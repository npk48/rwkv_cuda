#ifndef _RWKV_MODEL_HPP_
#define _RWKV_MODEL_HPP_

#include <string>
#include <algorithm>

#include "safe_tensors.hpp"

#include "rwkv.cuh"


/*

emb.weight
ln0.weight/bias

each rwkv layer has 18 tensors
	ln1.weight/bias
	ln2.weight/bias

	att.time_decay/time_first	fp32
	att.time_mix_k/v/r
	att.k/v/r
	att.out

	ffn.time_mix_k/r
	ffn.k/v/r

ln_out.weight/bias
head.weight

*/

struct rwkv_layer_state_cuda_t
{
	cuda::tensor_t att_xx; // fp16 / int8
	cuda::tensor_t att_aa; // fp32
	cuda::tensor_t att_bb; // fp32
	cuda::tensor_t att_pp; // fp32
	cuda::tensor_t ffn_xx; // fp16 / int8
};

typedef std::vector<rwkv_layer_state_cuda_t> rwkv_state_cuda_t;

struct rwkv_emb_cuda_t
{
	cuda::tensor_t emb_weight;

	cuda::tensor_t ln0_weight;
	cuda::tensor_t ln0_bias;
};

struct rwkv_layer_cuda_t
{
	cuda::tensor_t ln1_weight;
	cuda::tensor_t ln1_bias;

	cuda::tensor_t ln2_weight;
	cuda::tensor_t ln2_bias;

	cuda::tensor_t att_time_decay;			// fp32 w[x] = -torch.exp(w[x].float())
	cuda::tensor_t att_time_first;			// fp32 w[x] = w[x].float()

	cuda::tensor_t att_time_mix_k;
	cuda::tensor_t att_time_mix_v;
	cuda::tensor_t att_time_mix_r;

	cuda::tensor_t att_key_weight;			// transpose
	cuda::tensor_t att_value_weight;		// transpose
	cuda::tensor_t att_receptance_weight;	// transpose

	cuda::tensor_t att_output_weight;		// transpose w[x] = w[x] / (2 ** int(layer_id / 6))

	cuda::tensor_t ffn_time_mix_k;
	cuda::tensor_t ffn_time_mix_r;

	cuda::tensor_t ffn_key_weight;			// transpose
	cuda::tensor_t ffn_value_weight;		// transpose w[x] = w[x] / (2 ** int(layer_id / 6))
	cuda::tensor_t ffn_receptance_weight;	// transpose
};

struct rwkv_head_cuda_t
{
	cuda::tensor_t ln_out_weight;
	cuda::tensor_t ln_out_bias;

	cuda::tensor_t head_weight;				// transpose
};

class rwkv_model_cuda_t
{
public:
	rwkv_emb_cuda_t emb;
	std::vector<rwkv_layer_cuda_t> layers;
	rwkv_head_cuda_t head;

	std::vector<cuda::data_type_t> layer_strategy;

	uint32_t layer_count;
	uint32_t emb_size;
	uint32_t vocab_size;

public:
	void load(safe_tensors_model_t model)
	{
		this->layer_count = (model.layers.size() - 6) / 18;
		this->vocab_size = model.layers["emb.weight"].shape[0];
		this->emb_size = model.layers["emb.weight"].shape[1];

		auto load_tensor = [&](std::string name)
		{
			auto& layer = model.layers[name];
			auto& shape = layer.shape;

			cuda::tensor_shape_t tensor_shape;

			if (shape.size() >= 3)
			{
				tensor_shape.x = shape[2];
				tensor_shape.y = shape[1];
				tensor_shape.z = shape[0];
			}
			else if (shape.size() >= 2)
			{
				tensor_shape.x = shape[1];
				tensor_shape.y = shape[0];
				tensor_shape.z = 1;
			}
			else
			{
				tensor_shape.x = shape[0];
				tensor_shape.y = 1;
				tensor_shape.z = 1;
			}

			auto tensor = cuda::create_tensor<half>(tensor_shape);

			cuda::load_fp16(
				(half*)tensor.data,
				(half*)layer.data,
				tensor_shape.x *
				tensor_shape.y *
				tensor_shape.z
			);

			return tensor;
		};

		cuda::tensor_t tensor;

		// load emb
		this->emb.emb_weight = load_tensor("emb.weight");
		this->emb.ln0_weight = load_tensor("blocks.0.ln0.weight");
		this->emb.ln0_bias = load_tensor("blocks.0.ln0.bias");

		// precompute embedding
		//inspect_tensor(this->emb.emb_weight);
		tensor = cuda::create_tensor<half>(this->emb.emb_weight.shape);
		cuda::layernorm<half>((half*)this->emb.emb_weight.data, (half*)this->emb.ln0_weight.data, (half*)this->emb.ln0_bias.data, (half*)tensor.data, tensor.shape.y, tensor.shape.x);
		std::swap(tensor, this->emb.emb_weight);
		cuda::free_tensor(tensor);
		cuda::free_tensor(this->emb.ln0_weight);
		cuda::free_tensor(this->emb.ln0_bias);

		// load layers
		for (uint32_t i = 0; i < layer_count; i++)
		{
			// calculate layer 0 in fp32 else pf16;
			this->layer_strategy.push_back(i == 0 ? cuda::data_type_t::fp32 : cuda::data_type_t::fp16);

			std::string prefix = "blocks." + std::to_string(i) + ".";

			rwkv_layer_cuda_t layer;

			layer.ln1_weight = load_tensor(prefix + "ln1.weight");
			layer.ln1_bias = load_tensor(prefix + "ln1.bias");

			layer.ln2_weight = load_tensor(prefix + "ln2.weight");
			layer.ln2_bias = load_tensor(prefix + "ln2.bias");

			// fp32 w[x] = -torch.exp(w[x].float())
			tensor = load_tensor(prefix + "att.time_decay");
			layer.att_time_decay = cuda::create_tensor<float>(tensor.shape);
			cuda::convert_att_time_decay(tensor, layer.att_time_decay);
			cuda::free_tensor(tensor);

			// fp32 w[x] = w[x].float()
			tensor = load_tensor(prefix + "att.time_first");
			layer.att_time_first = cuda::create_tensor<float>(tensor.shape);
			cuda::convert<half, float>((half*)tensor.data, (float*)layer.att_time_first.data, tensor.byte_size / 2);
			cuda::free_tensor(tensor);

			layer.att_time_mix_r = load_tensor(prefix + "att.time_mix_r");
			layer.att_time_mix_k = load_tensor(prefix + "att.time_mix_k");
			layer.att_time_mix_v = load_tensor(prefix + "att.time_mix_v");

			// transpose
			tensor = load_tensor(prefix + "att.key.weight");
			layer.att_key_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.att_key_weight.data);
			cuda::free_tensor(tensor);

			// column major no need to transpose
			//layer.att_key_weight = load_tensor(prefix + "att.key.weight");

			// transpose
			tensor = load_tensor(prefix + "att.value.weight");
			layer.att_value_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.att_value_weight.data);
			cuda::free_tensor(tensor);

			//layer.att_value_weight = load_tensor(prefix + "att.value.weight");

			// transpose
			tensor = load_tensor(prefix + "att.receptance.weight");
			layer.att_receptance_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.att_receptance_weight.data);
			cuda::free_tensor(tensor);
			// layer.att_receptance_weight = load_tensor(prefix + "att.receptance.weight");

			// transpose w[x] then w[x] / (2 ** int(layer_id / 6))
			// tensor = load_tensor(prefix + "att.output.weight");
			// layer.att_output_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			// cuda::transpose_tensor(tensor, layer.att_output_weight);
			// cuda::free_tensor(tensor);
			// cuda::scale_tensor(layer.att_output_weight, 1.f / std::powf(2, (float)(uint32_t(i / 6))));
			tensor = load_tensor(prefix + "att.output.weight");
			layer.att_output_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.att_output_weight.data);
			cuda::element_wise_scale<half>(layer.att_output_weight.byte_size / 2, (half*)layer.att_output_weight.data, 1.f / std::powf(2, (float)(uint32_t(i / 6))), (half*)layer.att_output_weight.data);
			cuda::free_tensor(tensor);

			//layer.att_output_weight = load_tensor(prefix + "att.output.weight");
			//cuda::element_wise_scale<half>(layer.att_output_weight.byte_size / 2, (half*)layer.att_output_weight.data, 1.f / std::powf(2, (float)(uint32_t(i / 6))), (half*)layer.att_output_weight.data);

			layer.ffn_time_mix_k = load_tensor(prefix + "ffn.time_mix_k");
			layer.ffn_time_mix_r = load_tensor(prefix + "ffn.time_mix_r");

			// transpose
			// tensor = load_tensor(prefix + "ffn.key.weight");
			// layer.ffn_key_weight = cuda::create_tensor_fp16({ tensor.shape.y, tensor.shape.x, 1 });
			// cuda::transpose_tensor(tensor, layer.ffn_key_weight);
			// cuda::free_tensor(tensor);
			tensor = load_tensor(prefix + "ffn.key.weight");
			layer.ffn_key_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.ffn_key_weight.data);
			cuda::free_tensor(tensor);

			//layer.ffn_key_weight = load_tensor(prefix + "ffn.key.weight");

			// transpose w[x] = w[x] / (2 ** int(layer_id / 6))
			// tensor = load_tensor(prefix + "ffn.value.weight");
			// layer.ffn_value_weight = cuda::create_tensor_fp16({ tensor.shape.y, tensor.shape.x, 1 });
			// cuda::transpose_tensor(tensor, layer.ffn_value_weight);
			// cuda::free_tensor(tensor);
			// cuda::scale_tensor(layer.ffn_value_weight, 1.f / std::powf(2, (float)(uint32_t(i / 6))));
			tensor = load_tensor(prefix + "ffn.value.weight");
			layer.ffn_value_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.ffn_value_weight.data);
			cuda::element_wise_scale<half>(layer.ffn_value_weight.byte_size / 2, (half*)layer.ffn_value_weight.data, 1.f / std::powf(2, (float)(uint32_t(i / 6))), (half*)layer.ffn_value_weight.data);
			cuda::free_tensor(tensor);

			//layer.ffn_value_weight = load_tensor(prefix + "ffn.value.weight");
			//cuda::element_wise_scale<half>(layer.ffn_value_weight.byte_size / 2, (half*)layer.ffn_value_weight.data, 1.f / std::powf(2, (float)(uint32_t(i / 6))), (half*)layer.ffn_value_weight.data);


			// transpose
			// tensor = load_tensor(prefix + "ffn.receptance.weight");
			// layer.ffn_receptance_weight = cuda::create_tensor_fp16({ tensor.shape.y, tensor.shape.x, 1 });
			// cuda::transpose_tensor(tensor, layer.ffn_receptance_weight);
			// cuda::free_tensor(tensor);
			tensor = load_tensor(prefix + "ffn.receptance.weight");
			layer.ffn_receptance_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
			cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)layer.ffn_receptance_weight.data);
			cuda::free_tensor(tensor);

			//layer.ffn_receptance_weight = load_tensor(prefix + "ffn.receptance.weight");

			this->layers.push_back(layer);
		}

		// load head
		this->head.ln_out_weight = load_tensor("ln_out.weight");
		this->head.ln_out_bias = load_tensor("ln_out.bias");

		// transpose
		// tensor = load_tensor("head.weight");
		// this->head.head_weight = cuda::create_tensor_fp16({ tensor.shape.y, tensor.shape.x, 1 });
		// cuda::transpose_tensor(tensor, this->head.head_weight);
		// cuda::free_tensor(tensor);
		tensor = load_tensor("head.weight");
		this->head.head_weight = cuda::create_tensor<half>({ tensor.shape.y, tensor.shape.x, 1 });
		cuda::transpose<half>(tensor.shape.y, tensor.shape.x, (half*)tensor.data, (half*)this->head.head_weight.data);
		cuda::free_tensor(tensor);

		//this->head.head_weight = load_tensor("head.weight");
	}

	rwkv_state_cuda_t create_state()
	{
		rwkv_state_cuda_t state;

		for (uint32_t i = 0; i < layer_count; i++)
		{
			rwkv_layer_state_cuda_t layer_state;

			auto& strategy = this->layer_strategy[i];

			auto x = cuda::tensor_shape_t(this->emb_size);

			layer_state.att_xx = strategy == cuda::data_type_t::fp16 ? cuda::create_tensor<half>(x) : cuda::create_tensor<float>(x);
			cuda::zero_memory(layer_state.att_xx.data, layer_state.att_xx.byte_size);

			layer_state.att_aa = cuda::create_tensor<float>(x);
			cuda::zero_memory(layer_state.att_aa.data, layer_state.att_aa.byte_size);

			layer_state.att_bb = cuda::create_tensor<float>(x);
			cuda::zero_memory(layer_state.att_bb.data, layer_state.att_bb.byte_size);

			layer_state.att_pp = cuda::create_tensor<float>(x);
			cuda::fill_fp32((float*)layer_state.att_pp.data, -1e30, this->emb_size);

			layer_state.ffn_xx = strategy == cuda::data_type_t::fp16 ? cuda::create_tensor<half>(x) : cuda::create_tensor<float>(x);
			cuda::zero_memory(layer_state.ffn_xx.data, layer_state.ffn_xx.byte_size);

			state.push_back(layer_state);
		}

		return state;
	}

	rwkv_state_cuda_t duplicate_state(rwkv_state_cuda_t state)
	{
		auto dup_state = this->create_state();

		for (uint32_t i = 0; i < layer_count; i++)
		{
			cuda::copy(dup_state[i].att_xx.data, state[i].att_xx.data, state[i].att_xx.byte_size);
			cuda::copy(dup_state[i].att_aa.data, state[i].att_aa.data, state[i].att_aa.byte_size);
			cuda::copy(dup_state[i].att_bb.data, state[i].att_bb.data, state[i].att_bb.byte_size);
			cuda::copy(dup_state[i].att_pp.data, state[i].att_pp.data, state[i].att_pp.byte_size);
			cuda::copy(dup_state[i].ffn_xx.data, state[i].ffn_xx.data, state[i].ffn_xx.byte_size);
		}

		return dup_state;
	}

public:
	std::vector<float> forward(rwkv_state_cuda_t& state, std::vector<uint16_t> tokens)
	{
		uint64_t token_size = tokens.size();

		auto input_shape = cuda::tensor_shape_t(emb_size, token_size);

		auto x0 = cuda::create_tensor<half>(input_shape);
		auto x1 = cuda::create_tensor<half>(input_shape);

		cuda::token_to_emb(&tokens[0], token_size, emb.emb_weight, x0);
		//cuda::sync();

		for (uint32_t i = 0; i < this->layer_count; i++)
		{
			auto& layer = this->layers[i];
			auto& layer_state = state[i];
			auto& strategy = this->layer_strategy[i];

			if (strategy != cuda::data_type_t::fp16 && x0.type == cuda::data_type_t::fp16)
			{
				auto x0_fp32 = cuda::create_tensor<float>(x0.shape);
				auto x1_fp32 = cuda::create_tensor<float>(x0.shape);

				cuda::convert<half, float>((half*)x0.data, (float*)x0_fp32.data, x0.shape.x);
				cuda::convert<half, float>((half*)x1.data, (float*)x1_fp32.data, x0.shape.x);

				cuda::free_tensor(x0);
				cuda::free_tensor(x1);

				std::swap(x0_fp32, x0);
				std::swap(x1_fp32, x1);
				//inspect_tensor(x0);
			}
			else if (strategy != cuda::data_type_t::fp32 && x0.type == cuda::data_type_t::fp32)
			{
				auto x0_fp16 = cuda::create_tensor<half>(x0.shape);
				auto x1_fp16 = cuda::create_tensor<half>(x0.shape);

				cuda::convert<float, half>((float*)x0.data, (half*)x0_fp16.data, x0.shape.x);
				cuda::convert<float, half>((float*)x1.data, (half*)x1_fp16.data, x0.shape.x);

				cuda::free_tensor(x0);
				cuda::free_tensor(x1);

				std::swap(x0_fp16, x0);
				std::swap(x1_fp16, x1);
			}

			if(x0.type == cuda::data_type_t::fp16)
				cuda::time_mix_one<half>(
					x0,
					layer_state.att_xx, layer_state.att_aa, layer_state.att_bb, layer_state.att_pp,
					layer.ln1_weight, layer.ln1_bias,
					layer.att_time_mix_r, layer.att_time_mix_k, layer.att_time_mix_v,
					layer.att_time_decay, layer.att_time_first,
					layer.att_key_weight, layer.att_value_weight, layer.att_receptance_weight, layer.att_output_weight
				);
			else
				cuda::time_mix_one<float>(
					x0,
					layer_state.att_xx, layer_state.att_aa, layer_state.att_bb, layer_state.att_pp,
					layer.ln1_weight, layer.ln1_bias,
					layer.att_time_mix_r, layer.att_time_mix_k, layer.att_time_mix_v,
					layer.att_time_decay, layer.att_time_first,
					layer.att_key_weight, layer.att_value_weight, layer.att_receptance_weight, layer.att_output_weight
					);


			if (x0.type == cuda::data_type_t::fp16)
				cuda::channel_mix_one<half>(
					x0,
					layer_state.ffn_xx,
					layer.ln2_weight, layer.ln2_bias,
					layer.ffn_time_mix_r, layer.ffn_time_mix_k,
					layer.ffn_key_weight, layer.ffn_value_weight, layer.ffn_receptance_weight
				);
			else
				cuda::channel_mix_one<float>(
					x0,
					layer_state.ffn_xx,
					layer.ln2_weight, layer.ln2_bias,
					layer.ffn_time_mix_r, layer.ffn_time_mix_k,
					layer.ffn_key_weight, layer.ffn_value_weight, layer.ffn_receptance_weight
					);
			
			//cuda::inspect_tensor(x0);

			if ((i + 1) % 6 == 0)
				cuda::element_wise_scale<half>(x0.shape.x, (half*)x0.data, 0.5f, (half*)x0.data);
		}

		cuda::layernorm<half>((half*)x0.data, (half*)head.ln_out_weight.data, (half*)head.ln_out_bias.data, (half*)x1.data, 1, x0.shape.x);
		//cuda::sync();
		std::swap(x1, x0);

		//inspect_tensor(x0);
		//inspect_tensor(head.head_weight);

		std::vector<float> logits(vocab_size);
		cuda::emb_to_logits(x0, token_size, head.head_weight, &logits[0]);
		//cuda::sync();

		cuda::free_tensor(x0);
		cuda::free_tensor(x1);

		// std::vector<float> softmax(vocab_size);
		// 
		// // exp(x-max(x)) / sum(exp(x-max(x)))
		// auto max_x = *std::max_element(logits.begin(), logits.end());
		// 
		// float sum_x = 0.f;
		// 
		// for (uint32_t i = 0; i < logits.size(); i++) 
		// 	sum_x += std::exp(logits[i] - max_x);
		// 
		// for (int i = 0; i < logits.size(); i++)
		// 	softmax[i] = std::exp(logits[i] - max_x) / sum_x;

		return logits;
	}
};

#endif