#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../kernel/gemm.cuh"
#include "../kernel/layernorm.cuh"
#include "../util.hpp"

#include "lib_rwkv.h"

#include "../tokenizer.hpp"
#include "../safe_tensors.hpp"
#include "../rwkv_model.hpp"
#include "../sampler.hpp"

static auto tokenizer = trie_tokenizer_t();
static auto model = rwkv_model_cuda_t();


EXPORT_API void rwkv_init(
	const char* vocab_file_path,
	const char* safe_tensor_path,
	uint32_t* vocab_size_out,
	uint32_t* emb_size_out,
	uint32_t* state_size_out
)
{
	tokenizer.load(vocab_file_path);

	auto tensors = safe_tensors_model_t();
	tensors.load(safe_tensor_path);

	model.load(tensors);

	*vocab_size_out = model.vocab_size;
	*emb_size_out = model.emb_size;
	*state_size_out = model.emb_size * model.layer_count * 5 * sizeof(float);
}

EXPORT_API rwkv_state_t rwkv_new_state()
{
	auto ret = new rwkv_state_cuda_t();

	*ret = model.create_state();

	return reinterpret_cast<void*>(ret);
}

EXPORT_API rwkv_state_t rwkv_dup_state(rwkv_state_t state)
{
	auto src = reinterpret_cast<rwkv_state_cuda_t*>(state);

	auto ret = new rwkv_state_cuda_t();

	*ret = model.duplicate_state(*src);

	return reinterpret_cast<void*>(ret);
}

EXPORT_API void rwkv_dump_state(rwkv_state_t state, uint8_t* state_data)
{
	auto src = reinterpret_cast<rwkv_state_cuda_t*>(state);
	auto blk_size = model.emb_size * sizeof(float);

	uint32_t offset = 0;

	for (uint32_t i = 0; i < model.layer_count; i++)
	{
		cudaMemcpy(state_data + offset, (*src)[i].att_xx.data, blk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost); offset += blk_size;
		cudaMemcpy(state_data + offset, (*src)[i].att_aa.data, blk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost); offset += blk_size;
		cudaMemcpy(state_data + offset, (*src)[i].att_bb.data, blk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost); offset += blk_size;
		cudaMemcpy(state_data + offset, (*src)[i].att_pp.data, blk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost); offset += blk_size;
		cudaMemcpy(state_data + offset, (*src)[i].ffn_xx.data, blk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost); offset += blk_size;
	}
}

EXPORT_API void rwkv_load_state(rwkv_state_t state, uint8_t* state_data)
{
	auto dst = reinterpret_cast<rwkv_state_cuda_t*>(state);
	auto blk_size = model.emb_size * sizeof(float);

	uint32_t offset = 0;

	for (uint32_t i = 0; i < model.layer_count; i++)
	{
		cudaMemcpy((*dst)[i].att_xx.data, state_data + offset, blk_size, cudaMemcpyKind::cudaMemcpyHostToDevice); offset += blk_size;
		cudaMemcpy((*dst)[i].att_aa.data, state_data + offset, blk_size, cudaMemcpyKind::cudaMemcpyHostToDevice); offset += blk_size;
		cudaMemcpy((*dst)[i].att_bb.data, state_data + offset, blk_size, cudaMemcpyKind::cudaMemcpyHostToDevice); offset += blk_size;
		cudaMemcpy((*dst)[i].att_pp.data, state_data + offset, blk_size, cudaMemcpyKind::cudaMemcpyHostToDevice); offset += blk_size;
		cudaMemcpy((*dst)[i].ffn_xx.data, state_data + offset, blk_size, cudaMemcpyKind::cudaMemcpyHostToDevice); offset += blk_size;
	}
}

EXPORT_API void rwkv_free_state(rwkv_state_t state)
{
	auto src = reinterpret_cast<rwkv_state_cuda_t*>(state);

	for (uint32_t i = 0; i < model.layer_count; i++)
	{
		cudaFree((*src)[i].att_xx.data);
		cudaFree((*src)[i].att_aa.data);
		cudaFree((*src)[i].att_bb.data);
		cudaFree((*src)[i].att_pp.data);
		cudaFree((*src)[i].ffn_xx.data);
	}

	delete src;
}

EXPORT_API uint32_t rwkv_str2tok(const char* str, uint16_t* token_out, uint32_t max_token_out)
{
	auto toks = tokenizer.encode(str);

	memcpy_s(token_out, max_token_out * sizeof(uint16_t), &toks[0], toks.size() * sizeof(uint16_t));

	return toks.size();
}

EXPORT_API uint32_t rwkv_tok2str(char* str, uint16_t* token_in, uint32_t token_size)
{
	auto toks = std::vector<uint16_t>(token_size);

	toks.assign(token_in, token_in + token_size);

	auto decoded = tokenizer.decode(toks);

	if (decoded.size() == 0) 
	{
		str[0] = 0;
		return 1;
	}
	else
	{
		memcpy(str, &decoded[0], decoded.size());
		return decoded.size();
	}
}

EXPORT_API uint32_t rwkv_sample_typical(float* logits, uint32_t logits_size, float tau, float temp)
{
	auto logits_ = std::vector<float>(logits_size);

	logits_.assign(logits, logits + logits_size);

	return sampler::typical(logits_, tau, temp);
}

EXPORT_API uint32_t rwkv_forward(
	rwkv_state_t state,
	uint16_t* token_in, uint32_t token_size,
	float* logits_out
)
{
	auto st = reinterpret_cast<rwkv_state_cuda_t*>(state);

	auto toks = std::vector<uint16_t>(token_size);

	toks.assign(token_in, token_in + token_size);

	auto logits = model.forward(*st, toks);

	memcpy(logits_out, &logits[0], logits.size() * sizeof(float));

	return logits.size();
}