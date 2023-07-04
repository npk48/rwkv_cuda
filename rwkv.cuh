#ifndef _RWKV_CUH_
#define _RWKV_CUH_

#include <stdint.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernel/gemm.cuh"
#include "kernel/layernorm.cuh"

#include "kernel/util.cuh"

#include "util.hpp"

namespace cuda
{
	namespace kernel
	{
		__global__ void convert_att_time_decay(float* dst, half* src, const uint32_t count)
		{
			uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

			if (idx < count)
				dst[idx] = -exp(__half2float(src[idx]));
		}

		template<typename T, const uint32_t TM>
		__global__ void mix_rkv(
			const uint32_t m, const uint32_t k,
			T* xx, T* sx,
			half* mix_r, half* mix_k, half* mix_v,
			T* rx, T* kx, T* vx
		)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t block_id = blockIdx.x;
			const uint32_t thread_num = blockDim.x;

			const uint32_t thread_per_token = k / TM;

			T xx_tile[TM];
			T sx_tile[TM];

			half mix_r_tile[TM];
			half mix_k_tile[TM];
			half mix_v_tile[TM];

			xx += (block_id * thread_num + thread_id) * TM;
			sx += (block_id * thread_num + thread_id) * TM;

			mix_r += (thread_id % thread_per_token) * TM;
			mix_k += (thread_id % thread_per_token) * TM;
			mix_v += (thread_id % thread_per_token) * TM;

			rx += (block_id * thread_num + thread_id) * TM;
			kx += (block_id * thread_num + thread_id) * TM;
			vx += (block_id * thread_num + thread_id) * TM;

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					xx_tile[ld_idx + i] = xx[ld_idx + i];
					sx_tile[ld_idx + i] = sx[ld_idx + i];

					mix_r_tile[ld_idx + i] = mix_r[ld_idx + i];
					mix_k_tile[ld_idx + i] = mix_k[ld_idx + i];
					mix_v_tile[ld_idx + i] = mix_v[ld_idx + i];
				}
			}

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					rx[ld_idx + i] = (T)((float)xx_tile[ld_idx + i] * (float)mix_r_tile[ld_idx + i] + (float)sx_tile[ld_idx + i] * (1.0f - (float)mix_r_tile[ld_idx + i]));
					kx[ld_idx + i] = (T)((float)xx_tile[ld_idx + i] * (float)mix_k_tile[ld_idx + i] + (float)sx_tile[ld_idx + i] * (1.0f - (float)mix_k_tile[ld_idx + i]));
					vx[ld_idx + i] = (T)((float)xx_tile[ld_idx + i] * (float)mix_v_tile[ld_idx + i] + (float)sx_tile[ld_idx + i] * (1.0f - (float)mix_v_tile[ld_idx + i]));
				}
			}

		}

		template<typename T, const uint32_t TM>
		__global__ void mix_rk(
			const uint32_t m, const uint32_t k,
			T* xx, T* sx,
			half* mix_r, half* mix_k,
			T* rx, T* kx
		)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t block_id = blockIdx.x;
			const uint32_t thread_num = blockDim.x;

			const uint32_t thread_per_token = k / TM;

			T xx_tile[TM];
			T sx_tile[TM];

			half mix_r_tile[TM];
			half mix_k_tile[TM];

			xx += (block_id * thread_num + thread_id) * TM;
			sx += (block_id * thread_num + thread_id) * TM;

			mix_r += (thread_id % thread_per_token) * TM;
			mix_k += (thread_id % thread_per_token) * TM;

			rx += (block_id * thread_num + thread_id) * TM;
			kx += (block_id * thread_num + thread_id) * TM;

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					xx_tile[ld_idx + i] = xx[ld_idx + i];
					sx_tile[ld_idx + i] = sx[ld_idx + i];

					mix_r_tile[ld_idx + i] = mix_r[ld_idx + i];
					mix_k_tile[ld_idx + i] = mix_k[ld_idx + i];
				}
			}

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					rx[ld_idx + i] = (T)((float)xx_tile[ld_idx + i] * (float)mix_r_tile[ld_idx + i] + (float)sx_tile[ld_idx + i] * (1.0f - (float)mix_r_tile[ld_idx + i]));
					kx[ld_idx + i] = (T)((float)xx_tile[ld_idx + i] * (float)mix_k_tile[ld_idx + i] + (float)sx_tile[ld_idx + i] * (1.0f - (float)mix_k_tile[ld_idx + i]));
				}
			}

		}

		template<typename T>//, const uint32_t TM>
		__global__ void forward_wkv_one(
			const uint32_t m, const uint32_t k,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			T* key, T* value, T* wkv
		)
		{
			const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			float ww = att_time_first[idx] + (float)key[idx];
			float p = max(att_pp[idx], ww);
			float e1 = exp(att_pp[idx] - p);
			float e2 = exp(ww - p);

			wkv[idx] = (T)((e1 * att_aa[idx] + e2 * (float)value[idx]) / (e1 * att_bb[idx] + e2));

			ww = att_time_decay[idx] + att_pp[idx];
			p = max(ww, (float)key[idx]);
			e1 = exp(ww - p);
			e2 = exp((float)key[idx] - p);

			att_aa[idx] = e1 * att_aa[idx] + e2 * (float)value[idx];
			att_bb[idx] = e1 * att_bb[idx] + e2;
			att_pp[idx] = p;
		}
	}

	inline cudaError_t convert_att_time_decay(tensor_t& from, tensor_t& to)
	{
		uint32_t count = from.shape.x * from.shape.y * from.shape.z;

		kernel::convert_att_time_decay<<<(count * 16 + 15) / 16, 15 >>>((float*)to.data, (half*)from.data, count);

		return cudaGetLastError();
	}

	inline cudaError_t token_to_emb(uint16_t* tokens, uint32_t token_size, tensor_t emb_weight, tensor_t& emb)
	{
		auto emb_size = emb_weight.shape.x;

		for (uint64_t i = 0; i < token_size; i++)
			cuda::copy_fp16(&((half*)emb.data)[i * emb_size], &((half*)emb_weight.data)[((uint64_t)tokens[i]) * emb_size], emb_size);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t mix_rkv(
		const uint32_t m, const uint32_t k,
		T* xx, T* sx,
		half* mix_r, half* mix_k, half* mix_v,
		T* rx, T* kx, T* vx
	)
	{
		const uint32_t TM = 48;
		const uint32_t thread_dim = 16;
		const uint32_t block_dim = m * k / (thread_dim * TM);

		kernel::mix_rkv<T, TM><<<block_dim, thread_dim>>>(m, k, xx, sx, mix_r, mix_k, mix_v, rx, kx, vx);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t mix_rk(
		const uint32_t m, const uint32_t k,
		T* xx, T* sx,
		half* mix_r, half* mix_k,
		T* rx, T* kx
	)
	{
		const uint32_t TM = 48;
		const uint32_t thread_dim = 16;
		const uint32_t block_dim = m * k / (thread_dim * TM);

		kernel::mix_rk<T, TM><<<block_dim, thread_dim>>>(m, k, xx, sx, mix_r, mix_k, rx, kx);

		return cudaGetLastError();
	}

	template<typename T>//, const uint32_t TM>
	inline cudaError_t forward_wkv_one(
		const uint32_t emb_size,
		float* att_aa, float* att_bb, float* att_pp,
		float* att_time_decay, float* att_time_first,
		T* key, T* value, T* wkv
	)
	{
		const uint32_t thread_dim = 16;
		const uint32_t block_dim = (emb_size + 15) / 16;

		kernel::forward_wkv_one<T><<<block_dim, thread_dim>>>(1, emb_size, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);

		return cudaGetLastError();
	}

	template<typename T>
	inline void time_mix_one(
		tensor_t& x,
		tensor_t& att_xx, tensor_t& att_aa, tensor_t& att_bb, tensor_t& att_pp,
		tensor_t& ln1_weight, tensor_t& ln1_bias,
		tensor_t& att_time_mix_r, tensor_t& att_time_mix_k, tensor_t& att_time_mix_v,
		tensor_t& att_time_decay, tensor_t& att_time_first,
		tensor_t& att_key_weight, tensor_t& att_value_weight, tensor_t& att_receptance_weight, tensor_t& att_output_weight
	)
	{
		const uint32_t emb_size = x.shape.x;
		const uint32_t token_size = 1;

		auto result = cudaSuccess;

		auto xx = cuda::create_tensor<T>(x.shape);

		result = cuda::layernorm<T>((T*)x.data, (half*)ln1_weight.data, (half*)ln1_bias.data, (T*)xx.data, token_size, emb_size);
		
		auto rx = cuda::create_tensor<T>(x.shape);
		auto kx = cuda::create_tensor<T>(x.shape);
		auto vx = cuda::create_tensor<T>(x.shape);

		result = cuda::mix_rkv<T>(token_size, emb_size, (T*)xx.data, (T*)att_xx.data, (half*)att_time_mix_r.data, (half*)att_time_mix_k.data, (half*)att_time_mix_v.data, (T*)rx.data, (T*)kx.data, (T*)vx.data);
		
		result = cuda::copy<T>((T*)att_xx.data, (T*)xx.data, emb_size);

		// m = y, k = x
		auto r = cuda::create_tensor<T>({ att_receptance_weight.shape.x, 1, 1 });
		result = cuda::gemv_sigmoid<T>((half*)att_receptance_weight.data, (T*)rx.data, (T*)r.data, att_receptance_weight.shape.x, att_receptance_weight.shape.y);

		auto k = cuda::create_tensor<T>({ att_key_weight.shape.x, 1, 1 });
		result = cuda::gemv<T>((half*)att_key_weight.data, (T*)kx.data, (T*)k.data, att_key_weight.shape.x, att_key_weight.shape.y);

		auto v = cuda::create_tensor<T>({ att_value_weight.shape.x, 1, 1 });
		result = cuda::gemv<T>((half*)att_value_weight.data, (T*)vx.data, (T*)v.data, att_value_weight.shape.x, att_value_weight.shape.y);

		auto wkv = cuda::create_tensor<T>(x.shape);
		result = cuda::forward_wkv_one<T>(emb_size, (float*)att_aa.data, (float*)att_bb.data, (float*)att_pp.data, (float*)att_time_decay.data, (float*)att_time_first.data, (T*)k.data, (T*)v.data, (T*)wkv.data);

		auto r_dot_y = cuda::create_tensor<T>(x.shape);
		result = cuda::element_wise_product<T>(emb_size, (T*)r.data, (T*)wkv.data, (T*)r_dot_y.data);

		auto out = cuda::create_tensor<T>({ att_output_weight.shape.x, 1, 1 });
		result = cuda::gemv<T>((half*)att_output_weight.data, (T*)r_dot_y.data, (T*)out.data, att_output_weight.shape.x, att_output_weight.shape.y);

		result = cuda::element_wise_add<T>(emb_size, (T*)x.data, (T*)out.data, (T*)x.data);

		cuda::free_tensor(xx);
		cuda::free_tensor(rx);
		cuda::free_tensor(kx);
		cuda::free_tensor(vx);

		cuda::free_tensor(r);
		cuda::free_tensor(k);
		cuda::free_tensor(v);
		cuda::free_tensor(wkv);
		cuda::free_tensor(r_dot_y);
		cuda::free_tensor(out);
	}

	template<typename T>
	inline void channel_mix_one(
		tensor_t& x,
		tensor_t& ffn_xx,
		tensor_t& ln2_weight, tensor_t& ln2_bias,
		tensor_t& ffn_time_mix_r, tensor_t& ffn_time_mix_k,
		tensor_t& ffn_key_weight, tensor_t& ffn_value_weight, tensor_t& ffn_receptance_weight
	)
	{
		const uint32_t emb_size = x.shape.x;
		const uint32_t token_size = 1;

		auto result = cudaSuccess;

		auto xx = cuda::create_tensor<T>(x.shape);

		result = cuda::layernorm<T>((T*)x.data, (half*)ln2_weight.data, (half*)ln2_bias.data, (T*)xx.data, token_size, emb_size);

		auto rx = cuda::create_tensor<T>(x.shape);
		auto kx = cuda::create_tensor<T>(x.shape);

		result = cuda::mix_rk<T>(token_size, emb_size, (T*)xx.data, (T*)ffn_xx.data, (half*)ffn_time_mix_r.data, (half*)ffn_time_mix_k.data, (T*)rx.data, (T*)kx.data);

		result = cuda::copy<T>((T*)ffn_xx.data, (T*)xx.data, emb_size);

		// m = y, k = x
		auto r = cuda::create_tensor<T>({ ffn_receptance_weight.shape.x, 1, 1 });
		result = cuda::gemv_sigmoid<T>((half*)ffn_receptance_weight.data, (T*)rx.data, (T*)r.data, ffn_receptance_weight.shape.x, ffn_receptance_weight.shape.y);

		auto vx = cuda::create_tensor<T>({ ffn_key_weight.shape.x, 1, 1 });
		result = cuda::gemv_square_relu<T>((half*)ffn_key_weight.data, (T*)kx.data, (T*)vx.data, ffn_key_weight.shape.x, ffn_key_weight.shape.y);
		// tbd square relu has problem
		//inspect_tensor(vx);

		auto vx_vw_product = cuda::create_tensor<T>({ ffn_value_weight.shape.x, 1, 1 });
		result = cuda::gemv<T>((half*)ffn_value_weight.data, (T*)vx.data, (T*)vx_vw_product.data, ffn_value_weight.shape.x, ffn_value_weight.shape.y);

		// wrong value
		//inspect_tensor(vx_vw_product);

		auto out = cuda::create_tensor<T>(x.shape);
		result = cuda::element_wise_product<T>(x.shape.x, (T*)r.data, (T*)vx_vw_product.data, (T*)out.data);
		
		//inspect_tensor(out);

		result = cuda::element_wise_add<T>(emb_size, (T*)x.data, (T*)out.data, (T*)x.data);

		cuda::free_tensor(xx);
		cuda::free_tensor(rx);
		cuda::free_tensor(kx);

		cuda::free_tensor(r);
		cuda::free_tensor(vx);
		cuda::free_tensor(vx_vw_product);
		cuda::free_tensor(out);
	}

	inline void emb_to_logits(tensor_t& x, uint32_t token_size, tensor_t& head_weight, float* logits)
	{
		auto logits_out = cuda::create_tensor<half>({ head_weight.shape.x, 1, 1 });
		auto result = cuda::gemv<half>((half*)head_weight.data, (half*)x.data, (half*)logits_out.data, head_weight.shape.x, head_weight.shape.y);


		cuda::dump_fp16(logits, (half*)logits_out.data, head_weight.shape.x);
		
		// // x @ head_weight
		// auto _logits_out = gemm<half>((half*)head_weight.data, (half*)x.data, (half*)x.data, );
		// // [y=65535, x=2]
		// auto logits_out = cuda::create_tensor_fp16(cuda::tensor_shape_t(_logits_out.shape.y, _logits_out.shape.x));
		// // [y=2, x=65535]
		// cuda::transpose_tensor(_logits_out, logits_out);
		// cuda::free_tensor(_logits_out);
		// 
		// cuda::dump_to_host_fp16(logits, &((half*)logits_out.data)[(logits_out.shape.x - 1) * logits_out.shape.y], logits_out.shape.y);
		// 
		// cuda::free_tensor(logits_out);
	}
}



#endif
