#ifndef _LAYERNORM_CUH_
#define _LAYERNORM_CUH_

#include <stdint.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "util.cuh"

namespace cuda
{
	namespace kernel
	{
		inline __device__ void welford(float* mean, float* m2, float* count, float val)
		{
			*count += 1;
			float delta1 = val - *mean;
			*mean += delta1 / (*count);
			float delta2 = val - *mean;
			*m2 += delta1 * delta2;
		}

		inline __device__ void welford(float* mean, float* m2, float* count, float b_mean, float b_m2, float b_count)
		{
			float new_count = *count + b_count;
			float nb_over_n = b_count / new_count;
			float delta = b_mean - *mean;
			*mean += delta * nb_over_n;
			*m2 += b_m2 + delta * delta * (*count) * nb_over_n;
			*count = new_count;
		}

		template<typename T, const uint32_t TM>
		__global__ void layernorm(float* shared_data, const uint32_t m, const uint32_t k, T* x, half* weight, half* bias, T* norm_x)
		{
			const float eps = 1e-05;

			const uint32_t thread_id = threadIdx.x;
			const uint32_t block_id = blockIdx.x;
			const uint32_t thread_num = blockDim.x;
			//const uint32_t block_num = m / thread_num;

			const uint32_t thread_per_token = k / TM; 

			//extern __shared__ float shared_data[];

			/*
				shared_weight [TM]
				shared_bias   [TM]

				shared_block_data   [  thread_num,  thread_num  ]
									   block_mean   block_m2 / block_dev   
			
			*/

			float* block_mean = &shared_data[block_id * thread_num * 2];
			float* block_m2 = &shared_data[block_id * thread_num * 2 + thread_num];
			float* block_dev = &shared_data[block_id * thread_num * 2 + thread_num];

			T x_tile[TM];

			x += (block_id * thread_num + thread_id) * TM;
			norm_x += (block_id * thread_num + thread_id) * TM;

			// calculate welford mean & var
			float local_mean = 0.f;
			float local_m2 = 0.f;
			float local_count = 0.f;
			float local_dev = 0.f;

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					x_tile[ld_idx + i] = x[ld_idx + i];

#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					welford(&local_mean, &local_m2, &local_count, x_tile[ld_idx + i]);
			}

			block_mean[thread_id] = local_mean;
			block_m2[thread_id] = local_m2;

			__syncthreads();

			if ((thread_id % thread_per_token) == 0)
			{
				for (uint32_t i = 1; i < thread_per_token; i++)
				{
					float b_mean = block_mean[thread_id + i];
					float b_m2 = block_m2[thread_id + i];
					float b_count = TM;
					welford(&local_mean, &local_m2, &local_count, b_mean, b_m2, b_count);
				}
				
				local_dev = sqrt(local_m2 / local_count);

				for (uint32_t i = 0; i < thread_per_token; i++)
				{
					block_mean[thread_id + i] = local_mean;
					block_dev[thread_id + i] = local_dev;
				}
			}

			__syncthreads();

			local_mean = block_mean[thread_id];
			local_dev = block_dev[thread_id];

			// start process layernorm
 
 			// weight & bias loaded, move to local reg
 			half weight_tile[TM];
 			half bias_tile[TM];
 
#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					weight_tile[ld_idx + i] = weight[(thread_id % thread_per_token) * TM + ld_idx + i];
					bias_tile[ld_idx + i] = bias[(thread_id % thread_per_token) * TM + ld_idx + i];
				}
			}
 
 			T thread_results[TM];
 
#pragma unroll
			for (uint32_t p_idx = 0; p_idx < TM; p_idx++)
			{
				float val = (float)x_tile[p_idx];
				thread_results[p_idx] = (T)((float)(bias_tile[p_idx]) + (float)(weight_tile[p_idx]) * (val - local_mean) / (local_dev + eps));
			}
 
 #pragma unroll
 			for (uint32_t p_idx = 0; p_idx < TM; p_idx += 8)
 			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					norm_x[p_idx + i] = thread_results[p_idx + i];
 			}

 		}

	}

	template<typename T>
	inline cudaError_t layernorm(T* x, half* weight, half* bias, T* norm_x, uint32_t m, uint32_t k)
	{

		if (k <= 768)
		{
			const uint32_t TM = 48;
			const uint32_t thread_dim = 16;
			const uint32_t block_dim = m * k / (thread_dim * TM);

			float* shared_data = 0;
			cudaMalloc(&shared_data, m * thread_dim * 2 * sizeof(float));

			kernel::layernorm<T, TM> << <block_dim, thread_dim >> > (shared_data, m, k, x, weight, bias, norm_x);

			cudaFree(shared_data);
		}
		else
		{
			const uint32_t TM = 160;
			const uint32_t thread_dim = 16;
			const uint32_t block_dim = m * k / (thread_dim * TM);

			float* shared_data = 0;
			cudaMalloc(&shared_data, m * thread_dim * 2 * sizeof(float));

			kernel::layernorm<T, TM> << <block_dim, thread_dim >> > (shared_data, m, k, x, weight, bias, norm_x);

			cudaFree(shared_data);
		}

		//const uint32_t TM = 48;
		// const uint32_t TM = 160;
		// const uint32_t thread_dim = 16;
		// const uint32_t block_dim = m * k / (thread_dim * TM);
		// 
		// float* shared_data = 0;
		// cudaMalloc(&shared_data, m * thread_dim * 2 * sizeof(float));
		// 
		// kernel::layernorm<T, TM><<<block_dim, thread_dim>>>(shared_data, m, k, x, weight, bias, norm_x);
		// 
		// cudaFree(shared_data);

		return cudaGetLastError();
	}

}

#endif