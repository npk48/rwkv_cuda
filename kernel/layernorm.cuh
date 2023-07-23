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
			*mean += __fdividef(delta1,*count);
			float delta2 = val - *mean;
			*m2 += delta1 * delta2;
		}

		inline __device__ void welford(float* mean, float* m2, float* count, float b_mean, float b_m2, float b_count)
		{
			float new_count = *count + b_count;
			float nb_over_n = __fdividef(b_count, new_count);
			float delta = b_mean - *mean;
			*mean += delta * nb_over_n;
			*m2 += b_m2 + delta * delta * (*count) * nb_over_n;
			*count = new_count;
		}

		inline __device__ void welford_warp_reduce(float* mean, float* m2, float* count, float thread_mean, float thread_m2, float thread_count)
		{
			*mean = thread_mean;
			*m2 = thread_m2;
			*count = thread_count;
			for (uint32_t mask = 32 / 2; mask > 0; mask /= 2) 
			{
				float b_mean = __shfl_down_sync(0xffffffff, *mean, mask, 32);
				float b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, 32);
				float b_count = __shfl_down_sync(0xffffffff, *count, mask, 32);
				welford(mean, m2, count, b_mean, b_m2, b_count);
			}

			*mean = __shfl_sync(0xffffffff, *mean, 0, 32);
			*m2 = __shfl_sync(0xffffffff, *m2, 0, 32);
			*count = __shfl_sync(0xffffffff, *count, 0, 32);
		}

		inline __device__ void welford_block_reduce(float* mean, float* m2, float* count, float thread_mean, float thread_m2, float thread_count)
		{
			__shared__ float mean_shared[32];
			__shared__ float m2_shared[32];
			__shared__ float count_shared[32];

			__shared__ float mean_result_broadcast;
			__shared__ float m2_result_broadcast;
			__shared__ float count_result_broadcast;

			const uint32_t lane_id = threadIdx.x % 32;
			const uint32_t warp_id = threadIdx.x / 32;

			float warp_mean = 0;
			float warp_m2 = 0;
			float warp_count = 0;

			welford_warp_reduce(&warp_mean, &warp_m2, &warp_count, thread_mean, thread_m2, thread_count);

			__syncthreads();

			if (lane_id == 0)
			{
				mean_shared[warp_id] = warp_mean;
				m2_shared[warp_id] = warp_m2;
				count_shared[warp_id] = warp_count;
			}

			__syncthreads();

			if (warp_id == 0)
			{
				if (threadIdx.x < blockDim.x / 32)
				{
					warp_mean = mean_shared[lane_id];
					warp_m2 = m2_shared[lane_id];
					warp_count = count_shared[lane_id];
				}
				else
				{
					warp_mean = 0.f;
					warp_m2 = 0.f;
					warp_count = 1.f; // division by zero fix
				}

				float block_mean = 0.f;
				float block_m2 = 0.f;
				float block_count = 0.f;

				__syncwarp();

				welford_warp_reduce(&block_mean, &block_m2, &block_count, warp_mean, warp_m2, warp_count);

				if (lane_id == 0)
				{
					mean_result_broadcast = block_mean;
					m2_result_broadcast = block_m2;
					count_result_broadcast = block_count - 32 + blockDim.x / 32;
				}
			}

			__syncthreads();

			*mean = mean_result_broadcast;
			*m2 = m2_result_broadcast;
			*count = count_result_broadcast;
		}

		// k <= 1024
		__global__ void layernorm_warp_per_row(const uint32_t m, const uint32_t k, float* x, half* weight, half* bias, float* norm_x)
		{
			const float eps = 1e-05;

			const uint32_t thread_id = threadIdx.x;
			const uint32_t blk_id = blockIdx.x;

			const uint32_t lane_id = thread_id % 32;
			const uint32_t warp_id = thread_id / 32;

			const uint32_t warp_count = blockDim.x / 32;

			const uint32_t item_per_thread = k / 32;		// 8

			float thread_mean = 0.f;
			float thread_m2 = 0.f;
			float thread_count = 0.f;

			x += blk_id * warp_count * k + warp_id * k + lane_id * item_per_thread;

			float4* ld_x_vec4 = reinterpret_cast<float4*>(x);

			float4 x_cached[8];

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread/4; ld_idx++)
			{
				float4 ld_x_val = ld_x_vec4[ld_idx];
				x_cached[ld_idx] = ld_x_val;

				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.x);
				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.y);
				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.z);
				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.w);
			}

			welford_warp_reduce(&thread_mean, &thread_m2, &thread_count, thread_mean, thread_m2, thread_count);

			float thread_denom = __frsqrt_rn(eps + __fdividef(thread_m2, thread_count));

			weight += lane_id * item_per_thread;
			bias += lane_id * item_per_thread;
			norm_x += blk_id * warp_count * k + warp_id * k + lane_id * item_per_thread;
			
			float4* ld_weight_vec4 = reinterpret_cast<float4*>(weight);
			float4* ld_bias_vec4 = reinterpret_cast<float4*>(bias);
			float4* st_norm_x_vec4 = reinterpret_cast<float4*>(norm_x);
			
			for (uint32_t idx = 0; idx < item_per_thread / 8; idx++)
			{
				//float4 ld_x_val[2] = { ld_x_vec4[idx * 2], ld_x_vec4[idx * 2 + 1] };
				float4 ld_x_val[2] = { x_cached[idx * 2], x_cached[idx * 2 + 1] };

				float4 ld_weight_val = ld_weight_vec4[idx];
				float4 ld_bias_val = ld_bias_vec4[idx];
				float4 st_norm_x_val[2];

				half2* ld_weight_vec_h1 = reinterpret_cast<half2*>(&ld_weight_val.x);
				half2* ld_weight_vec_h2 = reinterpret_cast<half2*>(&ld_weight_val.y);
				half2* ld_weight_vec_h3 = reinterpret_cast<half2*>(&ld_weight_val.z);
				half2* ld_weight_vec_h4 = reinterpret_cast<half2*>(&ld_weight_val.w);

				half2* ld_bias_vec_h1 = reinterpret_cast<half2*>(&ld_bias_val.x);
				half2* ld_bias_vec_h2 = reinterpret_cast<half2*>(&ld_bias_val.y);
				half2* ld_bias_vec_h3 = reinterpret_cast<half2*>(&ld_bias_val.z);
				half2* ld_bias_vec_h4 = reinterpret_cast<half2*>(&ld_bias_val.w);

				st_norm_x_val[0].x = (float)(ld_bias_vec_h1->x) + (float)(ld_weight_vec_h1->x) * (ld_x_val[0].x - thread_mean) * thread_denom;
				st_norm_x_val[0].y = (float)(ld_bias_vec_h1->y) + (float)(ld_weight_vec_h1->y) * (ld_x_val[0].y - thread_mean) * thread_denom;
				st_norm_x_val[0].z = (float)(ld_bias_vec_h2->x) + (float)(ld_weight_vec_h2->x) * (ld_x_val[0].z - thread_mean) * thread_denom;
				st_norm_x_val[0].w = (float)(ld_bias_vec_h2->y) + (float)(ld_weight_vec_h2->y) * (ld_x_val[0].w - thread_mean) * thread_denom;
				st_norm_x_val[1].x = (float)(ld_bias_vec_h3->x) + (float)(ld_weight_vec_h3->x) * (ld_x_val[1].x - thread_mean) * thread_denom;
				st_norm_x_val[1].y = (float)(ld_bias_vec_h3->y) + (float)(ld_weight_vec_h3->y) * (ld_x_val[1].y - thread_mean) * thread_denom;
				st_norm_x_val[1].z = (float)(ld_bias_vec_h4->x) + (float)(ld_weight_vec_h4->x) * (ld_x_val[1].z - thread_mean) * thread_denom;
				st_norm_x_val[1].w = (float)(ld_bias_vec_h4->y) + (float)(ld_weight_vec_h4->y) * (ld_x_val[1].w - thread_mean) * thread_denom;
			
				st_norm_x_vec4[idx * 2] = st_norm_x_val[0];
				st_norm_x_vec4[idx * 2 + 1] = st_norm_x_val[1];
			}
		}

		// k > 1024 several warp for one row
		__global__ void layernorm_block_per_row(const uint32_t m, const uint32_t k, float* x, half* weight, half* bias, float* norm_x)
		{
			const float eps = 1e-05;

			const uint32_t thread_id = threadIdx.x;
			const uint32_t blk_id = blockIdx.x;

			const uint32_t lane_id = thread_id % 32;
			const uint32_t warp_id = thread_id / 32;

			const uint32_t warp_count = blockDim.x / 32;			// 10

			const uint32_t item_per_warp = k / warp_count;			// 256
			const uint32_t item_per_thread = item_per_warp / 32;	// 8

			float thread_mean = 0.f;
			float thread_m2 = 0.f;
			float thread_count = 0.f;

			x += blk_id * k + warp_id * item_per_warp + lane_id * item_per_thread;

			float4* ld_x_vec4 = reinterpret_cast<float4*>(x);

			extern __shared__ float x_shared[];
			float4* x_cached = reinterpret_cast<float4*>(x_shared + warp_id * item_per_warp + lane_id * item_per_thread);

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread / 4; ld_idx++)
			{
				float4 ld_x_val = ld_x_vec4[ld_idx];
				x_cached[ld_idx] = ld_x_val;

				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.x);
				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.y);
				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.z);
				welford(&thread_mean, &thread_m2, &thread_count, ld_x_val.w);
			}

			welford_block_reduce(&thread_mean, &thread_m2, &thread_count, thread_mean, thread_m2, thread_count);

			float thread_denom = __frsqrt_rn(eps + __fdividef(thread_m2, thread_count));

			weight += warp_id * item_per_warp + lane_id * item_per_thread;
			bias += warp_id * item_per_warp + lane_id * item_per_thread;
			norm_x += blk_id * k + warp_id * item_per_warp + lane_id * item_per_thread;

			float4* ld_weight_vec4 = reinterpret_cast<float4*>(weight);
			float4* ld_bias_vec4 = reinterpret_cast<float4*>(bias);
			float4* st_norm_x_vec4 = reinterpret_cast<float4*>(norm_x);

			for (uint32_t idx = 0; idx < item_per_thread / 8; idx++)
			{
				float4 ld_x_val[2] = { x_cached[idx * 2], x_cached[idx * 2 + 1] };

				float4 ld_weight_val = ld_weight_vec4[idx];
				float4 ld_bias_val = ld_bias_vec4[idx];
				float4 st_norm_x_val[2];

				half2* ld_weight_vec_h1 = reinterpret_cast<half2*>(&ld_weight_val.x);
				half2* ld_weight_vec_h2 = reinterpret_cast<half2*>(&ld_weight_val.y);
				half2* ld_weight_vec_h3 = reinterpret_cast<half2*>(&ld_weight_val.z);
				half2* ld_weight_vec_h4 = reinterpret_cast<half2*>(&ld_weight_val.w);

				half2* ld_bias_vec_h1 = reinterpret_cast<half2*>(&ld_bias_val.x);
				half2* ld_bias_vec_h2 = reinterpret_cast<half2*>(&ld_bias_val.y);
				half2* ld_bias_vec_h3 = reinterpret_cast<half2*>(&ld_bias_val.z);
				half2* ld_bias_vec_h4 = reinterpret_cast<half2*>(&ld_bias_val.w);

				st_norm_x_val[0].x = (float)(ld_bias_vec_h1->x) + (float)(ld_weight_vec_h1->x) * (ld_x_val[0].x - thread_mean) * thread_denom;
				st_norm_x_val[0].y = (float)(ld_bias_vec_h1->y) + (float)(ld_weight_vec_h1->y) * (ld_x_val[0].y - thread_mean) * thread_denom;
				st_norm_x_val[0].z = (float)(ld_bias_vec_h2->x) + (float)(ld_weight_vec_h2->x) * (ld_x_val[0].z - thread_mean) * thread_denom;
				st_norm_x_val[0].w = (float)(ld_bias_vec_h2->y) + (float)(ld_weight_vec_h2->y) * (ld_x_val[0].w - thread_mean) * thread_denom;
				st_norm_x_val[1].x = (float)(ld_bias_vec_h3->x) + (float)(ld_weight_vec_h3->x) * (ld_x_val[1].x - thread_mean) * thread_denom;
				st_norm_x_val[1].y = (float)(ld_bias_vec_h3->y) + (float)(ld_weight_vec_h3->y) * (ld_x_val[1].y - thread_mean) * thread_denom;
				st_norm_x_val[1].z = (float)(ld_bias_vec_h4->x) + (float)(ld_weight_vec_h4->x) * (ld_x_val[1].z - thread_mean) * thread_denom;
				st_norm_x_val[1].w = (float)(ld_bias_vec_h4->y) + (float)(ld_weight_vec_h4->y) * (ld_x_val[1].w - thread_mean) * thread_denom;

				st_norm_x_vec4[idx * 2] = st_norm_x_val[0];
				st_norm_x_vec4[idx * 2 + 1] = st_norm_x_val[1];
			}
		}

		// k <= 1024
		__global__ void layernorm_fp16_warp_per_row(const uint32_t m, const uint32_t k, half* x, half* weight, half* bias, half* norm_x)
		{
			const float eps = 1e-05;

			const uint32_t thread_id = threadIdx.x;
			const uint32_t blk_id = blockIdx.x;

			const uint32_t lane_id = thread_id % 32;
			const uint32_t warp_id = thread_id / 32;

			const uint32_t warp_count = blockDim.x / 32;

			const uint32_t item_per_thread = k / 32;		// 8

			float thread_mean = 0.f;
			float thread_m2 = 0.f;
			float thread_count = 0.f;

			x += blk_id * warp_count * k + warp_id * k + lane_id * item_per_thread;

			float4* ld_x_vec4 = reinterpret_cast<float4*>(x);

			float4 x_cached[8];

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread / 8; ld_idx++)
			{
				float4 ld_x_val = ld_x_vec4[ld_idx];
				x_cached[ld_idx] = ld_x_val;

				half2* ld_x_vec4_h1 = reinterpret_cast<half2*>(&ld_x_val.x);
				half2* ld_x_vec4_h2 = reinterpret_cast<half2*>(&ld_x_val.y);
				half2* ld_x_vec4_h3 = reinterpret_cast<half2*>(&ld_x_val.z);
				half2* ld_x_vec4_h4 = reinterpret_cast<half2*>(&ld_x_val.w);

				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h1->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h1->y);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h2->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h2->y);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h3->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h3->y);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h4->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h4->y);
			}

			welford_warp_reduce(&thread_mean, &thread_m2, &thread_count, thread_mean, thread_m2, thread_count);

			float thread_denom = __frsqrt_rn(eps + __fdividef(thread_m2, thread_count));

			weight += lane_id * item_per_thread;
			bias += lane_id * item_per_thread;
			norm_x += blk_id * warp_count * k + warp_id * k + lane_id * item_per_thread;

			float4* ld_weight_vec4 = reinterpret_cast<float4*>(weight);
			float4* ld_bias_vec4 = reinterpret_cast<float4*>(bias);
			float4* st_norm_x_vec4 = reinterpret_cast<float4*>(norm_x);

			for (uint32_t idx = 0; idx < item_per_thread / 8; idx++)
			{
				float4 ld_x_val = x_cached[idx];

				float4 ld_weight_val = ld_weight_vec4[idx];
				float4 ld_bias_val = ld_bias_vec4[idx];
				float4 st_norm_x_val;

				half2* ld_x_vec_h1 = reinterpret_cast<half2*>(&ld_x_val.x);
				half2* ld_x_vec_h2 = reinterpret_cast<half2*>(&ld_x_val.y);
				half2* ld_x_vec_h3 = reinterpret_cast<half2*>(&ld_x_val.z);
				half2* ld_x_vec_h4 = reinterpret_cast<half2*>(&ld_x_val.w);

				half2* ld_weight_vec_h1 = reinterpret_cast<half2*>(&ld_weight_val.x);
				half2* ld_weight_vec_h2 = reinterpret_cast<half2*>(&ld_weight_val.y);
				half2* ld_weight_vec_h3 = reinterpret_cast<half2*>(&ld_weight_val.z);
				half2* ld_weight_vec_h4 = reinterpret_cast<half2*>(&ld_weight_val.w);

				half2* ld_bias_vec_h1 = reinterpret_cast<half2*>(&ld_bias_val.x);
				half2* ld_bias_vec_h2 = reinterpret_cast<half2*>(&ld_bias_val.y);
				half2* ld_bias_vec_h3 = reinterpret_cast<half2*>(&ld_bias_val.z);
				half2* ld_bias_vec_h4 = reinterpret_cast<half2*>(&ld_bias_val.w);

				half2* st_norm_x_vec_h1 = reinterpret_cast<half2*>(&st_norm_x_val.x);
				half2* st_norm_x_vec_h2 = reinterpret_cast<half2*>(&st_norm_x_val.y);
				half2* st_norm_x_vec_h3 = reinterpret_cast<half2*>(&st_norm_x_val.z);
				half2* st_norm_x_vec_h4 = reinterpret_cast<half2*>(&st_norm_x_val.w);

				st_norm_x_vec_h1->x = (half)((float)(ld_bias_vec_h1->x) + (float)(ld_weight_vec_h1->x) * ((float)ld_x_vec_h1->x - thread_mean) * thread_denom);
				st_norm_x_vec_h1->y = (half)((float)(ld_bias_vec_h1->y) + (float)(ld_weight_vec_h1->y) * ((float)ld_x_vec_h1->y - thread_mean) * thread_denom);
				st_norm_x_vec_h2->x = (half)((float)(ld_bias_vec_h2->x) + (float)(ld_weight_vec_h2->x) * ((float)ld_x_vec_h2->x - thread_mean) * thread_denom);
				st_norm_x_vec_h2->y = (half)((float)(ld_bias_vec_h2->y) + (float)(ld_weight_vec_h2->y) * ((float)ld_x_vec_h2->y - thread_mean) * thread_denom);
				st_norm_x_vec_h3->x = (half)((float)(ld_bias_vec_h3->x) + (float)(ld_weight_vec_h3->x) * ((float)ld_x_vec_h3->x - thread_mean) * thread_denom);
				st_norm_x_vec_h3->y = (half)((float)(ld_bias_vec_h3->y) + (float)(ld_weight_vec_h3->y) * ((float)ld_x_vec_h3->y - thread_mean) * thread_denom);
				st_norm_x_vec_h4->x = (half)((float)(ld_bias_vec_h4->x) + (float)(ld_weight_vec_h4->x) * ((float)ld_x_vec_h4->x - thread_mean) * thread_denom);
				st_norm_x_vec_h4->y = (half)((float)(ld_bias_vec_h4->y) + (float)(ld_weight_vec_h4->y) * ((float)ld_x_vec_h4->y - thread_mean) * thread_denom);

				st_norm_x_vec4[idx] = st_norm_x_val;
			}
		}

		// k > 1024 several warp for one row
		__global__ void layernorm_fp16_block_per_row(const uint32_t m, const uint32_t k, half* x, half* weight, half* bias, half* norm_x)
		{
			const float eps = 1e-05;

			const uint32_t thread_id = threadIdx.x;
			const uint32_t blk_id = blockIdx.x;

			const uint32_t lane_id = thread_id % 32;
			const uint32_t warp_id = thread_id / 32;

			const uint32_t warp_count = blockDim.x / 32;			// 10

			const uint32_t item_per_warp = k / warp_count;			// 256
			const uint32_t item_per_thread = item_per_warp / 32;	// 8

			float thread_mean = 0.f;
			float thread_m2 = 0.f;
			float thread_count = 0.f;

			x += blk_id * k + warp_id * item_per_warp + lane_id * item_per_thread;

			float4* ld_x_vec4 = reinterpret_cast<float4*>(x);

			extern __shared__ float x_shared[];
			float4* x_cached = reinterpret_cast<float4*>(x_shared + (warp_id * item_per_warp + lane_id * item_per_thread)/2);

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread / 8; ld_idx++)
			{
				float4 ld_x_val = ld_x_vec4[ld_idx];
				x_cached[ld_idx] = ld_x_val;

				half2* ld_x_vec4_h1 = reinterpret_cast<half2*>(&ld_x_val.x);
				half2* ld_x_vec4_h2 = reinterpret_cast<half2*>(&ld_x_val.y);
				half2* ld_x_vec4_h3 = reinterpret_cast<half2*>(&ld_x_val.z);
				half2* ld_x_vec4_h4 = reinterpret_cast<half2*>(&ld_x_val.w);

				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h1->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h1->y);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h2->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h2->y);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h3->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h3->y);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h4->x);
				welford(&thread_mean, &thread_m2, &thread_count, (float)ld_x_vec4_h4->y);
			}

			welford_block_reduce(&thread_mean, &thread_m2, &thread_count, thread_mean, thread_m2, thread_count);

			float thread_denom = __frsqrt_rn(eps + __fdividef(thread_m2, thread_count));

			weight += warp_id * item_per_warp + lane_id * item_per_thread;
			bias += warp_id * item_per_warp + lane_id * item_per_thread;
			norm_x += blk_id * k + warp_id * item_per_warp + lane_id * item_per_thread;

			float4* ld_weight_vec4 = reinterpret_cast<float4*>(weight);
			float4* ld_bias_vec4 = reinterpret_cast<float4*>(bias);
			float4* st_norm_x_vec4 = reinterpret_cast<float4*>(norm_x);

			for (uint32_t idx = 0; idx < item_per_thread / 8; idx++)
			{
				float4 ld_x_val = x_cached[idx];

				float4 ld_weight_val = ld_weight_vec4[idx];
				float4 ld_bias_val = ld_bias_vec4[idx];
				float4 st_norm_x_val;

				half2* ld_x_vec_h1 = reinterpret_cast<half2*>(&ld_x_val.x);
				half2* ld_x_vec_h2 = reinterpret_cast<half2*>(&ld_x_val.y);
				half2* ld_x_vec_h3 = reinterpret_cast<half2*>(&ld_x_val.z);
				half2* ld_x_vec_h4 = reinterpret_cast<half2*>(&ld_x_val.w);

				half2* ld_weight_vec_h1 = reinterpret_cast<half2*>(&ld_weight_val.x);
				half2* ld_weight_vec_h2 = reinterpret_cast<half2*>(&ld_weight_val.y);
				half2* ld_weight_vec_h3 = reinterpret_cast<half2*>(&ld_weight_val.z);
				half2* ld_weight_vec_h4 = reinterpret_cast<half2*>(&ld_weight_val.w);

				half2* ld_bias_vec_h1 = reinterpret_cast<half2*>(&ld_bias_val.x);
				half2* ld_bias_vec_h2 = reinterpret_cast<half2*>(&ld_bias_val.y);
				half2* ld_bias_vec_h3 = reinterpret_cast<half2*>(&ld_bias_val.z);
				half2* ld_bias_vec_h4 = reinterpret_cast<half2*>(&ld_bias_val.w);

				half2* st_norm_x_vec_h1 = reinterpret_cast<half2*>(&st_norm_x_val.x);
				half2* st_norm_x_vec_h2 = reinterpret_cast<half2*>(&st_norm_x_val.y);
				half2* st_norm_x_vec_h3 = reinterpret_cast<half2*>(&st_norm_x_val.z);
				half2* st_norm_x_vec_h4 = reinterpret_cast<half2*>(&st_norm_x_val.w);

				st_norm_x_vec_h1->x = (half)((float)(ld_bias_vec_h1->x) + (float)(ld_weight_vec_h1->x) * ((float)ld_x_vec_h1->x - thread_mean) * thread_denom);
				st_norm_x_vec_h1->y = (half)((float)(ld_bias_vec_h1->y) + (float)(ld_weight_vec_h1->y) * ((float)ld_x_vec_h1->y - thread_mean) * thread_denom);
				st_norm_x_vec_h2->x = (half)((float)(ld_bias_vec_h2->x) + (float)(ld_weight_vec_h2->x) * ((float)ld_x_vec_h2->x - thread_mean) * thread_denom);
				st_norm_x_vec_h2->y = (half)((float)(ld_bias_vec_h2->y) + (float)(ld_weight_vec_h2->y) * ((float)ld_x_vec_h2->y - thread_mean) * thread_denom);
				st_norm_x_vec_h3->x = (half)((float)(ld_bias_vec_h3->x) + (float)(ld_weight_vec_h3->x) * ((float)ld_x_vec_h3->x - thread_mean) * thread_denom);
				st_norm_x_vec_h3->y = (half)((float)(ld_bias_vec_h3->y) + (float)(ld_weight_vec_h3->y) * ((float)ld_x_vec_h3->y - thread_mean) * thread_denom);
				st_norm_x_vec_h4->x = (half)((float)(ld_bias_vec_h4->x) + (float)(ld_weight_vec_h4->x) * ((float)ld_x_vec_h4->x - thread_mean) * thread_denom);
				st_norm_x_vec_h4->y = (half)((float)(ld_bias_vec_h4->y) + (float)(ld_weight_vec_h4->y) * ((float)ld_x_vec_h4->y - thread_mean) * thread_denom);

				st_norm_x_vec4[idx] = st_norm_x_val;
			}
		}

	}

	template<typename T>
	inline cudaError_t layernorm(T* x, half* weight, half* bias, T* norm_x, uint32_t m, uint32_t k)
	{
		if constexpr (sizeof(T) == 2)
		{
			if (k <= 1024)
				kernel::layernorm_fp16_warp_per_row<<<m, 32>>>(m, k, x, weight, bias, norm_x);
			else
				kernel::layernorm_fp16_block_per_row<<<m, 320, k*sizeof(half)>>>(m, k, x, weight, bias, norm_x);
		}
		else
		{
			if (k <= 1024)
				kernel::layernorm_warp_per_row << <m, 32 >> > (m, k, x, weight, bias, norm_x);
			else
				kernel::layernorm_block_per_row << <m, 320, k * sizeof(float) >> > (m, k, x, weight, bias, norm_x);
		}

		return cudaGetLastError();
	}

}

#endif