#ifndef _SOFTMAX_CUH_
#define _SOFTMAX_CUH_

#include <stdint.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuda
{
	namespace kernel
	{

		inline __device__ float warp_reduce_sum(float val)
		{
			for (uint32_t mask = 32 / 2; mask > 0; mask /= 2)
				val += __shfl_xor_sync(0xffffffff, val, mask, 32);
			return val;
		}

		inline __device__ float warp_reduce_max(float val)
		{
			for (uint32_t mask = 32 / 2; mask > 0; mask /= 2)
				val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
			return val;
		}

		inline __device__ float block_reduce_sum(float val)
		{
			__shared__ float sum_shared[32];

			__shared__ float sum_broadcast;

			const uint32_t lane_id = threadIdx.x % 32;
			const uint32_t warp_id = threadIdx.x / 32;

			float warp_sum = 0;

			warp_sum = warp_reduce_sum(val);

			__syncthreads();

			if (lane_id == 0)
				sum_shared[warp_id] = warp_sum;

			__syncthreads();

			if (warp_id == 0)
			{
				if (threadIdx.x < blockDim.x / 32)
					warp_sum = sum_shared[lane_id];
				else
					warp_sum = 0.f;

				float block_sum = 0.f;

				__syncwarp();

				block_sum = warp_reduce_sum(warp_sum);

				if (lane_id == 0)
					sum_broadcast = block_sum;
			}

			__syncthreads();

			return sum_broadcast;
		}

		inline __device__ float block_reduce_max(float val)
		{
			__shared__ float max_shared[32];

			__shared__ float max_broadcast;

			const uint32_t lane_id = threadIdx.x % 32;
			const uint32_t warp_id = threadIdx.x / 32;

			float warp_max = 0;

			warp_max = warp_reduce_max(val);

			__syncthreads();

			if (lane_id == 0)
				max_shared[warp_id] = warp_max;

			__syncthreads();

			if (warp_id == 0)
			{
				if (threadIdx.x < blockDim.x / 32)
					warp_max = max_shared[lane_id];
				else
					warp_max = 0.f;

				float block_max = 0.f;

				__syncwarp();

				block_max = warp_reduce_max(warp_max);

				if (lane_id == 0)
					max_broadcast = block_max;
			}

			__syncthreads();

			return max_broadcast;
		}

		// np.exp(z)/sum(np.exp(z))

		__global__ void softmax_warp_per_row(const uint32_t m, const uint32_t k, float* x, float* norm_x)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t blk_id = blockIdx.x;

			const uint32_t lane_id = thread_id % 32;
			const uint32_t warp_id = thread_id / 32;

			const uint32_t warp_count = blockDim.x / 32;

			const uint32_t item_per_thread = k / 32;		// 8

			float thread_sum = 0.f;

			x += blk_id * warp_count * k + warp_id * k + lane_id * item_per_thread;

			float4* ld_x_vec4 = reinterpret_cast<float4*>(x);

			float4 x_cached[8];

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread / 4; ld_idx++)
			{
				float4 ld_x_val = ld_x_vec4[ld_idx];

				ld_x_val.x = exp(ld_x_val.x);
				ld_x_val.y = exp(ld_x_val.y);
				ld_x_val.z = exp(ld_x_val.z);
				ld_x_val.w = exp(ld_x_val.w);

				x_cached[ld_idx] = ld_x_val;

				thread_sum += ld_x_val.x;
				thread_sum += ld_x_val.y;
				thread_sum += ld_x_val.z;
				thread_sum += ld_x_val.w;
			}

			float warp_sum = warp_reduce_sum(thread_sum);

			norm_x += blk_id * warp_count * k + warp_id * k + lane_id * item_per_thread;
			float4* st_norm_x_vec4 = reinterpret_cast<float4*>(norm_x);

			for (uint32_t idx = 0; idx < item_per_thread / 8; idx++)
			{
				float4 ld_x_val[2] = { x_cached[idx * 2], x_cached[idx * 2 + 1] };
				float4 st_norm_x_val[2];

				st_norm_x_val[0].x = ld_x_val[0].x / warp_sum;
				st_norm_x_val[0].y = ld_x_val[0].y / warp_sum;
				st_norm_x_val[0].z = ld_x_val[0].z / warp_sum;
				st_norm_x_val[0].w = ld_x_val[0].w / warp_sum;
				st_norm_x_val[1].x = ld_x_val[1].x / warp_sum;
				st_norm_x_val[1].y = ld_x_val[1].y / warp_sum;
				st_norm_x_val[1].z = ld_x_val[1].z / warp_sum;
				st_norm_x_val[1].w = ld_x_val[1].w / warp_sum;

				st_norm_x_vec4[idx * 2] = st_norm_x_val[0];
				st_norm_x_vec4[idx * 2 + 1] = st_norm_x_val[1];
			}

		}

		__global__ void softmax_block_per_row(const uint32_t m, const uint32_t k, float* x, float* norm_x)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t blk_id = blockIdx.x;

			const uint32_t lane_id = thread_id % 32;
			const uint32_t warp_id = thread_id / 32;

			const uint32_t warp_count = blockDim.x / 32;			// 10

			const uint32_t item_per_warp = k / warp_count;			// 256
			const uint32_t item_per_thread = item_per_warp / 32;	// 8

			float thread_max = 0.f;
			float thread_sum = 0.f;

			x += blk_id * k + warp_id * item_per_warp + lane_id * item_per_thread;

			float4* ld_x_vec4 = reinterpret_cast<float4*>(x);

			// extern __shared__ float x_shared[];
			// float4* x_cached = reinterpret_cast<float4*>(x_shared + warp_id * item_per_warp + lane_id * item_per_thread);
			float4 x_cached[16];

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread / 4; ld_idx++)
			{
				float4 ld_x_val = ld_x_vec4[ld_idx];

				thread_max = max(thread_max, ld_x_val.x);
				thread_max = max(thread_max, ld_x_val.y);
				thread_max = max(thread_max, ld_x_val.z);
				thread_max = max(thread_max, ld_x_val.w);

				x_cached[ld_idx] = ld_x_val;
			}

			float block_max = block_reduce_max(thread_max);

			for (uint32_t ld_idx = 0; ld_idx < item_per_thread / 4; ld_idx++)
			{
				float4 ld_x_val = x_cached[ld_idx];

				ld_x_val.x = exp(ld_x_val.x - block_max);
				ld_x_val.y = exp(ld_x_val.y - block_max);
				ld_x_val.z = exp(ld_x_val.z - block_max);
				ld_x_val.w = exp(ld_x_val.w - block_max);

				x_cached[ld_idx] = ld_x_val;

				thread_sum += ld_x_val.x;
				thread_sum += ld_x_val.y;
				thread_sum += ld_x_val.z;
				thread_sum += ld_x_val.w;
			}

			float warp_sum = block_reduce_sum(thread_sum);

			norm_x += blk_id * k + warp_id * item_per_warp + lane_id * item_per_thread;

			float4* st_norm_x_vec4 = reinterpret_cast<float4*>(norm_x);

			for (uint32_t idx = 0; idx < item_per_thread / 8; idx++)
			{
				float4 ld_x_val[2] = { x_cached[idx * 2], x_cached[idx * 2 + 1] };
				float4 st_norm_x_val[2];

				st_norm_x_val[0].x = ld_x_val[0].x / warp_sum;
				st_norm_x_val[0].y = ld_x_val[0].y / warp_sum;
				st_norm_x_val[0].z = ld_x_val[0].z / warp_sum;
				st_norm_x_val[0].w = ld_x_val[0].w / warp_sum;
				st_norm_x_val[1].x = ld_x_val[1].x / warp_sum;
				st_norm_x_val[1].y = ld_x_val[1].y / warp_sum;
				st_norm_x_val[1].z = ld_x_val[1].z / warp_sum;
				st_norm_x_val[1].w = ld_x_val[1].w / warp_sum;

				st_norm_x_vec4[idx * 2] = st_norm_x_val[0];
				st_norm_x_vec4[idx * 2 + 1] = st_norm_x_val[1];
			}

		}

		// radix sort

		// prefix sum
	}

	inline cudaError_t softmax(float* x, float* norm_x, uint32_t m, uint32_t k)
	{
		if (k <= 1024)
			kernel::softmax_warp_per_row << <m, 32 >> > (m, k, x, norm_x);
		else
			// too large x shared
			kernel::softmax_block_per_row << <m, 32 * 32 >> > (m, k, x, norm_x);

		return cudaGetLastError();
	}

	inline void softmax_(float* x, float* norm_x, uint32_t m, uint32_t k)
	{
		float* dx = nullptr;
		cudaMalloc(&dx, m * k * sizeof(float));
		float* dnormx = nullptr; 
		cudaMalloc(&dnormx, m * k * sizeof(float));

		cudaMemcpy(dx, x, m * k * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		softmax(dx, dnormx, m, k);

		cudaMemcpy(norm_x, dnormx, m * k * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaFree(dx);
		cudaFree(dnormx);
	}
}

#endif