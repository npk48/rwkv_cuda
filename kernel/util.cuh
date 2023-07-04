#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#include <stdint.h>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdio.h>

namespace cuda
{
	// compute type
#if __CUDA_ARCH__ < 700
	typedef float CT;
#else
	typedef half CT;
#endif

	namespace kernel
	{
		__global__ void half_to_float(float* dst, half* src, const uint32_t count)
		{
			uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

			if (idx < count)
				dst[idx] = __half2float(src[idx]);
		}

		__global__ void float_to_half(half* dst, float* src, const uint32_t count)
		{
			uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

			if (idx < count)
				dst[idx] = __float2half(src[idx]);
		}

		template<typename T, const uint32_t TM>
		__global__ void element_wise_product(const uint32_t m, T* a, T* b, T* c)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t block_id = blockIdx.x;
			const uint32_t thread_num = blockDim.x;

			T a_tile[TM];
			T b_tile[TM];
			T c_tile[TM];

			a += (block_id * thread_num + thread_id) * TM;
			b += (block_id * thread_num + thread_id) * TM;
			c += (block_id * thread_num + thread_id) * TM;

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					a_tile[ld_idx + i] = a[ld_idx + i];
					b_tile[ld_idx + i] = b[ld_idx + i];
				}
			}

#pragma unroll
			for (uint32_t p_idx = 0; p_idx < TM; p_idx++)
				c_tile[p_idx] = (T)((float)a_tile[p_idx] * (float)b_tile[p_idx]);

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					c[ld_idx + i] = c_tile[ld_idx + i];
			}

		}

		template<typename T, const uint32_t TM>
		__global__ void element_wise_add(const uint32_t m, T* a, T* b, T* c)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t block_id = blockIdx.x;
			const uint32_t thread_num = blockDim.x;

			T a_tile[TM];
			T b_tile[TM];
			T c_tile[TM];

			a += (block_id * thread_num + thread_id) * TM;
			b += (block_id * thread_num + thread_id) * TM;
			c += (block_id * thread_num + thread_id) * TM;

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
				{
					a_tile[ld_idx + i] = a[ld_idx + i];
					b_tile[ld_idx + i] = b[ld_idx + i];
				}
			}

#pragma unroll
			for (uint32_t p_idx = 0; p_idx < TM; p_idx++)
				c_tile[p_idx] = (T)((float)a_tile[p_idx] + (float)b_tile[p_idx]);

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					c[ld_idx + i] = c_tile[ld_idx + i];
			}

		}

		template<typename T, const uint32_t TM>
		__global__ void element_wise_scale(const uint32_t m, T* a, float b, T* c)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t block_id = blockIdx.x;
			const uint32_t thread_num = blockDim.x;

			T a_tile[TM];
			T c_tile[TM];

			a += (block_id * thread_num + thread_id) * TM;
			c += (block_id * thread_num + thread_id) * TM;

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					a_tile[ld_idx + i] = a[ld_idx + i];

#pragma unroll
			for (uint32_t p_idx = 0; p_idx < TM; p_idx++)
				c_tile[p_idx] = (T)((float)a_tile[p_idx] * (float)b);

#pragma unroll
			for (uint32_t ld_idx = 0; ld_idx < TM; ld_idx += 8)
			{
#pragma unroll
				for (uint32_t i = 0; i < 8; i++)
					c[ld_idx + i] = c_tile[ld_idx + i];
			}

		}
		
		template<typename T>
		__global__ void transpose(T* from, T* to, uint32_t width, uint32_t height)
		{
		    __shared__ T block[16][16 + 1];
		
		    uint32_t x_index = blockIdx.x * 16 + threadIdx.x;
		    uint32_t y_index = blockIdx.y * 16 + threadIdx.y;
		
		    if ((x_index < width) && (y_index < height))
		    {
		        uint32_t index_in = y_index * width + x_index;
		        block[threadIdx.y][threadIdx.x] = from[index_in];
		    }
		
		    __syncthreads();
		
		    x_index = blockIdx.y * 16 + threadIdx.x;
		    y_index = blockIdx.x * 16 + threadIdx.y;
		
		    if ((x_index < height) && (y_index < width))
		    {
		        uint32_t index_out = y_index * height + x_index;
		        to[index_out] = block[threadIdx.x][threadIdx.y];
		    }
		}
	}

	inline cudaError_t dump_fp16(float* h_dst, half* d_src, uint32_t count)
	{
		float* fp32;

		assert(cudaMalloc(&fp32, sizeof(float) * count) == cudaSuccess);

		const uint32_t block_dim = (count * 16 + 15) / 16;
		const uint32_t thread_dim = 16;

		kernel::half_to_float<<<block_dim, thread_dim>>>(fp32, d_src, count);

		assert(cudaMemcpy(h_dst, fp32, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyDeviceToHost) == cudaSuccess);

		assert(cudaFree(fp32) == cudaSuccess);

		return cudaGetLastError();
	}

	template<typename T_FROM, typename T_TO>
	inline cudaError_t convert(T_FROM* from, T_TO* to, uint32_t count)
	{
		const uint32_t block_dim = (count * 16 + 15) / 16;
		const uint32_t thread_dim = 16;

		if constexpr (sizeof(T_FROM) == 2 && sizeof(T_TO) == 4)
			kernel::half_to_float<<<block_dim, thread_dim>>>((float*)to, (half*)from, count);
		else if (sizeof(T_FROM) == 4 && sizeof(T_TO) == 2)
			kernel::float_to_half<<<block_dim, thread_dim>>>((half*)to, (float*)from, count);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t transpose(const uint32_t m, const uint32_t k, T* from, T* to)
	{
		dim3 grid((k + 15) / 16, (m + 15) / 16, 1);
		dim3 threads(16, 16, 1);

		if constexpr (sizeof(T) == 2)
			kernel::transpose<half><<<grid, threads>>> ((half*)from, (half*)to, k, m);
		else
			kernel::transpose<float><<<grid, threads>>> ((float*)from, (float*)to, k, m);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_product(const uint32_t count, T* a, T* b, T* c)
	{
		const uint32_t TM = 48;
		const uint32_t thread_dim = 16;
		const uint32_t block_dim = count / (thread_dim * TM);

		kernel::element_wise_product<T, TM><<<block_dim, thread_dim>>>(count, a, b, c);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_add(const uint32_t count, T* a, T* b, T* c)
	{
		const uint32_t TM = 48;
		const uint32_t thread_dim = 16;
		const uint32_t block_dim = count / (thread_dim * TM);

		kernel::element_wise_add<T, TM> << <block_dim, thread_dim >> > (count, a, b, c);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_scale(const uint32_t count, T* a, float b, T* c)
	{
		const uint32_t TM = 48;
		const uint32_t thread_dim = 16;
		const uint32_t block_dim = count / (thread_dim * TM);

		kernel::element_wise_scale<T, TM> << <block_dim, thread_dim >> > (count, a, b, c);

		return cudaGetLastError();
	}
}


#endif