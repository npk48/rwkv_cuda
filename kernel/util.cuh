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
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < count)
				dst[idx] = __half2float(src[idx]);
		}

		__global__ void float_to_half(half* dst, float* src, const uint32_t count)
		{
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < count)
				dst[idx] = __float2half(src[idx]);
		}

		template<typename T>
		__global__ void element_wise_product(const uint32_t m, T* a, T* b, T* c)
		{
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx >= m)
				return;

			c[idx] = (T)((float)a[idx] * (float)b[idx]);

		}

		template<typename T>
		__global__ void element_wise_add(const uint32_t m, T* a, T* b, T* c)
		{
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx >= m)
				return;

			c[idx] = (T)((float)a[idx] + (float)b[idx]);

		}

		template<typename T>
		__global__ void element_wise_scale(const uint32_t m, T* a, float b, T* c)
		{
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx >= m)
				return;

			c[idx] = (T)((float)a[idx] * (float)b);

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

		const uint32_t block_dim = (count + 15) / 16;
		const uint32_t thread_dim = 16;

		kernel::half_to_float<<<block_dim, thread_dim>>>(fp32, d_src, count);

		assert(cudaMemcpy(h_dst, fp32, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyDeviceToHost) == cudaSuccess);

		assert(cudaFree(fp32) == cudaSuccess);

		return cudaGetLastError();
	}

	template<typename T_FROM, typename T_TO>
	inline cudaError_t convert(T_FROM* from, T_TO* to, uint32_t count)
	{
		const uint32_t block_dim = (count + 15) / 16;
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
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = (count + thread_dim - 1) / thread_dim;

		kernel::element_wise_product<T><<<block_dim, thread_dim>>>(count, a, b, c);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_add(const uint32_t count, T* a, T* b, T* c)
	{
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = (count + thread_dim - 1) / thread_dim;

		kernel::element_wise_add<T> << <block_dim, thread_dim >> > (count, a, b, c);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_scale(const uint32_t count, T* a, float b, T* c)
	{
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = (count + thread_dim - 1) / thread_dim;

		kernel::element_wise_scale<T> << <block_dim, thread_dim >> > (count, a, b, c);

		return cudaGetLastError();
	}
}


#endif