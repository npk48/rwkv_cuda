#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#include <stdint.h>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <nvfunctional>

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
		template<typename T_DST, typename T_SRC>
		using el_op1_t = nvstd::function<T_DST(T_SRC)>;
		template<typename T_DST, typename T_SRC>
		using el_op2_t = nvstd::function<T_DST(T_SRC, T_SRC)>;
		template<typename T_DST, typename T_SRC>
		using el_op3_t = nvstd::function<T_DST(T_SRC, T_SRC, T_SRC)>;
		template<typename T_DST, typename T_SRC>
		using el_op4_t = nvstd::function<T_DST(T_SRC, T_SRC, T_SRC, T_SRC)>;

		__device__ __forceinline__ void el_foreach8_op1_fp32_fp16(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, float* dst, half* src, el_op1_t<float, half> op)
		{
			src += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			dst += blk_id * thread_count * 8; // float 0 - 8 = 2 * float4

			float4* ld_vec4 = reinterpret_cast<float4*>(src);
			float4* st_vec4 = reinterpret_cast<float4*>(dst);

			float4 ld_val = ld_vec4[thread_id];;
			float4 st_val[2];

			half2* ld_vec_h1 = (half2*)&ld_val.x;
			half2* ld_vec_h2 = (half2*)&ld_val.y;
			half2* ld_vec_h3 = (half2*)&ld_val.z;
			half2* ld_vec_h4 = (half2*)&ld_val.w;

			st_val[0].x = op(ld_vec_h1->x);
			st_val[0].y = op(ld_vec_h1->y);
			st_val[0].z = op(ld_vec_h2->x);
			st_val[0].w = op(ld_vec_h2->y);
			st_val[1].x = op(ld_vec_h3->x);
			st_val[1].y = op(ld_vec_h3->y);
			st_val[1].z = op(ld_vec_h4->x);
			st_val[1].w = op(ld_vec_h4->y);

			st_vec4[thread_id * 2] = st_val[0];
			st_vec4[thread_id * 2 + 1] = st_val[1];
		}

		__device__ __forceinline__ void el_foreach8_op1_fp16_fp32(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, half* dst, float* src, el_op1_t<half, float> op)
		{
			src += blk_id * thread_count * 8; // float 0 - 8 = 2 * float4
			dst += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
		
			float4* ld_vec4 = reinterpret_cast<float4*>(src);
			float4* st_vec4 = reinterpret_cast<float4*>(dst);
		
			float4 ld_val[] = { ld_vec4[thread_id * 2], ld_vec4[thread_id * 2 + 1] };
			float4 st_val;
		
			half2* st_vec_h1 = (half2*)&st_val.x;
			half2* st_vec_h2 = (half2*)&st_val.y;
			half2* st_vec_h3 = (half2*)&st_val.z;
			half2* st_vec_h4 = (half2*)&st_val.w;
		
			st_vec_h1->x = op(ld_val[0].x);
			st_vec_h1->y = op(ld_val[0].y);
			st_vec_h2->x = op(ld_val[0].z);
			st_vec_h2->y = op(ld_val[0].w);
			st_vec_h3->x = op(ld_val[1].x);
			st_vec_h3->y = op(ld_val[1].y);
			st_vec_h4->x = op(ld_val[1].z);
			st_vec_h4->y = op(ld_val[1].w);

			st_vec4[thread_id] = st_val;
		}

		__device__ __forceinline__ void el_foreach8_op2_fp16_fp16(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, half* dst, half* src_a, half* src_b, el_op2_t<half, half> op)
		{
			src_a += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			src_b += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			dst += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4

			float4* ld_a_vec4 = reinterpret_cast<float4*>(src_a);
			float4* ld_b_vec4 = reinterpret_cast<float4*>(src_b);
			float4* st_vec4 = reinterpret_cast<float4*>(dst);

			float4 ld_a_val = ld_a_vec4[thread_id];
			float4 ld_b_val = ld_b_vec4[thread_id];
			float4 st_val;

			half2* ld_a_vec_h1 = (half2*)&ld_a_val.x;
			half2* ld_a_vec_h2 = (half2*)&ld_a_val.y;
			half2* ld_a_vec_h3 = (half2*)&ld_a_val.z;
			half2* ld_a_vec_h4 = (half2*)&ld_a_val.w;

			half2* ld_b_vec_h1 = (half2*)&ld_b_val.x;
			half2* ld_b_vec_h2 = (half2*)&ld_b_val.y;
			half2* ld_b_vec_h3 = (half2*)&ld_b_val.z;
			half2* ld_b_vec_h4 = (half2*)&ld_b_val.w;

			half2* st_vec_h1 = (half2*)&st_val.x;
			half2* st_vec_h2 = (half2*)&st_val.y;
			half2* st_vec_h3 = (half2*)&st_val.z;
			half2* st_vec_h4 = (half2*)&st_val.w;

			st_vec_h1->x = op(ld_a_vec_h1->x, ld_b_vec_h1->x);
			st_vec_h1->y = op(ld_a_vec_h1->y, ld_b_vec_h1->y);
			st_vec_h2->x = op(ld_a_vec_h2->x, ld_b_vec_h2->x);
			st_vec_h2->y = op(ld_a_vec_h2->y, ld_b_vec_h2->y);
			st_vec_h3->x = op(ld_a_vec_h3->x, ld_b_vec_h3->x);
			st_vec_h3->y = op(ld_a_vec_h3->y, ld_b_vec_h3->y);
			st_vec_h4->x = op(ld_a_vec_h4->x, ld_b_vec_h4->x);
			st_vec_h4->y = op(ld_a_vec_h4->y, ld_b_vec_h4->y);

			st_vec4[thread_id] = st_val;
		}

		__device__ __forceinline__ void el_foreach8_op2_fp32_fp32(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, float* dst, float* src_a, float* src_b, el_op2_t<float, float> op)
		{
			src_a += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			src_b += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			dst += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4

			float4* ld_a_vec4 = reinterpret_cast<float4*>(src_a);
			float4* ld_b_vec4 = reinterpret_cast<float4*>(src_b);
			float4* st_vec4 = reinterpret_cast<float4*>(dst);

			float4 ld_a_val[2] = { ld_a_vec4[thread_id * 2], ld_a_vec4[thread_id * 2 + 1] };
			float4 ld_b_val[2] = { ld_b_vec4[thread_id * 2], ld_b_vec4[thread_id * 2 + 1] };
			float4 st_val[2];

			st_val[0].x = op(ld_a_val[0].x, ld_b_val[0].x);
			st_val[0].y = op(ld_a_val[0].y, ld_b_val[0].y);
			st_val[0].z = op(ld_a_val[0].z, ld_b_val[0].z);
			st_val[0].w = op(ld_a_val[0].w, ld_b_val[0].w);
			st_val[1].x = op(ld_a_val[1].x, ld_b_val[1].x);
			st_val[1].y = op(ld_a_val[1].y, ld_b_val[1].y);
			st_val[1].z = op(ld_a_val[1].z, ld_b_val[1].z);
			st_val[1].w = op(ld_a_val[1].w, ld_b_val[1].w);

			st_vec4[thread_id * 2] = st_val[0];
			st_vec4[thread_id * 2 + 1] = st_val[1];
		}

		template<typename T>
		__device__ __forceinline__ void el_foreach8_op2(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, T* dst, T* src_a, T* src_b, el_op2_t<T, T> op)
		{
			if constexpr (sizeof(T) == 2)
				el_foreach8_op2_fp16_fp16(blk_id, thread_id, thread_count, dst, src_a, src_b, op);
			else
				el_foreach8_op2_fp32_fp32(blk_id, thread_id, thread_count, dst, src_a, src_b, op);
		}

		__device__ __forceinline__ void el_foreach8_op1_fp16_scalar(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, half* dst, half* src_a, half b, el_op2_t<half, half> op)
		{
			src_a += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			dst += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4

			float4* ld_a_vec4 = reinterpret_cast<float4*>(src_a);
			float4* st_vec4 = reinterpret_cast<float4*>(dst);

			float4 ld_a_val = ld_a_vec4[thread_id];
			float4 st_val;

			half2* ld_a_vec_h1 = (half2*)&ld_a_val.x;
			half2* ld_a_vec_h2 = (half2*)&ld_a_val.y;
			half2* ld_a_vec_h3 = (half2*)&ld_a_val.z;
			half2* ld_a_vec_h4 = (half2*)&ld_a_val.w;

			half2* st_vec_h1 = (half2*)&st_val.x;
			half2* st_vec_h2 = (half2*)&st_val.y;
			half2* st_vec_h3 = (half2*)&st_val.z;
			half2* st_vec_h4 = (half2*)&st_val.w;

			st_vec_h1->x = op(ld_a_vec_h1->x, b);
			st_vec_h1->y = op(ld_a_vec_h1->y, b);
			st_vec_h2->x = op(ld_a_vec_h2->x, b);
			st_vec_h2->y = op(ld_a_vec_h2->y, b);
			st_vec_h3->x = op(ld_a_vec_h3->x, b);
			st_vec_h3->y = op(ld_a_vec_h3->y, b);
			st_vec_h4->x = op(ld_a_vec_h4->x, b);
			st_vec_h4->y = op(ld_a_vec_h4->y, b);

			st_vec4[thread_id] = st_val;
		}

		__device__ __forceinline__ void el_foreach8_op1_fp32_scalar(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, float* dst, float* src_a, float b, el_op2_t<float, float> op)
		{
			src_a += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4
			dst += blk_id * thread_count * 8; // half 0 - 8 = 1 * float4

			float4* ld_a_vec4 = reinterpret_cast<float4*>(src_a);
			float4* st_vec4 = reinterpret_cast<float4*>(dst);

			float4 ld_a_val[2] = { ld_a_vec4[thread_id * 2], ld_a_vec4[thread_id * 2 + 1] };
			float4 st_val[2];

			st_val[0].x = op(ld_a_val[0].x, b);
			st_val[0].y = op(ld_a_val[0].y, b);
			st_val[0].z = op(ld_a_val[0].z, b);
			st_val[0].w = op(ld_a_val[0].w, b);
			st_val[1].x = op(ld_a_val[1].x, b);
			st_val[1].y = op(ld_a_val[1].y, b);
			st_val[1].z = op(ld_a_val[1].z, b);
			st_val[1].w = op(ld_a_val[1].w, b);

			st_vec4[thread_id * 2] = st_val[0];
			st_vec4[thread_id * 2 + 1] = st_val[1];
		}

		template<typename T>
		__device__ __forceinline__ void el_foreach8_op1_scalar(uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, T* dst, T* src_a, T b, el_op2_t<T, T> op)
		{
			if constexpr (sizeof(T) == 2)
				el_foreach8_op1_fp16_scalar(blk_id, thread_id, thread_count, dst, src_a, b, op);
			else
				el_foreach8_op1_fp32_scalar(blk_id, thread_id, thread_count, dst, src_a, b, op);
		}

		__global__ void half_to_float(float* dst, half* src, const uint32_t count)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			el_foreach8_op1_fp32_fp16(blk_id, thread_id, thread_count, dst, src, [](half val) {
				return __half2float(val);
			});
			
		}

		__global__ void float_to_half(half* dst, float* src, const uint32_t count)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			el_foreach8_op1_fp16_fp32(blk_id, thread_id, thread_count, dst, src, [](float val) {
				return __float2half(val);
			});
		}

		template<typename T>
		__global__ void element_wise_product(const uint32_t m, T* a, T* b, T* c)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			el_foreach8_op2<T>(blk_id, thread_id, thread_count, c, a, b, [](auto val_a, auto val_b) {
				return (float)val_a * (float)val_b;
			});
		}

		template<typename T>
		__global__ void element_wise_add(const uint32_t m, T* a, T* b, T* c)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			el_foreach8_op2<T>(blk_id, thread_id, thread_count, c, a, b, [](auto val_a, auto val_b) {
				return (float)val_a + (float)val_b;
			});
		}

		template<typename T>
		__global__ void element_wise_scale(const uint32_t m, T* a, float b, T* c)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			el_foreach8_op1_scalar<T>(blk_id, thread_id, thread_count, c, a, b, [](auto val_a, auto val_b) {
				return (float)val_a * (float)val_b;
			});
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
		float* fp32 = 0;

		cudaMalloc(&fp32, sizeof(float) * count);

		const uint32_t block_dim = count / 256;
		const uint32_t thread_dim = 32;

		kernel::half_to_float<<<block_dim, thread_dim>>>(fp32, d_src, count);

		cudaMemcpy(h_dst, fp32, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaFree(fp32);

		return cudaGetLastError();
	}

	template<typename T_FROM, typename T_TO>
	inline cudaError_t convert(T_FROM* from, T_TO* to, uint32_t count)
	{
		const uint32_t block_dim = count / 256;
		const uint32_t thread_dim = 32;

		if constexpr (sizeof(T_FROM) == 2 && sizeof(T_TO) == 4)
			kernel::half_to_float<<<block_dim, thread_dim >>>((float*)to, (half*)from, count);
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
		const uint32_t block_dim = count / 256;

		kernel::element_wise_product<T><<<block_dim, thread_dim>>>(count, a, b, c);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_add(const uint32_t count, T* a, T* b, T* c)
	{
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = count / 256;

		kernel::element_wise_add<T> << <block_dim, thread_dim >> > (count, a, b, c);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t element_wise_scale(const uint32_t count, T* a, float b, T* c)
	{
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = count / 256;

		kernel::element_wise_scale<T> << <block_dim, thread_dim >> > (count, a, b, c);

		return cudaGetLastError();
	}
}


#endif