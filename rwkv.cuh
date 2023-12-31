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
		__device__ __forceinline__ void mix_rkv_fp16(
			const uint32_t k, uint32_t blk_id, uint32_t thread_id, uint32_t thread_count, 
			half* mix_r, half* mix_k, half* mix_v, 
			half* xx, half* sx, 
			half* rx, half* kx, half* vx
		)
		{
			mix_r += (blk_id * thread_count * 8) % k;
			mix_k += (blk_id * thread_count * 8) % k;
			mix_v += (blk_id * thread_count * 8) % k;

			xx += blk_id * thread_count * 8;
			sx += blk_id * thread_count * 8;

			rx += blk_id * thread_count * 8;
			kx += blk_id * thread_count * 8;
			vx += blk_id * thread_count * 8;

			float4* ld_mix_r_vec4 = reinterpret_cast<float4*>(mix_r);
			float4* ld_mix_k_vec4 = reinterpret_cast<float4*>(mix_k);
			float4* ld_mix_v_vec4 = reinterpret_cast<float4*>(mix_v);

			float4* ld_xx_vec4 = reinterpret_cast<float4*>(xx);
			float4* ld_sx_vec4 = reinterpret_cast<float4*>(sx);

			float4* st_rx_vec4 = reinterpret_cast<float4*>(rx);
			float4* st_kx_vec4 = reinterpret_cast<float4*>(kx);
			float4* st_vx_vec4 = reinterpret_cast<float4*>(vx);

			float4 ld_mix_r_val = ld_mix_r_vec4[thread_id];
			float4 ld_mix_k_val = ld_mix_k_vec4[thread_id];
			float4 ld_mix_v_val = ld_mix_v_vec4[thread_id];

			float4 ld_xx_val = ld_xx_vec4[thread_id];
			float4 ld_sx_val = ld_sx_vec4[thread_id];

			float4 st_rx_val;
			float4 st_kx_val;
			float4 st_vx_val;

			half2* ld_mix_r_vec_h1 = (half2*)&ld_mix_r_val.x;
			half2* ld_mix_r_vec_h2 = (half2*)&ld_mix_r_val.y;
			half2* ld_mix_r_vec_h3 = (half2*)&ld_mix_r_val.z;
			half2* ld_mix_r_vec_h4 = (half2*)&ld_mix_r_val.w;

			half2* ld_mix_k_vec_h1 = (half2*)&ld_mix_k_val.x;
			half2* ld_mix_k_vec_h2 = (half2*)&ld_mix_k_val.y;
			half2* ld_mix_k_vec_h3 = (half2*)&ld_mix_k_val.z;
			half2* ld_mix_k_vec_h4 = (half2*)&ld_mix_k_val.w;

			half2* ld_mix_v_vec_h1 = (half2*)&ld_mix_v_val.x;
			half2* ld_mix_v_vec_h2 = (half2*)&ld_mix_v_val.y;
			half2* ld_mix_v_vec_h3 = (half2*)&ld_mix_v_val.z;
			half2* ld_mix_v_vec_h4 = (half2*)&ld_mix_v_val.w;

			half2* ld_xx_vec_h1 = (half2*)&ld_xx_val.x;
			half2* ld_xx_vec_h2 = (half2*)&ld_xx_val.y;
			half2* ld_xx_vec_h3 = (half2*)&ld_xx_val.z;
			half2* ld_xx_vec_h4 = (half2*)&ld_xx_val.w;

			half2* ld_sx_vec_h1 = (half2*)&ld_sx_val.x;
			half2* ld_sx_vec_h2 = (half2*)&ld_sx_val.y;
			half2* ld_sx_vec_h3 = (half2*)&ld_sx_val.z;
			half2* ld_sx_vec_h4 = (half2*)&ld_sx_val.w;

			half2* st_rx_vec_h1 = (half2*)&st_rx_val.x;
			half2* st_rx_vec_h2 = (half2*)&st_rx_val.y;
			half2* st_rx_vec_h3 = (half2*)&st_rx_val.z;
			half2* st_rx_vec_h4 = (half2*)&st_rx_val.w;

			half2* st_kx_vec_h1 = (half2*)&st_kx_val.x;
			half2* st_kx_vec_h2 = (half2*)&st_kx_val.y;
			half2* st_kx_vec_h3 = (half2*)&st_kx_val.z;
			half2* st_kx_vec_h4 = (half2*)&st_kx_val.w;

			half2* st_vx_vec_h1 = (half2*)&st_vx_val.x;
			half2* st_vx_vec_h2 = (half2*)&st_vx_val.y;
			half2* st_vx_vec_h3 = (half2*)&st_vx_val.z;
			half2* st_vx_vec_h4 = (half2*)&st_vx_val.w;

			st_rx_vec_h1->x = (half)(((float)ld_xx_vec_h1->x) * (float)ld_mix_r_vec_h1->x + ((float)ld_sx_vec_h1->x) * (1.0f - (float)ld_mix_r_vec_h1->x));
			st_rx_vec_h1->y = (half)(((float)ld_xx_vec_h1->y) * (float)ld_mix_r_vec_h1->y + ((float)ld_sx_vec_h1->y) * (1.0f - (float)ld_mix_r_vec_h1->y));
			st_rx_vec_h2->x = (half)(((float)ld_xx_vec_h2->x) * (float)ld_mix_r_vec_h2->x + ((float)ld_sx_vec_h2->x) * (1.0f - (float)ld_mix_r_vec_h2->x));
			st_rx_vec_h2->y = (half)(((float)ld_xx_vec_h2->y) * (float)ld_mix_r_vec_h2->y + ((float)ld_sx_vec_h2->y) * (1.0f - (float)ld_mix_r_vec_h2->y));
			st_rx_vec_h3->x = (half)(((float)ld_xx_vec_h3->x) * (float)ld_mix_r_vec_h3->x + ((float)ld_sx_vec_h3->x) * (1.0f - (float)ld_mix_r_vec_h3->x));
			st_rx_vec_h3->y = (half)(((float)ld_xx_vec_h3->y) * (float)ld_mix_r_vec_h3->y + ((float)ld_sx_vec_h3->y) * (1.0f - (float)ld_mix_r_vec_h3->y));
			st_rx_vec_h4->x = (half)(((float)ld_xx_vec_h4->x) * (float)ld_mix_r_vec_h4->x + ((float)ld_sx_vec_h4->x) * (1.0f - (float)ld_mix_r_vec_h4->x));
			st_rx_vec_h4->y = (half)(((float)ld_xx_vec_h4->y) * (float)ld_mix_r_vec_h4->y + ((float)ld_sx_vec_h4->y) * (1.0f - (float)ld_mix_r_vec_h4->y));

			st_kx_vec_h1->x = (half)(((float)ld_xx_vec_h1->x) * (float)ld_mix_k_vec_h1->x + ((float)ld_sx_vec_h1->x) * (1.0f - (float)ld_mix_k_vec_h1->x));
			st_kx_vec_h1->y = (half)(((float)ld_xx_vec_h1->y) * (float)ld_mix_k_vec_h1->y + ((float)ld_sx_vec_h1->y) * (1.0f - (float)ld_mix_k_vec_h1->y));
			st_kx_vec_h2->x = (half)(((float)ld_xx_vec_h2->x) * (float)ld_mix_k_vec_h2->x + ((float)ld_sx_vec_h2->x) * (1.0f - (float)ld_mix_k_vec_h2->x));
			st_kx_vec_h2->y = (half)(((float)ld_xx_vec_h2->y) * (float)ld_mix_k_vec_h2->y + ((float)ld_sx_vec_h2->y) * (1.0f - (float)ld_mix_k_vec_h2->y));
			st_kx_vec_h3->x = (half)(((float)ld_xx_vec_h3->x) * (float)ld_mix_k_vec_h3->x + ((float)ld_sx_vec_h3->x) * (1.0f - (float)ld_mix_k_vec_h3->x));
			st_kx_vec_h3->y = (half)(((float)ld_xx_vec_h3->y) * (float)ld_mix_k_vec_h3->y + ((float)ld_sx_vec_h3->y) * (1.0f - (float)ld_mix_k_vec_h3->y));
			st_kx_vec_h4->x = (half)(((float)ld_xx_vec_h4->x) * (float)ld_mix_k_vec_h4->x + ((float)ld_sx_vec_h4->x) * (1.0f - (float)ld_mix_k_vec_h4->x));
			st_kx_vec_h4->y = (half)(((float)ld_xx_vec_h4->y) * (float)ld_mix_k_vec_h4->y + ((float)ld_sx_vec_h4->y) * (1.0f - (float)ld_mix_k_vec_h4->y));

			st_vx_vec_h1->x = (half)(((float)ld_xx_vec_h1->x) * (float)ld_mix_v_vec_h1->x + ((float)ld_sx_vec_h1->x) * (1.0f - (float)ld_mix_v_vec_h1->x));
			st_vx_vec_h1->y = (half)(((float)ld_xx_vec_h1->y) * (float)ld_mix_v_vec_h1->y + ((float)ld_sx_vec_h1->y) * (1.0f - (float)ld_mix_v_vec_h1->y));
			st_vx_vec_h2->x = (half)(((float)ld_xx_vec_h2->x) * (float)ld_mix_v_vec_h2->x + ((float)ld_sx_vec_h2->x) * (1.0f - (float)ld_mix_v_vec_h2->x));
			st_vx_vec_h2->y = (half)(((float)ld_xx_vec_h2->y) * (float)ld_mix_v_vec_h2->y + ((float)ld_sx_vec_h2->y) * (1.0f - (float)ld_mix_v_vec_h2->y));
			st_vx_vec_h3->x = (half)(((float)ld_xx_vec_h3->x) * (float)ld_mix_v_vec_h3->x + ((float)ld_sx_vec_h3->x) * (1.0f - (float)ld_mix_v_vec_h3->x));
			st_vx_vec_h3->y = (half)(((float)ld_xx_vec_h3->y) * (float)ld_mix_v_vec_h3->y + ((float)ld_sx_vec_h3->y) * (1.0f - (float)ld_mix_v_vec_h3->y));
			st_vx_vec_h4->x = (half)(((float)ld_xx_vec_h4->x) * (float)ld_mix_v_vec_h4->x + ((float)ld_sx_vec_h4->x) * (1.0f - (float)ld_mix_v_vec_h4->x));
			st_vx_vec_h4->y = (half)(((float)ld_xx_vec_h4->y) * (float)ld_mix_v_vec_h4->y + ((float)ld_sx_vec_h4->y) * (1.0f - (float)ld_mix_v_vec_h4->y));

			st_rx_vec4[thread_id] = st_rx_val;
			st_kx_vec4[thread_id] = st_kx_val;
			st_vx_vec4[thread_id] = st_vx_val;
		}

		__device__ __forceinline__ void mix_rkv_fp32(
			const uint32_t k, uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			half* mix_r, half* mix_k, half* mix_v,
			float* xx, float* sx,
			float* rx, float* kx, float* vx
		)
		{
			mix_r += (blk_id * thread_count * 8) % k;
			mix_k += (blk_id * thread_count * 8) % k;
			mix_v += (blk_id * thread_count * 8) % k;

			xx += blk_id * thread_count * 8;
			sx += blk_id * thread_count * 8;

			rx += blk_id * thread_count * 8;
			kx += blk_id * thread_count * 8;
			vx += blk_id * thread_count * 8;

			float4* ld_mix_r_vec4 = reinterpret_cast<float4*>(mix_r);
			float4* ld_mix_k_vec4 = reinterpret_cast<float4*>(mix_k);
			float4* ld_mix_v_vec4 = reinterpret_cast<float4*>(mix_v);

			float4* ld_xx_vec4 = reinterpret_cast<float4*>(xx);
			float4* ld_sx_vec4 = reinterpret_cast<float4*>(sx);

			float4* st_rx_vec4 = reinterpret_cast<float4*>(rx);
			float4* st_kx_vec4 = reinterpret_cast<float4*>(kx);
			float4* st_vx_vec4 = reinterpret_cast<float4*>(vx);

			float4 ld_mix_r_val = ld_mix_r_vec4[thread_id];
			float4 ld_mix_k_val = ld_mix_k_vec4[thread_id];
			float4 ld_mix_v_val = ld_mix_v_vec4[thread_id];

			float4 ld_xx_val[2] = { ld_xx_vec4[thread_id * 2], ld_xx_vec4[thread_id * 2 + 1]};
			float4 ld_sx_val[2] = { ld_sx_vec4[thread_id * 2], ld_sx_vec4[thread_id * 2 + 1]};

			float4 st_rx_val[2];
			float4 st_kx_val[2];
			float4 st_vx_val[2];

			half2* ld_mix_r_vec_h1 = (half2*)&ld_mix_r_val.x;
			half2* ld_mix_r_vec_h2 = (half2*)&ld_mix_r_val.y;
			half2* ld_mix_r_vec_h3 = (half2*)&ld_mix_r_val.z;
			half2* ld_mix_r_vec_h4 = (half2*)&ld_mix_r_val.w;

			half2* ld_mix_k_vec_h1 = (half2*)&ld_mix_k_val.x;
			half2* ld_mix_k_vec_h2 = (half2*)&ld_mix_k_val.y;
			half2* ld_mix_k_vec_h3 = (half2*)&ld_mix_k_val.z;
			half2* ld_mix_k_vec_h4 = (half2*)&ld_mix_k_val.w;

			half2* ld_mix_v_vec_h1 = (half2*)&ld_mix_v_val.x;
			half2* ld_mix_v_vec_h2 = (half2*)&ld_mix_v_val.y;
			half2* ld_mix_v_vec_h3 = (half2*)&ld_mix_v_val.z;
			half2* ld_mix_v_vec_h4 = (half2*)&ld_mix_v_val.w;

			st_rx_val[0].x = ld_xx_val[0].x * (float)ld_mix_r_vec_h1->x + ld_sx_val[0].x * (1.0f - (float)ld_mix_r_vec_h1->x);
			st_rx_val[0].y = ld_xx_val[0].y * (float)ld_mix_r_vec_h1->y + ld_sx_val[0].y * (1.0f - (float)ld_mix_r_vec_h1->y);
			st_rx_val[0].z = ld_xx_val[0].z * (float)ld_mix_r_vec_h2->x + ld_sx_val[0].z * (1.0f - (float)ld_mix_r_vec_h2->x);
			st_rx_val[0].w = ld_xx_val[0].w * (float)ld_mix_r_vec_h2->y + ld_sx_val[0].w * (1.0f - (float)ld_mix_r_vec_h2->y);
			st_rx_val[1].x = ld_xx_val[1].x * (float)ld_mix_r_vec_h3->x + ld_sx_val[1].x * (1.0f - (float)ld_mix_r_vec_h3->x);
			st_rx_val[1].y = ld_xx_val[1].y * (float)ld_mix_r_vec_h3->y + ld_sx_val[1].y * (1.0f - (float)ld_mix_r_vec_h3->y);
			st_rx_val[1].z = ld_xx_val[1].z * (float)ld_mix_r_vec_h4->x + ld_sx_val[1].z * (1.0f - (float)ld_mix_r_vec_h4->x);
			st_rx_val[1].w = ld_xx_val[1].w * (float)ld_mix_r_vec_h4->y + ld_sx_val[1].w * (1.0f - (float)ld_mix_r_vec_h4->y);

			st_kx_val[0].x = ld_xx_val[0].x * (float)ld_mix_k_vec_h1->x + ld_sx_val[0].x * (1.0f - (float)ld_mix_k_vec_h1->x);
			st_kx_val[0].y = ld_xx_val[0].y * (float)ld_mix_k_vec_h1->y + ld_sx_val[0].y * (1.0f - (float)ld_mix_k_vec_h1->y);
			st_kx_val[0].z = ld_xx_val[0].z * (float)ld_mix_k_vec_h2->x + ld_sx_val[0].z * (1.0f - (float)ld_mix_k_vec_h2->x);
			st_kx_val[0].w = ld_xx_val[0].w * (float)ld_mix_k_vec_h2->y + ld_sx_val[0].w * (1.0f - (float)ld_mix_k_vec_h2->y);
			st_kx_val[1].x = ld_xx_val[1].x * (float)ld_mix_k_vec_h3->x + ld_sx_val[1].x * (1.0f - (float)ld_mix_k_vec_h3->x);
			st_kx_val[1].y = ld_xx_val[1].y * (float)ld_mix_k_vec_h3->y + ld_sx_val[1].y * (1.0f - (float)ld_mix_k_vec_h3->y);
			st_kx_val[1].z = ld_xx_val[1].z * (float)ld_mix_k_vec_h4->x + ld_sx_val[1].z * (1.0f - (float)ld_mix_k_vec_h4->x);
			st_kx_val[1].w = ld_xx_val[1].w * (float)ld_mix_k_vec_h4->y + ld_sx_val[1].w * (1.0f - (float)ld_mix_k_vec_h4->y);

			st_vx_val[0].x = ld_xx_val[0].x * (float)ld_mix_v_vec_h1->x + ld_sx_val[0].x * (1.0f - (float)ld_mix_v_vec_h1->x);
			st_vx_val[0].y = ld_xx_val[0].y * (float)ld_mix_v_vec_h1->y + ld_sx_val[0].y * (1.0f - (float)ld_mix_v_vec_h1->y);
			st_vx_val[0].z = ld_xx_val[0].z * (float)ld_mix_v_vec_h2->x + ld_sx_val[0].z * (1.0f - (float)ld_mix_v_vec_h2->x);
			st_vx_val[0].w = ld_xx_val[0].w * (float)ld_mix_v_vec_h2->y + ld_sx_val[0].w * (1.0f - (float)ld_mix_v_vec_h2->y);
			st_vx_val[1].x = ld_xx_val[1].x * (float)ld_mix_v_vec_h3->x + ld_sx_val[1].x * (1.0f - (float)ld_mix_v_vec_h3->x);
			st_vx_val[1].y = ld_xx_val[1].y * (float)ld_mix_v_vec_h3->y + ld_sx_val[1].y * (1.0f - (float)ld_mix_v_vec_h3->y);
			st_vx_val[1].z = ld_xx_val[1].z * (float)ld_mix_v_vec_h4->x + ld_sx_val[1].z * (1.0f - (float)ld_mix_v_vec_h4->x);
			st_vx_val[1].w = ld_xx_val[1].w * (float)ld_mix_v_vec_h4->y + ld_sx_val[1].w * (1.0f - (float)ld_mix_v_vec_h4->y);

			st_rx_vec4[thread_id * 2] = st_rx_val[0];
			st_rx_vec4[thread_id * 2 + 1] = st_rx_val[1];
			st_kx_vec4[thread_id * 2] = st_kx_val[0];
			st_kx_vec4[thread_id * 2 + 1] = st_kx_val[1];
			st_vx_vec4[thread_id * 2] = st_vx_val[0];
			st_vx_vec4[thread_id * 2 + 1] = st_vx_val[1];
		}

		template<typename T>
		__global__ void mix_rkv(
			const uint32_t m, const uint32_t k,
			T* xx, T* sx,
			half* mix_r, half* mix_k, half* mix_v,
			T* rx, T* kx, T* vx
		)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;
			
			if constexpr (sizeof(T) == 2)
				mix_rkv_fp16(k, blk_id, thread_id, thread_count, mix_r, mix_k, mix_v, xx, sx, rx, kx, vx);
			else
				mix_rkv_fp32(k, blk_id, thread_id, thread_count, mix_r, mix_k, mix_v, xx, sx, rx, kx, vx);
		}

		__device__ __forceinline__ void mix_rk_fp16(
			const uint32_t k, uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			half* mix_r, half* mix_k, 
			half* xx, half* sx,
			half* rx, half* kx
		)
		{
			mix_r += (blk_id * thread_count * 8) % k;
			mix_k += (blk_id * thread_count * 8) % k;

			xx += blk_id * thread_count * 8;
			sx += blk_id * thread_count * 8;

			rx += blk_id * thread_count * 8;
			kx += blk_id * thread_count * 8;

			float4* ld_mix_r_vec4 = reinterpret_cast<float4*>(mix_r);
			float4* ld_mix_k_vec4 = reinterpret_cast<float4*>(mix_k);

			float4* ld_xx_vec4 = reinterpret_cast<float4*>(xx);
			float4* ld_sx_vec4 = reinterpret_cast<float4*>(sx);

			float4* st_rx_vec4 = reinterpret_cast<float4*>(rx);
			float4* st_kx_vec4 = reinterpret_cast<float4*>(kx);

			float4 ld_mix_r_val = ld_mix_r_vec4[thread_id];
			float4 ld_mix_k_val = ld_mix_k_vec4[thread_id];

			float4 ld_xx_val = ld_xx_vec4[thread_id];
			float4 ld_sx_val = ld_sx_vec4[thread_id];

			float4 st_rx_val;
			float4 st_kx_val;

			half2* ld_mix_r_vec_h1 = (half2*)&ld_mix_r_val.x;
			half2* ld_mix_r_vec_h2 = (half2*)&ld_mix_r_val.y;
			half2* ld_mix_r_vec_h3 = (half2*)&ld_mix_r_val.z;
			half2* ld_mix_r_vec_h4 = (half2*)&ld_mix_r_val.w;

			half2* ld_mix_k_vec_h1 = (half2*)&ld_mix_k_val.x;
			half2* ld_mix_k_vec_h2 = (half2*)&ld_mix_k_val.y;
			half2* ld_mix_k_vec_h3 = (half2*)&ld_mix_k_val.z;
			half2* ld_mix_k_vec_h4 = (half2*)&ld_mix_k_val.w;

			half2* ld_xx_vec_h1 = (half2*)&ld_xx_val.x;
			half2* ld_xx_vec_h2 = (half2*)&ld_xx_val.y;
			half2* ld_xx_vec_h3 = (half2*)&ld_xx_val.z;
			half2* ld_xx_vec_h4 = (half2*)&ld_xx_val.w;

			half2* ld_sx_vec_h1 = (half2*)&ld_sx_val.x;
			half2* ld_sx_vec_h2 = (half2*)&ld_sx_val.y;
			half2* ld_sx_vec_h3 = (half2*)&ld_sx_val.z;
			half2* ld_sx_vec_h4 = (half2*)&ld_sx_val.w;

			half2* st_rx_vec_h1 = (half2*)&st_rx_val.x;
			half2* st_rx_vec_h2 = (half2*)&st_rx_val.y;
			half2* st_rx_vec_h3 = (half2*)&st_rx_val.z;
			half2* st_rx_vec_h4 = (half2*)&st_rx_val.w;

			half2* st_kx_vec_h1 = (half2*)&st_kx_val.x;
			half2* st_kx_vec_h2 = (half2*)&st_kx_val.y;
			half2* st_kx_vec_h3 = (half2*)&st_kx_val.z;
			half2* st_kx_vec_h4 = (half2*)&st_kx_val.w;

			st_rx_vec_h1->x = (half)(((float)ld_xx_vec_h1->x) * (float)ld_mix_r_vec_h1->x + ((float)ld_sx_vec_h1->x) * (1.0f - (float)ld_mix_r_vec_h1->x));
			st_rx_vec_h1->y = (half)(((float)ld_xx_vec_h1->y) * (float)ld_mix_r_vec_h1->y + ((float)ld_sx_vec_h1->y) * (1.0f - (float)ld_mix_r_vec_h1->y));
			st_rx_vec_h2->x = (half)(((float)ld_xx_vec_h2->x) * (float)ld_mix_r_vec_h2->x + ((float)ld_sx_vec_h2->x) * (1.0f - (float)ld_mix_r_vec_h2->x));
			st_rx_vec_h2->y = (half)(((float)ld_xx_vec_h2->y) * (float)ld_mix_r_vec_h2->y + ((float)ld_sx_vec_h2->y) * (1.0f - (float)ld_mix_r_vec_h2->y));
			st_rx_vec_h3->x = (half)(((float)ld_xx_vec_h3->x) * (float)ld_mix_r_vec_h3->x + ((float)ld_sx_vec_h3->x) * (1.0f - (float)ld_mix_r_vec_h3->x));
			st_rx_vec_h3->y = (half)(((float)ld_xx_vec_h3->y) * (float)ld_mix_r_vec_h3->y + ((float)ld_sx_vec_h3->y) * (1.0f - (float)ld_mix_r_vec_h3->y));
			st_rx_vec_h4->x = (half)(((float)ld_xx_vec_h4->x) * (float)ld_mix_r_vec_h4->x + ((float)ld_sx_vec_h4->x) * (1.0f - (float)ld_mix_r_vec_h4->x));
			st_rx_vec_h4->y = (half)(((float)ld_xx_vec_h4->y) * (float)ld_mix_r_vec_h4->y + ((float)ld_sx_vec_h4->y) * (1.0f - (float)ld_mix_r_vec_h4->y));

			st_kx_vec_h1->x = (half)(((float)ld_xx_vec_h1->x) * (float)ld_mix_k_vec_h1->x + ((float)ld_sx_vec_h1->x) * (1.0f - (float)ld_mix_k_vec_h1->x));
			st_kx_vec_h1->y = (half)(((float)ld_xx_vec_h1->y) * (float)ld_mix_k_vec_h1->y + ((float)ld_sx_vec_h1->y) * (1.0f - (float)ld_mix_k_vec_h1->y));
			st_kx_vec_h2->x = (half)(((float)ld_xx_vec_h2->x) * (float)ld_mix_k_vec_h2->x + ((float)ld_sx_vec_h2->x) * (1.0f - (float)ld_mix_k_vec_h2->x));
			st_kx_vec_h2->y = (half)(((float)ld_xx_vec_h2->y) * (float)ld_mix_k_vec_h2->y + ((float)ld_sx_vec_h2->y) * (1.0f - (float)ld_mix_k_vec_h2->y));
			st_kx_vec_h3->x = (half)(((float)ld_xx_vec_h3->x) * (float)ld_mix_k_vec_h3->x + ((float)ld_sx_vec_h3->x) * (1.0f - (float)ld_mix_k_vec_h3->x));
			st_kx_vec_h3->y = (half)(((float)ld_xx_vec_h3->y) * (float)ld_mix_k_vec_h3->y + ((float)ld_sx_vec_h3->y) * (1.0f - (float)ld_mix_k_vec_h3->y));
			st_kx_vec_h4->x = (half)(((float)ld_xx_vec_h4->x) * (float)ld_mix_k_vec_h4->x + ((float)ld_sx_vec_h4->x) * (1.0f - (float)ld_mix_k_vec_h4->x));
			st_kx_vec_h4->y = (half)(((float)ld_xx_vec_h4->y) * (float)ld_mix_k_vec_h4->y + ((float)ld_sx_vec_h4->y) * (1.0f - (float)ld_mix_k_vec_h4->y));

			st_rx_vec4[thread_id] = st_rx_val;
			st_kx_vec4[thread_id] = st_kx_val;
		}

		__device__ __forceinline__ void mix_rk_fp32(
			const uint32_t k, uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			half* mix_r, half* mix_k, 
			float* xx, float* sx,
			float* rx, float* kx
		)
		{
			mix_r += (blk_id * thread_count * 8) % k;
			mix_k += (blk_id * thread_count * 8) % k;

			xx += blk_id * thread_count * 8;
			sx += blk_id * thread_count * 8;

			rx += blk_id * thread_count * 8;
			kx += blk_id * thread_count * 8;

			float4* ld_mix_r_vec4 = reinterpret_cast<float4*>(mix_r);
			float4* ld_mix_k_vec4 = reinterpret_cast<float4*>(mix_k);

			float4* ld_xx_vec4 = reinterpret_cast<float4*>(xx);
			float4* ld_sx_vec4 = reinterpret_cast<float4*>(sx);

			float4* st_rx_vec4 = reinterpret_cast<float4*>(rx);
			float4* st_kx_vec4 = reinterpret_cast<float4*>(kx);

			float4 ld_mix_r_val = ld_mix_r_vec4[thread_id];
			float4 ld_mix_k_val = ld_mix_k_vec4[thread_id];

			float4 ld_xx_val[2] = { ld_xx_vec4[thread_id * 2], ld_xx_vec4[thread_id * 2 + 1] };
			float4 ld_sx_val[2] = { ld_sx_vec4[thread_id * 2], ld_sx_vec4[thread_id * 2 + 1] };

			float4 st_rx_val[2];
			float4 st_kx_val[2];

			half2* ld_mix_r_vec_h1 = (half2*)&ld_mix_r_val.x;
			half2* ld_mix_r_vec_h2 = (half2*)&ld_mix_r_val.y;
			half2* ld_mix_r_vec_h3 = (half2*)&ld_mix_r_val.z;
			half2* ld_mix_r_vec_h4 = (half2*)&ld_mix_r_val.w;

			half2* ld_mix_k_vec_h1 = (half2*)&ld_mix_k_val.x;
			half2* ld_mix_k_vec_h2 = (half2*)&ld_mix_k_val.y;
			half2* ld_mix_k_vec_h3 = (half2*)&ld_mix_k_val.z;
			half2* ld_mix_k_vec_h4 = (half2*)&ld_mix_k_val.w;

			st_rx_val[0].x = ld_xx_val[0].x * (float)ld_mix_r_vec_h1->x + ld_sx_val[0].x * (1.0f - (float)ld_mix_r_vec_h1->x);
			st_rx_val[0].y = ld_xx_val[0].y * (float)ld_mix_r_vec_h1->y + ld_sx_val[0].y * (1.0f - (float)ld_mix_r_vec_h1->y);
			st_rx_val[0].z = ld_xx_val[0].z * (float)ld_mix_r_vec_h2->x + ld_sx_val[0].z * (1.0f - (float)ld_mix_r_vec_h2->x);
			st_rx_val[0].w = ld_xx_val[0].w * (float)ld_mix_r_vec_h2->y + ld_sx_val[0].w * (1.0f - (float)ld_mix_r_vec_h2->y);
			st_rx_val[1].x = ld_xx_val[1].x * (float)ld_mix_r_vec_h3->x + ld_sx_val[1].x * (1.0f - (float)ld_mix_r_vec_h3->x);
			st_rx_val[1].y = ld_xx_val[1].y * (float)ld_mix_r_vec_h3->y + ld_sx_val[1].y * (1.0f - (float)ld_mix_r_vec_h3->y);
			st_rx_val[1].z = ld_xx_val[1].z * (float)ld_mix_r_vec_h4->x + ld_sx_val[1].z * (1.0f - (float)ld_mix_r_vec_h4->x);
			st_rx_val[1].w = ld_xx_val[1].w * (float)ld_mix_r_vec_h4->y + ld_sx_val[1].w * (1.0f - (float)ld_mix_r_vec_h4->y);

			st_kx_val[0].x = ld_xx_val[0].x * (float)ld_mix_k_vec_h1->x + ld_sx_val[0].x * (1.0f - (float)ld_mix_k_vec_h1->x);
			st_kx_val[0].y = ld_xx_val[0].y * (float)ld_mix_k_vec_h1->y + ld_sx_val[0].y * (1.0f - (float)ld_mix_k_vec_h1->y);
			st_kx_val[0].z = ld_xx_val[0].z * (float)ld_mix_k_vec_h2->x + ld_sx_val[0].z * (1.0f - (float)ld_mix_k_vec_h2->x);
			st_kx_val[0].w = ld_xx_val[0].w * (float)ld_mix_k_vec_h2->y + ld_sx_val[0].w * (1.0f - (float)ld_mix_k_vec_h2->y);
			st_kx_val[1].x = ld_xx_val[1].x * (float)ld_mix_k_vec_h3->x + ld_sx_val[1].x * (1.0f - (float)ld_mix_k_vec_h3->x);
			st_kx_val[1].y = ld_xx_val[1].y * (float)ld_mix_k_vec_h3->y + ld_sx_val[1].y * (1.0f - (float)ld_mix_k_vec_h3->y);
			st_kx_val[1].z = ld_xx_val[1].z * (float)ld_mix_k_vec_h4->x + ld_sx_val[1].z * (1.0f - (float)ld_mix_k_vec_h4->x);
			st_kx_val[1].w = ld_xx_val[1].w * (float)ld_mix_k_vec_h4->y + ld_sx_val[1].w * (1.0f - (float)ld_mix_k_vec_h4->y);

			st_rx_vec4[thread_id * 2] = st_rx_val[0];
			st_rx_vec4[thread_id * 2 + 1] = st_rx_val[1];
			st_kx_vec4[thread_id * 2] = st_kx_val[0];
			st_kx_vec4[thread_id * 2 + 1] = st_kx_val[1];
		}

		template<typename T>
		__global__ void mix_rk(
			const uint32_t m, const uint32_t k,
			T* xx, T* sx,
			half* mix_r, half* mix_k,
			T* rx, T* kx
		)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			if constexpr (sizeof(T) == 2)
				mix_rk_fp16(k, blk_id, thread_id, thread_count, mix_r, mix_k, xx, sx, rx, kx);
			else
				mix_rk_fp32(k, blk_id, thread_id, thread_count, mix_r, mix_k, xx, sx, rx, kx);
		}

		template<typename T>
		__device__ __forceinline__ void cal_wkv(
			float& att_aa, float& att_bb, float& att_pp,
			float att_time_decay, float att_time_first,
			T key, T value, T& wkv
		)
		{
			float ww = att_time_first + (float)key;
			float p = max(att_pp, ww);
			float e1 = exp(att_pp - p);
			float e2 = exp(ww - p);

			wkv = (T)((e1 * att_aa + e2 * (float)value) / (e1 * att_bb + e2));

			ww = att_time_decay + att_pp;
			p = max(ww, (float)key);
			e1 = exp(ww - p);
			e2 = exp((float)key - p);

			att_aa = e1 * att_aa + e2 * (float)value;
			att_bb = e1 * att_bb + e2;
			att_pp = p;
		}

		__device__ __forceinline__ void forward_wkv_one_fp16(
			uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			half* key, half* value, half* wkv
		)
		{
			att_aa += blk_id * thread_count * 8;
			att_bb += blk_id * thread_count * 8;
			att_pp += blk_id * thread_count * 8;

			att_time_decay += blk_id * thread_count * 8;
			att_time_first += blk_id * thread_count * 8;

			key += blk_id * thread_count * 8;
			value += blk_id * thread_count * 8;

			wkv += blk_id * thread_count * 8;

			float4* ld_att_aa_vec4 = reinterpret_cast<float4*>(att_aa);
			float4* ld_att_bb_vec4 = reinterpret_cast<float4*>(att_bb);
			float4* ld_att_pp_vec4 = reinterpret_cast<float4*>(att_pp);

			float4* ld_att_time_decay_vec4 = reinterpret_cast<float4*>(att_time_decay);
			float4* ld_att_time_first_vec4 = reinterpret_cast<float4*>(att_time_first);

			float4* ld_key_vec4 = reinterpret_cast<float4*>(key);
			float4* ld_value_vec4 = reinterpret_cast<float4*>(value);

			float4* st_wkv_vec4 = reinterpret_cast<float4*>(wkv);

			float4 ld_att_aa_val[2] = { ld_att_aa_vec4[thread_id * 2], ld_att_aa_vec4[thread_id * 2 + 1] };
			float4 ld_att_bb_val[2] = { ld_att_bb_vec4[thread_id * 2], ld_att_bb_vec4[thread_id * 2 + 1] };
			float4 ld_att_pp_val[2] = { ld_att_pp_vec4[thread_id * 2], ld_att_pp_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_decay_val[2] = { ld_att_time_decay_vec4[thread_id * 2], ld_att_time_decay_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_first_val[2] = { ld_att_time_first_vec4[thread_id * 2], ld_att_time_first_vec4[thread_id * 2 + 1] };

			float4 ld_key_val = ld_key_vec4[thread_id];
			float4 ld_value_val = ld_value_vec4[thread_id];

			float4 st_wkv_val;

			half2* ld_key_vec_h1 = (half2*)&ld_key_val.x;
			half2* ld_key_vec_h2 = (half2*)&ld_key_val.y;
			half2* ld_key_vec_h3 = (half2*)&ld_key_val.z;
			half2* ld_key_vec_h4 = (half2*)&ld_key_val.w;

			half2* ld_value_vec_h1 = (half2*)&ld_value_val.x;
			half2* ld_value_vec_h2 = (half2*)&ld_value_val.y;
			half2* ld_value_vec_h3 = (half2*)&ld_value_val.z;
			half2* ld_value_vec_h4 = (half2*)&ld_value_val.w;

			half2* st_wkv_vec_h1 = (half2*)&st_wkv_val.x;
			half2* st_wkv_vec_h2 = (half2*)&st_wkv_val.y;
			half2* st_wkv_vec_h3 = (half2*)&st_wkv_val.z;
			half2* st_wkv_vec_h4 = (half2*)&st_wkv_val.w;

			cal_wkv<half>(ld_att_aa_val[0].x, ld_att_bb_val[0].x, ld_att_pp_val[0].x, ld_att_time_decay_val[0].x, ld_att_time_first_val[0].x, ld_key_vec_h1->x, ld_value_vec_h1->x, st_wkv_vec_h1->x);
			cal_wkv<half>(ld_att_aa_val[0].y, ld_att_bb_val[0].y, ld_att_pp_val[0].y, ld_att_time_decay_val[0].y, ld_att_time_first_val[0].y, ld_key_vec_h1->y, ld_value_vec_h1->y, st_wkv_vec_h1->y);
			cal_wkv<half>(ld_att_aa_val[0].z, ld_att_bb_val[0].z, ld_att_pp_val[0].z, ld_att_time_decay_val[0].z, ld_att_time_first_val[0].z, ld_key_vec_h2->x, ld_value_vec_h2->x, st_wkv_vec_h2->x);
			cal_wkv<half>(ld_att_aa_val[0].w, ld_att_bb_val[0].w, ld_att_pp_val[0].w, ld_att_time_decay_val[0].w, ld_att_time_first_val[0].w, ld_key_vec_h2->y, ld_value_vec_h2->y, st_wkv_vec_h2->y);
			cal_wkv<half>(ld_att_aa_val[1].x, ld_att_bb_val[1].x, ld_att_pp_val[1].x, ld_att_time_decay_val[1].x, ld_att_time_first_val[1].x, ld_key_vec_h3->x, ld_value_vec_h3->x, st_wkv_vec_h3->x);
			cal_wkv<half>(ld_att_aa_val[1].y, ld_att_bb_val[1].y, ld_att_pp_val[1].y, ld_att_time_decay_val[1].y, ld_att_time_first_val[1].y, ld_key_vec_h3->y, ld_value_vec_h3->y, st_wkv_vec_h3->y);
			cal_wkv<half>(ld_att_aa_val[1].z, ld_att_bb_val[1].z, ld_att_pp_val[1].z, ld_att_time_decay_val[1].z, ld_att_time_first_val[1].z, ld_key_vec_h4->x, ld_value_vec_h4->x, st_wkv_vec_h4->x);
			cal_wkv<half>(ld_att_aa_val[1].w, ld_att_bb_val[1].w, ld_att_pp_val[1].w, ld_att_time_decay_val[1].w, ld_att_time_first_val[1].w, ld_key_vec_h4->y, ld_value_vec_h4->y, st_wkv_vec_h4->y);

			st_wkv_vec4[thread_id] = st_wkv_val;
			ld_att_aa_vec4[thread_id * 2] = ld_att_aa_val[0];
			ld_att_aa_vec4[thread_id * 2 + 1] = ld_att_aa_val[1];
			ld_att_bb_vec4[thread_id * 2] = ld_att_bb_val[0];
			ld_att_bb_vec4[thread_id * 2 + 1] = ld_att_bb_val[1];
			ld_att_pp_vec4[thread_id * 2] = ld_att_pp_val[0];
			ld_att_pp_vec4[thread_id * 2 + 1] = ld_att_pp_val[1];
		}

		__device__ __forceinline__ void forward_wkv_one_fp32(
			uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			float* key, float* value, float* wkv
		)
		{
			att_aa += blk_id * thread_count * 8;
			att_bb += blk_id * thread_count * 8;
			att_pp += blk_id * thread_count * 8;

			att_time_decay += blk_id * thread_count * 8;
			att_time_first += blk_id * thread_count * 8;

			key += blk_id * thread_count * 8;
			value += blk_id * thread_count * 8;

			wkv += blk_id * thread_count * 8;

			float4* ld_att_aa_vec4 = reinterpret_cast<float4*>(att_aa);
			float4* ld_att_bb_vec4 = reinterpret_cast<float4*>(att_bb);
			float4* ld_att_pp_vec4 = reinterpret_cast<float4*>(att_pp);

			float4* ld_att_time_decay_vec4 = reinterpret_cast<float4*>(att_time_decay);
			float4* ld_att_time_first_vec4 = reinterpret_cast<float4*>(att_time_first);

			float4* ld_key_vec4 = reinterpret_cast<float4*>(key);
			float4* ld_value_vec4 = reinterpret_cast<float4*>(value);

			float4* st_wkv_vec4 = reinterpret_cast<float4*>(wkv);

			float4 ld_att_aa_val[2] = { ld_att_aa_vec4[thread_id * 2], ld_att_aa_vec4[thread_id * 2 + 1] };
			float4 ld_att_bb_val[2] = { ld_att_bb_vec4[thread_id * 2], ld_att_bb_vec4[thread_id * 2 + 1] };
			float4 ld_att_pp_val[2] = { ld_att_pp_vec4[thread_id * 2], ld_att_pp_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_decay_val[2] = { ld_att_time_decay_vec4[thread_id * 2], ld_att_time_decay_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_first_val[2] = { ld_att_time_first_vec4[thread_id * 2], ld_att_time_first_vec4[thread_id * 2 + 1] };

			float4 ld_key_val[2] = { ld_key_vec4[thread_id * 2], ld_key_vec4[thread_id * 2 + 1] };
			float4 ld_value_val[2] = { ld_value_vec4[thread_id * 2], ld_value_vec4[thread_id * 2 + 1] };

			float4 st_wkv_val[2];

			cal_wkv<float>(ld_att_aa_val[0].x, ld_att_bb_val[0].x, ld_att_pp_val[0].x, ld_att_time_decay_val[0].x, ld_att_time_first_val[0].x, ld_key_val[0].x, ld_value_val[0].x, st_wkv_val[0].x);
			cal_wkv<float>(ld_att_aa_val[0].y, ld_att_bb_val[0].y, ld_att_pp_val[0].y, ld_att_time_decay_val[0].y, ld_att_time_first_val[0].y, ld_key_val[0].y, ld_value_val[0].y, st_wkv_val[0].y);
			cal_wkv<float>(ld_att_aa_val[0].z, ld_att_bb_val[0].z, ld_att_pp_val[0].z, ld_att_time_decay_val[0].z, ld_att_time_first_val[0].z, ld_key_val[0].z, ld_value_val[0].z, st_wkv_val[0].z);
			cal_wkv<float>(ld_att_aa_val[0].w, ld_att_bb_val[0].w, ld_att_pp_val[0].w, ld_att_time_decay_val[0].w, ld_att_time_first_val[0].w, ld_key_val[0].w, ld_value_val[0].w, st_wkv_val[0].w);
			cal_wkv<float>(ld_att_aa_val[1].x, ld_att_bb_val[1].x, ld_att_pp_val[1].x, ld_att_time_decay_val[1].x, ld_att_time_first_val[1].x, ld_key_val[1].x, ld_value_val[1].x, st_wkv_val[1].x);
			cal_wkv<float>(ld_att_aa_val[1].y, ld_att_bb_val[1].y, ld_att_pp_val[1].y, ld_att_time_decay_val[1].y, ld_att_time_first_val[1].y, ld_key_val[1].y, ld_value_val[1].y, st_wkv_val[1].y);
			cal_wkv<float>(ld_att_aa_val[1].z, ld_att_bb_val[1].z, ld_att_pp_val[1].z, ld_att_time_decay_val[1].z, ld_att_time_first_val[1].z, ld_key_val[1].z, ld_value_val[1].z, st_wkv_val[1].z);
			cal_wkv<float>(ld_att_aa_val[1].w, ld_att_bb_val[1].w, ld_att_pp_val[1].w, ld_att_time_decay_val[1].w, ld_att_time_first_val[1].w, ld_key_val[1].w, ld_value_val[1].w, st_wkv_val[1].w);

			st_wkv_vec4[thread_id * 2] = st_wkv_val[0];
			st_wkv_vec4[thread_id * 2 + 1] = st_wkv_val[1];
			ld_att_aa_vec4[thread_id * 2] = ld_att_aa_val[0];
			ld_att_aa_vec4[thread_id * 2 + 1] = ld_att_aa_val[1];
			ld_att_bb_vec4[thread_id * 2] = ld_att_bb_val[0];
			ld_att_bb_vec4[thread_id * 2 + 1] = ld_att_bb_val[1];
			ld_att_pp_vec4[thread_id * 2] = ld_att_pp_val[0];
			ld_att_pp_vec4[thread_id * 2 + 1] = ld_att_pp_val[1];
		}

		template<typename T>
		__global__ void forward_wkv_one(
			const uint32_t m, const uint32_t k,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			T* key, T* value, T* wkv
		)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			if constexpr (sizeof(T) == 2)
				forward_wkv_one_fp16(blk_id, thread_id, thread_count, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);
			else
				forward_wkv_one_fp32(blk_id, thread_id, thread_count, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);
		}

		__device__ __forceinline__ void forward_wkv_seq_fp16(
			const uint32_t m, const uint32_t k, 
			uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			half* key, half* value, half* wkv
		)
		{
			att_aa += blk_id * thread_count * 8;
			att_bb += blk_id * thread_count * 8;
			att_pp += blk_id * thread_count * 8;

			att_time_decay += blk_id * thread_count * 8;
			att_time_first += blk_id * thread_count * 8;

			float4* ld_att_aa_vec4 = reinterpret_cast<float4*>(att_aa);
			float4* ld_att_bb_vec4 = reinterpret_cast<float4*>(att_bb);
			float4* ld_att_pp_vec4 = reinterpret_cast<float4*>(att_pp);

			float4* ld_att_time_decay_vec4 = reinterpret_cast<float4*>(att_time_decay);
			float4* ld_att_time_first_vec4 = reinterpret_cast<float4*>(att_time_first);

			float4 ld_att_aa_val[2] = { ld_att_aa_vec4[thread_id * 2], ld_att_aa_vec4[thread_id * 2 + 1] };
			float4 ld_att_bb_val[2] = { ld_att_bb_vec4[thread_id * 2], ld_att_bb_vec4[thread_id * 2 + 1] };
			float4 ld_att_pp_val[2] = { ld_att_pp_vec4[thread_id * 2], ld_att_pp_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_decay_val[2] = { ld_att_time_decay_vec4[thread_id * 2], ld_att_time_decay_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_first_val[2] = { ld_att_time_first_vec4[thread_id * 2], ld_att_time_first_vec4[thread_id * 2 + 1] };

			float4 st_wkv_val;

			half2* st_wkv_vec_h1 = (half2*)&st_wkv_val.x;
			half2* st_wkv_vec_h2 = (half2*)&st_wkv_val.y;
			half2* st_wkv_vec_h3 = (half2*)&st_wkv_val.z;
			half2* st_wkv_vec_h4 = (half2*)&st_wkv_val.w;

			for(uint32_t i=0; i<m; i++)
			{
				key += blk_id * thread_count * 8 + i * k;
				value += blk_id * thread_count * 8 + i * k;

				float4* ld_key_vec4 = reinterpret_cast<float4*>(key);
				float4* ld_value_vec4 = reinterpret_cast<float4*>(value);

				float4 ld_key_val = ld_key_vec4[thread_id];
				float4 ld_value_val = ld_value_vec4[thread_id];

				half2* ld_key_vec_h1 = (half2*)&ld_key_val.x;
				half2* ld_key_vec_h2 = (half2*)&ld_key_val.y;
				half2* ld_key_vec_h3 = (half2*)&ld_key_val.z;
				half2* ld_key_vec_h4 = (half2*)&ld_key_val.w;

				half2* ld_value_vec_h1 = (half2*)&ld_value_val.x;
				half2* ld_value_vec_h2 = (half2*)&ld_value_val.y;
				half2* ld_value_vec_h3 = (half2*)&ld_value_val.z;
				half2* ld_value_vec_h4 = (half2*)&ld_value_val.w;

				wkv += blk_id * thread_count * 8 + i * k;

				float4* st_wkv_vec4 = reinterpret_cast<float4*>(wkv);

				cal_wkv<half>(ld_att_aa_val[0].x, ld_att_bb_val[0].x, ld_att_pp_val[0].x, ld_att_time_decay_val[0].x, ld_att_time_first_val[0].x, ld_key_vec_h1->x, ld_value_vec_h1->x, st_wkv_vec_h1->x);
				cal_wkv<half>(ld_att_aa_val[0].y, ld_att_bb_val[0].y, ld_att_pp_val[0].y, ld_att_time_decay_val[0].y, ld_att_time_first_val[0].y, ld_key_vec_h1->y, ld_value_vec_h1->y, st_wkv_vec_h1->y);
				cal_wkv<half>(ld_att_aa_val[0].z, ld_att_bb_val[0].z, ld_att_pp_val[0].z, ld_att_time_decay_val[0].z, ld_att_time_first_val[0].z, ld_key_vec_h2->x, ld_value_vec_h2->x, st_wkv_vec_h2->x);
				cal_wkv<half>(ld_att_aa_val[0].w, ld_att_bb_val[0].w, ld_att_pp_val[0].w, ld_att_time_decay_val[0].w, ld_att_time_first_val[0].w, ld_key_vec_h2->y, ld_value_vec_h2->y, st_wkv_vec_h2->y);
				cal_wkv<half>(ld_att_aa_val[1].x, ld_att_bb_val[1].x, ld_att_pp_val[1].x, ld_att_time_decay_val[1].x, ld_att_time_first_val[1].x, ld_key_vec_h3->x, ld_value_vec_h3->x, st_wkv_vec_h3->x);
				cal_wkv<half>(ld_att_aa_val[1].y, ld_att_bb_val[1].y, ld_att_pp_val[1].y, ld_att_time_decay_val[1].y, ld_att_time_first_val[1].y, ld_key_vec_h3->y, ld_value_vec_h3->y, st_wkv_vec_h3->y);
				cal_wkv<half>(ld_att_aa_val[1].z, ld_att_bb_val[1].z, ld_att_pp_val[1].z, ld_att_time_decay_val[1].z, ld_att_time_first_val[1].z, ld_key_vec_h4->x, ld_value_vec_h4->x, st_wkv_vec_h4->x);
				cal_wkv<half>(ld_att_aa_val[1].w, ld_att_bb_val[1].w, ld_att_pp_val[1].w, ld_att_time_decay_val[1].w, ld_att_time_first_val[1].w, ld_key_vec_h4->y, ld_value_vec_h4->y, st_wkv_vec_h4->y);
			
				st_wkv_vec4[thread_id] = st_wkv_val;
			}
			
			ld_att_aa_vec4[thread_id * 2] = ld_att_aa_val[0];
			ld_att_aa_vec4[thread_id * 2 + 1] = ld_att_aa_val[1];
			ld_att_bb_vec4[thread_id * 2] = ld_att_bb_val[0];
			ld_att_bb_vec4[thread_id * 2 + 1] = ld_att_bb_val[1];
			ld_att_pp_vec4[thread_id * 2] = ld_att_pp_val[0];
			ld_att_pp_vec4[thread_id * 2 + 1] = ld_att_pp_val[1];
		}

		__device__ __forceinline__ void forward_wkv_seq_fp32(
			const uint32_t m, const uint32_t k,
			uint32_t blk_id, uint32_t thread_id, uint32_t thread_count,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			float* key, float* value, float* wkv
		)
		{
			att_aa += blk_id * thread_count * 8;
			att_bb += blk_id * thread_count * 8;
			att_pp += blk_id * thread_count * 8;

			att_time_decay += blk_id * thread_count * 8;
			att_time_first += blk_id * thread_count * 8;

			float4* ld_att_aa_vec4 = reinterpret_cast<float4*>(att_aa);
			float4* ld_att_bb_vec4 = reinterpret_cast<float4*>(att_bb);
			float4* ld_att_pp_vec4 = reinterpret_cast<float4*>(att_pp);

			float4* ld_att_time_decay_vec4 = reinterpret_cast<float4*>(att_time_decay);
			float4* ld_att_time_first_vec4 = reinterpret_cast<float4*>(att_time_first);

			float4 ld_att_aa_val[2] = { ld_att_aa_vec4[thread_id * 2], ld_att_aa_vec4[thread_id * 2 + 1] };
			float4 ld_att_bb_val[2] = { ld_att_bb_vec4[thread_id * 2], ld_att_bb_vec4[thread_id * 2 + 1] };
			float4 ld_att_pp_val[2] = { ld_att_pp_vec4[thread_id * 2], ld_att_pp_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_decay_val[2] = { ld_att_time_decay_vec4[thread_id * 2], ld_att_time_decay_vec4[thread_id * 2 + 1] };
			float4 ld_att_time_first_val[2] = { ld_att_time_first_vec4[thread_id * 2], ld_att_time_first_vec4[thread_id * 2 + 1] };

			float4 st_wkv_val[2];

			for(uint32_t i=0; i<m; i++)
			{
				auto _key = key + blk_id * thread_count * 8 + i * k;
				auto _value = value + blk_id * thread_count * 8 + i * k;

				float4* ld_key_vec4 = reinterpret_cast<float4*>(_key);
				float4* ld_value_vec4 = reinterpret_cast<float4*>(_value);

				float4 ld_key_val[2] = { ld_key_vec4[thread_id * 2], ld_key_vec4[thread_id * 2 + 1] };
				float4 ld_value_val[2] = { ld_value_vec4[thread_id * 2], ld_value_vec4[thread_id * 2 + 1] };

				auto _wkv = wkv + blk_id * thread_count * 8 + i * k;
				float4* st_wkv_vec4 = reinterpret_cast<float4*>(_wkv);

				cal_wkv<float>(ld_att_aa_val[0].x, ld_att_bb_val[0].x, ld_att_pp_val[0].x, ld_att_time_decay_val[0].x, ld_att_time_first_val[0].x, ld_key_val[0].x, ld_value_val[0].x, st_wkv_val[0].x);
				cal_wkv<float>(ld_att_aa_val[0].y, ld_att_bb_val[0].y, ld_att_pp_val[0].y, ld_att_time_decay_val[0].y, ld_att_time_first_val[0].y, ld_key_val[0].y, ld_value_val[0].y, st_wkv_val[0].y);
				cal_wkv<float>(ld_att_aa_val[0].z, ld_att_bb_val[0].z, ld_att_pp_val[0].z, ld_att_time_decay_val[0].z, ld_att_time_first_val[0].z, ld_key_val[0].z, ld_value_val[0].z, st_wkv_val[0].z);
				cal_wkv<float>(ld_att_aa_val[0].w, ld_att_bb_val[0].w, ld_att_pp_val[0].w, ld_att_time_decay_val[0].w, ld_att_time_first_val[0].w, ld_key_val[0].w, ld_value_val[0].w, st_wkv_val[0].w);
				cal_wkv<float>(ld_att_aa_val[1].x, ld_att_bb_val[1].x, ld_att_pp_val[1].x, ld_att_time_decay_val[1].x, ld_att_time_first_val[1].x, ld_key_val[1].x, ld_value_val[1].x, st_wkv_val[1].x);
				cal_wkv<float>(ld_att_aa_val[1].y, ld_att_bb_val[1].y, ld_att_pp_val[1].y, ld_att_time_decay_val[1].y, ld_att_time_first_val[1].y, ld_key_val[1].y, ld_value_val[1].y, st_wkv_val[1].y);
				cal_wkv<float>(ld_att_aa_val[1].z, ld_att_bb_val[1].z, ld_att_pp_val[1].z, ld_att_time_decay_val[1].z, ld_att_time_first_val[1].z, ld_key_val[1].z, ld_value_val[1].z, st_wkv_val[1].z);
				cal_wkv<float>(ld_att_aa_val[1].w, ld_att_bb_val[1].w, ld_att_pp_val[1].w, ld_att_time_decay_val[1].w, ld_att_time_first_val[1].w, ld_key_val[1].w, ld_value_val[1].w, st_wkv_val[1].w);
				
				st_wkv_vec4[thread_id * 2] = st_wkv_val[0];
				st_wkv_vec4[thread_id * 2 + 1] = st_wkv_val[1];
			}
			
			ld_att_aa_vec4[thread_id * 2] = ld_att_aa_val[0];
			ld_att_aa_vec4[thread_id * 2 + 1] = ld_att_aa_val[1];
			ld_att_bb_vec4[thread_id * 2] = ld_att_bb_val[0];
			ld_att_bb_vec4[thread_id * 2 + 1] = ld_att_bb_val[1];
			ld_att_pp_vec4[thread_id * 2] = ld_att_pp_val[0];
			ld_att_pp_vec4[thread_id * 2 + 1] = ld_att_pp_val[1];
		}

		template<typename T>
		__global__ void forward_wkv_seq(
			const uint32_t m, const uint32_t k,
			float* att_aa, float* att_bb, float* att_pp,
			float* att_time_decay, float* att_time_first,
			T* key, T* value, T* wkv
		)
		{
			const uint32_t thread_id = threadIdx.x;
			const uint32_t thread_count = blockDim.x;
			const uint32_t blk_id = blockIdx.x;

			if constexpr (sizeof(T) == 2)
				forward_wkv_seq_fp16(m, k, blk_id, thread_id, thread_count, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);
			else
				forward_wkv_seq_fp32(m, k, blk_id, thread_id, thread_count, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);
		}
	}

	inline cudaError_t convert_att_time_decay(tensor_t& from, tensor_t& to)
	{
		const uint32_t count = from.shape.x * from.shape.y * from.shape.z;

		using minus_exp_t = elementwise::functor::minus_exp::functor_t<float, half>;

		elementwise::unary(minus_exp_t(), count, (float*)to.data, (half*)from.data);

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
		const uint32_t count = m * k;
		const uint32_t block_dim = count / 256;
		const uint32_t thread_dim = 32;

		kernel::mix_rkv<T><<<block_dim, thread_dim>>>(m, k, xx, sx, mix_r, mix_k, mix_v, rx, kx, vx);
		
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
		const uint32_t count = m * k;
		const uint32_t block_dim = count / 256;
		const uint32_t thread_dim = 32;

		kernel::mix_rk<T> << <block_dim, thread_dim >> > (m, k, xx, sx, mix_r, mix_k, rx, kx);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t forward_wkv_one(
		const uint32_t emb_size,
		float* att_aa, float* att_bb, float* att_pp,
		float* att_time_decay, float* att_time_first,
		T* key, T* value, T* wkv
	)
	{
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = emb_size / 256;

		kernel::forward_wkv_one<T><<<block_dim, thread_dim>>>(1, emb_size, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);

		return cudaGetLastError();
	}

	template<typename T>
	inline cudaError_t forward_wkv_seq(
		const uint32_t token_size, const uint32_t emb_size,
		float* att_aa, float* att_bb, float* att_pp,
		float* att_time_decay, float* att_time_first,
		T* key, T* value, T* wkv
	)
	{
		const uint32_t thread_dim = 32;
		const uint32_t block_dim = emb_size / 256;

		kernel::forward_wkv_seq<T> << <block_dim, thread_dim >> > (token_size, emb_size, att_aa, att_bb, att_pp, att_time_decay, att_time_first, key, value, wkv);

		return cudaGetLastError();
	}

	inline void time_mix_one(
		tensor_t& x,
		tensor_t& att_xx, tensor_t& att_aa, tensor_t& att_bb, tensor_t& att_pp,
		tensor_t& ln1_weight, tensor_t& ln1_bias,
		tensor_t& att_time_mix_r, tensor_t& att_time_mix_k, tensor_t& att_time_mix_v,
		tensor_t& att_time_decay, tensor_t& att_time_first,
		tensor_t& att_key_weight, tensor_t& att_value_weight, tensor_t& att_receptance_weight, tensor_t& att_output_weight
	)
	{
		using T = float;
		const uint32_t emb_size = x.shape.x;
		const uint32_t token_size = 1;

		auto result = cudaSuccess;

		auto xx = cuda::create_tensor<T>(x.shape);

		cuda::inspect_tensor(x);
		result = cuda::layernorm<T>((T*)x.data, (half*)ln1_weight.data, (half*)ln1_bias.data, (T*)xx.data, token_size, emb_size);	
		cuda::inspect_tensor(xx);

		auto rx = cuda::create_tensor<T>(x.shape);
		auto kx = cuda::create_tensor<T>(x.shape);
		auto vx = cuda::create_tensor<T>(x.shape);

		cuda::inspect_tensor(xx);
		cuda::inspect_tensor(att_xx);
		result = cuda::mix_rkv<T>(token_size, emb_size, (T*)xx.data, (T*)att_xx.data, (half*)att_time_mix_r.data, (half*)att_time_mix_k.data, (half*)att_time_mix_v.data, (T*)rx.data, (T*)kx.data, (T*)vx.data);
		cuda::sync();
		cuda::inspect_tensor(rx);
		cuda::inspect_tensor(kx);
		cuda::inspect_tensor(vx);

		result = cuda::copy<T>((T*)att_xx.data, (T*)xx.data, emb_size);

		// m = y, k = x
		auto r = cuda::create_tensor<T>({ att_receptance_weight.shape.x, 1, 1 });
		result = cuda::gemv_sigmoid((half*)att_receptance_weight.data, (T*)rx.data, (T*)r.data, att_receptance_weight.shape.x, att_receptance_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(r);

		auto k = cuda::create_tensor<T>({ att_key_weight.shape.x, 1, 1 });
		result = cuda::gemv((half*)att_key_weight.data, (T*)kx.data, (T*)k.data, att_key_weight.shape.x, att_key_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(k);

		cuda::inspect_tensor(att_value_weight);
		auto v = cuda::create_tensor<T>({ att_value_weight.shape.x, 1, 1 });
		result = cuda::gemv((half*)att_value_weight.data, (T*)vx.data, (T*)v.data, att_value_weight.shape.x, att_value_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(v);

		auto wkv = cuda::create_tensor<T>(x.shape);
		result = cuda::forward_wkv_one<T>(emb_size, (float*)att_aa.data, (float*)att_bb.data, (float*)att_pp.data, (float*)att_time_decay.data, (float*)att_time_first.data, (T*)k.data, (T*)v.data, (T*)wkv.data);
		cuda::sync();
		cuda::inspect_tensor(wkv);

		auto r_dot_y = cuda::create_tensor<T>(x.shape);
		result = cuda::element_wise_product<T>(emb_size, (T*)r.data, (T*)wkv.data, (T*)r_dot_y.data);
		cuda::sync();
		cuda::inspect_tensor(r_dot_y);

		auto out = cuda::create_tensor<T>({ att_output_weight.shape.x, 1, 1 });
		result = cuda::gemv((half*)att_output_weight.data, (T*)r_dot_y.data, (T*)out.data, att_output_weight.shape.x, att_output_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(out);

		result = cuda::element_wise_add<T>(emb_size, (T*)x.data, (T*)out.data, (T*)x.data);
		cuda::sync();
		cuda::inspect_tensor(x);

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

	inline void channel_mix_one(
		tensor_t& x,
		tensor_t& ffn_xx,
		tensor_t& ln2_weight, tensor_t& ln2_bias,
		tensor_t& ffn_time_mix_r, tensor_t& ffn_time_mix_k,
		tensor_t& ffn_key_weight, tensor_t& ffn_value_weight, tensor_t& ffn_receptance_weight
	)
	{

		using T = float;

		const uint32_t emb_size = x.shape.x;
		const uint32_t token_size = 1;

		auto result = cudaSuccess;

		// numeric problem

		auto xx = cuda::create_tensor<T>(x.shape);

		result = cuda::layernorm<T>((T*)x.data, (half*)ln2_weight.data, (half*)ln2_bias.data, (T*)xx.data, token_size, emb_size);
		cuda::sync();
		cuda::inspect_tensor(xx);

		auto rx = cuda::create_tensor<T>(x.shape);
		auto kx = cuda::create_tensor<T>(x.shape);

		result = cuda::mix_rk<T>(token_size, emb_size, (T*)xx.data, (T*)ffn_xx.data, (half*)ffn_time_mix_r.data, (half*)ffn_time_mix_k.data, (T*)rx.data, (T*)kx.data);
		cuda::sync();
		cuda::inspect_tensor(rx);
		cuda::inspect_tensor(kx);

		result = cuda::copy<T>((T*)ffn_xx.data, (T*)xx.data, emb_size);
		cuda::sync();

		// m = y, k = x
		auto r = cuda::create_tensor<T>({ ffn_receptance_weight.shape.x, 1, 1 });
		result = cuda::gemv_sigmoid((half*)ffn_receptance_weight.data, (T*)rx.data, (T*)r.data, ffn_receptance_weight.shape.x, ffn_receptance_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(r);

		//cuda::inspect_tensor(ffn_key_weight);
		auto vx = cuda::create_tensor<T>({ ffn_key_weight.shape.x, 1, 1 });
		result = cuda::gemv_square_relu((half*)ffn_key_weight.data, (T*)kx.data, (T*)vx.data, ffn_key_weight.shape.x, ffn_key_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(vx);
		// tbd square relu has problem
		//inspect_tensor(vx);

		auto vx_vw_product = cuda::create_tensor<T>({ ffn_value_weight.shape.x, 1, 1 });
		result = cuda::gemv((half*)ffn_value_weight.data, (T*)vx.data, (T*)vx_vw_product.data, ffn_value_weight.shape.x, ffn_value_weight.shape.y);
		cuda::sync();
		cuda::inspect_tensor(vx_vw_product);

		// wrong value
		//inspect_tensor(vx_vw_product);

		auto out = cuda::create_tensor<T>(x.shape);
		result = cuda::element_wise_product<T>(x.shape.x, (T*)r.data, (T*)vx_vw_product.data, (T*)out.data);
		cuda::sync();
		cuda::inspect_tensor(out);
		
		//inspect_tensor(out);

		result = cuda::element_wise_add<T>(emb_size, (T*)x.data, (T*)out.data, (T*)x.data);
		cuda::sync();

		cuda::free_tensor(xx);
		cuda::free_tensor(rx);
		cuda::free_tensor(kx);

		cuda::free_tensor(r);
		cuda::free_tensor(vx);
		cuda::free_tensor(vx_vw_product);
		cuda::free_tensor(out);
	}

	/*
		def cuda_att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
			T, C = x.size()
			xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
			sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
			kx = xx * k_mix + sx * (1 - k_mix)
			vx = xx * v_mix + sx * (1 - v_mix)
			rx = xx * r_mix + sx * (1 - r_mix)

			r = torch.sigmoid(rx @ rw)
			k = kx @ kw
			v = vx @ vw
			y, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)

			out = (r * y) @ ow
			return x + out, xx[-1,:], aa, bb, pp
	*/

	inline void time_mix_seq(
		tensor_t& x,
		tensor_t& att_xx, tensor_t& att_aa, tensor_t& att_bb, tensor_t& att_pp,
		tensor_t& ln1_weight, tensor_t& ln1_bias,
		tensor_t& att_time_mix_r, tensor_t& att_time_mix_k, tensor_t& att_time_mix_v,
		tensor_t& att_time_decay, tensor_t& att_time_first,
		tensor_t& att_key_weight, tensor_t& att_value_weight, tensor_t& att_receptance_weight, tensor_t& att_output_weight
	)
	{
		using T = float;
		const uint32_t emb_size = x.shape.x;
		const uint32_t token_size = x.shape.y;

		auto result = cudaSuccess;

		auto xx = cuda::create_tensor<T>(x.shape);

		cuda::inspect_tensor(x);
		// xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
		result = cuda::layernorm<T>((T*)x.data, (half*)ln1_weight.data, (half*)ln1_bias.data, (T*)xx.data, token_size, emb_size);
		cuda::inspect_tensor(xx);

		auto rx = cuda::create_tensor<T>(x.shape);
		auto kx = cuda::create_tensor<T>(x.shape);
		auto vx = cuda::create_tensor<T>(x.shape);

		// sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
		auto sx = cuda::create_tensor<T>(x.shape);
		{
			cuda::copy_fp32((float*)sx.data, (float*)att_xx.data, emb_size);
			cuda::copy_fp32(((float*)sx.data) + emb_size, (float*)xx.data, emb_size * (token_size - 1));
		}
		cuda::inspect_tensor(sx);

		// kx = xx * k_mix + sx * (1 - k_mix)
		// vx = xx * v_mix + sx * (1 - v_mix)
		// rx = xx * r_mix + sx * (1 - r_mix)
		cuda::inspect_tensor(xx);
		cuda::inspect_tensor(sx);
		result = cuda::mix_rkv<T>(token_size, emb_size, (T*)xx.data, (T*)sx.data, (half*)att_time_mix_r.data, (half*)att_time_mix_k.data, (half*)att_time_mix_v.data, (T*)rx.data, (T*)kx.data, (T*)vx.data);
		//cuda::sync();
		cuda::inspect_tensor(rx);
		cuda::inspect_tensor(kx);
		cuda::inspect_tensor(vx);

		result = cuda::copy<T>((T*)att_xx.data, ((T*)xx.data + emb_size * (token_size - 1)), emb_size);

		// m = y, k = x
		// a = m * k 
		// b = k * n
		// c = m * n
		
		// r = torch.sigmoid(rx @ rw)
		auto r = cuda::create_tensor<T>({ att_receptance_weight.shape.x, rx.shape.y, 1 });
		result = cuda::gemm_sigmoid<float, half, float>((float*)rx.data, (half*)att_receptance_weight.data, (float*)r.data, rx.shape.y, rx.shape.x, att_receptance_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(r);

		// k = kx @ kw
		auto k = cuda::create_tensor<T>({ att_key_weight.shape.x, kx.shape.y, 1 });
		result = cuda::gemm<float, half, float>((float*)kx.data, (half*)att_key_weight.data, (float*)k.data, kx.shape.y, kx.shape.x, att_key_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(k);
		
		// v = vx @ vw
		cuda::inspect_tensor(att_value_weight);
		auto v = cuda::create_tensor<T>({ att_value_weight.shape.x, vx.shape.y, 1 });
		result = cuda::gemm<float, half, float>((float*)vx.data, (half*)att_value_weight.data, (float*)v.data, vx.shape.y, vx.shape.x, att_value_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(v);
		
		// y, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)
		auto wkv = cuda::create_tensor<T>(x.shape);
		result = cuda::forward_wkv_seq<T>(token_size, emb_size, (float*)att_aa.data, (float*)att_bb.data, (float*)att_pp.data, (float*)att_time_decay.data, (float*)att_time_first.data, (T*)k.data, (T*)v.data, (T*)wkv.data);
		//cuda::sync();
		cuda::inspect_tensor(wkv);
		
		// out = (r * y) @ ow
		auto r_dot_y = cuda::create_tensor<T>(x.shape);
		result = cuda::element_wise_product<T>(token_size * emb_size, (T*)r.data, (T*)wkv.data, (T*)r_dot_y.data);
		//cuda::sync();
		cuda::inspect_tensor(r_dot_y);
		
		auto out = cuda::create_tensor<T>({ att_output_weight.shape.x, r_dot_y.shape.y, 1 });
		result = cuda::gemm<float, half, float>((float*)r_dot_y.data, (half*)att_output_weight.data, (float*)out.data, r_dot_y.shape.y, r_dot_y.shape.x, att_output_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(out);
		
		result = cuda::element_wise_add<T>(token_size * emb_size, (T*)x.data, (T*)out.data, (T*)x.data);
		//cuda::sync();
		cuda::inspect_tensor(x);
		
		cuda::free_tensor(xx);
		cuda::free_tensor(sx);
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

	/*
		def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
			xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
			sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
			kx = xx * k_mix + sx * (1 - k_mix)
			rx = xx * r_mix + sx * (1 - r_mix)

			r = torch.sigmoid(rx @ rw)
			vx = torch.square(torch.relu(kx @ kw))
			out = r * (vx @ vw)
			return x + out, xx[-1,:]
	*/
	inline void channel_mix_seq(
		tensor_t& x,
		tensor_t& ffn_xx,
		tensor_t& ln2_weight, tensor_t& ln2_bias,
		tensor_t& ffn_time_mix_r, tensor_t& ffn_time_mix_k,
		tensor_t& ffn_key_weight, tensor_t& ffn_value_weight, tensor_t& ffn_receptance_weight
	)
	{
		using T = float;

		const uint32_t emb_size = x.shape.x;
		const uint32_t token_size = x.shape.y;

		auto result = cudaSuccess;

		auto xx = cuda::create_tensor<T>(x.shape);

		// xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
		result = cuda::layernorm<T>((T*)x.data, (half*)ln2_weight.data, (half*)ln2_bias.data, (T*)xx.data, token_size, emb_size);
		//cuda::sync();
		cuda::inspect_tensor(xx);

		auto rx = cuda::create_tensor<T>(x.shape);
		auto kx = cuda::create_tensor<T>(x.shape);

		// sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
		auto sx = cuda::create_tensor<T>(x.shape);
		{
			cuda::copy_fp32((float*)sx.data, (float*)ffn_xx.data, emb_size);
			cuda::copy_fp32(((float*)sx.data) + emb_size, (float*)xx.data, emb_size * (token_size - 1));
		}

		// kx = xx * k_mix + sx * (1 - k_mix)
		// rx = xx * r_mix + sx * (1 - r_mix)
		result = cuda::mix_rk<T>(token_size, emb_size, (T*)xx.data, (T*)sx.data, (half*)ffn_time_mix_r.data, (half*)ffn_time_mix_k.data, (T*)rx.data, (T*)kx.data);
		//cuda::sync();
		cuda::inspect_tensor(rx);
		cuda::inspect_tensor(kx);

		result = cuda::copy<T>((T*)ffn_xx.data, ((T*)xx.data + emb_size * (token_size - 1)), emb_size);
		//cuda::sync();

		// m = y, k = x
		// r = torch.sigmoid(rx @ rw)
		auto r = cuda::create_tensor<T>({ ffn_receptance_weight.shape.x, rx.shape.y, 1 });
		result = cuda::gemm_sigmoid<float, half, float>((float*)rx.data, (half*)ffn_receptance_weight.data, (float*)r.data, rx.shape.y, rx.shape.x, ffn_receptance_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(r);

		//cuda::inspect_tensor(ffn_key_weight);
		// vx = torch.square(torch.relu(kx @ kw))
		auto vx = cuda::create_tensor<T>({ ffn_key_weight.shape.x, kx.shape.y, 1 });
		result = cuda::gemm_square_relu<float, half, float>((float*)kx.data, (half*)ffn_key_weight.data, (float*)vx.data, kx.shape.y, kx.shape.x, ffn_key_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(vx);
		// tbd square relu has problem
		//inspect_tensor(vx);

		// out = r * (vx @ vw)
		auto vx_vw_product = cuda::create_tensor<T>({ ffn_value_weight.shape.x, vx.shape.y, 1 });
		result = cuda::gemm<float, half, float>((float*)vx.data, (half*)ffn_value_weight.data, (float*)vx_vw_product.data, vx.shape.y, vx.shape.x, ffn_value_weight.shape.x);
		//cuda::sync();
		cuda::inspect_tensor(vx_vw_product);

		// wrong value
		//inspect_tensor(vx_vw_product);

		auto out = cuda::create_tensor<T>(x.shape);
		result = cuda::element_wise_product<T>(emb_size * token_size, (T*)r.data, (T*)vx_vw_product.data, (T*)out.data);
		//cuda::sync();
		cuda::inspect_tensor(out);

		//inspect_tensor(out);

		// out = r * (vx @ vw)
		result = cuda::element_wise_add<T>(emb_size * token_size, (T*)x.data, (T*)out.data, (T*)x.data);
		//cuda::sync();

		cuda::free_tensor(xx);
		cuda::free_tensor(sx);
		cuda::free_tensor(rx);
		cuda::free_tensor(kx);

		cuda::free_tensor(r);
		cuda::free_tensor(vx);
		cuda::free_tensor(vx_vw_product);
		cuda::free_tensor(out);
	}

	inline cudaError_t token_to_emb(uint16_t* tokens, uint32_t token_size, tensor_t emb_weight, tensor_t& emb)
	{
		auto emb_size = emb_weight.shape.x;

		half* dst = (half*)emb.data;
		half* src = (half*)emb_weight.data;

		for (uint32_t i = 0; i < token_size; i++)
			cuda::load_fp16(dst + i * emb_size, src + tokens[i] * emb_size, emb_size);

		return cudaGetLastError();
	}

	template<typename T>
	inline void emb_to_logits(tensor_t& x, uint32_t token_size, tensor_t& head_weight, float* logits)
	{
		auto logits_out = cuda::create_tensor<T>({ head_weight.shape.x, 1, 1 });
		auto result = cuda::gemv((half*)head_weight.data, (T*)x.data + (token_size -1) * x.shape.x, (T*)logits_out.data, head_weight.shape.x, head_weight.shape.y);
		//cuda::sync();

		cuda::dump_fp32(logits, (float*)logits_out.data, head_weight.shape.x);

		cuda::free_tensor(logits_out);
	}
}



#endif
