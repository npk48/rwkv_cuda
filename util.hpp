#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <vector>

namespace cuda
{
	struct timer_t
	{
		cudaEvent_t evt_start;
		cudaEvent_t evt_stop;

		timer_t()
		{
			cudaEventCreate(&evt_start);
			cudaEventCreate(&evt_stop);
		}

		~timer_t()
		{
			cudaEventDestroy(evt_start);
			cudaEventDestroy(evt_stop);
		}

		void start()
		{
			cudaEventRecord(evt_start, 0);
		}

		void stop()
		{
			cudaEventRecord(evt_stop, 0);
		}

		float elapsed_ms()
		{
			float elapsed;
			cudaEventSynchronize(evt_stop);
			cudaEventElapsedTime(&elapsed, evt_start, evt_stop);
			return elapsed;
		}
	};

	template<typename T>
	inline T* malloc(uint32_t quantity)
	{
		T* ptr = 0;
		cudaMalloc(&ptr, quantity * sizeof(T));
		return ptr;
	}

	inline void free(void* ptr)
	{
		cudaFree(ptr);
	}

	inline void fill_fp16(half* d_dst, float value, uint64_t count)
	{
		half fp16 = __float2half(value);
		cuMemsetD16((CUdeviceptr)(d_dst), *(uint16_t*)&fp16, count);
	}

	inline void fill_fp32(float* d_dst, float value, uint64_t count)
	{
		float fp32 = value;
		cuMemsetD32((CUdeviceptr)(d_dst), *(uint32_t*)&fp32, count);
	}

	template<typename T>
	inline void fill(T* dst, float value, uint64_t count)
	{
		if constexpr (sizeof(T) == 2)
			fill_fp16((half*)dst, value, count);
		else
			fill_fp32((float*)dst, value, count);
	}

	inline void zero_memory(void* d_dst, uint64_t size)
	{
		cudaMemset(d_dst, 0, size);
	}

	// inline void dump_fp16(float* h_dst, half* d_src, uint32_t count)
	// {
	// 	std::vector<half> tmp(count);
	// 	assert(cudaMemcpy(&tmp[0], d_src, sizeof(half) * count, cudaMemcpyKind::cudaMemcpyDeviceToHost) == cudaSuccess);
	// 	for (uint32_t i = 0; i < count; i++)
	// 		h_dst[i] = __half2float(tmp[i]);
	// }

	inline void dump(void* h_dst, void* d_src, uint64_t size)
	{
		cudaMemcpy(h_dst, d_src, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	inline void dump_fp32(float* h_dst, float* d_src, uint32_t count)
	{
		cudaMemcpy(&h_dst[0], d_src, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	inline void load_fp16(half* d_dst, half* h_src, uint32_t count)
	{
		cudaMemcpy(d_dst, h_src, sizeof(half) * count, cudaMemcpyKind::cudaMemcpyHostToDevice);
	}

	inline void load_fp32(float* d_dst, float* h_src, uint32_t count)
	{
		cudaMemcpy(d_dst, h_src, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyHostToDevice);
	}

	inline void copy_fp16(half* d_dst, half* d_src, uint32_t count)
	{
		cudaMemcpy(d_dst, d_src, sizeof(half) * count, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	}

	inline void copy_fp32(float* d_dst, float* d_src, uint32_t count)
	{
		cudaMemcpy(d_dst, d_src, sizeof(float) * count, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	}

	template<typename T>
	inline cudaError_t copy(T* d_dst, T* d_src, uint32_t count)
	{
		return cudaMemcpy(d_dst, d_src, sizeof(T) * count, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	}

	static inline void sync()
	{
		cudaDeviceSynchronize();
	}

	enum data_type_t
	{
		fp16,
		bf16,
		fp32,

		int4,
		int8
	};

	struct tensor_shape_t
	{
		uint32_t x;
		uint32_t y;
		uint32_t z;

		tensor_shape_t(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}
	};

	struct tensor_t
	{
		uint8_t* data;
		data_type_t type;

		tensor_shape_t shape;
		uint64_t byte_size;
	};

	template<typename T>
	inline tensor_t create_tensor(tensor_shape_t shape)
	{
		tensor_t tensor = { nullptr, sizeof(T) == 2 ? data_type_t::fp16 : data_type_t::fp32, shape };

		uint64_t mem_size = sizeof(T);
		mem_size *= shape.x;
		mem_size *= shape.y;
		mem_size *= shape.z;

		tensor.byte_size = mem_size;

		cudaMalloc((void**)&tensor.data, mem_size);

		return tensor;
	}

	inline void free_tensor(tensor_t& tensor)
	{
		cudaFree(tensor.data);
	}

	inline void inspect_tensor(cuda::tensor_t& tensor)
	{
		//return;
		static bool do_inspect = false;
		if (!do_inspect) return;
		
		auto tensor_size = tensor.shape.x * tensor.shape.y * tensor.shape.z;

		//std::vector<float> h_tensor(tensor_size);
		float* h_tensor = new float[tensor_size];

		if (tensor.type == cuda::data_type_t::fp32)
			cuda::dump_fp32(&h_tensor[0], (float*)tensor.data, tensor_size);

		delete[] h_tensor;
		do_inspect = false;
	}
}

#endif