#include "thrust.hpp"

#include "util.hpp"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace thrust
{
	void argsort(uint32_t* index_out, float* arr, uint32_t size)
	{
		auto const seq_iter = thrust::make_counting_iterator(static_cast<uint32_t>(0));

		thrust::device_ptr<uint32_t> dev_index = thrust::device_pointer_cast(index_out);
		thrust::copy(seq_iter, seq_iter + size, dev_index);

		thrust::sort(dev_index, dev_index + size,
			[arr] __host__ __device__(uint32_t left_idx, uint32_t right_idx)
		{
			return arr[left_idx] < arr[right_idx];
		}
		);

		// thrust::raw_pointer_cast(&dev_index[0]);
	}

	void argsort_host(uint32_t* index_out, float* arr, uint32_t size)
	{
		float* dev_arr = nullptr;
		cudaMalloc(&dev_arr, size * sizeof(float));
		cudaMemcpy(dev_arr, arr, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		uint32_t* dev_index = nullptr;
		cudaMalloc(&dev_index, size * sizeof(uint32_t));

        thrust::argsort(dev_index, dev_arr, size);

        cudaMemcpy(index_out, dev_index, size * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaFree(dev_index);
		cudaFree(dev_arr);
	}

	void cumsum(float* arr_out, float* arr_in, uint32_t size)
	{
		thrust::device_ptr<float> dev_arr_in = thrust::device_pointer_cast(arr_in);
		thrust::device_ptr<float> dev_arr_out = thrust::device_pointer_cast(arr_out);

		thrust::inclusive_scan(thrust::device, dev_arr_in, dev_arr_in + size, dev_arr_out);
	}

	void cumsum_host(float* arr_out, float* arr_in, uint32_t size)
	{
		float* dev_arr = nullptr;
		cudaMalloc(&dev_arr, size * sizeof(float));
		cudaMemcpy(dev_arr, arr_in, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		float* dev_sum = nullptr;
		cudaMalloc(&dev_sum, size * sizeof(float));

		cumsum(dev_sum, dev_arr, size);

		cudaMemcpy(arr_out, dev_sum, size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaFree(dev_sum);
		cudaFree(dev_arr);
	}
}