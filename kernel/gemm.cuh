#ifndef _GEMM_CUH_
#define _GEMM_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cutlass/arch/mma_sm75.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/gemm/kernel/gemv.h>

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>


#include "activation.cuh"

namespace cuda
{
	// a * b = c
	// a = m * k 
	// b = k * n
	// c = m * n

	using mma_op_class = cutlass::arch::OpClassSimt;
	using sm_arch = cutlass::arch::Sm75;

	template<typename TA, typename TB, typename TC>
	inline cudaError_t gemm(
		TA* a, TB* b,
		TC* c,
		int m, int k, int n
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, TA, TB, TC, float>;

		using gemm = typename cutlass::gemm::device::Gemm <
			TA, cutlass::layout::RowMajor,
			TB, cutlass::layout::ColumnMajor,
			TC, cutlass::layout::RowMajor,
			float,
			mma_op_class,
			sm_arch,
			typename default_config::ThreadblockShape,
			typename default_config::WarpShape,
			typename default_config::InstructionShape,
			epilogue_op_nop<TC>
		>;

		typename gemm gemm_op;

		typename gemm::Arguments args(
			{ m, n, k },
			{ a, k },
			{ b, k },
			{ c, n },
			{ c, n },
			{ 1, 0 }
		);

		auto status = gemm_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

	template<typename TA, typename TB, typename TC>
	inline cudaError_t gemm_sigmoid(
		TA* a, TB* b,
		TC* c,
		int m, int k, int n
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, TA, TB, TC, float>;

		using gemm = typename cutlass::gemm::device::Gemm <
			TA, cutlass::layout::RowMajor,
			TB, cutlass::layout::ColumnMajor,
			TC, cutlass::layout::RowMajor,
			float,
			mma_op_class,
			sm_arch,
			typename default_config::ThreadblockShape,
			typename default_config::WarpShape,
			typename default_config::InstructionShape,
			epilogue_op_sigmoid<TC>
		>;

		typename gemm gemm_op;

		typename gemm::Arguments args(
			{ m, n, k },
			{ a, k },
			{ b, k },
			{ c, n },
			{ c, n },
			{ 1, 0 }
		);

		auto status = gemm_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

	template<typename TA, typename TB, typename TC>
	inline cudaError_t gemm_square_relu(
		TA* a, TB* b,
		TC* c,
		int m, int k, int n
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, TA, TB, TC, float>;

		using gemm = typename cutlass::gemm::device::Gemm <
			TA, cutlass::layout::RowMajor,
			TB, cutlass::layout::ColumnMajor,
			TC, cutlass::layout::RowMajor,
			float,
			mma_op_class,
			sm_arch,
			typename default_config::ThreadblockShape,
			typename default_config::WarpShape,
			typename default_config::InstructionShape,
			epilogue_op_square_relu<TC>
		>;

		typename gemm gemm_op;

		typename gemm::Arguments args(
			{ m, n, k },
			{ a, k },
			{ b, k },
			{ c, n },
			{ c, n },
			{ 1, 0 }
		);

		auto status = gemm_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

#define USE_CUTLASS_GEMV
#ifdef USE_CUTLASS_GEMV

	template<typename T>
	inline cudaError_t gemv(
		half* a, T* b,
		T* c,
		int m, int k
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, half, T, T, float>;

		using gemv = cutlass::gemm::device::Gemv<
			cutlass::gemm::kernel::Gemv<
			half, cutlass::layout::RowMajor,
			T,
			T,
			float,
			epilogue_op_nop<T>,
			8,	// Number of elements involved in a global access.
			32,	// Number of threads in the thread block.
			16	// Number of threads in the k dimension.
			>
		>;

		typename gemv gemv_op;

		typename gemv::Arguments args(
			{ m, k },
			1, // batch count
			{ 1, 0 },
			{ a, m },
			b,
			c,
			c,
			m*k, k, m, m // batch stride a/b/c/d
		);

		auto status = gemv_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

	template<typename T>
	inline cudaError_t gemv_sigmoid(
		half* a, T* b,
		T* c,
		int m, int k
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, half, T, T, float>;

		using gemv = cutlass::gemm::device::Gemv<
			cutlass::gemm::kernel::Gemv<
			half, cutlass::layout::RowMajor,
			T,
			T,
			float,
			epilogue_op_sigmoid<T>,
			8,	// Number of elements involved in a global access.
			32,	// Number of threads in the thread block.
			16	// Number of threads in the k dimension.
			>
		>;

		typename gemv gemv_op;

		typename gemv::Arguments args(
			{ m, k },
			1, // batch count
			{ 1, 0 },
			{ a, m },
			b,
			c,
			c,
			m * k, k, m, m // batch stride a/b/c/d
		);

		auto status = gemv_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

	template<typename T>
	inline cudaError_t gemv_square_relu(
		half* a, T* b,
		T* c,
		int m, int k
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, half, T, T, float>;

		using gemv = cutlass::gemm::device::Gemv<
			cutlass::gemm::kernel::Gemv<
			half, cutlass::layout::RowMajor,
			T,
			T,
			float,
			epilogue_op_square_relu<T>,
			8,	// Number of elements involved in a global access.
			32,	// Number of threads in the thread block.
			16	// Number of threads in the k dimension.
			>
		>;

		typename gemv gemv_op;

		typename gemv::Arguments args(
			{ m, k },
			1, // batch count
			{ 1, 0 },
			{ a, m },
			b,
			c,
			c,
			m * k, k, m, m // batch stride a/b/c/d
		);

		auto status = gemv_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

#else

	namespace kernel
	{
		__device__ __forceinline__ float simple_gemv_sigmoid(float x)
		{
			return 1.f / (1.f + exp(-x));
		}
	
		__device__ __forceinline__ float simple_gemv_square_relu(float x)
		{
			return (x > 0.f ? x : 0.f) * (x > 0.f ? x : 0.f);
		}

		const uint32_t gemv_op_nop = 0;
		const uint32_t gemv_op_sigmoid = 1;
		const uint32_t gemv_op_square_relu = 2;

	
		template<uint32_t op>
		__global__ void simple_gemv(half* a, float* b, float* c, uint32_t m, uint32_t k)
		{
			// one thread per row
			uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

			if (tid < m) 
			{
				float sum = 0;

#pragma unroll
				for (uint32_t i = 0; i < k; i++)
					sum += b[i] * (float)a[i * m + tid];

				if constexpr (op == gemv_op_nop)
					c[tid] = sum;
				else if constexpr (op == gemv_op_sigmoid)
					c[tid] = simple_gemv_sigmoid(sum);
				else if constexpr (op == gemv_op_square_relu)
					c[tid] = simple_gemv_square_relu(sum);
			}
		}

		template<uint32_t op>
		__global__ void fast_gemv(half* a, float* b, float* c, uint32_t m, uint32_t k)
		{
			const uint32_t warp_size = 32;

			const uint32_t
				i = blockDim.x * blockIdx.x + threadIdx.x,
				lane_id = i % warp_size,
				j_beg_last = k / warp_size * warp_size;

			float sum = 0;
			for (uint32_t j_beg = 0; j_beg < j_beg_last; j_beg += warp_size)
			{
				const float val = b[j_beg + lane_id];
				for (uint32_t j = 0; j < warp_size; ++j) 
					sum += (float)a[(j + j_beg) * m + i] * __shfl_sync(0xffffffff, val, j, warp_size);
			}
			{
				const uint32_t val = j_beg_last + lane_id < k ? b[j_beg_last + lane_id] : 0;
				for (size_t j = 0; j < k - j_beg_last; ++j) 
					sum += (float)a[(j + j_beg_last) * m + i] * __shfl_sync(0xffffffff, val, j, warp_size);
			}
			if (i < m)
			{
				if constexpr (op == gemv_op_nop)
					c[i] = sum;
				else if constexpr (op == gemv_op_sigmoid)
					c[i] = simple_gemv_sigmoid(sum);
				else if constexpr (op == gemv_op_square_relu)
					c[i] = simple_gemv_square_relu(sum);
			}
		}
	}

	inline cudaError_t gemv(
		half* a, float* b,
		float* c,
		uint32_t m, uint32_t k
	)
	{
		dim3 dim_grid((m + 31) / 32);
		dim3 dim_block(32);

		kernel::fast_gemv<kernel::gemv_op_nop><<<dim_grid, dim_block>>>(a, b, c, m, k);
		
		return cudaGetLastError();
	}
	
	inline cudaError_t gemv_sigmoid(
		half* a, float* b,
		float* c,
		uint32_t m, uint32_t k
	)
	{
		dim3 dim_grid((m + 31) / 32);
		dim3 dim_block(32);

		kernel::fast_gemv<kernel::gemv_op_sigmoid><<<dim_grid, dim_block>>>(a, b, c, m, k);

		return cudaGetLastError();
	}
	
	inline cudaError_t gemv_square_relu(
		half* a, float* b,
		float* c,
		uint32_t m, uint32_t k
	)
	{
		dim3 dim_grid((m + 31) / 32);
		dim3 dim_block(32);

		kernel::fast_gemv<kernel::gemv_op_square_relu><<<dim_grid, dim_block>>>(a, b, c, m, k);

		return cudaGetLastError();
	}

#endif

}

#endif