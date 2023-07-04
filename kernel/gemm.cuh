#ifndef _GEMM_CUH_
#define _GEMM_CUH_

#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/gemm/kernel/gemv.h>

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>

#include "activation.cuh"

namespace cuda
{
#if __CUDA_ARCH__ < 700
	using mma_op_class = cutlass::arch::OpClassSimt;
	using sm_arch = cutlass::arch::Sm61;
#else
	using mma_op = cutlass::arch::OpClassTensorOp;
	using sm_arch = cutlass::arch::Sm75;
#endif

	// a * b = c
	// a = m * k 
	// b = k * n
	// c = m * n

	template<typename T>
	inline cudaError_t gemm(
		half* a, T* b,
		T* c,
		int m, int k, int n
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, half, T, T, float>;

		using gemm = typename cutlass::gemm::device::Gemm <
			half, cutlass::layout::ColumnMajor,
			T, cutlass::layout::ColumnMajor,
			T, cutlass::layout::ColumnMajor,
			float,
			mma_op_class,
			sm_arch,
			typename default_config::ThreadblockShape,
			typename default_config::WarpShape,
			typename default_config::InstructionShape,
			epilogue_op_nop<T>
		>;

		typename gemm gemm_op;

		typename gemm::Arguments args(
			{ m, n, k },
			{ a, m },
			{ b, k },
			{ c, m },
			{ c, m },
			{ 1, 0 }
		);

		auto status = gemm_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

	template<typename T>
	inline cudaError_t gemm_sigmoid(
		half* a, T* b,
		T* c,
		int m, int k, int n
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, half, T, T, float>;

		using gemm = typename cutlass::gemm::device::Gemm <
			half, cutlass::layout::ColumnMajor,
			T, cutlass::layout::ColumnMajor,
			T, cutlass::layout::ColumnMajor,
			float,
			mma_op_class,
			sm_arch,
			typename default_config::ThreadblockShape,
			typename default_config::WarpShape,
			typename default_config::InstructionShape,
			epilogue_op_sigmoid<T>
		>;

		typename gemm gemm_op;

		typename gemm::Arguments args(
			{ m, n, k },
			{ a, m },
			{ b, k },
			{ c, m },
			{ c, m },
			{ 1, 0 }
		);

		auto status = gemm_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

	template<typename T>
	inline cudaError_t gemm_square_relu(
		half* a, T* b,
		T* c,
		int m, int k, int n
	)
	{
		using default_config = typename cutlass::gemm::device::DefaultGemmConfiguration<mma_op_class, sm_arch, half, T, T, float>;

		using gemm = typename cutlass::gemm::device::Gemm <
			half, cutlass::layout::ColumnMajor,
			T, cutlass::layout::ColumnMajor,
			T, cutlass::layout::ColumnMajor,
			float,
			mma_op_class,
			sm_arch,
			typename default_config::ThreadblockShape,
			typename default_config::WarpShape,
			typename default_config::InstructionShape,
			epilogue_op_square_relu<T>
		>;

		typename gemm gemm_op;

		typename gemm::Arguments args(
			{ m, n, k },
			{ a, m },
			{ b, k },
			{ c, m },
			{ c, m },
			{ 1, 0 }
		);

		auto status = gemm_op(args);

		return (status != cutlass::Status::kSuccess) ? cudaErrorUnknown : cudaSuccess;
	}

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
			half, cutlass::layout::ColumnMajor,
			T,
			T,
			float,
			epilogue_op_nop<T>
			>
		>;

		typename gemv gemv_op;

		typename gemv::Arguments args(
			{ m, k },
			{1, 0},
			{ a, m },
			b, 
			c, 
			c,
			1, 1, 1
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
			half, cutlass::layout::ColumnMajor,
			T,
			T,
			float,
			epilogue_op_sigmoid<T>
			>
		>;

		typename gemv gemv_op;

		typename gemv::Arguments args(
			{ m, k },
			{ 1, 0 },
			{ a, m },
			b,
			c,
			c,
			1, 1, 1
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
			half, cutlass::layout::ColumnMajor,
			T,
			T,
			float,
			epilogue_op_square_relu<T>
			>
		>;

		typename gemv gemv_op;

		typename gemv::Arguments args(
			{ m, k },
			{ 1, 0 },
			{ a, m },
			b,
			c,
			c,
			1, 1, 1
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
			return max(0.f, x) * max(0.f, x);
		}

		const uint32_t gemv_op_nop = 0;
		const uint32_t gemv_op_sigmoid = 1;
		const uint32_t gemv_op_square_relu = 2;

	
		template<typename T, uint32_t op>
		__global__ void simple_gemv(half* a, T* b, T* c, int m, int k)
		{
			uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

			if (tid < m) 
			{
				float sum = 0;

				for (uint32_t i = 0; i < k; i++)
					sum += (float)b[i] * (float)a[i * m + tid];

				if constexpr (op == gemv_op_nop)
					c[tid] = (T)sum;
				else if constexpr (op == gemv_op_sigmoid)
					c[tid] = (T)simple_gemv_sigmoid(sum);
				else if constexpr (op == gemv_op_square_relu)
					c[tid] = (T)simple_gemv_square_relu(sum);
			}
		}
	}

	template<typename T>
	inline cudaError_t gemv(
		half* a, T* b,
		T* c,
		int m, int k
	)
	{
		dim3 dim_grid((m + 31) / 32);
		dim3 dim_block(32);

		kernel::simple_gemv<T, kernel::gemv_op_nop><<<dim_grid, dim_block>>>(a, b, c, m, k);

		return cudaGetLastError();
	}
	
	template<typename T>
	inline cudaError_t gemv_sigmoid(
		half* a, T* b,
		T* c,
		int m, int k
	)
	{
		dim3 dim_grid((m + 31) / 32);
		dim3 dim_block(32);

		kernel::simple_gemv<T, kernel::gemv_op_sigmoid> << <dim_grid, dim_block >> > (a, b, c, m, k);

		return cudaGetLastError();
	}
	
	template<typename T>
	inline cudaError_t gemv_square_relu(
		half* a, T* b,
		T* c,
		int m, int k
	)
	{
		dim3 dim_grid((m + 31) / 32);
		dim3 dim_block(32);

		kernel::simple_gemv<T, kernel::gemv_op_square_relu> << <dim_grid, dim_block >> > (a, b, c, m, k);

		return cudaGetLastError();
	}

#endif

}

#endif