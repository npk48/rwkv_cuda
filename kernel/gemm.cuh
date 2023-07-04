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

}

#endif