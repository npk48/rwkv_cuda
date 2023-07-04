#ifndef _ACTIVATION_CUH_
#define _ACTIVATION_CUH_

#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>

namespace cutlass {
namespace epilogue {
namespace thread {

	template <typename T>
	struct SquareRelu {
		CUTLASS_HOST_DEVICE
			T operator()(T const& scalar) const {
			return (T)((scalar > T(0) ? scalar : T(0)) * (scalar > T(0) ? scalar : T(0)));
		}

		using Params = LinearCombinationGenericParams<T>;

		CUTLASS_HOST_DEVICE
			T operator()(T const& scalar, Params const& params_) const {
			return this->operator()(scalar);
		}
	};

	template <typename T, int N>
	struct SquareRelu<Array<T, N> > {
		CUTLASS_HOST_DEVICE
			Array<T, N> operator()(Array<T, N> const& value) const {
			Array<T, N> y;
			SquareRelu<T> square_relu_op;

			CUTLASS_PRAGMA_UNROLL
				for (int i = 0; i < N; ++i) {
					y[i] = square_relu_op(value[i]);
				}

			return y;
		}

		using Params = LinearCombinationGenericParams<T>;

		CUTLASS_HOST_DEVICE
			Array<T, N> operator()(Array<T, N> const& value, Params const& params_) const {
			return this->operator()(value);
		}
	};

	template <
		typename ElementOutput_,                             
		int Count,                                           
		typename ElementAccumulator_ = ElementOutput_,       
		typename ElementCompute_ = ElementOutput_,           
		ScaleType::Kind Scale = ScaleType::Default,          
		FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
	>
		using LinearCombinationSquareRelu = LinearCombinationGeneric<SquareRelu, ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Scale, Round, true>;

}
}
}

namespace cuda 
{
	template<typename T>
	using epilogue_op_sigmoid = cutlass::epilogue::thread::LinearCombinationSigmoid<
		T,
		1,
		float,
		float,
		cutlass::epilogue::thread::ScaleType::Nothing>;

	template<typename T>
	using epilogue_op_square_relu = cutlass::epilogue::thread::LinearCombinationSquareRelu<
		T,
		1,
		float,
		float,
		cutlass::epilogue::thread::ScaleType::Nothing>;

	template<typename T>
	using epilogue_op_nop = cutlass::epilogue::thread::LinearCombination<
		T,
		1,
		float,
		float,
		cutlass::epilogue::thread::ScaleType::Nothing>;
}

#endif