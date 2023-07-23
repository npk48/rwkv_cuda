#ifndef _ELEMENTWISE_CUH_
#define _ELEMENTWISE_CUH_

#include <stdint.h>
#include <type_traits>
#include <algorithm>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cuda
{
	namespace elementwise
	{
		constexpr uint32_t block_size = 256;
		constexpr uint32_t warp_size = 32;

		namespace util
		{
			constexpr uint64_t min(uint64_t a, uint64_t b) { return a < b ? a : b; }
			constexpr uint64_t max(uint64_t a, uint64_t b) { return a > b ? a : b; }

			inline uint32_t get_block_count(uint64_t n)
			{
				int dev;
				int sm_count;
				int thread_per_sm;

				cudaGetDevice(&dev);
				cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
				cudaDeviceGetAttribute(&thread_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);

				return max(1, min((n + block_size - 1) / block_size, sm_count * thread_per_sm / block_size * warp_size));
			}

			template<typename T, uint32_t pack_size>
			struct type_of_pack_t
			{
				using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
			};

			template<typename T, uint32_t pack_size>
			using pack_type_t = typename type_of_pack_t<T, pack_size>::type;

			template<typename T, uint32_t pack_size>
			union pack_t
			{
				static_assert(sizeof(pack_type_t<T, pack_size>) == sizeof(T) * pack_size, "");
				__device__ pack_t() {}
				pack_type_t<T, pack_size> storage;
				T elem[pack_size];
			};

			template<typename T, uint32_t pack_size>
			struct alignas(sizeof(T)* pack_size) packed_t
			{
				__device__ packed_t() { }
				union
				{
					T elem[pack_size];
				};
			};

			constexpr uint32_t max_pack_bytes = 128 / 8;
			constexpr uint32_t max_pack_size = 8;

			template<typename T>
			constexpr uint32_t pack_size()
			{
				return min(max_pack_bytes / sizeof(T), max_pack_size);
			}

			template<typename T, typename U, typename... Args>
			constexpr uint32_t pack_size()
			{
				return min(pack_size<T>(), pack_size<U, Args...>());
			}

			template<typename T>
			class has_apply_2_t
			{
				typedef char one;
				struct two { char x[2]; };

				template<typename C>
				static one test(decltype(&C::apply2));
				template<typename C>
				static two test(...);

			public:
				enum { value = sizeof(test<T>(0)) == sizeof(char) };
			};

			template<uint32_t pack_size, typename FunctorT, typename R, typename... IN>
			__device__ typename std::enable_if<has_apply_2_t<FunctorT>::value == true && pack_size % 2 == 0, packed_t<R, pack_size>>::type
				apply_pack(const FunctorT& functor, const packed_t<IN, pack_size>... in) 
			{
				packed_t<R, pack_size> ret;
				#pragma unroll
				for (uint32_t j = 0; j < pack_size; j += 2) { functor.apply2(ret.elem + j, (in.elem + j)...); }
				return ret;
			}

			template<uint32_t pack_size, typename FunctorT, typename R, typename... IN>
			__device__ typename std::enable_if<has_apply_2_t<FunctorT>::value == false || pack_size % 2 != 0, packed_t<R, pack_size>>::type
				apply_pack(const FunctorT& functor, const packed_t<IN, pack_size>... in)
			{
				packed_t<R, pack_size> ret;
				#pragma unroll
				for (uint32_t j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in.elem[j])...); }
				return ret;
			}

			template<size_t pack_size>
			inline bool is_aligned_for_pack() 
			{
				return true;
			}

			template<size_t pack_size, typename T, typename... Args>
			inline bool is_aligned_for_pack(const T* ptr, const Args*... others) 
			{
				return reinterpret_cast<uintptr_t>(ptr) % sizeof(pack_t<T, pack_size>) == 0 && is_aligned_for_pack<pack_size, Args...>(others...);
			}
		}

		template<uint64_t pack_size, typename FactoryT, typename R, typename... IN>
		__global__ void __launch_bounds__(block_size) apply_generic(
			FactoryT factory,
			uint64_t n_pack,
			util::packed_t<R, pack_size>* pack_r,
			const util::packed_t<IN, pack_size>*... pack_in,
			uint64_t n_tail,
			R* tail_r,
			const IN*... tail_in)
		{
			auto functor = factory();
			const uint32_t global_tid = blockIdx.x * block_size + threadIdx.x;

			for (uint64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x)
				pack_r[i] = util::apply_pack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);

			if (global_tid < n_tail)
				tail_r[global_tid] = functor((tail_in[global_tid])...);
		}

		template<uint64_t pack_size, typename FactoryT, typename R, typename... IN>
		inline void launch_kernel_generic(FactoryT factory, uint64_t n, R* r, const IN*... in)
		{
			const uint64_t n_pack = n / pack_size;
			const uint64_t tail_offset = n_pack * pack_size;
			const uint64_t n_tail = n - tail_offset;

			uint32_t block_count = util::get_block_count(n_pack);

			apply_generic<pack_size, FactoryT, R, IN...> << <block_count, block_size >> > (
				factory, n_pack, reinterpret_cast<util::packed_t<R, pack_size>*>(r),
				(reinterpret_cast<const util::packed_t<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
				(in + tail_offset)...
				);
		}

		template<typename FactoryT, typename R, typename... IN>
		inline void launch_kernel(FactoryT factory, uint64_t n, R* r, const IN*... in)
		{
			constexpr int max_pack_size = util::pack_size<R, IN...>();
			if (util::is_aligned_for_pack<max_pack_size, R, IN...>(r, in...))
				launch_kernel_generic<max_pack_size, FactoryT, R, IN...>(factory, n, r, in...);
			else
				launch_kernel_generic<1, FactoryT, R, IN...>(factory, n, r, in...);
		}

		template<typename FunctorT>
		struct simple_factory_t 
		{
			explicit simple_factory_t(FunctorT functor) : tpl(functor) {}
			__device__ FunctorT operator()() const { return tpl; }

		private:
			FunctorT tpl;
		};

		template<typename FactoryT, typename R, typename A>
		inline void unary_with_factory(FactoryT factory, int64_t n, R* r, const A* a) 
		{
			launch_kernel<FactoryT, R, A>(factory, n, r, a);
		}

		template<typename FactoryT, typename R, typename A, typename B>
		inline void binary_with_factory(FactoryT factory, int64_t n, R* r, const A* a, const B* b)
		{
			launch_kernel<FactoryT, R, A, B>(factory, n, r, a, b);
		}

		template<typename FactoryT, typename R, typename A, typename B, typename C>
		inline void ternary_with_factory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
			const C* c)
		{
			launch_kernel<FactoryT, R, A, B, C>(factory, n, r, a, b, c);
		}

		template<typename FunctorT, typename R, typename A>
		inline void unary(FunctorT functor, int64_t n, R* r, const A* a)
		{
			unary_with_factory(simple_factory_t<FunctorT>(functor), n, r, a);
		}

		template<typename FunctorT, typename R, typename A, typename B>
		inline void binary(FunctorT functor, int64_t n, R* r, const A* a, const B* b)
		{
			binary_with_factory(simple_factory_t<FunctorT>(functor), n, r, a, b);
		}

		template<typename FunctorT, typename R, typename A, typename B, typename C>
		inline void ternary(FunctorT functor, int64_t n, R* r, const A* a, const B* b, const C* c)
		{
			ternary_with_factory(simple_factory_t<FunctorT>(functor), n, r, a, b, c);
		}
	
		namespace functor
		{
			namespace cast
			{
				template<typename To, typename From, typename = void>
				struct functor_t 
				{
					__device__ To operator()(From from) const { return static_cast<To>(from); }
				};

				template<typename To>
				struct functor_t<To, half, typename std::enable_if<!std::is_same<To, half>::value>::type> 
				{
					__device__ To operator()(half from) const { return static_cast<To>(static_cast<float>(from)); }

					// __device__ void apply2(To* to, const half* from) const 
					// {
					// 	const float2 f2 = __half22float2(*reinterpret_cast<const half2*>(from));
					// 	to[0] = static_cast<To>(f2.x);
					// 	to[1] = static_cast<To>(f2.y);
					// }
				};
			}

			namespace product
			{
				template<typename DST, typename SRC, typename = void>
				struct functor_t
				{
					__device__ DST operator()(SRC a, SRC b) const { return static_cast<DST>(a * b); }
				};

				template<typename DST>
				struct functor_t<DST, half, typename std::enable_if<!std::is_same<DST, half>::value>::type>
				{
					__device__ DST operator()(half a, half b) const { return static_cast<DST>(a * b); }

					__device__ void apply2(DST* dst, const half* a, const half* b) const
					{
						half2 ha = *reinterpret_cast<const half2*>(a);
						half2 hb = *reinterpret_cast<const half2*>(b);

						const half2 h2 = ha * hb;
						const float2 f2 = __half22float2(h2);

						dst[0] = static_cast<DST>(f2.x);
						dst[1] = static_cast<DST>(f2.y);
					}
				};
			}

			namespace add
			{
				template<typename DST, typename SRC, typename = void>
				struct functor_t
				{
					__device__ DST operator()(SRC a, SRC b) const { return static_cast<DST>(a + b); }
				};

				template<typename DST>
				struct functor_t<DST, half, typename std::enable_if<!std::is_same<DST, half>::value>::type>
				{
					__device__ DST operator()(half a, half b) const { return static_cast<DST>(a * b); }

					__device__ void apply2(DST* dst, const half* a, const half* b) const
					{
						half2 ha = *reinterpret_cast<const half2*>(a);
						half2 hb = *reinterpret_cast<const half2*>(b);

						const half2 h2 = ha + hb;
						const float2 f2 = __half22float2(h2);

						dst[0] = static_cast<DST>(f2.x);
						dst[1] = static_cast<DST>(f2.y);
					}
				};
			}

			namespace minus_exp
			{
				template<typename DST, typename SRC, typename = void>
				struct functor_t
				{
					__device__ DST operator()(SRC val) const { return static_cast<DST>(-exp(static_cast<float>(val))); }
				};

				template<typename DST>
				struct functor_t<DST, half, typename std::enable_if<!std::is_same<DST, half>::value>::type>
				{
					__device__ DST operator()(half val) const { return static_cast<DST>(-exp(static_cast<half>(val))); }

					__device__ void apply2(DST* dst, const half* val) const
					{
						half2 ha = *reinterpret_cast<const half2*>(val);

						const half2 h2 = h2exp(ha);
						const float2 f2 = __half22float2(h2);

						dst[0] = -static_cast<DST>(f2.x);
						dst[1] = -static_cast<DST>(f2.y);
					}
				};
			}
		}
	}
}

#endif