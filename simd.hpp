#ifndef _SIMD_HPP_
#define _SIMD_HPP_

#include <stdint.h>
#include <immintrin.h>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>

namespace simd
{
    void softmax(std::vector<float>& vec);

    void sigmoid(std::vector<float>& vec);

    void log(std::vector<float>& vec);

    float sum_all(std::vector<float>& vec);

    void add(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void add(std::vector<float>& a, float b, std::vector<float>& c);

    void sub(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void sub(std::vector<float>& a, float b, std::vector<float>& c);

    void abs(std::vector<float>& vec);

    inline void argsort(std::vector<uint32_t>& sorted, std::vector<float>& src)
    {
        auto first = src.begin();
        std::less<std::iterator_traits<decltype(first)>::value_type> comparitor;
        std::iota(sorted.begin(), sorted.end(), 0);
        auto ind_comp = [&](const auto& li, const auto& ri) {return comparitor(*std::next(first, li), *std::next(first, ri)); };
        std::sort(sorted.begin(), sorted.end(), ind_comp);
    }

    void cumsum(std::vector<float>& dst, std::vector<float>& src);

    void cmp_lt(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void cmp_gt(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void cmp_eq(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void cmp_neq(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void cmp_le(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);

    void cmp_lt(std::vector<float>& a, float b, std::vector<float>& c);
    void cmp_gt(std::vector<float>& a, float b, std::vector<float>& c);
    void cmp_eq(std::vector<float>& a, float b, std::vector<float>& c);
    void cmp_neq(std::vector<float>& a, float b, std::vector<float>& c);
    void cmp_le(std::vector<float>& a, float b, std::vector<float>& c);

    void mul(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void mul(std::vector<float>& a, float b, std::vector<float>& c);

    void div(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void div(std::vector<float>& a, float b, std::vector<float>& c);

    void pow(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
    void pow(std::vector<float>& a, float b, std::vector<float>& c);

    // torch.multinomial(probs, num_samples=1)

    uint32_t multinomial_one(std::vector<float>& prob);
}

#endif