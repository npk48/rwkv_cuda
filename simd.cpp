#include "simd.hpp"
#include <math.h>

namespace simd
{
    namespace util
    {
#if defined(__GNUC__)
# define ALIGN16 __attribute__((aligned(32)))
# define ALIGN32 __attribute__((aligned(32)))
#elif defined(_WIN32)
# define ALIGN16 __declspec(align(16))
# define ALIGN32 __declspec(align(32))
#endif

#define _PS256_CONST(name, val) const ALIGN32 float _ps256_##name[8] = { val, val, val, val, val, val, val, val }
#define _PI32_CONST256(name, val) const ALIGN32 int _pi32_256_##name[8] = { val, val, val, val, val, val, val, val }
#define _PS256_CONST_TYPE(name, type, val) const ALIGN32 type _ps256_##name[8] = { val, val, val, val, val, val, val, val }


        _PS256_CONST(1, 1.0f);
        _PS256_CONST(0p5, 0.5f);

        _PS256_CONST(exp_hi, 88.3762626647949f);
        _PS256_CONST(exp_lo, -88.3762626647949f);

        _PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
        _PS256_CONST(cephes_exp_C1, 0.693359375);
        _PS256_CONST(cephes_exp_C2, -2.12194440e-4);

        _PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
        _PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
        _PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
        _PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
        _PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
        _PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

        _PI32_CONST256(0x7f, 0x7f);

        _PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
        _PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

        _PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
        _PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
        _PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
        _PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
        _PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
        _PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
        _PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
        _PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
        _PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
        _PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
        _PS256_CONST(cephes_log_q1, -2.12194440e-4f);
        _PS256_CONST(cephes_log_q2, 0.693359375f);

        inline __m256 exp256_ps(__m256 x)
        {
            __m256 tmp = _mm256_setzero_ps(), fx;
            __m256i imm0;
            __m256 one = *(__m256*)_ps256_1;

            x = _mm256_min_ps(x, *(__m256*)_ps256_exp_hi);
            x = _mm256_max_ps(x, *(__m256*)_ps256_exp_lo);

            /* express exp(x) as exp(g + n*log(2)) */
            fx = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_LOG2EF);
            fx = _mm256_add_ps(fx, *(__m256*)_ps256_0p5);

            /* how to perform a floorf with SSE: just below */
            //imm0 = _mm256_cvttps_epi32(fx);
            //tmp  = _mm256_cvtepi32_ps(imm0);

            tmp = _mm256_floor_ps(fx);

            /* if greater, substract 1 */
            //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
            __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
            mask = _mm256_and_ps(mask, one);
            fx = _mm256_sub_ps(tmp, mask);

            tmp = _mm256_mul_ps(fx, *(__m256*)_ps256_cephes_exp_C1);
            __m256 z = _mm256_mul_ps(fx, *(__m256*)_ps256_cephes_exp_C2);
            x = _mm256_sub_ps(x, tmp);
            x = _mm256_sub_ps(x, z);

            z = _mm256_mul_ps(x, x);

            __m256 y = *(__m256*)_ps256_cephes_exp_p0;
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p1);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p2);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p3);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p4);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p5);
            y = _mm256_mul_ps(y, z);
            y = _mm256_add_ps(y, x);
            y = _mm256_add_ps(y, one);

            /* build 2^n */
            imm0 = _mm256_cvttps_epi32(fx);
            // another two AVX2 instructions
            imm0 = _mm256_add_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
            imm0 = _mm256_slli_epi32(imm0, 23);
            __m256 pow2n = _mm256_castsi256_ps(imm0);
            y = _mm256_mul_ps(y, pow2n);
            return y;
        }

        inline float hsum_ps_sse3(__m128 v)
        {
            __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        inline float hsum256_ps_avx(__m256 v)
        {
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
            vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
            return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
            // (no wasted instructions, and all of them are the 4B minimum)
        }

        inline __m256 _mm256_abs_ps(__m256 x) 
        {
            static const ALIGN16 int _ps256_inv_sign_mask[8] = { ~0x80000000, ~0x80000000, ~0x80000000, ~0x80000000 ,~0x80000000, ~0x80000000, ~0x80000000, ~0x80000000 };
            return _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
        }

        inline __m256 scan_avx(__m256 x) 
        {
            __m256 t0, t1;

            t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
            t1 = _mm256_permute2f128_ps(t0, t0, 41);
            x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));

            t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
            t1 = _mm256_permute2f128_ps(t0, t0, 41);
            x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));

            x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 41));
            return x;
        }

        inline __m256  _mm256_cmplt_ps(__m256 a, __m256 b) { return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_LT_OQ), _mm256_set1_ps(1.0)); }
        inline __m256  _mm256_cmpgt_ps(__m256 a, __m256 b) { return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_GT_OQ), _mm256_set1_ps(1.0)); }
        inline __m256  _mm256_cmpeq_ps(__m256 a, __m256 b) { return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_EQ_OQ), _mm256_set1_ps(1.0)); }
        inline __m256  _mm256_cmpneq_ps(__m256 a, __m256 b) { return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_NEQ_OQ), _mm256_set1_ps(1.0)); }
        inline __m256  _mm256_cmple_ps(__m256 a, __m256 b) { return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_LE_OQ), _mm256_set1_ps(1.0)); }

        inline __m256i _wrap_mm256_slli_epi32(__m256i x, int  y) { return _mm256_slli_epi32(x, y); }
        inline __m256i _wrap_mm256_srli_epi32(__m256i x, int  y) { return _mm256_srli_epi32(x, y); }
        inline __m256i _wrap_mm256_sub_epi32(__m256i x, __m256i y) { return _mm256_sub_epi32(x, y); }
        inline __m256i _wrap_mm256_add_epi32(__m256i x, __m256i y) { return _mm256_add_epi32(x, y); }

        inline __m256 log256_ps(__m256 x) 
        {
            __m256i imm0;
            __m256 one = *(__m256*)_ps256_1;

            //__m256 invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
            __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

            x = _mm256_max_ps(x, *(__m256*)_ps256_min_norm_pos);  /* cut off denormalized stuff */

            // can be done with AVX2
            imm0 = _wrap_mm256_srli_epi32(_mm256_castps_si256(x), 23);

            /* keep only the fractional part */
            x = _mm256_and_ps(x, *(__m256*)_ps256_inv_mant_mask);
            x = _mm256_or_ps(x, *(__m256*)_ps256_0p5);

            // this is again another AVX2 instruction
            imm0 = _wrap_mm256_sub_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
            __m256 e = _mm256_cvtepi32_ps(imm0);

            e = _mm256_add_ps(e, one);

            /* part2:
               if( x < SQRTHF ) {
                 e -= 1;
                 x = x + x - 1.0;
               } else { x = x - 1.0; }
            */
            //__m256 mask = _mm256_cmplt_ps(x, *(__m256*)_ps256_cephes_SQRTHF);
            __m256 mask = _mm256_cmp_ps(x, *(__m256*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
            __m256 tmp = _mm256_and_ps(x, mask);
            x = _mm256_sub_ps(x, one);
            e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
            x = _mm256_add_ps(x, tmp);

            __m256 z = _mm256_mul_ps(x, x);

            __m256 y = *(__m256*)_ps256_cephes_log_p0;
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p1);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p2);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p3);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p4);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p5);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p6);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p7);
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p8);
            y = _mm256_mul_ps(y, x);

            y = _mm256_mul_ps(y, z);

            tmp = _mm256_mul_ps(e, *(__m256*)_ps256_cephes_log_q1);
            y = _mm256_add_ps(y, tmp);


            tmp = _mm256_mul_ps(z, *(__m256*)_ps256_0p5);
            y = _mm256_sub_ps(y, tmp);

            tmp = _mm256_mul_ps(e, *(__m256*)_ps256_cephes_log_q2);
            x = _mm256_add_ps(x, y);
            x = _mm256_add_ps(x, tmp);
            x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
            return x;
        }
    }

    void softmax(std::vector<float>& vec)
    {
        float sum = 0.0f;
        float max_val = *std::max_element(vec.begin(), vec.end());

        __m256 max_v = _mm256_set1_ps(max_val);
        __m256 sum_v = _mm256_set1_ps(0.0f);

        uint32_t i = 0;
        if (vec.size() >= 8)
        {
            for (i = 0; i <= (vec.size() - 8); i += 8)
            {
                __m256 ymm0 = _mm256_load_ps(&vec[i]);
                ymm0 = _mm256_sub_ps(ymm0, max_v);
                ymm0 = util::exp256_ps(ymm0);
                _mm256_store_ps(&vec[i], ymm0);
                sum_v = _mm256_add_ps(sum_v, ymm0);
            }
            sum += util::hsum256_ps_avx(sum_v);
        }
        

        for (; i < vec.size(); i++)
        {
            vec[i] = expf(vec[i] - max_val);
            sum += vec[i];
        }

        i = 0;
        if (vec.size() >= 8)
        {
            __m256 sum4 = _mm256_set1_ps(sum);
            for (i = 0; i <= (vec.size() - 8); i += 8)
            {
                __m256 ymm0 = _mm256_load_ps(&vec[i]);
                _mm256_store_ps(&vec[i], _mm256_div_ps(ymm0, sum4));
            }
        }

        for (; i < vec.size(); i++)
            vec[i] /= sum;
    }

    void sigmoid(std::vector<float>& vec)
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 zero = _mm256_set1_ps(0.0f);

        __m256 ymm0, ymm1, ymm2, ymm3;

        uint32_t i = 0;
        if (vec.size() >= 8)
        {
            for (i = 0; i <= (vec.size() - 16); i += 16)
            {
                ymm0 = _mm256_load_ps(&vec[i]);
                ymm1 = _mm256_load_ps(&vec[i + 8]);
                ymm0 = _mm256_sub_ps(zero, ymm0);
                ymm1 = _mm256_sub_ps(zero, ymm1);
                ymm2 = _mm256_add_ps(one, util::exp256_ps(ymm0));
                ymm3 = _mm256_add_ps(one, util::exp256_ps(ymm1));
                ymm2 = _mm256_div_ps(one, ymm2);
                ymm3 = _mm256_div_ps(one, ymm3);
                _mm256_store_ps(&vec[i], ymm2);
                _mm256_store_ps(&vec[i + 8], ymm3);
            }
        }
        for (; i < vec.size(); i++)
            vec[i] = 1.0f / (1.0f + expf(-vec[i]));
    }

    void log(std::vector<float>& vec)
    {
        __m256 ymm0, ymm1;

        uint32_t i = 0;
        if (vec.size() >= 8)
        {
            for (i = 0; i <= (vec.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&vec[i]);
                ymm1 = _mm256_log_ps(ymm0);
                _mm256_store_ps(&vec[i], ymm1);
            }
        }

        for (; i < vec.size(); i++)
            vec[i] = logf(vec[i]);
    }

    float sum_all(std::vector<float>& vec)
    {
        float sum = 0.f;

        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (vec.size() >= 8)
        {
            for (i = 0; i <= (vec.size() - 16); i += 16)
            {
                ymm0 = _mm256_load_ps(&vec[i]);
                ymm1 = _mm256_load_ps(&vec[i + 8]);
                ymm2 = _mm256_add_ps(ymm0, ymm1);
                sum += util::hsum256_ps_avx(ymm2);
            }
        }


        for (; i < vec.size(); i++)
            sum += vec[i];

        return sum;
    }

    void add(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = _mm256_add_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] + b[i];
    }

    void add(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = _mm256_add_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] + b;
    }

    void sub(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = _mm256_sub_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] - b[i];
    }

    void sub(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = _mm256_sub_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] - b;
    }

    void abs(std::vector<float>& vec)
    {
        __m256 ymm0, ymm1;

        uint32_t i = 0;
        if (vec.size() >= 8)
        {
            for (i = 0; i <= (vec.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&vec[i]);
                ymm1 = util::_mm256_abs_ps(ymm0);
                _mm256_store_ps(&vec[i], ymm1);
            }
        }

        for (; i < vec.size(); i++)
            vec[i] = fabs(vec[i]);
    }

    void cumsum(std::vector<float>& dst, std::vector<float>& src)
    {
        __m256 offset = _mm256_setzero_ps();

        uint32_t i = 0;
        float sum = 0;
        if (dst.size() >= 8)
        {
            for (i = 0; i <= (dst.size() - 8); i += 8)
            {
                __m256 x = _mm256_loadu_ps(&src[i]);
                __m256 out = util::scan_avx(x);
                out = _mm256_add_ps(out, offset);
                _mm256_storeu_ps(&dst[i], out);
                __m256 t0 = _mm256_permute2f128_ps(out, out, 0x11);
                offset = _mm256_permute_ps(t0, 0xff);
            }
            sum = dst[i - 1];
        }

        for (; i < dst.size(); i++) 
        {
            sum += src[i];
            dst[i] = sum;
        }
    }

    void cmp_lt(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = util::_mm256_cmplt_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] < b[i] ? 1.f : 0.f;
    }
    
    void cmp_lt(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;

        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = util::_mm256_cmplt_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] < b ? 1.f : 0.f;
    }

    void cmp_gt(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = util::_mm256_cmpgt_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] > b[i] ? 1.f : 0.f;
    }

    void cmp_gt(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = util::_mm256_cmpgt_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] > b ? 1.f : 0.f;
    }

    void cmp_eq(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = util::_mm256_cmpeq_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] == b[i] ? 1.f : 0.f;
    }

    void cmp_eq(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = util::_mm256_cmpeq_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] == b ? 1.f : 0.f;
    }

    void cmp_neq(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = util::_mm256_cmpneq_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] != b[i] ? 1.f : 0.f;
    }

    void cmp_neq(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = util::_mm256_cmpneq_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] != b ? 1.f : 0.f;
    }

    void cmp_le(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = util::_mm256_cmple_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] <= b[i] ? 1.f : 0.f;
    }

    void cmp_le(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;

        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = util::_mm256_cmple_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] <= b ? 1.f : 0.f;
    }

    void mul(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = _mm256_mul_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] * b[i];
    }

    void mul(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = _mm256_mul_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] * b;
    }

    void div(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = _mm256_div_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] / b[i];
    }

    void div(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                ymm2 = _mm256_div_ps(ymm0, ymm1);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = a[i] / b;
    }

    void pow(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        __m256 zero = _mm256_set1_ps(0.f);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                auto zero_fix = util::_mm256_cmpneq_ps(ymm0, zero);
                ymm1 = _mm256_load_ps(&b[i]);
                ymm2 = util::exp256_ps(_mm256_mul_ps(ymm1, util::log256_ps(ymm0)));
                ymm2 = _mm256_mul_ps(ymm2, zero_fix);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = powf(a[i], b[i]);
    }

    void pow(std::vector<float>& a, float b, std::vector<float>& c)
    {
        __m256 ymm0, ymm1, ymm2;

        ymm1 = _mm256_set1_ps(b);
        __m256 zero = _mm256_set1_ps(0.f);

        uint32_t i = 0;
        if (c.size() >= 8)
        {
            for (i = 0; i <= (c.size() - 8); i += 8)
            {
                ymm0 = _mm256_load_ps(&a[i]);
                auto zero_fix = util::_mm256_cmpneq_ps(ymm0, zero);
                ymm2 = util::exp256_ps(_mm256_mul_ps(ymm1, util::log256_ps(ymm0)));
                ymm2 = _mm256_mul_ps(ymm2, zero_fix);
                _mm256_store_ps(&c[i], ymm2);
            }
        }

        for (; i < c.size(); i++)
            c[i] = powf(a[i], b);
    }

    uint32_t multinomial_one(std::vector<float>& prob)
    {
        static std::random_device random_device;
        static std::mt19937 gen(random_device());
        static std::exponential_distribution<float> exp_d(1);

        std::vector<float> q(prob.size());

        auto sample = []() {return exp_d(gen);};

        std::generate(q.begin(), q.end(), sample);

        div(prob, q, q);

        auto max_el = std::max_element(q.begin(), q.end());

        return max_el - q.begin();
    }
}