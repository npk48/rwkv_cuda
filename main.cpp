
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#include "kernel/gemm.cuh"
#include "kernel/layernorm.cuh"
#include "util.hpp"

#include "safe_tensors.hpp"
#include "tokenizer.hpp"

#include "rwkv_model.hpp"

#include "simd.hpp"
#include "sampler.hpp"

#ifdef _WIN32
#include <windows.h>
static inline std::string gbk_to_utf8(std::string src_str)
{
    int len = MultiByteToWideChar(CP_ACP, 0, src_str.c_str(), -1, NULL, 0);
    wchar_t* wstr = new wchar_t[len + 1];
    memset(wstr, 0, len + 1);
    MultiByteToWideChar(CP_ACP, 0, src_str.c_str(), -1, wstr, len);
    len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    char* str = new char[len + 1];
    memset(str, 0, len + 1);
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
    std::string strTemp = str;
    if (wstr) delete[] wstr;
    if (str) delete[] str;
    return strTemp;
}

static inline std::string utf8_to_gbk(std::string src_str)
{
    int len = MultiByteToWideChar(CP_UTF8, 0, src_str.c_str(), -1, NULL, 0);
    wchar_t* wstr = new wchar_t[len + 1];
    memset(wstr, 0, len + 1);
    MultiByteToWideChar(CP_UTF8, 0, src_str.c_str(), -1, wstr, len);
    len = WideCharToMultiByte(CP_ACP, 0, wstr, -1, NULL, 0, NULL, NULL);
    char* str = new char[len + 1];
    memset(str, 0, len + 1);
    WideCharToMultiByte(CP_ACP, 0, wstr, -1, str, len, NULL, NULL);
    std::string strTemp = str;
    if (wstr) delete[] wstr;
    if (str) delete[] str;
    return strTemp;
}

#else

static inline std::string gbk_to_utf8(std::string src_str) { return src_str; }
static inline std::string utf8_to_gbk(std::string src_str) { return src_str; }

#endif

#include "thrust.hpp"

int main()
{
    {
        cudaDeviceReset();
        cudaSetDevice(0);


        //std::vector<float> test_prob(65536, 123.f);
        //cuda::softmax_(&test_prob[0], &test_prob[0], 1, 65536);

        // m = rows of output c
        // n = cols of output c
        // k = inner gemm dim
        
        // m * k , 3 * 4
        float a[] = {
           1, 2, 3, 4,
           5, 6, 7, 8,
           9, 10, 11, 12
        };

        // k * n , 4 * 4
        float b[] = {
            10, 20, 30, 40,
            11, 21, 31, 41,
            12, 22, 32, 42,
            13, 23, 33, 43
        };

        // m * n , 3 * 4
        float c[] = {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        };

        auto da = cuda::malloc<float>(3 * 4);
        auto db = cuda::malloc<half>(4 * 4);
        auto dc = cuda::malloc<float>(3 * 4);

        {
            cuda::load_fp32(da, a, 3 * 4);

            auto db_fp32 = cuda::malloc<float>(4 * 4);
            cuda::load_fp32(db_fp32, b, 4 * 4);
            cuda::convert<float, half>(db_fp32, db, 4 * 4);
            cuda::free(db_fp32);

            //cuda::load_fp32(db, b, 4 * 4);
            cuda::load_fp32(dc, c, 3 * 4);
        }

        cuda::gemm<float, half, float>(da, db, dc, 3, 4, 4);
        cuda::dump_fp32(c, dc, 3 * 4);

        /*
            300  310
            700  726
            1100 1142
        */

        int asdf = 0;

    }
    {
        int nDevices;
        printf("list cuda devices:\n");
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            printf("  Memory Clock Rate (KHz): %d\n",
                prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        }
    }
    auto err = cudaDeviceReset();
    err = cudaSetDevice(0);

    auto test_tensor = cuda::create_tensor<half>({ 256, 1, 1 });

    cuda::fill_fp16((half*)test_tensor.data, 1.234, 256);

    std::vector<float> test_htensor(256);

    cuda::dump_fp16(&test_htensor[0], (half*)test_tensor.data, 256);

    //std::vector<float> test_logits = { -0.1445, -0.9658,  0.0528, -0.9129 };
    //sampler::typical(test_logits, 0.6, 0.6);

    auto tokenizer = trie_tokenizer_t();
    tokenizer.load("c:/work/rwkv_cuda/model/rwkv_vocab_v20230424.txt");
    //tokenizer.load("rwkv_vocab_v20230424.txt");
    printf("tokenizer loaded\n");

    auto model_tensors = safe_tensors_model_t();
    //model_tensors.load("E:/work/rwkv_core/util/0.1b.fp16.safetensors");
    //model_tensors.load("0.1b.fp16.safetensors");
    model_tensors.load("c:/work/rwkv_cuda/model/3b.fp16.safetensors");
    printf("model parsed\n");

    auto rwkv_model = rwkv_model_cuda_t();
    printf("loading model\n");
    rwkv_model.load(model_tensors);
    printf("model loaded\n");

    auto rwkv_state = rwkv_model.create_state();

    std::string text1 = "嘉祥:香草快去准备开店吧\n\n香草:";
    //std::string text2 = "it's";
    std::string text2 = "it's a nice weather";
    std::string text3 = "hello world";

    auto token_ids = tokenizer.encode(text1);

    std::vector<float> logits;
    //logits = rwkv_model.forward(rwkv_state, token_ids);

    cuda::timer_t timer;

    timer.start();

    //for (uint32_t i = 0; i < token_ids.size(); i++)
    //    logits = rwkv_model.forward(rwkv_state, std::vector<uint16_t>(1, token_ids[i]));
    logits = rwkv_model.forward(rwkv_state, token_ids);

    //auto it = std::max_element(logits.begin(), logits.end());

    uint16_t out_token_id = sampler::typical(logits, 0.6, 0.2);//it - logits.begin();

    auto out_word = tokenizer.decode({ out_token_id });

    auto gbk_str = utf8_to_gbk(text1.c_str());
    printf(gbk_str.c_str());
    gbk_str = utf8_to_gbk(out_word.c_str());
    printf(gbk_str.c_str());

    for (uint32_t i = 0; i < 100; i++)
    {
        token_ids = out_word.size() == 0 ? std::vector<uint16_t>({ 0 }) : tokenizer.encode(out_word);

        logits = rwkv_model.forward(rwkv_state, token_ids);
        //simd::softmax(logits);

        //it = std::max_element(logits.begin(), logits.end());

        out_token_id = sampler::typical(logits, 0.6, 0.2); //it - logits.begin();

        out_word = tokenizer.decode({ out_token_id });

        gbk_str = utf8_to_gbk(out_word.c_str());
        printf(gbk_str.c_str());
    }

    timer.stop();
    printf("Time to generate:  %3.1f ms \n", timer.elapsed_ms());

    return 0;
}

/*
time_mix:
    1. xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)

    2. kx = xx * k_mix + sx * (1 - k_mix)
       vx = xx * v_mix + sx * (1 - v_mix)
       rx = xx * r_mix + sx * (1 - r_mix)

    3. r = torch.sigmoid(rx @ rw)

    4. k = kx @ kw

    5. v = vx @ vw

    7. wkv, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)

    8. out = (r * wkv) @ ow

    9. x = x + out

*/

/*

channel_mix:
    1. xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)

    2.  kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

    3. r = torch.sigmoid(rx @ rw)

    4. vx = torch.square(torch.relu(kx @ kw))

    5. out = r * (vx @ vw)

    6. x = out + x
*/