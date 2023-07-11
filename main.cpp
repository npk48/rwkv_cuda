
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>


#include "cutlass/gemm/device/gemm.h"

#include "kernel/gemm.cuh"
#include "kernel/layernorm.cuh"
#include "util.hpp"

#include "safe_tensors.hpp"
#include "tokenizer.hpp"

#include "rwkv_model.hpp"

#include "simd.hpp"
#include "sampler.hpp"

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

int main()
{
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

    //std::vector<float> test_logits = { -0.1445, -0.9658,  0.0528, -0.9129 };
    //sampler::typical(test_logits, 0.6, 0.6);

    auto tokenizer = trie_tokenizer_t();
    //tokenizer.load("E:/work/rwkv_core/rwkv_vocab_v20230424.txt");
    tokenizer.load("rwkv_vocab_v20230424.txt");
    printf("tokenizer loaded\n");

    auto model_tensors = safe_tensors_model_t();
    //model_tensors.load("E:/work/rwkv_core/util/0.1b.fp16.safetensors");
    model_tensors.load("0.1b.fp16.safetensors");
    //model_tensors.load("E:/work/rwkv_cuda/3b.fp16.safetensors");
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

    for (uint32_t i = 0; i < token_ids.size(); i++)
        logits = rwkv_model.forward(rwkv_state, std::vector<uint16_t>(1, token_ids[i]));

    //auto it = std::max_element(logits.begin(), logits.end());

    uint16_t out_token_id = sampler::typical(logits, 0.6, 0.2);//it - logits.begin();

    auto out_word = tokenizer.decode({ out_token_id });

    auto gbk_str = utf8_to_gbk(text1.c_str());
    printf(gbk_str.c_str());
    gbk_str = utf8_to_gbk(out_word.c_str());
    printf(gbk_str.c_str());

    cuda::timer_t timer;

    timer.start();

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

int main2()
{
    const uint32_t m = 768;
    const uint32_t k = 768;
    const uint32_t n = 1;

    auto a = cuda::malloc<half>(m * k);
    auto b = cuda::malloc<half>(k * n);
    auto c = cuda::malloc<half>(m * n);

    cuda::fill_fp16(a, 1.f/100.f, m * k);
    cuda::fill_fp16(b, 2.f/100.f, k * n);

    //auto result = cuda::gemm<half>(a, b, c, m, k, n);
    auto result = cuda::gemv<half>(a, b, c, m, k);

    std::vector<float> h_c(m * n);

    cuda::dump_fp16(&h_c[0], c, m * n);

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