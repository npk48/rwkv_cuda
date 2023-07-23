#ifndef _LIB_RWKV_H_
#define _LIB_RWKV_H_

#include <stdint.h>

#define EXPORT_API extern "C" __declspec(dllexport)

// 

EXPORT_API void rwkv_init(
	const char* vocab_file_path, 
	const char* safe_tensor_path,
	uint32_t* vocab_size_out,
	uint32_t* emb_size_out,
	uint32_t* state_size_out
);

// state api

typedef void* rwkv_state_t;

EXPORT_API rwkv_state_t rwkv_new_state();

EXPORT_API rwkv_state_t rwkv_dup_state(rwkv_state_t state);

EXPORT_API void rwkv_dump_state(rwkv_state_t state, uint8_t* state_data);

EXPORT_API void rwkv_load_state(rwkv_state_t state, uint8_t* state_data);

EXPORT_API void rwkv_free_state(rwkv_state_t state);

// tokenizer api

EXPORT_API uint32_t rwkv_str2tok(const char* str, uint16_t* token_out, uint32_t max_token_out);

EXPORT_API uint32_t rwkv_tok2str(char* str, uint16_t* token_in, uint32_t token_size);

// sampler api

EXPORT_API uint32_t rwkv_sample_typical(float* logits, uint32_t logits_size, float tau, float temp);

// forward api

EXPORT_API uint32_t rwkv_forward(
	rwkv_state_t state,
	uint16_t* token_in, uint32_t token_size,
	float* logits_out
);

#endif