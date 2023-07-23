import process from 'process';
import koffi from 'koffi';

const lib = koffi.load('./rwkv_cuda.dll');

const rwkv_init = lib.func('void rwkv_init(const char* vocab_path, const char* tensor_path, _Out_ uint32_t* vocab_size, _Out_ uint32_t* emb_size, _Out_ uint32_t* state_size)');

const rwkv_new_state = lib.func('void* rwkv_new_state()');

const rwkv_str2tok = lib.func('uint32_t rwkv_str2tok(const char* str, _Out_ uint16_t* token_out, uint32_t max_token_out)');
const rwkv_tok2str = lib.func('uint32_t rwkv_tok2str(_Out_ uint8_t* str, uint16_t* token_in, uint32_t token_size)');

const rwkv_sample_typical = lib.func('uint32_t rwkv_sample_typical(float* logits, uint32_t logits_size, float tau, float temp)');

const rwkv_forward = lib.func('uint32_t rwkv_forward(void* state, uint16_t* token_in, uint32_t token_size, _Out_ float* logits_out)')

//const rwkv_init = lib.func('rwkv_init', 'void', ['str', 'str', ]);

const vocab_file = "c:/work/rwkv_cuda/model/rwkv_vocab_v20230424.txt";
const tensor_file = "c:/work/rwkv_cuda/model/3b.fp16.safetensors";

let vocab_size = [0];
let emb_size = [0];
let state_size = [0];

rwkv_init(vocab_file, tensor_file, vocab_size, emb_size, state_size);

let state = rwkv_new_state();

let prompt = "嘉祥:香草快去准备开店吧\n\n香草:";
let toks = [... new Array(32).keys()];
let tok_size = rwkv_str2tok(prompt, toks, toks.length);

let logits = [... new Array(vocab_size[0]).keys()];
let logits_size = rwkv_forward(state, toks, tok_size, logits);

let tok_id = rwkv_sample_typical(logits, logits_size, 0.6, 0.2);

let str_out = new Array(16).fill(0);

let str_size = rwkv_tok2str(str_out, [tok_id], 1);

process.stdout.write(prompt);

const decoder = new TextDecoder('UTF-8');

const toString = (bytes) => {
    const array = new Uint8Array(bytes);
  	return decoder.decode(array);
};

process.stdout.write(toString(str_out.slice(0, str_size)));

for(let i=0; i<100; i++)
{
    logits_size = rwkv_forward(state, [tok_id], 1, logits);
    tok_id = rwkv_sample_typical(logits, logits_size, 0.6, 0.2);
    str_size = rwkv_tok2str(str_out, [tok_id], 1);
    
    process.stdout.write(toString(str_out.slice(0, str_size)));
}
