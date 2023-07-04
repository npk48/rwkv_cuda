#ifndef _TOKENIZER_HPP_
#define _TOKENIZER_HPP_

#include <stdint.h>
#include <vector>
#include <string>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

struct trie_t 
{
     trie_t(uint8_t ch = '\0')
         : ch(ch), to(256, nullptr), value(-1)
     {
 
     }
 
     trie_t* add(std::vector<uint8_t> key, int32_t idx = 0, int32_t val = -1)
     {
         if (idx == key.size()) 
         {
             value = val;
             return this;
         }
 
         uint8_t ch = key[idx];

         if (ch >= 256)
             ch = 0;

         if (to[ch] == nullptr)
             to[ch] = new trie_t(ch);
 
         return to[ch]->add(key, idx + 1, val);
     }
 
     std::tuple<uint32_t, int32_t> find_longest(std::vector<uint8_t> key, uint32_t idx = 0) 
     {
        trie_t* u = this;

        uint8_t ch = key[idx];

        std::tuple<uint32_t, int32_t> ret;

        if (u->to[ch] == nullptr)
            return { idx+1, 65535 };

        while (u->to[ch] != nullptr) 
        {
            u = u->to[ch];
            idx++;

            if (u->value != -1 ) 
                ret = { idx, u->value };

            if (idx == key.size())
                break;

            ch = key[idx];
        }

        return ret;
     }
 
     uint8_t ch;
     std::vector<trie_t*> to;
     int32_t value;
};



class trie_tokenizer_t 
{
public:
    void load(const char* file_name) 
    {
        std::ifstream f(file_name, std::ios::in | std::ios::binary);
        std::string line;

        while (std::getline(f, line)) {

            auto split_a = line.find(' ');
            auto split_b = line.rfind(' ');

            int32_t idx = std::stoi(line.substr(0, split_a));
            std::string token = line.substr(split_a + 1, split_b - split_a - 1);
            int32_t len = std::stoi(line.substr(split_b + 1));

            if (token[0] == '\'')
                token = token.substr(1, token.size() - 2);
            else if (token[0] == '\"')
                token = token.substr(1, token.size() - 2);
            else if (token[0] == 'b')
                token = token.substr(1, token.size() - 2);

            std::vector<uint8_t> encoded_x = fix_utf8_escape(token);

            idx2token[idx] = encoded_x;
        }
        f.close();

        for (const auto& kv : idx2token)
            token2idx[kv.second] = kv.first;

        for (const auto& kv : token2idx)
             trie_root.add(kv.first, 0, kv.second);
    }

    std::vector<uint16_t> encode_byte(std::vector<uint8_t> src)
    {
        uint32_t idx = 0;
        std::vector<uint16_t> tokens;

        while (idx < src.size()) 
        {
            auto result = trie_root.find_longest(src, idx);
            auto new_idx = std::get<0>(result);
            auto value = std::get<1>(result);
            idx = new_idx;
            tokens.push_back((uint16_t)value);
        }
        return tokens;
    }

	std::vector<uint16_t> encode(std::string text)
	{
        return encode_byte(std::vector<uint8_t>(text.begin(), text.end()));
	}

    std::string decode_byte(uint16_t token_id)
    {
        return std::string(this->idx2token[token_id].begin(), this->idx2token[token_id].end());
    }

	std::string decode(std::vector<uint16_t> token_ids)
	{
        std::string result = "";

        for (auto& token_id : token_ids)
            result += decode_byte(token_id);

        return result;
	}

	std::vector<std::string> tokenize(std::string text)
	{
        std::vector<std::string> result;

        auto token_ids = encode(text);
        for (auto& token_id : token_ids)
            result.push_back(decode_byte(token_id));

        return result;
	}

private:
    static inline std::vector<uint8_t> fix_utf8_escape(std::string token)
    {
        /*
            sequence need to be unescaped
            [
                "\\symbol", ["\\", "symbol"]
                "\\",       ["\\"]
                "\\t",      ["\\", "t"]
                "\\n",      ["\\", "n"]
                "\\r",      ["\\", "r"]
                "\\x12",    ["\\", "x", "1", "2"]
                "\\u1234",  ["\\", "u", "1", "2", "3", "4"]
            ]
        */

        std::vector<uint8_t> result;
        while (token.size())
        {
            char c = token[0];
            if (c == '\\') {
                if (token[1] == 't') {
                    result.push_back('\t');
                    token = token.substr(2);
                }
                else if (token[1] == 'n') {
                    result.push_back('\n');
                    token = token.substr(2);
                }
                else if (token[1] == 'r') {
                    result.push_back('\r');
                    token = token.substr(2);
                }
                else if (token[1] == 'x') {
                    result.push_back(std::stoi(token.substr(2, 2), 0, 16));
                    token = token.substr(4);
                }
                else if (token[1] == 'u') {
                    result.push_back(std::stoi(token.substr(2, 4), 0, 16));
                    token = token.substr(6);
                }
                else
                {
                    result.push_back(token[1]);
                    token = token.substr(2);
                }
            }
            else {
                result.push_back(c);
                token = token.substr(1);
            }
        }

        return result;
    }

private:
    std::map<int32_t, std::vector<uint8_t>> idx2token;
    std::map<std::vector<uint8_t>, int32_t> token2idx;

    trie_t trie_root;
};


#endif