#ifndef _SAFE_TENSORS_HPP_
#define _SAFE_TENSORS_HPP_

#include <stdint.h>
#include <fstream>

#include <vector>
#include <string>
#include <map>
#include <tuple>

#include "json.hpp"
#include "mmap.hpp"

struct safe_tensors_t
{
	uint8_t* data;
	uint64_t size;

	std::string dtype;
	std::vector<uint64_t> shape;
	std::pair<uint64_t, uint64_t> data_offsets;

	int file_descriptor;
	uint64_t file_size;
	uint64_t file_offset;
	uint8_t* file_mapping;

	void load()
	{
		this->file_mapping = (uint8_t*)mmap(0, this->file_size, PROT_READ, MAP_SHARED, this->file_descriptor, 0);
		this->data = this->file_mapping + this->file_offset;
	}

	void unload()
	{
		munmap(this->file_mapping, this->file_size);
	}
};

class safe_tensors_model_t
{
public:
	std::map<std::string, safe_tensors_t> layers;

public:
	void load(const char* path)
	{
		this->file_descriptor = _open(path, 0);

		_lseek(this->file_descriptor, 0, SEEK_END);

		this->file_size = _lseek(this->file_descriptor, 0, SEEK_CUR);

		this->file_mapping = (uint8_t*)mmap(0, this->file_size, PROT_READ, MAP_SHARED, this->file_descriptor, 0);

		this->parse_header();

		munmap(this->file_mapping, this->file_size);
	}

	void unload()
	{
		//munmap(this->file_mapping, this->file_size);
		_close(this->file_descriptor);
	}

private:
	int file_descriptor;
	uint8_t* file_mapping;
	uint64_t file_size;
	uint64_t header_length;

private:
	void parse_header()
	{
		this->header_length = *(uint64_t*)this->file_mapping;
		std::string header_json = "";
		header_json.resize(this->header_length);
		memcpy((void*)header_json.c_str(), &this->file_mapping[8], this->header_length);

		auto json = nlohmann::json::parse(header_json);

		for (auto& item : json.items())
		{
			if (item.key() == "__metadata__")
				continue;

			auto name = std::string(item.key());
			auto& info = item.value();

			this->layers[name] = { (uint8_t*)0, (uint64_t)0, info["dtype"], info["shape"], info["data_offsets"], this->file_descriptor, this->file_size, 0, 0 };
			this->layers[name].data = 0;// this->file_mapping + 8 + this->header_length + std::get<0>(this->layers[name].data_offsets);
			this->layers[name].size = std::get<1>(this->layers[name].data_offsets) - std::get<0>(this->layers[name].data_offsets);
			this->layers[name].file_offset = 8 + this->header_length + std::get<0>(this->layers[name].data_offsets);
		}
	}

};

#endif