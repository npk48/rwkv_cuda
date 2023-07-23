#ifndef _THRUST_HPP_
#define _THRUST_HPP_

#include <stdint.h>

namespace thrust
{
	void argsort(uint32_t* index_out, float* arr, uint32_t size);

	void argsort_host(uint32_t* index_out, float* arr, uint32_t size);

	void cumsum(float* arr_out, float* arr_in, uint32_t size);

	void cumsum_host(float* arr_out, float* arr_in, uint32_t size);
}

#endif