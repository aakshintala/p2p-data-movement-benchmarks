/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_UTILITY_H_
#define __CUDA_UTILITY_H_


#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
// uncomment to disable assert()
#include <cassert>

/**
 * Asserts if not CUDA_SUCCESS
 * @ingroup util
 */
#define CUDACER(x) cudaCheckError((x), #x, __FILE__, __LINE__)
#define CUDA_ASSERT(x) assert(CUDACER(x) == cudaSuccess)

/*
 * define this if you want all cuda calls to be printed
 */
//#define CUDA_TRACE



/**
 * cudaCheckError
 * @ingroup util
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
	if( retval == cudaSuccess)
		return cudaSuccess;
#endif

	std::cout << "[cuda] " << txt <<std::endl;

	if( retval != cudaSuccess )
	{
		std::cout << "[cuda] error" << retval << " " << cudaGetErrorString(retval) <<std::endl;
		std::cout << "[cuda] " << file <<" " <<line <<std::endl;
	}

	return retval;
}

#endif
