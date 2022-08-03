/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// A simple timer class

#ifdef __CUDACC__

// use CUDA's high-resolution timers when possible
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <string>

void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}

struct timer
{
  cudaEvent_t start;
  cudaEvent_t end;

  timer(void)
  {
    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&end));
    restart();
  }

  ~timer(void)
  {
    cuda_safe_call(cudaEventDestroy(start));
    cuda_safe_call(cudaEventDestroy(end));
  }

  void restart(void)
  {
    cuda_safe_call(cudaEventRecord(start, 0));
  }

  double elapsed(void)
  {
    cuda_safe_call(cudaEventRecord(end, 0));
    cuda_safe_call(cudaEventSynchronize(end));

    float ms_elapsed;
    cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
    return ms_elapsed / 1e3;
  }

  double epsilon(void)
  {
    return 0.5e-6;
  }
};

#else

// fallback to clock()
#include <ctime>

struct timer
{
  clock_t start;
  clock_t end;

  timer(void)
  {
    restart();
  }

  ~timer(void)
  {
  }

  void restart(void)
  {
    start = clock();
  }

  double elapsed(void)
  {
    end = clock();

    return static_cast<double>(end - start) / static_cast<double>(CLOCKS_PER_SEC);
  }

  double epsilon(void)
  {
    return 1.0 / static_cast<double>(CLOCKS_PER_SEC);
  }
};

#endif

