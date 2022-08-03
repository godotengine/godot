// Copyright 2022 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef MANIFOLD_USE_CUDA
#include <cuda_runtime.h>

namespace manifold {
int CUDA_ENABLED = -1;
void check_cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  CUDA_ENABLED = device_count != 0;
}
}  // namespace manifold
#else
namespace manifold {
void check_cuda_available() {}
}  // namespace manifold
#endif
