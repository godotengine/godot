/*
 *  Copyright 2019-2020 NVIDIA Corporation
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

#include <cstdio>
#include <set>
#include <string>

int main(int argc, char** argv) {
  std::set<std::string> archs;
  int devices;
  if ((cudaGetDeviceCount(&devices) == cudaSuccess) && (devices > 0)) {
    for (int dev = 0; dev < devices; ++dev) {
      char buff[32];
      cudaDeviceProp prop;
      if(cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;
      sprintf(buff, "%d%d", prop.major, prop.minor);
      archs.insert(buff);
    }
  }
  if (archs.empty()) {
    printf("NONE");
  } else {
    bool first = true;
    for(const auto& arch : archs) {
      printf(first ? "%s" : ";%s", arch.c_str());
      first = false;
    }
  }
  printf("\n");
}
