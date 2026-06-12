// Copyright 2016 The Draco Authors.
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
//
#include "draco/core/cycle_timer.h"

namespace draco {
void DracoTimer::Start() {
#ifdef _WIN32
  QueryPerformanceCounter(&tv_start_);
#else
  gettimeofday(&tv_start_, nullptr);
#endif
}

void DracoTimer::Stop() {
#ifdef _WIN32
  QueryPerformanceCounter(&tv_end_);
#else
  gettimeofday(&tv_end_, nullptr);
#endif
}

int64_t DracoTimer::GetInMs() {
#ifdef _WIN32
  LARGE_INTEGER elapsed = {0};
  elapsed.QuadPart = tv_end_.QuadPart - tv_start_.QuadPart;

  LARGE_INTEGER frequency = {0};
  QueryPerformanceFrequency(&frequency);
  return elapsed.QuadPart * 1000 / frequency.QuadPart;
#else
  const int64_t seconds = (tv_end_.tv_sec - tv_start_.tv_sec) * 1000;
  const int64_t milliseconds = (tv_end_.tv_usec - tv_start_.tv_usec) / 1000;
  return seconds + milliseconds;
#endif
}

}  // namespace draco
