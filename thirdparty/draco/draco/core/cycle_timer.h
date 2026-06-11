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
#ifndef DRACO_CORE_CYCLE_TIMER_H_
#define DRACO_CORE_CYCLE_TIMER_H_

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef LARGE_INTEGER DracoTimeVal;
#else
#include <sys/time.h>
typedef timeval DracoTimeVal;
#endif

#include <cinttypes>
#include <cstddef>

namespace draco {

class DracoTimer {
 public:
  DracoTimer() {}
  ~DracoTimer() {}
  void Start();
  void Stop();
  int64_t GetInMs();

 private:
  DracoTimeVal tv_start_;
  DracoTimeVal tv_end_;
};

typedef DracoTimer CycleTimer;

}  // namespace draco

#endif  // DRACO_CORE_CYCLE_TIMER_H_
