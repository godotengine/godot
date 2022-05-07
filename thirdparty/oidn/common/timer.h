// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "platform.h"
#include <chrono>

namespace oidn {

  class Timer
  {
  private:
    using clock = std::chrono::high_resolution_clock;

    std::chrono::time_point<clock> start;

  public:
    Timer()
    {
      reset();
    }

    void reset()
    {
      start = clock::now();
    }

    double query() const
    {
      auto end = clock::now();
      return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }
  };

} // namespace oidn
