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
#include <mutex>
#include <condition_variable>

namespace oidn {

  class Barrier
  {
  private:
    std::mutex m;
    std::condition_variable cv;
    volatile int count;

  public:
    Barrier(int count) : count(count) {}

    void wait()
    {
      std::unique_lock<std::mutex> lk(m);
      count--;

      if (count == 0)
      {
        lk.unlock();
        cv.notify_all();
      }
      else
      {
        cv.wait(lk, [&]{ return count == 0; });
      }
    }
  };

} // namespace oidn
