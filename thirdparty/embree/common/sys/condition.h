// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#include "mutex.h"

namespace embree
{
  class ConditionSys
  {
  public:
    ConditionSys();
    ~ConditionSys();
    void wait( class MutexSys& mutex );
    void notify_all();

    template<typename Predicate>
      __forceinline void wait( class MutexSys& mutex, const Predicate& pred )
    {
      while (!pred()) wait(mutex);
    }

  private:
    ConditionSys (const ConditionSys& other) DELETED; // do not implement
    ConditionSys& operator= (const ConditionSys& other) DELETED; // do not implement

  protected:
    void* cond;
  };
}
