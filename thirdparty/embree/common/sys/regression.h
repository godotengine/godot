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

#include "platform.h"

#include <vector>

namespace embree
{
  /*! virtual interface for all regression tests */
  struct RegressionTest 
  { 
    RegressionTest (std::string name) : name(name) {}
    virtual bool run() = 0;
    std::string name;
  };
 
  /*! registers a regression test */
  void registerRegressionTest(RegressionTest* test);

  /*! run all regression tests */
  RegressionTest* getRegressionTest(size_t index);
}
