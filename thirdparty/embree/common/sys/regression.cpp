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

#include "regression.h"

namespace embree
{
  static std::unique_ptr<std::vector<RegressionTest*>> regression_tests;

  void registerRegressionTest(RegressionTest* test) 
  {
    if (!regression_tests) 
      regression_tests = std::unique_ptr<std::vector<RegressionTest*>>(new std::vector<RegressionTest*>);

    regression_tests->push_back(test);
  }

  RegressionTest* getRegressionTest(size_t index)
  {
    if (!regression_tests) 
      return nullptr;

    if (index >= regression_tests->size())
      return nullptr;
    
    return (*regression_tests)[index];
  }
}
