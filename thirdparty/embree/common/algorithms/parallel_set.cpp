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

#include "parallel_set.h"
#include "../sys/regression.h"

namespace embree
{
  struct parallel_set_regression_test : public RegressionTest
  {
    parallel_set_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      bool passed = true;

      /* create vector with random numbers */
      const size_t N = 10000;
      std::vector<uint32_t> unsorted(N);
      for (size_t i=0; i<N; i++) unsorted[i] = 2*rand();
      
      /* created set from numbers */
      parallel_set<uint32_t> sorted;
      sorted.init(unsorted);

      /* check that all elements are in the set */
      for (size_t i=0; i<N; i++) {
	passed &= sorted.lookup(unsorted[i]);
      }

      /* check that these elements are not in the set */
      for (size_t i=0; i<N; i++) {
	passed &= !sorted.lookup(unsorted[i]+1);
      }

      return passed;
    }
  };

  parallel_set_regression_test parallel_set_regression("parallel_set_regression_test");
}
