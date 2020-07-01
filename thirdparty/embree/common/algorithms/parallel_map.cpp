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

#include "parallel_map.h"
#include "../sys/regression.h"

namespace embree
{
  struct parallel_map_regression_test : public RegressionTest
  {
    parallel_map_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      bool passed = true;

      /* create key/value vectors with random numbers */
      const size_t N = 10000;
      std::vector<uint32_t> keys(N);
      std::vector<uint32_t> vals(N);
      for (size_t i=0; i<N; i++) keys[i] = 2*unsigned(i)*647382649;
      for (size_t i=0; i<N; i++) std::swap(keys[i],keys[rand()%N]);
      for (size_t i=0; i<N; i++) vals[i] = 2*rand();
      
      /* create map */
      parallel_map<uint32_t,uint32_t> map;
      map.init(keys,vals);

      /* check that all keys are properly mapped */
      for (size_t i=0; i<N; i++) {
        const uint32_t* val = map.lookup(keys[i]);
        passed &= val && (*val == vals[i]);
      }

      /* check that these keys are not in the map */
      for (size_t i=0; i<N; i++) {
        passed &= !map.lookup(keys[i]+1);
      }

      return passed;
    }
  };

  parallel_map_regression_test parallel_map_regression("parallel_map_regression_test");
}
