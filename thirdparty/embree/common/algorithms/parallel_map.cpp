// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
