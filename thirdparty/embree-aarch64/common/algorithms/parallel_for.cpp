// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "parallel_for.h"
#include "../sys/regression.h"

namespace embree
{
  struct parallel_for_regression_test : public RegressionTest
  {
    parallel_for_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      bool passed = true;

      const size_t M = 10;
      for (size_t N=10; N<10000000; N=size_t(2.1*N))
      {
        /* sequentially calculate sum of squares */
        size_t sum0 = 0;
        for (size_t i=0; i<N; i++) {
          sum0 += i*i;
        }

        /* parallel calculation of sum of squares */
        for (size_t m=0; m<M; m++)
        {
          std::atomic<size_t> sum1(0);
          parallel_for( size_t(0), size_t(N), size_t(1024), [&](const range<size_t>& r) 
          {
            size_t s = 0;
            for (size_t i=r.begin(); i<r.end(); i++) 
              s += i*i;
            sum1 += s;
          });
          passed = sum0 == sum1;
        }
      }
      
      return passed;
    }
  };

  parallel_for_regression_test parallel_for_regression("parallel_for_regression_test");
}
