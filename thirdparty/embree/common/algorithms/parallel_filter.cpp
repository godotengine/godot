// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "parallel_filter.h"
#include "../sys/regression.h"
#include <map>

namespace embree
{
  struct parallel_filter_regression_test : public RegressionTest
  {
    parallel_filter_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      bool passed = true;
      auto pred = [&]( uint32_t v ) { return (v & 0x3) == 0; };
      
      for (size_t N=10; N<1000000; N=size_t(2.1*N))
      {
        size_t N0 = rand() % N;
        
	/* initialize array with random numbers */
	std::vector<uint32_t> src(N);
        std::map<uint32_t,int> m;
	for (size_t i=0; i<N; i++) src[i] = rand();

        /* count elements up */
	for (size_t i=N0; i<N; i++)
          if (pred(src[i]))
            m[src[i]] = 0;
        for (size_t i=N0; i<N; i++)
          if (pred(src[i]))
            m[src[i]]++;

        /* filter array */
        //size_t M = sequential_filter(src.data(),N0,N,pred);
        size_t M = parallel_filter(src.data(),N0,N,size_t(1024),pred);
        
	/* check if filtered data is correct */
	for (size_t i=N0; i<M; i++) {
          passed &= pred(src[i]);
          m[src[i]]--;
        }
	for (size_t i=N0; i<M; i++)
          passed &= (m[src[i]] == 0);
      }

      return passed;
    }
  };

  parallel_filter_regression_test parallel_filter_regression("parallel_filter_regression");
}
