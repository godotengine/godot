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

#include "parallel_for_for.h"
#include "../sys/regression.h"

namespace embree
{
  struct parallel_for_for_regression_test : public RegressionTest
  {
    parallel_for_for_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      bool passed = true;

      /* create vector with random numbers */
      size_t sum0 = 0;
      size_t K = 0;
      const size_t M = 1000;
      std::vector<std::vector<size_t>* > array2(M);
      for (size_t i=0; i<M; i++) {
        const size_t N = rand() % 1024;
        K+=N;
        array2[i] = new std::vector<size_t>(N);
        for (size_t j=0; j<N; j++) 
          sum0 += (*array2[i])[j] = rand();
      }

      /* array to test global index */
      std::vector<atomic<size_t>> verify_k(K);
      for (size_t i=0; i<K; i++) verify_k[i].store(0);

      /* add all numbers using parallel_for_for */
      std::atomic<size_t> sum1(0);
      parallel_for_for( array2, size_t(1), [&](std::vector<size_t>* v, const range<size_t>& r, size_t k) -> size_t
      {
        size_t s = 0;
	for (size_t i=r.begin(); i<r.end(); i++) {
	  s += (*v)[i];
          verify_k[k++]++;
        }
        sum1 += s;
	return sum1;
      });
      passed &= (sum0 == sum1);

      /* check global index */
      for (size_t i=0; i<K; i++) 
        passed &= (verify_k[i] == 1);

      /* delete vectors again */
      for (size_t i=0; i<array2.size(); i++)
	delete array2[i];
      
      return passed;
    }
  };

  parallel_for_for_regression_test parallel_for_for_regression("parallel_for_for_regression_test");
}
