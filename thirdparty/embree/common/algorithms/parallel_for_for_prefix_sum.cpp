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

#include "parallel_for_for_prefix_sum.h"
#include "../sys/regression.h"

namespace embree
{
  struct parallel_for_for_prefix_sum_regression_test : public RegressionTest
  {
    parallel_for_for_prefix_sum_regression_test(const char* name) : RegressionTest(name) {
      registerRegressionTest(this);
    }
    
    bool run ()
    {
      bool passed = true;

      /* create vector with random numbers */
      const size_t M = 10;
      std::vector<atomic<size_t>> flattened;
      typedef std::vector<std::vector<size_t>* > ArrayArray;
      ArrayArray array2(M);
      size_t K = 0;
      for (size_t i=0; i<M; i++) {
        const size_t N = rand() % 10;
        K += N;
        array2[i] = new std::vector<size_t>(N);
        for (size_t j=0; j<N; j++) 
          (*array2[i])[j] = rand() % 10;
      }
  
      /* array to test global index */
      std::vector<atomic<size_t>> verify_k(K);
      for (size_t i=0; i<K; i++) verify_k[i].store(0);

      ParallelForForPrefixSumState<size_t> state(array2,size_t(1));
  
      /* dry run only counts */
      size_t S = parallel_for_for_prefix_sum0( state, array2, size_t(0), [&](std::vector<size_t>* v, const range<size_t>& r, size_t k) -> size_t
      {
        size_t s = 0;
	for (size_t i=r.begin(); i<r.end(); i++) {
          s += (*v)[i];
          verify_k[k++]++;
        }
        return s;
      }, [](size_t v0, size_t v1) { return v0+v1; });
      
      /* create properly sized output array */
      flattened.resize(S);
      for (auto& a : flattened) a.store(0);

      /* now we actually fill the flattened array */
      parallel_for_for_prefix_sum1( state, array2, size_t(0), [&](std::vector<size_t>* v, const range<size_t>& r, size_t k, const size_t base) -> size_t
      {
        size_t s = 0;
	for (size_t i=r.begin(); i<r.end(); i++) {
          for (size_t j=0; j<(*v)[i]; j++) {
            flattened[base+s+j]++;
          }
          s += (*v)[i];
          verify_k[k++]++;
        }
        return s;
      }, [](size_t v0, size_t v1) { return v0+v1; });

      /* check global index */
      for (size_t i=0; i<K; i++) 
        passed &= (verify_k[i] == 2);

      /* check if each element was assigned exactly once */
      for (size_t i=0; i<flattened.size(); i++)
        passed &= (flattened[i] == 1);
      
      /* delete arrays again */
      for (size_t i=0; i<array2.size(); i++)
	delete array2[i];

      return passed;
    }
  };

  parallel_for_for_prefix_sum_regression_test parallel_for_for_prefix_sum_regression("parallel_for_for_prefix_sum_regression_test");
}
