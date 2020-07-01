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

#include "parallel_sort.h"

namespace embree
{
  /* implementation of a set of values with parallel construction */
  template<typename T>
  class parallel_set
  {
  public:

    /*! default constructor for the parallel set */
    parallel_set () {}

    /*! construction from vector */
    template<typename Vector>
      parallel_set (const Vector& in) { init(in); }

    /*! initialized the parallel set from a vector */
    template<typename Vector>
      void init(const Vector& in) 
    {
      /* copy data to internal vector */
      vec.resize(in.size());
      parallel_for( size_t(0), in.size(), size_t(4*4096), [&](const range<size_t>& r) {
	for (size_t i=r.begin(); i<r.end(); i++) 
	  vec[i] = in[i];
      });

      /* sort the data */
      std::vector<T> temp(in.size());
      radix_sort<T>(vec.data(),temp.data(),vec.size());
    }

    /*! tests if some element is in the set */
    __forceinline bool lookup(const T& elt) const {
      return std::binary_search(vec.begin(), vec.end(), elt);
    }

    /*! clears all state */
    void clear() {
      vec.clear();
    }

  private:
    std::vector<T> vec;   //!< vector containing sorted elements
  };
}
