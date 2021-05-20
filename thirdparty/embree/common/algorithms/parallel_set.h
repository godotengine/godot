// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
