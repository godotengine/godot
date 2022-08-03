// Copyright 2021 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <math.h>
#include <../third_party/thrust/thrust/binary_search.h>
#include <../third_party/thrust/thrust/gather.h>
#include <../third_party/thrust/thrust/remove.h>
#include <../third_party/thrust/thrust/sort.h>
#include <../third_party/thrust/thrust/unique.h>

#include "optional_assert.h"
#include "par.h"
#include "structs.h"
#include "utils.h"
#include "vec_dh.h"

namespace manifold {

/** @ingroup Private */
class SparseIndices {
  // COO-style sparse matrix storage. Values corresponding to these indicies are
  // stored in vectors separate from this class, but having the same length.
 public:
  SparseIndices(int size = 0) : p(size), q(size) {}
  typedef typename VecDH<int>::Iter Iter;
  typedef typename thrust::zip_iterator<thrust::tuple<Iter, Iter>> Zip;
  Zip beginPQ() { return zip(p.begin(), q.begin()); }
  Zip endPQ() { return zip(p.end(), q.end()); }
  Iter begin(bool use_q) { return use_q ? q.begin() : p.begin(); }
  Iter end(bool use_q) { return use_q ? q.end() : p.end(); }
  int* ptrD(bool use_q) { return use_q ? q.ptrD() : p.ptrD(); }
  thrust::pair<int*, int*> ptrDpq(int idx = 0) {
    return thrust::make_pair(p.ptrD() + idx, q.ptrD() + idx);
  }
  const thrust::pair<const int*, const int*> ptrDpq(int idx = 0) const {
    return thrust::make_pair(p.ptrD() + idx, q.ptrD() + idx);
  }
  const VecDH<int>& Get(bool use_q) const { return use_q ? q : p; }
  VecDH<int> Copy(bool use_q) const {
    VecDH<int> out = use_q ? q : p;
    return out;
  }

  typedef typename VecDH<int>::IterC IterC;
  typedef typename thrust::zip_iterator<thrust::tuple<IterC, IterC>> ZipC;
  ZipC beginPQ() const { return zip(p.begin(), q.begin()); }
  ZipC endPQ() const { return zip(p.end(), q.end()); }
  IterC begin(bool use_q) const { return use_q ? q.begin() : p.begin(); }
  IterC end(bool use_q) const { return use_q ? q.end() : p.end(); }
  const int* ptrD(bool use_q) const { return use_q ? q.ptrD() : p.ptrD(); }

  ZipC beginHpq() const { return zip(p.begin(), q.begin()); }
  ZipC endHpq() const { return zip(p.end(), q.end()); }

  int size() const { return p.size(); }
  void SwapPQ() { p.swap(q); }

  void Sort() { sort(autoPolicy(size()), beginPQ(), endPQ()); }

  void Resize(int size) {
    p.resize(size, -1);
    q.resize(size, -1);
  }

  void Unique() {
    Sort();
    int newSize =
        unique<decltype(beginPQ())>(autoPolicy(size()), beginPQ(), endPQ()) -
        beginPQ();
    Resize(newSize);
  }

  struct firstZero {
    __host__ __device__ bool operator()(thrust::tuple<int, int, int> x) const {
      return thrust::get<0>(x) == 0;
    }
  };

  size_t RemoveZeros(VecDH<int>& S) {
    ASSERT(S.size() == p.size(), userErr,
           "Different number of values than indicies!");
    auto zBegin = zip(S.begin(), begin(false), begin(true));
    auto zEnd = zip(S.end(), end(false), end(true));
    size_t size = remove_if<decltype(zBegin)>(autoPolicy(S.size()), zBegin,
                                              zEnd, firstZero()) -
                  zBegin;
    S.resize(size, -1);
    p.resize(size, -1);
    q.resize(size, -1);
    return size;
  }

  template <typename T>
  struct firstNonFinite {
    __host__ __device__ bool NotFinite(float v) const { return !isfinite(v); }
    __host__ __device__ bool NotFinite(glm::vec2 v) const {
      return !isfinite(v[0]);
    }
    __host__ __device__ bool NotFinite(glm::vec3 v) const {
      return !isfinite(v[0]);
    }
    __host__ __device__ bool NotFinite(glm::vec4 v) const {
      return !isfinite(v[0]);
    }

    __host__ __device__ bool operator()(
        thrust::tuple<T, int, int, int> x) const {
      bool result = NotFinite(thrust::get<0>(x));
      return result;
    }
  };

  template <typename T>
  size_t KeepFinite(VecDH<T>& v, VecDH<int>& x) {
    ASSERT(x.size() == p.size(), userErr,
           "Different number of values than indicies!");
    auto zBegin = zip(v.begin(), x.begin(), begin(false), begin(true));
    auto zEnd = zip(v.end(), x.end(), end(false), end(true));
    size_t size = remove_if<decltype(zBegin)>(autoPolicy(v.size()), zBegin,
                                              zEnd, firstNonFinite<T>()) -
                  zBegin;
    v.resize(size);
    x.resize(size, -1);
    p.resize(size, -1);
    q.resize(size, -1);
    return size;
  }

  template <typename Iter, typename T>
  VecDH<T> Gather(const VecDH<T>& val, const Iter pqBegin, const Iter pqEnd,
                  T missingVal) const {
    ASSERT(val.size() == p.size(), userErr,
           "Different number of values than indicies!");
    size_t size = pqEnd - pqBegin;
    VecDH<T> result(size);
    VecDH<char> found(size);
    VecDH<int> temp(size);
    auto policy = autoPolicy(size);
    fill(policy, result.begin(), result.end(), missingVal);
    binary_search(policy, beginPQ(), endPQ(), pqBegin, pqEnd, found.begin());
    lower_bound(policy, beginPQ(), endPQ(), pqBegin, pqEnd, temp.begin());
    gather_if(policy, temp.begin(), temp.end(), found.begin(), val.begin(),
              result.begin());
    return result;
  }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    const auto& p = Get(0);
    const auto& q = Get(1);
    std::cout << "SparseIndices = " << std::endl;
    for (int i = 0; i < size(); ++i) {
      std::cout << i << ", p = " << p[i] << ", q = " << q[i] << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 private:
  VecDH<int> p, q;
};
}  // namespace manifold
