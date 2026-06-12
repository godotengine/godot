// Copyright 2016 The Draco Authors.
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
//
#ifndef DRACO_CORE_DRACO_INDEX_TYPE_VECTOR_H_
#define DRACO_CORE_DRACO_INDEX_TYPE_VECTOR_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "draco/core/draco_index_type.h"

namespace draco {

// A wrapper around the standard std::vector that supports indexing of the
// vector entries using the strongly typed indices as defined in
// draco_index_type.h.
// TODO(ostava): Make the interface more complete. It's currently missing some
// features.
template <class IndexTypeT, class ValueTypeT>
class IndexTypeVector {
 public:
  typedef typename std::vector<ValueTypeT>::const_reference const_reference;
  typedef typename std::vector<ValueTypeT>::reference reference;
  typedef typename std::vector<ValueTypeT>::iterator iterator;
  typedef typename std::vector<ValueTypeT>::const_iterator const_iterator;

  IndexTypeVector() {}
  explicit IndexTypeVector(size_t size) : vector_(size) {}
  IndexTypeVector(size_t size, const ValueTypeT &val) : vector_(size, val) {}

  iterator begin() { return vector_.begin(); }
  const_iterator begin() const { return vector_.begin(); }
  iterator end() { return vector_.end(); }
  const_iterator end() const { return vector_.end(); }

  void clear() { vector_.clear(); }
  void reserve(size_t size) { vector_.reserve(size); }
  void resize(size_t size) { vector_.resize(size); }
  void resize(size_t size, const ValueTypeT &val) { vector_.resize(size, val); }
  void assign(size_t size, const ValueTypeT &val) { vector_.assign(size, val); }
  iterator erase(iterator position) { return vector_.erase(position); }

  void swap(IndexTypeVector<IndexTypeT, ValueTypeT> &arg) {
    vector_.swap(arg.vector_);
  }

  size_t size() const { return vector_.size(); }
  bool empty() const { return vector_.empty(); }

  void push_back(const ValueTypeT &val) { vector_.push_back(val); }
  void push_back(ValueTypeT &&val) { vector_.push_back(std::move(val)); }

  template <typename... Args>
  void emplace_back(Args &&...args) {
    vector_.emplace_back(std::forward<Args>(args)...);
  }

  inline reference operator[](const IndexTypeT &index) {
    return vector_[index.value()];
  }
  inline const_reference operator[](const IndexTypeT &index) const {
    return vector_[index.value()];
  }
  inline reference at(const IndexTypeT &index) {
    return vector_[index.value()];
  }
  inline const_reference at(const IndexTypeT &index) const {
    return vector_[index.value()];
  }
  const ValueTypeT *data() const { return vector_.data(); }

 private:
  std::vector<ValueTypeT> vector_;
};

}  // namespace draco

#endif  // DRACO_CORE_DRACO_INDEX_TYPE_VECTOR_H_
