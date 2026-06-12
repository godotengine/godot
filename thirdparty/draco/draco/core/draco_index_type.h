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
// This files provides a basic framework for strongly typed indices that are
// used within the Draco library. The motivation of using strongly typed indices
// is to prevent bugs caused by mixing up incompatible indices, such as indexing
// mesh faces with point indices and vice versa.
//
// Usage:
//      Define strongly typed index using macro:
//
//        DEFINE_NEW_DRACO_INDEX_TYPE(value_type, name)
//
//      where |value_type| is the data type of the index value (such as int32_t)
//      and |name| is a unique typename of the new index.
//
//      E.g., we can define new index types as:
//
//        DEFINE_NEW_DRACO_INDEX_TYPE(int, PointIndex)
//        DEFINE_NEW_DRACO_INDEX_TYPE(int, FaceIndex)
//
//      The new types can then be used in the similar way as the regular weakly
//      typed indices (such as int32, int64, ...), but they cannot be
//      accidentally misassigned. E.g.:
//
//        PointIndex point_index(10);
//        FaceIndex face_index;
//        face_index = point_index;  // Compile error!
//
//      One can still cast one type to another explicitly by accessing the index
//      value directly using the .value() method:
//
//        face_index = FaceIndex(point_index.value());  // Compiles OK.
//
//      Strongly typed indices support most of the common binary and unary
//      operators and support for additional operators can be added if
//      necessary.

#ifndef DRACO_CORE_DRACO_INDEX_TYPE_H_
#define DRACO_CORE_DRACO_INDEX_TYPE_H_

#include <ostream>

#include "draco/draco_features.h"

namespace draco {

#define DEFINE_NEW_DRACO_INDEX_TYPE(value_type, name) \
  struct name##_tag_type_ {};                         \
  typedef IndexType<value_type, name##_tag_type_> name;

template <class ValueTypeT, class TagT>
class IndexType {
 public:
  typedef IndexType<ValueTypeT, TagT> ThisIndexType;
  typedef ValueTypeT ValueType;

  constexpr IndexType() : value_(ValueTypeT()) {}
  constexpr explicit IndexType(ValueTypeT value) : value_(value) {}

  constexpr ValueTypeT value() const { return value_; }

  constexpr bool operator==(const IndexType &i) const {
    return value_ == i.value_;
  }
  constexpr bool operator==(const ValueTypeT &val) const {
    return value_ == val;
  }
  constexpr bool operator!=(const IndexType &i) const {
    return value_ != i.value_;
  }
  constexpr bool operator!=(const ValueTypeT &val) const {
    return value_ != val;
  }
  constexpr bool operator<(const IndexType &i) const {
    return value_ < i.value_;
  }
  constexpr bool operator<(const ValueTypeT &val) const { return value_ < val; }
  constexpr bool operator>(const IndexType &i) const {
    return value_ > i.value_;
  }
  constexpr bool operator>(const ValueTypeT &val) const { return value_ > val; }
  constexpr bool operator>=(const IndexType &i) const {
    return value_ >= i.value_;
  }
  constexpr bool operator>=(const ValueTypeT &val) const {
    return value_ >= val;
  }

  inline ThisIndexType &operator++() {
    ++value_;
    return *this;
  }
  inline ThisIndexType operator++(int) {
    const ThisIndexType ret(value_);
    ++value_;
    return ret;
  }

  inline ThisIndexType &operator--() {
    --value_;
    return *this;
  }
  inline ThisIndexType operator--(int) {
    const ThisIndexType ret(value_);
    --value_;
    return ret;
  }

  constexpr ThisIndexType operator+(const IndexType &i) const {
    return ThisIndexType(value_ + i.value_);
  }
  constexpr ThisIndexType operator+(const ValueTypeT &val) const {
    return ThisIndexType(value_ + val);
  }
  constexpr ThisIndexType operator-(const IndexType &i) const {
    return ThisIndexType(value_ - i.value_);
  }
  constexpr ThisIndexType operator-(const ValueTypeT &val) const {
    return ThisIndexType(value_ - val);
  }

  inline ThisIndexType &operator+=(const IndexType &i) {
    value_ += i.value_;
    return *this;
  }
  inline ThisIndexType operator+=(const ValueTypeT &val) {
    value_ += val;
    return *this;
  }
  inline ThisIndexType &operator-=(const IndexType &i) {
    value_ -= i.value_;
    return *this;
  }
  inline ThisIndexType operator-=(const ValueTypeT &val) {
    value_ -= val;
    return *this;
  }
  inline ThisIndexType &operator=(const ThisIndexType &i) {
    value_ = i.value_;
    return *this;
  }
  inline ThisIndexType &operator=(const ValueTypeT &val) {
    value_ = val;
    return *this;
  }

 private:
  ValueTypeT value_;
};

// Stream operator << provided for logging purposes.
template <class ValueTypeT, class TagT>
std::ostream &operator<<(std::ostream &os, IndexType<ValueTypeT, TagT> index) {
  return os << index.value();
}

}  // namespace draco

// Specialize std::hash for the strongly indexed types.
namespace std {

template <class ValueTypeT, class TagT>
struct hash<draco::IndexType<ValueTypeT, TagT>> {
  size_t operator()(const draco::IndexType<ValueTypeT, TagT> &i) const {
    return static_cast<size_t>(i.value());
  }
};

}  // namespace std

#endif  // DRACO_CORE_DRACO_INDEX_TYPE_H_
