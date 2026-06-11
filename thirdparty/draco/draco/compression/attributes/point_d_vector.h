// Copyright 2018 The Draco Authors.
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

#ifndef DRACO_COMPRESSION_ATTRIBUTES_POINT_D_VECTOR_H_
#define DRACO_COMPRESSION_ATTRIBUTES_POINT_D_VECTOR_H_

#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "draco/core/macros.h"

namespace draco {

// The main class of this file is PointDVector providing an interface similar to
// std::vector<PointD> for arbitrary number of dimensions (without a template
// argument). PointDVectorIterator is a random access iterator, which allows for
// compatibility with existing algorithms. PseudoPointD provides for a view on
// the individual items in a contiguous block of memory, which is compatible
// with the swap function and is returned by a dereference of
// PointDVectorIterator. Swap functions provide for compatibility/specialization
// that allows these classes to work with currently utilized STL functions.

// This class allows for swap functionality from the RandomIterator
// It seems problematic to bring this inside PointDVector due to templating.
template <typename internal_t>
class PseudoPointD {
 public:
  PseudoPointD(internal_t *mem, internal_t dimension)
      : mem_(mem), dimension_(dimension) {}

  // Specifically copies referenced memory
  void swap(PseudoPointD &other) noexcept {
    for (internal_t dim = 0; dim < dimension_; dim += 1) {
      std::swap(mem_[dim], other.mem_[dim]);
    }
  }

  PseudoPointD(const PseudoPointD &other)
      : mem_(other.mem_), dimension_(other.dimension_) {}

  const internal_t &operator[](const size_t &n) const {
    DRACO_DCHECK_LT(n, dimension_);
    return mem_[n];
  }
  internal_t &operator[](const size_t &n) {
    DRACO_DCHECK_LT(n, dimension_);
    return mem_[n];
  }

  bool operator==(const PseudoPointD &other) const {
    for (auto dim = 0; dim < dimension_; dim += 1) {
      if (mem_[dim] != other.mem_[dim]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const PseudoPointD &other) const {
    return !this->operator==(other);
  }

 private:
  internal_t *const mem_;
  const internal_t dimension_;
};

// It seems problematic to bring this inside PointDVector due to templating.
template <typename internal_t>
void swap(draco::PseudoPointD<internal_t> &&a,
          draco::PseudoPointD<internal_t> &&b) noexcept {
  a.swap(b);
};
template <typename internal_t>
void swap(draco::PseudoPointD<internal_t> &a,
          draco::PseudoPointD<internal_t> &b) noexcept {
  a.swap(b);
};

template <typename internal_t>
class PointDVector {
 public:
  PointDVector(const uint32_t n_items, const uint32_t dimensionality)
      : n_items_(n_items),
        dimensionality_(dimensionality),
        item_size_bytes_(dimensionality * sizeof(internal_t)),
        data_(n_items * dimensionality),
        data0_(data_.data()) {}
  // random access iterator
  class PointDVectorIterator {
    friend class PointDVector;

   public:
    // Iterator traits expected by std libraries.
    using iterator_category = std::random_access_iterator_tag;
    using value_type = size_t;
    using difference_type = size_t;
    using pointer = PointDVector *;
    using reference = PointDVector &;

    // std::iter_swap is called inside of std::partition and needs this
    // specialized support
    PseudoPointD<internal_t> operator*() const {
      return PseudoPointD<internal_t>(vec_->data0_ + item_ * dimensionality_,
                                      dimensionality_);
    }
    const PointDVectorIterator &operator++() {
      item_ += 1;
      return *this;
    }
    const PointDVectorIterator &operator--() {
      item_ -= 1;
      return *this;
    }
    PointDVectorIterator operator++(int32_t) {
      PointDVectorIterator copy(*this);
      item_ += 1;
      return copy;
    }
    PointDVectorIterator operator--(int32_t) {
      PointDVectorIterator copy(*this);
      item_ -= 1;
      return copy;
    }
    PointDVectorIterator &operator=(const PointDVectorIterator &other) {
      this->item_ = other.item_;
      return *this;
    }

    bool operator==(const PointDVectorIterator &ref) const {
      return item_ == ref.item_;
    }
    bool operator!=(const PointDVectorIterator &ref) const {
      return item_ != ref.item_;
    }
    bool operator<(const PointDVectorIterator &ref) const {
      return item_ < ref.item_;
    }
    bool operator>(const PointDVectorIterator &ref) const {
      return item_ > ref.item_;
    }
    bool operator<=(const PointDVectorIterator &ref) const {
      return item_ <= ref.item_;
    }
    bool operator>=(const PointDVectorIterator &ref) const {
      return item_ >= ref.item_;
    }

    PointDVectorIterator operator+(const int32_t &add) const {
      PointDVectorIterator copy(vec_, item_ + add);
      return copy;
    }
    PointDVectorIterator &operator+=(const int32_t &add) {
      item_ += add;
      return *this;
    }
    PointDVectorIterator operator-(const int32_t &sub) const {
      PointDVectorIterator copy(vec_, item_ - sub);
      return copy;
    }
    size_t operator-(const PointDVectorIterator &sub) const {
      return (item_ - sub.item_);
    }

    PointDVectorIterator &operator-=(const int32_t &sub) {
      item_ -= sub;
      return *this;
    }

    internal_t *operator[](const size_t &n) const {
      return vec_->data0_ + (item_ + n) * dimensionality_;
    }

   protected:
    explicit PointDVectorIterator(PointDVector *vec, size_t start_item)
        : item_(start_item), vec_(vec), dimensionality_(vec->dimensionality_) {}

   private:
    size_t item_;  // this counts the item that should be referenced.
    PointDVector *const vec_;        // the thing that we're iterating on
    const uint32_t dimensionality_;  // local copy from vec_
  };

  PointDVectorIterator begin() { return PointDVectorIterator(this, 0); }
  PointDVectorIterator end() { return PointDVectorIterator(this, n_items_); }

  // operator[] allows for unprotected user-side usage of operator[] on the
  // return value AS IF it were a natively indexable type like Point3*
  internal_t *operator[](const uint32_t index) {
    DRACO_DCHECK_LT(index, n_items_);
    return data0_ + index * dimensionality_;
  }
  const internal_t *operator[](const uint32_t index) const {
    DRACO_DCHECK_LT(index, n_items_);
    return data0_ + index * dimensionality_;
  }

  uint32_t size() const { return n_items_; }
  size_t GetBufferSize() const { return data_.size(); }

  // copy a single contiguous 'item' from one PointDVector into this one.
  void CopyItem(const PointDVector &source, const internal_t source_index,
                const internal_t destination_index) {
    DRACO_DCHECK(&source != this ||
                 (&source == this && source_index != destination_index));
    DRACO_DCHECK_LT(destination_index, n_items_);
    DRACO_DCHECK_LT(source_index, source.n_items_);

    // DRACO_DCHECK_EQ(source.n_items_, n_items_); // not technically necessary
    DRACO_DCHECK_EQ(source.dimensionality_, dimensionality_);

    const internal_t *ref = source[source_index];
    internal_t *const dest = this->operator[](destination_index);
    std::memcpy(dest, ref, item_size_bytes_);
  }

  // Copy data directly off of an attribute buffer interleaved into internal
  // memory.
  void CopyAttribute(
      // The dimensionality of the attribute being integrated
      const internal_t attribute_dimensionality,
      // The offset in dimensions to insert this attribute.
      const internal_t offset_dimensionality, const internal_t index,
      // The direct pointer to the data
      const void *const attribute_item_data) {
    // chunk copy
    const size_t copy_size = sizeof(internal_t) * attribute_dimensionality;

    // a multiply and add can be optimized away with an iterator
    std::memcpy(data0_ + index * dimensionality_ + offset_dimensionality,
                attribute_item_data, copy_size);
  }
  // Copy data off of a contiguous buffer interleaved into internal memory
  void CopyAttribute(
      // The dimensionality of the attribute being integrated
      const internal_t attribute_dimensionality,
      // The offset in dimensions to insert this attribute.
      const internal_t offset_dimensionality,
      const internal_t *const attribute_mem) {
    DRACO_DCHECK_LT(offset_dimensionality,
                    dimensionality_ - attribute_dimensionality);
    // degenerate case block copy the whole buffer.
    if (dimensionality_ == attribute_dimensionality) {
      DRACO_DCHECK_EQ(offset_dimensionality, 0);
      const size_t copy_size =
          sizeof(internal_t) * attribute_dimensionality * n_items_;
      std::memcpy(data0_, attribute_mem, copy_size);
    } else {  // chunk copy
      const size_t copy_size = sizeof(internal_t) * attribute_dimensionality;
      internal_t *internal_data;
      const internal_t *attribute_data;
      internal_t item;
      for (internal_data = data0_ + offset_dimensionality,
          attribute_data = attribute_mem, item = 0;
           item < n_items_; internal_data += dimensionality_,
          attribute_data += attribute_dimensionality, item += 1) {
        std::memcpy(internal_data, attribute_data, copy_size);
      }
    }
  }

 private:
  // internal parameters.
  const uint32_t n_items_;
  const uint32_t dimensionality_;  // The dimension of the points in the buffer
  const uint32_t item_size_bytes_;
  std::vector<internal_t> data_;  // contiguously stored data.  Never resized.
  internal_t *const data0_;       // raw pointer to base data.
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_POINT_D_VECTOR_H_
