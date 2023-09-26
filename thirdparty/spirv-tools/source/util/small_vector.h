// Copyright (c) 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_UTIL_SMALL_VECTOR_H_
#define SOURCE_UTIL_SMALL_VECTOR_H_

#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "source/util/make_unique.h"

namespace spvtools {
namespace utils {

// The |SmallVector| class is intended to be a drop-in replacement for
// |std::vector|.  The difference is in the implementation. A |SmallVector| is
// optimized for when the number of elements in the vector are small.  Small is
// defined by the template parameter |small_size|.
//
// Note that |SmallVector| is not always faster than an |std::vector|, so you
// should experiment with different values for |small_size| and compare to
// using and |std::vector|.
//
// TODO: I have implemented the public member functions from |std::vector| that
// I needed.  If others are needed they should be implemented. Do not implement
// public member functions that are not defined by std::vector.
template <class T, size_t small_size>
class SmallVector {
 public:
  using iterator = T*;
  using const_iterator = const T*;

  SmallVector()
      : size_(0),
        small_data_(reinterpret_cast<T*>(buffer)),
        large_data_(nullptr) {}

  SmallVector(const SmallVector& that) : SmallVector() { *this = that; }

  SmallVector(SmallVector&& that) : SmallVector() { *this = std::move(that); }

  SmallVector(const std::vector<T>& vec) : SmallVector() {
    if (vec.size() > small_size) {
      large_data_ = MakeUnique<std::vector<T>>(vec);
    } else {
      size_ = vec.size();
      for (uint32_t i = 0; i < size_; i++) {
        new (small_data_ + i) T(vec[i]);
      }
    }
  }

  template <class InputIt>
  SmallVector(InputIt first, InputIt last) : SmallVector() {
    insert(end(), first, last);
  }

  SmallVector(std::vector<T>&& vec) : SmallVector() {
    if (vec.size() > small_size) {
      large_data_ = MakeUnique<std::vector<T>>(std::move(vec));
    } else {
      size_ = vec.size();
      for (uint32_t i = 0; i < size_; i++) {
        new (small_data_ + i) T(std::move(vec[i]));
      }
    }
    vec.clear();
  }

  SmallVector(std::initializer_list<T> init_list) : SmallVector() {
    if (init_list.size() < small_size) {
      for (auto it = init_list.begin(); it != init_list.end(); ++it) {
        new (small_data_ + (size_++)) T(std::move(*it));
      }
    } else {
      large_data_ = MakeUnique<std::vector<T>>(std::move(init_list));
    }
  }

  SmallVector(size_t s, const T& v) : SmallVector() { resize(s, v); }

  virtual ~SmallVector() {
    for (T* p = small_data_; p < small_data_ + size_; ++p) {
      p->~T();
    }
  }

  SmallVector& operator=(const SmallVector& that) {
    assert(small_data_);
    if (that.large_data_) {
      if (large_data_) {
        *large_data_ = *that.large_data_;
      } else {
        large_data_ = MakeUnique<std::vector<T>>(*that.large_data_);
      }
    } else {
      large_data_.reset(nullptr);
      size_t i = 0;
      // Do a copy for any element in |this| that is already constructed.
      for (; i < size_ && i < that.size_; ++i) {
        small_data_[i] = that.small_data_[i];
      }

      if (i >= that.size_) {
        // If the size of |this| becomes smaller after the assignment, then
        // destroy any extra elements.
        for (; i < size_; ++i) {
          small_data_[i].~T();
        }
      } else {
        // If the size of |this| becomes larger after the assignement, copy
        // construct the new elements that are needed.
        for (; i < that.size_; ++i) {
          new (small_data_ + i) T(that.small_data_[i]);
        }
      }
      size_ = that.size_;
    }
    return *this;
  }

  SmallVector& operator=(SmallVector&& that) {
    if (that.large_data_) {
      large_data_.reset(that.large_data_.release());
    } else {
      large_data_.reset(nullptr);
      size_t i = 0;
      // Do a move for any element in |this| that is already constructed.
      for (; i < size_ && i < that.size_; ++i) {
        small_data_[i] = std::move(that.small_data_[i]);
      }

      if (i >= that.size_) {
        // If the size of |this| becomes smaller after the assignment, then
        // destroy any extra elements.
        for (; i < size_; ++i) {
          small_data_[i].~T();
        }
      } else {
        // If the size of |this| becomes larger after the assignement, move
        // construct the new elements that are needed.
        for (; i < that.size_; ++i) {
          new (small_data_ + i) T(std::move(that.small_data_[i]));
        }
      }
      size_ = that.size_;
    }

    // Reset |that| because all of the data has been moved to |this|.
    that.DestructSmallData();
    return *this;
  }

  template <class OtherVector>
  friend bool operator==(const SmallVector& lhs, const OtherVector& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }

    auto rit = rhs.begin();
    for (auto lit = lhs.begin(); lit != lhs.end(); ++lit, ++rit) {
      if (*lit != *rit) {
        return false;
      }
    }
    return true;
  }

// Avoid infinite recursion from rewritten operators in C++20
#if __cplusplus <= 201703L
  friend bool operator==(const std::vector<T>& lhs, const SmallVector& rhs) {
    return rhs == lhs;
  }
#endif

  friend bool operator!=(const SmallVector& lhs, const std::vector<T>& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator!=(const std::vector<T>& lhs, const SmallVector& rhs) {
    return rhs != lhs;
  }

  T& operator[](size_t i) {
    if (!large_data_) {
      return small_data_[i];
    } else {
      return (*large_data_)[i];
    }
  }

  const T& operator[](size_t i) const {
    if (!large_data_) {
      return small_data_[i];
    } else {
      return (*large_data_)[i];
    }
  }

  size_t size() const {
    if (!large_data_) {
      return size_;
    } else {
      return large_data_->size();
    }
  }

  iterator begin() {
    if (large_data_) {
      return large_data_->data();
    } else {
      return small_data_;
    }
  }

  const_iterator begin() const {
    if (large_data_) {
      return large_data_->data();
    } else {
      return small_data_;
    }
  }

  const_iterator cbegin() const { return begin(); }

  iterator end() {
    if (large_data_) {
      return large_data_->data() + large_data_->size();
    } else {
      return small_data_ + size_;
    }
  }

  const_iterator end() const {
    if (large_data_) {
      return large_data_->data() + large_data_->size();
    } else {
      return small_data_ + size_;
    }
  }

  const_iterator cend() const { return end(); }

  T* data() { return begin(); }

  const T* data() const { return cbegin(); }

  T& front() { return (*this)[0]; }

  const T& front() const { return (*this)[0]; }

  iterator erase(const_iterator pos) { return erase(pos, pos + 1); }

  iterator erase(const_iterator first, const_iterator last) {
    if (large_data_) {
      size_t start_index = first - large_data_->data();
      size_t end_index = last - large_data_->data();
      auto r = large_data_->erase(large_data_->begin() + start_index,
                                  large_data_->begin() + end_index);
      return large_data_->data() + (r - large_data_->begin());
    }

    // Since C++11, std::vector has |const_iterator| for the parameters, so I
    // follow that.  However, I need iterators to modify the current container,
    // which is not const.  This is why I cast away the const.
    iterator f = const_cast<iterator>(first);
    iterator l = const_cast<iterator>(last);
    iterator e = end();

    size_t num_of_del_elements = last - first;
    iterator ret = f;
    if (first == last) {
      return ret;
    }

    // Move |last| and any elements after it their earlier position.
    while (l != e) {
      *f = std::move(*l);
      ++f;
      ++l;
    }

    // Destroy the elements that were supposed to be deleted.
    while (f != l) {
      f->~T();
      ++f;
    }

    // Update the size.
    size_ -= num_of_del_elements;
    return ret;
  }

  void push_back(const T& value) {
    if (!large_data_ && size_ == small_size) {
      MoveToLargeData();
    }

    if (large_data_) {
      large_data_->push_back(value);
      return;
    }

    new (small_data_ + size_) T(value);
    ++size_;
  }

  void push_back(T&& value) {
    if (!large_data_ && size_ == small_size) {
      MoveToLargeData();
    }

    if (large_data_) {
      large_data_->push_back(std::move(value));
      return;
    }

    new (small_data_ + size_) T(std::move(value));
    ++size_;
  }

  void pop_back() {
    if (large_data_) {
      large_data_->pop_back();
    } else {
      --size_;
      small_data_[size_].~T();
    }
  }

  template <class InputIt>
  iterator insert(iterator pos, InputIt first, InputIt last) {
    size_t element_idx = (pos - begin());
    size_t num_of_new_elements = std::distance(first, last);
    size_t new_size = size_ + num_of_new_elements;
    if (!large_data_ && new_size > small_size) {
      MoveToLargeData();
    }

    if (large_data_) {
      typename std::vector<T>::iterator new_pos =
          large_data_->begin() + element_idx;
      large_data_->insert(new_pos, first, last);
      return begin() + element_idx;
    }

    // Move |pos| and all of the elements after it over |num_of_new_elements|
    // places.  We start at the end and work backwards, to make sure we do not
    // overwrite data that we have not moved yet.
    for (iterator i = begin() + new_size - 1, j = end() - 1; j >= pos;
         --i, --j) {
      if (i >= begin() + size_) {
        new (i) T(std::move(*j));
      } else {
        *i = std::move(*j);
      }
    }

    // Copy the new elements into position.
    iterator p = pos;
    for (; first != last; ++p, ++first) {
      if (p >= small_data_ + size_) {
        new (p) T(*first);
      } else {
        *p = *first;
      }
    }

    // Update the size.
    size_ += num_of_new_elements;
    return pos;
  }

  bool empty() const {
    if (large_data_) {
      return large_data_->empty();
    }
    return size_ == 0;
  }

  void clear() {
    if (large_data_) {
      large_data_->clear();
    } else {
      DestructSmallData();
    }
  }

  template <class... Args>
  void emplace_back(Args&&... args) {
    if (!large_data_ && size_ == small_size) {
      MoveToLargeData();
    }

    if (large_data_) {
      large_data_->emplace_back(std::forward<Args>(args)...);
    } else {
      new (small_data_ + size_) T(std::forward<Args>(args)...);
      ++size_;
    }
  }

  void resize(size_t new_size, const T& v) {
    if (!large_data_ && new_size > small_size) {
      MoveToLargeData();
    }

    if (large_data_) {
      large_data_->resize(new_size, v);
      return;
    }

    // If |new_size| < |size_|, then destroy the extra elements.
    for (size_t i = new_size; i < size_; ++i) {
      small_data_[i].~T();
    }

    // If |new_size| > |size_|, the copy construct the new elements.
    for (size_t i = size_; i < new_size; ++i) {
      new (small_data_ + i) T(v);
    }

    // Update the size.
    size_ = new_size;
  }

 private:
  // Moves all of the element from |small_data_| into a new std::vector that can
  // be access through |large_data|.
  void MoveToLargeData() {
    assert(!large_data_);
    large_data_ = MakeUnique<std::vector<T>>();
    for (size_t i = 0; i < size_; ++i) {
      large_data_->emplace_back(std::move(small_data_[i]));
    }
    DestructSmallData();
  }

  // Destroys all of the elements in |small_data_| that have been constructed.
  void DestructSmallData() {
    for (size_t i = 0; i < size_; ++i) {
      small_data_[i].~T();
    }
    size_ = 0;
  }

  // The number of elements in |small_data_| that have been constructed.
  size_t size_;

  // The pointed used to access the array of elements when the number of
  // elements is small.
  T* small_data_;

  // The actual data used to store the array elements.  It must never be used
  // directly, but must only be accessed through |small_data_|.
  typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type
      buffer[small_size];

  // A pointer to a vector that is used to store the elements of the vector when
  // this size exceeds |small_size|.  If |large_data_| is nullptr, then the data
  // is stored in |small_data_|.  Otherwise, the data is stored in
  // |large_data_|.
  std::unique_ptr<std::vector<T>> large_data_;
};  // namespace utils

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_SMALL_VECTOR_H_
