// Copyright 2018 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef UTIL_POD_ARRAY_H_
#define UTIL_POD_ARRAY_H_

#include <memory>
#include <type_traits>

namespace re2 {

template <typename T>
class PODArray {
 public:
  static_assert(std::is_pod<T>::value,
                "T must be POD");

  PODArray()
      : ptr_() {}
  explicit PODArray(int len)
      : ptr_(std::allocator<T>().allocate(len), Deleter(len)) {}

  T* data() const {
    return ptr_.get();
  }

  int size() const {
    return ptr_.get_deleter().len_;
  }

  T& operator[](int pos) const {
    return ptr_[pos];
  }

 private:
  struct Deleter {
    Deleter()
        : len_(0) {}
    explicit Deleter(int len)
        : len_(len) {}

    void operator()(T* ptr) const {
      std::allocator<T>().deallocate(ptr, len_);
    }

    int len_;
  };

  std::unique_ptr<T[], Deleter> ptr_;
};

}  // namespace re2

#endif  // UTIL_POD_ARRAY_H_
