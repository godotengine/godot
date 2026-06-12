/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_BUFFER_H_
#define VPX_TEST_BUFFER_H_

#include <stdio.h>

#include <limits>

#include "gtest/gtest.h"

#include "test/acm_random.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

namespace libvpx_test {

template <typename T>
class Buffer {
 public:
  Buffer(int width, int height, int top_padding, int left_padding,
         int right_padding, int bottom_padding)
      : width_(width), height_(height), top_padding_(top_padding),
        left_padding_(left_padding), right_padding_(right_padding),
        bottom_padding_(bottom_padding), alignment_(0), padding_value_(0),
        stride_(0), raw_size_(0), num_elements_(0), raw_buffer_(nullptr) {}

  Buffer(int width, int height, int top_padding, int left_padding,
         int right_padding, int bottom_padding, unsigned int alignment)
      : width_(width), height_(height), top_padding_(top_padding),
        left_padding_(left_padding), right_padding_(right_padding),
        bottom_padding_(bottom_padding), alignment_(alignment),
        padding_value_(0), stride_(0), raw_size_(0), num_elements_(0),
        raw_buffer_(nullptr) {}

  Buffer(int width, int height, int padding)
      : width_(width), height_(height), top_padding_(padding),
        left_padding_(padding), right_padding_(padding),
        bottom_padding_(padding), alignment_(0), padding_value_(0), stride_(0),
        raw_size_(0), num_elements_(0), raw_buffer_(nullptr) {}

  Buffer(int width, int height, int padding, unsigned int alignment)
      : width_(width), height_(height), top_padding_(padding),
        left_padding_(padding), right_padding_(padding),
        bottom_padding_(padding), alignment_(alignment), padding_value_(0),
        stride_(0), raw_size_(0), num_elements_(0), raw_buffer_(nullptr) {}

  ~Buffer() {
    if (alignment_) {
      vpx_free(raw_buffer_);
    } else {
      delete[] raw_buffer_;
    }
  }

  T *TopLeftPixel() const;

  int stride() const { return stride_; }

  // Set the buffer (excluding padding) to 'value'.
  void Set(const T value);

  // Set the buffer (excluding padding) to the output of ACMRandom function
  // 'rand_func'.
  void Set(ACMRandom *rand_class, T (ACMRandom::*rand_func)());

  // Set the buffer (excluding padding) to the output of ACMRandom function
  // 'RandRange' with range 'low' to 'high' which typically must be within
  // testing::internal::Random::kMaxRange (1u << 31). However, because we want
  // to allow negative low (and high) values, it is restricted to INT32_MAX
  // here.
  void Set(ACMRandom *rand_class, const T low, const T high);

  // Copy the contents of Buffer 'a' (excluding padding).
  void CopyFrom(const Buffer<T> &a);

  void DumpBuffer() const;

  // Highlight the differences between two buffers if they are the same size.
  void PrintDifference(const Buffer<T> &a) const;

  bool HasPadding() const;

  // Sets all the values in the buffer to 'padding_value'.
  void SetPadding(const T padding_value);

  // Checks if all the values (excluding padding) are equal to 'value' if the
  // Buffers are the same size.
  bool CheckValues(const T value) const;

  // Check that padding matches the expected value or there is no padding.
  bool CheckPadding() const;

  // Compare the non-padding portion of two buffers if they are the same size.
  bool CheckValues(const Buffer<T> &a) const;

  bool Init() {
    if (raw_buffer_ != nullptr) return false;
    EXPECT_GT(width_, 0);
    EXPECT_GT(height_, 0);
    EXPECT_GE(top_padding_, 0);
    EXPECT_GE(left_padding_, 0);
    EXPECT_GE(right_padding_, 0);
    EXPECT_GE(bottom_padding_, 0);
    stride_ = left_padding_ + width_ + right_padding_;
    num_elements_ = stride_ * (top_padding_ + height_ + bottom_padding_);
    raw_size_ = num_elements_ * sizeof(T);
    if (alignment_) {
      EXPECT_GE(alignment_, sizeof(T));
      // Ensure alignment of the first value will be preserved.
      EXPECT_EQ((left_padding_ * sizeof(T)) % alignment_, 0u);
      // Ensure alignment of the subsequent rows will be preserved when there is
      // a stride.
      if (stride_ != width_) {
        EXPECT_EQ((stride_ * sizeof(T)) % alignment_, 0u);
      }
      raw_buffer_ = reinterpret_cast<T *>(vpx_memalign(alignment_, raw_size_));
    } else {
      raw_buffer_ = new (std::nothrow) T[num_elements_];
    }
    EXPECT_NE(raw_buffer_, nullptr);
    SetPadding(std::numeric_limits<T>::max());
    return !::testing::Test::HasFailure();
  }

 private:
  bool BufferSizesMatch(const Buffer<T> &a) const;

  const int width_;
  const int height_;
  const int top_padding_;
  const int left_padding_;
  const int right_padding_;
  const int bottom_padding_;
  const unsigned int alignment_;
  T padding_value_;
  int stride_;
  int raw_size_;
  int num_elements_;
  T *raw_buffer_;
};

template <typename T>
T *Buffer<T>::TopLeftPixel() const {
  if (!raw_buffer_) return nullptr;
  return raw_buffer_ + (top_padding_ * stride_) + left_padding_;
}

template <typename T>
void Buffer<T>::Set(const T value) {
  if (!raw_buffer_) return;
  T *src = TopLeftPixel();
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      src[width] = value;
    }
    src += stride_;
  }
}

template <typename T>
void Buffer<T>::Set(ACMRandom *rand_class, T (ACMRandom::*rand_func)()) {
  if (!raw_buffer_) return;
  T *src = TopLeftPixel();
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      src[width] = (*rand_class.*rand_func)();
    }
    src += stride_;
  }
}

template <typename T>
void Buffer<T>::Set(ACMRandom *rand_class, const T low, const T high) {
  if (!raw_buffer_) return;

  EXPECT_LE(low, high);
  EXPECT_LE(static_cast<int64_t>(high) - low,
            std::numeric_limits<int32_t>::max());

  T *src = TopLeftPixel();
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      // 'low' will be promoted to unsigned given the return type of RandRange.
      // Store the value as an int to avoid unsigned overflow warnings when
      // 'low' is negative.
      const int32_t value =
          static_cast<int32_t>((*rand_class).RandRange(high - low));
      src[width] = static_cast<T>(value + low);
    }
    src += stride_;
  }
}

template <typename T>
void Buffer<T>::CopyFrom(const Buffer<T> &a) {
  if (!raw_buffer_) return;
  if (!BufferSizesMatch(a)) return;

  T *a_src = a.TopLeftPixel();
  T *b_src = this->TopLeftPixel();
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      b_src[width] = a_src[width];
    }
    a_src += a.stride();
    b_src += this->stride();
  }
}

template <typename T>
void Buffer<T>::DumpBuffer() const {
  if (!raw_buffer_) return;
  for (int height = 0; height < height_ + top_padding_ + bottom_padding_;
       ++height) {
    for (int width = 0; width < stride_; ++width) {
      printf("%4d", raw_buffer_[height + width * stride_]);
    }
    printf("\n");
  }
}

template <typename T>
bool Buffer<T>::HasPadding() const {
  if (!raw_buffer_) return false;
  return top_padding_ || left_padding_ || right_padding_ || bottom_padding_;
}

template <typename T>
void Buffer<T>::PrintDifference(const Buffer<T> &a) const {
  if (!raw_buffer_) return;
  if (!BufferSizesMatch(a)) return;

  T *a_src = a.TopLeftPixel();
  T *b_src = TopLeftPixel();

  printf("This buffer:\n");
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      if (a_src[width] != b_src[width]) {
        printf("*%3d", b_src[width]);
      } else {
        printf("%4d", b_src[width]);
      }
    }
    printf("\n");
    a_src += a.stride();
    b_src += this->stride();
  }

  a_src = a.TopLeftPixel();
  b_src = TopLeftPixel();

  printf("Reference buffer:\n");
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      if (a_src[width] != b_src[width]) {
        printf("*%3d", a_src[width]);
      } else {
        printf("%4d", a_src[width]);
      }
    }
    printf("\n");
    a_src += a.stride();
    b_src += this->stride();
  }
}

template <typename T>
void Buffer<T>::SetPadding(const T padding_value) {
  if (!raw_buffer_) return;
  padding_value_ = padding_value;

  T *src = raw_buffer_;
  for (int i = 0; i < num_elements_; ++i) {
    src[i] = padding_value;
  }
}

template <typename T>
bool Buffer<T>::CheckValues(const T value) const {
  if (!raw_buffer_) return false;
  T *src = TopLeftPixel();
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      if (value != src[width]) {
        return false;
      }
    }
    src += stride_;
  }
  return true;
}

template <typename T>
bool Buffer<T>::CheckPadding() const {
  if (!raw_buffer_) return false;
  if (!HasPadding()) return true;

  // Top padding.
  T const *top = raw_buffer_;
  for (int i = 0; i < stride_ * top_padding_; ++i) {
    if (padding_value_ != top[i]) {
      return false;
    }
  }

  // Left padding.
  T const *left = TopLeftPixel() - left_padding_;
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < left_padding_; ++width) {
      if (padding_value_ != left[width]) {
        return false;
      }
    }
    left += stride_;
  }

  // Right padding.
  T const *right = TopLeftPixel() + width_;
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < right_padding_; ++width) {
      if (padding_value_ != right[width]) {
        return false;
      }
    }
    right += stride_;
  }

  // Bottom padding
  T const *bottom = raw_buffer_ + (top_padding_ + height_) * stride_;
  for (int i = 0; i < stride_ * bottom_padding_; ++i) {
    if (padding_value_ != bottom[i]) {
      return false;
    }
  }

  return true;
}

template <typename T>
bool Buffer<T>::CheckValues(const Buffer<T> &a) const {
  if (!raw_buffer_) return false;
  if (!BufferSizesMatch(a)) return false;

  T *a_src = a.TopLeftPixel();
  T *b_src = this->TopLeftPixel();
  for (int height = 0; height < height_; ++height) {
    for (int width = 0; width < width_; ++width) {
      if (a_src[width] != b_src[width]) {
        return false;
      }
    }
    a_src += a.stride();
    b_src += this->stride();
  }
  return true;
}

template <typename T>
bool Buffer<T>::BufferSizesMatch(const Buffer<T> &a) const {
  if (!raw_buffer_) return false;
  if (a.width_ != this->width_ || a.height_ != this->height_) {
    printf(
        "Reference buffer of size %dx%d does not match this buffer which is "
        "size %dx%d\n",
        a.width_, a.height_, this->width_, this->height_);
    return false;
  }

  return true;
}
}  // namespace libvpx_test
#endif  // VPX_TEST_BUFFER_H_
