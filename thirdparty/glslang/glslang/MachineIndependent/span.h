#pragma once

//
// Copyright (C) 2023 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

// Partial implementation of std::span for C++11
// Replace with std::span if repo standard is bumped to C++20
//
// This code was copied from https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/layers/containers/custom_containers.h
template <typename T>
class span {
  public:
    using pointer = T *;
    using const_pointer = T const *;
    using iterator = pointer;
    using const_iterator = const_pointer;

    span() = default;
    span(pointer start, size_t n) : data_(start), count_(n) {}
    template <typename Iterator>
    span(Iterator start, Iterator end) : data_(&(*start)), count_(end - start) {}
    template <typename Container>
    span(Container &c) : data_(c.data()), count_(c.size()) {}

    iterator begin() { return data_; }
    const_iterator begin() const { return data_; }

    iterator end() { return data_ + count_; }
    const_iterator end() const { return data_ + count_; }

    T &operator[](int i) { return data_[i]; }
    const T &operator[](int i) const { return data_[i]; }

    T &front() { return *data_; }
    const T &front() const { return *data_; }

    T &back() { return *(data_ + (count_ - 1)); }
    const T &back() const { return *(data_ + (count_ - 1)); }

    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

  private:
    pointer data_ = {};
    size_t count_ = 0;
};

//
// Allow type inference that using the constructor doesn't allow in C++11
template <typename T>
span<T> make_span(T *begin, size_t count) {
    return span<T>(begin, count);
}
template <typename T>
span<T> make_span(T *begin, T *end) {
    return make_span<T>(begin, end);
}
