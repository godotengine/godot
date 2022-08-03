/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// TODO: This can probably be removed.

#pragma once

#include <thrust/detail/config.h>

#include <thrust/system/cuda/detail/util.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {
namespace alignment_of_detail {


  template <typename T>
  class alignment_of_impl;

  template <typename T, std::size_t size_diff>
  struct helper
  {
    static const std::size_t value = size_diff;
  };

  template <typename T>
  class helper<T, 0>
  {
  public:
    static const std::size_t value = alignment_of_impl<T>::value;
  };

  template <typename T>
  class alignment_of_impl
  {
  private:
    struct big
    {
      T    x;
      char c;
    };

  public:
    static const std::size_t value = helper<big, sizeof(big) - sizeof(T)>::value;
  };


}    // end alignment_of_detail


template <typename T>
struct alignment_of
    : alignment_of_detail::alignment_of_impl<T>
{
};


template <std::size_t Align>
struct aligned_type;

// __align__ is CUDA-specific, so guard it
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

// implementing aligned_type portably is tricky:

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
// implement aligned_type with specialization because MSVC
// requires literals as arguments to declspec(align(n))
template <>
struct aligned_type<1>
{
  struct __align__(1) type{};
};

template <>
struct aligned_type<2>
{
  struct __align__(2) type{};
};

template <>
struct aligned_type<4>
{
  struct __align__(4) type{};
};

template <>
struct aligned_type<8>
{
  struct __align__(8) type{};
};

template <>
struct aligned_type<16>
{
  struct __align__(16) type{};
};

template <>
struct aligned_type<32>
{
  struct __align__(32) type{};
};

template <>
struct aligned_type<64>
{
  struct __align__(64) type{};
};

template <>
struct aligned_type<128>
{
  struct __align__(128) type{};
};

template <>
struct aligned_type<256>
{
  struct __align__(256) type{};
};

template <>
struct aligned_type<512>
{
  struct __align__(512) type{};
};

template <>
struct aligned_type<1024>
{
  struct __align__(1024) type{};
};

template <>
struct aligned_type<2048>
{
  struct __align__(2048) type{};
};

template <>
struct aligned_type<4096>
{
  struct __align__(4096) type{};
};

template <>
struct aligned_type<8192>
{
  struct __align__(8192) type{};
};
#elif (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) && (THRUST_GCC_VERSION < 40300)
// implement aligned_type with specialization because gcc 4.2
// requires literals as arguments to __attribute__(aligned(n))
template <>
struct aligned_type<1>
{
  struct __align__(1) type{};
};

template <>
struct aligned_type<2>
{
  struct __align__(2) type{};
};

template <>
struct aligned_type<4>
{
  struct __align__(4) type{};
};

template <>
struct aligned_type<8>
{
  struct __align__(8) type{};
};

template <>
struct aligned_type<16>
{
  struct __align__(16) type{};
};

template <>
struct aligned_type<32>
{
  struct __align__(32) type{};
};

template <>
struct aligned_type<64>
{
  struct __align__(64) type{};
};

template <>
struct aligned_type<128>
{
  struct __align__(128) type{};
};

#else
// assume the compiler allows template parameters as
// arguments to __align__
template <std::size_t Align>
struct aligned_type
{
  struct __align__(Align) type{};
};
#endif    // THRUST_HOST_COMPILER
#else
template <std::size_t Align>
struct aligned_type
{
  struct type
  {
  };
};
#endif    // THRUST_DEVICE_COMPILER


template <std::size_t Len, std::size_t Align>
struct aligned_storage
{
  union type
  {
    unsigned char data[Len];

    typename aligned_type<Align>::type align;
  };
};


}    // end cuda_

THRUST_NAMESPACE_END
