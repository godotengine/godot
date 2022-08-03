/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/cstdint.h>
#include <thrust/scatter.h>

#include <limits>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{
namespace radix_sort_detail
{


template <typename T>
struct RadixEncoder : public thrust::identity<T>
{};


template <>
struct RadixEncoder<char> : public thrust::unary_function<char, unsigned char>
{
  __host__ __device__
  unsigned char operator()(char x) const
  {
    if(std::numeric_limits<char>::is_signed)
    {
      return static_cast<unsigned char>(x) ^ static_cast<unsigned char>(1) << (8 * sizeof(unsigned char) - 1);
    }
    else
    {
      return x;
    }
  }
};

template <>
struct RadixEncoder<signed char> : public thrust::unary_function<signed char, unsigned char>
{
  __host__ __device__
  unsigned char operator()(signed char x) const
  {
    return static_cast<unsigned char>(x) ^ static_cast<unsigned char>(1) << (8 * sizeof(unsigned char) - 1);
  }
};

template <>
struct RadixEncoder<short> : public thrust::unary_function<short, unsigned short>
{
  __host__ __device__
  unsigned short operator()(short x) const
  {
    return static_cast<unsigned short>(x) ^ static_cast<unsigned short>(1) << (8 * sizeof(unsigned short) - 1);
  }
};

template <>
struct RadixEncoder<int> : public thrust::unary_function<int, unsigned int>
{
  __host__ __device__
  unsigned long operator()(long x) const
  {
    return x ^ static_cast<unsigned int>(1) << (8 * sizeof(unsigned int) - 1);
  }
};

template <>
struct RadixEncoder<long> : public thrust::unary_function<long, unsigned long>
{
  __host__ __device__
  unsigned long operator()(long x) const
  {
    return x ^ static_cast<unsigned long>(1) << (8 * sizeof(unsigned long) - 1);
  }
};

template <>
struct RadixEncoder<long long> : public thrust::unary_function<long long, unsigned long long>
{
  __host__ __device__
  unsigned long long operator()(long long x) const
  {
    return x ^ static_cast<unsigned long long>(1) << (8 * sizeof(unsigned long long) - 1);
  }
};

// ideally we'd use uint32 here and uint64 below
template <>
struct RadixEncoder<float> : public thrust::unary_function<float, thrust::detail::uint32_t>
{
  __host__ __device__
  thrust::detail::uint32_t operator()(float x) const
  {
    union { float f; thrust::detail::uint32_t i; } u;
    u.f = x;
    thrust::detail::uint32_t mask = -static_cast<thrust::detail::int32_t>(u.i >> 31) | (static_cast<thrust::detail::uint32_t>(1) << 31);
    return u.i ^ mask;
  }
};

template <>
struct RadixEncoder<double> : public thrust::unary_function<double, thrust::detail::uint64_t>
{
  __host__ __device__
  thrust::detail::uint64_t operator()(double x) const
  {
    union { double f; thrust::detail::uint64_t i; } u;
    u.f = x;
    thrust::detail::uint64_t mask = -static_cast<thrust::detail::int64_t>(u.i >> 63) | (static_cast<thrust::detail::uint64_t>(1) << 63);
    return u.i ^ mask;
  }
};


// this functor returns a key's to its histogram bucket count and post-increments the bucket
template<unsigned int RadixBits, typename KeyType>
  struct bucket_functor
{
  typedef RadixEncoder<KeyType> Encoder;
  typedef typename Encoder::result_type EncodedType;
  typedef size_t result_type;
  static const EncodedType BitMask = static_cast<EncodedType>((1 << RadixBits) - 1);

  Encoder encode;
  EncodedType bit_shift;
  size_t *histogram;

  __host__ __device__
  bucket_functor(EncodedType bit_shift, size_t *histogram)
    : encode(),
      bit_shift(bit_shift),
      histogram(histogram)
  {}

  inline __host__ __device__
  size_t operator()(KeyType key)
  {
    const EncodedType x = encode(key);

    // note that we mutate the histogram here
    return histogram[(x >> bit_shift) & BitMask]++;
  }
};


template<unsigned int RadixBits,
         typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Integer>
inline __host__ __device__
void radix_shuffle_n(sequential::execution_policy<DerivedPolicy> &exec,
                     RandomAccessIterator1 first,
                     const size_t n,
                     RandomAccessIterator2 result,
                     Integer bit_shift,
                     size_t *histogram)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;

  // note that we are going to mutate the histogram during this sequential scatter
  thrust::scatter(exec,
                  first, first + n,
                  thrust::make_transform_iterator(first, bucket_functor<RadixBits,KeyType>(bit_shift, histogram)),
                  result);
}


template<unsigned int RadixBits,
         typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename Integer>
__host__ __device__
void radix_shuffle_n(sequential::execution_policy<DerivedPolicy> &exec,
                     RandomAccessIterator1 keys_first,
                     RandomAccessIterator2 values_first,
                     const size_t n,
                     RandomAccessIterator3 keys_result,
                     RandomAccessIterator4 values_result,
                     Integer bit_shift,
                     size_t *histogram)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;

  // note that we are going to mutate the histogram during this sequential scatter
  thrust::scatter(exec,
                  thrust::make_zip_iterator(thrust::make_tuple(keys_first, values_first)),
                  thrust::make_zip_iterator(thrust::make_tuple(keys_first + n, values_first + n)),
                  thrust::make_transform_iterator(keys_first, bucket_functor<RadixBits,KeyType>(bit_shift, histogram)),
                  thrust::make_zip_iterator(thrust::make_tuple(keys_result, values_result)));
}


template<unsigned int RadixBits,
         bool HasValues,
         typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__host__ __device__
void radix_sort(sequential::execution_policy<DerivedPolicy> &exec,
                RandomAccessIterator1 keys1,
                RandomAccessIterator2 keys2,
                RandomAccessIterator3 vals1,
                RandomAccessIterator4 vals2,
                const size_t N)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;

  typedef RadixEncoder<KeyType> Encoder;
  typedef typename Encoder::result_type EncodedType;

  const unsigned int NumHistograms = (8 * sizeof(EncodedType) + (RadixBits - 1)) / RadixBits;
  const unsigned int HistogramSize =  1 << RadixBits;

  const EncodedType BitMask = static_cast<EncodedType>((1 << RadixBits) - 1);

  Encoder encode;

  // storage for histograms
  size_t histograms[NumHistograms][HistogramSize] = {{0}};

  // see which passes can be eliminated
  bool skip_shuffle[NumHistograms] = {false};

  // false if most recent data is stored in (keys1,vals1)
  bool flip = false;

  // compute histograms
  for(size_t i = 0; i < N; i++)
  {
    const EncodedType x = encode(keys1[i]);

    for(unsigned int j = 0; j < NumHistograms; j++)
    {
      const auto BitShift = static_cast<EncodedType>(RadixBits * j);
      histograms[j][(x >> BitShift) & BitMask]++;
    }
  }

  // scan histograms
  for(unsigned int i = 0; i < NumHistograms; i++)
  {
    size_t sum = 0;

    for(unsigned int j = 0; j < HistogramSize; j++)
    {
      size_t bin = histograms[i][j];

      if(bin == N)
        skip_shuffle[i] = true;

      histograms[i][j] = sum;

      sum = sum + bin;
    }
  }

  // shuffle keys and (optionally) values
  for(unsigned int i = 0; i < NumHistograms; i++)
  {
    const EncodedType BitShift = static_cast<EncodedType>(RadixBits * i);

    if(!skip_shuffle[i])
    {
      if(flip)
      {
        if(HasValues)
        {
          radix_shuffle_n<RadixBits>(exec, keys2, vals2, N, keys1, vals1, BitShift, histograms[i]);
        }
        else
        {
          radix_shuffle_n<RadixBits>(exec, keys2, N, keys1, BitShift, histograms[i]);
        }
      }
      else
      {
        if(HasValues)
        {
          radix_shuffle_n<RadixBits>(exec, keys1, vals1, N, keys2, vals2, BitShift, histograms[i]);
        }
        else
        {
          radix_shuffle_n<RadixBits>(exec, keys1, N, keys2, BitShift, histograms[i]);
        }
      }

      flip = (flip) ? false : true;
    }
  }

  // ensure final values are in (keys1,vals1)
  if(flip)
  {
    thrust::copy(exec, keys2, keys2 + N, keys1);

    if(HasValues)
    {
      thrust::copy(exec, vals2, vals2 + N, vals1);
    }
  }
}


// Select best radix sort parameters based on sizeof(T) and input size
// These particular values were determined through empirical testing on a Core i7 950 CPU
template <size_t KeySize>
struct radix_sort_dispatcher
{
};

template <>
struct radix_sort_dispatcher<1>
{
  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  const size_t N)
  {
    radix_sort_detail::radix_sort<8,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
  }

  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  RandomAccessIterator3 vals1, RandomAccessIterator4 vals2,
                  const size_t N)
  {
    radix_sort_detail::radix_sort<8,true>(exec, keys1, keys2, vals1, vals2, N);
  }
};


template <>
struct radix_sort_dispatcher<2>
{
  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  const size_t N)
  {
#ifdef __QNX__
    // XXX war for nvbug 200193674
    const bool condition = true;
#else
    const bool condition = N < (1 << 16);
#endif
    if (condition)
    {
      radix_sort_detail::radix_sort<8,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    }
    else
    {
      radix_sort_detail::radix_sort<16,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    }
  }


  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  RandomAccessIterator3 vals1, RandomAccessIterator4 vals2,
                  const size_t N)
  {
#ifdef __QNX__
    // XXX war for nvbug 200193674
    const bool condition = true;
#else
    const bool condition = N < (1 << 15);
#endif
    if (condition)
    {
      radix_sort_detail::radix_sort<8,true>(exec, keys1, keys2, vals1, vals2, N);
    }
    else
    {
      radix_sort_detail::radix_sort<16,true>(exec, keys1, keys2, vals1, vals2, N);
    }
  }
};


template <>
struct radix_sort_dispatcher<4>
{
  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  const size_t N)
  {
    if(N < (1 << 22))
    {
      radix_sort_detail::radix_sort<8,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    }
    else
    {
      radix_sort_detail::radix_sort<4,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    }
  }

  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  RandomAccessIterator3 vals1, RandomAccessIterator4 vals2,
                  const size_t N)
  {
    if(N < (1 << 22))
    {
      radix_sort_detail::radix_sort<8,true>(exec, keys1, keys2, vals1, vals2, N);
    }
    else
    {
      radix_sort_detail::radix_sort<3,true>(exec, keys1, keys2, vals1, vals2, N);
    }
  }
};


template <>
struct radix_sort_dispatcher<8>
{
  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  const size_t N)
  {
    if(N < (1 << 21))
    {
      radix_sort_detail::radix_sort<8,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    }
    else
    {
      radix_sort_detail::radix_sort<4,false>(exec, keys1, keys2, static_cast<int *>(0), static_cast<int *>(0), N);
    }
  }

  template<typename DerivedPolicy,
           typename RandomAccessIterator1,
           typename RandomAccessIterator2,
           typename RandomAccessIterator3,
           typename RandomAccessIterator4>
  __host__ __device__
  void operator()(sequential::execution_policy<DerivedPolicy> &exec,
                  RandomAccessIterator1 keys1, RandomAccessIterator2 keys2,
                  RandomAccessIterator3 vals1, RandomAccessIterator4 vals2,
                  const size_t N)
  {
    if(N < (1 << 21))
    {
      radix_sort_detail::radix_sort<8,true>(exec, keys1, keys2, vals1, vals2, N);
    }
    else
    {
      radix_sort_detail::radix_sort<3,true>(exec, keys1, keys2, vals1, vals2, N);
    }
  }
};


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void radix_sort(sequential::execution_policy<DerivedPolicy> &exec,
                RandomAccessIterator1 keys1,
                RandomAccessIterator2 keys2,
                const size_t N)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  radix_sort_dispatcher<sizeof(KeyType)>()(exec, keys1, keys2, N);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4>
__host__ __device__
void radix_sort(sequential::execution_policy<DerivedPolicy> &exec,
                RandomAccessIterator1 keys1,
                RandomAccessIterator2 keys2,
                RandomAccessIterator3 vals1,
                RandomAccessIterator4 vals2,
                const size_t N)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  radix_sort_dispatcher<sizeof(KeyType)>()(exec, keys1, keys2, vals1, vals2, N);
}


} // namespace radix_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
void stable_radix_sort(sequential::execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type KeyType;

  size_t N = last - first;

  thrust::detail::temporary_array<KeyType, DerivedPolicy> temp(exec, N);

  radix_sort_detail::radix_sort(exec, first, temp.begin(), N);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
void stable_radix_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type ValueType;

  size_t N = last1 - first1;

  thrust::detail::temporary_array<KeyType, DerivedPolicy>   temp1(exec, N);
  thrust::detail::temporary_array<ValueType, DerivedPolicy> temp2(exec, N);

  radix_sort_detail::radix_sort(exec, first1, temp1.begin(), first2, temp2.begin(), N);
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

