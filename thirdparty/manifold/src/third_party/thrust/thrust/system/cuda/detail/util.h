/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights meserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <cstdio>
#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#include <cub/detail/device_synchronize.cuh>
#include <cub/util_arch.cuh>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {

inline __host__ __device__
cudaStream_t
default_stream()
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return cudaStreamPerThread;
#else
  return cudaStreamLegacy;
#endif
}

// Fallback implementation of the customization point.
template <class Derived>
__host__ __device__
cudaStream_t
get_stream(execution_policy<Derived> &)
{
  return default_stream();
}

// Entry point/interface.
template <class Derived>
__host__ __device__ cudaStream_t
stream(execution_policy<Derived> &policy)
{
  return get_stream(derived_cast(policy));
}


// Fallback implementation of the customization point.
template <class Derived>
__host__ __device__
bool
must_perform_optional_stream_synchronization(execution_policy<Derived> &)
{
  return true;
}

// Entry point/interface.
template <class Derived>
__host__ __device__ bool
must_perform_optional_synchronization(execution_policy<Derived> &policy)
{
  return must_perform_optional_stream_synchronization(derived_cast(policy));
}


// Fallback implementation of the customization point.
__thrust_exec_check_disable__
template <class Derived>
__host__ __device__
cudaError_t
synchronize_stream(execution_policy<Derived> &policy)
{
  cudaError_t result;
  if (THRUST_IS_HOST_CODE) {
    #if THRUST_INCLUDE_HOST_CODE
      cudaStreamSynchronize(stream(policy));
      result = cudaGetLastError();
    #endif
  } else {
    #if THRUST_INCLUDE_DEVICE_CODE
      #if __THRUST_HAS_CUDART__
        THRUST_UNUSED_VAR(policy);
        cub::detail::device_synchronize();
        result = cudaGetLastError();
      #else
        THRUST_UNUSED_VAR(policy);
        result = cudaSuccess;
      #endif
    #endif
  }
  return result;
}

// Entry point/interface.
template <class Policy>
__host__ __device__
cudaError_t
synchronize(Policy &policy)
{
  return synchronize_stream(derived_cast(policy));
}

// Fallback implementation of the customization point.
__thrust_exec_check_disable__
template <class Derived>
__host__ __device__
cudaError_t
synchronize_stream_optional(execution_policy<Derived> &policy)
{
  cudaError_t result;
  if (THRUST_IS_HOST_CODE) {
    #if THRUST_INCLUDE_HOST_CODE
      if(must_perform_optional_synchronization(policy)){
        cudaStreamSynchronize(stream(policy));
        result = cudaGetLastError();
      }else{
        result = cudaSuccess;
      }
    #endif
  } else {
    #if THRUST_INCLUDE_DEVICE_CODE
      #if __THRUST_HAS_CUDART__
        if(must_perform_optional_synchronization(policy)){
          cub::detail::device_synchronize();
          result = cudaGetLastError();
        }else{
          result = cudaSuccess;
        }
      #else
        THRUST_UNUSED_VAR(policy);
        result = cudaSuccess;
      #endif
    #endif
  }
  return result;
}

// Entry point/interface.
template <class Policy>
__host__ __device__
cudaError_t
synchronize_optional(Policy &policy)
{
  return synchronize_stream_optional(derived_cast(policy));
}

template <class Type>
THRUST_HOST_FUNCTION cudaError_t
trivial_copy_from_device(Type *       dst,
                         Type const * src,
                         size_t       count,
                         cudaStream_t stream)
{
  cudaError status = cudaSuccess;
  if (count == 0) return status;

  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyDeviceToHost,
                             stream);
  cudaStreamSynchronize(stream);
  return status;
}

template <class Type>
THRUST_HOST_FUNCTION cudaError_t
trivial_copy_to_device(Type *       dst,
                       Type const * src,
                       size_t       count,
                       cudaStream_t stream)
{
  cudaError status = cudaSuccess;
  if (count == 0) return status;

  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyHostToDevice,
                             stream);
  cudaStreamSynchronize(stream);
  return status;
}

template <class Policy, class Type>
__host__ __device__ cudaError_t
trivial_copy_device_to_device(Policy &    policy,
                              Type *      dst,
                              Type const *src,
                              size_t      count)
{
  cudaError_t  status = cudaSuccess;
  if (count == 0) return status;

  cudaStream_t stream = cuda_cub::stream(policy);
  //
  status = ::cudaMemcpyAsync(dst,
                             src,
                             sizeof(Type) * count,
                             cudaMemcpyDeviceToDevice,
                             stream);
  cuda_cub::synchronize(policy);
  return status;
}

inline void __host__ __device__
terminate()
{
  if (THRUST_IS_DEVICE_CODE) {
    #if THRUST_INCLUDE_DEVICE_CODE
      asm("trap;");
    #endif
  } else {
    #if THRUST_INCLUDE_HOST_CODE
      std::terminate();
    #endif
  }
}

__host__  __device__
inline void throw_on_error(cudaError_t status)
{
#if __THRUST_HAS_CUDART__
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
  cudaGetLastError();
#endif

  if (cudaSuccess != status)
  {
    if (THRUST_IS_HOST_CODE) {
      #if THRUST_INCLUDE_HOST_CODE
        throw thrust::system_error(status, thrust::cuda_category());
      #endif
    } else {
      #if THRUST_INCLUDE_DEVICE_CODE
        #if __THRUST_HAS_CUDART__
          printf("Thrust CUDA backend error: %s: %s\n",
                 cudaGetErrorName(status),
                 cudaGetErrorString(status));
        #else
          printf("Thrust CUDA backend error: %d\n",
                 static_cast<int>(status));
        #endif
        cuda_cub::terminate();
      #endif
    }
  }
}

__host__ __device__
inline void throw_on_error(cudaError_t status, char const *msg)
{
#if __THRUST_HAS_CUDART__
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.
  cudaGetLastError();
#endif

  if (cudaSuccess != status)
  {
    if (THRUST_IS_HOST_CODE) {
      #if THRUST_INCLUDE_HOST_CODE
        throw thrust::system_error(status, thrust::cuda_category(), msg);
      #endif
    } else {
      #if THRUST_INCLUDE_DEVICE_CODE
        #if __THRUST_HAS_CUDART__
          printf("Thrust CUDA backend error: %s: %s: %s\n",
                 cudaGetErrorName(status),
                 cudaGetErrorString(status),
                 msg);
        #else
          printf("Thrust CUDA backend error: %d: %s \n",
                 static_cast<int>(status),
                 msg);
        #endif
        cuda_cub::terminate();
      #endif
    }
  }
}

// FIXME: Move the iterators elsewhere.

template <class ValueType,
          class InputIt,
          class UnaryOp>
struct transform_input_iterator_t
{
  typedef transform_input_iterator_t                         self_t;
  typedef typename iterator_traits<InputIt>::difference_type difference_type;
  typedef ValueType                                          value_type;
  typedef void                                               pointer;
  typedef value_type                                         reference;
  typedef std::random_access_iterator_tag                    iterator_category;

  InputIt         input;
  mutable UnaryOp op;

  __host__ __device__ __forceinline__
  transform_input_iterator_t(InputIt input, UnaryOp op)
      : input(input), op(op) {}

#if THRUST_CPP_DIALECT >= 2011
  transform_input_iterator_t(const self_t &) = default;
#endif

  // UnaryOp might not be copy assignable, such as when it is a lambda.  Define
  // an explicit copy assignment operator that doesn't try to assign it.
  self_t& operator=(const self_t& o)
  {
    input = o.input;
    return *this;
  }

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++input;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    typename thrust::iterator_value<InputIt>::type x = *input;
    return op(x);
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    typename thrust::iterator_value<InputIt>::type x = *input;
    return op(x);
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input + n, op);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    input += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input - n, op);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    input -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return input - other.input;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return op(input[n]);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input == rhs.input);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input != rhs.input);
  }
};    // struct transform_input_iterarot_t

template <class ValueType,
          class InputIt1,
          class InputIt2,
          class BinaryOp>
struct transform_pair_of_input_iterators_t
{
  typedef transform_pair_of_input_iterators_t                 self_t;
  typedef typename iterator_traits<InputIt1>::difference_type difference_type;
  typedef ValueType                                           value_type;
  typedef void                                                pointer;
  typedef value_type                                          reference;
  typedef std::random_access_iterator_tag                     iterator_category;

  InputIt1         input1;
  InputIt2         input2;
  mutable BinaryOp op;

  __host__ __device__ __forceinline__
  transform_pair_of_input_iterators_t(InputIt1 input1_,
                                      InputIt2 input2_,
                                      BinaryOp op_)
      : input1(input1_), input2(input2_), op(op_) {}

#if THRUST_CPP_DIALECT >= 2011
  transform_pair_of_input_iterators_t(const self_t &) = default;
#endif

  // BinaryOp might not be copy assignable, such as when it is a lambda.
  // Define an explicit copy assignment operator that doesn't try to assign it.
  self_t& operator=(const self_t& o)
  {
    input1 = o.input1;
    input2 = o.input2;
    return *this;
  }

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++input1;
    ++input2;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++input1;
    ++input2;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return op(*input1, *input2);
  }
  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return op(*input1, *input2);
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(input1 + n, input2 + n, op);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    input1 += n;
    input2 += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(input1 - n, input2 - n, op);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    input1 -= n;
    input2 -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return input1 - other.input1;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return op(input1[n], input2[n]);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (input1 == rhs.input1) && (input2 == rhs.input2);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (input1 != rhs.input1) || (input2 != rhs.input2);
  }

};    // struct transform_pair_of_input_iterators_t


struct identity
{
  template <class T>
  __host__ __device__ T const &
  operator()(T const &t) const
  {
    return t;
  }

  template <class T>
  __host__ __device__ T &
  operator()(T &t) const
  {
    return t;
  }
};


template <class T>
struct counting_iterator_t
{
  typedef counting_iterator_t             self_t;
  typedef T                               difference_type;
  typedef T                               value_type;
  typedef void                            pointer;
  typedef T                               reference;
  typedef std::random_access_iterator_tag iterator_category;

  T count;

  __host__ __device__ __forceinline__
  counting_iterator_t(T count_) : count(count_) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_t operator++(int)
  {
    self_t retval = *this;
    ++count;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_t operator++()
  {
    ++count;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const
  {
    return count;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*()
  {
    return count;
  }

  /// Addition
  __host__ __device__ __forceinline__ self_t operator+(difference_type n) const
  {
    return self_t(count + n);
  }

  /// Addition assignment
  __host__ __device__ __forceinline__ self_t &operator+=(difference_type n)
  {
    count += n;
    return *this;
  }

  /// Subtraction
  __host__ __device__ __forceinline__ self_t operator-(difference_type n) const
  {
    return self_t(count - n);
  }

  /// Subtraction assignment
  __host__ __device__ __forceinline__ self_t &operator-=(difference_type n)
  {
    count -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_t other) const
  {
    return count - other.count;
  }

  /// Array subscript
  __host__ __device__ __forceinline__ reference operator[](difference_type n) const
  {
    return count + n;
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_t &rhs) const
  {
    return (count == rhs.count);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_t &rhs) const
  {
    return (count != rhs.count);
  }

};    // struct count_iterator_t

}    // cuda_

THRUST_NAMESPACE_END
