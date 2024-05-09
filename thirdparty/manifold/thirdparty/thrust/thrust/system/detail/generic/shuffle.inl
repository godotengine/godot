/*
 *  Copyright 2008-20120 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/system/detail/generic/shuffle.h>

#include <cstdint>

THRUST_NAMESPACE_BEGIN
namespace system {
namespace detail {
namespace generic {

// An implementation of a Feistel cipher for operating on 64 bit keys
class feistel_bijection {
  struct round_state {
    std::uint32_t left;
    std::uint32_t right;
  };

 public:
  template <class URBG>
  __host__ __device__ feistel_bijection(std::uint64_t m, URBG&& g) {
    std::uint64_t total_bits = get_cipher_bits(m);
    // Half bits rounded down
    left_side_bits = total_bits / 2;
    left_side_mask = (1ull << left_side_bits) - 1;
    // Half the bits rounded up
    right_side_bits = total_bits - left_side_bits;
    right_side_mask = (1ull << right_side_bits) - 1;

    for (std::uint32_t i = 0; i < num_rounds; i++) {
      key[i] = g();
    }
  }

  __host__ __device__ std::uint64_t nearest_power_of_two() const {
    return 1ull << (left_side_bits + right_side_bits);
  }

  __host__ __device__ std::uint64_t operator()(const std::uint64_t val) const {
    std::uint32_t state[2] = { static_cast<std::uint32_t>( val >> right_side_bits ), static_cast<std::uint32_t>( val & right_side_mask ) };
    for( std::uint32_t i = 0; i < num_rounds; i++ )
    {
        std::uint32_t hi, lo;
        constexpr std::uint64_t M0 = UINT64_C( 0xD2B74407B1CE6E93 );
        mulhilo( M0, state[0], hi, lo );
        lo = ( lo << ( right_side_bits - left_side_bits ) ) | state[1] >> left_side_bits;
        state[0] = ( ( hi ^ key[i] ) ^ state[1] ) & left_side_mask;
        state[1] = lo & right_side_mask;
    }
    // Combine the left and right sides together to get result
    return static_cast<std::uint64_t>(state[0] << right_side_bits) | static_cast<std::uint64_t>(state[1]);
  }

 private:
   // Perform 64 bit multiplication and save result in two 32 bit int
   static __host__ __device__ void mulhilo( std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo )
   {
       std::uint64_t product = a * b;
       hi = static_cast<std::uint32_t>( product >> 32 );
       lo = static_cast<std::uint32_t>( product );
   }

  // Find the nearest power of two
  static __host__ __device__ std::uint64_t get_cipher_bits(std::uint64_t m) {
    if (m <= 16) return 4;
    std::uint64_t i = 0;
    m--;
    while (m != 0) {
      i++;
      m >>= 1;
    }
    return i;
  }

  static constexpr std::uint32_t num_rounds = 24;
  std::uint64_t right_side_bits;
  std::uint64_t left_side_bits;
  std::uint64_t right_side_mask;
  std::uint64_t left_side_mask;
  std::uint32_t key[num_rounds];
};

struct key_flag_tuple {
  std::uint64_t key;
  std::uint64_t flag;
};

// scan only flags
struct key_flag_scan_op {
  __host__ __device__ key_flag_tuple operator()(const key_flag_tuple& a,
                                                const key_flag_tuple& b) {
    return {b.key, a.flag + b.flag};
  }
};

struct construct_key_flag_op {
  std::uint64_t m;
  feistel_bijection bijection;
  __host__ __device__ construct_key_flag_op(std::uint64_t m,
                                            feistel_bijection bijection)
      : m(m), bijection(bijection) {}
  __host__ __device__ key_flag_tuple operator()(std::uint64_t idx) {
    auto gather_key = bijection(idx);
    return key_flag_tuple{gather_key, (gather_key < m) ? 1ull : 0ull};
  }
};

template <typename InputIterT, typename OutputIterT>
struct write_output_op {
  std::uint64_t m;
  InputIterT in;
  OutputIterT out;
  // flag contains inclusive scan of valid keys
  // perform gather using valid keys
  __thrust_exec_check_disable__
  __host__ __device__ std::size_t operator()(key_flag_tuple x) {
    if (x.key < m) {
      // -1 because inclusive scan
      out[x.flag - 1] = in[x.key];
    }
    return 0;  // Discarded
  }
};

template <typename ExecutionPolicy, typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(
    thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first,
    RandomIterator last, URBG&& g) {
  using InputType = typename thrust::iterator_value_t<RandomIterator>;

  // copy input to temp buffer
  thrust::detail::temporary_array<InputType, ExecutionPolicy> temp(exec, first,
                                                                   last);
  thrust::shuffle_copy(exec, temp.begin(), temp.end(), first, g);
}

template <typename ExecutionPolicy, typename RandomIterator,
          typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(
    thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first,
    RandomIterator last, OutputIterator result, URBG&& g) {
  // m is the length of the input
  // we have an available bijection of length n via a feistel cipher
  std::size_t m = last - first;
  feistel_bijection bijection(m, g);
  std::uint64_t n = bijection.nearest_power_of_two();

  // perform stream compaction over length n bijection to get length m
  // pseudorandom bijection over the original input
  thrust::counting_iterator<std::uint64_t> indices(0);
  thrust::transform_iterator<construct_key_flag_op, decltype(indices),
                             key_flag_tuple>
      key_flag_it(indices, construct_key_flag_op(m, bijection));
  write_output_op<RandomIterator, decltype(result)> write_functor{m, first,
                                                                  result};
  auto gather_output_it = thrust::make_transform_output_iterator(
      thrust::discard_iterator<std::size_t>(), write_functor);
  // the feistel_bijection outputs a stream of permuted indices in range [0,n)
  // flag each value < m and compact it, so we have a set of permuted indices in
  // range [0,m) each thread gathers an input element according to its
  // pseudorandom permuted index
  thrust::inclusive_scan(exec, key_flag_it, key_flag_it + n, gather_output_it,
                         key_flag_scan_op());
}

}  // end namespace generic
}  // end namespace detail
}  // end namespace system
THRUST_NAMESPACE_END
