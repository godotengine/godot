#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <assert.h>

#include "include/timer.h"

// This examples compares sorting performance using Array of Structures (AoS)
// and Structure of Arrays (SoA) data layout.  Legacy applications will often
// store data in C/C++ structs, such as MyStruct defined below.  Although
// Thrust can process array of structs, it is typically less efficient than
// the equivalent structure of arrays layout.  In this particular example,
// the optimized SoA approach is approximately *five times faster* than the
// traditional AoS method.  Therefore, it is almost always worthwhile to
// convert AoS data structures to SoA.

struct MyStruct
{
  int key;
  float value;

  __host__ __device__
    bool operator<(const MyStruct other) const
    {
      return key < other.key;
    }
};

void initialize_keys(thrust::device_vector<int>& keys)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<int> h_keys(keys.size());

  for(size_t i = 0; i < h_keys.size(); i++)
    h_keys[i] = dist(rng);

  keys = h_keys;
}


void initialize_keys(thrust::device_vector<MyStruct>& structures)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 2147483647);

  thrust::host_vector<MyStruct> h_structures(structures.size());

  for(size_t i = 0; i < h_structures.size(); i++)
    h_structures[i].key = dist(rng);

  structures = h_structures;
}

int main(void)
{
  size_t N = 2 * 1024 * 1024;

  // Sort Key-Value pairs using Array of Structures (AoS) storage
  {
    thrust::device_vector<MyStruct> structures(N);

    initialize_keys(structures);

    timer t;

    thrust::sort(structures.begin(), structures.end());
    assert(thrust::is_sorted(structures.begin(), structures.end()));

    std::cout << "AoS sort took " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  // Sort Key-Value pairs using Structure of Arrays (SoA) storage
  {
    thrust::device_vector<int>   keys(N);
    thrust::device_vector<float> values(N);

    initialize_keys(keys);

    timer t;

    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    assert(thrust::is_sorted(keys.begin(), keys.end()));

    std::cout << "SoA sort took " << 1e3 * t.elapsed() << " milliseconds" << std::endl;
  }

  return 0;
}

