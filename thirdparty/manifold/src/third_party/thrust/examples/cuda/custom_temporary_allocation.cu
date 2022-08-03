#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <cassert>

// This example demonstrates how to control how Thrust allocates temporary
// storage during algorithms such as thrust::sort. The idea will be to create a
// simple cache of allocations to search when temporary storage is requested.
// If a hit is found in the cache, we quickly return the cached allocation
// instead of resorting to the more expensive thrust::cuda::malloc.

// Note: Thrust now has its own caching allocator layer; if you just need a
// caching allocator, you ought to use that. This example is still useful
// as a demonstration of how to use a Thrust custom allocator.

// Note: this implementation cached_allocator is not thread-safe. If multiple
// (host) threads use the same cached_allocator then they should gain exclusive
// access to the allocator before accessing its methods.

struct not_my_pointer
{
  not_my_pointer(void* p)
    : message()
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  virtual ~not_my_pointer() {}

  virtual const char* what() const
  {
    return message.c_str();
  }

private:
  std::string message;
};

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator
{
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator()
  {
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    std::cout << "cached_allocator::allocate(): num_bytes == "
              << num_bytes
              << std::endl;

    char *result = 0;

    // Search the cache for a free block.
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end())
    {
      std::cout << "cached_allocator::allocate(): found a free block"
                << std::endl;

      result = free_block->second;

      // Erase from the `free_blocks` map.
      free_blocks.erase(free_block);
    }
    else
    {
      // No allocation of the right size exists, so create a new one with
      // `thrust::cuda::malloc`.
      try
      {
        std::cout << "cached_allocator::allocate(): allocating new block"
                  << std::endl;

        // Allocate memory and convert the resulting `thrust::cuda::pointer` to
        // a raw pointer.
        result = thrust::cuda::malloc<char>(num_bytes).get();
      }
      catch (std::runtime_error&)
      {
        throw;
      }
    }

    // Insert the allocated pointer into the `allocated_blocks` map.
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t)
  {
    std::cout << "cached_allocator::deallocate(): ptr == "
              << reinterpret_cast<void*>(ptr) << std::endl;

    // Erase the allocated block from the allocated blocks map.
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);

    if (iter == allocated_blocks.end())
      throw not_my_pointer(reinterpret_cast<void*>(ptr));

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // Insert the block into the free blocks map.
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {
    std::cout << "cached_allocator::free_all()" << std::endl;

    // Deallocate all outstanding blocks in both lists.
    for ( free_blocks_type::iterator i = free_blocks.begin()
        ; i != free_blocks.end()
        ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for( allocated_blocks_type::iterator i = allocated_blocks.begin()
       ; i != allocated_blocks.end()
       ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

int main()
{
  std::size_t num_elements = 32768;

  thrust::host_vector<int> h_input(num_elements);

  // Generate random input.
  thrust::generate(h_input.begin(), h_input.end(), rand);

  thrust::cuda::vector<int> d_input = h_input;
  thrust::cuda::vector<int> d_result(num_elements);

  std::size_t num_trials = 5;

  cached_allocator alloc;

  for (std::size_t i = 0; i < num_trials; ++i)
  {
    d_result = d_input;

    // Pass alloc through cuda::par as the first parameter to sort
    // to cause allocations to be handled by alloc during sort.
    thrust::sort(thrust::cuda::par(alloc), d_result.begin(), d_result.end());

    // Ensure the result is sorted.
    assert(thrust::is_sorted(d_result.begin(), d_result.end()));
  }

  return 0;
}

