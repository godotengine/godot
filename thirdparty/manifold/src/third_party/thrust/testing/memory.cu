#include <iostream>
#include <unittest/unittest.h>
#include <thrust/memory.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>

// Define a new system class, as the my_system one is already used with a thrust::sort template definition
// that calls back into sort.cu
class my_memory_system : public thrust::device_execution_policy<my_memory_system>
{
  public:
    my_memory_system(int)
      : correctly_dispatched(false),
        num_copies(0)
    {}

    my_memory_system(const my_memory_system &other)
      : correctly_dispatched(false),
        num_copies(other.num_copies + 1)
    {}

    void validate_dispatch()
    {
      correctly_dispatched = (num_copies == 0);
    }

    bool is_valid()
    {
      return correctly_dispatched;
    }

  private:
    bool correctly_dispatched;

    // count the number of copies so that we can validate
    // that dispatch does not introduce any
    unsigned int num_copies;


    // disallow default construction
    my_memory_system();
};

namespace my_old_namespace
{

struct my_old_temporary_allocation_system
  : public thrust::device_execution_policy<my_old_temporary_allocation_system>
{
};

template <typename T>
thrust::pair<thrust::pointer<T, my_old_temporary_allocation_system>, std::ptrdiff_t>
get_temporary_buffer(my_old_temporary_allocation_system, std::ptrdiff_t)
{
  thrust::pointer<T, my_old_temporary_allocation_system> const
    result(reinterpret_cast<T*>(4217));

  return thrust::make_pair(result, 314);
}

template<typename Pointer>
void return_temporary_buffer(my_old_temporary_allocation_system, Pointer p)
{
  typedef typename thrust::detail::pointer_traits<Pointer>::raw_pointer RP;
  ASSERT_EQUAL(p.get(), reinterpret_cast<RP>(4217));
}

} // my_old_namespace

namespace my_new_namespace
{

struct my_new_temporary_allocation_system
  : public thrust::device_execution_policy<my_new_temporary_allocation_system>
{
};

template <typename T>
thrust::pair<thrust::pointer<T, my_new_temporary_allocation_system>, std::ptrdiff_t>
get_temporary_buffer(my_new_temporary_allocation_system, std::ptrdiff_t)
{
  thrust::pointer<T, my_new_temporary_allocation_system> const
    result(reinterpret_cast<T*>(1742));

  return thrust::make_pair(result, 413);
}

template<typename Pointer>
void return_temporary_buffer(my_new_temporary_allocation_system, Pointer)
{
  // This should never be called (the three-argument with size overload below
  // should be preferred) and shouldn't be ambiguous.
  ASSERT_EQUAL(true, false);
}

template<typename Pointer>
void return_temporary_buffer(my_new_temporary_allocation_system, Pointer p, std::ptrdiff_t n)
{
  typedef typename thrust::detail::pointer_traits<Pointer>::raw_pointer RP;
  ASSERT_EQUAL(p.get(), reinterpret_cast<RP>(1742));
  ASSERT_EQUAL(n, 413);
}

} // my_new_namespace

template<typename T1, typename T2>
bool are_same(const T1 &, const T2 &)
{
  return false;
}


template<typename T>
bool are_same(const T &, const T &)
{
  return true;
}


void TestSelectSystemDifferentTypes()
{
  using thrust::system::detail::generic::select_system;

  my_memory_system my_sys(0);
  thrust::device_system_tag device_sys;

  // select_system(my_system, device_system_tag) should return device_system_tag (the minimum tag)
  bool is_device_system_tag = are_same(device_sys, select_system(my_sys, device_sys));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(device_system_tag, my_tag) should return device_system_tag (the minimum tag)
  is_device_system_tag = are_same(device_sys, select_system(device_sys, my_sys));
  ASSERT_EQUAL(true, is_device_system_tag);
}
DECLARE_UNITTEST(TestSelectSystemDifferentTypes);


void TestSelectSystemSameTypes()
{
  using thrust::system::detail::generic::select_system;

  my_memory_system my_sys(0);
  thrust::device_system_tag device_sys;
  thrust::host_system_tag host_sys;

  // select_system(host_system_tag, host_system_tag) should return host_system_tag
  bool is_host_system_tag = are_same(host_sys, select_system(host_sys, host_sys));
  ASSERT_EQUAL(true, is_host_system_tag);

  // select_system(device_system_tag, device_system_tag) should return device_system_tag
  bool is_device_system_tag = are_same(device_sys, select_system(device_sys, device_sys));
  ASSERT_EQUAL(true, is_device_system_tag);

  // select_system(my_system, my_system) should return my_system
  bool is_my_system = are_same(my_sys, select_system(my_sys, my_sys));
  ASSERT_EQUAL(true, is_my_system);
}
DECLARE_UNITTEST(TestSelectSystemSameTypes);


void TestGetTemporaryBuffer()
{
  const std::ptrdiff_t n = 9001;

  thrust::device_system_tag dev_tag;
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz = thrust::get_temporary_buffer<int>(dev_tag, n);

  ASSERT_EQUAL(ptr_and_sz.second, n);

  const int ref_val = 13;
  thrust::device_vector<int> ref(n, ref_val);

  thrust::fill_n(ptr_and_sz.first, n, ref_val);

  ASSERT_EQUAL(true, thrust::all_of(ptr_and_sz.first, ptr_and_sz.first + n, thrust::placeholders::_1 == ref_val));

  thrust::return_temporary_buffer(dev_tag, ptr_and_sz.first, ptr_and_sz.second);
}
DECLARE_UNITTEST(TestGetTemporaryBuffer);


void TestMalloc()
{
  const std::ptrdiff_t n = 9001;

  thrust::device_system_tag dev_tag;
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  pointer ptr = pointer(static_cast<int*>(thrust::malloc(dev_tag, sizeof(int) * n).get()));

  const int ref_val = 13;
  thrust::device_vector<int> ref(n, ref_val);

  thrust::fill_n(ptr, n, ref_val);

  ASSERT_EQUAL(true, thrust::all_of(ptr, ptr + n, thrust::placeholders::_1 == ref_val));

  thrust::free(dev_tag, ptr);
}
DECLARE_UNITTEST(TestMalloc);


thrust::pointer<void,my_memory_system>
  malloc(my_memory_system &system, std::size_t)
{
  system.validate_dispatch();

  return thrust::pointer<void,my_memory_system>();
}


void TestMallocDispatchExplicit()
{
  const size_t n = 0;

  my_memory_system sys(0);
  thrust::malloc(sys, n);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMallocDispatchExplicit);


template<typename Pointer>
void free(my_memory_system &system, Pointer)
{
  system.validate_dispatch();
}


void TestFreeDispatchExplicit()
{
  thrust::pointer<my_memory_system,void> ptr;

  my_memory_system sys(0);
  thrust::free(sys, ptr);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFreeDispatchExplicit);


template<typename T>
  thrust::pair<thrust::pointer<T,my_memory_system>, std::ptrdiff_t>
    get_temporary_buffer(my_memory_system &system, std::ptrdiff_t n)
{
  system.validate_dispatch();

  thrust::device_system_tag device_sys;
  thrust::pair<thrust::pointer<T, thrust::device_system_tag>, std::ptrdiff_t> result = thrust::get_temporary_buffer<T>(device_sys, n);
  return thrust::make_pair(thrust::pointer<T,my_memory_system>(result.first.get()), result.second);
}


void TestGetTemporaryBufferDispatchExplicit()
{
  const std::ptrdiff_t n = 9001;

  my_memory_system sys(0);
  typedef thrust::pointer<int, thrust::device_system_tag> pointer;
  thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz = thrust::get_temporary_buffer<int>(sys, n);

  ASSERT_EQUAL(ptr_and_sz.second, n);
  ASSERT_EQUAL(true, sys.is_valid());

  const int ref_val = 13;
  thrust::device_vector<int> ref(n, ref_val);

  thrust::fill_n(ptr_and_sz.first, n, ref_val);

  ASSERT_EQUAL(true, thrust::all_of(ptr_and_sz.first, ptr_and_sz.first + n, thrust::placeholders::_1 == ref_val));

  thrust::return_temporary_buffer(sys, ptr_and_sz.first, ptr_and_sz.second);
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchExplicit);


void TestGetTemporaryBufferDispatchImplicit()
{
  if(are_same(thrust::device_system_tag(), thrust::system::cpp::tag()))
  {
    // XXX cpp uses the internal scalar backend, which currently elides user tags
    KNOWN_FAILURE;
  }
  else
  {
    thrust::device_vector<int> vec(9001);

    thrust::sequence(vec.begin(), vec.end());
    thrust::reverse(vec.begin(), vec.end());

    // call something we know will invoke get_temporary_buffer
    my_memory_system sys(0);
    thrust::sort(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, thrust::is_sorted(vec.begin(), vec.end()));
    ASSERT_EQUAL(true, sys.is_valid());
  }
}
DECLARE_UNITTEST(TestGetTemporaryBufferDispatchImplicit);


void TestTemporaryBufferOldCustomization()
{
  typedef my_old_namespace::my_old_temporary_allocation_system system;
  typedef thrust::pointer<int, system> pointer;
  typedef thrust::pair<pointer, std::ptrdiff_t> pointer_and_size;

  system sys;

  {
    pointer_and_size ps = thrust::get_temporary_buffer<int>(sys, 0);

    // The magic values are defined in `my_old_namespace` above.
    ASSERT_EQUAL(ps.first.get(), reinterpret_cast<int*>(4217));
    ASSERT_EQUAL(ps.second, 314);

    thrust::return_temporary_buffer(sys, ps.first, ps.second);
  }
}
DECLARE_UNITTEST(TestTemporaryBufferOldCustomization);


void TestTemporaryBufferNewCustomization()
{
  typedef my_new_namespace::my_new_temporary_allocation_system system;
  typedef thrust::pointer<int, system> pointer;
  typedef thrust::pair<pointer, std::ptrdiff_t> pointer_and_size;

  system sys;

  {
    pointer_and_size ps = thrust::get_temporary_buffer<int>(sys, 0);

    // The magic values are defined in `my_new_namespace` above.
    ASSERT_EQUAL(ps.first.get(), reinterpret_cast<int*>(1742));
    ASSERT_EQUAL(ps.second, 413);

    thrust::return_temporary_buffer(sys, ps.first, ps.second);
  }
}
DECLARE_UNITTEST(TestTemporaryBufferNewCustomization);
