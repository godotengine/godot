#include <unittest/unittest.h>

#include <thrust/sequence.h>
#include <thrust/allocate_unique.h>
#include <thrust/universal_vector.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <numeric>
#include <vector>

namespace
{

// The managed_memory_pointer class should be identified as a
// contiguous_iterator
THRUST_STATIC_ASSERT(
  thrust::is_contiguous_iterator<thrust::universal_allocator<int>::pointer>::value);

template <typename T>
struct some_object {
  some_object(T data)
      : m_data(data)
  {}

  void setter(T data) { m_data = data; }
  T getter() const { return m_data; }

private:
  T m_data;
};

} // namespace

template <typename T>
void TestUniversalAllocateUnique()
{
  // Simple test to ensure that pointers created with universal_memory_resource
  // can be dereferenced and used with STL code. This is necessary as some
  // STL implementations break when using fancy references that overload
  // operator&, so universal_memory_resource uses a special pointer type that
  // returns regular C++ references that can be safely used host-side.

  // These operations fail to compile with fancy references:
  auto raw = thrust::allocate_unique<T>(thrust::universal_allocator<T>{}, 42);
  auto obj = thrust::allocate_unique<some_object<T>>(
    thrust::universal_allocator<some_object<T> >{}, 42
  );

  static_assert(
    std::is_same<decltype(raw.get()),
                 thrust::universal_ptr<T> >::value,
    "Unexpected pointer type returned from std::unique_ptr::get.");
  static_assert(
    std::is_same<decltype(obj.get()),
                 thrust::universal_ptr<some_object<T> > >::value,
    "Unexpected pointer type returned from std::unique_ptr::get.");

  ASSERT_EQUAL(*raw, T(42));
  ASSERT_EQUAL(*raw.get(), T(42));
  ASSERT_EQUAL(obj->getter(), T(42));
  ASSERT_EQUAL((*obj).getter(), T(42));
  ASSERT_EQUAL(obj.get()->getter(), T(42));
  ASSERT_EQUAL((*obj.get()).getter(), T(42));
}
DECLARE_GENERIC_UNITTEST(TestUniversalAllocateUnique);

template <typename T>
void TestUniversalIterationRaw()
{
  auto array = thrust::allocate_unique_n<T>(
    thrust::universal_allocator<T>{}, 6, 42);

  static_assert(
    std::is_same<decltype(array.get()), thrust::universal_ptr<T> >::value,
    "Unexpected pointer type returned from std::unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    ASSERT_EQUAL(*iter, T(42));
    ASSERT_EQUAL(*iter.get(), T(42));
  }
}
DECLARE_GENERIC_UNITTEST(TestUniversalIterationRaw);

template <typename T>
void TestUniversalIterationObj()
{
  auto array = thrust::allocate_unique_n<some_object<T>>(
    thrust::universal_allocator<some_object<T>>{}, 6, 42);

  static_assert(
    std::is_same<decltype(array.get()),
                 thrust::universal_ptr<some_object<T>>>::value,
    "Unexpected pointer type returned from std::unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    ASSERT_EQUAL(iter->getter(), T(42));
    ASSERT_EQUAL((*iter).getter(), T(42));
    ASSERT_EQUAL(iter.get()->getter(), T(42));
    ASSERT_EQUAL((*iter.get()).getter(), T(42));
  }
}
DECLARE_GENERIC_UNITTEST(TestUniversalIterationObj);

template <typename T>
void TestUniversalRawPointerCast()
{
  auto obj = thrust::allocate_unique<T>(thrust::universal_allocator<T>{}, 42);

  static_assert(
    std::is_same<decltype(obj.get()), thrust::universal_ptr<T>>::value,
    "Unexpected pointer type returned from std::unique_ptr::get.");

  static_assert(
    std::is_same<decltype(thrust::raw_pointer_cast(obj.get())), T*>::value,
    "Unexpected pointer type returned from thrust::raw_pointer_cast.");

  *thrust::raw_pointer_cast(obj.get()) = T(17);

  ASSERT_EQUAL(*obj, T(17));
}
DECLARE_GENERIC_UNITTEST(TestUniversalRawPointerCast);

template <typename T>
void TestUniversalThrustVector(std::size_t const n)
{
  thrust::host_vector<T>      host(n);
  thrust::universal_vector<T> universal(n);

  static_assert(
    std::is_same<typename std::decay<decltype(universal)>::type::pointer,
                 thrust::universal_ptr<T>>::value,
    "Unexpected thrust::universal_vector pointer type.");

  thrust::sequence(host.begin(), host.end(), 0);
  thrust::sequence(universal.begin(), universal.end(), 0);

  ASSERT_EQUAL(host.size(), n);
  ASSERT_EQUAL(universal.size(), n);
  ASSERT_EQUAL(host, universal);
}
DECLARE_VARIABLE_UNITTEST(TestUniversalThrustVector);

// Verify that a std::vector using the universal allocator will work with
// Standard Library algorithms.
template <typename T>
void TestUniversalStdVector(std::size_t const n)
{
  std::vector<T>                                 host(n);
  std::vector<T, thrust::universal_allocator<T>> universal(n);

  static_assert(
    std::is_same<typename std::decay<decltype(universal)>::type::pointer,
                 thrust::universal_ptr<T>>::value,
    "Unexpected std::vector pointer type.");

  std::iota(host.begin(), host.end(), 0);
  std::iota(universal.begin(), universal.end(), 0);

  ASSERT_EQUAL(host.size(), n);
  ASSERT_EQUAL(universal.size(), n);
  ASSERT_EQUAL(host, universal);
}
DECLARE_VARIABLE_UNITTEST(TestUniversalStdVector);

