#include <unittest/unittest.h>
#include <thrust/detail/config.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cpp/vector.h>
#include <memory>

template <typename T>
struct my_allocator_with_custom_construct1
  : thrust::device_malloc_allocator<T>
{
  __host__ __device__
  my_allocator_with_custom_construct1()
  {}

  __host__ __device__
  void construct(T *p)
  {
    *p = 13;
  }
};

template <typename T>
void TestAllocatorCustomDefaultConstruct(size_t n)
{
  thrust::device_vector<T> ref(n, 13);
  thrust::device_vector<T, my_allocator_with_custom_construct1<T> > vec(n);

  ASSERT_EQUAL_QUIET(ref, vec);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorCustomDefaultConstruct);

template <typename T>
struct my_allocator_with_custom_construct2
  : thrust::device_malloc_allocator<T>
{
  __host__ __device__
  my_allocator_with_custom_construct2()
  {}

  template <typename Arg>
  __host__ __device__
  void construct(T *p, const Arg &)
  {
    *p = 13;
  }
};

template <typename T>
void TestAllocatorCustomCopyConstruct(size_t n)
{
  thrust::device_vector<T> ref(n, 13);
  thrust::device_vector<T> copy_from(n, 7);
  thrust::device_vector<T, my_allocator_with_custom_construct2<T> >
    vec(copy_from.begin(), copy_from.end());

  ASSERT_EQUAL_QUIET(ref, vec);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorCustomCopyConstruct);

template <typename T>
struct my_allocator_with_custom_destroy
{
  typedef T         value_type;
  typedef T &       reference;
  typedef const T & const_reference;

  static bool g_state;

  __host__
  my_allocator_with_custom_destroy(){}

  __host__
  my_allocator_with_custom_destroy(const my_allocator_with_custom_destroy &other)
    : use_me_to_alloc(other.use_me_to_alloc)
  {}

  __host__
  ~my_allocator_with_custom_destroy(){}

  __host__ __device__
  void destroy(T *)
  {
#if !__CUDA_ARCH__
    g_state = true;
#endif
  }

  value_type *allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type *ptr, std::ptrdiff_t n)
  {
    use_me_to_alloc.deallocate(ptr,n);
  }

  bool operator==(const my_allocator_with_custom_destroy &) const
  {
    return true;
  }

  bool operator!=(const my_allocator_with_custom_destroy &other) const
  {
    return !(*this == other);
  }

  typedef thrust::detail::true_type is_always_equal;

  // use composition rather than inheritance
  // to avoid inheriting std::allocator's member
  // function destroy
  std::allocator<T> use_me_to_alloc;
};

template <typename T>
bool my_allocator_with_custom_destroy<T>::g_state = false;

template <typename T>
void TestAllocatorCustomDestroy(size_t n)
{
  {
    thrust::cpp::vector<T, my_allocator_with_custom_destroy<T> > vec(n);
  } // destroy everything

  if (0 < n)
    ASSERT_EQUAL(true, my_allocator_with_custom_destroy<T>::g_state);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorCustomDestroy);

template <typename T>
struct my_minimal_allocator
{
  typedef T         value_type;

  // XXX ideally, we shouldn't require
  //     these two typedefs
  typedef T &       reference;
  typedef const T & const_reference;

  __host__
  my_minimal_allocator(){}

  __host__
  my_minimal_allocator(const my_minimal_allocator &other)
    : use_me_to_alloc(other.use_me_to_alloc)
  {}

  __host__
  ~my_minimal_allocator(){}

  value_type *allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type *ptr, std::ptrdiff_t n)
  {
    use_me_to_alloc.deallocate(ptr,n);
  }

  std::allocator<T> use_me_to_alloc;
};

template <typename T>
void TestAllocatorMinimal(size_t n)
{
  thrust::cpp::vector<int, my_minimal_allocator<int> > vec(n, 13);

  // XXX copy to h_vec because ASSERT_EQUAL doesn't know about cpp::vector
  thrust::host_vector<int> h_vec(vec.begin(), vec.end());
  thrust::host_vector<int> ref(n, 13);

  ASSERT_EQUAL(ref, h_vec);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorMinimal);

void TestAllocatorTraitsRebind()
{
  ASSERT_EQUAL(
    (thrust::detail::is_same<
      typename thrust::detail::allocator_traits<
        thrust::device_malloc_allocator<int>
      >::template rebind_traits<float>::other,
      typename thrust::detail::allocator_traits<
        thrust::device_malloc_allocator<float>
      >
    >::value),
    true
  );

  ASSERT_EQUAL(
    (thrust::detail::is_same<
      typename thrust::detail::allocator_traits<
        my_minimal_allocator<int>
      >::template rebind_traits<float>::other,
      typename thrust::detail::allocator_traits<
        my_minimal_allocator<float>
      >
    >::value),
    true
  );
}
DECLARE_UNITTEST(TestAllocatorTraitsRebind);

#if THRUST_CPP_DIALECT >= 2011
void TestAllocatorTraitsRebindCpp11()
{
  ASSERT_EQUAL(
    (thrust::detail::is_same<
      typename thrust::detail::allocator_traits<
        thrust::device_malloc_allocator<int>
      >::template rebind_alloc<float>,
      thrust::device_malloc_allocator<float>
    >::value),
    true
  );

  ASSERT_EQUAL(
    (thrust::detail::is_same<
      typename thrust::detail::allocator_traits<
        my_minimal_allocator<int>
      >::template rebind_alloc<float>,
      my_minimal_allocator<float>
    >::value),
    true
  );

  ASSERT_EQUAL(
    (thrust::detail::is_same<
      typename thrust::detail::allocator_traits<
        thrust::device_malloc_allocator<int>
      >::template rebind_traits<float>,
      typename thrust::detail::allocator_traits<
        thrust::device_malloc_allocator<float>
      >
    >::value),
    true
  );

  ASSERT_EQUAL(
    (thrust::detail::is_same<
      typename thrust::detail::allocator_traits<
        my_minimal_allocator<int>
      >::template rebind_traits<float>,
      typename thrust::detail::allocator_traits<
        my_minimal_allocator<float>
      >
    >::value),
    true
  );
}
DECLARE_UNITTEST(TestAllocatorTraitsRebindCpp11);
#endif // C++11

