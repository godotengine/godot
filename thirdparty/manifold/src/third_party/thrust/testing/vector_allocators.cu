#include <unittest/unittest.h>

#include <thrust/detail/config.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename BaseAlloc, bool PropagateOnSwap>
class stateful_allocator : public BaseAlloc
{
  typedef thrust::detail::allocator_traits<BaseAlloc> base_traits;

public:
    stateful_allocator(int i) : state(i)
    {
    }

    ~stateful_allocator() {}

    stateful_allocator(const stateful_allocator &other)
        : BaseAlloc(other), state(other.state)
    {
    }

    stateful_allocator & operator=(const stateful_allocator & other)
    {
        state = other.state;
        return *this;
    }

#if THRUST_CPP_DIALECT >= 2011
    stateful_allocator(stateful_allocator && other)
        : BaseAlloc(std::move(other)), state(other.state)
    {
        other.state = 0;
    }

    stateful_allocator & operator=(stateful_allocator && other)
    {
        state = other.state;
        other.state = 0;
        return *this;
    }
#endif

    static int last_allocated;
    static int last_deallocated;

    typedef typename base_traits::pointer pointer;
    typedef typename base_traits::const_pointer const_pointer;
    typedef typename base_traits::reference reference;
    typedef typename base_traits::const_reference const_reference;

    pointer allocate(std::size_t size)
    {
        BaseAlloc alloc;
        last_allocated = state;
        return base_traits::allocate(alloc, size);
    }

    void deallocate(pointer ptr, std::size_t size)
    {
        BaseAlloc alloc;
        last_deallocated = state;
        return base_traits::deallocate(alloc, ptr, size);
    }

    static void construct(pointer ptr)
    {
      BaseAlloc alloc;
      return base_traits::construct(alloc, ptr);
    }

    static void destroy(pointer ptr)
    {
      BaseAlloc alloc;
      return base_traits::destroy(alloc, ptr);
    }

    bool operator==(const stateful_allocator &rhs) const
    {
        return state == rhs.state;
    }

    bool operator!=(const stateful_allocator &rhs) const
    {
        return state != rhs.state;
    }

    friend std::ostream & operator<<(std::ostream &os,
        const stateful_allocator & alloc)
    {
        os << "stateful_alloc(" << alloc.state << ")";
        return os;
    }

    typedef thrust::detail::false_type is_always_equal;
    typedef thrust::detail::true_type propagate_on_container_copy_assignment;
    typedef thrust::detail::true_type propagate_on_container_move_assignment;
    typedef thrust::detail::integral_constant<bool, PropagateOnSwap> propagate_on_container_swap;

private:
    int state;
};

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_allocated = 0;

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_deallocated = 0;

typedef stateful_allocator<std::allocator<int>, true> host_alloc;
typedef stateful_allocator<thrust::device_allocator<int>, true> device_alloc;

typedef thrust::host_vector<int, host_alloc> host_vector;
typedef thrust::device_vector<int, device_alloc> device_vector;

typedef stateful_allocator<std::allocator<int>, false> host_alloc_nsp;
typedef stateful_allocator<thrust::device_allocator<int>, false> device_alloc_nsp;

typedef thrust::host_vector<int, host_alloc_nsp> host_vector_nsp;
typedef thrust::device_vector<int, device_alloc_nsp> device_vector_nsp;

template<typename Vector>
void TestVectorAllocatorConstructors()
{
    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(alloc1);
    ASSERT_EQUAL(v1.get_allocator(), alloc1);

    Vector v2(10, alloc1);
    ASSERT_EQUAL(v2.size(), 10u);
    ASSERT_EQUAL(v2.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v3(10, 17, alloc1);
    ASSERT_EQUAL((v3 == std::vector<int>(10, 17)), true);
    ASSERT_EQUAL(v3.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v4(v3, alloc2);
    ASSERT_EQUAL((v3 == v4), true);
    ASSERT_EQUAL(v4.get_allocator(), alloc2);
    ASSERT_EQUAL(Alloc::last_allocated, 2);
    Alloc::last_allocated = 0;

#if THRUST_CPP_DIALECT >= 2011
    // FIXME: uncomment this after the vector_base(vector_base&&, const Alloc&)
    // is fixed and implemented
    // Vector v5(std::move(v3), alloc2);
    // ASSERT_EQUAL((v4 == v5), true);
    // ASSERT_EQUAL(v5.get_allocator(), alloc2);
    // ASSERT_EQUAL(Alloc::last_allocated, 1);
    // Alloc::last_allocated = 0;
#endif

    Vector v6(v4.begin(), v4.end(), alloc2);
    ASSERT_EQUAL((v4 == v6), true);
    ASSERT_EQUAL(v6.get_allocator(), alloc2);
    ASSERT_EQUAL(Alloc::last_allocated, 2);
}

void TestVectorAllocatorConstructorsHost()
{
    TestVectorAllocatorConstructors<host_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorConstructorsHost);

void TestVectorAllocatorConstructorsDevice()
{
    TestVectorAllocatorConstructors<device_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorConstructorsDevice);

template<typename Vector>
void TestVectorAllocatorPropagateOnCopyAssignment()
{
    ASSERT_EQUAL(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = v1;
    ASSERT_EQUAL((v1 == v2), true);
    ASSERT_EQUAL(v2.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 1);
    ASSERT_EQUAL(Alloc::last_deallocated, 2);
}

void TestVectorAllocatorPropagateOnCopyAssignmentHost()
{
    TestVectorAllocatorPropagateOnCopyAssignment<host_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnCopyAssignmentHost);

void TestVectorAllocatorPropagateOnCopyAssignmentDevice()
{
    TestVectorAllocatorPropagateOnCopyAssignment<device_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnCopyAssignmentDevice);

#if THRUST_CPP_DIALECT >= 2011
template<typename Vector>
void TestVectorAllocatorPropagateOnMoveAssignment()
{
    typedef typename Vector::allocator_type Alloc;
    ASSERT_EQUAL(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    {
    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = std::move(v1);
    ASSERT_EQUAL(v2.get_allocator(), alloc1);
    ASSERT_EQUAL(Alloc::last_allocated, 2);
    ASSERT_EQUAL(Alloc::last_deallocated, 2);
    }

    ASSERT_EQUAL(Alloc::last_deallocated, 1);
}

void TestVectorAllocatorPropagateOnMoveAssignmentHost()
{
    TestVectorAllocatorPropagateOnMoveAssignment<host_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnMoveAssignmentHost);

void TestVectorAllocatorPropagateOnMoveAssignmentDevice()
{
    TestVectorAllocatorPropagateOnMoveAssignment<device_vector>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnMoveAssignmentDevice);
#endif

template<typename Vector>
void TestVectorAllocatorPropagateOnSwap()
{
    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(17, alloc1);
    thrust::swap(v1, v2);

    ASSERT_EQUAL(v1.size(), 17u);
    ASSERT_EQUAL(v2.size(), 10u);

    Vector v3(15, alloc1);
    Vector v4(31, alloc2);
    ASSERT_THROWS(thrust::swap(v3, v4), thrust::detail::allocator_mismatch_on_swap);
}

void TestVectorAllocatorPropagateOnSwapHost()
{
    TestVectorAllocatorPropagateOnSwap<host_vector_nsp>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnSwapHost);

void TestVectorAllocatorPropagateOnSwapDevice()
{
    TestVectorAllocatorPropagateOnSwap<device_vector_nsp>();
}
DECLARE_UNITTEST(TestVectorAllocatorPropagateOnSwapDevice);
