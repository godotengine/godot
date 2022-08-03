#include <unittest/unittest.h>
#include <thrust/detail/alignment.h>

struct alignof_mock_0
{
    char a;
    char b;
}; // size: 2 * sizeof(char), alignment: sizeof(char)

struct alignof_mock_1
{
    int n;
    char c;
    // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_2
{
    int n;
    char c;
    // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_3
{
    char c;
    // sizeof(int) - sizeof(char) bytes of padding
    int n;
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_4
{
    char c0;
    // sizeof(int) - sizeof(char) bytes of padding
    int n;
    char c1;
    // sizeof(int) - sizeof(char) bytes of padding
}; // size: 3 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_5
{
    char c0;
    char c1;
    // sizeof(int) - 2 * sizeof(char) bytes of padding
    int n;
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_6
{
    int n;
    char c0;
    char c1;
    // sizeof(int) - 2 * sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

void test_alignof_mocks_sizes()
{
    ASSERT_EQUAL(sizeof(alignof_mock_0), 2 * sizeof(char));
    ASSERT_EQUAL(sizeof(alignof_mock_1), 2 * sizeof(int));
    ASSERT_EQUAL(sizeof(alignof_mock_2), 2 * sizeof(int));
    ASSERT_EQUAL(sizeof(alignof_mock_3), 2 * sizeof(int));
    ASSERT_EQUAL(sizeof(alignof_mock_4), 3 * sizeof(int));
    ASSERT_EQUAL(sizeof(alignof_mock_5), 2 * sizeof(int));
    ASSERT_EQUAL(sizeof(alignof_mock_6), 2 * sizeof(int));
}
DECLARE_UNITTEST(test_alignof_mocks_sizes);

void test_alignof()
{
    ASSERT_EQUAL(THRUST_ALIGNOF(bool)                  , sizeof(bool));
    ASSERT_EQUAL(THRUST_ALIGNOF(signed char)           , sizeof(signed char));
    ASSERT_EQUAL(THRUST_ALIGNOF(unsigned char)         , sizeof(unsigned char));
    ASSERT_EQUAL(THRUST_ALIGNOF(char)                  , sizeof(char));
    ASSERT_EQUAL(THRUST_ALIGNOF(short int)             , sizeof(short int));
    ASSERT_EQUAL(THRUST_ALIGNOF(unsigned short int)    , sizeof(unsigned short int));
    ASSERT_EQUAL(THRUST_ALIGNOF(int)                   , sizeof(int));
    ASSERT_EQUAL(THRUST_ALIGNOF(unsigned int)          , sizeof(unsigned int));
    ASSERT_EQUAL(THRUST_ALIGNOF(long int)              , sizeof(long int));
    ASSERT_EQUAL(THRUST_ALIGNOF(unsigned long int)     , sizeof(unsigned long int));
    ASSERT_EQUAL(THRUST_ALIGNOF(long long int)         , sizeof(long long int));
    ASSERT_EQUAL(THRUST_ALIGNOF(unsigned long long int), sizeof(unsigned long long int));
    ASSERT_EQUAL(THRUST_ALIGNOF(float)                 , sizeof(float));
    ASSERT_EQUAL(THRUST_ALIGNOF(double)                , sizeof(double));
    ASSERT_EQUAL(THRUST_ALIGNOF(long double)           , sizeof(long double));

    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_0), sizeof(char));
    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_1), sizeof(int));
    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_2), sizeof(int));
    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_3), sizeof(int));
    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_4), sizeof(int));
    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_5), sizeof(int));
    ASSERT_EQUAL(THRUST_ALIGNOF(alignof_mock_6), sizeof(int));
}
DECLARE_UNITTEST(test_alignof);

void test_alignment_of()
{
    ASSERT_EQUAL(
        thrust::detail::alignment_of<bool>::value
      , sizeof(bool)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<signed char>::value
      , sizeof(signed char)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<unsigned char>::value
      , sizeof(unsigned char)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<char>::value
      , sizeof(char)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<short int>::value
      , sizeof(short int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<unsigned short int>::value
      , sizeof(unsigned short int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<int>::value
      , sizeof(int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<unsigned int>::value
      , sizeof(unsigned int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<long int>::value
      , sizeof(long int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<unsigned long int>::value
      , sizeof(unsigned long int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<long long int>::value
      , sizeof(long long int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<unsigned long long int>::value
      , sizeof(unsigned long long int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<float>::value
      , sizeof(float)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<double>::value
      , sizeof(double)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<long double>::value
      , sizeof(long double)
    );

    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_0>::value
      , sizeof(char)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_1>::value
      , sizeof(int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_2>::value
      , sizeof(int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_3>::value
      , sizeof(int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_4>::value
      , sizeof(int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_5>::value
      , sizeof(int)
    );
    ASSERT_EQUAL(
        thrust::detail::alignment_of<alignof_mock_6>::value
      , sizeof(int)
    );
}
DECLARE_UNITTEST(test_alignment_of);

template <std::size_t Align>
void test_aligned_type_instantiation()
{
    typedef typename thrust::detail::aligned_type<Align>::type type;
    ASSERT_GEQUAL(sizeof(type), 1lu);
    ASSERT_EQUAL(THRUST_ALIGNOF(type), Align);
    ASSERT_EQUAL(thrust::detail::alignment_of<type>::value, Align);
}

void test_aligned_type()
{
    test_aligned_type_instantiation<1>();
    test_aligned_type_instantiation<2>();
    test_aligned_type_instantiation<4>();
    test_aligned_type_instantiation<8>();
    test_aligned_type_instantiation<16>();
    test_aligned_type_instantiation<32>();
    test_aligned_type_instantiation<64>();
    test_aligned_type_instantiation<128>();
}
DECLARE_UNITTEST(test_aligned_type);

template <std::size_t Len, std::size_t Align>
void test_aligned_storage_instantiation(thrust::detail::true_type /* Align is valid */)
{
    typedef typename thrust::detail::aligned_storage<Len, Align>::type type;
    ASSERT_GEQUAL(sizeof(type), Len);
    ASSERT_EQUAL(THRUST_ALIGNOF(type), Align);
    ASSERT_EQUAL(thrust::detail::alignment_of<type>::value, Align);
}

template <std::size_t Len, std::size_t Align>
void test_aligned_storage_instantiation(thrust::detail::false_type /* Align is invalid */)
{
  // no-op -- alignment is > max_align_t and MSVC complains loudly.
}

template <std::size_t Len, std::size_t Align>
void test_aligned_storage_instantiation()
{
  typedef thrust::detail::integral_constant<
      bool, Align <= THRUST_ALIGNOF(thrust::detail::max_align_t)>
      ValidAlign;
  test_aligned_storage_instantiation<Len, Align>(ValidAlign());
}

template <std::size_t Len>
void test_aligned_storage_size()
{
    test_aligned_storage_instantiation<Len, 1>();
    test_aligned_storage_instantiation<Len, 2>();
    test_aligned_storage_instantiation<Len, 4>();
    test_aligned_storage_instantiation<Len, 8>();
    test_aligned_storage_instantiation<Len, 16>();
    test_aligned_storage_instantiation<Len, 32>();
    test_aligned_storage_instantiation<Len, 64>();
    test_aligned_storage_instantiation<Len, 128>();
}

void test_aligned_storage()
{
    test_aligned_storage_size<1>();
    test_aligned_storage_size<2>();
    test_aligned_storage_size<4>();
    test_aligned_storage_size<8>();
    test_aligned_storage_size<16>();
    test_aligned_storage_size<32>();
    test_aligned_storage_size<64>();
    test_aligned_storage_size<128>();
    test_aligned_storage_size<256>();
    test_aligned_storage_size<512>();
    test_aligned_storage_size<1024>();
    test_aligned_storage_size<2048>();
    test_aligned_storage_size<4096>();
    test_aligned_storage_size<8192>();
    test_aligned_storage_size<16384>();

    test_aligned_storage_size<3>();
    test_aligned_storage_size<5>();
    test_aligned_storage_size<7>();

    test_aligned_storage_size<17>();
    test_aligned_storage_size<42>();

    test_aligned_storage_size<10000>();
}
DECLARE_UNITTEST(test_aligned_storage);

void test_max_align_t()
{
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(bool)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(signed char)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(unsigned char)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(char)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(short int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(unsigned short int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(unsigned int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(long int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(unsigned long int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(long long int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(unsigned long long int)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(float)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(double)
    );
    ASSERT_GEQUAL(
        THRUST_ALIGNOF(thrust::detail::max_align_t)
      , THRUST_ALIGNOF(long double)
    );
}
DECLARE_UNITTEST(test_max_align_t);

void test_aligned_reinterpret_cast()
{
    thrust::detail::aligned_type<1>* a1 = 0;

    thrust::detail::aligned_type<2>* a2 = 0;

    // Cast to type with stricter (larger) alignment requirement.
    a2 = thrust::detail::aligned_reinterpret_cast<
        thrust::detail::aligned_type<2>*
    >(a1);

    // Cast to type with less strict (smaller) alignment requirement.
    a1 = thrust::detail::aligned_reinterpret_cast<
        thrust::detail::aligned_type<1>*
    >(a2);
}
DECLARE_UNITTEST(test_aligned_reinterpret_cast);

