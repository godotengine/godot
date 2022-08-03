#include <unittest/unittest.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


template<typename ForwardIterator>
void sequence(my_system &system, ForwardIterator, ForwardIterator)
{
    system.validate_dispatch();
}

void TestSequenceDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::sequence(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSequenceDispatchExplicit);


template<typename ForwardIterator>
void sequence(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
}

void TestSequenceDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::sequence(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.end()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSequenceDispatchImplicit);


template <class Vector>
void TestSequenceSimple()
{
    using value_type = typename Vector::value_type;
    Vector v(5);

    thrust::sequence(v.begin(), v.end());

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);

    thrust::sequence(v.begin(), v.end(), value_type{10});

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);
    ASSERT_EQUAL(v[3], 13);
    ASSERT_EQUAL(v[4], 14);
    
    thrust::sequence(v.begin(), v.end(), value_type{10}, value_type{2});

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 12);
    ASSERT_EQUAL(v[2], 14);
    ASSERT_EQUAL(v[3], 16);
    ASSERT_EQUAL(v[4], 18);
}
DECLARE_VECTOR_UNITTEST(TestSequenceSimple);


template <typename T>
void TestSequence(size_t n)
{
    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    thrust::sequence(h_data.begin(), h_data.end());
    thrust::sequence(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10));
    thrust::sequence(d_data.begin(), d_data.end(), T(10));

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);
    
    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSequence);


template <typename T>
void TestSequenceToDiscardIterator(size_t n)
{
    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    thrust::sequence(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(13),
                     T(10),
                     T(2));

    // nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(TestSequenceToDiscardIterator);


void TestSequenceComplex()
{
  thrust::device_vector<thrust::complex<double> > m(64);
  thrust::sequence(m.begin(), m.end());
}
DECLARE_UNITTEST(TestSequenceComplex);

// A class that doesnt accept conversion from size_t but can be multiplied by a scalar
struct Vector
{
    Vector() = default;
    // Explicitly disable construction from size_t
    Vector(std::size_t) = delete;
    __host__ __device__ Vector(int x_, int y_) : x{x_}, y{y_} {}
    Vector(const Vector&) = default;
    Vector &operator=(const Vector&) = default;

    int x, y;
};

// Vector-Vector addition
__host__ __device__ Vector operator+(const Vector a, const Vector b)
{
  return Vector{a.x + b.x, a.y + b.y};
}

// Vector-Scalar Multiplication
// Multiplication by std::size_t is required by thrust::sequence.
__host__ __device__ Vector operator*(const std::size_t a, const Vector b)
{
  return Vector{static_cast<int>(a) * b.x, static_cast<int>(a) * b.y};
}
__host__ __device__ Vector operator*(const Vector b, const std::size_t a)
{
  return Vector{static_cast<int>(a) * b.x, static_cast<int>(a) * b.y};
}

void TestSequenceNoSizeTConversion()
{
    thrust::device_vector<Vector> m(64);
    thrust::sequence(m.begin(), m.end(), ::Vector{0, 0}, ::Vector{1, 2});

    for (std::size_t i = 0; i < m.size(); ++i)
    {
        const ::Vector v = m[i];
        ASSERT_EQUAL(static_cast<std::size_t>(v.x), i);
        ASSERT_EQUAL(static_cast<std::size_t>(v.y), 2 * i);
    }
}
DECLARE_UNITTEST(TestSequenceNoSizeTConversion);
