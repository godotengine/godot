#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <string>

// this functor clamps a value to the range [lo, hi]
template <typename T>
struct clamp : public thrust::unary_function<T,T>
{
    T lo, hi;

    __host__ __device__
    clamp(T _lo, T _hi) : lo(_lo), hi(_hi) {}

    __host__ __device__
    T operator()(T x)
    {
        if (x < lo)
            return lo;
        else if (x < hi)
            return x;
        else
            return hi;
    }
};

template <typename T>
struct simple_negate : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(T x)
    {
        return -x;
    }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));  
    std::cout << "\n";
}


int main(void)
{
    // clamp values to the range [1, 5]
    int lo = 1;
    int hi = 5;

    // define some types
    typedef thrust::device_vector<int> Vector;
    typedef Vector::iterator           VectorIterator;

    // initialize values
    Vector values(8);

    values[0] =  2;
    values[1] =  5;
    values[2] =  7;
    values[3] =  1;
    values[4] =  6;
    values[5] =  0;
    values[6] =  3;
    values[7] =  8;
    
    print_range("values         ", values.begin(), values.end());

    // define some more types
    typedef thrust::transform_iterator<clamp<int>, VectorIterator> ClampedVectorIterator;

    // create a transform_iterator that applies clamp() to the values array
    ClampedVectorIterator cv_begin = thrust::make_transform_iterator(values.begin(), clamp<int>(lo, hi));
    ClampedVectorIterator cv_end   = cv_begin + values.size();
    
    // now [clamped_begin, clamped_end) defines a sequence of clamped values
    print_range("clamped values ", cv_begin, cv_end);



    ////
    // compute the sum of the clamped sequence with reduce()
    std::cout << "sum of clamped values : " << thrust::reduce(cv_begin, cv_end) << "\n";



    ////
    // combine transform_iterator with other fancy iterators like counting_iterator
    typedef thrust::counting_iterator<int>                           CountingIterator;
    typedef thrust::transform_iterator<clamp<int>, CountingIterator> ClampedCountingIterator;

    CountingIterator count_begin(0);
    CountingIterator count_end(10);
    
    print_range("sequence         ", count_begin, count_end);

    ClampedCountingIterator cs_begin = thrust::make_transform_iterator(count_begin, clamp<int>(lo, hi));
    ClampedCountingIterator cs_end   = thrust::make_transform_iterator(count_end,   clamp<int>(lo, hi));

    print_range("clamped sequence ", cs_begin, cs_end);



    ////
    // combine transform_iterator with another transform_iterator
    typedef thrust::transform_iterator<thrust::negate<int>, ClampedCountingIterator> NegatedClampedCountingIterator;
    
    NegatedClampedCountingIterator ncs_begin = thrust::make_transform_iterator(cs_begin, thrust::negate<int>());
    NegatedClampedCountingIterator ncs_end   = thrust::make_transform_iterator(cs_end,   thrust::negate<int>());

    print_range("negated sequence ", ncs_begin, ncs_end);


    ////
    // when a functor does not define result_type, a third template argument must be provided
    typedef thrust::transform_iterator<simple_negate<int>, VectorIterator, int> NegatedVectorIterator;

    NegatedVectorIterator nv_begin(values.begin(), simple_negate<int>());
    NegatedVectorIterator nv_end(values.end(), simple_negate<int>());
    
    print_range("negated values ", nv_begin, nv_end);

    return 0;
}

