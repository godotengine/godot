#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>

// this example illustrates how to make strided access to a range of values
// examples:
//   strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6] 
//   strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
//   ...

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

int main(void)
{
    thrust::device_vector<int> data(8);
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;
    data[5] = 60;
    data[6] = 70;
    data[7] = 80;

    // print the initial data
    std::cout << "data: ";
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    typedef thrust::device_vector<int>::iterator Iterator;
    
    // create strided_range with indices [0,2,4,6]
    strided_range<Iterator> evens(data.begin(), data.end(), 2);
    std::cout << "sum of even indices: " << thrust::reduce(evens.begin(), evens.end()) << std::endl;
    
    // create strided_range with indices [1,3,5,7]
    strided_range<Iterator> odds(data.begin() + 1, data.end(), 2);
    std::cout << "sum of odd indices:  " << thrust::reduce(odds.begin(), odds.end()) << std::endl;

    // set odd elements to 0 with fill()
    std::cout << "setting odd indices to zero: ";
    thrust::fill(odds.begin(), odds.end(), 0);
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    return 0;
}
