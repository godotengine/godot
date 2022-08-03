#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>

// this example illustrates how to make repeated access to a range of values
// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3] 
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] 
//   ...

template <typename Iterator>
class repeated_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct repeat_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type repeats;

        repeat_functor(difference_type repeats)
            : repeats(repeats) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return i / repeats;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<repeat_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the repeated_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    repeated_range(Iterator first, Iterator last, difference_type repeats)
        : first(first), last(last), repeats(repeats) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
    }

    iterator end(void) const
    {
        return begin() + repeats * (last - first);
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type repeats;
};

int main(void)
{
    thrust::device_vector<int> data(4);
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;

    // print the initial data
    std::cout << "range        ";
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    typedef thrust::device_vector<int>::iterator Iterator;
  
    // create repeated_range with elements repeated twice
    repeated_range<Iterator> twice(data.begin(), data.end(), 2);
    std::cout << "repeated x2: ";
    thrust::copy(twice.begin(), twice.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;
    
    // create repeated_range with elements repeated x3
    repeated_range<Iterator> thrice(data.begin(), data.end(), 3);
    std::cout << "repeated x3: ";
    thrust::copy(thrice.begin(), thrice.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    return 0;
}
