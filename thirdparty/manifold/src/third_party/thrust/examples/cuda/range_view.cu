#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>


// This example demonstrates the use of a view: a non-owning wrapper for an
// iterator range which presents a container-like interface to the user.
//
// For example, a view of a device_vector's data can be helpful when we wish to
// access that data from a device function. Even though device_vectors are not
// accessible from device functions, the range_view class allows us to access
// and manipulate its data as if we were manipulating a real container.

template<class Iterator>
class range_view
{
public:
  typedef Iterator iterator;
  typedef typename thrust::iterator_traits<iterator>::value_type value_type;
  typedef typename thrust::iterator_traits<iterator>::pointer pointer;
  typedef typename thrust::iterator_traits<iterator>::difference_type difference_type;
  typedef typename thrust::iterator_traits<iterator>::reference reference;

private:
  const iterator first;
  const iterator last;


public:
  __host__ __device__
  range_view(Iterator first, Iterator last)
      : first(first), last(last) {}
  __host__ __device__
  ~range_view() {}

  __host__ __device__
  difference_type size() const { return thrust::distance(first, last); }


  __host__ __device__
  reference operator[](difference_type n)
  {
    return *(first + n);
  }
  __host__ __device__
  const reference operator[](difference_type n) const
  {
    return *(first + n);
  }

  __host__ __device__
  iterator begin() 
  {
    return first;
  }
  __host__ __device__
  const iterator cbegin() const
  {
    return first;
  }
  __host__ __device__
  iterator end() 
  {
    return last;
  }
  __host__ __device__
  const iterator cend() const
  {
    return last;
  }


  __host__ __device__
  thrust::reverse_iterator<iterator> rbegin()
  {
    return thrust::reverse_iterator<iterator>(end());
  }
  __host__ __device__
  const thrust::reverse_iterator<const iterator> crbegin() const 
  {
    return thrust::reverse_iterator<const iterator>(cend());
  }
  __host__ __device__
  thrust::reverse_iterator<iterator> rend()
  {
    return thrust::reverse_iterator<iterator>(begin());
  }
  __host__ __device__
  const thrust::reverse_iterator<const iterator> crend() const 
  {
    return thrust::reverse_iterator<const iterator>(cbegin());
  }
  __host__ __device__
  reference front() 
  {
    return *begin();
  }
  __host__ __device__
  const reference front()  const
  {
    return *cbegin();
  }

  __host__ __device__
  reference back() 
  {
    return *end();
  }
  __host__ __device__
  const reference back()  const
  {
    return *cend();
  }

  __host__ __device__
  bool empty() const 
  {
    return size() == 0;
  }

};

// This helper function creates a range_view from iterator and the number of
// elements
template <class Iterator, class Size>
range_view<Iterator>
__host__ __device__
make_range_view(Iterator first, Size n)
{
  return range_view<Iterator>(first, first+n);
}

// This helper function creates a range_view from a pair of iterators
template <class Iterator>
range_view<Iterator>
__host__ __device__
make_range_view(Iterator first, Iterator last)
{
  return range_view<Iterator>(first, last);
}

// This helper function creates a range_view from a Vector
template <class Vector>
range_view<typename Vector::iterator>
__host__
make_range_view(Vector& v)
{
  return range_view<typename Vector::iterator>(v.begin(), v.end());
}


// This saxpy functor stores view of X, Y, Z array, and accesses them in
// vector-like way
template<class View1, class View2, class View3>
struct saxpy_functor : public thrust::unary_function<int,void>
{
  const float a;
  View1 x;
  View2 y;
  View3 z;

  __host__ __device__
  saxpy_functor(float _a, View1 _x, View2 _y, View3 _z)
      : a(_a), x(_x), y(_y), z(_z)
  {
  }

  __host__ __device__ 
  void operator()(int i) 
  {
    z[i] = a * x[i] + y[i];
  }
};

// saxpy function, which can either be called form host or device
// The views are passed by value
template<class View1, class View2, class View3>
__host__ __device__
void saxpy(float A, View1 X, View2 Y, View3 Z)
{
  // Z = A * X + Y
  const int size = X.size();
  thrust::for_each(thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      saxpy_functor<View1,View2,View3>(A,X,Y,Z));
}

struct f1 : public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) const
  {
    return x*3;
  }
};

int main()
{
  using std::cout;
  using std::endl;

  // initialize host arrays
  float x[4] = {1.0, 1.0, 1.0, 1.0};
  float y[4] = {1.0, 2.0, 3.0, 4.0};
  float z[4] = {0.0};

  thrust::device_vector<float> X(x, x + 4);
  thrust::device_vector<float> Y(y, y + 4);
  thrust::device_vector<float> Z(z, z + 4);

  saxpy(
      2.0, 

      // make a range view of a pair of transform_iterators
      make_range_view(thrust::make_transform_iterator(X.cbegin(), f1()),
                      thrust::make_transform_iterator(X.cend(), f1())),

      // range view of normal_iterators
      make_range_view(Y.begin(), thrust::distance(Y.begin(), Y.end())),

      // range view of naked pointers
      make_range_view(Z.data().get(), 4));

  // print values from original device_vector<float> Z 
  // to ensure that range view was mapped to this vector
  for (std::size_t i = 0, n = Z.size(); i < n; ++i)
  {
    cout << "z[" << i << "]= " << Z[i] << endl;
  }


  return 0;
}

