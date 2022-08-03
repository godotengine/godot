#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct hash32
{
  __host__ __device__
  unsigned int operator()(unsigned int h) const
  {
    h = ~h + (h << 15);
    h =  h ^ (h >> 12);
    h =  h + (h <<  2);
    h =  h ^ (h >>  4);
    h =  h + (h <<  3) + (h << 11);
    h =  h ^ (h >> 16);
    return h;
  }
};

struct hash64
{
  __host__ __device__
  unsigned long long operator()(unsigned long long h) const
  {
    h = ~h + (h << 21);
    h =  h ^ (h >> 24);
    h = (h + (h <<  3)) + (h << 8);
    h =  h ^ (h >> 14);
    h = (h + (h <<  2)) + (h << 4);
    h =  h ^ (h >> 28);
    h =  h + (h << 31);
    return h;
  }
};

struct hashtofloat
{
  __host__ __device__
  float operator()(unsigned int h) const
  {
    return static_cast<float>(hash32()(h)) / 4294967296.0f;
  }
};

struct hashtodouble
{
  __host__ __device__
  double operator()(unsigned long long h) const
  {
    return static_cast<double>(hash64()(h)) / 18446744073709551616.0;
  }
};



template <typename Vector, typename T>
void _randomize(Vector& v, T)
{
    thrust::transform(thrust::counting_iterator<unsigned int>(0), 
                      thrust::counting_iterator<unsigned int>(0) + v.size(),
                      v.begin(),
                      hash32());
}

template <typename Vector>
void _randomize(Vector& v, long long)
{
    thrust::transform(thrust::counting_iterator<unsigned long long>(0), 
                      thrust::counting_iterator<unsigned long long>(0) + v.size(),
                      v.begin(),
                      hash64());
}

template <typename Vector>
void _randomize(Vector& v, float)
{
    thrust::transform(thrust::counting_iterator<unsigned int>(0), 
                      thrust::counting_iterator<unsigned int>(0) + v.size(),
                      v.begin(),
                      hashtofloat());
}

template <typename Vector>
void _randomize(Vector& v, double)
{
    thrust::transform(thrust::counting_iterator<unsigned long long>(0), 
                      thrust::counting_iterator<unsigned long long>(0) + v.size(),
                      v.begin(),
                      hashtodouble());
}

// fill Vector with random values
template <typename Vector>
void randomize(Vector& v)
{
    _randomize(v, typename Vector::value_type());
}


