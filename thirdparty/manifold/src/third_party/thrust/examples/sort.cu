#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <iostream>
#include <iomanip>

// Helper routines

void initialize(thrust::device_vector<int>& v)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(10, 99);
  for(size_t i = 0; i < v.size(); i++)
    v[i] = dist(rng);
}

void initialize(thrust::device_vector<float>& v)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(2, 19);
  for(size_t i = 0; i < v.size(); i++)
    v[i] = dist(rng) / 2.0f;
}

void initialize(thrust::device_vector< thrust::pair<int,int> >& v)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(0,9);
  for(size_t i = 0; i < v.size(); i++)
  {
    int a = dist(rng);
    int b = dist(rng);
    v[i] = thrust::make_pair(a,b);
  }
}

void initialize(thrust::device_vector<int>& v1, thrust::device_vector<int>& v2)
{
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(10, 99);
  for(size_t i = 0; i < v1.size(); i++)
  {
    v1[i] = dist(rng);
    v2[i] = static_cast<int>(i);
  }
}

void print(const thrust::device_vector<int>& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << " " << v[i];
  std::cout << "\n";
}

void print(const thrust::device_vector<float>& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << " " << std::fixed << std::setprecision(1) << v[i];
  std::cout << "\n";
}

void print(const thrust::device_vector< thrust::pair<int,int> >& v)
{
  for(size_t i = 0; i < v.size(); i++)
  {
    thrust::pair<int,int> p = v[i];
    std::cout << " (" << p.first << "," << p.second << ")";
  }
  std::cout << "\n";
}

void print(thrust::device_vector<int>& v1, thrust::device_vector<int> v2)
{
  for(size_t i = 0; i < v1.size(); i++)
    std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
  std::cout << "\n";
}


// user-defined comparison operator that acts like less<int>,
// except even numbers are considered to be smaller than odd numbers
struct evens_before_odds
{
  __host__ __device__
  bool operator()(int x, int y)
  {
    if (x % 2 == y % 2)
      return x < y;
    else if (x % 2)
      return false;
    else
      return true;
  }
};


int main(void)
{
  size_t N = 16;

  std::cout << "sorting integers\n";
  {
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
  }
  
  std::cout << "\nsorting integers (descending)\n";
  {
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end(), thrust::greater<int>());
    print(keys);
  }
  
  std::cout << "\nsorting integers (user-defined comparison)\n";
  {
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end(), evens_before_odds());
    print(keys);
  }

  std::cout << "\nsorting floats\n";
  {
    thrust::device_vector<float> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
  }
  
  std::cout << "\nsorting pairs\n";
  {
    thrust::device_vector< thrust::pair<int,int> > keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
  }
  
  std::cout << "\nkey-value sorting\n";
  {
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);
    initialize(keys, values);
    print(keys, values);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    print(keys, values);
  }
  
  std::cout << "\nkey-value sorting (descending)\n";
  {
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);
    initialize(keys, values);
    print(keys, values);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin(), thrust::greater<int>());
    print(keys, values);
  }

  return 0;
}

