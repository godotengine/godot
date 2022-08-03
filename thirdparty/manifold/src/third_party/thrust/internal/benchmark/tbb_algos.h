#pragma once

#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>
#include <tbb/tbb_thread.h>

#include <cstdef> // For std::size_t.

#include <cassert>

template <typename T>
struct NegateBody
{ 
  void operator()(T& x) const
  {
    x = -x;
  }
};

template <typename Vector>
struct ForBody
{ 
  typedef typename Vector::value_type T;

private:
  Vector& v;

public: 
  ForBody(Vector& x) : v(x) {}    

  void operator()(tbb::blocked_range<std::size_t> const& r) const
  { 
    for (std::size_t i = r.begin(); i != r.end(); ++i)  
      v[i] = -v[i];
  }
};

template <typename Vector>
struct ReduceBody
{ 
  typedef typename Vector::value_type T;

private:
  Vector& v;

public: 
  T sum;  

  ReduceBody(Vector& x) : v(x), sum(0) {}    

  ReduceBody(ReduceBody& x, tbb::split) : v(x.v), sum(0) {}

  void operator()(tbb::blocked_range<std::size_t> const& r)
  { 
    for (std::size_t i = r.begin(); i != r.end(); ++i)  
      sum += v[i];
  }
  
  void join(ReduceBody const& x) { sum += x.sum; } 
};

template <typename Vector>
struct ScanBody
{ 
  typedef typename Vector::value_type T;

private:
  Vector& v; 

public: 
  T sum; 

  ScanBody(Vector& x) : sum(0), v(x) {} 

  ScanBody(ScanBody& x, tbb::split) : v(x.v), sum(0) {} 

  template <typename Tag> 
  void operator()(tbb::blocked_range<std::size_t> const& r, Tag)
  {
    T temp = sum; 
    for (std::size_t i = r.begin(); i < r.end(); ++i)
    { 
      temp = temp + x[i]; 
      if (Tag::is_final_scan()) 
        x[i] = temp; 
    }        
    sum = temp; 
  }

  void assign(ScanBody const& x) { sum = x.sum; } 

  T get_sum() const { return sum; } 

  void reverse_join(ScanBody const& x) { sum = x.sum + sum;} 
};

template <typename Vector>
struct CopyBody
{ 
  typedef typename Vector::value_type T;

private:
  Vector &v;
  Vector &u;

public: 
  CopyBody(Vector& x, Vector& y) : v(x), u(y) {}    

  void operator()(tbb::blocked_range<size_t> const& r) const
  { 
    for (std::size_t i = r.begin(); i != r.end(); ++i)  
      v[i] = u[i];
  }
};

template <typename Vector>
typename Vector::value_type tbb_reduce(Vector& v)
{
  ReduceBody<Vector> body(v);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v.size()), body);
  return body.sum;
}

template <typename Vector>
void tbb_sort(Vector& v)
{
  tbb::parallel_sort(v.begin(), v.end());
}

template <typename Vector>
void tbb_transform(Vector& v)
{
  ForBody<Vector> body(v);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, v.size()), body);
}

template <typename Vector>
void tbb_scan(Vector& v)
{
  ScanBody<Vector> body(v);
  tbb::parallel_scan(tbb::blocked_range<size_t>(0, v.size()), body);
}

template <typename Vector>
void tbb_copy(Vector& v, Vector& u)
{
  CopyBody<Vector> body(v, u);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, v.size()), body);
}

void test_tbb()
{
  std::size_t elements = 1 << 20;

  std::vector<int> A(elements);
  std::vector<int> B(elements);
  std::vector<int> C(elements);
  std::vector<int> D(elements);

  randomize(A);
  randomize(B);
  assert(std::accumulate(A.begin(), A.end(), 0) == tbb_reduce(A));
  
  randomize(A);
  randomize(B);
  std::transform(A.begin(), A.end(), A.begin(), thrust::negate<int>());
  tbb_transform(B);
  assert(A == B);
 
  randomize(A);
  randomize(B);
  std::partial_sum(A.begin(), A.end(), A.begin());
  tbb_scan(B);
  assert(A == B);

  randomize(A);
  randomize(B);
  std::sort(A.begin(), A.end());
  tbb_sort(B);
  assert(A == B);

  randomize(A);
  randomize(B);
  randomize(C);
  randomize(D);
  std::copy(A.begin(), A.end(), C.begin());
  tbb_copy(B, D);
  assert(A == B);
  assert(C == D);
}

