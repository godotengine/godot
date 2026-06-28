// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec2.h"
#include "vec3.h"
#include "bbox.h"

namespace embree
{
  template<typename V>
    struct Interval
    {
      V lower, upper;
      
      __forceinline Interval() {}
      __forceinline Interval           ( const Interval& other ) { lower = other.lower; upper = other.upper; }
      __forceinline Interval& operator=( const Interval& other ) { lower = other.lower; upper = other.upper; return *this; }

      __forceinline Interval(const V& a) : lower(a), upper(a) {}
      __forceinline Interval(const V& lower, const V& upper) : lower(lower), upper(upper) {}
      __forceinline Interval(const BBox<V>& a) : lower(a.lower), upper(a.upper) {}
          
      /*! tests if box is empty */
      //__forceinline bool empty() const { return lower > upper; }
      
      /*! computes the size of the interval */
      __forceinline V size() const { return upper - lower; }
      
      __forceinline V center() const { return 0.5f*(lower+upper); }
      
      __forceinline const Interval& extend(const Interval& other) { lower = min(lower,other.lower); upper = max(upper,other.upper); return *this; }
      __forceinline const Interval& extend(const V   & other) { lower = min(lower,other      ); upper = max(upper,other      ); return *this; }
      
      __forceinline friend Interval operator +( const Interval& a, const Interval& b ) {
        return Interval(a.lower+b.lower,a.upper+b.upper);
      }
      
      __forceinline friend Interval operator -( const Interval& a, const Interval& b ) {
        return Interval(a.lower-b.upper,a.upper-b.lower);
      }
      
      __forceinline friend Interval operator -( const Interval& a, const V& b ) {
        return Interval(a.lower-b,a.upper-b);
      }
      
      __forceinline friend Interval operator *( const Interval& a, const Interval& b )
      {
        const V ll = a.lower*b.lower;
        const V lu = a.lower*b.upper;
        const V ul = a.upper*b.lower;
        const V uu = a.upper*b.upper;
        return Interval(min(ll,lu,ul,uu),max(ll,lu,ul,uu));
      }
      
      __forceinline friend Interval merge( const Interval& a, const Interval& b) {
        return Interval(min(a.lower,b.lower),max(a.upper,b.upper));
      }
      
      __forceinline friend Interval merge( const Interval& a, const Interval& b, const Interval& c) {
        return merge(merge(a,b),c);
      }
      
      __forceinline friend Interval merge( const Interval& a, const Interval& b, const Interval& c, const Interval& d) {
        return merge(merge(a,b),merge(c,d));
      }
      
      /*! intersect bounding boxes */
      __forceinline friend const Interval intersect( const Interval& a, const Interval& b ) { return Interval(max(a.lower, b.lower), min(a.upper, b.upper)); }
      __forceinline friend const Interval intersect( const Interval& a, const Interval& b, const Interval& c ) { return intersect(a,intersect(b,c)); }
      __forceinline friend const Interval intersect( const Interval& a, const Interval& b, const Interval& c, const Interval& d ) { return intersect(intersect(a,b),intersect(c,d)); }       
      
      friend embree_ostream operator<<(embree_ostream cout, const Interval& a) {
        return cout << "[" << a.lower << ", " << a.upper << "]";
      }
      
      ////////////////////////////////////////////////////////////////////////////////
      /// Constants
      ////////////////////////////////////////////////////////////////////////////////
      
      __forceinline Interval( EmptyTy ) : lower(pos_inf), upper(neg_inf) {}
      __forceinline Interval( FullTy  ) : lower(neg_inf), upper(pos_inf) {}
    };

  __forceinline bool isEmpty(const Interval<float>& v) { 
    return v.lower > v.upper;
  }

  __forceinline vboolx isEmpty(const Interval<vfloatx>& v) {
    return v.lower > v.upper;
  }
  
  /*! subset relation */
  template<typename T> __forceinline bool subset( const Interval<T>& a, const Interval<T>& b ) { 
    return (a.lower > b.lower) && (a.upper < b.upper);
  }

  template<typename T> __forceinline bool subset( const Vec2<Interval<T>>& a, const Vec2<Interval<T>>& b ) { 
    return subset(a.x,b.x) && subset(a.y,b.y);
  }

  template<typename T> __forceinline const Vec2<Interval<T>> intersect( const Vec2<Interval<T>>& a, const Vec2<Interval<T>>& b ) {
    return Vec2<Interval<T>>(intersect(a.x,b.x),intersect(a.y,b.y));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Interval<T> select ( bool s, const Interval<T>& t, const Interval<T>& f ) {
    return Interval<T>(select(s,t.lower,f.lower),select(s,t.upper,f.upper));
  }

  template<typename T> __forceinline Interval<T> select ( const typename T::Bool& s, const Interval<T>& t, const Interval<T>& f ) {
    return Interval<T>(select(s,t.lower,f.lower),select(s,t.upper,f.upper));
  }

  __forceinline int numRoots(const Interval<float>& p0, const Interval<float>& p1)
  {
    float eps = 1E-4f;
    bool neg0 = p0.lower < eps; bool pos0 = p0.upper > -eps;
    bool neg1 = p1.lower < eps; bool pos1 = p1.upper > -eps;
    return (neg0 && pos1) || (pos0 && neg1) || (neg0 && pos0) || (neg1 && pos1);
  }
  
  typedef Interval<float> Interval1f;
  typedef Vec2<Interval<float>> Interval2f;
  typedef Vec3<Interval<float>> Interval3f;

inline void swap(float& a, float& b) { float tmp = a; a = b; b = tmp; }

inline Interval1f shift(const Interval1f& v, float shift) { return Interval1f(v.lower + shift, v.upper + shift); }

#define TWO_PI (2.0*M_PI)
inline Interval1f sin(Interval1f interval)
{
  if (interval.upper-interval.lower >= M_PI) { return Interval1f(-1.0, 1.0); }
  if (interval.upper > TWO_PI)                 { interval = shift(interval, -TWO_PI*floor(interval.upper/TWO_PI)); }
  if (interval.lower < 0)                      { interval = shift(interval, -TWO_PI*floor(interval.lower/TWO_PI)); }
  float sinLower = sin(interval.lower);
  float sinUpper = sin(interval.upper);
  if (sinLower > sinUpper) swap(sinLower, sinUpper);
  if (interval.lower <       M_PI / 2.0 && interval.upper >       M_PI / 2.0) sinUpper =  1.0;
  if (interval.lower < 3.0 * M_PI / 2.0 && interval.upper > 3.0 * M_PI / 2.0) sinLower = -1.0;
  return Interval1f(sinLower, sinUpper);
}

inline Interval1f cos(Interval1f interval)
{
  if (interval.upper-interval.lower >= M_PI) { return Interval1f(-1.0, 1.0); }
  if (interval.upper > TWO_PI)                 { interval = shift(interval, -TWO_PI*floor(interval.upper/TWO_PI)); }
  if (interval.lower < 0)                      { interval = shift(interval, -TWO_PI*floor(interval.lower/TWO_PI)); }
  float cosLower = cos(interval.lower);
  float cosUpper = cos(interval.upper);
  if (cosLower > cosUpper) swap(cosLower, cosUpper);
  if (interval.lower < M_PI && interval.upper > M_PI) cosLower = -1.0;
  return Interval1f(cosLower, cosUpper);
}
#undef TWO_PI
}
