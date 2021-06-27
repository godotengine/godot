// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec2.h"
#include "vec3.h"

namespace embree
{
  namespace internal {

    template <typename T> __forceinline T divideByTwo(const T& v) { return v / T(2); }
    template <> __forceinline float divideByTwo<float>(const float& v) { return v * 0.5f; }
    template <> __forceinline double divideByTwo<double>(const double& v) { return v * 0.5; }

  } // namespace internal
  template<typename T>
  struct BBox
  {
    T lower, upper;

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline BBox           ( )                   { }
    template<typename T1>
    __forceinline BBox           ( const BBox<T1>& other ) : lower(other.lower), upper(other.upper) {}
    __forceinline BBox& operator=( const BBox& other ) { lower = other.lower; upper = other.upper; return *this; }

    __forceinline BBox ( const T& v                     ) : lower(v), upper(v) {}
    __forceinline BBox ( const T& lower, const T& upper ) : lower(lower), upper(upper) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Extending Bounds
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const BBox& extend(const BBox& other) { lower = min(lower,other.lower); upper = max(upper,other.upper); return *this; }
    __forceinline const BBox& extend(const T   & other) { lower = min(lower,other      ); upper = max(upper,other      ); return *this; }

    /*! tests if box is empty */
    __forceinline bool empty() const { for (int i=0; i<T::N; i++) if (lower[i] > upper[i]) return true; return false; }

    /*! computes the size of the box */
    __forceinline T size() const { return upper - lower; }

    /*! computes the center of the box */
    __forceinline T center() const { return internal::divideByTwo<T>(lower+upper); }

    /*! computes twice the center of the box */
    __forceinline T center2() const { return lower+upper; }

    /*! merges two boxes */
    __forceinline static const BBox merge (const BBox& a, const BBox& b) {
      return BBox(min(a.lower, b.lower), max(a.upper, b.upper));
    }

     /*! enlarge box by some scaling factor */
    __forceinline BBox enlarge_by(const float a) const {
      return BBox(lower - T(a)*abs(lower), upper + T(a)*abs(upper));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline BBox( EmptyTy ) : lower(pos_inf), upper(neg_inf) {}
    __forceinline BBox( FullTy  ) : lower(neg_inf), upper(pos_inf) {}
    __forceinline BBox( FalseTy ) : lower(pos_inf), upper(neg_inf) {}
    __forceinline BBox( TrueTy  ) : lower(neg_inf), upper(pos_inf) {}
    __forceinline BBox( NegInfTy ): lower(pos_inf), upper(neg_inf) {}
    __forceinline BBox( PosInfTy ): lower(neg_inf), upper(pos_inf) {}
  };

  template<> __forceinline bool BBox<float>::empty() const {
    return lower > upper;
  }

#if defined(__SSE__)
  template<> __forceinline bool BBox<Vec3fa>::empty() const {
    return !all(le_mask(lower,upper));
  }
  template<> __forceinline bool BBox<Vec3fx>::empty() const {
    return !all(le_mask(lower,upper));
  }
#endif

  /*! tests if box is finite */
  __forceinline bool isvalid( const BBox<Vec3fa>& v ) {
    return all(gt_mask(v.lower,Vec3fa_t(-FLT_LARGE)) & lt_mask(v.upper,Vec3fa_t(+FLT_LARGE)));
  }

  /*! tests if box is finite and non-empty*/
  __forceinline bool isvalid_non_empty( const BBox<Vec3fa>& v ) {
    return all(gt_mask(v.lower,Vec3fa_t(-FLT_LARGE)) & lt_mask(v.upper,Vec3fa_t(+FLT_LARGE)) & le_mask(v.lower,v.upper));
  }
  
  /*! tests if box has finite entries */
  __forceinline bool is_finite( const BBox<Vec3fa>& b) {
    return is_finite(b.lower) && is_finite(b.upper);
  }

  /*! test if point contained in box */
  __forceinline bool inside ( const BBox<Vec3fa>& b, const Vec3fa& p ) { return all(ge_mask(p,b.lower) & le_mask(p,b.upper)); }

  /*! computes the center of the box */
  template<typename T> __forceinline const T center2(const BBox<T>& box) { return box.lower + box.upper; }
  template<typename T> __forceinline const T center (const BBox<T>& box) { return internal::divideByTwo<T>(center2(box)); }

  /*! computes the volume of a bounding box */
  __forceinline float volume    ( const BBox<Vec3fa>& b ) { return reduce_mul(b.size()); }
  __forceinline float safeVolume( const BBox<Vec3fa>& b ) { if (b.empty()) return 0.0f; else return volume(b); }

  /*! computes the volume of a bounding box */
  __forceinline float volume( const BBox<Vec3f>& b )  { return reduce_mul(b.size()); }

  /*! computes the surface area of a bounding box */
  template<typename T> __forceinline const T area( const BBox<Vec2<T> >& b ) { const Vec2<T> d = b.size(); return d.x*d.y; }

  template<typename T> __forceinline const T halfArea( const BBox<Vec3<T> >& b ) { return halfArea(b.size()); }
  template<typename T> __forceinline const T     area( const BBox<Vec3<T> >& b ) { return T(2)*halfArea(b); }

  __forceinline float halfArea( const BBox<Vec3fa>& b ) { return halfArea(b.size()); }
  __forceinline float     area( const BBox<Vec3fa>& b ) { return 2.0f*halfArea(b); }

  __forceinline float halfArea( const BBox<Vec3fx>& b ) { return halfArea(b.size()); }
  __forceinline float     area( const BBox<Vec3fx>& b ) { return 2.0f*halfArea(b); }

  template<typename Vec> __forceinline float safeArea( const BBox<Vec>& b ) { if (b.empty()) return 0.0f; else return area(b); }

  template<typename T> __forceinline float expectedApproxHalfArea(const BBox<T>& box) {
    return halfArea(box);
  }

  /*! merges bounding boxes and points */
  template<typename T> __forceinline const BBox<T> merge( const BBox<T>& a, const       T& b ) { return BBox<T>(min(a.lower, b    ), max(a.upper, b    )); }
  template<typename T> __forceinline const BBox<T> merge( const       T& a, const BBox<T>& b ) { return BBox<T>(min(a    , b.lower), max(a    , b.upper)); }
  template<typename T> __forceinline const BBox<T> merge( const BBox<T>& a, const BBox<T>& b ) { return BBox<T>(min(a.lower, b.lower), max(a.upper, b.upper)); }

  /*! Merges three boxes. */
  template<typename T> __forceinline const BBox<T> merge( const BBox<T>& a, const BBox<T>& b, const BBox<T>& c ) { return merge(a,merge(b,c)); }

  /*! Merges four boxes. */
  template<typename T> __forceinline BBox<T> merge(const BBox<T>& a, const BBox<T>& b, const BBox<T>& c, const BBox<T>& d) {
    return merge(merge(a,b),merge(c,d));
  }

  /*! Comparison Operators */
  template<typename T> __forceinline bool operator==( const BBox<T>& a, const BBox<T>& b ) { return a.lower == b.lower && a.upper == b.upper; }
  template<typename T> __forceinline bool operator!=( const BBox<T>& a, const BBox<T>& b ) { return a.lower != b.lower || a.upper != b.upper; }

  /*! scaling */
  template<typename T> __forceinline BBox<T> operator *( const float& a, const BBox<T>& b ) { return BBox<T>(a*b.lower,a*b.upper); }
  template<typename T> __forceinline BBox<T> operator *( const     T& a, const BBox<T>& b ) { return BBox<T>(a*b.lower,a*b.upper); }

  /*! translations */
  template<typename T> __forceinline BBox<T> operator +( const BBox<T>& a, const BBox<T>& b ) { return BBox<T>(a.lower+b.lower,a.upper+b.upper); }
  template<typename T> __forceinline BBox<T> operator -( const BBox<T>& a, const BBox<T>& b ) { return BBox<T>(a.lower-b.lower,a.upper-b.upper); }
  template<typename T> __forceinline BBox<T> operator +( const BBox<T>& a, const      T & b ) { return BBox<T>(a.lower+b      ,a.upper+b      ); }
  template<typename T> __forceinline BBox<T> operator -( const BBox<T>& a, const      T & b ) { return BBox<T>(a.lower-b      ,a.upper-b      ); }

  /*! extension */
  template<typename T> __forceinline BBox<T> enlarge(const BBox<T>& a, const T& b) { return BBox<T>(a.lower-b, a.upper+b); }

  /*! intersect bounding boxes */
  template<typename T> __forceinline const BBox<T> intersect( const BBox<T>& a, const BBox<T>& b ) { return BBox<T>(max(a.lower, b.lower), min(a.upper, b.upper)); }
  template<typename T> __forceinline const BBox<T> intersect( const BBox<T>& a, const BBox<T>& b, const BBox<T>& c ) { return intersect(a,intersect(b,c)); }
  template<typename T> __forceinline const BBox<T> intersect( const BBox<T>& a, const BBox<T>& b, const BBox<T>& c, const BBox<T>& d ) { return intersect(intersect(a,b),intersect(c,d)); }

  /*! subtract bounds from each other */
  template<typename T> __forceinline void subtract(const BBox<T>& a, const BBox<T>& b, BBox<T>& c, BBox<T>& d)
  {
    c.lower = a.lower;
    c.upper = min(a.upper,b.lower);
    d.lower = max(a.lower,b.upper);
    d.upper = a.upper;
  }

  /*! tests if bounding boxes (and points) are disjoint (empty intersection) */
  template<typename T> __inline bool disjoint( const BBox<T>& a, const BBox<T>& b ) { return intersect(a,b).empty(); }
  template<typename T> __inline bool disjoint( const BBox<T>& a, const       T& b ) { return disjoint(a,BBox<T>(b)); }
  template<typename T> __inline bool disjoint( const       T& a, const BBox<T>& b ) { return disjoint(BBox<T>(a),b); }

  /*! tests if bounding boxes (and points) are conjoint (non-empty intersection) */
  template<typename T> __inline bool conjoint( const BBox<T>& a, const BBox<T>& b ) { return !intersect(a,b).empty(); }
  template<typename T> __inline bool conjoint( const BBox<T>& a, const       T& b ) { return conjoint(a,BBox<T>(b)); }
  template<typename T> __inline bool conjoint( const       T& a, const BBox<T>& b ) { return conjoint(BBox<T>(a),b); }

  /*! subset relation */
  template<typename T> __inline bool subset( const BBox<T>& a, const BBox<T>& b )
  { 
    for ( size_t i = 0; i < T::N; i++ ) if ( a.lower[i] < b.lower[i] ) return false;
    for ( size_t i = 0; i < T::N; i++ ) if ( a.upper[i] > b.upper[i] ) return false;
    return true; 
  }

  template<> __inline bool subset( const BBox<Vec3fa>& a, const BBox<Vec3fa>& b ) {
    return all(ge_mask(a.lower,b.lower)) & all(le_mask(a.upper,b.upper));
  }

  template<> __inline bool subset( const BBox<Vec3fx>& a, const BBox<Vec3fx>& b ) {
    return all(ge_mask(a.lower,b.lower)) & all(le_mask(a.upper,b.upper));
  }
  
  /*! blending */
  template<typename T>
    __forceinline BBox<T> lerp(const BBox<T>& b0, const BBox<T>& b1, const float t) {
    return BBox<T>(lerp(b0.lower,b1.lower,t),lerp(b0.upper,b1.upper,t));
  }

  /*! output operator */
  template<typename T> __forceinline embree_ostream operator<<(embree_ostream cout, const BBox<T>& box) {
    return cout << "[" << box.lower << "; " << box.upper << "]";
  }

  /*! default template instantiations */
  typedef BBox<float> BBox1f;
  typedef BBox<Vec2f> BBox2f;
  typedef BBox<Vec2fa> BBox2fa;
  typedef BBox<Vec3f> BBox3f;
  typedef BBox<Vec3fa> BBox3fa;
  typedef BBox<Vec3fx> BBox3fx;
  typedef BBox<Vec3ff> BBox3ff;
}

////////////////////////////////////////////////////////////////////////////////
/// SSE / AVX / MIC specializations
////////////////////////////////////////////////////////////////////////////////

#if defined __SSE__
#include "../simd/sse.h"
#endif

#if defined __AVX__
#include "../simd/avx.h"
#endif

#if defined(__AVX512F__)
#include "../simd/avx512.h"
#endif

namespace embree
{
  template<int N>
    __forceinline BBox<Vec3<vfloat<N>>> transpose(const BBox3fa* bounds);
  
  template<>
    __forceinline BBox<Vec3<vfloat4>> transpose<4>(const BBox3fa* bounds)
  {
    BBox<Vec3<vfloat4>> dest;
    
    transpose((vfloat4&)bounds[0].lower,
              (vfloat4&)bounds[1].lower,
              (vfloat4&)bounds[2].lower,
              (vfloat4&)bounds[3].lower,
              dest.lower.x,
              dest.lower.y,
              dest.lower.z);
    
    transpose((vfloat4&)bounds[0].upper,
              (vfloat4&)bounds[1].upper,
              (vfloat4&)bounds[2].upper,
              (vfloat4&)bounds[3].upper,
              dest.upper.x,
              dest.upper.y,
              dest.upper.z);
    
    return dest;
  }
  
#if defined(__AVX__)
  template<>
    __forceinline BBox<Vec3<vfloat8>> transpose<8>(const BBox3fa* bounds)
  {
    BBox<Vec3<vfloat8>> dest;
    
    transpose((vfloat4&)bounds[0].lower,
              (vfloat4&)bounds[1].lower,
              (vfloat4&)bounds[2].lower,
              (vfloat4&)bounds[3].lower,
              (vfloat4&)bounds[4].lower,
              (vfloat4&)bounds[5].lower,
              (vfloat4&)bounds[6].lower,
              (vfloat4&)bounds[7].lower,
              dest.lower.x,
              dest.lower.y,
              dest.lower.z);
    
    transpose((vfloat4&)bounds[0].upper,
              (vfloat4&)bounds[1].upper,
              (vfloat4&)bounds[2].upper,
              (vfloat4&)bounds[3].upper,
              (vfloat4&)bounds[4].upper,
              (vfloat4&)bounds[5].upper,
              (vfloat4&)bounds[6].upper,
              (vfloat4&)bounds[7].upper,
              dest.upper.x,
              dest.upper.y,
              dest.upper.z);
    
    return dest;
  }
#endif
  
  template<int N>
    __forceinline BBox3fa merge(const BBox3fa* bounds);
  
  template<>
    __forceinline BBox3fa merge<4>(const BBox3fa* bounds)
  {
    const Vec3fa lower = min(min(bounds[0].lower,bounds[1].lower),
                             min(bounds[2].lower,bounds[3].lower));
    const Vec3fa upper = max(max(bounds[0].upper,bounds[1].upper),
                             max(bounds[2].upper,bounds[3].upper));
    return BBox3fa(lower,upper);
  }
  
#if defined(__AVX__)
  template<>
    __forceinline BBox3fa merge<8>(const BBox3fa* bounds)
  {
    const Vec3fa lower = min(min(min(bounds[0].lower,bounds[1].lower),min(bounds[2].lower,bounds[3].lower)),
                             min(min(bounds[4].lower,bounds[5].lower),min(bounds[6].lower,bounds[7].lower)));
    const Vec3fa upper = max(max(max(bounds[0].upper,bounds[1].upper),max(bounds[2].upper,bounds[3].upper)),
                             max(max(bounds[4].upper,bounds[5].upper),max(bounds[6].upper,bounds[7].upper)));
    return BBox3fa(lower,upper);
  }
#endif
}

