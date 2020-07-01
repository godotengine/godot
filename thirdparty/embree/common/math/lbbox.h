// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "bbox.h"
#include "range.h"

namespace embree
{
  template<typename T>
    __forceinline std::pair<T,T> globalLinear(const std::pair<T,T>& v, const BBox1f& dt)
  {
    const float rcp_dt_size = float(1.0f)/dt.size();
    const T g0 = lerp(v.first,v.second,-dt.lower*rcp_dt_size);
    const T g1 = lerp(v.first,v.second,(1.0f-dt.lower)*rcp_dt_size);
    return std::make_pair(g0,g1);
  }

  template<typename T>
  struct LBBox
  {
  public:
    __forceinline LBBox () {}

    __forceinline LBBox ( const LBBox& other ) { 
      bounds0 = other.bounds0; bounds1 = other.bounds1; 
    }
    __forceinline LBBox& operator= ( const LBBox& other ) { 
      bounds0 = other.bounds0; bounds1 = other.bounds1; return *this; 
    }

    __forceinline LBBox (EmptyTy) 
      : bounds0(EmptyTy()), bounds1(EmptyTy()) {}
    
    __forceinline explicit LBBox ( const BBox<T>& bounds) 
      : bounds0(bounds), bounds1(bounds) { }
    
    __forceinline LBBox ( const BBox<T>& bounds0, const BBox<T>& bounds1) 
      : bounds0(bounds0), bounds1(bounds1) { }

    LBBox ( const avector<BBox<T>>& bounds ) 
    {
      assert(bounds.size());
      BBox<T> b0 = bounds.front();
      BBox<T> b1 = bounds.back();
      for (size_t i=1; i<bounds.size()-1; i++) {
        const float f = float(i)/float(bounds.size()-1);
        const BBox<T> bt = lerp(b0,b1,f);
        const T dlower = min(bounds[i].lower-bt.lower,T(zero));
        const T dupper = max(bounds[i].upper-bt.upper,T(zero));
        b0.lower += dlower; b1.lower += dlower;
        b0.upper += dupper; b1.upper += dupper;
      }
      bounds0 = b0;
      bounds1 = b1;
    }

    /*! calculates the linear bounds of a primitive for the specified time range */
    template<typename BoundsFunc>
    __forceinline LBBox(const BoundsFunc& bounds, const BBox1f& time_range, float numTimeSegments)
    {
      const float lower = time_range.lower*numTimeSegments;
      const float upper = time_range.upper*numTimeSegments;
      const float ilowerf = floor(lower);
      const float iupperf = ceil(upper);
      const int ilower = (int)ilowerf;
      const int iupper = (int)iupperf;

      const BBox<T> blower0 = bounds(ilower);
      const BBox<T> bupper1 = bounds(iupper);

      if (iupper-ilower == 1) {
        bounds0 = lerp(blower0, bupper1, lower-ilowerf);
        bounds1 = lerp(bupper1, blower0, iupperf-upper);
        return;
      }

      const BBox<T> blower1 = bounds(ilower+1);
      const BBox<T> bupper0 = bounds(iupper-1);
      BBox<T> b0 = lerp(blower0, blower1, lower-ilowerf);
      BBox<T> b1 = lerp(bupper1, bupper0, iupperf-upper);

      for (int i = ilower+1; i < iupper; i++)
      {
        const float f = (float(i)/numTimeSegments - time_range.lower) / time_range.size();
        const BBox<T> bt = lerp(b0, b1, f);
        const BBox<T> bi = bounds(i);
        const T dlower = min(bi.lower-bt.lower, T(zero));
        const T dupper = max(bi.upper-bt.upper, T(zero));
        b0.lower += dlower; b1.lower += dlower;
        b0.upper += dupper; b1.upper += dupper;
      }

      bounds0 = b0;
      bounds1 = b1;
    }

    /*! calculates the linear bounds of a primitive for the specified time range */
    template<typename BoundsFunc>
    __forceinline LBBox(const BoundsFunc& bounds, const BBox1f& time_range_in, const BBox1f& geom_time_range, float geom_time_segments)
    {
      /* normalize global time_range_in to local geom_time_range */
      const BBox1f time_range((time_range_in.lower-geom_time_range.lower)/geom_time_range.size(),
                              (time_range_in.upper-geom_time_range.lower)/geom_time_range.size());
        
      const float lower = time_range.lower*geom_time_segments;
      const float upper = time_range.upper*geom_time_segments;
      const float ilowerf = floor(lower);
      const float iupperf = ceil(upper);
      const float ilowerfc = max(0.0f,ilowerf);
      const float iupperfc = min(iupperf,geom_time_segments);
      const int   ilowerc = (int)ilowerfc;
      const int   iupperc = (int)iupperfc;
      assert(iupperc-ilowerc > 0);

      /* this larger iteration range guarantees that we process borders of geom_time_range is (partially) inside time_range_in */
      const int ilower_iter = max(-1,(int)ilowerf);
      const int iupper_iter = min((int)iupperf,(int)geom_time_segments+1);
        
      const BBox<T> blower0 = bounds(ilowerc);
      const BBox<T> bupper1 = bounds(iupperc);
      if (iupper_iter-ilower_iter == 1) {
        bounds0 = lerp(blower0, bupper1, max(0.0f,lower-ilowerfc));
        bounds1 = lerp(bupper1, blower0, max(0.0f,iupperfc-upper));
        return;
      }

      const BBox<T> blower1 = bounds(ilowerc+1);
      const BBox<T> bupper0 = bounds(iupperc-1);
      BBox<T> b0 = lerp(blower0, blower1, max(0.0f,lower-ilowerfc));
      BBox<T> b1 = lerp(bupper1, bupper0, max(0.0f,iupperfc-upper));

      for (int i = ilower_iter+1; i < iupper_iter; i++)
      {
        const float f = (float(i)/geom_time_segments - time_range.lower) / time_range.size();
        const BBox<T> bt = lerp(b0, b1, f);
        const BBox<T> bi = bounds(i);
        const T dlower = min(bi.lower-bt.lower, T(zero));
        const T dupper = max(bi.upper-bt.upper, T(zero));
        b0.lower += dlower; b1.lower += dlower;
        b0.upper += dupper; b1.upper += dupper;
      }

      bounds0 = b0;
      bounds1 = b1;
    }

    /*! calculates the linear bounds of a primitive for the specified time range */
    template<typename BoundsFunc>
    __forceinline LBBox(const BoundsFunc& bounds, const range<int>& time_range, int numTimeSegments)
    {
      const int ilower = time_range.begin();
      const int iupper = time_range.end();

      BBox<T> b0 = bounds(ilower);
      BBox<T> b1 = bounds(iupper);

      if (iupper-ilower == 1)
      {
        bounds0 = b0;
        bounds1 = b1;
        return;
      }
  
      for (int i = ilower+1; i<iupper; i++)
      {
        const float f = float(i - time_range.begin()) / float(time_range.size());
        const BBox<T> bt = lerp(b0, b1, f);
        const BBox<T> bi = bounds(i);
        const T dlower = min(bi.lower-bt.lower, T(zero));
        const T dupper = max(bi.upper-bt.upper, T(zero));
        b0.lower += dlower; b1.lower += dlower;
        b0.upper += dupper; b1.upper += dupper;
      }

      bounds0 = b0;
      bounds1 = b1;
    }

  public:

    __forceinline bool empty() const {
      return bounds().empty();
    }

    __forceinline BBox<T> bounds () const {
      return merge(bounds0,bounds1);
    }

    __forceinline BBox<T> interpolate( const float t ) const {
      return lerp(bounds0,bounds1,t);
    }

    __forceinline LBBox<T> interpolate( const BBox1f& dt ) const {
      return LBBox<T>(interpolate(dt.lower),interpolate(dt.upper));
    }

    __forceinline void extend( const LBBox& other ) {
      bounds0.extend(other.bounds0);
      bounds1.extend(other.bounds1);
    }

    __forceinline float expectedHalfArea() const;

    __forceinline float expectedHalfArea(const BBox1f& dt) const {
      return interpolate(dt).expectedHalfArea();
    }

    __forceinline float expectedApproxHalfArea() const {
      return 0.5f*(halfArea(bounds0) + halfArea(bounds1));
    }

    /* calculates bounds for [0,1] time range from bounds in dt time range */
    __forceinline LBBox global(const BBox1f& dt) const 
    {
      const float rcp_dt_size = 1.0f/dt.size();
      const BBox<T> b0 = interpolate(-dt.lower*rcp_dt_size);
      const BBox<T> b1 = interpolate((1.0f-dt.lower)*rcp_dt_size);
      return LBBox(b0,b1);
    }

    /*! Comparison Operators */
    template<typename TT> friend __forceinline bool operator==( const LBBox<TT>& a, const LBBox<TT>& b ) { return a.bounds0 == b.bounds0 && a.bounds1 == b.bounds1; }
    template<typename TT> friend __forceinline bool operator!=( const LBBox<TT>& a, const LBBox<TT>& b ) { return a.bounds0 != b.bounds0 || a.bounds1 != b.bounds1; }

    /*! output operator */
    friend __forceinline std::ostream& operator<<(std::ostream& cout, const LBBox& box) {
      return cout << "LBBox { " << box.bounds0 << "; " << box.bounds1 << " }";
    }

  public:
    BBox<T> bounds0, bounds1;
  };

  /*! tests if box is finite */
  template<typename T>
    __forceinline bool isvalid( const LBBox<T>& v ) {
    return isvalid(v.bounds0) && isvalid(v.bounds1);
  }
  
  template<typename T>
    __forceinline T expectedArea(const T& a0, const T& a1, const T& b0, const T& b1)
  {
    const T da = a1-a0;
    const T db = b1-b0;
    return a0*b0+(a0*db+da*b0)*T(0.5f) + da*db*T(1.0f/3.0f);
  }
  
  template<> __forceinline float LBBox<Vec3fa>::expectedHalfArea() const 
  {
    const Vec3fa d0 = bounds0.size();
    const Vec3fa d1 = bounds1.size();
    return reduce_add(expectedArea(Vec3fa(d0.x,d0.y,d0.z),
                                   Vec3fa(d1.x,d1.y,d1.z),
                                   Vec3fa(d0.y,d0.z,d0.x),
                                   Vec3fa(d1.y,d1.z,d1.x)));
  }

  template<typename T>
  __forceinline float expectedApproxHalfArea(const LBBox<T>& box) {
    return box.expectedApproxHalfArea(); 
  }

  template<typename T>
  __forceinline LBBox<T> merge(const LBBox<T>& a, const LBBox<T>& b) {
    return LBBox<T>(merge(a.bounds0, b.bounds0), merge(a.bounds1, b.bounds1));
  }

   /*! subset relation */
  template<typename T> __inline bool subset( const LBBox<T>& a, const LBBox<T>& b ) {
    return subset(a.bounds0,b.bounds0) && subset(a.bounds1,b.bounds1);
  }

  /*! default template instantiations */
  typedef LBBox<float> LBBox1f;
  typedef LBBox<Vec2f> LBBox2f;
  typedef LBBox<Vec3f> LBBox3f;
  typedef LBBox<Vec3fa> LBBox3fa;
}
