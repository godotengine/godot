// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"

namespace embree
{
  /* Point query structure for closest point query */
  template<int K>
  struct RTC_ALIGN(16) PointQueryK 
  {
    /* Default construction does nothing */
    __forceinline PointQueryK() {}

    /* Constructs a ray from origin, direction, and ray segment. Near
     * has to be smaller than far */
    __forceinline PointQueryK(const Vec3vf<K>& p, const vfloat<K>& radius = inf, const vfloat<K>& time = zero)
      : p(p), time(time), radius(radius) {}

    /* Returns the size of the ray */
    static __forceinline size_t size() { return K; }

    /* Calculates if this is a valid ray that does not cause issues during traversal */
    __forceinline vbool<K> valid() const
    {
      const vbool<K> vx = (abs(p.x) <= vfloat<K>(FLT_LARGE));
      const vbool<K> vy = (abs(p.y) <= vfloat<K>(FLT_LARGE));
      const vbool<K> vz = (abs(p.z) <= vfloat<K>(FLT_LARGE));
      const vbool<K> vn = radius >= vfloat<K>(0);
      const vbool<K> vf = abs(time) < vfloat<K>(inf);
      return vx & vy & vz & vn & vf;
    }

    __forceinline void get(PointQueryK<1>* ray) const;
    __forceinline void get(size_t i, PointQueryK<1>& ray) const;
    __forceinline void set(const PointQueryK<1>* ray);
    __forceinline void set(size_t i, const PointQueryK<1>& ray);

    Vec3vf<K> p;      // location of the query point
    vfloat<K> time;   // time for motion blur
    vfloat<K> radius; // radius for the point query
  };
  
  /* Specialization for a single point query */
  template<>
  struct RTC_ALIGN(16) PointQueryK<1>
  {
    /* Default construction does nothing */
    __forceinline PointQueryK() {}

    /* Constructs a ray from origin, direction, and ray segment. Near
     *  has to be smaller than far */
    __forceinline PointQueryK(const Vec3fa& p, float radius = inf, float time = zero)
      : p(p), time(time), radius(radius) {}

    /* Calculates if this is a valid ray that does not cause issues during traversal */
    __forceinline bool valid() const {
      return all(le_mask(abs(Vec3fa(p)), Vec3fa(FLT_LARGE)) & le_mask(Vec3fa(0.f), Vec3fa(radius))) && abs(time) < float(inf);
    }

    Vec3f p;  
    float time;
    float radius;
  };
  
  /* Converts point query packet to single point query */
  template<int K>
  __forceinline void PointQueryK<K>::get(PointQueryK<1>* query) const
  {
    for (size_t i = 0; i < K; i++) // FIXME: use SIMD transpose
    {
      query[i].p.x    = p.x[i]; 
      query[i].p.y    = p.y[i]; 
      query[i].p.z    = p.z[i];
      query[i].time   = time[i];
      query[i].radius = radius[i]; 
    }
  }

  /* Extracts a single point query out of a point query packet*/
  template<int K>
  __forceinline void PointQueryK<K>::get(size_t i, PointQueryK<1>& query) const
  {
    query.p.x    = p.x[i]; 
    query.p.y    = p.y[i]; 
    query.p.z    = p.z[i];
    query.radius = radius[i];  
    query.time   = time[i];  
  }

  /* Converts single point query to point query packet */
  template<int K>
  __forceinline void PointQueryK<K>::set(const PointQueryK<1>* query)
  {
    for (size_t i = 0; i < K; i++)
    {
      p.x[i]    = query[i].p.x;
      p.y[i]    = query[i].p.y;
      p.z[i]    = query[i].p.z;
      radius[i] = query[i].radius; 
      time[i]   = query[i].time; 
    }
  }

  /* inserts a single point query into a point query packet element */
  template<int K>
  __forceinline void PointQueryK<K>::set(size_t i, const PointQueryK<1>& query)
  {
    p.x[i]    = query.p.x;
    p.y[i]    = query.p.y;
    p.z[i]    = query.p.z;
    radius[i] = query.radius; 
    time[i]   = query.time; 
  }

  /* Shortcuts */
  typedef PointQueryK<1>  PointQuery;
  typedef PointQueryK<4>  PointQuery4;
  typedef PointQueryK<8>  PointQuery8;
  typedef PointQueryK<16> PointQuery16;
  struct PointQueryN;

  /* Outputs point query to stream */
  template<int K>
  __forceinline embree_ostream operator <<(embree_ostream cout, const PointQueryK<K>& query)
  {
    cout << "{ " << embree_endl
        << "  p = "    << query.p      << embree_endl
        << "  r = "    << query.radius << embree_endl
        << "  time = " << query.time   << embree_endl
        << "}";
    return cout;
  }
}
