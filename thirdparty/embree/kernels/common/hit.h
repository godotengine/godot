// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "ray.h"
#include "instance_stack.h"

namespace embree
{
  /* Hit structure for K hits */
  template<int K>
    struct HitK
  {
    /* Default construction does nothing */
    __forceinline HitK() {}

    /* Constructs a hit */
    __forceinline HitK(const RTCIntersectContext* context, const vuint<K>& geomID, const vuint<K>& primID, const vfloat<K>& u, const vfloat<K>& v, const Vec3vf<K>& Ng)
      : Ng(Ng), u(u), v(v), primID(primID), geomID(geomID) 
    {
      for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l)
        instID[l] = RTC_INVALID_GEOMETRY_ID;
      instance_id_stack::copy_UV<K>(context->instID, instID);
    }

    /* Returns the size of the hit */
    static __forceinline size_t size() { return K; }

  public:
    Vec3vf<K> Ng;  // geometry normal
    vfloat<K> u;         // barycentric u coordinate of hit
    vfloat<K> v;         // barycentric v coordinate of hit
    vuint<K> primID;      // primitive ID
    vuint<K> geomID;      // geometry ID
    vuint<K> instID[RTC_MAX_INSTANCE_LEVEL_COUNT];      // instance ID
  };

  /* Specialization for a single hit */
  template<>
    struct __aligned(16) HitK<1>
  {
     /* Default construction does nothing */
    __forceinline HitK() {}

    /* Constructs a hit */
    __forceinline HitK(const RTCIntersectContext* context, unsigned int geomID, unsigned int primID, float u, float v, const Vec3fa& Ng)
      : Ng(Ng.x,Ng.y,Ng.z), u(u), v(v), primID(primID), geomID(geomID)
    {
      instance_id_stack::copy_UU(context->instID, instID);
    }

    /* Returns the size of the hit */
    static __forceinline size_t size() { return 1; }

  public:
    Vec3<float> Ng;  // geometry normal
    float u;         // barycentric u coordinate of hit
    float v;         // barycentric v coordinate of hit
    unsigned int primID;      // primitive ID
    unsigned int geomID;      // geometry ID
    unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT];      // instance ID
  };

  /* Shortcuts */
  typedef HitK<1>  Hit;
  typedef HitK<4>  Hit4;
  typedef HitK<8>  Hit8;
  typedef HitK<16> Hit16;

  /* Outputs hit to stream */
  template<int K>
  __forceinline embree_ostream operator<<(embree_ostream cout, const HitK<K>& ray)
  {
    cout << "{ " << embree_endl
         << "  Ng = " << ray.Ng <<  embree_endl
         << "  u = " << ray.u <<  embree_endl
         << "  v = " << ray.v << embree_endl
         << "  primID = " << ray.primID <<  embree_endl
         << "  geomID = " << ray.geomID << embree_endl
         << "  instID =";
    for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l)
    {
      cout << " " << ray.instID[l];
    }
    cout << embree_endl;
    return cout << "}";
  }

  template<typename Hit>
    __forceinline void copyHitToRay(RayHit& ray, const Hit& hit)
  {
    ray.Ng   = hit.Ng;
    ray.u    = hit.u;
    ray.v    = hit.v;
    ray.primID = hit.primID;
    ray.geomID = hit.geomID;
    instance_id_stack::copy_UU(hit.instID, ray.instID);
  }

  template<int K>
    __forceinline void copyHitToRay(const vbool<K> &mask, RayHitK<K> &ray, const HitK<K> &hit)
  {
    vfloat<K>::storeu(mask,&ray.Ng.x, hit.Ng.x);
    vfloat<K>::storeu(mask,&ray.Ng.y, hit.Ng.y);
    vfloat<K>::storeu(mask,&ray.Ng.z, hit.Ng.z);
    vfloat<K>::storeu(mask,&ray.u, hit.u);
    vfloat<K>::storeu(mask,&ray.v, hit.v);
    vuint<K>::storeu(mask,&ray.primID, hit.primID);
    vuint<K>::storeu(mask,&ray.geomID, hit.geomID);
    instance_id_stack::copy_VV<K>(hit.instID, ray.instID, mask);
  }
}
