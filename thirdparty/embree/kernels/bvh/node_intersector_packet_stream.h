// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_intersector.h"

namespace embree
{
  namespace isa
  {
    //////////////////////////////////////////////////////////////////////////////////////
    // Ray packet structure used in stream traversal
    //////////////////////////////////////////////////////////////////////////////////////

    template<int K, bool robust>
    struct TravRayKStream;

    /* Fast variant */
    template<int K>
    struct TravRayKStream<K, false>
    {
      __forceinline TravRayKStream() {}

      __forceinline TravRayKStream(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar)
      {
        init(ray_org, ray_dir);
        tnear = ray_tnear;
        tfar = ray_tfar;
      }

      __forceinline void init(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir)
      {
        rdir = rcp_safe(ray_dir);
        org_rdir = ray_org * rdir;
      }

      Vec3vf<K> rdir;
      Vec3vf<K> org_rdir;
      vfloat<K> tnear;
      vfloat<K> tfar;
    };

    template<int K>
    using TravRayKStreamFast = TravRayKStream<K, false>;

    /* Robust variant */
    template<int K>
    struct TravRayKStream<K, true>
    {
      __forceinline TravRayKStream() {}

      __forceinline TravRayKStream(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar)
      {
        init(ray_org, ray_dir);
        tnear = ray_tnear;
        tfar = ray_tfar;
      }

      __forceinline void init(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir)
      {
        rdir = vfloat<K>(1.0f)/(zero_fix(ray_dir));
        org = ray_org;
      }

      Vec3vf<K> rdir;
      Vec3vf<K> org;
      vfloat<K> tnear;
      vfloat<K> tfar;
    };

    template<int K>
    using TravRayKStreamRobust = TravRayKStream<K, true>;

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline size_t intersectNode1(const typename BVHN<N>::AABBNode* __restrict__ node,
                                        const TravRayKStreamFast<K>& ray, size_t k, const NearFarPrecalculations& nf)
    {
      const vfloat<N> bminX = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearX));
      const vfloat<N> bminY = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearY));
      const vfloat<N> bminZ = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearZ));
      const vfloat<N> bmaxX = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farX));
      const vfloat<N> bmaxY = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farY));
      const vfloat<N> bmaxZ = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farZ));

      const vfloat<N> rminX = msub(bminX, vfloat<N>(ray.rdir.x[k]), vfloat<N>(ray.org_rdir.x[k]));
      const vfloat<N> rminY = msub(bminY, vfloat<N>(ray.rdir.y[k]), vfloat<N>(ray.org_rdir.y[k]));
      const vfloat<N> rminZ = msub(bminZ, vfloat<N>(ray.rdir.z[k]), vfloat<N>(ray.org_rdir.z[k]));
      const vfloat<N> rmaxX = msub(bmaxX, vfloat<N>(ray.rdir.x[k]), vfloat<N>(ray.org_rdir.x[k]));
      const vfloat<N> rmaxY = msub(bmaxY, vfloat<N>(ray.rdir.y[k]), vfloat<N>(ray.org_rdir.y[k]));
      const vfloat<N> rmaxZ = msub(bmaxZ, vfloat<N>(ray.rdir.z[k]), vfloat<N>(ray.org_rdir.z[k]));
      const vfloat<N> rmin  = maxi(rminX, rminY, rminZ, vfloat<N>(ray.tnear[k]));
      const vfloat<N> rmax  = mini(rmaxX, rmaxY, rmaxZ, vfloat<N>(ray.tfar[k]));

      const vbool<N> vmask_first_hit = rmin <= rmax;

      return movemask(vmask_first_hit) & (((size_t)1 << N)-1);
    }

    template<int N, int K>
    __forceinline size_t intersectNodeK(const typename BVHN<N>::AABBNode* __restrict__ node, size_t i,
                                        const TravRayKStreamFast<K>& ray, const NearFarPrecalculations& nf)
    {
      char* ptr = (char*)&node->lower_x + i*sizeof(float);
      const vfloat<K> bminX = *(const float*)(ptr + nf.nearX);
      const vfloat<K> bminY = *(const float*)(ptr + nf.nearY);
      const vfloat<K> bminZ = *(const float*)(ptr + nf.nearZ);
      const vfloat<K> bmaxX = *(const float*)(ptr + nf.farX);
      const vfloat<K> bmaxY = *(const float*)(ptr + nf.farY);
      const vfloat<K> bmaxZ = *(const float*)(ptr + nf.farZ);

      const vfloat<K> rminX = msub(bminX, ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> rminY = msub(bminY, ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> rminZ = msub(bminZ, ray.rdir.z, ray.org_rdir.z);
      const vfloat<K> rmaxX = msub(bmaxX, ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> rmaxY = msub(bmaxY, ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> rmaxZ = msub(bmaxZ, ray.rdir.z, ray.org_rdir.z);

      const vfloat<K> rmin  = maxi(rminX, rminY, rminZ, ray.tnear);
      const vfloat<K> rmax  = mini(rmaxX, rmaxY, rmaxZ, ray.tfar);

      const vbool<K> vmask_first_hit = rmin <= rmax;

      return movemask(vmask_first_hit);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline size_t intersectNode1(const typename BVHN<N>::AABBNode* __restrict__ node,
                                        const TravRayKStreamRobust<K>& ray, size_t k, const NearFarPrecalculations& nf)
    {
      const vfloat<N> bminX = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearX));
      const vfloat<N> bminY = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearY));
      const vfloat<N> bminZ = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearZ));
      const vfloat<N> bmaxX = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farX));
      const vfloat<N> bmaxY = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farY));
      const vfloat<N> bmaxZ = vfloat<N>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farZ));

      const vfloat<N> rminX = (bminX - vfloat<N>(ray.org.x[k])) * vfloat<N>(ray.rdir.x[k]);
      const vfloat<N> rminY = (bminY - vfloat<N>(ray.org.y[k])) * vfloat<N>(ray.rdir.y[k]);
      const vfloat<N> rminZ = (bminZ - vfloat<N>(ray.org.z[k])) * vfloat<N>(ray.rdir.z[k]);
      const vfloat<N> rmaxX = (bmaxX - vfloat<N>(ray.org.x[k])) * vfloat<N>(ray.rdir.x[k]);
      const vfloat<N> rmaxY = (bmaxY - vfloat<N>(ray.org.y[k])) * vfloat<N>(ray.rdir.y[k]);
      const vfloat<N> rmaxZ = (bmaxZ - vfloat<N>(ray.org.z[k])) * vfloat<N>(ray.rdir.z[k]);
      const float round_up = 1.0f+3.0f*float(ulp); // FIXME: use per instruction rounding for AVX512
      const vfloat<N> rmin  =            max(rminX, rminY, rminZ, vfloat<N>(ray.tnear[k]));
      const vfloat<N> rmax  = round_up  *min(rmaxX, rmaxY, rmaxZ, vfloat<N>(ray.tfar[k]));

      const vbool<N> vmask_first_hit = rmin <= rmax;

      return movemask(vmask_first_hit) & (((size_t)1 << N)-1);
    }

    template<int N, int K>
    __forceinline size_t intersectNodeK(const typename BVHN<N>::AABBNode* __restrict__ node, size_t i,
                                        const TravRayKStreamRobust<K>& ray, const NearFarPrecalculations& nf)
    {
      char *ptr = (char*)&node->lower_x + i*sizeof(float);
      const vfloat<K> bminX = *(const float*)(ptr + nf.nearX);
      const vfloat<K> bminY = *(const float*)(ptr + nf.nearY);
      const vfloat<K> bminZ = *(const float*)(ptr + nf.nearZ);
      const vfloat<K> bmaxX = *(const float*)(ptr + nf.farX);
      const vfloat<K> bmaxY = *(const float*)(ptr + nf.farY);
      const vfloat<K> bmaxZ = *(const float*)(ptr + nf.farZ);

      const vfloat<K> rminX = (bminX - ray.org.x) * ray.rdir.x;
      const vfloat<K> rminY = (bminY - ray.org.y) * ray.rdir.y;
      const vfloat<K> rminZ = (bminZ - ray.org.z) * ray.rdir.z;
      const vfloat<K> rmaxX = (bmaxX - ray.org.x) * ray.rdir.x;
      const vfloat<K> rmaxY = (bmaxY - ray.org.y) * ray.rdir.y;
      const vfloat<K> rmaxZ = (bmaxZ - ray.org.z) * ray.rdir.z;

      const float round_up  = 1.0f+3.0f*float(ulp);
      const vfloat<K> rmin  =            max(rminX, rminY, rminZ, vfloat<K>(ray.tnear));
      const vfloat<K> rmax  = round_up * min(rmaxX, rmaxY, rmaxZ, vfloat<K>(ray.tfar));

      const vbool<K> vmask_first_hit = rmin <= rmax;

      return movemask(vmask_first_hit);
    }
  }
}
