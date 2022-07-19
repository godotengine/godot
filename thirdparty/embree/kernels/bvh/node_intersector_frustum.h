// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_intersector.h"

namespace embree
{
  namespace isa
  {
    //////////////////////////////////////////////////////////////////////////////////////
    // Frustum structure used in hybrid and stream traversal
    //////////////////////////////////////////////////////////////////////////////////////

    /*
       Optimized frustum test. We calculate t=(p-org)/dir in ray/box
       intersection. We assume the rays are split by octant, thus
       dir intervals are either positive or negative in each
       dimension.

       Case 1: dir.min >= 0 && dir.max >= 0:
         t_min = (p_min - org_max) / dir_max = (p_min - org_max)*rdir_min = p_min*rdir_min - org_max*rdir_min
         t_max = (p_max - org_min) / dir_min = (p_max - org_min)*rdir_max = p_max*rdir_max - org_min*rdir_max

       Case 2: dir.min < 0 && dir.max < 0:
         t_min = (p_max - org_min) / dir_min = (p_max - org_min)*rdir_max = p_max*rdir_max - org_min*rdir_max
         t_max = (p_min - org_max) / dir_max = (p_min - org_max)*rdir_min = p_min*rdir_min - org_max*rdir_min
    */

    template<bool robust>
    struct Frustum;
    
    /* Fast variant */
    template<>
    struct Frustum<false>
    {
      __forceinline Frustum() {}

      template<int K>
      __forceinline Frustum(const vbool<K>& valid, const Vec3vf<K>& org, const Vec3vf<K>& rdir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar, int N)
      {
        init(valid, org, rdir, ray_tnear, ray_tfar, N);
      }

      template<int K>
      __forceinline void init(const vbool<K>& valid, const Vec3vf<K>& org, const Vec3vf<K>& rdir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar, int N)
      {
        const Vec3fa reduced_min_org(reduce_min(select(valid, org.x, pos_inf)),
                                     reduce_min(select(valid, org.y, pos_inf)),
                                     reduce_min(select(valid, org.z, pos_inf)));

        const Vec3fa reduced_max_org(reduce_max(select(valid, org.x, neg_inf)),
                                     reduce_max(select(valid, org.y, neg_inf)),
                                     reduce_max(select(valid, org.z, neg_inf)));

        const Vec3fa reduced_min_rdir(reduce_min(select(valid, rdir.x, pos_inf)),
                                      reduce_min(select(valid, rdir.y, pos_inf)),
                                      reduce_min(select(valid, rdir.z, pos_inf)));

        const Vec3fa reduced_max_rdir(reduce_max(select(valid, rdir.x, neg_inf)),
                                      reduce_max(select(valid, rdir.y, neg_inf)),
                                      reduce_max(select(valid, rdir.z, neg_inf)));

        const float reduced_min_dist = reduce_min(select(valid, ray_tnear, vfloat<K>(pos_inf)));
        const float reduced_max_dist = reduce_max(select(valid, ray_tfar , vfloat<K>(neg_inf)));

        init(reduced_min_org, reduced_max_org, reduced_min_rdir, reduced_max_rdir, reduced_min_dist, reduced_max_dist, N);
      }

      __forceinline void init(const Vec3fa& reduced_min_org,
                              const Vec3fa& reduced_max_org,
                              const Vec3fa& reduced_min_rdir,
                              const Vec3fa& reduced_max_rdir,
                              float reduced_min_dist,
                              float reduced_max_dist,
                              int N)
      {
        const Vec3ba pos_rdir = ge_mask(reduced_min_rdir, Vec3fa(zero));

        min_rdir = select(pos_rdir, reduced_min_rdir, reduced_max_rdir);
        max_rdir = select(pos_rdir, reduced_max_rdir, reduced_min_rdir);

#if defined (__aarch64__)
        neg_min_org_rdir = -(min_rdir * select(pos_rdir, reduced_max_org, reduced_min_org));
        neg_max_org_rdir = -(max_rdir * select(pos_rdir, reduced_min_org, reduced_max_org));
#else
        min_org_rdir = min_rdir * select(pos_rdir, reduced_max_org, reduced_min_org);
        max_org_rdir = max_rdir * select(pos_rdir, reduced_min_org, reduced_max_org);
#endif
        min_dist = reduced_min_dist;
        max_dist = reduced_max_dist;

        nf = NearFarPrecalculations(min_rdir, N);
      }

      template<int K>
      __forceinline void updateMaxDist(const vfloat<K>& ray_tfar)
      {
        max_dist = reduce_max(ray_tfar);
      }

      NearFarPrecalculations nf;

      Vec3fa min_rdir;
      Vec3fa max_rdir;

#if defined (__aarch64__)
      Vec3fa neg_min_org_rdir;
      Vec3fa neg_max_org_rdir;
#else
      Vec3fa min_org_rdir;
      Vec3fa max_org_rdir;
#endif
      float min_dist;
      float max_dist;
    };

    typedef Frustum<false> FrustumFast;

    /* Robust variant */
    template<>
    struct Frustum<true>
    {
      __forceinline Frustum() {}

      template<int K>
      __forceinline Frustum(const vbool<K>& valid, const Vec3vf<K>& org, const Vec3vf<K>& rdir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar, int N)
      {
        init(valid, org, rdir, ray_tnear, ray_tfar, N);
      }

      template<int K>
      __forceinline void init(const vbool<K>& valid, const Vec3vf<K>& org, const Vec3vf<K>& rdir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar, int N)
      {
        const Vec3fa reduced_min_org(reduce_min(select(valid, org.x, pos_inf)),
                                     reduce_min(select(valid, org.y, pos_inf)),
                                     reduce_min(select(valid, org.z, pos_inf)));

        const Vec3fa reduced_max_org(reduce_max(select(valid, org.x, neg_inf)),
                                     reduce_max(select(valid, org.y, neg_inf)),
                                     reduce_max(select(valid, org.z, neg_inf)));

        const Vec3fa reduced_min_rdir(reduce_min(select(valid, rdir.x, pos_inf)),
                                      reduce_min(select(valid, rdir.y, pos_inf)),
                                      reduce_min(select(valid, rdir.z, pos_inf)));

        const Vec3fa reduced_max_rdir(reduce_max(select(valid, rdir.x, neg_inf)),
                                      reduce_max(select(valid, rdir.y, neg_inf)),
                                      reduce_max(select(valid, rdir.z, neg_inf)));

        const float reduced_min_dist = reduce_min(select(valid, ray_tnear, vfloat<K>(pos_inf)));
        const float reduced_max_dist = reduce_max(select(valid, ray_tfar , vfloat<K>(neg_inf)));

        init(reduced_min_org, reduced_max_org, reduced_min_rdir, reduced_max_rdir, reduced_min_dist, reduced_max_dist, N);
      }

      __forceinline void init(const Vec3fa& reduced_min_org,
                              const Vec3fa& reduced_max_org,
                              const Vec3fa& reduced_min_rdir,
                              const Vec3fa& reduced_max_rdir,
                              float reduced_min_dist,
                              float reduced_max_dist,
                              int N)
      {
        const Vec3ba pos_rdir = ge_mask(reduced_min_rdir, Vec3fa(zero));
        min_rdir = select(pos_rdir, reduced_min_rdir, reduced_max_rdir);
        max_rdir = select(pos_rdir, reduced_max_rdir, reduced_min_rdir);

        min_org = select(pos_rdir, reduced_max_org, reduced_min_org);
        max_org = select(pos_rdir, reduced_min_org, reduced_max_org);

        min_dist = reduced_min_dist;
        max_dist = reduced_max_dist;

        nf = NearFarPrecalculations(min_rdir, N);
      }

      template<int K>
      __forceinline void updateMaxDist(const vfloat<K>& ray_tfar)
      {
        max_dist = reduce_max(ray_tfar);
      }

      NearFarPrecalculations nf;

      Vec3fa min_rdir;
      Vec3fa max_rdir;

      Vec3fa min_org;
      Vec3fa max_org;

      float min_dist;
      float max_dist;
    };

    typedef Frustum<true> FrustumRobust;

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx>
    __forceinline size_t intersectNodeFrustum(const typename BVHN<N>::AABBNode* __restrict__ node,
                                       const FrustumFast& frustum, vfloat<Nx>& dist)
    {
      const vfloat<Nx> bminX = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.nearX);
      const vfloat<Nx> bminY = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.nearY);
      const vfloat<Nx> bminZ = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.nearZ);
      const vfloat<Nx> bmaxX = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.farX);
      const vfloat<Nx> bmaxY = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.farY);
      const vfloat<Nx> bmaxZ = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.farZ);

#if defined (__aarch64__)
      const vfloat<Nx> fminX = madd(bminX, vfloat<Nx>(frustum.min_rdir.x), vfloat<Nx>(frustum.neg_min_org_rdir.x));
      const vfloat<Nx> fminY = madd(bminY, vfloat<Nx>(frustum.min_rdir.y), vfloat<Nx>(frustum.neg_min_org_rdir.y));
      const vfloat<Nx> fminZ = madd(bminZ, vfloat<Nx>(frustum.min_rdir.z), vfloat<Nx>(frustum.neg_min_org_rdir.z));
      const vfloat<Nx> fmaxX = madd(bmaxX, vfloat<Nx>(frustum.max_rdir.x), vfloat<Nx>(frustum.neg_max_org_rdir.x));
      const vfloat<Nx> fmaxY = madd(bmaxY, vfloat<Nx>(frustum.max_rdir.y), vfloat<Nx>(frustum.neg_max_org_rdir.y));
      const vfloat<Nx> fmaxZ = madd(bmaxZ, vfloat<Nx>(frustum.max_rdir.z), vfloat<Nx>(frustum.neg_max_org_rdir.z));
#else
      const vfloat<Nx> fminX = msub(bminX, vfloat<Nx>(frustum.min_rdir.x), vfloat<Nx>(frustum.min_org_rdir.x));
      const vfloat<Nx> fminY = msub(bminY, vfloat<Nx>(frustum.min_rdir.y), vfloat<Nx>(frustum.min_org_rdir.y));
      const vfloat<Nx> fminZ = msub(bminZ, vfloat<Nx>(frustum.min_rdir.z), vfloat<Nx>(frustum.min_org_rdir.z));
      const vfloat<Nx> fmaxX = msub(bmaxX, vfloat<Nx>(frustum.max_rdir.x), vfloat<Nx>(frustum.max_org_rdir.x));
      const vfloat<Nx> fmaxY = msub(bmaxY, vfloat<Nx>(frustum.max_rdir.y), vfloat<Nx>(frustum.max_org_rdir.y));
      const vfloat<Nx> fmaxZ = msub(bmaxZ, vfloat<Nx>(frustum.max_rdir.z), vfloat<Nx>(frustum.max_org_rdir.z));
#endif
      const vfloat<Nx> fmin  = maxi(fminX, fminY, fminZ, vfloat<Nx>(frustum.min_dist));
      dist = fmin;
      const vfloat<Nx> fmax  = mini(fmaxX, fmaxY, fmaxZ, vfloat<Nx>(frustum.max_dist));
      const vbool<Nx> vmask_node_hit = fmin <= fmax;
      size_t m_node = movemask(vmask_node_hit) & (((size_t)1 << N)-1);
      return m_node;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx>
    __forceinline size_t intersectNodeFrustum(const typename BVHN<N>::AABBNode* __restrict__ node,
                                       const FrustumRobust& frustum, vfloat<Nx>& dist)
    {
      const vfloat<Nx> bminX = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.nearX);
      const vfloat<Nx> bminY = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.nearY);
      const vfloat<Nx> bminZ = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.nearZ);
      const vfloat<Nx> bmaxX = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.farX);
      const vfloat<Nx> bmaxY = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.farY);
      const vfloat<Nx> bmaxZ = *(const vfloat<N>*)((const char*)&node->lower_x + frustum.nf.farZ);

      const vfloat<Nx> fminX = (bminX - vfloat<Nx>(frustum.min_org.x)) * vfloat<Nx>(frustum.min_rdir.x);
      const vfloat<Nx> fminY = (bminY - vfloat<Nx>(frustum.min_org.y)) * vfloat<Nx>(frustum.min_rdir.y);
      const vfloat<Nx> fminZ = (bminZ - vfloat<Nx>(frustum.min_org.z)) * vfloat<Nx>(frustum.min_rdir.z);
      const vfloat<Nx> fmaxX = (bmaxX - vfloat<Nx>(frustum.max_org.x)) * vfloat<Nx>(frustum.max_rdir.x);
      const vfloat<Nx> fmaxY = (bmaxY - vfloat<Nx>(frustum.max_org.y)) * vfloat<Nx>(frustum.max_rdir.y);
      const vfloat<Nx> fmaxZ = (bmaxZ - vfloat<Nx>(frustum.max_org.z)) * vfloat<Nx>(frustum.max_rdir.z);

      const float round_down = 1.0f-2.0f*float(ulp); // FIXME: use per instruction rounding for AVX512
      const float round_up   = 1.0f+2.0f*float(ulp);
      const vfloat<Nx> fmin  = max(fminX, fminY, fminZ, vfloat<Nx>(frustum.min_dist));
      dist = fmin;
      const vfloat<Nx> fmax  = min(fmaxX, fmaxY, fmaxZ, vfloat<Nx>(frustum.max_dist));
      const vbool<Nx> vmask_node_hit = (round_down*fmin <= round_up*fmax);
      size_t m_node = movemask(vmask_node_hit) & (((size_t)1 << N)-1);
      return m_node;
    }
  }
}
