// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_intersector.h"

namespace embree
{
  namespace isa
  {
    //////////////////////////////////////////////////////////////////////////////////////
    // Ray packet structure used in hybrid traversal
    //////////////////////////////////////////////////////////////////////////////////////

    template<int K, bool robust>
    struct TravRayK;

    /* Fast variant */
    template<int K>
    struct TravRayK<K, false>
    {
      __forceinline TravRayK() {}

      __forceinline TravRayK(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, int N)
      {
        init(ray_org, ray_dir, N);
      }

      __forceinline TravRayK(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar, int N)
      {
        init(ray_org, ray_dir, N);
        tnear = ray_tnear;
        tfar = ray_tfar;
      }

      __forceinline void init(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, int N)
      {
        org = ray_org;
        dir = ray_dir;
        rdir = rcp_safe(ray_dir);
#if defined(__aarch64__)
        neg_org_rdir = -(org * rdir);
#elif defined(__AVX2__)
        org_rdir = org * rdir;
#endif

        if (N)
        {
          const int size = sizeof(float)*N;
          nearXYZ.x = select(rdir.x >= 0.0f, vint<K>(0*size), vint<K>(1*size));
          nearXYZ.y = select(rdir.y >= 0.0f, vint<K>(2*size), vint<K>(3*size));
          nearXYZ.z = select(rdir.z >= 0.0f, vint<K>(4*size), vint<K>(5*size));
        }
      }

      Vec3vf<K> org;
      Vec3vf<K> dir;
      Vec3vf<K> rdir;
#if defined(__aarch64__)
      Vec3vf<K> neg_org_rdir;
#elif defined(__AVX2__)
      Vec3vf<K> org_rdir;
#endif
      Vec3vi<K> nearXYZ;
      vfloat<K> tnear;
      vfloat<K> tfar;
    };

    template<int K>
    using TravRayKFast = TravRayK<K, false>;

    /* Robust variant */
    template<int K>
    struct TravRayK<K, true>
    {
      __forceinline TravRayK() {}

      __forceinline TravRayK(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, int N)
      {
        init(ray_org, ray_dir, N);
      }

      __forceinline TravRayK(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, const vfloat<K>& ray_tnear, const vfloat<K>& ray_tfar, int N)
      {
        init(ray_org, ray_dir, N);
        tnear = ray_tnear;
        tfar = ray_tfar;
      }

      __forceinline void init(const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir, int N)
      {
        org = ray_org;
        dir = ray_dir;
        rdir = vfloat<K>(1.0f)/(zero_fix(ray_dir));

        if (N)
        {
          const int size = sizeof(float)*N;
          nearXYZ.x = select(rdir.x >= 0.0f, vint<K>(0*size), vint<K>(1*size));
          nearXYZ.y = select(rdir.y >= 0.0f, vint<K>(2*size), vint<K>(3*size));
          nearXYZ.z = select(rdir.z >= 0.0f, vint<K>(4*size), vint<K>(5*size));
        }
      }

      Vec3vf<K> org;
      Vec3vf<K> dir;
      Vec3vf<K> rdir;
      Vec3vi<K> nearXYZ;
      vfloat<K> tnear;
      vfloat<K> tfar;
    };

    template<int K>
    using TravRayKRobust = TravRayK<K, true>;

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectNodeK(const typename BVHN<N>::AABBNode* node, size_t i,
                                         const TravRayKFast<K>& ray, vfloat<K>& dist)

    {
#if defined(__aarch64__)
      const vfloat<K> lclipMinX = madd(node->lower_x[i], ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMinY = madd(node->lower_y[i], ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMinZ = madd(node->lower_z[i], ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<K> lclipMaxX = madd(node->upper_x[i], ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMaxY = madd(node->upper_y[i], ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMaxZ = madd(node->upper_z[i], ray.rdir.z, ray.neg_org_rdir.z);
#elif defined(__AVX2__)
      const vfloat<K> lclipMinX = msub(node->lower_x[i], ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMinY = msub(node->lower_y[i], ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMinZ = msub(node->lower_z[i], ray.rdir.z, ray.org_rdir.z);
      const vfloat<K> lclipMaxX = msub(node->upper_x[i], ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMaxY = msub(node->upper_y[i], ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMaxZ = msub(node->upper_z[i], ray.rdir.z, ray.org_rdir.z);
  #else
      const vfloat<K> lclipMinX = (node->lower_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (node->lower_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (node->lower_z[i] - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (node->upper_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (node->upper_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (node->upper_z[i] - ray.org.z) * ray.rdir.z;
  #endif

  #if defined(__AVX512F__) // SKX
      if (K == 16)
      {
        /* use mixed float/int min/max */
        const vfloat<K> lnearP = maxi(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = mini(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
        const vbool<K> lhit    = asInt(maxi(lnearP, ray.tnear)) <= asInt(mini(lfarP, ray.tfar));
        dist = lnearP;
        return lhit;
      }
      else
  #endif
      {
        const vfloat<K> lnearP = maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY), mini(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY), maxi(lclipMinZ, lclipMaxZ));
  #if defined(__AVX512F__) // SKX
        const vbool<K> lhit    = asInt(maxi(lnearP, ray.tnear)) <= asInt(mini(lfarP, ray.tfar));
  #else
        const vbool<K> lhit    = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
  #endif
        dist = lnearP;
        return lhit;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectNodeKRobust(const typename BVHN<N>::AABBNode* node, size_t i,
                                               const TravRayKRobust<K>& ray, vfloat<K>& dist)
    {
      // FIXME: use per instruction rounding for AVX512
      const vfloat<K> lclipMinX = (node->lower_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (node->lower_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (node->lower_z[i] - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (node->upper_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (node->upper_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (node->upper_z[i] - ray.org.z) * ray.rdir.z;
      const float round_up   = 1.0f+3.0f*float(ulp);
      const float round_down = 1.0f-3.0f*float(ulp);
      const vfloat<K> lnearP = round_down*max(max(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY)), min(lclipMinZ, lclipMaxZ));
      const vfloat<K> lfarP  = round_up  *min(min(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY)), max(lclipMinZ, lclipMaxZ));
      const vbool<K> lhit   = max(lnearP, ray.tnear) <= min(lfarP, ray.tfar);
      dist = lnearP;
      return lhit;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectNodeK(const typename BVHN<N>::AABBNodeMB* node, const size_t i,
                                         const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist)
    {
      const vfloat<K> vlower_x = madd(time, vfloat<K>(node->lower_dx[i]), vfloat<K>(node->lower_x[i]));
      const vfloat<K> vlower_y = madd(time, vfloat<K>(node->lower_dy[i]), vfloat<K>(node->lower_y[i]));
      const vfloat<K> vlower_z = madd(time, vfloat<K>(node->lower_dz[i]), vfloat<K>(node->lower_z[i]));
      const vfloat<K> vupper_x = madd(time, vfloat<K>(node->upper_dx[i]), vfloat<K>(node->upper_x[i]));
      const vfloat<K> vupper_y = madd(time, vfloat<K>(node->upper_dy[i]), vfloat<K>(node->upper_y[i]));
      const vfloat<K> vupper_z = madd(time, vfloat<K>(node->upper_dz[i]), vfloat<K>(node->upper_z[i]));

#if defined(__aarch64__)
      const vfloat<K> lclipMinX = madd(vlower_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMinY = madd(vlower_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMinZ = madd(vlower_z, ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<K> lclipMaxX = madd(vupper_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMaxY = madd(vupper_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMaxZ = madd(vupper_z, ray.rdir.z, ray.neg_org_rdir.z);
#elif defined(__AVX2__)
      const vfloat<K> lclipMinX = msub(vlower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMinY = msub(vlower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMinZ = msub(vlower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat<K> lclipMaxX = msub(vupper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMaxY = msub(vupper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMaxZ = msub(vupper_z, ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat<K> lclipMinX = (vlower_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (vlower_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (vlower_z - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (vupper_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (vupper_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (vupper_z - ray.org.z) * ray.rdir.z;
#endif

#if defined(__AVX512F__) // SKX
      if (K == 16)
      {
        /* use mixed float/int min/max */
        const vfloat<K> lnearP = maxi(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = mini(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
        const vbool<K> lhit    = asInt(maxi(lnearP, ray.tnear)) <= asInt(mini(lfarP, ray.tfar));
        dist = lnearP;
        return lhit;
      }
      else
#endif
      {
        const vfloat<K> lnearP = maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY), mini(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY), maxi(lclipMinZ, lclipMaxZ));
#if defined(__AVX512F__) // SKX
        const vbool<K> lhit    = asInt(maxi(lnearP, ray.tnear)) <= asInt(mini(lfarP, ray.tfar));
#else
        const vbool<K> lhit    = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
#endif
        dist = lnearP;
        return lhit;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectNodeKRobust(const typename BVHN<N>::AABBNodeMB* node, const size_t i,
                                               const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist)
    {
      const vfloat<K> vlower_x = madd(time, vfloat<K>(node->lower_dx[i]), vfloat<K>(node->lower_x[i]));
      const vfloat<K> vlower_y = madd(time, vfloat<K>(node->lower_dy[i]), vfloat<K>(node->lower_y[i]));
      const vfloat<K> vlower_z = madd(time, vfloat<K>(node->lower_dz[i]), vfloat<K>(node->lower_z[i]));
      const vfloat<K> vupper_x = madd(time, vfloat<K>(node->upper_dx[i]), vfloat<K>(node->upper_x[i]));
      const vfloat<K> vupper_y = madd(time, vfloat<K>(node->upper_dy[i]), vfloat<K>(node->upper_y[i]));
      const vfloat<K> vupper_z = madd(time, vfloat<K>(node->upper_dz[i]), vfloat<K>(node->upper_z[i]));

      const vfloat<K> lclipMinX = (vlower_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (vlower_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (vlower_z - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (vupper_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (vupper_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (vupper_z - ray.org.z) * ray.rdir.z;

      const float round_up   = 1.0f+3.0f*float(ulp);
      const float round_down = 1.0f-3.0f*float(ulp);

#if defined(__AVX512F__) // SKX
      if (K == 16)
      {
        const vfloat<K> lnearP = round_down*maxi(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = round_up  *mini(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
        const vbool<K>  lhit   = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
        dist = lnearP;
        return lhit;
      }
      else
#endif
      {
        const vfloat<K> lnearP = round_down*maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY), mini(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = round_up  *mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY), maxi(lclipMinZ, lclipMaxZ));
        const vbool<K>  lhit   = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
        dist = lnearP;
        return lhit;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNodeMB4D intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectNodeKMB4D(const typename BVHN<N>::NodeRef ref, const size_t i,
                                             const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist)
    {
      const typename BVHN<N>::AABBNodeMB* node = ref.getAABBNodeMB();

      const vfloat<K> vlower_x = madd(time, vfloat<K>(node->lower_dx[i]), vfloat<K>(node->lower_x[i]));
      const vfloat<K> vlower_y = madd(time, vfloat<K>(node->lower_dy[i]), vfloat<K>(node->lower_y[i]));
      const vfloat<K> vlower_z = madd(time, vfloat<K>(node->lower_dz[i]), vfloat<K>(node->lower_z[i]));
      const vfloat<K> vupper_x = madd(time, vfloat<K>(node->upper_dx[i]), vfloat<K>(node->upper_x[i]));
      const vfloat<K> vupper_y = madd(time, vfloat<K>(node->upper_dy[i]), vfloat<K>(node->upper_y[i]));
      const vfloat<K> vupper_z = madd(time, vfloat<K>(node->upper_dz[i]), vfloat<K>(node->upper_z[i]));

#if defined(__aarch64__)
      const vfloat<K> lclipMinX = madd(vlower_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMinY = madd(vlower_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMinZ = madd(vlower_z, ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<K> lclipMaxX = madd(vupper_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMaxY = madd(vupper_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMaxZ = madd(vupper_z, ray.rdir.z, ray.neg_org_rdir.z);
#elif defined(__AVX2__)
      const vfloat<K> lclipMinX = msub(vlower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMinY = msub(vlower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMinZ = msub(vlower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat<K> lclipMaxX = msub(vupper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMaxY = msub(vupper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMaxZ = msub(vupper_z, ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat<K> lclipMinX = (vlower_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (vlower_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (vlower_z - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (vupper_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (vupper_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (vupper_z - ray.org.z) * ray.rdir.z;
#endif

      const vfloat<K> lnearP = maxi(maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY)), mini(lclipMinZ, lclipMaxZ));
      const vfloat<K> lfarP  = mini(mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY)), maxi(lclipMinZ, lclipMaxZ));
      vbool<K> lhit = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
      if (unlikely(ref.isAABBNodeMB4D())) {
        const typename BVHN<N>::AABBNodeMB4D* node1 = (const typename BVHN<N>::AABBNodeMB4D*) node;
        lhit = lhit & (vfloat<K>(node1->lower_t[i]) <= time) & (time < vfloat<K>(node1->upper_t[i]));
      }
      dist = lnearP;
      return lhit;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNodeMB4D intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectNodeKMB4DRobust(const typename BVHN<N>::NodeRef ref, const size_t i,
                                                    const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist)
    {
      const typename BVHN<N>::AABBNodeMB* node = ref.getAABBNodeMB();

      const vfloat<K> vlower_x = madd(time, vfloat<K>(node->lower_dx[i]), vfloat<K>(node->lower_x[i]));
      const vfloat<K> vlower_y = madd(time, vfloat<K>(node->lower_dy[i]), vfloat<K>(node->lower_y[i]));
      const vfloat<K> vlower_z = madd(time, vfloat<K>(node->lower_dz[i]), vfloat<K>(node->lower_z[i]));
      const vfloat<K> vupper_x = madd(time, vfloat<K>(node->upper_dx[i]), vfloat<K>(node->upper_x[i]));
      const vfloat<K> vupper_y = madd(time, vfloat<K>(node->upper_dy[i]), vfloat<K>(node->upper_y[i]));
      const vfloat<K> vupper_z = madd(time, vfloat<K>(node->upper_dz[i]), vfloat<K>(node->upper_z[i]));

      const vfloat<K> lclipMinX = (vlower_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (vlower_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (vlower_z - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (vupper_x - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (vupper_y - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (vupper_z - ray.org.z) * ray.rdir.z;

      const float round_up   = 1.0f+3.0f*float(ulp);
      const float round_down = 1.0f-3.0f*float(ulp);
      const vfloat<K> lnearP = round_down*maxi(maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY)), mini(lclipMinZ, lclipMaxZ));
      const vfloat<K> lfarP  = round_up  *mini(mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY)), maxi(lclipMinZ, lclipMaxZ));
      vbool<K> lhit = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);

      if (unlikely(ref.isAABBNodeMB4D())) {
        const typename BVHN<N>::AABBNodeMB4D* node1 = (const typename BVHN<N>::AABBNodeMB4D*) node;
        lhit = lhit & (vfloat<K>(node1->lower_t[i]) <= time) & (time < vfloat<K>(node1->upper_t[i]));
      }
      dist = lnearP;
      return lhit;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast OBBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K, bool robust>
    __forceinline vbool<K> intersectNodeK(const typename BVHN<N>::OBBNode* node, const size_t i,
                                          const TravRayK<K,robust>& ray, vfloat<K>& dist)
    {
      const AffineSpace3vf<K> naabb(Vec3f(node->naabb.l.vx.x[i], node->naabb.l.vx.y[i], node->naabb.l.vx.z[i]),
                                    Vec3f(node->naabb.l.vy.x[i], node->naabb.l.vy.y[i], node->naabb.l.vy.z[i]),
                                    Vec3f(node->naabb.l.vz.x[i], node->naabb.l.vz.y[i], node->naabb.l.vz.z[i]),
                                    Vec3f(node->naabb.p   .x[i], node->naabb.p   .y[i], node->naabb.p   .z[i]));

      const Vec3vf<K> dir = xfmVector(naabb, ray.dir);
      const Vec3vf<K> nrdir = Vec3vf<K>(vfloat<K>(-1.0f)) * rcp_safe(dir); // FIXME: negate instead of mul with -1?
      const Vec3vf<K> org = xfmPoint(naabb, ray.org);

      const vfloat<K> lclipMinX = org.x * nrdir.x; // (Vec3fa(zero) - org) * rdir;
      const vfloat<K> lclipMinY = org.y * nrdir.y;
      const vfloat<K> lclipMinZ = org.z * nrdir.z;
      const vfloat<K> lclipMaxX  = lclipMinX - nrdir.x; // (Vec3fa(one) - org) * rdir;
      const vfloat<K> lclipMaxY  = lclipMinY - nrdir.y;
      const vfloat<K> lclipMaxZ  = lclipMinZ - nrdir.z;

      vfloat<K> lnearP = maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY), mini(lclipMinZ, lclipMaxZ));
      vfloat<K> lfarP  = mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY), maxi(lclipMinZ, lclipMaxZ));
      if (robust) {
        lnearP = lnearP*vfloat<K>(1.0f-3.0f*float(ulp));
        lfarP  = lfarP *vfloat<K>(1.0f+3.0f*float(ulp));
      }
      const vbool<K> lhit    = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
      dist = lnearP;
      return lhit;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast OBBNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K, bool robust>
    __forceinline vbool<K> intersectNodeK(const typename BVHN<N>::OBBNodeMB* node, const size_t i,
                                          const TravRayK<K,robust>& ray, const vfloat<K>& time, vfloat<K>& dist)
    {
      const AffineSpace3vf<K> xfm(Vec3f(node->space0.l.vx.x[i], node->space0.l.vx.y[i], node->space0.l.vx.z[i]),
                                  Vec3f(node->space0.l.vy.x[i], node->space0.l.vy.y[i], node->space0.l.vy.z[i]),
                                  Vec3f(node->space0.l.vz.x[i], node->space0.l.vz.y[i], node->space0.l.vz.z[i]),
                                  Vec3f(node->space0.p   .x[i], node->space0.p   .y[i], node->space0.p   .z[i]));

      const Vec3vf<K> b0_lower = zero;
      const Vec3vf<K> b0_upper = one;
      const Vec3vf<K> b1_lower(node->b1.lower.x[i], node->b1.lower.y[i], node->b1.lower.z[i]);
      const Vec3vf<K> b1_upper(node->b1.upper.x[i], node->b1.upper.y[i], node->b1.upper.z[i]);
      const Vec3vf<K> lower = lerp(b0_lower, b1_lower, time);
      const Vec3vf<K> upper = lerp(b0_upper, b1_upper, time);

      const Vec3vf<K> dir = xfmVector(xfm, ray.dir);
      const Vec3vf<K> rdir = rcp_safe(dir);
      const Vec3vf<K> org = xfmPoint(xfm, ray.org);

      const vfloat<K> lclipMinX = (lower.x - org.x) * rdir.x;
      const vfloat<K> lclipMinY = (lower.y - org.y) * rdir.y;
      const vfloat<K> lclipMinZ = (lower.z - org.z) * rdir.z;
      const vfloat<K> lclipMaxX  = (upper.x - org.x) * rdir.x;
      const vfloat<K> lclipMaxY  = (upper.y - org.y) * rdir.y;
      const vfloat<K> lclipMaxZ  = (upper.z - org.z) * rdir.z;

      vfloat<K> lnearP = maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY), mini(lclipMinZ, lclipMaxZ));
      vfloat<K> lfarP  = mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY), maxi(lclipMinZ, lclipMaxZ));
      if (robust) {
        lnearP = lnearP*vfloat<K>(1.0f-3.0f*float(ulp));
        lfarP  = lfarP *vfloat<K>(1.0f+3.0f*float(ulp));
      }
        
      const vbool<K> lhit    = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
      dist = lnearP;
      return lhit;
    }



    //////////////////////////////////////////////////////////////////////////////////////
    // QuantizedBaseNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int K>
    __forceinline vbool<K> intersectQuantizedNodeK(const typename BVHN<N>::QuantizedBaseNode* node, size_t i,
                                                   const TravRayK<K,false>& ray, vfloat<K>& dist)

    {
      assert(movemask(node->validMask()) & ((size_t)1 << i));
      const vfloat<N> lower_x = node->dequantizeLowerX();
      const vfloat<N> upper_x = node->dequantizeUpperX();
      const vfloat<N> lower_y = node->dequantizeLowerY();
      const vfloat<N> upper_y = node->dequantizeUpperY();
      const vfloat<N> lower_z = node->dequantizeLowerZ();
      const vfloat<N> upper_z = node->dequantizeUpperZ();

  #if defined(__aarch64__)
      const vfloat<K> lclipMinX = madd(lower_x[i], ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMinY = madd(lower_y[i], ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMinZ = madd(lower_z[i], ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<K> lclipMaxX = madd(upper_x[i], ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<K> lclipMaxY = madd(upper_y[i], ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<K> lclipMaxZ = madd(upper_z[i], ray.rdir.z, ray.neg_org_rdir.z);
  #elif defined(__AVX2__)
      const vfloat<K> lclipMinX = msub(lower_x[i], ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMinY = msub(lower_y[i], ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMinZ = msub(lower_z[i], ray.rdir.z, ray.org_rdir.z);
      const vfloat<K> lclipMaxX = msub(upper_x[i], ray.rdir.x, ray.org_rdir.x);
      const vfloat<K> lclipMaxY = msub(upper_y[i], ray.rdir.y, ray.org_rdir.y);
      const vfloat<K> lclipMaxZ = msub(upper_z[i], ray.rdir.z, ray.org_rdir.z);
  #else
      const vfloat<K> lclipMinX = (lower_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (lower_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (lower_z[i] - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (upper_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (upper_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (upper_z[i] - ray.org.z) * ray.rdir.z;
  #endif

  #if defined(__AVX512F__) // SKX
      if (K == 16)
      {
        /* use mixed float/int min/max */
        const vfloat<K> lnearP = maxi(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = mini(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
        const vbool<K> lhit    = asInt(maxi(lnearP, ray.tnear)) <= asInt(mini(lfarP, ray.tfar));
        dist = lnearP;
        return lhit;
      }
      else
  #endif
      {
        const vfloat<K> lnearP = maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY), mini(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY), maxi(lclipMinZ, lclipMaxZ));
  #if defined(__AVX512F__) // SKX
        const vbool<K> lhit    = asInt(maxi(lnearP, ray.tnear)) <= asInt(mini(lfarP, ray.tfar));
  #else
        const vbool<K> lhit    = maxi(lnearP, ray.tnear) <= mini(lfarP, ray.tfar);
  #endif
        dist = lnearP;
        return lhit;
      }
    }

    template<int N, int K>
    __forceinline vbool<K> intersectQuantizedNodeK(const typename BVHN<N>::QuantizedBaseNode* node, size_t i,
          const TravRayK<K,true>& ray, vfloat<K>& dist)

    {
      assert(movemask(node->validMask()) & ((size_t)1 << i));
      const vfloat<N> lower_x = node->dequantizeLowerX();
      const vfloat<N> upper_x = node->dequantizeUpperX();
      const vfloat<N> lower_y = node->dequantizeLowerY();
      const vfloat<N> upper_y = node->dequantizeUpperY();
      const vfloat<N> lower_z = node->dequantizeLowerZ();
      const vfloat<N> upper_z = node->dequantizeUpperZ();

      const vfloat<K> lclipMinX = (lower_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMinY = (lower_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMinZ = (lower_z[i] - ray.org.z) * ray.rdir.z;
      const vfloat<K> lclipMaxX = (upper_x[i] - ray.org.x) * ray.rdir.x;
      const vfloat<K> lclipMaxY = (upper_y[i] - ray.org.y) * ray.rdir.y;
      const vfloat<K> lclipMaxZ = (upper_z[i] - ray.org.z) * ray.rdir.z;

      const float round_up   = 1.0f+3.0f*float(ulp);
      const float round_down = 1.0f-3.0f*float(ulp);

      const vfloat<K> lnearP = round_down*max(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
      const vfloat<K> lfarP  = round_up  *min(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
      const vbool<K> lhit    = max(lnearP, ray.tnear) <= min(lfarP, ray.tfar);
      dist = lnearP;
      return lhit;
      }

    template<int N, int K>
      __forceinline vbool<K> intersectQuantizedNodeMBK(const typename BVHN<N>::QuantizedBaseNodeMB* node, const size_t i,
          const TravRayK<K,false>& ray, const vfloat<K>& time, vfloat<K>& dist)

    {
        assert(movemask(node->validMask()) & ((size_t)1 << i));

        const vfloat<K> lower_x = node->template dequantizeLowerX<K>(i,time);
        const vfloat<K> upper_x = node->template dequantizeUpperX<K>(i,time);
        const vfloat<K> lower_y = node->template dequantizeLowerY<K>(i,time);
        const vfloat<K> upper_y = node->template dequantizeUpperY<K>(i,time);
        const vfloat<K> lower_z = node->template dequantizeLowerZ<K>(i,time);
        const vfloat<K> upper_z = node->template dequantizeUpperZ<K>(i,time);
        
#if defined(__aarch64__)
        const vfloat<K> lclipMinX = madd(lower_x, ray.rdir.x, ray.neg_org_rdir.x);
        const vfloat<K> lclipMinY = madd(lower_y, ray.rdir.y, ray.neg_org_rdir.y);
        const vfloat<K> lclipMinZ = madd(lower_z, ray.rdir.z, ray.neg_org_rdir.z);
        const vfloat<K> lclipMaxX = madd(upper_x, ray.rdir.x, ray.neg_org_rdir.x);
        const vfloat<K> lclipMaxY = madd(upper_y, ray.rdir.y, ray.neg_org_rdir.y);
        const vfloat<K> lclipMaxZ = madd(upper_z, ray.rdir.z, ray.neg_org_rdir.z);
#elif defined(__AVX2__)
        const vfloat<K> lclipMinX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
        const vfloat<K> lclipMinY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
        const vfloat<K> lclipMinZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
        const vfloat<K> lclipMaxX = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
        const vfloat<K> lclipMaxY = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
        const vfloat<K> lclipMaxZ = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#else
        const vfloat<K> lclipMinX = (lower_x - ray.org.x) * ray.rdir.x;
        const vfloat<K> lclipMinY = (lower_y - ray.org.y) * ray.rdir.y;
        const vfloat<K> lclipMinZ = (lower_z - ray.org.z) * ray.rdir.z;
        const vfloat<K> lclipMaxX = (upper_x - ray.org.x) * ray.rdir.x;
        const vfloat<K> lclipMaxY = (upper_y - ray.org.y) * ray.rdir.y;
        const vfloat<K> lclipMaxZ = (upper_z - ray.org.z) * ray.rdir.z;
  #endif
        const vfloat<K> lnearP = max(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = min(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
        const vbool<K> lhit    = max(lnearP, ray.tnear) <= min(lfarP, ray.tfar);
        dist = lnearP;
        return lhit;
      }


    template<int N, int K>
      __forceinline vbool<K> intersectQuantizedNodeMBK(const typename BVHN<N>::QuantizedBaseNodeMB* node, const size_t i,
          const TravRayK<K,true>& ray, const vfloat<K>& time, vfloat<K>& dist)

    {
        assert(movemask(node->validMask()) & ((size_t)1 << i));

        const vfloat<K> lower_x = node->template dequantizeLowerX<K>(i,time);
        const vfloat<K> upper_x = node->template dequantizeUpperX<K>(i,time);
        const vfloat<K> lower_y = node->template dequantizeLowerY<K>(i,time);
        const vfloat<K> upper_y = node->template dequantizeUpperY<K>(i,time);
        const vfloat<K> lower_z = node->template dequantizeLowerZ<K>(i,time);
        const vfloat<K> upper_z = node->template dequantizeUpperZ<K>(i,time);

        const vfloat<K> lclipMinX = (lower_x - ray.org.x) * ray.rdir.x;
        const vfloat<K> lclipMinY = (lower_y - ray.org.y) * ray.rdir.y;
        const vfloat<K> lclipMinZ = (lower_z - ray.org.z) * ray.rdir.z;
        const vfloat<K> lclipMaxX = (upper_x - ray.org.x) * ray.rdir.x;
        const vfloat<K> lclipMaxY = (upper_y - ray.org.y) * ray.rdir.y;
        const vfloat<K> lclipMaxZ = (upper_z - ray.org.z) * ray.rdir.z;

        const float round_up   = 1.0f+3.0f*float(ulp);
        const float round_down = 1.0f-3.0f*float(ulp);

        const vfloat<K> lnearP = round_down*max(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY), min(lclipMinZ, lclipMaxZ));
        const vfloat<K> lfarP  = round_up  *min(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY), max(lclipMinZ, lclipMaxZ));
        const vbool<K> lhit    = max(lnearP, ray.tnear) <= min(lfarP, ray.tfar);
        dist = lnearP;
        return lhit;
      }


    //////////////////////////////////////////////////////////////////////////////////////
    // Node intersectors used in hybrid traversal
    //////////////////////////////////////////////////////////////////////////////////////

    /*! Intersects N nodes with K rays */
    template<int N, int K, int types, bool robust>
    struct BVHNNodeIntersectorK;

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN1, false>
    {
      /* vmask is both an input and an output parameter! Its initial value should be the parent node
         hit mask, which is used for correctly computing the current hit mask. The parent hit mask
         is actually required only for motion blur node intersections (because different rays may
         have different times), so for regular nodes vmask is simply overwritten. */
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        vmask = intersectNodeK<N,K>(node.getAABBNode(), i, ray, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        vmask = intersectNodeKRobust<N,K>(node.getAABBNode(), i, ray, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        vmask = intersectNodeK<N,K>(node.getAABBNodeMB(), i, ray, time, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        vmask = intersectNodeKRobust<N,K>(node.getAABBNodeMB(), i, ray, time, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN1_UN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        if (likely(node.isAABBNode()))              vmask = intersectNodeK<N,K>(node.getAABBNode(), i, ray, dist);
        else /*if (unlikely(node.isOBBNode()))*/ vmask = intersectNodeK<N,K>(node.ungetAABBNode(), i, ray, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN1_UN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        if (likely(node.isAABBNode()))              vmask = intersectNodeKRobust<N,K>(node.getAABBNode(), i, ray, dist);
        else /*if (unlikely(node.isOBBNode()))*/ vmask = intersectNodeK<N,K>(node.ungetAABBNode(), i, ray, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2_UN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        if (likely(node.isAABBNodeMB()))              vmask = intersectNodeK<N,K>(node.getAABBNodeMB(), i, ray, time, dist);
        else /*if (unlikely(node.isOBBNodeMB()))*/ vmask = intersectNodeK<N,K>(node.ungetAABBNodeMB(), i, ray, time, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2_UN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        if (likely(node.isAABBNodeMB()))              vmask = intersectNodeKRobust<N,K>(node.getAABBNodeMB(), i, ray, time, dist);
        else /*if (unlikely(node.isOBBNodeMB()))*/ vmask = intersectNodeK<N,K>(node.ungetAABBNodeMB(), i, ray, time, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2_AN4D, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        vmask &= intersectNodeKMB4D<N,K>(node, i, ray, time, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2_AN4D, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        vmask &= intersectNodeKMB4DRobust<N,K>(node, i, ray, time, dist);
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2_AN4D_UN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKFast<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        if (likely(node.isAABBNodeMB() || node.isAABBNodeMB4D())) {
          vmask &= intersectNodeKMB4D<N,K>(node, i, ray, time, dist);
        } else /*if (unlikely(node.isOBBNodeMB()))*/ {
          assert(node.isOBBNodeMB());
          vmask &= intersectNodeK<N,K>(node.ungetAABBNodeMB(), i, ray, time, dist);
        }
        return true;
      }
    };

    template<int N, int K>
    struct BVHNNodeIntersectorK<N, K, BVH_AN2_AN4D_UN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, size_t i,
                                          const TravRayKRobust<K>& ray, const vfloat<K>& time, vfloat<K>& dist, vbool<K>& vmask)
      {
        if (likely(node.isAABBNodeMB() || node.isAABBNodeMB4D())) {
          vmask &= intersectNodeKMB4DRobust<N,K>(node, i, ray, time, dist);
        } else /*if (unlikely(node.isOBBNodeMB()))*/ {
          assert(node.isOBBNodeMB());
          vmask &= intersectNodeK<N,K>(node.ungetAABBNodeMB(), i, ray, time, dist);
        }
        return true;
      }
    };


    /*! Intersects N nodes with K rays */
    template<int N, int K, bool robust>
    struct BVHNQuantizedBaseNodeIntersectorK;

    template<int N, int K>
    struct BVHNQuantizedBaseNodeIntersectorK<N, K, false>
    {
      static __forceinline vbool<K> intersectK(const typename BVHN<N>::QuantizedBaseNode* node, const size_t i,
                                              const TravRayK<K,false>& ray, vfloat<K>& dist)
      {
        return intersectQuantizedNodeK<N,K>(node,i,ray,dist);
      }

      static __forceinline vbool<K> intersectK(const typename BVHN<N>::QuantizedBaseNodeMB* node, const size_t i,
                                               const TravRayK<K,false>& ray, const vfloat<K>& time, vfloat<K>& dist)
      {
        return intersectQuantizedNodeMBK<N,K>(node,i,ray,time,dist);
      }

    };

    template<int N, int K>
    struct BVHNQuantizedBaseNodeIntersectorK<N, K, true>
    {
      static __forceinline vbool<K> intersectK(const typename BVHN<N>::QuantizedBaseNode* node, const size_t i,
                                               const TravRayK<K,true>& ray, vfloat<K>& dist)
      {
        return intersectQuantizedNodeK<N,K>(node,i,ray,dist);
      }

      static __forceinline vbool<K> intersectK(const typename BVHN<N>::QuantizedBaseNodeMB* node, const size_t i,
          const TravRayK<K,true>& ray, const vfloat<K>& time, vfloat<K>& dist)
      {
        return intersectQuantizedNodeMBK<N,K>(node,i,ray,time,dist);
      }
    };


  }
}
