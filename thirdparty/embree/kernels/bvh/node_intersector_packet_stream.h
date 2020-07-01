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
    // Fast AlignedNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx, int K>
    __forceinline size_t intersectNode1(const typename BVHN<N>::AlignedNode* __restrict__ node,
                                        const TravRayKStreamFast<K>& ray, size_t k, const NearFarPrecalculations& nf)
    {
      const vfloat<Nx> bminX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearX));
      const vfloat<Nx> bminY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearY));
      const vfloat<Nx> bminZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearZ));
      const vfloat<Nx> bmaxX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farX));
      const vfloat<Nx> bmaxY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farY));
      const vfloat<Nx> bmaxZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farZ));

      const vfloat<Nx> rminX = msub(bminX, vfloat<Nx>(ray.rdir.x[k]), vfloat<Nx>(ray.org_rdir.x[k]));
      const vfloat<Nx> rminY = msub(bminY, vfloat<Nx>(ray.rdir.y[k]), vfloat<Nx>(ray.org_rdir.y[k]));
      const vfloat<Nx> rminZ = msub(bminZ, vfloat<Nx>(ray.rdir.z[k]), vfloat<Nx>(ray.org_rdir.z[k]));
      const vfloat<Nx> rmaxX = msub(bmaxX, vfloat<Nx>(ray.rdir.x[k]), vfloat<Nx>(ray.org_rdir.x[k]));
      const vfloat<Nx> rmaxY = msub(bmaxY, vfloat<Nx>(ray.rdir.y[k]), vfloat<Nx>(ray.org_rdir.y[k]));
      const vfloat<Nx> rmaxZ = msub(bmaxZ, vfloat<Nx>(ray.rdir.z[k]), vfloat<Nx>(ray.org_rdir.z[k]));
      const vfloat<Nx> rmin  = maxi(rminX, rminY, rminZ, vfloat<Nx>(ray.tnear[k]));
      const vfloat<Nx> rmax  = mini(rmaxX, rmaxY, rmaxZ, vfloat<Nx>(ray.tfar[k]));

      const vbool<Nx> vmask_first_hit = rmin <= rmax;

      return movemask(vmask_first_hit) & (((size_t)1 << N)-1);
    }

    template<int N, int K>
    __forceinline size_t intersectNodeK(const typename BVHN<N>::AlignedNode* __restrict__ node, size_t i,
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
    // Robust AlignedNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx, int K>
    __forceinline size_t intersectNode1(const typename BVHN<N>::AlignedNode* __restrict__ node,
                                        const TravRayKStreamRobust<K>& ray, size_t k, const NearFarPrecalculations& nf)
    {
      const vfloat<Nx> bminX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearX));
      const vfloat<Nx> bminY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearY));
      const vfloat<Nx> bminZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.nearZ));
      const vfloat<Nx> bmaxX = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farX));
      const vfloat<Nx> bmaxY = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farY));
      const vfloat<Nx> bmaxZ = vfloat<Nx>(*(const vfloat<N>*)((const char*)&node->lower_x + nf.farZ));

      const vfloat<Nx> rminX = (bminX - vfloat<Nx>(ray.org.x[k])) * vfloat<Nx>(ray.rdir.x[k]);
      const vfloat<Nx> rminY = (bminY - vfloat<Nx>(ray.org.y[k])) * vfloat<Nx>(ray.rdir.y[k]);
      const vfloat<Nx> rminZ = (bminZ - vfloat<Nx>(ray.org.z[k])) * vfloat<Nx>(ray.rdir.z[k]);
      const vfloat<Nx> rmaxX = (bmaxX - vfloat<Nx>(ray.org.x[k])) * vfloat<Nx>(ray.rdir.x[k]);
      const vfloat<Nx> rmaxY = (bmaxY - vfloat<Nx>(ray.org.y[k])) * vfloat<Nx>(ray.rdir.y[k]);
      const vfloat<Nx> rmaxZ = (bmaxZ - vfloat<Nx>(ray.org.z[k])) * vfloat<Nx>(ray.rdir.z[k]);
      const float round_up = 1.0f+3.0f*float(ulp); // FIXME: use per instruction rounding for AVX512
      const vfloat<Nx> rmin  =            max(rminX, rminY, rminZ, vfloat<Nx>(ray.tnear[k]));
      const vfloat<Nx> rmax  = round_up  *min(rmaxX, rmaxY, rmaxZ, vfloat<Nx>(ray.tfar[k]));

      const vbool<Nx> vmask_first_hit = rmin <= rmax;

      return movemask(vmask_first_hit) & (((size_t)1 << N)-1);
    }

    template<int N, int K>
    __forceinline size_t intersectNodeK(const typename BVHN<N>::AlignedNode* __restrict__ node, size_t i,
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
