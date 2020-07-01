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

#include "curveNv.h"
#include "curveNi_intersector.h"

namespace embree
{
  namespace isa
  {
    template<int M>
      struct CurveNvIntersector1 : public CurveNiIntersector1<M>
    {
      typedef CurveNv<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_t(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = CurveNiIntersector1<M>::intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = (CurveGeometry*) context->scene->get(geomID);
          const Vec3fa a0 = Vec3fa::loadu(&prim.vertices(i,N)[0]);
          const Vec3fa a1 = Vec3fa::loadu(&prim.vertices(i,N)[1]);
          const Vec3fa a2 = Vec3fa::loadu(&prim.vertices(i,N)[2]);
          const Vec3fa a3 = Vec3fa::loadu(&prim.vertices(i,N)[3]);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            prefetchL1(&prim.vertices(i1,N)[0]);
            prefetchL1(&prim.vertices(i1,N)[4]);
            if (mask1) {
              const size_t i2 = bsf(mask1);
              prefetchL2(&prim.vertices(i2,N)[0]);
              prefetchL2(&prim.vertices(i2,N)[4]);
            }
          }

          Intersector().intersect(pre,ray,geom,primID,a0,a1,a2,a3,Epilog(ray,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
      }

      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_t(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = CurveNiIntersector1<M>::intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = (CurveGeometry*) context->scene->get(geomID);
          const Vec3fa a0 = Vec3fa::loadu(&prim.vertices(i,N)[0]);
          const Vec3fa a1 = Vec3fa::loadu(&prim.vertices(i,N)[1]);
          const Vec3fa a2 = Vec3fa::loadu(&prim.vertices(i,N)[2]);
          const Vec3fa a3 = Vec3fa::loadu(&prim.vertices(i,N)[3]);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            prefetchL1(&prim.vertices(i1,N)[0]);
            prefetchL1(&prim.vertices(i1,N)[4]);
            if (mask1) {
              const size_t i2 = bsf(mask1);
              prefetchL2(&prim.vertices(i2,N)[0]);
              prefetchL2(&prim.vertices(i2,N)[4]);
            }
          }
          
          if (Intersector().intersect(pre,ray,geom,primID,a0,a1,a2,a3,Epilog(ray,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
        return false;
      }
    };

    template<int M, int K>
      struct CurveNvIntersectorK : public CurveNiIntersectorK<M,K>
    {
      typedef CurveNv<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_t(Precalculations& pre, RayHitK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = CurveNiIntersectorK<M,K>::intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = (CurveGeometry*) context->scene->get(geomID);
          const Vec3fa a0 = Vec3fa::loadu(&prim.vertices(i,N)[0]);
          const Vec3fa a1 = Vec3fa::loadu(&prim.vertices(i,N)[1]);
          const Vec3fa a2 = Vec3fa::loadu(&prim.vertices(i,N)[2]);
          const Vec3fa a3 = Vec3fa::loadu(&prim.vertices(i,N)[3]);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            prefetchL1(&prim.vertices(i1,N)[0]);
            prefetchL1(&prim.vertices(i1,N)[4]);
            if (mask1) {
              const size_t i2 = bsf(mask1);
              prefetchL2(&prim.vertices(i2,N)[0]);
              prefetchL2(&prim.vertices(i2,N)[4]);
            }
          }

          Intersector().intersect(pre,ray,k,geom,primID,a0,a1,a2,a3,Epilog(ray,k,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
      }

      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_t(Precalculations& pre, RayK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = CurveNiIntersectorK<M,K>::intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = (CurveGeometry*) context->scene->get(geomID);
          const Vec3fa a0 = Vec3fa::loadu(&prim.vertices(i,N)[0]);
          const Vec3fa a1 = Vec3fa::loadu(&prim.vertices(i,N)[1]);
          const Vec3fa a2 = Vec3fa::loadu(&prim.vertices(i,N)[2]);
          const Vec3fa a3 = Vec3fa::loadu(&prim.vertices(i,N)[3]);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            prefetchL1(&prim.vertices(i1,N)[0]);
            prefetchL1(&prim.vertices(i1,N)[4]);
            if (mask1) {
              const size_t i2 = bsf(mask1);
              prefetchL2(&prim.vertices(i2,N)[0]);
              prefetchL2(&prim.vertices(i2,N)[4]);
            }
          }

          if (Intersector().intersect(pre,ray,k,geom,primID,a0,a1,a2,a3,Epilog(ray,k,context,geomID,primID)))
            return true;

          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
        return false;
      }
    };
  }
}
