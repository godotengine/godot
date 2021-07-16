// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curveNi.h"

namespace embree
{
  namespace isa
  {
    template<int M>
    struct CurveNiIntersector1
    {
      typedef CurveNi<M> Primitive;
      typedef Vec3vf<M> Vec3vfM;
      typedef LinearSpace3<Vec3vfM>LinearSpace3vfM;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline vbool<M> intersect(Ray& ray, const Primitive& prim, vfloat<M>& tNear_o)
      {
        const size_t N = prim.N;
        const vfloat4 offset_scale = vfloat4::loadu(prim.offset(N));
        const Vec3fa offset = Vec3fa(offset_scale);
        const Vec3fa scale = Vec3fa(shuffle<3,3,3,3>(offset_scale));
        const Vec3fa org1 = (ray.org-offset)*scale;
        const Vec3fa dir1 = ray.dir*scale;
        
        const LinearSpace3vfM space(vfloat<M>::load(prim.bounds_vx_x(N)), vfloat<M>::load(prim.bounds_vx_y(N)), vfloat<M>::load(prim.bounds_vx_z(N)),
                                    vfloat<M>::load(prim.bounds_vy_x(N)), vfloat<M>::load(prim.bounds_vy_y(N)), vfloat<M>::load(prim.bounds_vy_z(N)),
                                    vfloat<M>::load(prim.bounds_vz_x(N)), vfloat<M>::load(prim.bounds_vz_y(N)), vfloat<M>::load(prim.bounds_vz_z(N)));

        const Vec3vfM dir2 = xfmVector(space,Vec3vfM(dir1));
        const Vec3vfM org2 = xfmPoint (space,Vec3vfM(org1));
        const Vec3vfM rcp_dir2 = rcp_safe(dir2);
       
        const vfloat<M> t_lower_x = (vfloat<M>::load(prim.bounds_vx_lower(N))-vfloat<M>(org2.x))*vfloat<M>(rcp_dir2.x);
        const vfloat<M> t_upper_x = (vfloat<M>::load(prim.bounds_vx_upper(N))-vfloat<M>(org2.x))*vfloat<M>(rcp_dir2.x);
        const vfloat<M> t_lower_y = (vfloat<M>::load(prim.bounds_vy_lower(N))-vfloat<M>(org2.y))*vfloat<M>(rcp_dir2.y);
        const vfloat<M> t_upper_y = (vfloat<M>::load(prim.bounds_vy_upper(N))-vfloat<M>(org2.y))*vfloat<M>(rcp_dir2.y);
        const vfloat<M> t_lower_z = (vfloat<M>::load(prim.bounds_vz_lower(N))-vfloat<M>(org2.z))*vfloat<M>(rcp_dir2.z);
        const vfloat<M> t_upper_z = (vfloat<M>::load(prim.bounds_vz_upper(N))-vfloat<M>(org2.z))*vfloat<M>(rcp_dir2.z);

        const vfloat<M> round_up  (1.0f+3.0f*float(ulp));
        const vfloat<M> round_down(1.0f-3.0f*float(ulp));
        const vfloat<M> tNear = round_down*max(mini(t_lower_x,t_upper_x),mini(t_lower_y,t_upper_y),mini(t_lower_z,t_upper_z),vfloat<M>(ray.tnear()));
        const vfloat<M> tFar  = round_up  *min(maxi(t_lower_x,t_upper_x),maxi(t_lower_y,t_upper_y),maxi(t_lower_z,t_upper_z),vfloat<M>(ray.tfar));
        tNear_o = tNear;
        return (vint<M>(step) < vint<M>(prim.N)) & (tNear <= tFar);
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_t(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff a0,a1,a2,a3; geom->gather(a0,a1,a2,a3,geom->curve(primID));

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }
          
          Intersector().intersect(pre,ray,context,geom,primID,a0,a1,a2,a3,Epilog(ray,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
      }

      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_t(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff a0,a1,a2,a3; geom->gather(a0,a1,a2,a3,geom->curve(primID));
         
          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }

          if (Intersector().intersect(pre,ray,context,geom,primID,a0,a1,a2,a3,Epilog(ray,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
        return false;
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_n(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          
          unsigned int vertexID = geom->curve(primID);
          Vec3ff a0,a1,a2,a3; Vec3fa n0,n1,n2,n3; geom->gather(a0,a1,a2,a3,n0,n1,n2,n3,vertexID);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }
          
          Intersector().intersect(pre,ray,context,geom,primID,a0,a1,a2,a3,n0,n1,n2,n3,Epilog(ray,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
      }

      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_n(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);

          unsigned int vertexID = geom->curve(primID);
          Vec3ff a0,a1,a2,a3; Vec3fa n0,n1,n2,n3; geom->gather(a0,a1,a2,a3,n0,n1,n2,n3,vertexID);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }

          if (Intersector().intersect(pre,ray,context,geom,primID,a0,a1,a2,a3,n0,n1,n2,n3,Epilog(ray,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
        return false;
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_h(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; geom->gather_hermite(p0,t0,p1,t1,geom->curve(primID));
          Intersector().intersect(pre,ray,context,geom,primID,p0,t0,p1,t1,Epilog(ray,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
      }

      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_h(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; geom->gather_hermite(p0,t0,p1,t1,geom->curve(primID));
          if (Intersector().intersect(pre,ray,context,geom,primID,p0,t0,p1,t1,Epilog(ray,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
        return false;
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_hn(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; Vec3fa n0,dn0,n1,dn1; geom->gather_hermite(p0,t0,n0,dn0,p1,t1,n1,dn1,geom->curve(primID));
          Intersector().intersect(pre,ray,context,geom,primID,p0,t0,p1,t1,n0,dn0,n1,dn1,Epilog(ray,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
      }

      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_hn(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; Vec3fa n0,dn0,n1,dn1; geom->gather_hermite(p0,t0,n0,dn0,p1,t1,n1,dn1,geom->curve(primID));
          if (Intersector().intersect(pre,ray,context,geom,primID,p0,t0,p1,t1,n0,dn0,n1,dn1,Epilog(ray,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar));
        }
        return false;
      }
    };

    template<int M, int K>
      struct CurveNiIntersectorK
    {
      typedef CurveNi<M> Primitive;
      typedef Vec3vf<M> Vec3vfM;
      typedef LinearSpace3<Vec3vfM>LinearSpace3vfM;
      typedef CurvePrecalculationsK<K> Precalculations;
      
      static __forceinline vbool<M> intersect(RayK<K>& ray, const size_t k, const Primitive& prim, vfloat<M>& tNear_o)
      {
        const size_t N = prim.N;
        const vfloat4 offset_scale = vfloat4::loadu(prim.offset(N));
        const Vec3fa offset = Vec3fa(offset_scale);
        const Vec3fa scale = Vec3fa(shuffle<3,3,3,3>(offset_scale));

        const Vec3fa ray_org(ray.org.x[k],ray.org.y[k],ray.org.z[k]);
        const Vec3fa ray_dir(ray.dir.x[k],ray.dir.y[k],ray.dir.z[k]);
        const Vec3fa org1 = (ray_org-offset)*scale;
        const Vec3fa dir1 = ray_dir*scale;
        
        const LinearSpace3vfM space(vfloat<M>::load(prim.bounds_vx_x(N)), vfloat<M>::load(prim.bounds_vx_y(N)), vfloat<M>::load(prim.bounds_vx_z(N)),
                                    vfloat<M>::load(prim.bounds_vy_x(N)), vfloat<M>::load(prim.bounds_vy_y(N)), vfloat<M>::load(prim.bounds_vy_z(N)),
                                    vfloat<M>::load(prim.bounds_vz_x(N)), vfloat<M>::load(prim.bounds_vz_y(N)), vfloat<M>::load(prim.bounds_vz_z(N)));

        const Vec3vfM dir2 = xfmVector(space,Vec3vfM(dir1));
        const Vec3vfM org2 = xfmPoint (space,Vec3vfM(org1));
        const Vec3vfM rcp_dir2 = rcp_safe(dir2);
       
        const vfloat<M> t_lower_x = (vfloat<M>::load(prim.bounds_vx_lower(N))-vfloat<M>(org2.x))*vfloat<M>(rcp_dir2.x);
        const vfloat<M> t_upper_x = (vfloat<M>::load(prim.bounds_vx_upper(N))-vfloat<M>(org2.x))*vfloat<M>(rcp_dir2.x);
        const vfloat<M> t_lower_y = (vfloat<M>::load(prim.bounds_vy_lower(N))-vfloat<M>(org2.y))*vfloat<M>(rcp_dir2.y);
        const vfloat<M> t_upper_y = (vfloat<M>::load(prim.bounds_vy_upper(N))-vfloat<M>(org2.y))*vfloat<M>(rcp_dir2.y);
        const vfloat<M> t_lower_z = (vfloat<M>::load(prim.bounds_vz_lower(N))-vfloat<M>(org2.z))*vfloat<M>(rcp_dir2.z);
        const vfloat<M> t_upper_z = (vfloat<M>::load(prim.bounds_vz_upper(N))-vfloat<M>(org2.z))*vfloat<M>(rcp_dir2.z);

        const vfloat<M> round_up  (1.0f+3.0f*float(ulp));
        const vfloat<M> round_down(1.0f-3.0f*float(ulp));
        const vfloat<M> tNear = round_down*max(mini(t_lower_x,t_upper_x),mini(t_lower_y,t_upper_y),mini(t_lower_z,t_upper_z),vfloat<M>(ray.tnear()[k]));
        const vfloat<M> tFar  = round_up  *min(maxi(t_lower_x,t_upper_x),maxi(t_lower_y,t_upper_y),maxi(t_lower_z,t_upper_z),vfloat<M>(ray.tfar[k]));
        tNear_o = tNear;
        return (vint<M>(step) < vint<M>(prim.N)) & (tNear <= tFar);
      }
      
      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_t(Precalculations& pre, RayHitK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff a0,a1,a2,a3; geom->gather(a0,a1,a2,a3,geom->curve(primID));

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }

          Intersector().intersect(pre,ray,k,context,geom,primID,a0,a1,a2,a3,Epilog(ray,k,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
      }
      
      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_t(Precalculations& pre, RayK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff a0,a1,a2,a3; geom->gather(a0,a1,a2,a3,geom->curve(primID));

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }
          
          if (Intersector().intersect(pre,ray,k,context,geom,primID,a0,a1,a2,a3,Epilog(ray,k,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
        return false;
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_n(Precalculations& pre, RayHitK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);

          unsigned int vertexID = geom->curve(primID);
          Vec3ff a0,a1,a2,a3; Vec3fa n0,n1,n2,n3; geom->gather(a0,a1,a2,a3,n0,n1,n2,n3,vertexID);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }

          Intersector().intersect(pre,ray,k,context,geom,primID,a0,a1,a2,a3,n0,n1,n2,n3,Epilog(ray,k,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
      }
      
      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_n(Precalculations& pre, RayK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);

          unsigned int vertexID = geom->curve(primID);
          Vec3ff a0,a1,a2,a3; Vec3fa n0,n1,n2,n3; geom->gather(a0,a1,a2,a3,n0,n1,n2,n3,vertexID);

          size_t mask1 = mask;
          const size_t i1 = bscf(mask1);
          if (mask) {
            const unsigned int primID1 = prim.primID(N)[i1];
            geom->prefetchL1_vertices(geom->curve(primID1));
            if (mask1) {
              const size_t i2 = bsf(mask1);
              const unsigned int primID2 = prim.primID(N)[i2];
              geom->prefetchL2_vertices(geom->curve(primID2));
            }
          }

          if (Intersector().intersect(pre,ray,k,context,geom,primID,a0,a1,a2,a3,n0,n1,n2,n3,Epilog(ray,k,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
        return false;
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_h(Precalculations& pre, RayHitK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; geom->gather_hermite(p0,t0,p1,t1,geom->curve(primID));
          Intersector().intersect(pre,ray,k,context,geom,primID,p0,t0,p1,t1,Epilog(ray,k,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
      }
      
      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_h(Precalculations& pre, RayK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; geom->gather_hermite(p0,t0,p1,t1,geom->curve(primID));
          if (Intersector().intersect(pre,ray,k,context,geom,primID,p0,t0,p1,t1,Epilog(ray,k,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
        return false;
      }

      template<typename Intersector, typename Epilog>
        static __forceinline void intersect_hn(Precalculations& pre, RayHitK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(normal.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; Vec3fa n0,dn0,n1,dn1; geom->gather_hermite(p0,t0,n0,dn0,p1,t1,n1,dn1,geom->curve(primID));
          Intersector().intersect(pre,ray,k,context,geom,primID,p0,t0,p1,t1,n0,dn0,n1,dn1,Epilog(ray,k,context,geomID,primID));
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
      }
      
      template<typename Intersector, typename Epilog>
        static __forceinline bool occluded_hn(Precalculations& pre, RayK<K>& ray, const size_t k, IntersectContext* context, const Primitive& prim)
      {
        vfloat<M> tNear;
        vbool<M> valid = intersect(ray,k,prim,tNear);

        const size_t N = prim.N;
        size_t mask = movemask(valid);
        while (mask)
        {
          const size_t i = bscf(mask);
          STAT3(shadow.trav_prims,1,1,1);
          const unsigned int geomID = prim.geomID(N);
          const unsigned int primID = prim.primID(N)[i];
          const CurveGeometry* geom = context->scene->get<CurveGeometry>(geomID);
          Vec3ff p0,t0,p1,t1; Vec3fa n0,dn0,n1,dn1; geom->gather_hermite(p0,t0,n0,dn0,p1,t1,n1,dn1,geom->curve(primID));
          if (Intersector().intersect(pre,ray,k,context,geom,primID,p0,t0,p1,t1,n0,dn0,n1,dn1,Epilog(ray,k,context,geomID,primID)))
            return true;
          
          mask &= movemask(tNear <= vfloat<M>(ray.tfar[k]));
        }
        return false;
      }
    };
  }
}
