// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "curve_intersector_precalculations.h"

namespace embree
{
  namespace isa
  {
    namespace __coneline_internal 
    {
      template<int M, typename Epilog, typename ray_tfar_func>
        static __forceinline bool intersectCone(const vbool<M>& valid_i,
                                                const Vec3vf<M>& ray_org_in, const Vec3vf<M>& ray_dir, 
                                                const vfloat<M>& ray_tnear, const ray_tfar_func& ray_tfar,
                                                const Vec4vf<M>& v0, const Vec4vf<M>& v1,
                                                const vbool<M>& cL, const vbool<M>& cR,
                                                const Epilog& epilog)
      {   
        vbool<M> valid = valid_i;

        /* move ray origin closer to make calculations numerically stable */
        const vfloat<M> dOdO = sqr(ray_dir);
        const vfloat<M> rcp_dOdO = rcp(dOdO);
        const Vec3vf<M> center = vfloat<M>(0.5f)*(v0.xyz()+v1.xyz());
        const vfloat<M> dt = dot(center-ray_org_in,ray_dir)*rcp_dOdO;
        const Vec3vf<M> ray_org = ray_org_in + dt*ray_dir;

        const Vec3vf<M> dP = v1.xyz() - v0.xyz();
        const Vec3vf<M> p0 = ray_org - v0.xyz();
        const Vec3vf<M> p1 = ray_org - v1.xyz();
        
        const vfloat<M> dPdP  = sqr(dP);
        const vfloat<M> dP0   = dot(p0,dP);
        const vfloat<M> dP1   = dot(p1,dP); 
        const vfloat<M> dOdP  = dot(ray_dir,dP);

        // intersect cone body
        const vfloat<M> dr  = v0.w - v1.w;
        const vfloat<M> hy  = dPdP + sqr(dr);
        const vfloat<M> dO0 = dot(ray_dir,p0);
        const vfloat<M> OO  = sqr(p0);
        const vfloat<M> dPdP2 = sqr(dPdP);
        const vfloat<M> dPdPr0 = dPdP*v0.w;
        
        const vfloat<M> A = dPdP2     - sqr(dOdP)*hy;
        const vfloat<M> B = dPdP2*dO0 - dP0*dOdP*hy   + dPdPr0*(dr*dOdP);
        const vfloat<M> C = dPdP2*OO  - sqr(dP0)*hy   + dPdPr0*(2.0f*dr*dP0 - dPdPr0);
        
        const vfloat<M> D = B*B - A*C;
        valid &= D >= 0.0f;
        if (unlikely(none(valid))) {
          return false;
        }

        /* standard case for "non-parallel" rays */
        const vfloat<M> Q = sqrt(D);
        const vfloat<M> rcp_A = rcp(A);
        /* special case for rays that are "parallel" to the cone - assume miss */
        const vbool<M> isParallel = abs(A) <= min_rcp_input;

        vfloat<M> t_cone_lower = select (isParallel, neg_inf, (-B-Q)*rcp_A);
        vfloat<M> t_cone_upper = select (isParallel, pos_inf, (-B+Q)*rcp_A);
        const vfloat<M> y_lower = dP0 + t_cone_lower*dOdP;
        const vfloat<M> y_upper = dP0 + t_cone_upper*dOdP;
        t_cone_lower = select(valid & y_lower > 0.0f & y_lower < dPdP, t_cone_lower, pos_inf);
        t_cone_upper = select(valid & y_upper > 0.0f & y_upper < dPdP, t_cone_upper, neg_inf);

        const vbool<M> hitDisk0 = valid & cL;
        const vbool<M> hitDisk1 = valid & cR;
        const vfloat<M> rcp_dOdP = rcp(dOdP);
        const vfloat<M> t_disk0 = select (hitDisk0, select (sqr(p0*dOdP-ray_dir*dP0)<(sqr(v0.w)*sqr(dOdP)), -dP0*rcp_dOdP, pos_inf), pos_inf);
        const vfloat<M> t_disk1 = select (hitDisk1, select (sqr(p1*dOdP-ray_dir*dP1)<(sqr(v1.w)*sqr(dOdP)), -dP1*rcp_dOdP, pos_inf), pos_inf);
        const vfloat<M> t_disk_lower = min(t_disk0, t_disk1);
        const vfloat<M> t_disk_upper = max(t_disk0, t_disk1);

        const vfloat<M> t_lower = min(t_cone_lower, t_disk_lower);
        const vfloat<M> t_upper = max(t_cone_upper, select(t_lower==t_disk_lower, 
                                                      select(t_disk_upper==vfloat<M>(pos_inf),neg_inf,t_disk_upper), 
                                                      select(t_disk_lower==vfloat<M>(pos_inf),neg_inf,t_disk_lower)));

        const vbool<M> valid_lower = valid & ray_tnear <= dt+t_lower & dt+t_lower <= ray_tfar() & t_lower != vfloat<M>(pos_inf);
        const vbool<M> valid_upper = valid & ray_tnear <= dt+t_upper & dt+t_upper <= ray_tfar() & t_upper != vfloat<M>(neg_inf);

        const vbool<M> valid_first = valid_lower | valid_upper;
        if (unlikely(none(valid_first)))
          return false;

        const vfloat<M> t_first = select(valid_lower, t_lower, t_upper);
        const vfloat<M> y_first = select(valid_lower, y_lower, y_upper);

        const vfloat<M> rcp_dPdP = rcp(dPdP);
        const Vec3vf<M> dP2drr0dP = dPdP*dr*v0.w*dP;
        const Vec3vf<M> dPhy = dP*hy;
        const vbool<M> cone_hit_first = valid & (t_first == t_cone_lower | t_first == t_cone_upper);
        const vbool<M> disk0_hit_first = valid & (t_first == t_disk0);
        const Vec3vf<M> Ng_first = select(cone_hit_first, dPdP2*(p0+t_first*ray_dir)+dP2drr0dP-dPhy*y_first, select(disk0_hit_first, -dP, dP));
        const vfloat<M> u_first = select(cone_hit_first, y_first*rcp_dPdP, select(disk0_hit_first, vfloat<M>(zero), vfloat<M>(one)));

        /* invoke intersection filter for first hit */
        RoundLineIntersectorHitM<M> hit(u_first,zero,dt+t_first,Ng_first);
        const bool is_hit_first = epilog(valid_first, hit);

        /* check for possible second hits before potentially accepted hit */
        const vfloat<M> t_second = t_upper;
        const vfloat<M> y_second = y_upper;
        const vbool<M> valid_second = valid_lower & valid_upper & (dt+t_upper <= ray_tfar());
        if (unlikely(none(valid_second)))
          return is_hit_first;
        
        /* invoke intersection filter for second hit */
        const vbool<M> cone_hit_second = t_second == t_cone_lower | t_second == t_cone_upper;
        const vbool<M> disk0_hit_second = t_second == t_disk0;
        const Vec3vf<M> Ng_second = select(cone_hit_second, dPdP2*(p0+t_second*ray_dir)+dP2drr0dP-dPhy*y_second, select(disk0_hit_second, -dP, dP));
        const vfloat<M> u_second = select(cone_hit_second, y_second*rcp_dPdP, select(disk0_hit_first, vfloat<M>(zero), vfloat<M>(one)));

        hit = RoundLineIntersectorHitM<M>(u_second,zero,dt+t_second,Ng_second);
        const bool is_hit_second = epilog(valid_second, hit);
        
        return is_hit_first | is_hit_second;
      }
    }

    template<int M>
      struct ConeLineIntersectorHitM
      {
        __forceinline ConeLineIntersectorHitM() {}
        
        __forceinline ConeLineIntersectorHitM(const vfloat<M>& u, const vfloat<M>& v, const vfloat<M>& t, const Vec3vf<M>& Ng)
          : vu(u), vv(v), vt(t), vNg(Ng) {}
	
        __forceinline void finalize() {}
	
        __forceinline Vec2f uv (const size_t i) const { return Vec2f(vu[i],vv[i]); }
        __forceinline float t  (const size_t i) const { return vt[i]; }
        __forceinline Vec3fa Ng(const size_t i) const { return Vec3fa(vNg.x[i],vNg.y[i],vNg.z[i]); }
	
      public:
        vfloat<M> vu;
        vfloat<M> vv;
        vfloat<M> vt;
        Vec3vf<M> vNg;
      };
    
    template<int M>
      struct ConeCurveIntersector1
      {
        typedef CurvePrecalculations1 Precalculations;
        
        struct ray_tfar {
          Ray& ray;
          __forceinline ray_tfar(Ray& ray) : ray(ray) {}
          __forceinline vfloat<M> operator() () const { return ray.tfar; };
        };

        template<typename Epilog>
        static __forceinline bool intersect(const vbool<M>& valid_i,
                                            Ray& ray,
                                            IntersectContext* context,
                                            const LineSegments* geom,
                                            const Precalculations& pre,
                                            const Vec4vf<M>& v0i, const Vec4vf<M>& v1i,
                                            const vbool<M>& cL, const vbool<M>& cR,
                                            const Epilog& epilog)
        {
          const Vec3vf<M> ray_org(ray.org.x, ray.org.y, ray.org.z);
          const Vec3vf<M> ray_dir(ray.dir.x, ray.dir.y, ray.dir.z);
          const vfloat<M> ray_tnear(ray.tnear());
          const Vec4vf<M> v0 = enlargeRadiusToMinWidth<M>(context,geom,ray_org,v0i);
          const Vec4vf<M> v1 = enlargeRadiusToMinWidth<M>(context,geom,ray_org,v1i);
          return  __coneline_internal::intersectCone<M>(valid_i,ray_org,ray_dir,ray_tnear,ray_tfar(ray),v0,v1,cL,cR,epilog);
        }
      };
    
    template<int M, int K>
      struct ConeCurveIntersectorK
      {
        typedef CurvePrecalculationsK<K> Precalculations;
        
        struct ray_tfar {
          RayK<K>& ray;
          size_t k;
          __forceinline ray_tfar(RayK<K>& ray, size_t k) : ray(ray), k(k) {}
          __forceinline vfloat<M> operator() () const { return ray.tfar[k]; };
        };
        
        template<typename Epilog>
        static __forceinline bool intersect(const vbool<M>& valid_i,
                                            RayK<K>& ray, size_t k,
                                            IntersectContext* context,
                                            const LineSegments* geom,
                                            const Precalculations& pre,
                                            const Vec4vf<M>& v0i, const Vec4vf<M>& v1i,
                                            const vbool<M>& cL, const vbool<M>& cR,
                                            const Epilog& epilog)
        {
          const Vec3vf<M> ray_org(ray.org.x[k], ray.org.y[k], ray.org.z[k]);
          const Vec3vf<M> ray_dir(ray.dir.x[k], ray.dir.y[k], ray.dir.z[k]);
          const vfloat<M> ray_tnear = ray.tnear()[k];
          const Vec4vf<M> v0 = enlargeRadiusToMinWidth<M>(context,geom,ray_org,v0i);
          const Vec4vf<M> v1 = enlargeRadiusToMinWidth<M>(context,geom,ray_org,v1i);
          return __coneline_internal::intersectCone<M>(valid_i,ray_org,ray_dir,ray_tnear,ray_tfar(ray,k),v0,v1,cL,cR,epilog);
        }
      };
  }
}
