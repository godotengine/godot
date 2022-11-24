// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "curve_intersector_precalculations.h"
#include "curve_intersector_sweep.h"
#include "../subdiv/linear_bezier_patch.h"

#define DBG(x)

namespace embree
{
  namespace isa
  {
    template<typename Ray, typename Epilog>
      struct TensorLinearCubicBezierSurfaceIntersector
      {
        const LinearSpace3fa& ray_space;
        Ray& ray;
        TensorLinearCubicBezierSurface3fa curve3d;
        TensorLinearCubicBezierSurface2fa curve2d;
        float eps;
        const Epilog& epilog;
        bool isHit;

        __forceinline TensorLinearCubicBezierSurfaceIntersector (const LinearSpace3fa& ray_space, Ray& ray, const TensorLinearCubicBezierSurface3fa& curve3d, const Epilog& epilog)
          : ray_space(ray_space), ray(ray), curve3d(curve3d), epilog(epilog), isHit(false)
        {
          const TensorLinearCubicBezierSurface3fa curve3dray = curve3d.xfm(ray_space,ray.org);
          curve2d = TensorLinearCubicBezierSurface2fa(CubicBezierCurve2fa(curve3dray.L),CubicBezierCurve2fa(curve3dray.R));
          const BBox2fa b2 = curve2d.bounds();
          eps = 8.0f*float(ulp)*reduce_max(max(abs(b2.lower),abs(b2.upper)));
        }
        
        __forceinline Interval1f solve_linear(const float u0, const float u1, const float& p0, const float& p1)
        {
          if (p1 == p0) {
            if (p0 == 0.0f) return Interval1f(u0,u1);
            else return Interval1f(empty);
          }
          const float t = -p0/(p1-p0);
          const float tt = lerp(u0,u1,t);
          return Interval1f(tt);
        }

        __forceinline void solve_linear(const float u0, const float u1, const Interval1f& p0, const Interval1f& p1, Interval1f& u)
        {
          if (sign(p0.lower) != sign(p0.upper)) u.extend(u0);
          if (sign(p0.lower) != sign(p1.lower)) u.extend(solve_linear(u0,u1,p0.lower,p1.lower));
          if (sign(p0.upper) != sign(p1.upper)) u.extend(solve_linear(u0,u1,p0.upper,p1.upper));
          if (sign(p1.lower) != sign(p1.upper)) u.extend(u1);
        }

        __forceinline Interval1f bezier_clipping(const CubicBezierCurve<Interval1f>& curve)
        {
          Interval1f u = empty;
          solve_linear(0.0f/3.0f,1.0f/3.0f,curve.v0,curve.v1,u);
          solve_linear(0.0f/3.0f,2.0f/3.0f,curve.v0,curve.v2,u);
          solve_linear(0.0f/3.0f,3.0f/3.0f,curve.v0,curve.v3,u);
          solve_linear(1.0f/3.0f,2.0f/3.0f,curve.v1,curve.v2,u);
          solve_linear(1.0f/3.0f,3.0f/3.0f,curve.v1,curve.v3,u);
          solve_linear(2.0f/3.0f,3.0f/3.0f,curve.v2,curve.v3,u);
          return intersect(u,Interval1f(0.0f,1.0f));
        }
        
        __forceinline Interval1f bezier_clipping(const LinearBezierCurve<Interval1f>& curve)
        {
          Interval1f v = empty;
          solve_linear(0.0f,1.0f,curve.v0,curve.v1,v);
          return intersect(v,Interval1f(0.0f,1.0f));
        }

        __forceinline void solve_bezier_clipping(BBox1f cu, BBox1f cv, const TensorLinearCubicBezierSurface2fa& curve2)
        {
          BBox2fa bounds = curve2.bounds();
          if (bounds.upper.x < 0.0f) return;
          if (bounds.upper.y < 0.0f) return;
          if (bounds.lower.x > 0.0f) return;
          if (bounds.lower.y > 0.0f) return;
          
          if (max(cu.size(),cv.size()) < 1E-4f)
          {
            const float u = cu.center();
            const float v = cv.center();
            TensorLinearCubicBezierSurface1f curve_z = curve3d.xfm(ray_space.row2(),ray.org);
            const float t = curve_z.eval(u,v);
            if (ray.tnear() <= t && t <= ray.tfar) {
              const Vec3fa Ng = cross(curve3d.eval_du(u,v),curve3d.eval_dv(u,v));
              BezierCurveHit hit(t,u,v,Ng);
              isHit |= epilog(hit);
            }
            return;
          }
          
          const Vec2fa dv = curve2.axis_v();
          const TensorLinearCubicBezierSurface1f curve1v = curve2.xfm(dv);
          LinearBezierCurve<Interval1f> curve0v = curve1v.reduce_u();
          if (!curve0v.hasRoot()) return;
          
          const Interval1f v = bezier_clipping(curve0v);
          if (isEmpty(v)) return;
          TensorLinearCubicBezierSurface2fa curve2a = curve2.clip_v(v);
          cv = BBox1f(lerp(cv.lower,cv.upper,v.lower),lerp(cv.lower,cv.upper,v.upper));

          const Vec2fa du = curve2.axis_u();
          const TensorLinearCubicBezierSurface1f curve1u = curve2a.xfm(du);
          CubicBezierCurve<Interval1f> curve0u = curve1u.reduce_v();         
          int roots = curve0u.maxRoots();
          if (roots == 0) return;
          
          if (roots == 1)
          {
            const Interval1f u = bezier_clipping(curve0u);
            if (isEmpty(u)) return;
            TensorLinearCubicBezierSurface2fa curve2b = curve2a.clip_u(u);
            cu = BBox1f(lerp(cu.lower,cu.upper,u.lower),lerp(cu.lower,cu.upper,u.upper));
            solve_bezier_clipping(cu,cv,curve2b);
            return;
          }

          TensorLinearCubicBezierSurface2fa curve2l, curve2r;
          curve2a.split_u(curve2l,curve2r);
          solve_bezier_clipping(BBox1f(cu.lower,cu.center()),cv,curve2l);
          solve_bezier_clipping(BBox1f(cu.center(),cu.upper),cv,curve2r);
        }
        
        __forceinline bool solve_bezier_clipping()
        {
          solve_bezier_clipping(BBox1f(0.0f,1.0f),BBox1f(0.0f,1.0f),curve2d);
          return isHit;
        }

        __forceinline void solve_newton_raphson(BBox1f cu, BBox1f cv)
        {
          Vec2fa uv(cu.center(),cv.center());
          const Vec2fa dfdu = curve2d.eval_du(uv.x,uv.y);
          const Vec2fa dfdv = curve2d.eval_dv(uv.x,uv.y);
          const LinearSpace2fa rcp_J = rcp(LinearSpace2fa(dfdu,dfdv));
          solve_newton_raphson_loop(cu,cv,uv,dfdu,dfdv,rcp_J);
        }

        __forceinline void solve_newton_raphson_loop(BBox1f cu, BBox1f cv, const Vec2fa& uv_in, const Vec2fa& dfdu, const Vec2fa& dfdv, const LinearSpace2fa& rcp_J)
        {
          Vec2fa uv = uv_in;
          
          for (size_t i=0; i<200; i++)
          {
            const Vec2fa f = curve2d.eval(uv.x,uv.y);
            const Vec2fa duv = rcp_J*f;
            uv -= duv;

            if (max(abs(f.x),abs(f.y)) < eps)
            {
              const float u = uv.x;
              const float v = uv.y;
              if (!(u >= 0.0f && u <= 1.0f)) return; // rejects NaNs
              if (!(v >= 0.0f && v <= 1.0f)) return; // rejects NaNs
              const TensorLinearCubicBezierSurface1f curve_z = curve3d.xfm(ray_space.row2(),ray.org);
              const float t = curve_z.eval(u,v);
              if (!(ray.tnear() <= t && t <= ray.tfar)) return; // rejects NaNs
              const Vec3fa Ng = cross(curve3d.eval_du(u,v),curve3d.eval_dv(u,v));
              BezierCurveHit hit(t,u,v,Ng);
              isHit |= epilog(hit);
              return;
            }
          }       
        }

        __forceinline bool clip_v(BBox1f& cu, BBox1f& cv)
        {
          const Vec2fa dv = curve2d.eval_dv(cu.lower,cv.lower);
          const TensorLinearCubicBezierSurface1f curve1v = curve2d.xfm(dv).clip(cu,cv);
          LinearBezierCurve<Interval1f> curve0v = curve1v.reduce_u();
          if (!curve0v.hasRoot()) return false;
          Interval1f v = bezier_clipping(curve0v);
          if (isEmpty(v)) return false;
          v = intersect(v + Interval1f(-0.1f,+0.1f),Interval1f(0.0f,1.0f));
          cv = BBox1f(lerp(cv.lower,cv.upper,v.lower),lerp(cv.lower,cv.upper,v.upper));
          return true;
        }

        __forceinline bool solve_krawczyk(bool very_small, BBox1f& cu, BBox1f& cv)
        {
          /* perform bezier clipping in v-direction to get tight v-bounds */
          TensorLinearCubicBezierSurface2fa curve2 = curve2d.clip(cu,cv);
          const Vec2fa dv = curve2.axis_v();
          const TensorLinearCubicBezierSurface1f curve1v = curve2.xfm(dv);
          LinearBezierCurve<Interval1f> curve0v = curve1v.reduce_u();
          if (unlikely(!curve0v.hasRoot())) return true;
          Interval1f v = bezier_clipping(curve0v);
          if (unlikely(isEmpty(v))) return true;
          v = intersect(v + Interval1f(-0.1f,+0.1f),Interval1f(0.0f,1.0f));
          curve2 = curve2.clip_v(v);
          cv = BBox1f(lerp(cv.lower,cv.upper,v.lower),lerp(cv.lower,cv.upper,v.upper));

          /* perform one newton raphson iteration */
          Vec2fa c(cu.center(),cv.center());
          Vec2fa f,dfdu,dfdv; curve2d.eval(c.x,c.y,f,dfdu,dfdv);
          const LinearSpace2fa rcp_J = rcp(LinearSpace2fa(dfdu,dfdv));
          const Vec2fa c1 = c - rcp_J*f;
          
          /* calculate bounds of derivatives */
          const BBox2fa bounds_du = (1.0f/cu.size())*curve2.derivative_u().bounds();
          const BBox2fa bounds_dv = (1.0f/cv.size())*curve2.derivative_v().bounds();

          /* calculate krawczyk test */
          LinearSpace2<Vec2<Interval1f>> I(Interval1f(1.0f), Interval1f(0.0f),
                                           Interval1f(0.0f), Interval1f(1.0f));

          LinearSpace2<Vec2<Interval1f>> G(Interval1f(bounds_du.lower.x,bounds_du.upper.x), Interval1f(bounds_dv.lower.x,bounds_dv.upper.x),
                                           Interval1f(bounds_du.lower.y,bounds_du.upper.y), Interval1f(bounds_dv.lower.y,bounds_dv.upper.y));

          const LinearSpace2<Vec2f> rcp_J2(rcp_J);
          const LinearSpace2<Vec2<Interval1f>> rcp_Ji(rcp_J2);
          
          const Vec2<Interval1f> x(cu,cv);
          const Vec2<Interval1f> K = Vec2<Interval1f>(Vec2f(c1)) + (I - rcp_Ji*G)*(x-Vec2<Interval1f>(Vec2f(c)));

          /* test if there is no solution */
          const Vec2<Interval1f> KK = intersect(K,x);
          if (unlikely(isEmpty(KK.x) || isEmpty(KK.y))) return true;

          /* exit if convergence cannot get proven, but terminate if we are very small */
          if (unlikely(!subset(K,x) && !very_small)) return false;

          /* solve using newton raphson iteration of convergence is guaranteed */
          solve_newton_raphson_loop(cu,cv,c1,dfdu,dfdv,rcp_J);
          return true;
        }

        __forceinline void solve_newton_raphson_no_recursion(BBox1f cu, BBox1f cv)
        {
           if (!clip_v(cu,cv)) return;
           return solve_newton_raphson(cu,cv);
        }
        
        __forceinline void solve_newton_raphson_recursion(BBox1f cu, BBox1f cv)
        {
          unsigned int sptr = 0;
          const unsigned int stack_size = 4;
          unsigned int mask_stack[stack_size];
          BBox1f cu_stack[stack_size];
          BBox1f cv_stack[stack_size];
          goto entry;
          
          /* terminate if stack is empty */
          while (sptr)
          {
            /* pop from stack */
            {
              sptr--;
              size_t mask = mask_stack[sptr];
              cu = cu_stack[sptr];
              cv = cv_stack[sptr];
              const size_t i = bscf(mask);
              mask_stack[sptr] = mask;
              if (mask) sptr++; // there are still items on the stack
              
              /* process next element recurse into each hit curve segment */
              const float u0 = float(i+0)*(1.0f/(VSIZEX-1));
              const float u1 = float(i+1)*(1.0f/(VSIZEX-1));
              const BBox1f cui(lerp(cu.lower,cu.upper,u0),lerp(cu.lower,cu.upper,u1));
              cu = cui;
            }

#if 0
            solve_newton_raphson_no_recursion(cu,cv);
            continue;
            
#else
            /* we assume convergence for small u ranges and verify using krawczyk */
            if (cu.size() < 1.0f/6.0f) {
              const bool very_small = cu.size() < 0.001f || sptr >= stack_size;
              if (solve_krawczyk(very_small,cu,cv)) {
                continue;
              }
            }
#endif

          entry:
          
            /* split the curve into VSIZEX-1 segments in u-direction */
            vboolx valid = true;
            TensorLinearCubicBezierSurface<Vec2vfx> subcurves = curve2d.clip_v(cv).vsplit_u(valid,cu);
            
            /* slabs test in u-direction */
            Vec2vfx ndv = cross(subcurves.axis_v());
            BBox<vfloatx> boundsv = subcurves.vxfm(ndv).bounds();
            valid &= boundsv.lower <= eps;
            valid &= boundsv.upper >= -eps;
            if (none(valid)) continue;

            /* slabs test in v-direction */
            Vec2vfx ndu = cross(subcurves.axis_u());
            BBox<vfloatx> boundsu = subcurves.vxfm(ndu).bounds();
            valid &= boundsu.lower <= eps;
            valid &= boundsu.upper >= -eps;
            if (none(valid)) continue;

            /* push valid segments to stack */
            assert(sptr < stack_size);
            mask_stack [sptr] = movemask(valid);
            cu_stack   [sptr] = cu;
            cv_stack   [sptr] = cv;
            sptr++;
          }
        }
        
        __forceinline bool solve_newton_raphson_main()
        {
          BBox1f vu(0.0f,1.0f);
          BBox1f vv(0.0f,1.0f);
          solve_newton_raphson_recursion(vu,vv);
          return isHit;
        }
      };


    template<template<typename Ty> class SourceCurve>
      struct OrientedCurve1Intersector1
    {
      //template<typename Ty> using Curve = SourceCurve<Ty>;
      typedef SourceCurve<Vec3ff> SourceCurve3ff;
      typedef SourceCurve<Vec3fa> SourceCurve3fa;
      
      __forceinline OrientedCurve1Intersector1() {}
      
      __forceinline OrientedCurve1Intersector1(const Ray& ray, const void* ptr) {}
      
      template<typename Epilog>
      __noinline bool intersect(const CurvePrecalculations1& pre, Ray& ray,
                                IntersectContext* context,
                                const CurveGeometry* geom, const unsigned int primID, 
                                const Vec3ff& v0i, const Vec3ff& v1i, const Vec3ff& v2i, const Vec3ff& v3i,
                                const Vec3fa& n0i, const Vec3fa& n1i, const Vec3fa& n2i, const Vec3fa& n3i,
                                const Epilog& epilog) const
      {
        STAT3(normal.trav_prims,1,1,1);

        SourceCurve3ff ccurve(v0i,v1i,v2i,v3i);
        SourceCurve3fa ncurve(n0i,n1i,n2i,n3i);
        ccurve = enlargeRadiusToMinWidth(context,geom,ray.org,ccurve);
        TensorLinearCubicBezierSurface3fa curve = TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(ccurve,ncurve);
        //return TensorLinearCubicBezierSurfaceIntersector<Ray,Epilog>(pre.ray_space,ray,curve,epilog).solve_bezier_clipping();
        return TensorLinearCubicBezierSurfaceIntersector<Ray,Epilog>(pre.ray_space,ray,curve,epilog).solve_newton_raphson_main();
      }

      template<typename Epilog>
      __noinline bool intersect(const CurvePrecalculations1& pre, Ray& ray,
                                IntersectContext* context,
                                const CurveGeometry* geom, const unsigned int primID,
                                const TensorLinearCubicBezierSurface3fa& curve, const Epilog& epilog) const
      {
        STAT3(normal.trav_prims,1,1,1);
        //return TensorLinearCubicBezierSurfaceIntersector<Ray,Epilog>(pre.ray_space,ray,curve,epilog).solve_bezier_clipping();
        return TensorLinearCubicBezierSurfaceIntersector<Ray,Epilog>(pre.ray_space,ray,curve,epilog).solve_newton_raphson_main();
      }
    };

    template<template<typename Ty> class SourceCurve, int K>
      struct OrientedCurve1IntersectorK
    {
      //template<typename Ty> using Curve = SourceCurve<Ty>;
      typedef SourceCurve<Vec3ff> SourceCurve3ff;
      typedef SourceCurve<Vec3fa> SourceCurve3fa;
      
      struct Ray1
      {
        __forceinline Ray1(RayK<K>& ray, size_t k)
          : org(ray.org.x[k],ray.org.y[k],ray.org.z[k]), dir(ray.dir.x[k],ray.dir.y[k],ray.dir.z[k]), _tnear(ray.tnear()[k]), tfar(ray.tfar[k]) {}

        Vec3fa org;
        Vec3fa dir;
        float _tnear;
        float& tfar;

        __forceinline float& tnear() { return _tnear; }
        //__forceinline float& tfar()  { return _tfar; }
        __forceinline const float& tnear() const { return _tnear; }
        //__forceinline const float& tfar()  const { return _tfar; }
      };

      template<typename Epilog>
      __forceinline bool intersect(const CurvePrecalculationsK<K>& pre, RayK<K>& vray, size_t k,
                                   IntersectContext* context,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const Vec3ff& v0i, const Vec3ff& v1i, const Vec3ff& v2i, const Vec3ff& v3i,
                                   const Vec3fa& n0i, const Vec3fa& n1i, const Vec3fa& n2i, const Vec3fa& n3i,
                                   const Epilog& epilog)
      {
        STAT3(normal.trav_prims,1,1,1);
        Ray1 ray(vray,k);
        SourceCurve3ff ccurve(v0i,v1i,v2i,v3i);
        SourceCurve3fa ncurve(n0i,n1i,n2i,n3i);
        ccurve = enlargeRadiusToMinWidth(context,geom,ray.org,ccurve);
        TensorLinearCubicBezierSurface3fa curve = TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(ccurve,ncurve);
        //return TensorLinearCubicBezierSurfaceIntersector<Ray1,Epilog>(pre.ray_space[k],ray,curve,epilog).solve_bezier_clipping();
        return TensorLinearCubicBezierSurfaceIntersector<Ray1,Epilog>(pre.ray_space[k],ray,curve,epilog).solve_newton_raphson_main();
      }

      template<typename Epilog>
      __forceinline bool intersect(const CurvePrecalculationsK<K>& pre, RayK<K>& vray, size_t k,
                                   IntersectContext* context,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const TensorLinearCubicBezierSurface3fa& curve,
                                   const Epilog& epilog)
      {
        STAT3(normal.trav_prims,1,1,1);
        Ray1 ray(vray,k);
        //return TensorLinearCubicBezierSurfaceIntersector<Ray1,Epilog>(pre.ray_space[k],ray,curve,epilog).solve_bezier_clipping();
        return TensorLinearCubicBezierSurfaceIntersector<Ray1,Epilog>(pre.ray_space[k],ray,curve,epilog).solve_newton_raphson_main();
      }
    };
  }
}
