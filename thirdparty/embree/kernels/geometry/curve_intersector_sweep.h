// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "cylinder.h"
#include "plane.h"
#include "line_intersector.h"
#include "curve_intersector_precalculations.h"

namespace embree
{
  namespace isa
  {
    static const size_t numJacobianIterations = 5;
#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    static const size_t numBezierSubdivisions = 2;
#elif defined(__AVX__)
    static const size_t numBezierSubdivisions = 2;
#else
    static const size_t numBezierSubdivisions = 3;
#endif

    struct BezierCurveHit
    {
      __forceinline BezierCurveHit() {}

      __forceinline BezierCurveHit(const float t, const float u, const Vec3fa& Ng)
        : t(t), u(u), v(0.0f), Ng(Ng) {}

      __forceinline BezierCurveHit(const float t, const float u, const float v, const Vec3fa& Ng)
        : t(t), u(u), v(v), Ng(Ng) {}
      
      __forceinline void finalize() {}
      
    public:
      float t;
      float u;
      float v; 
      Vec3fa Ng;
    };
    
    template<typename NativeCurve3ff, typename Ray, typename Epilog>
    __forceinline bool intersect_bezier_iterative_debug(const Ray& ray, const float dt, const NativeCurve3ff& curve, size_t i,
                                                        const vfloatx& u, const BBox<vfloatx>& tp, const BBox<vfloatx>& h0, const BBox<vfloatx>& h1, 
                                                        const Vec3vfx& Ng, const Vec4vfx& dP0du, const Vec4vfx& dP3du,
                                                        const Epilog& epilog)
    {
      if (tp.lower[i]+dt > ray.tfar) return false;
      Vec3fa Ng_o = Vec3fa(Ng.x[i],Ng.y[i],Ng.z[i]);
      if (h0.lower[i] == tp.lower[i]) Ng_o = -Vec3fa(dP0du.x[i],dP0du.y[i],dP0du.z[i]);
      if (h1.lower[i] == tp.lower[i]) Ng_o = +Vec3fa(dP3du.x[i],dP3du.y[i],dP3du.z[i]);
      BezierCurveHit hit(tp.lower[i]+dt,u[i],Ng_o);
      return epilog(hit);
    }

    template<typename NativeCurve3ff, typename Ray, typename Epilog> 
     __forceinline bool intersect_bezier_iterative_jacobian(const Ray& ray, const float dt, const NativeCurve3ff& curve, float u, float t, const Epilog& epilog)
    {
      const Vec3fa org = zero;
      const Vec3fa dir = ray.dir;
      const float length_ray_dir = length(dir);

      /* error of curve evaluations is proportional to largest coordinate */
      const BBox3ff box = curve.bounds();
      const float P_err = 16.0f*float(ulp)*reduce_max(max(abs(box.lower),abs(box.upper)));
     
      for (size_t i=0; i<numJacobianIterations; i++) 
      {
        const Vec3fa Q = madd(Vec3fa(t),dir,org);
        //const Vec3fa dQdu = zero;
        const Vec3fa dQdt = dir;
        const float Q_err = 16.0f*float(ulp)*length_ray_dir*t; // works as org=zero here
           
        Vec3ff P,dPdu,ddPdu; curve.eval(u,P,dPdu,ddPdu);
        //const Vec3fa dPdt = zero;

        const Vec3fa R = Q-P;
        const float len_R = length(R); //reduce_max(abs(R));
        const float R_err = max(Q_err,P_err);
        const Vec3fa dRdu = /*dQdu*/-dPdu;
        const Vec3fa dRdt = dQdt;//-dPdt;

        const Vec3fa T = normalize(dPdu);
        const Vec3fa dTdu = dnormalize(dPdu,ddPdu);
        //const Vec3fa dTdt = zero;
        const float cos_err = P_err/length(dPdu);

        /* Error estimate for dot(R,T):

           dot(R,T) = cos(R,T) |R| |T|
                    = (cos(R,T) +- cos_error) * (|R| +- |R|_err) * (|T| +- |T|_err)
                    = cos(R,T)*|R|*|T| 
                      +- cos(R,T)*(|R|*|T|_err + |T|*|R|_err)
                      +- cos_error*(|R| + |T|)
                      +- lower order terms
           with cos(R,T) being in [0,1] and |T| = 1 we get:
             dot(R,T)_err = |R|*|T|_err + |R|_err = cos_error*(|R|+1)
        */
              
        const float f = dot(R,T);
        const float f_err = len_R*P_err + R_err + cos_err*(1.0f+len_R);
        const float dfdu = dot(dRdu,T) + dot(R,dTdu);
        const float dfdt = dot(dRdt,T);// + dot(R,dTdt);

        const float K = dot(R,R)-sqr(f);
        const float dKdu = /*2.0f*/(dot(R,dRdu)-f*dfdu);
        const float dKdt = /*2.0f*/(dot(R,dRdt)-f*dfdt);
        const float rsqrt_K = rsqrt(K);

        const float g = sqrt(K)-P.w;
        const float g_err = R_err + f_err + 16.0f*float(ulp)*box.upper.w;
        const float dgdu = /*0.5f*/dKdu*rsqrt_K-dPdu.w;
        const float dgdt = /*0.5f*/dKdt*rsqrt_K;//-dPdt.w;

        const LinearSpace2f J = LinearSpace2f(dfdu,dfdt,dgdu,dgdt);
        const Vec2f dut = rcp(J)*Vec2f(f,g);
        const Vec2f ut = Vec2f(u,t) - dut;
        u = ut.x; t = ut.y;

        if (abs(f) < f_err && abs(g) < g_err)
        {
          t+=dt;
          if (!(ray.tnear() <= t && t <= ray.tfar)) return false; // rejects NaNs
          if (!(u >= 0.0f && u <= 1.0f)) return false; // rejects NaNs
          const Vec3fa R = normalize(Q-P);
          const Vec3fa U = madd(Vec3fa(dPdu.w),R,dPdu);
          const Vec3fa V = cross(dPdu,R);
          BezierCurveHit hit(t,u,cross(V,U));
          return epilog(hit);
        }
      }
      return false;
    }

#if !defined(__SYCL_DEVICE_ONLY__)
    
    template<typename NativeCurve3ff, typename Ray, typename Epilog>
    __forceinline bool intersect_bezier_recursive_jacobian(const Ray& ray, const float dt, const NativeCurve3ff& curve, const Epilog& epilog)
    {
      float u0 = 0.0f;
      float u1 = 1.0f;
      unsigned int depth = 1;
        
#if defined(__AVX__)
      enum { VSIZEX_ = 8 };
      typedef vbool8 vboolx; // maximally 8-wide to work around KNL issues
      typedef vint8 vintx; 
      typedef vfloat8 vfloatx;
#else
      enum { VSIZEX_ = 4 };
      typedef vbool4 vboolx;
      typedef vint4 vintx; 
      typedef vfloat4 vfloatx;
#endif
    
      unsigned int maxDepth = numBezierSubdivisions;
      bool found = false;
      const Vec3fa org = zero;
      const Vec3fa dir = ray.dir;

      unsigned int sptr = 0;
      const unsigned int stack_size = numBezierSubdivisions+1; // +1 because of unstable workaround below
      struct StackEntry {
        vboolx valid;
        vfloatx tlower;
        float u0;
        float u1;
        unsigned int depth;
      };
      StackEntry stack[stack_size];
      goto entry;

       /* terminate if stack is empty */
      while (sptr)
      {
        /* pop from stack */
        {
          sptr--;
          vboolx valid = stack[sptr].valid;
          const vfloatx tlower = stack[sptr].tlower;
          valid &= tlower+dt <= ray.tfar;
          if (none(valid)) continue;
          u0 = stack[sptr].u0;
          u1 = stack[sptr].u1;
          depth = stack[sptr].depth;
          const size_t i = select_min(valid,tlower); clear(valid,i);
          stack[sptr].valid = valid;
          if (any(valid)) sptr++; // there are still items on the stack

          /* process next segment */
          const vfloatx vu0 = lerp(u0,u1,vfloatx(step)*(1.0f/(vfloatx::size-1)));
          u0 = vu0[i+0];
          u1 = vu0[i+1];
        }
      entry:

        /* subdivide curve */
        const float dscale = (u1-u0)*(1.0f/(3.0f*(vfloatx::size-1)));
        const vfloatx vu0 = lerp(u0,u1,vfloatx(step)*(1.0f/(vfloatx::size-1)));
        Vec4vfx P0, dP0du; curve.template veval<VSIZEX_>(vu0,P0,dP0du); dP0du = dP0du * Vec4vfx(dscale);
        const Vec4vfx P3 = shift_right_1(P0);
        const Vec4vfx dP3du = shift_right_1(dP0du); 
        const Vec4vfx P1 = P0 + dP0du; 
        const Vec4vfx P2 = P3 - dP3du;
        
        /* calculate bounding cylinders */
        const vfloatx rr1 = sqr_point_to_line_distance(Vec3vfx(dP0du),Vec3vfx(P3-P0));
        const vfloatx rr2 = sqr_point_to_line_distance(Vec3vfx(dP3du),Vec3vfx(P3-P0));
        const vfloatx maxr12 = sqrt(max(rr1,rr2));
        const vfloatx one_plus_ulp  = 1.0f+2.0f*float(ulp);
        const vfloatx one_minus_ulp = 1.0f-2.0f*float(ulp);
        vfloatx r_outer = max(P0.w,P1.w,P2.w,P3.w)+maxr12;
        vfloatx r_inner = min(P0.w,P1.w,P2.w,P3.w)-maxr12;
        r_outer = one_plus_ulp*r_outer;
        r_inner = max(0.0f,one_minus_ulp*r_inner);
        const CylinderN<vfloatx::size> cylinder_outer(Vec3vfx(P0),Vec3vfx(P3),r_outer);
        const CylinderN<vfloatx::size> cylinder_inner(Vec3vfx(P0),Vec3vfx(P3),r_inner);
        vboolx valid = true; clear(valid,vfloatx::size-1);
        
        /* intersect with outer cylinder */
        BBox<vfloatx> tc_outer; vfloatx u_outer0; Vec3vfx Ng_outer0; vfloatx u_outer1; Vec3vfx Ng_outer1;
        valid &= cylinder_outer.intersect(org,dir,tc_outer,u_outer0,Ng_outer0,u_outer1,Ng_outer1);
        if (none(valid)) continue;
        
        /* intersect with cap-planes */
        BBox<vfloatx> tp(ray.tnear()-dt,ray.tfar-dt);
        tp = embree::intersect(tp,tc_outer);
        BBox<vfloatx> h0 = HalfPlaneN<vfloatx::size>(Vec3vfx(P0),+Vec3vfx(dP0du)).intersect(org,dir);
        tp = embree::intersect(tp,h0);
        BBox<vfloatx> h1 = HalfPlaneN<vfloatx::size>(Vec3vfx(P3),-Vec3vfx(dP3du)).intersect(org,dir);
        tp = embree::intersect(tp,h1);
        valid &= tp.lower <= tp.upper;
        if (none(valid)) continue;
        
        /* clamp and correct u parameter */
        u_outer0 = clamp(u_outer0,vfloatx(0.0f),vfloatx(1.0f));
        u_outer1 = clamp(u_outer1,vfloatx(0.0f),vfloatx(1.0f));
        u_outer0 = lerp(u0,u1,(vfloatx(step)+u_outer0)*(1.0f/float(vfloatx::size)));
        u_outer1 = lerp(u0,u1,(vfloatx(step)+u_outer1)*(1.0f/float(vfloatx::size)));
        
        /* intersect with inner cylinder */
        BBox<vfloatx> tc_inner;
        vfloatx u_inner0 = zero; Vec3vfx Ng_inner0 = zero; vfloatx u_inner1 = zero; Vec3vfx Ng_inner1 = zero;
        const vboolx valid_inner = cylinder_inner.intersect(org,dir,tc_inner,u_inner0,Ng_inner0,u_inner1,Ng_inner1);
        
        /* at the unstable area we subdivide deeper */
        const vboolx unstable0 = (!valid_inner) | (abs(dot(Vec3vfx(Vec3fa(ray.dir)),Ng_inner0)) < 0.3f);
        const vboolx unstable1 = (!valid_inner) | (abs(dot(Vec3vfx(Vec3fa(ray.dir)),Ng_inner1)) < 0.3f);
      
        /* subtract the inner interval from the current hit interval */
        BBox<vfloatx> tp0, tp1;
        subtract(tp,tc_inner,tp0,tp1);
        vboolx valid0 = valid & (tp0.lower <= tp0.upper);
        vboolx valid1 = valid & (tp1.lower <= tp1.upper);
        if (none(valid0 | valid1)) continue;
        
        /* iterate over all first hits front to back */
        const vintx termDepth0 = select(unstable0,vintx(maxDepth+1),vintx(maxDepth));
        vboolx recursion_valid0 = valid0 & (depth < termDepth0);
        valid0 &= depth >= termDepth0;
        
        while (any(valid0))
        {
          const size_t i = select_min(valid0,tp0.lower); clear(valid0,i);
          found = found | intersect_bezier_iterative_jacobian(ray,dt,curve,u_outer0[i],tp0.lower[i],epilog);
          //found = found | intersect_bezier_iterative_debug   (ray,dt,curve,i,u_outer0,tp0,h0,h1,Ng_outer0,dP0du,dP3du,epilog);
          valid0 &= tp0.lower+dt <= ray.tfar;
        }
        valid1 &= tp1.lower+dt <= ray.tfar;
        
        /* iterate over all second hits front to back */
        const vintx termDepth1 = select(unstable1,vintx(maxDepth+1),vintx(maxDepth));
        vboolx recursion_valid1 = valid1 & (depth < termDepth1);
        valid1 &= depth >= termDepth1;
        while (any(valid1))
        {
          const size_t i = select_min(valid1,tp1.lower); clear(valid1,i);
          found = found | intersect_bezier_iterative_jacobian(ray,dt,curve,u_outer1[i],tp1.upper[i],epilog);
          //found = found | intersect_bezier_iterative_debug   (ray,dt,curve,i,u_outer1,tp1,h0,h1,Ng_outer1,dP0du,dP3du,epilog);
          valid1 &= tp1.lower+dt <= ray.tfar;
        }

        /* push valid segments to stack */
        recursion_valid0 &= tp0.lower+dt <= ray.tfar;
        recursion_valid1 &= tp1.lower+dt <= ray.tfar;
        const vboolx recursion_valid = recursion_valid0 | recursion_valid1;
        if (any(recursion_valid))
        {
          assert(sptr < stack_size);
          stack[sptr].valid = recursion_valid;
          stack[sptr].tlower = select(recursion_valid0,tp0.lower,tp1.lower);
          stack[sptr].u0 = u0;
          stack[sptr].u1 = u1;
          stack[sptr].depth = depth+1;
          sptr++;
        }
      }
      return found;
    }

#else
    
     template<typename NativeCurve3ff, typename Ray, typename Epilog>
     __forceinline bool intersect_bezier_recursive_jacobian(const Ray& ray, const float dt, const NativeCurve3ff& curve, const Epilog& epilog)
    {
      const Vec3fa org = zero;
      const Vec3fa dir = ray.dir;
      const unsigned int max_depth = 7;
      
      bool found = false;

      struct ShortStack
      {
        /* pushes both children */
        __forceinline void push() {
          depth++;
        }

        /* pops next node */
        __forceinline void pop() {
          short_stack += (1<<(31-depth));
          depth = 31-bsf(short_stack);
        }
        
        unsigned int depth = 0;
        unsigned int short_stack = 0;
      };

      ShortStack stack;

      do
      {
        const float u0 = (stack.short_stack+0*(1<<(31-stack.depth)))/float(0x80000000);
        const float u1 = (stack.short_stack+1*(1<<(31-stack.depth)))/float(0x80000000);
      
        /* subdivide bezier curve */
        Vec3ff P0, dP0du; curve.eval(u0,P0,dP0du); dP0du = dP0du * (u1-u0);
        Vec3ff P3, dP3du; curve.eval(u1,P3,dP3du); dP3du = dP3du * (u1-u0);
        const Vec3ff P1 = P0 + dP0du*(1.0f/3.0f); 
        const Vec3ff P2 = P3 - dP3du*(1.0f/3.0f);

        /* check if curve is well behaved, by checking deviation of tangents from straight line */
        const Vec3ff W = Vec3ff(P3-P0,0.0f);
        const Vec3ff dQ0 = abs(3.0f*(P1-P0) - W);
        const Vec3ff dQ1 = abs(3.0f*(P2-P1) - W);
        const Vec3ff dQ2 = abs(3.0f*(P3-P2) - W);
        const Vec3ff max_dQ = max(dQ0,dQ1,dQ2);
        const float m = max(max_dQ.x,max_dQ.y,max_dQ.z); //,max_dQ.w);
        const float l = length(Vec3f(W));
        const bool well_behaved = m < 0.2f*l;

        if (!well_behaved && stack.depth < max_depth) {
          stack.push();
          continue;
        }
        
        /* calculate bounding cylinders */
        const float rr1 = sqr_point_to_line_distance(Vec3f(dP0du),Vec3f(P3-P0));
        const float rr2 = sqr_point_to_line_distance(Vec3f(dP3du),Vec3f(P3-P0));
        const float maxr12 = sqrt(max(rr1,rr2));
        const float one_plus_ulp  = 1.0f+2.0f*float(ulp);
        const float one_minus_ulp = 1.0f-2.0f*float(ulp);
        float r_outer = max(P0.w,P1.w,P2.w,P3.w)+maxr12;
        float r_inner = min(P0.w,P1.w,P2.w,P3.w)-maxr12;
        r_outer = one_plus_ulp*r_outer;
        r_inner = max(0.0f,one_minus_ulp*r_inner);
        const Cylinder cylinder_outer(Vec3f(P0),Vec3f(P3),r_outer);
        const Cylinder cylinder_inner(Vec3f(P0),Vec3f(P3),r_inner);
        
        /* intersect with outer cylinder */
        BBox<float> tc_outer; float u_outer0; Vec3fa Ng_outer0; float u_outer1; Vec3fa Ng_outer1;
        if (!cylinder_outer.intersect(org,dir,tc_outer,u_outer0,Ng_outer0,u_outer1,Ng_outer1))
        {
          stack.pop();
          continue;
        }
                
        /* intersect with cap-planes */
        BBox<float> tp(ray.tnear()-dt,ray.tfar-dt);
        tp = embree::intersect(tp,tc_outer);
        BBox<float> h0 = HalfPlane(Vec3f(P0),+Vec3f(dP0du)).intersect(org,dir);
        tp = embree::intersect(tp,h0);
        BBox<float> h1 = HalfPlane(Vec3f(P3),-Vec3f(dP3du)).intersect(org,dir);
        tp = embree::intersect(tp,h1);
        if (tp.lower > tp.upper)
        {
          stack.pop();
          continue;
        }
        
        bool valid = true;
        
        /* clamp and correct u parameter */
        u_outer0 = clamp(u_outer0,float(0.0f),float(1.0f));
        u_outer1 = clamp(u_outer1,float(0.0f),float(1.0f));
        u_outer0 = lerp(u0,u1,u_outer0);
        u_outer1 = lerp(u0,u1,u_outer1);
        
        /* intersect with inner cylinder */
        BBox<float> tc_inner;
        float u_inner0 = zero; Vec3fa Ng_inner0 = zero; float u_inner1 = zero; Vec3fa Ng_inner1 = zero;
        const bool valid_inner =  cylinder_inner.intersect(org,dir,tc_inner,u_inner0,Ng_inner0,u_inner1,Ng_inner1);

        /* subtract the inner interval from the current hit interval */
        BBox<float> tp0, tp1;
        subtract(tp,tc_inner,tp0,tp1);
        bool valid0 = valid & (tp0.lower <= tp0.upper);
        bool valid1 = valid & (tp1.lower <= tp1.upper);
        if (!(valid0 | valid1))
        {
          stack.pop();
          continue;
        }

        /* at the unstable area we subdivide deeper */
        const bool unstable0 = valid0 && ((!valid_inner) | (abs(dot(Vec3fa(ray.dir),Ng_inner0)) < 0.3f));
        const bool unstable1 = valid1 && ((!valid_inner) | (abs(dot(Vec3fa(ray.dir),Ng_inner1)) < 0.3f));
    
        if ((unstable0 | unstable1) && (stack.depth < max_depth)) {
           stack.push();
           continue;
         }

        if (valid0)
          found |= intersect_bezier_iterative_jacobian(ray,dt,curve,u_outer0,tp0.lower,epilog);
          
        /* the far hit cannot be closer, thus skip if we hit entry already */
        valid1 &= tp1.lower+dt <= ray.tfar;
        
        /* iterate over second hit */
        if (valid1)
          found |= intersect_bezier_iterative_jacobian(ray,dt,curve,u_outer1,tp1.upper,epilog);

        stack.pop();
        
      } while (stack.short_stack != 0x80000000);

      return found;
    }

#endif
    
    template<template<typename Ty> class NativeCurve>
    struct SweepCurve1Intersector1
    {
      typedef NativeCurve<Vec3ff> NativeCurve3ff;
      
      template<typename Ray, typename Epilog>
      __forceinline bool intersect(const CurvePrecalculations1& pre, Ray& ray,
                                RayQueryContext* context,
                                const CurveGeometry* geom, const unsigned int primID,
                                const Vec3ff& v0, const Vec3ff& v1, const Vec3ff& v2, const Vec3ff& v3,
                                const Epilog& epilog)
      {
        STAT3(normal.trav_prims,1,1,1);

        /* move ray closer to make intersection stable */
        NativeCurve3ff curve0(v0,v1,v2,v3);
        curve0 = enlargeRadiusToMinWidth(context,geom,ray.org,curve0);
        const float dt = dot(curve0.center()-ray.org,ray.dir)*rcp(dot(ray.dir,ray.dir));
        const Vec3ff ref(madd(Vec3fa(dt),ray.dir,ray.org),0.0f);
        const NativeCurve3ff curve1 = curve0-ref;
        return intersect_bezier_recursive_jacobian(ray,dt,curve1,epilog);
      }
    };

    template<template<typename Ty> class NativeCurve, int K>
    struct SweepCurve1IntersectorK
    {
      typedef NativeCurve<Vec3ff> NativeCurve3ff;
      
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
                                   RayQueryContext* context,
                                   const CurveGeometry* geom, const unsigned int primID,
                                   const Vec3ff& v0, const Vec3ff& v1, const Vec3ff& v2, const Vec3ff& v3,
                                   const Epilog& epilog)
      {
        STAT3(normal.trav_prims,1,1,1);
        Ray1 ray(vray,k);

        /* move ray closer to make intersection stable */
        NativeCurve3ff curve0(v0,v1,v2,v3);
        curve0 = enlargeRadiusToMinWidth(context,geom,ray.org,curve0);
        const float dt = dot(curve0.center()-ray.org,ray.dir)*rcp(dot(ray.dir,ray.dir));
        const Vec3ff ref(madd(Vec3fa(dt),ray.dir,ray.org),0.0f);
        const NativeCurve3ff curve1 = curve0-ref;
        return intersect_bezier_recursive_jacobian(ray,dt,curve1,epilog);
      }
    };
  }
}
