// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"

namespace embree
{
  namespace isa
  {
    struct Cone
    {
      const Vec3fa p0; //!< start position of cone
      const Vec3fa p1; //!< end position of cone
      const float r0;  //!< start radius of cone
      const float r1;  //!< end radius of cone

      __forceinline Cone(const Vec3fa& p0, const float r0, const Vec3fa& p1, const float r1) 
        : p0(p0), p1(p1), r0(r0), r1(r1) {}

      __forceinline bool intersect(const Vec3fa& org, const Vec3fa& dir, 
                                   BBox1f& t_o, 
                                   float& u0_o, Vec3fa& Ng0_o, 
                                   float& u1_o, Vec3fa& Ng1_o) const 
      {
        /* calculate quadratic equation to solve */
        const Vec3fa v0 = p0-org;
        const Vec3fa v1 = p1-org;
        
        const float rl = rcp_length(v1-v0);
        const Vec3fa P0 = v0, dP = (v1-v0)*rl;
        const float dr = (r1-r0)*rl;
        const Vec3fa O = -P0, dO = dir;
        
        const float dOdO = dot(dO,dO);
        const float OdO = dot(dO,O);
        const float OO = dot(O,O);
        const float dOz = dot(dP,dO);
        const float Oz = dot(dP,O);

        const float R = r0 + Oz*dr;          
        const float A = dOdO - sqr(dOz) * (1.0f+sqr(dr));
        const float B = 2.0f * (OdO - dOz*(Oz + R*dr));
        const float C = OO - (sqr(Oz) + sqr(R));

        /* we miss the cone if determinant is smaller than zero */
        const float D = B*B - 4.0f*A*C;
        if (D < 0.0f) return false;

        /* special case for rays that are "parallel" to the cone */
        const float eps = float(1<<8)*float(ulp)*max(abs(dOdO),abs(sqr(dOz)));
        if (unlikely(abs(A) < eps))
        {
          /* cylinder case */
          if (abs(dr) < 16.0f*float(ulp)) {
            if (C <= 0.0f) { t_o = BBox1f(neg_inf,pos_inf); return true; } 
            else           { t_o = BBox1f(pos_inf,neg_inf); return false; }
          }

          /* cone case */
          else 
          {
            /* if we hit the negative cone there cannot be a hit */
            const float t = -C/B;
            const float z0 = Oz+t*dOz;
            const float z0r = r0+z0*dr;
            if (z0r < 0.0f) return false;

            /* test if we start inside or outside the cone */
            if (dOz*dr > 0.0f) t_o = BBox1f(t,pos_inf);
            else               t_o = BBox1f(neg_inf,t);
          }
        }

        /* standard case for "non-parallel" rays */
        else
        {
          const float Q = sqrt(D);
          const float rcp_2A = rcp(2.0f*A);
          t_o.lower = (-B-Q)*rcp_2A;
          t_o.upper = (-B+Q)*rcp_2A;
          
          /* standard case where both hits are on same cone */
          if (likely(A > 0.0f)) {
            const float z0 = Oz+t_o.lower*dOz;
            const float z0r = r0+z0*dr;
            if (z0r < 0.0f) return false;
          } 

          /* special case where the hits are on the positive and negative cone */
          else 
          {
            /* depending on the ray direction and the open direction
             * of the cone we have a hit from inside or outside the
             * cone */
            if (dOz*dr > 0) t_o.upper = pos_inf;
            else            t_o.lower = neg_inf;
          }
        }

        /* calculates u and Ng for near hit */
        {
          u0_o = (Oz+t_o.lower*dOz)*rl;
          const Vec3fa Pr = t_o.lower*dir;
          const Vec3fa Pl = v0 + u0_o*(v1-v0);
          const Vec3fa R = normalize(Pr-Pl);
          const Vec3fa U = (p1-p0)+(r1-r0)*R;
          const Vec3fa V = cross(p1-p0,R);
          Ng0_o = cross(V,U);
        }

        /* calculates u and Ng for far hit */
        {
          u1_o = (Oz+t_o.upper*dOz)*rl;
          const Vec3fa Pr = t_o.upper*dir;
          const Vec3fa Pl = v0 + u1_o*(v1-v0);
          const Vec3fa R = normalize(Pr-Pl);
          const Vec3fa U = (p1-p0)+(r1-r0)*R;
          const Vec3fa V = cross(p1-p0,R);
          Ng1_o = cross(V,U);
        }
        return true;
      }

      __forceinline bool intersect(const Vec3fa& org, const Vec3fa& dir, BBox1f& t_o) const 
      {
        float u0_o; Vec3fa Ng0_o; float u1_o; Vec3fa Ng1_o;
        return intersect(org,dir,t_o,u0_o,Ng0_o,u1_o,Ng1_o);
      }

      static bool verify(const size_t id, const Cone& cone, const Ray& ray, bool shouldhit, const float t0, const float t1)
      {
        float eps = 0.001f;
        BBox1f t; bool hit;
        hit = cone.intersect(ray.org,ray.dir,t);

        bool failed = hit != shouldhit;
        if (shouldhit) failed |= std::isinf(t0) ? t0 != t.lower : (t0 == -1E6) ? t.lower > -1E6f : abs(t0-t.lower) > eps;
        if (shouldhit) failed |= std::isinf(t1) ? t1 != t.upper : (t1 == +1E6) ? t.upper < +1E6f : abs(t1-t.upper) > eps;
        if (!failed) return true;
        embree_cout << "Cone test " << id << " failed: cone = " << cone << ", ray = " << ray << ", hit = " << hit << ", t = " << t << embree_endl; 
        return false;
      }

      /* verify cone class */
      static bool verify()
      {
        bool passed = true;
        const Cone cone0(Vec3fa(0.0f,0.0f,0.0f),0.0f,Vec3fa(1.0f,0.0f,0.0f),1.0f);
        passed &= verify(0,cone0,Ray(Vec3fa(-2.0f,1.0f,0.0f),Vec3fa(+1.0f,+0.0f,+0.0f),0.0f,float(inf)),true,3.0f,pos_inf);
        passed &= verify(1,cone0,Ray(Vec3fa(+2.0f,1.0f,0.0f),Vec3fa(-1.0f,+0.0f,+0.0f),0.0f,float(inf)),true,neg_inf,1.0f);
        passed &= verify(2,cone0,Ray(Vec3fa(-1.0f,0.0f,2.0f),Vec3fa(+0.0f,+0.0f,-1.0f),0.0f,float(inf)),false,0.0f,0.0f);
        passed &= verify(3,cone0,Ray(Vec3fa(+1.0f,0.0f,2.0f),Vec3fa(+0.0f,+0.0f,-1.0f),0.0f,float(inf)),true,1.0f,3.0f);
        passed &= verify(4,cone0,Ray(Vec3fa(-1.0f,0.0f,0.0f),Vec3fa(+1.0f,+0.0f,+0.0f),0.0f,float(inf)),true,1.0f,pos_inf);
        passed &= verify(5,cone0,Ray(Vec3fa(+1.0f,0.0f,0.0f),Vec3fa(-1.0f,+0.0f,+0.0f),0.0f,float(inf)),true,neg_inf,1.0f);
        passed &= verify(6,cone0,Ray(Vec3fa(+0.0f,0.0f,1.0f),Vec3fa(+0.0f,+0.0f,-1.0f),0.0f,float(inf)),true,1.0f,1.0f);
        passed &= verify(7,cone0,Ray(Vec3fa(+0.0f,1.0f,0.0f),Vec3fa(-1.0f,-1.0f,+0.0f),0.0f,float(inf)),false,0.0f,0.0f);
        passed &= verify(8,cone0,Ray(Vec3fa(+0.0f,1.0f,0.0f),Vec3fa(+1.0f,-1.0f,+0.0f),0.0f,float(inf)),true,0.5f,+1E6);
        passed &= verify(9,cone0,Ray(Vec3fa(+0.0f,1.0f,0.0f),Vec3fa(-1.0f,+1.0f,+0.0f),0.0f,float(inf)),true,-1E6,-0.5f);
        const Cone cone1(Vec3fa(0.0f,0.0f,0.0f),1.0f,Vec3fa(1.0f,0.0f,0.0f),0.0f);
        passed &= verify(10,cone1,Ray(Vec3fa(-2.0f,1.0f,0.0f),Vec3fa(+1.0f,+0.0f,+0.0f),0.0f,float(inf)),true,neg_inf,2.0f);
        passed &= verify(11,cone1,Ray(Vec3fa(-1.0f,0.0f,2.0f),Vec3fa(+0.0f,+0.0f,-1.0f),0.0f,float(inf)),true,0.0f,4.0f);
        const Cone cylinder(Vec3fa(0.0f,0.0f,0.0f),1.0f,Vec3fa(1.0f,0.0f,0.0f),1.0f);
        passed &= verify(12,cylinder,Ray(Vec3fa(-2.0f,1.0f,0.0f),Vec3fa( 0.0f,-1.0f,+0.0f),0.0f,float(inf)),true,0.0f,2.0f);
        passed &= verify(13,cylinder,Ray(Vec3fa(+2.0f,1.0f,0.0f),Vec3fa( 0.0f,-1.0f,+0.0f),0.0f,float(inf)),true,0.0f,2.0f);
        passed &= verify(14,cylinder,Ray(Vec3fa(+2.0f,1.0f,2.0f),Vec3fa( 0.0f,-1.0f,+0.0f),0.0f,float(inf)),false,0.0f,0.0f);
        passed &= verify(15,cylinder,Ray(Vec3fa(+0.0f,0.0f,0.0f),Vec3fa( 1.0f, 0.0f,+0.0f),0.0f,float(inf)),true,neg_inf,pos_inf);
        passed &= verify(16,cylinder,Ray(Vec3fa(+0.0f,0.0f,0.0f),Vec3fa(-1.0f, 0.0f,+0.0f),0.0f,float(inf)),true,neg_inf,pos_inf);
        passed &= verify(17,cylinder,Ray(Vec3fa(+0.0f,2.0f,0.0f),Vec3fa( 1.0f, 0.0f,+0.0f),0.0f,float(inf)),false,pos_inf,neg_inf);
        passed &= verify(18,cylinder,Ray(Vec3fa(+0.0f,2.0f,0.0f),Vec3fa(-1.0f, 0.0f,+0.0f),0.0f,float(inf)),false,pos_inf,neg_inf);
        return passed;
      }

      /*! output operator */
      friend __forceinline embree_ostream operator<<(embree_ostream cout, const Cone& c) {
        return cout << "Cone { p0 = " << c.p0 << ", r0 = " << c.r0 << ", p1 = " << c.p1 << ", r1 = " << c.r1 << "}";
      }
    };

    template<int N>
      struct ConeN
    {
      typedef Vec3<vfloat<N>> Vec3vfN;
      
      const Vec3vfN p0;     //!< start position of cone
      const Vec3vfN p1;     //!< end position of cone
      const vfloat<N> r0;   //!< start radius of cone
      const vfloat<N> r1;   //!< end radius of cone

      __forceinline ConeN(const Vec3vfN& p0, const vfloat<N>& r0, const Vec3vfN& p1, const vfloat<N>& r1) 
        : p0(p0), p1(p1), r0(r0), r1(r1) {}

      __forceinline Cone operator[] (const size_t i) const
      {
        assert(i<N);
        return Cone(Vec3fa(p0.x[i],p0.y[i],p0.z[i]),r0[i],Vec3fa(p1.x[i],p1.y[i],p1.z[i]),r1[i]);
      }

      __forceinline vbool<N> intersect(const Vec3fa& org, const Vec3fa& dir, 
                                       BBox<vfloat<N>>& t_o, 
                                       vfloat<N>& u0_o, Vec3vfN& Ng0_o, 
                                       vfloat<N>& u1_o, Vec3vfN& Ng1_o) const
      {
        /* calculate quadratic equation to solve */
        const Vec3vfN v0 = p0-Vec3vfN(org);
        const Vec3vfN v1 = p1-Vec3vfN(org);

        const vfloat<N> rl = rcp_length(v1-v0);
        const Vec3vfN P0 = v0, dP = (v1-v0)*rl;
        const vfloat<N> dr = (r1-r0)*rl;
        const Vec3vfN O = -P0, dO = dir;
       
        const vfloat<N> dOdO = dot(dO,dO);
        const vfloat<N> OdO = dot(dO,O);
        const vfloat<N> OO = dot(O,O);
        const vfloat<N> dOz = dot(dP,dO);
        const vfloat<N> Oz = dot(dP,O);
        
        const vfloat<N> R = r0 + Oz*dr;          
        const vfloat<N> A = dOdO - sqr(dOz) * (vfloat<N>(1.0f)+sqr(dr));
        const vfloat<N> B = 2.0f * (OdO - dOz*(Oz + R*dr));
        const vfloat<N> C = OO - (sqr(Oz) + sqr(R));

        /* we miss the cone if determinant is smaller than zero */
        const vfloat<N> D = B*B - 4.0f*A*C;
        vbool<N> valid = D >= 0.0f;
        if (none(valid)) return valid;

        /* special case for rays that are "parallel" to the cone */
        const vfloat<N> eps = float(1<<8)*float(ulp)*max(abs(dOdO),abs(sqr(dOz)));
        const vbool<N> validt = valid &  (abs(A) < eps);
        const vbool<N> validf = valid & !(abs(A) < eps);
        if (unlikely(any(validt)))
        {
          const vboolx validtt = validt & (abs(dr) <  16.0f*float(ulp));
          const vboolx validtf = validt & (abs(dr) >= 16.0f*float(ulp));
          
          /* cylinder case */
          if (unlikely(any(validtt))) 
          {
            t_o.lower = select(validtt, select(C <= 0.0f, vfloat<N>(neg_inf), vfloat<N>(pos_inf)), t_o.lower);
            t_o.upper = select(validtt, select(C <= 0.0f, vfloat<N>(pos_inf), vfloat<N>(neg_inf)), t_o.upper);
            valid &= !validtt | C <= 0.0f;
          }

          /* cone case */
          if (any(validtf)) 
          {
            /* if we hit the negative cone there cannot be a hit */
            const vfloat<N> t = -C/B;
            const vfloat<N> z0 = Oz+t*dOz;
            const vfloat<N> z0r = r0+z0*dr;
            valid &= !validtf | z0r >= 0.0f;

            /* test if we start inside or outside the cone */
            t_o.lower = select(validtf, select(dOz*dr > 0.0f, t, vfloat<N>(neg_inf)), t_o.lower);
            t_o.upper = select(validtf, select(dOz*dr > 0.0f, vfloat<N>(pos_inf), t), t_o.upper);
          }
        }

        /* standard case for "non-parallel" rays */
        if (likely(any(validf)))
        {
          const vfloat<N> Q = sqrt(D);
          const vfloat<N> rcp_2A = 0.5f*rcp(A);
          t_o.lower = select(validf, (-B-Q)*rcp_2A, t_o.lower);
          t_o.upper = select(validf, (-B+Q)*rcp_2A, t_o.upper);
          
          /* standard case where both hits are on same cone */
          const vbool<N> validft = validf &   A>0.0f;
          const vbool<N> validff = validf & !(A>0.0f);
          if (any(validft)) {
            const vfloat<N> z0 = Oz+t_o.lower*dOz;
            const vfloat<N> z0r = r0+z0*dr;
            valid &= !validft | z0r >= 0.0f;
          } 

          /* special case where the hits are on the positive and negative cone */
          if (any(validff)) {
            /* depending on the ray direction and the open direction
             * of the cone we have a hit from inside or outside the
             * cone */
            t_o.lower = select(validff, select(dOz*dr > 0.0f, t_o.lower, float(neg_inf)), t_o.lower);
            t_o.upper = select(validff, select(dOz*dr > 0.0f, float(pos_inf), t_o.upper), t_o.upper);
          }
        }

        /* calculates u and Ng for near hit */
        {
          u0_o = (Oz+t_o.lower*dOz)*rl;
          const Vec3vfN Pr = t_o.lower*Vec3vfN(dir);
          const Vec3vfN Pl = v0 + u0_o*(v1-v0);
          const Vec3vfN R = normalize(Pr-Pl);
          const Vec3vfN U = (p1-p0)+(r1-r0)*R;
          const Vec3vfN V = cross(p1-p0,R);
          Ng0_o = cross(V,U);
        }

        /* calculates u and Ng for far hit */
        {
          u1_o = (Oz+t_o.upper*dOz)*rl;
          const Vec3vfN Pr = t_o.lower*Vec3vfN(dir);
          const Vec3vfN Pl = v0 + u1_o*(v1-v0);
          const Vec3vfN R = normalize(Pr-Pl);
          const Vec3vfN U = (p1-p0)+(r1-r0)*R;
          const Vec3vfN V = cross(p1-p0,R);
          Ng1_o = cross(V,U);
        }
        return valid;
      }
 
      __forceinline vbool<N> intersect(const Vec3fa& org, const Vec3fa& dir, BBox<vfloat<N>>& t_o) const
      {
        vfloat<N> u0_o; Vec3vfN Ng0_o; vfloat<N> u1_o; Vec3vfN Ng1_o;
        return intersect(org,dir,t_o,u0_o,Ng0_o,u1_o,Ng1_o);
      }
    };
  }
}

