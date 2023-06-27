// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "primitive.h"

namespace embree
{ 
  namespace isa
  {
    struct TriangleTriangleIntersector
    {
      __forceinline static float T(float pa0, float pa1, float da0, float da1) {
        return pa0 + (pa1-pa0)*da0/(da0-da1);
      }
      
      __forceinline static bool point_line_side(const Vec2f& p, const Vec2f& a0, const Vec2f& a1) {
        return det(p-a0,a0-a1) >= 0.0f;
      }
      
      __forceinline static bool point_inside_triangle(const Vec2f& p, const Vec2f& a, const Vec2f& b, const Vec2f& c) 
      {
        const bool pab = point_line_side(p,a,b); 
        const bool pbc = point_line_side(p,b,c);
        const bool pca = point_line_side(p,c,a);
        return pab == pbc && pab == pca;
      }
      
      __forceinline static bool intersect_line_line(const Vec2f& a0, const Vec2f& a1, const Vec2f& b0, const Vec2f& b1)
      {
        const bool different_sides0 = point_line_side(b0,a0,a1) != point_line_side(b1,a0,a1);
        const bool different_sides1 = point_line_side(a0,b0,b1) != point_line_side(a1,b0,b1);
        return different_sides0 && different_sides1;
      }
      
      __forceinline static bool intersect_triangle_triangle (const Vec2f& a0, const Vec2f& a1, const Vec2f& a2, 
                                                             const Vec2f& b0, const Vec2f& b1, const Vec2f& b2)
      {
        const bool a01_b01 = intersect_line_line(a0,a1,b0,b1); 
        if (a01_b01) return true;
        const bool a01_b12 = intersect_line_line(a0,a1,b1,b2);
        if (a01_b12) return true;
        const bool a01_b20 = intersect_line_line(a0,a1,b2,b0);
        if (a01_b20) return true;
        const bool a12_b01 = intersect_line_line(a1,a2,b0,b1);
        if (a12_b01) return true;
        const bool a12_b12 = intersect_line_line(a1,a2,b1,b2);
        if (a12_b12) return true;
        const bool a12_b20 = intersect_line_line(a1,a2,b2,b0);
        if (a12_b20) return true;
        const bool a20_b01 = intersect_line_line(a2,a0,b0,b1);
        if (a20_b01) return true;
        const bool a20_b12 = intersect_line_line(a2,a0,b1,b2);
        if (a20_b12) return true;
        const bool a20_b20 = intersect_line_line(a2,a0,b2,b0);
        if (a20_b20) return true;
        
        bool a_in_b = point_inside_triangle(a0,b0,b1,b2) && point_inside_triangle(a1,b0,b1,b2) && point_inside_triangle(a2,b0,b1,b2);
        if (a_in_b) return true;
        
        bool b_in_a = point_inside_triangle(b0,a0,a1,a2) && point_inside_triangle(b1,a0,a1,a2) && point_inside_triangle(b2,a0,a1,a2);
        if (b_in_a) return true;
        
        return false;
      }
      
      static bool intersect_triangle_triangle (const Vec3fa& a0, const Vec3fa& a1, const Vec3fa& a2,
                                               const Vec3fa& b0, const Vec3fa& b1, const Vec3fa& b2)
      {
        const float eps = 1E-5f;
        
        /* calculate triangle planes */
        const Vec3fa Na = cross(a1-a0,a2-a0);
        const float  Ca = dot(Na,a0);
        const Vec3fa Nb = cross(b1-b0,b2-b0);
        const float  Cb = dot(Nb,b0);
        
        /* project triangle A onto plane B */
        const float da0 = dot(Nb,a0)-Cb;
        const float da1 = dot(Nb,a1)-Cb;
        const float da2 = dot(Nb,a2)-Cb;
        if (max(da0,da1,da2) < -eps) return false;
        if (min(da0,da1,da2) > +eps) return false;
        //CSTAT(bvh_collide_prim_intersections4++);
        
        /* project triangle B onto plane A */
        const float db0 = dot(Na,b0)-Ca;
        const float db1 = dot(Na,b1)-Ca;
        const float db2 = dot(Na,b2)-Ca;
        if (max(db0,db1,db2) < -eps) return false;
        if (min(db0,db1,db2) > +eps) return false;
        //CSTAT(bvh_collide_prim_intersections5++);
        
        if (unlikely((std::fabs(da0) < eps && std::fabs(da1) < eps && std::fabs(da2) < eps) ||
                     (std::fabs(db0) < eps && std::fabs(db1) < eps && std::fabs(db2) < eps)))
        {
          const size_t dz = maxDim(Na);
          const size_t dx = (dz+1)%3;
          const size_t dy = (dx+1)%3;
          const Vec2f A0(a0[dx],a0[dy]);
          const Vec2f A1(a1[dx],a1[dy]);
          const Vec2f A2(a2[dx],a2[dy]);
          const Vec2f B0(b0[dx],b0[dy]);
          const Vec2f B1(b1[dx],b1[dy]);
          const Vec2f B2(b2[dx],b2[dy]);
          return intersect_triangle_triangle(A0,A1,A2,B0,B1,B2);
        }
        
        const Vec3fa D = cross(Na,Nb);
        const float pa0 = dot(D,a0);
        const float pa1 = dot(D,a1);
        const float pa2 = dot(D,a2);
        const float pb0 = dot(D,b0);
        const float pb1 = dot(D,b1);
        const float pb2 = dot(D,b2);
        
        BBox1f ba = empty;
        if (min(da0,da1) <= 0.0f && max(da0,da1) >= 0.0f && abs(da0-da1) > 0.0f) ba.extend(T(pa0,pa1,da0,da1));
        if (min(da1,da2) <= 0.0f && max(da1,da2) >= 0.0f && abs(da1-da2) > 0.0f) ba.extend(T(pa1,pa2,da1,da2));
        if (min(da2,da0) <= 0.0f && max(da2,da0) >= 0.0f && abs(da2-da0) > 0.0f) ba.extend(T(pa2,pa0,da2,da0));
        
        BBox1f bb = empty;
        if (min(db0,db1) <= 0.0f && max(db0,db1) >= 0.0f && abs(db0-db1) > 0.0f) bb.extend(T(pb0,pb1,db0,db1));
        if (min(db1,db2) <= 0.0f && max(db1,db2) >= 0.0f && abs(db1-db2) > 0.0f) bb.extend(T(pb1,pb2,db1,db2));
        if (min(db2,db0) <= 0.0f && max(db2,db0) >= 0.0f && abs(db2-db0) > 0.0f) bb.extend(T(pb2,pb0,db2,db0));
        
        return conjoint(ba,bb);
      }
    };
  }
}

  
