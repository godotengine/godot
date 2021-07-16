// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace embree
{
  namespace isa
  {
    /*! Intersects a ray with a quad with backface culling
     *  enabled. The quad v0,v1,v2,v3 is split into two triangles
     *  v0,v1,v3 and v2,v3,v1. The edge v1,v2 decides which of the two
     *  triangles gets intersected. */
    template<int N>
    __forceinline vbool<N> intersect_quad_backface_culling(const vbool<N>& valid0,
                                                           const Vec3fa& ray_org,
                                                           const Vec3fa& ray_dir,
                                                           const float ray_tnear,
                                                           const float ray_tfar,
                                                           const Vec3vf<N>& quad_v0,
                                                           const Vec3vf<N>& quad_v1,
                                                           const Vec3vf<N>& quad_v2,
                                                           const Vec3vf<N>& quad_v3,
                                                           vfloat<N>& u_o,
                                                           vfloat<N>& v_o,
                                                           vfloat<N>& t_o)
    {
      /* calculate vertices relative to ray origin */
      vbool<N> valid = valid0;
      const Vec3vf<N> O = Vec3vf<N>(ray_org);
      const Vec3vf<N> D = Vec3vf<N>(ray_dir);
      const Vec3vf<N> va = quad_v0-O;
      const Vec3vf<N> vb = quad_v1-O;
      const Vec3vf<N> vc = quad_v2-O;
      const Vec3vf<N> vd = quad_v3-O;

      const Vec3vf<N> edb = vb-vd;
      const vfloat<N> WW = dot(cross(vd,edb),D);
      const Vec3vf<N> v0 = select(WW <= 0.0f,va,vc);
      const Vec3vf<N> v1 = select(WW <= 0.0f,vb,vd);
      const Vec3vf<N> v2 = select(WW <= 0.0f,vd,vb);

      /* calculate edges */
      const Vec3vf<N> e0 = v2-v0;
      const Vec3vf<N> e1 = v0-v1;

      /* perform edge tests */
      const vfloat<N> U = dot(cross(v0,e0),D);
      const vfloat<N> V = dot(cross(v1,e1),D);
      valid &= max(U,V) <= 0.0f;
      if (unlikely(none(valid))) return false;

      /* calculate geometry normal and denominator */
      const Vec3vf<N> Ng = cross(e1,e0);
      const vfloat<N> den = dot(Ng,D);
      const vfloat<N> rcpDen = rcp(den);

      /* perform depth test */
      const vfloat<N> t = rcpDen*dot(v0,Ng);
      valid &= vfloat<N>(ray_tnear) <= t & t <= vfloat<N>(ray_tfar);
      if (unlikely(none(valid))) return false;

      /* avoid division by 0 */
      valid &= den != vfloat<N>(zero);
      if (unlikely(none(valid))) return false;

      /* update hit information */
      t_o = t;
      u_o = U * rcpDen;
      v_o = V * rcpDen;
      u_o = select(WW <= 0.0f,u_o,1.0f-u_o);
      v_o = select(WW <= 0.0f,v_o,1.0f-v_o);
      return valid;
    }
  }
}
