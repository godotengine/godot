// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/default.h"
#include "bezier_curve.h"

namespace embree
{
  template<typename Vertex>
    struct HermiteCurveT : BezierCurveT<Vertex>
    {
      __forceinline HermiteCurveT() {}

      __forceinline HermiteCurveT(const BezierCurveT<Vertex>& curve)
        : BezierCurveT<Vertex>(curve) {}
      
      __forceinline HermiteCurveT(const Vertex& v0, const Vertex& t0, const Vertex& v1, const Vertex& t1)
        : BezierCurveT<Vertex>(v0,madd(1.0f/3.0f,t0,v0),nmadd(1.0f/3.0f,t1,v1),v1) {}

      __forceinline HermiteCurveT<Vec3ff> xfm_pr(const LinearSpace3fa& space, const Vec3fa& p) const
      {
        const Vec3ff q0(xfmVector(space,this->v0-p), this->v0.w);
        const Vec3ff q1(xfmVector(space,this->v1-p), this->v1.w);
        const Vec3ff q2(xfmVector(space,this->v2-p), this->v2.w);
        const Vec3ff q3(xfmVector(space,this->v3-p), this->v3.w);
        return BezierCurveT<Vec3ff>(q0,q1,q2,q3);
      }
    };

  template<typename CurveGeometry>
  __forceinline HermiteCurveT<Vec3ff> enlargeRadiusToMinWidth(const IntersectContext* context, const CurveGeometry* geom, const Vec3fa& ray_org, const HermiteCurveT<Vec3ff>& curve) {
    return HermiteCurveT<Vec3ff>(enlargeRadiusToMinWidth(context,geom,ray_org,BezierCurveT<Vec3ff>(curve)));
  }
  
  typedef HermiteCurveT<Vec3fa> HermiteCurve3fa;
}

