// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
 
#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    VirtualCurveIntersector* VirtualCurveIntersector4v()
    {
      static VirtualCurveIntersector function_local_static_prim = []()
      {
        VirtualCurveIntersector intersector;
        intersector.vtbl[Geometry::GTY_SPHERE_POINT] = SphereNiIntersectors<4>();
        intersector.vtbl[Geometry::GTY_DISC_POINT] = DiscNiIntersectors<4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_DISC_POINT] = OrientedDiscNiIntersectors<4>();
        intersector.vtbl[Geometry::GTY_CONE_LINEAR_CURVE ] = LinearConeNiIntersectors<4>();
        intersector.vtbl[Geometry::GTY_ROUND_LINEAR_CURVE ] = LinearRoundConeNiIntersectors<4>();
        intersector.vtbl[Geometry::GTY_FLAT_LINEAR_CURVE ] = LinearRibbonNiIntersectors<4>();
        intersector.vtbl[Geometry::GTY_ROUND_BEZIER_CURVE] = CurveNvIntersectors <BezierCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_BEZIER_CURVE ] = RibbonNvIntersectors<BezierCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_BEZIER_CURVE] = OrientedCurveNiIntersectors<BezierCurveT,4>();
        intersector.vtbl[Geometry::GTY_ROUND_BSPLINE_CURVE] = CurveNvIntersectors <BSplineCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_BSPLINE_CURVE ] = RibbonNvIntersectors<BSplineCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_BSPLINE_CURVE] = OrientedCurveNiIntersectors<BSplineCurveT,4>();
        intersector.vtbl[Geometry::GTY_ROUND_HERMITE_CURVE] = HermiteCurveNiIntersectors <HermiteCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_HERMITE_CURVE ] = HermiteRibbonNiIntersectors<HermiteCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_HERMITE_CURVE] = HermiteOrientedCurveNiIntersectors<HermiteCurveT,4>();
        intersector.vtbl[Geometry::GTY_ROUND_CATMULL_ROM_CURVE] = CurveNiIntersectors <CatmullRomCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_CATMULL_ROM_CURVE ] = RibbonNiIntersectors<CatmullRomCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_CATMULL_ROM_CURVE] = OrientedCurveNiIntersectors<CatmullRomCurveT,4>();
        return intersector;
      }();
      return &function_local_static_prim;
    }
  }
}
