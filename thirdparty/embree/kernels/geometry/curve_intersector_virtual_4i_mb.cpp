// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
 
#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    VirtualCurveIntersector* VirtualCurveIntersector4iMB()
    {
      static VirtualCurveIntersector function_local_static_prim = []()
      {
        VirtualCurveIntersector intersector;
        intersector.vtbl[Geometry::GTY_SPHERE_POINT] = SphereNiMBIntersectors<4>();
        intersector.vtbl[Geometry::GTY_DISC_POINT] = DiscNiMBIntersectors<4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_DISC_POINT] = OrientedDiscNiMBIntersectors<4>();
        intersector.vtbl[Geometry::GTY_CONE_LINEAR_CURVE ] = LinearConeNiMBIntersectors<4>();
        intersector.vtbl[Geometry::GTY_ROUND_LINEAR_CURVE ] = LinearRoundConeNiMBIntersectors<4>();
        intersector.vtbl[Geometry::GTY_FLAT_LINEAR_CURVE ] = LinearRibbonNiMBIntersectors<4>();
        intersector.vtbl[Geometry::GTY_ROUND_BEZIER_CURVE] = CurveNiMBIntersectors <BezierCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_BEZIER_CURVE ] = RibbonNiMBIntersectors<BezierCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_BEZIER_CURVE] = OrientedCurveNiMBIntersectors<BezierCurveT,4>();
        intersector.vtbl[Geometry::GTY_ROUND_BSPLINE_CURVE] = CurveNiMBIntersectors <BSplineCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_BSPLINE_CURVE ] = RibbonNiMBIntersectors<BSplineCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_BSPLINE_CURVE] = OrientedCurveNiMBIntersectors<BSplineCurveT,4>();
        intersector.vtbl[Geometry::GTY_ROUND_HERMITE_CURVE] = HermiteCurveNiMBIntersectors <HermiteCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_HERMITE_CURVE ] = HermiteRibbonNiMBIntersectors<HermiteCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_HERMITE_CURVE] = HermiteOrientedCurveNiMBIntersectors<HermiteCurveT,4>();
        intersector.vtbl[Geometry::GTY_ROUND_CATMULL_ROM_CURVE] = CurveNiMBIntersectors <CatmullRomCurveT,4>();
        intersector.vtbl[Geometry::GTY_FLAT_CATMULL_ROM_CURVE ] = RibbonNiMBIntersectors<CatmullRomCurveT,4>();
        intersector.vtbl[Geometry::GTY_ORIENTED_CATMULL_ROM_CURVE] = OrientedCurveNiMBIntersectors<CatmullRomCurveT,4>();
        return intersector;
      }();
      return &function_local_static_prim;
    }
  }
}
