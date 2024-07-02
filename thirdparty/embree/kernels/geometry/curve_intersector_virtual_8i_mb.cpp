// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
 
#include "curve_intersector_virtual.h"
#include "intersector_epilog.h"

#include "../subdiv/bezier_curve.h"
#include "../subdiv/bspline_curve.h"
#include "../subdiv/hermite_curve.h"
#include "../subdiv/catmullrom_curve.h"

#include "spherei_intersector.h"
#include "disci_intersector.h"

#include "linei_intersector.h"

#include "curveNi_intersector.h"
#include "curveNv_intersector.h"
#include "curveNi_mb_intersector.h"

#include "curve_intersector_distance.h"
#include "curve_intersector_ribbon.h"
#include "curve_intersector_oriented.h"
#include "curve_intersector_sweep.h"

namespace embree
{
  namespace isa
  {
#if defined (__AVX__)
    
    VirtualCurveIntersector* VirtualCurveIntersector8iMB()
    {
      static VirtualCurveIntersector function_local_static_prim = []()
      {
        VirtualCurveIntersector intersector;
        intersector.vtbl[Geometry::GTY_SPHERE_POINT] = SphereNiMBIntersectors<8>();
        intersector.vtbl[Geometry::GTY_DISC_POINT] = DiscNiMBIntersectors<8>();
        intersector.vtbl[Geometry::GTY_ORIENTED_DISC_POINT] = OrientedDiscNiMBIntersectors<8>();
        intersector.vtbl[Geometry::GTY_CONE_LINEAR_CURVE ] = LinearConeNiMBIntersectors<8>();
        intersector.vtbl[Geometry::GTY_ROUND_LINEAR_CURVE ] = LinearRoundConeNiMBIntersectors<8>();
        intersector.vtbl[Geometry::GTY_FLAT_LINEAR_CURVE ] = LinearRibbonNiMBIntersectors<8>();
        intersector.vtbl[Geometry::GTY_ROUND_BEZIER_CURVE] = CurveNiMBIntersectors <BezierCurveT,8>();
        intersector.vtbl[Geometry::GTY_FLAT_BEZIER_CURVE ] = RibbonNiMBIntersectors<BezierCurveT,8>();
        intersector.vtbl[Geometry::GTY_ORIENTED_BEZIER_CURVE] = OrientedCurveNiMBIntersectors<BezierCurveT,8>();
        intersector.vtbl[Geometry::GTY_ROUND_BSPLINE_CURVE] = CurveNiMBIntersectors <BSplineCurveT,8>();
        intersector.vtbl[Geometry::GTY_FLAT_BSPLINE_CURVE ] = RibbonNiMBIntersectors<BSplineCurveT,8>();
        intersector.vtbl[Geometry::GTY_ORIENTED_BSPLINE_CURVE] = OrientedCurveNiMBIntersectors<BSplineCurveT,8>();
        intersector.vtbl[Geometry::GTY_ROUND_HERMITE_CURVE] = HermiteCurveNiMBIntersectors <HermiteCurveT,8>();
        intersector.vtbl[Geometry::GTY_FLAT_HERMITE_CURVE ] = HermiteRibbonNiMBIntersectors<HermiteCurveT,8>();
        intersector.vtbl[Geometry::GTY_ORIENTED_HERMITE_CURVE] = HermiteOrientedCurveNiMBIntersectors<HermiteCurveT,8>();
        intersector.vtbl[Geometry::GTY_ROUND_CATMULL_ROM_CURVE] = CurveNiMBIntersectors <CatmullRomCurveT,8>();
        intersector.vtbl[Geometry::GTY_FLAT_CATMULL_ROM_CURVE ] = RibbonNiMBIntersectors<CatmullRomCurveT,8>();
        intersector.vtbl[Geometry::GTY_ORIENTED_CATMULL_ROM_CURVE] = OrientedCurveNiMBIntersectors<CatmullRomCurveT,8>();
        return intersector;
      }();
      return &function_local_static_prim;
    }
  
#endif
  }
}
