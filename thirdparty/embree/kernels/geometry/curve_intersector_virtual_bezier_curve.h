// Copyright 2020 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    void AddVirtualCurveBezierCurveInterector4i(VirtualCurveIntersector &prim);
    void AddVirtualCurveBezierCurveInterector4v(VirtualCurveIntersector &prim);
    void AddVirtualCurveBezierCurveInterector4iMB(VirtualCurveIntersector &prim);
#if defined(__AVX__)
    void AddVirtualCurveBezierCurveInterector8i(VirtualCurveIntersector &prim);
    void AddVirtualCurveBezierCurveInterector8v(VirtualCurveIntersector &prim);
    void AddVirtualCurveBezierCurveInterector8iMB(VirtualCurveIntersector &prim);
#endif
  }
}
