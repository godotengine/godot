// Copyright 2020 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    void AddVirtualCurveBSplineCurveInterector4i(VirtualCurveIntersector &prim);
    void AddVirtualCurveBSplineCurveInterector4v(VirtualCurveIntersector &prim);
    void AddVirtualCurveBSplineCurveInterector4iMB(VirtualCurveIntersector &prim);
#if defined(__AVX__)
    void AddVirtualCurveBSplineCurveInterector8i(VirtualCurveIntersector &prim);
    void AddVirtualCurveBSplineCurveInterector8v(VirtualCurveIntersector &prim);
    void AddVirtualCurveBSplineCurveInterector8iMB(VirtualCurveIntersector &prim);
#endif
  }
}
