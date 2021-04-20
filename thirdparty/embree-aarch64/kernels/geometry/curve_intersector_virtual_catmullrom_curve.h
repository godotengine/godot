// Copyright 2020 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    void AddVirtualCurveCatmullRomCurveInterector4i(VirtualCurveIntersector &prim);
    void AddVirtualCurveCatmullRomCurveInterector4v(VirtualCurveIntersector &prim);
    void AddVirtualCurveCatmullRomCurveInterector4iMB(VirtualCurveIntersector &prim);
#if defined(__AVX__)
    void AddVirtualCurveCatmullRomCurveInterector8i(VirtualCurveIntersector &prim);
    void AddVirtualCurveCatmullRomCurveInterector8v(VirtualCurveIntersector &prim);
    void AddVirtualCurveCatmullRomCurveInterector8iMB(VirtualCurveIntersector &prim);
#endif
  }
}
