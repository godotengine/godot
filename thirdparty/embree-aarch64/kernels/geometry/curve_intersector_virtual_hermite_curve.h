// Copyright 2020 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    void AddVirtualCurveHermiteCurveInterector4i(VirtualCurveIntersector &prim);
    void AddVirtualCurveHermiteCurveInterector4v(VirtualCurveIntersector &prim);
    void AddVirtualCurveHermiteCurveInterector4iMB(VirtualCurveIntersector &prim);
#if defined(__AVX__)
    void AddVirtualCurveHermiteCurveInterector8i(VirtualCurveIntersector &prim);
    void AddVirtualCurveHermiteCurveInterector8v(VirtualCurveIntersector &prim);
    void AddVirtualCurveHermiteCurveInterector8iMB(VirtualCurveIntersector &prim);
#endif
  }
}
