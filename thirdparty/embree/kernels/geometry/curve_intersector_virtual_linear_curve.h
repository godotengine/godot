// Copyright 2020 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    void AddVirtualCurveLinearCurveInterector4i(VirtualCurveIntersector &prim);
    void AddVirtualCurveLinearCurveInterector4v(VirtualCurveIntersector &prim);
    void AddVirtualCurveLinearCurveInterector4iMB(VirtualCurveIntersector &prim);
#if defined(__AVX__)
    void AddVirtualCurveLinearCurveInterector8i(VirtualCurveIntersector &prim);
    void AddVirtualCurveLinearCurveInterector8v(VirtualCurveIntersector &prim);
    void AddVirtualCurveLinearCurveInterector8iMB(VirtualCurveIntersector &prim);
#endif
  }
}
