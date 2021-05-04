// Copyright 2020 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "curve_intersector_virtual.h"

namespace embree
{
  namespace isa
  {
    void AddVirtualCurvePointInterector4i(VirtualCurveIntersector &prim);
    void AddVirtualCurvePointInterector4v(VirtualCurveIntersector &prim);
    void AddVirtualCurvePointInterector4iMB(VirtualCurveIntersector &prim);

#if defined (__AVX__)
    void AddVirtualCurvePointInterector8i(VirtualCurveIntersector &prim);
    void AddVirtualCurvePointInterector8v(VirtualCurveIntersector &prim);
    void AddVirtualCurvePointInterector8iMB(VirtualCurveIntersector &prim);
#endif
  }
}
