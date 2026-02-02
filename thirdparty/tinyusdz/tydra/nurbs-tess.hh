// SPDX-License-Identifier: Apache 2.0
// Simple NURBS tesselation

#pragma once

#include "render-data.hh"

namespace tinyusdz {

namespace tydra {

struct Nurbs;

class NurbsTesselator
{
  bool tesselate(const Nurbs &nurbs, uint32_t u_divs, uint32_t v_divs, RenderMesh &dst );
};

} // namespace tydra

} // namespace tinyusdz
