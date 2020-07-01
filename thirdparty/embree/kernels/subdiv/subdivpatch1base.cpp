// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "subdivpatch1base.h"

namespace embree
{
  SubdivPatch1Base::SubdivPatch1Base (const unsigned int gID,
                                      const unsigned int pID,
                                      const unsigned int subPatch,
                                      const SubdivMesh *const mesh,
                                      const size_t time,
                                      const Vec2f uv[4],
                                      const float edge_level[4],
                                      const int subdiv[4],
                                      const int simd_width)
  : flags(0), type(INVALID_PATCH), geom(gID), prim(pID), time_(unsigned(time))
  {
    static_assert(sizeof(SubdivPatch1Base) == 5 * 64, "SubdivPatch1Base has wrong size");

    const HalfEdge* edge = mesh->getHalfEdge(0,pID);

    if (edge->patch_type == HalfEdge::BILINEAR_PATCH)
    {
      type = BILINEAR_PATCH;
      new (patch_v) BilinearPatch3fa(edge,mesh->getVertexBuffer(time));
    }
    else if (edge->patch_type == HalfEdge::REGULAR_QUAD_PATCH) 
    {
#if PATCH_USE_BEZIER_PATCH 
      type = BEZIER_PATCH;
      new (patch_v) BezierPatch3fa(BSplinePatch3fa(CatmullClarkPatch3fa(edge,mesh->getVertexBuffer(time))));
#else
      type = BSPLINE_PATCH;
      new (patch_v) BSplinePatch3fa(CatmullClarkPatch3fa(edge,mesh->getVertexBuffer(time))); // FIXME: init BSpline directly from half edge structure
#endif      
    }
#if PATCH_USE_GREGORY == 2
    else if (edge->patch_type == HalfEdge::IRREGULAR_QUAD_PATCH) 
    {
      type = GREGORY_PATCH;
      new (patch_v) DenseGregoryPatch3fa(GregoryPatch3fa(CatmullClarkPatch3fa(edge,mesh->getVertexBuffer(time))));
    }
#endif
    else
    {
      type = EVAL_PATCH;
      set_edge(mesh->getHalfEdge(0,pID));
      set_subPatch(subPatch);
    }

    for (size_t i=0; i<4; i++) {
      u[i] = (unsigned short)clamp(uv[i].x * (0x10000/8.0f), 0.0f, float(0xFFFF));
      v[i] = (unsigned short)clamp(uv[i].y * (0x10000/8.0f), 0.0f, float(0xFFFF));
    }

    updateEdgeLevels(edge_level,subdiv,mesh,simd_width);
  }

  void SubdivPatch1Base::computeEdgeLevels(const float edge_level[4], const int subdiv[4], float level[4])
  {
    /* init discrete edge tessellation levels and grid resolution */
    assert( edge_level[0] >= 0.0f );
    assert( edge_level[1] >= 0.0f );
    assert( edge_level[2] >= 0.0f );
    assert( edge_level[3] >= 0.0f );

    level[0] = max(ceilf(adjustTessellationLevel(edge_level[0],subdiv[0])),1.0f);
    level[1] = max(ceilf(adjustTessellationLevel(edge_level[1],subdiv[1])),1.0f);
    level[2] = max(ceilf(adjustTessellationLevel(edge_level[2],subdiv[2])),1.0f);
    level[3] = max(ceilf(adjustTessellationLevel(edge_level[3],subdiv[3])),1.0f);
  }

  Vec2i SubdivPatch1Base::computeGridSize(const float level[4])
  {
    return Vec2i((int)max(level[0],level[2])+1,
                 (int)max(level[1],level[3])+1);
  }
  
  bool SubdivPatch1Base::updateEdgeLevels(const float edge_level[4], const int subdiv[4], const SubdivMesh *const mesh, const int simd_width)
  {
    /* calculate edge levels */
    float new_level[4];
    computeEdgeLevels(edge_level,subdiv,new_level);

    /* calculate if tessellation pattern changed */
    bool grid_changed = false;
    for (size_t i=0; i<4; i++) {
      grid_changed |= (int)new_level[i] != (int)level[i]; 
      level[i] = new_level[i];
    }

    /* compute grid resolution */
    Vec2i res = computeGridSize(level);
    grid_u_res = res.x; grid_v_res = res.y;
    grid_size_simd_blocks = ((grid_u_res*grid_v_res+simd_width-1)&(-simd_width)) / simd_width;

    /* need stiching? */
    flags &= ~TRANSITION_PATCH;
    const int int_edge_points0 = (int)level[0] + 1;
    const int int_edge_points1 = (int)level[1] + 1;
    const int int_edge_points2 = (int)level[2] + 1;
    const int int_edge_points3 = (int)level[3] + 1;
    if (int_edge_points0 < (int)grid_u_res ||
	int_edge_points2 < (int)grid_u_res ||
	int_edge_points1 < (int)grid_v_res ||
	int_edge_points3 < (int)grid_v_res) {
      flags |= TRANSITION_PATCH;
    }

    return grid_changed;
  }
}
