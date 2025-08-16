// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../geometry/primitive.h"
#include "bspline_patch.h"
#include "bezier_patch.h"
#include "gregory_patch.h"
#include "gregory_patch_dense.h"
#include "tessellation.h"
#include "tessellation_cache.h"
#include "gridrange.h"
#include "patch_eval_grid.h"
#include "feature_adaptive_eval_grid.h"
#include "../common/scene_subdiv_mesh.h"

namespace embree
{
  struct __aligned(64) SubdivPatch1Base
  {
  public:

    enum Type {
      INVALID_PATCH          = 0,
      BSPLINE_PATCH          = 1,  
      BEZIER_PATCH           = 2,  
      GREGORY_PATCH          = 3,
      EVAL_PATCH             = 5,
      BILINEAR_PATCH         = 6,
    };

    enum Flags {
      TRANSITION_PATCH       = 16, 
    };

    /*! Default constructor. */
    __forceinline SubdivPatch1Base () {}

    SubdivPatch1Base (const unsigned int gID,
                      const unsigned int pID,
                      const unsigned int subPatch,
                      const SubdivMesh *const mesh,
                      const size_t time,
                      const Vec2f uv[4],
                      const float edge_level[4],
                      const int subdiv[4],
                      const int simd_width);

    __forceinline bool needsStitching() const {
      return flags & TRANSITION_PATCH;      
    }

    __forceinline Vec2f getUV(const size_t i) const {
      return Vec2f((float)u[i],(float)v[i]) * (8.0f/0x10000);
    }

    static void computeEdgeLevels(const float edge_level[4], const int subdiv[4], float level[4]);
    static Vec2i computeGridSize(const float level[4]);
    bool updateEdgeLevels(const float edge_level[4], const int subdiv[4], const SubdivMesh *const mesh, const int simd_width);

  public:

    __forceinline size_t getGridBytes() const {
      const size_t grid_size_xyzuv = (grid_size_simd_blocks * VSIZEX) * 4;
      return 64*((grid_size_xyzuv+15) / 16);
    }

    __forceinline void write_lock()     { mtx.lock();   }
    __forceinline void write_unlock()   { mtx.unlock(); }
    __forceinline bool try_write_lock() { return mtx.try_lock(); }
    //__forceinline bool try_read_lock()  { return mtx.try_read_lock(); }

    __forceinline void resetRootRef() {
      //assert( mtx.hasInitialState() );
      root_ref = SharedLazyTessellationCache::Tag();
    }

    __forceinline SharedLazyTessellationCache::CacheEntry& entry() {
      return (SharedLazyTessellationCache::CacheEntry&) root_ref;
    }

  public:    
    __forceinline unsigned int geomID() const  {
      return geom;
    } 

    __forceinline unsigned int primID() const  {
      return prim;
    } 

  public:
    SharedLazyTessellationCache::Tag root_ref;
    SpinLock mtx;

    unsigned short u[4];                        //!< 16bit discretized u,v coordinates
    unsigned short v[4];
    float level[4];

    unsigned char flags;
    unsigned char type;
    unsigned short grid_u_res;
    unsigned int geom;                          //!< geometry ID of the subdivision mesh this patch belongs to
    unsigned int prim;                          //!< primitive ID of this subdivision patch
    unsigned short grid_v_res;

    unsigned short grid_size_simd_blocks;
    unsigned int time_;

    struct PatchHalfEdge {
      const HalfEdge* edge;
      unsigned subPatch;
    };

    Vec3fa patch_v[4][4];

    const HalfEdge *edge() const { return ((PatchHalfEdge*)patch_v)->edge; }
    unsigned time() const { return time_; }
    unsigned subPatch() const { return ((PatchHalfEdge*)patch_v)->subPatch; }

    void set_edge(const HalfEdge *h) const { ((PatchHalfEdge*)patch_v)->edge = h; }
    void set_subPatch(const unsigned s) const { ((PatchHalfEdge*)patch_v)->subPatch = s; }
  };

  namespace isa
  {
    Vec3fa patchEval(const SubdivPatch1Base& patch, const float uu, const float vv);
    Vec3fa patchNormal(const SubdivPatch1Base& patch, const float uu, const float vv);
    
    template<typename simdf>
      Vec3<simdf> patchEval(const SubdivPatch1Base& patch, const simdf& uu, const simdf& vv); 

    template<typename simdf>
      Vec3<simdf> patchNormal(const SubdivPatch1Base& patch, const simdf& uu, const simdf& vv); 
   

    /* eval grid over patch and stich edges when required */      
    void evalGrid(const SubdivPatch1Base& patch,
                  const unsigned x0, const unsigned x1,
                  const unsigned y0, const unsigned y1,
                  const unsigned swidth, const unsigned sheight,
                  float *__restrict__ const grid_x,
                  float *__restrict__ const grid_y,
                  float *__restrict__ const grid_z,
                  float *__restrict__ const grid_u,
                  float *__restrict__ const grid_v,
                  const SubdivMesh* const geom);

    /* eval grid over patch and stich edges when required */      
    BBox3fa evalGridBounds(const SubdivPatch1Base& patch,
                           const unsigned x0, const unsigned x1,
                           const unsigned y0, const unsigned y1,
                           const unsigned swidth, const unsigned sheight,
                           const SubdivMesh* const geom);
  }
}
