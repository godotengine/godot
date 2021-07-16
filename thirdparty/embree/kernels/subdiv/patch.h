// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "catmullclark_patch.h"
#include "bilinear_patch.h"
#include "bspline_patch.h"
#include "bezier_patch.h"
#include "gregory_patch.h"
#include "tessellation_cache.h"

#if 1
#define PATCH_DEBUG_SUBDIVISION(ptr,x,y,z)
#else
#define PATCH_DEBUG_SUBDIVISION(ptr,x,y,z)            \
  {                                                   \
    size_t hex = (size_t)ptr;                          \
    for (size_t i=0; i<4; i++) hex = hex ^ (hex >> 8);  \
    const float c = (float)(((hex >> 0) ^ (hex >> 4) ^ (hex >> 8) ^ (hex >> 12) ^ (hex >> 16))&0xf)/15.0f; \
    if (P) *P = Vertex(0.5f+0.5f*x,0.5f+0.5f*y,0.5f+0.5f*z,0.0f);         \
    }               
#endif

#define PATCH_MAX_CACHE_DEPTH 2
//#define PATCH_MIN_RESOLUTION 1     // FIXME: not yet completely implemented
#define PATCH_MAX_EVAL_DEPTH_IRREGULAR 10     // maximum evaluation depth at irregular vertices (has to be larger or equal than PATCH_MAX_CACHE_DEPTH)
#define PATCH_MAX_EVAL_DEPTH_CREASE 10       // maximum evaluation depth at crease features (has to be larger or equal than PATCH_MAX_CACHE_DEPTH)
#define PATCH_USE_GREGORY 1        // 0 = no gregory, 1 = fill, 2 = as early as possible

#if PATCH_USE_GREGORY==2
#define PATCH_USE_BEZIER_PATCH 1   // enable use of bezier instead of b-spline patches
#else
#define PATCH_USE_BEZIER_PATCH 0   // enable use of bezier instead of b-spline patches
#endif

#if PATCH_USE_BEZIER_PATCH
#  define RegularPatch  BezierPatch
#  define RegularPatchT BezierPatchT<Vertex,Vertex_t>
#else
#  define RegularPatch  BSplinePatch
#  define RegularPatchT BSplinePatchT<Vertex,Vertex_t>
#endif

#if PATCH_USE_GREGORY
#define IrregularFillPatch GregoryPatch
#define IrregularFillPatchT GregoryPatchT<Vertex,Vertex_t>
#else
#define IrregularFillPatch BilinearPatch
#define IrregularFillPatchT BilinearPatchT<Vertex,Vertex_t>
#endif

namespace embree
{
  template<typename Vertex, typename Vertex_t = Vertex>
    struct __aligned(64) PatchT
    {
    public:
    
    typedef GeneralCatmullClarkPatchT<Vertex,Vertex_t> GeneralCatmullClarkPatch;
    typedef CatmullClarkPatchT<Vertex,Vertex_t> CatmullClarkPatch;
    typedef CatmullClark1RingT<Vertex,Vertex_t> CatmullClarkRing;
    typedef BezierCurveT<Vertex> BezierCurve;
    
    enum Type {
      INVALID_PATCH = 0,
      BILINEAR_PATCH = 1,
      BSPLINE_PATCH = 2,  
      BEZIER_PATCH = 3,  
      GREGORY_PATCH = 4,
      SUBDIVIDED_GENERAL_PATCH = 7,
      SUBDIVIDED_QUAD_PATCH = 8,
      EVAL_PATCH = 9,
    };
    
    struct Ref
    {
      __forceinline Ref(void* p = nullptr) 
        : ptr((size_t)p) {}

      __forceinline operator bool() const { return ptr != 0; }
      __forceinline operator size_t() const { return ptr; }

      __forceinline Ref (Type ty, void* in) 
        : ptr(((size_t)in)+ty) { assert((((size_t)in) & 0xF) == 0); }

      __forceinline Type  type  () const { return (Type)(ptr & 0xF); }
      __forceinline void* object() const { return (void*) (ptr & ~0xF); }

      size_t ptr;
    };

    struct EvalPatch 
    {
      /* creates EvalPatch from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const CatmullClarkPatch& patch) 
      {
        size_t ofs = 0, bytes = patch.bytes();
        void* ptr = alloc(bytes);
        patch.serialize(ptr,ofs);
        assert(ofs == bytes);
        return Ref(EVAL_PATCH, ptr);
      }
    };

    struct BilinearPatch 
    {
      /* creates BilinearPatch from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const CatmullClarkPatch& patch,
                                   const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) {
        return Ref(BILINEAR_PATCH, new (alloc(sizeof(BilinearPatch))) BilinearPatch(patch));
      }

      __forceinline BilinearPatch (const CatmullClarkPatch& patch) 
        : patch(patch) {}

      /* creates BilinearPatch from 4 vertices */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const HalfEdge* edge, const char* vertices, size_t stride) {
        return Ref(BILINEAR_PATCH, new (alloc(sizeof(BilinearPatch))) BilinearPatch(edge,vertices,stride));
      }
      
      __forceinline BilinearPatch (const HalfEdge* edge, const char* vertices, size_t stride) 
        : patch(edge,vertices,stride) {}
      
    public:
      BilinearPatchT<Vertex,Vertex_t> patch;
    };
    
    struct BSplinePatch 
    {
      /* creates BSplinePatch from a half edge */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const HalfEdge* edge, const char* vertices, size_t stride) {
        return Ref(BSPLINE_PATCH, new (alloc(sizeof(BSplinePatch))) BSplinePatch(edge,vertices,stride));
      }
      
      __forceinline BSplinePatch (const HalfEdge* edge, const char* vertices, size_t stride) 
        : patch(edge,vertices,stride) {}
      
      /* creates BSplinePatch from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const CatmullClarkPatch& patch,
                                   const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) {
        return Ref(BSPLINE_PATCH, new (alloc(sizeof(BSplinePatch))) BSplinePatch(patch,border0,border1,border2,border3));
      }
      
      __forceinline BSplinePatch (const CatmullClarkPatch& patch, const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) 
        : patch(patch,border0,border1,border2,border3) {}
      
    public:
      BSplinePatchT<Vertex,Vertex_t> patch;
    };

    struct BezierPatch
    {
      /* creates BezierPatch from a half edge */
      template<typename Allocator>
        __noinline static Ref create(const Allocator& alloc, const HalfEdge* edge, const char* vertices, size_t stride) {
        return Ref(BEZIER_PATCH, new (alloc(sizeof(BezierPatch))) BezierPatch(edge,vertices,stride));
      }
      
      __forceinline BezierPatch (const HalfEdge* edge, const char* vertices, size_t stride) 
        : patch(edge,vertices,stride) {}
      
      /* creates Bezier from a CatmullClarkPatch */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const CatmullClarkPatch& patch,
                                   const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) {
        return Ref(BEZIER_PATCH, new (alloc(sizeof(BezierPatch))) BezierPatch(patch,border0,border1,border2,border3));
      }
      
      __forceinline BezierPatch (const CatmullClarkPatch& patch, const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) 
        : patch(patch,border0,border1,border2,border3) {}
      
    public:
      BezierPatchT<Vertex,Vertex_t> patch;
    };
    
    struct GregoryPatch
    {
      /* creates GregoryPatch from half edge */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const HalfEdge* edge, const char* vertices, size_t stride) {
        return Ref(GREGORY_PATCH, new (alloc(sizeof(GregoryPatch))) GregoryPatch(edge,vertices,stride));
      }
      
      __forceinline GregoryPatch (const HalfEdge* edge, const char* vertices, size_t stride) 
        : patch(CatmullClarkPatch(edge,vertices,stride)) {}
       
      /* creates GregoryPatch from CatmullClarkPatch */
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const CatmullClarkPatch& patch,
                                   const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) {
        return Ref(GREGORY_PATCH, new (alloc(sizeof(GregoryPatch))) GregoryPatch(patch,border0,border1,border2,border3));
      }
      
      __forceinline GregoryPatch (const CatmullClarkPatch& patch, const BezierCurve* border0, const BezierCurve* border1, const BezierCurve* border2, const BezierCurve* border3) 
        : patch(patch,border0,border1,border2,border3) {}
      
    public:
      GregoryPatchT<Vertex,Vertex_t> patch;
    };
    
    struct SubdividedQuadPatch
    {
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, Ref children[4]) {
        return Ref(SUBDIVIDED_QUAD_PATCH, new (alloc(sizeof(SubdividedQuadPatch))) SubdividedQuadPatch(children));
      }
      
      __forceinline SubdividedQuadPatch(Ref children[4]) {
        for (size_t i=0; i<4; i++) child[i] = children[i];
      }
      
    public:
      Ref child[4];
    };
    
    struct SubdividedGeneralPatch
    {
      template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, Ref* children, const unsigned N) {
        return Ref(SUBDIVIDED_GENERAL_PATCH, new (alloc(sizeof(SubdividedGeneralPatch))) SubdividedGeneralPatch(children,N));
      }
      
      __forceinline SubdividedGeneralPatch(Ref* children, const unsigned N) : N(N) {
        for (unsigned i=0; i<N; i++) child[i] = children[i];
      }
      
      unsigned N;
      Ref child[MAX_PATCH_VALENCE];
    };
    
    /*! Default constructor. */
    __forceinline PatchT () {}
    
    template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, const HalfEdge* edge, const char* vertices, size_t stride)
    {
      if (PATCH_MAX_CACHE_DEPTH == 0) 
        return nullptr;

      Ref child(0);
      switch (edge->patch_type) {
      case HalfEdge::BILINEAR_PATCH:       child = BilinearPatch::create(alloc,edge,vertices,stride); break; 
      case HalfEdge::REGULAR_QUAD_PATCH:   child = RegularPatch::create(alloc,edge,vertices,stride); break;
#if PATCH_USE_GREGORY == 2
      case HalfEdge::IRREGULAR_QUAD_PATCH: child = GregoryPatch::create(alloc,edge,vertices,stride); break;
#endif
      default: {
        GeneralCatmullClarkPatch patch(edge,vertices,stride);
        child = PatchT::create(alloc,patch,edge,vertices,stride,0);
      }
      }
      return child;
    }

    template<typename Allocator>
    __noinline static Ref create(const Allocator& alloc, GeneralCatmullClarkPatch& patch, const HalfEdge* edge, const char* vertices, size_t stride, size_t depth)
    {  
      /* convert into standard quad patch if possible */
      if (likely(patch.isQuadPatch())) 
      {
        CatmullClarkPatch qpatch; patch.init(qpatch);
        return PatchT::create(alloc,qpatch,edge,vertices,stride,depth);
      }
   
      /* do only cache up to some depth */
      if (depth >= PATCH_MAX_CACHE_DEPTH)
        return nullptr;
         
      /* subdivide patch */
      unsigned N;
      array_t<CatmullClarkPatch,GeneralCatmullClarkPatch::SIZE> patches; 
      patch.subdivide(patches,N);
      
      if (N == 4) 
      {
        Ref child[4];
#if PATCH_USE_GREGORY == 2
        BezierCurve borders[GeneralCatmullClarkPatch::SIZE]; patch.getLimitBorder(borders);
        BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
        BezierCurve border1l,border1r; borders[1].subdivide(border1l,border1r);
        BezierCurve border2l,border2r; borders[2].subdivide(border2l,border2r);
        BezierCurve border3l,border3r; borders[3].subdivide(border3l,border3r);
        GeneralCatmullClarkPatch::fix_quad_ring_order(patches);
        child[0] = PatchT::create(alloc,patches[0],edge,vertices,stride,depth+1,&border0l,nullptr,nullptr,&border3r);
        child[1] = PatchT::create(alloc,patches[1],edge,vertices,stride,depth+1,&border0r,&border1l,nullptr,nullptr);
        child[2] = PatchT::create(alloc,patches[2],edge,vertices,stride,depth+1,nullptr,&border1r,&border2l,nullptr);
        child[3] = PatchT::create(alloc,patches[3],edge,vertices,stride,depth+1,nullptr,nullptr,&border2r,&border3l);
#else
        GeneralCatmullClarkPatch::fix_quad_ring_order(patches);
        for (size_t i=0; i<4; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);
#endif
        return SubdividedQuadPatch::create(alloc,child);
      }
      else 
      {
        assert(N<MAX_PATCH_VALENCE);
        Ref child[MAX_PATCH_VALENCE];
        
#if PATCH_USE_GREGORY == 2
        BezierCurve borders[GeneralCatmullClarkPatch::SIZE]; 
        patch.getLimitBorder(borders);

        for (size_t i0=0; i0<N; i0++) {
          const size_t i2 = i0==0 ? N-1 : i0-1; 
          BezierCurve border0l,border0r; borders[i0].subdivide(border0l,border0r);
          BezierCurve border2l,border2r; borders[i2].subdivide(border2l,border2r);
          child[i0] = PatchT::create(alloc,patches[i0],edge,vertices,stride,depth+1, &border0l, nullptr, nullptr, &border2r);
        }
#else
        for (size_t i=0; i<N; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);
#endif
        return SubdividedGeneralPatch::create(alloc,child,N);
      }
      
      return nullptr;
    }

    static __forceinline bool final(const CatmullClarkPatch& patch, const typename CatmullClarkRing::Type type, size_t depth) 
    {
      const size_t max_eval_depth = (type & CatmullClarkRing::TYPE_CREASES) ? PATCH_MAX_EVAL_DEPTH_CREASE : PATCH_MAX_EVAL_DEPTH_IRREGULAR;
//#if PATCH_MIN_RESOLUTION
//      return patch.isFinalResolution(PATCH_MIN_RESOLUTION) || depth>=max_eval_depth;
//#else
      return depth>=max_eval_depth;
//#endif
    }

    template<typename Allocator>
      __noinline static Ref create(const Allocator& alloc, CatmullClarkPatch& patch, const HalfEdge* edge, const char* vertices, size_t stride, size_t depth,
                                   const BezierCurve* border0 = nullptr, const BezierCurve* border1 = nullptr, const BezierCurve* border2 = nullptr, const BezierCurve* border3 = nullptr)
    {
      const typename CatmullClarkPatch::Type ty = patch.type();
      if (unlikely(final(patch,ty,depth))) {
        if (ty & CatmullClarkRing::TYPE_REGULAR) return RegularPatch::create(alloc,patch,border0,border1,border2,border3); 
        else                                     return IrregularFillPatch::create(alloc,patch,border0,border1,border2,border3); 
      }
      else if (ty & CatmullClarkRing::TYPE_REGULAR_CREASES) { 
        assert(depth > 0); return RegularPatch::create(alloc,patch,border0,border1,border2,border3); 
      }
#if PATCH_USE_GREGORY == 2
      else if (ty & CatmullClarkRing::TYPE_GREGORY_CREASES) { 
        assert(depth > 0); return GregoryPatch::create(alloc,patch,border0,border1,border2,border3); 
      }
#endif
      else if (depth >= PATCH_MAX_CACHE_DEPTH) {
        return EvalPatch::create(alloc,patch); 
      }
      
      else 
      {
        Ref child[4];
        array_t<CatmullClarkPatch,4> patches; 
        patch.subdivide(patches);
        
        for (size_t i=0; i<4; i++)
          child[i] = PatchT::create(alloc,patches[i],edge,vertices,stride,depth+1);
        return SubdividedQuadPatch::create(alloc,child);
      }
    }
  };

  typedef PatchT<Vec3fa,Vec3fa_t> Patch3fa;
}
