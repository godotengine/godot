// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "patch.h"

namespace embree
{
  namespace isa
  {
    template<typename vbool, typename vint, typename vfloat, typename Vertex, typename Vertex_t = Vertex>
      struct FeatureAdaptiveEvalSimd
      {
      public:
        
        typedef PatchT<Vertex,Vertex_t> Patch;
        typedef typename Patch::Ref Ref;
        typedef GeneralCatmullClarkPatchT<Vertex,Vertex_t> GeneralCatmullClarkPatch;
        typedef CatmullClark1RingT<Vertex,Vertex_t> CatmullClarkRing;
        typedef CatmullClarkPatchT<Vertex,Vertex_t> CatmullClarkPatch;
        typedef BSplinePatchT<Vertex,Vertex_t> BSplinePatch;
        typedef BezierPatchT<Vertex,Vertex_t> BezierPatch;
        typedef GregoryPatchT<Vertex,Vertex_t> GregoryPatch;
        typedef BilinearPatchT<Vertex,Vertex_t> BilinearPatch;
        typedef BezierCurveT<Vertex> BezierCurve;

        FeatureAdaptiveEvalSimd (const HalfEdge* edge, const char* vertices, size_t stride, const vbool& valid, const vfloat& u, const vfloat& v, 
                                 float* P, float* dPdu, float* dPdv, float* ddPdudu, float* ddPdvdv, float* ddPdudv, const size_t dstride, const size_t N)
        : P(P), dPdu(dPdu), dPdv(dPdv), ddPdudu(ddPdudu), ddPdvdv(ddPdvdv), ddPdudv(ddPdudv), dstride(dstride), N(N)
        {
          switch (edge->patch_type) {
          case HalfEdge::BILINEAR_PATCH: BilinearPatch(edge,vertices,stride).eval(valid,u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,1.0f,dstride,N); break;
          case HalfEdge::REGULAR_QUAD_PATCH: RegularPatchT(edge,vertices,stride).eval(valid,u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,1.0f,dstride,N); break;
#if PATCH_USE_GREGORY == 2
          case HalfEdge::IRREGULAR_QUAD_PATCH: GregoryPatchT<Vertex,Vertex_t>(edge,vertices,stride).eval(valid,u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,1.0f,dstride,N); break;
#endif
          default: {
            GeneralCatmullClarkPatch patch(edge,vertices,stride);
            eval_direct(valid,patch,Vec2<vfloat>(u,v),0);
            break;
          }
          }
        }

        FeatureAdaptiveEvalSimd (const CatmullClarkPatch& patch, const vbool& valid, const vfloat& u, const vfloat& v, float dscale, size_t depth, 
                                 float* P, float* dPdu, float* dPdv, float* ddPdudu, float* ddPdvdv, float* ddPdudv, const size_t dstride, const size_t N)
        : P(P), dPdu(dPdu), dPdv(dPdv), ddPdudu(ddPdudu), ddPdvdv(ddPdvdv), ddPdudv(ddPdudv), dstride(dstride), N(N)
        {
          eval_direct(valid,patch,Vec2<vfloat>(u,v),dscale,depth);
        }

        template<size_t N>
        __forceinline void eval_quad_direct(const vbool& valid, array_t<CatmullClarkPatch,N>& patches, const Vec2<vfloat>& uv, float dscale, size_t depth)
        {
          const vfloat u = uv.x, v = uv.y;
          const vbool u0_mask = u < 0.5f, u1_mask = u >= 0.5f;
          const vbool v0_mask = v < 0.5f, v1_mask = v >= 0.5f;
          const vbool u0v0_mask = valid & u0_mask & v0_mask;
          const vbool u0v1_mask = valid & u0_mask & v1_mask;
          const vbool u1v0_mask = valid & u1_mask & v0_mask;
          const vbool u1v1_mask = valid & u1_mask & v1_mask;
          if (any(u0v0_mask)) eval_direct(u0v0_mask,patches[0],Vec2<vfloat>(2.0f*u,2.0f*v),2.0f*dscale,depth+1);
          if (any(u1v0_mask)) eval_direct(u1v0_mask,patches[1],Vec2<vfloat>(2.0f*u-1.0f,2.0f*v),2.0f*dscale,depth+1);
          if (any(u1v1_mask)) eval_direct(u1v1_mask,patches[2],Vec2<vfloat>(2.0f*u-1.0f,2.0f*v-1.0f),2.0f*dscale,depth+1);
          if (any(u0v1_mask)) eval_direct(u0v1_mask,patches[3],Vec2<vfloat>(2.0f*u,2.0f*v-1.0f),2.0f*dscale,depth+1);
        }
        
        template<size_t N>
        __forceinline void eval_general_quad_direct(const vbool& valid, const GeneralCatmullClarkPatch& patch, array_t<CatmullClarkPatch,N>& patches, const Vec2<vfloat>& uv, float dscale, size_t depth)
        {
#if PATCH_USE_GREGORY == 2
          BezierCurve borders[GeneralCatmullClarkPatch::SIZE]; patch.getLimitBorder(borders);
          BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
          BezierCurve border1l,border1r; borders[1].subdivide(border1l,border1r);
          BezierCurve border2l,border2r; borders[2].subdivide(border2l,border2r);
          BezierCurve border3l,border3r; borders[3].subdivide(border3l,border3r);
#endif
          GeneralCatmullClarkPatch::fix_quad_ring_order(patches);
          const vfloat u = uv.x, v = uv.y;
          const vbool u0_mask = u < 0.5f, u1_mask = u >= 0.5f;
          const vbool v0_mask = v < 0.5f, v1_mask = v >= 0.5f;
          const vbool u0v0_mask = valid & u0_mask & v0_mask;
          const vbool u0v1_mask = valid & u0_mask & v1_mask;
          const vbool u1v0_mask = valid & u1_mask & v0_mask;
          const vbool u1v1_mask = valid & u1_mask & v1_mask;
#if PATCH_USE_GREGORY == 2
          if (any(u0v0_mask)) eval_direct(u0v0_mask,patches[0],Vec2<vfloat>(2.0f*u,2.0f*v),2.0f*dscale,depth+1,&border0l,nullptr,nullptr,&border3r);
          if (any(u1v0_mask)) eval_direct(u1v0_mask,patches[1],Vec2<vfloat>(2.0f*u-1.0f,2.0f*v),2.0f*dscale,depth+1,&border0r,&border1l,nullptr,nullptr);
          if (any(u1v1_mask)) eval_direct(u1v1_mask,patches[2],Vec2<vfloat>(2.0f*u-1.0f,2.0f*v-1.0f),2.0f*dscale,depth+1,nullptr,&border1r,&border2l,nullptr);
          if (any(u0v1_mask)) eval_direct(u0v1_mask,patches[3],Vec2<vfloat>(2.0f*u,2.0f*v-1.0f),2.0f*dscale,depth+1,nullptr,nullptr,&border2r,&border3l);
#else
          if (any(u0v0_mask)) eval_direct(u0v0_mask,patches[0],Vec2<vfloat>(2.0f*u,2.0f*v),2.0f*dscale,depth+1);
          if (any(u1v0_mask)) eval_direct(u1v0_mask,patches[1],Vec2<vfloat>(2.0f*u-1.0f,2.0f*v),2.0f*dscale,depth+1);
          if (any(u1v1_mask)) eval_direct(u1v1_mask,patches[2],Vec2<vfloat>(2.0f*u-1.0f,2.0f*v-1.0f),2.0f*dscale,depth+1);
          if (any(u0v1_mask)) eval_direct(u0v1_mask,patches[3],Vec2<vfloat>(2.0f*u,2.0f*v-1.0f),2.0f*dscale,depth+1);
#endif
        }
        
        __forceinline bool final(const CatmullClarkPatch& patch, const typename CatmullClarkRing::Type type, size_t depth) 
        {
          const size_t max_eval_depth = (type & CatmullClarkRing::TYPE_CREASES) ? PATCH_MAX_EVAL_DEPTH_CREASE : PATCH_MAX_EVAL_DEPTH_IRREGULAR;
//#if PATCH_MIN_RESOLUTION
//          return patch.isFinalResolution(PATCH_MIN_RESOLUTION) || depth>=max_eval_depth;
//#else
          return depth>=max_eval_depth;
//#endif
        }

        void eval_direct(const vbool& valid, const CatmullClarkPatch& patch, const Vec2<vfloat>& uv, float dscale, size_t depth,
                         BezierCurve* border0 = nullptr, BezierCurve* border1 = nullptr, BezierCurve* border2 = nullptr, BezierCurve* border3 = nullptr)
        {
          typename CatmullClarkPatch::Type ty = patch.type();

          if (unlikely(final(patch,ty,depth)))
          {
            if (ty & CatmullClarkRing::TYPE_REGULAR) { 
              RegularPatch(patch,border0,border1,border2,border3).eval(valid,uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale,dstride,N);
            } else {
              IrregularFillPatch(patch,border0,border1,border2,border3).eval(valid,uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale,dstride,N);
            }
          }
          else if (ty & CatmullClarkRing::TYPE_REGULAR_CREASES) { 
            assert(depth > 0); RegularPatch(patch,border0,border1,border2,border3).eval(valid,uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale,dstride,N);
          }
#if PATCH_USE_GREGORY == 2
          else if (ty & CatmullClarkRing::TYPE_GREGORY_CREASES) { 
            assert(depth > 0); GregoryPatch(patch,border0,border1,border2,border3).eval(valid,uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale,dstride,N);
          }
#endif
          else
          {
            array_t<CatmullClarkPatch,4> patches; 
            patch.subdivide(patches); // FIXME: only have to generate one of the patches
            eval_quad_direct(valid,patches,uv,dscale,depth);
          }
        }  

        void eval_direct(const vbool& valid, const GeneralCatmullClarkPatch& patch, const Vec2<vfloat>& uv, const size_t depth) 
        {
          /* convert into standard quad patch if possible */
          if (likely(patch.isQuadPatch())) {
            CatmullClarkPatch qpatch; patch.init(qpatch);
            return eval_direct(valid,qpatch,uv,1.0f,depth);
          }
          
          /* subdivide patch */
          unsigned Nc;
          array_t<CatmullClarkPatch,GeneralCatmullClarkPatch::SIZE> patches; 
          patch.subdivide(patches,Nc); // FIXME: only have to generate one of the patches
          
          /* parametrization for quads */
          if (Nc == 4) 
            eval_general_quad_direct(valid,patch,patches,uv,1.0f,depth);
          
          /* parametrization for arbitrary polygons */
          else 
          {
            const vint l = (vint)floor(0.5f*uv.x); const vfloat u = 2.0f*frac(0.5f*uv.x)-0.5f; 
            const vint h = (vint)floor(0.5f*uv.y); const vfloat v = 2.0f*frac(0.5f*uv.y)-0.5f; 
            const vint i = (h<<2)+l; assert(all(valid,i<Nc));
            foreach_unique(valid,i,[&](const vbool& valid, const int i) {
#if PATCH_USE_GREGORY == 2
                BezierCurve borders[2]; patch.getLimitBorder(borders,i);
                BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
                BezierCurve border2l,border2r; borders[1].subdivide(border2l,border2r);
                eval_direct(valid,patches[i],Vec2<vfloat>(u,v),1.0f,depth+1, &border0l, nullptr, nullptr, &border2r);
#else
                eval_direct(valid,patches[i],Vec2<vfloat>(u,v),1.0f,depth+1);
#endif
              });
          }
        }

      private:
        float* const P;
        float* const dPdu;
        float* const dPdv;
        float* const ddPdudu;
        float* const ddPdvdv;
        float* const ddPdudv;
        const size_t dstride;
        const size_t N;
      };
  }
}
