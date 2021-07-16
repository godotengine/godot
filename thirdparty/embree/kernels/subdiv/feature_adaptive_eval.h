// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "patch.h"

namespace embree
{
  namespace isa
  {
    template<typename Vertex, typename Vertex_t = Vertex>
      struct FeatureAdaptiveEval
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
        
      public:
        
        FeatureAdaptiveEval (const HalfEdge* edge, const char* vertices, size_t stride, const float u, const float v, 
                             Vertex* P, Vertex* dPdu, Vertex* dPdv, Vertex* ddPdudu, Vertex* ddPdvdv, Vertex* ddPdudv)
        : P(P), dPdu(dPdu), dPdv(dPdv), ddPdudu(ddPdudu), ddPdvdv(ddPdvdv), ddPdudv(ddPdudv)
        {
          switch (edge->patch_type) {
          case HalfEdge::BILINEAR_PATCH: BilinearPatch(edge,vertices,stride).eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,1.0f); break;
          case HalfEdge::REGULAR_QUAD_PATCH: RegularPatchT(edge,vertices,stride).eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,1.0f); break;
#if PATCH_USE_GREGORY == 2
          case HalfEdge::IRREGULAR_QUAD_PATCH: GregoryPatch(edge,vertices,stride).eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,1.0f); break;
#endif
          default: {
            GeneralCatmullClarkPatch patch(edge,vertices,stride);
            eval(patch,Vec2f(u,v),0);
            break;
          }
          }
        }

        FeatureAdaptiveEval (CatmullClarkPatch& patch, const float u, const float v, float dscale, size_t depth, 
                             Vertex* P, Vertex* dPdu, Vertex* dPdv, Vertex* ddPdudu, Vertex* ddPdvdv, Vertex* ddPdudv)
        : P(P), dPdu(dPdu), dPdv(dPdv), ddPdudu(ddPdudu), ddPdvdv(ddPdvdv), ddPdudv(ddPdudv)
        {
          eval(patch,Vec2f(u,v),dscale,depth);
        }
        
        void eval_general_quad(const GeneralCatmullClarkPatch& patch, array_t<CatmullClarkPatch,GeneralCatmullClarkPatch::SIZE>& patches, const Vec2f& uv, size_t depth)
        {
          float u = uv.x, v = uv.y;
          if (v < 0.5f) {
            if (u < 0.5f) {
#if PATCH_USE_GREGORY == 2
              BezierCurve borders[2]; patch.getLimitBorder(borders,0);
              BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
              BezierCurve border2l,border2r; borders[1].subdivide(border2l,border2r);
              eval(patches[0],Vec2f(2.0f*u,2.0f*v),2.0f,depth+1, &border0l, nullptr, nullptr, &border2r);
#else
              eval(patches[0],Vec2f(2.0f*u,2.0f*v),2.0f,depth+1);
#endif
              if (dPdu && dPdv) {
                const Vertex dpdx = *dPdu, dpdy = *dPdv;
                *dPdu = dpdx; *dPdv = dpdy;
              }
            }
            else {
#if PATCH_USE_GREGORY == 2
              BezierCurve borders[2]; patch.getLimitBorder(borders,1);
              BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
              BezierCurve border2l,border2r; borders[1].subdivide(border2l,border2r);
              eval(patches[1],Vec2f(2.0f*v,2.0f-2.0f*u),2.0f,depth+1, &border0l, nullptr, nullptr, &border2r);
#else
              eval(patches[1],Vec2f(2.0f*v,2.0f-2.0f*u),2.0f,depth+1);
#endif
              if (dPdu && dPdv) {
                const Vertex dpdx = *dPdu, dpdy = *dPdv;
                *dPdu = -dpdy; *dPdv = dpdx;
              }
            }
          } else {
            if (u > 0.5f) {
#if PATCH_USE_GREGORY == 2
              BezierCurve borders[2]; patch.getLimitBorder(borders,2);
              BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
              BezierCurve border2l,border2r; borders[1].subdivide(border2l,border2r);
              eval(patches[2],Vec2f(2.0f-2.0f*u,2.0f-2.0f*v),2.0f,depth+1, &border0l, nullptr, nullptr, &border2r);
#else
              eval(patches[2],Vec2f(2.0f-2.0f*u,2.0f-2.0f*v),2.0f,depth+1);
#endif
              if (dPdu && dPdv) {
                const Vertex dpdx = *dPdu, dpdy = *dPdv;
                *dPdu = -dpdx; *dPdv = -dpdy;
              }
            }
            else {
#if PATCH_USE_GREGORY == 2
              BezierCurve borders[2]; patch.getLimitBorder(borders,3);
              BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
              BezierCurve border2l,border2r; borders[1].subdivide(border2l,border2r);
              eval(patches[3],Vec2f(2.0f-2.0f*v,2.0f*u),2.0f,depth+1, &border0l, nullptr, nullptr, &border2r);
#else
              eval(patches[3],Vec2f(2.0f-2.0f*v,2.0f*u),2.0f,depth+1);
#endif
              if (dPdu && dPdv) {
                const Vertex dpdx = *dPdu, dpdy = *dPdv;
                *dPdu = dpdy; *dPdv = -dpdx;
              }
            }
          }
        }

        __forceinline bool final(const CatmullClarkPatch& patch, const typename CatmullClarkRing::Type type, size_t depth) 
        {
          const int max_eval_depth = (type & CatmullClarkRing::TYPE_CREASES) ? PATCH_MAX_EVAL_DEPTH_CREASE : PATCH_MAX_EVAL_DEPTH_IRREGULAR;
//#if PATCH_MIN_RESOLUTION
//          return patch.isFinalResolution(PATCH_MIN_RESOLUTION) || depth>=(size_t)max_eval_depth;
//#else
          return depth>=(size_t)max_eval_depth;
//#endif
        }
        
        void eval(CatmullClarkPatch& patch, Vec2f uv, float dscale, size_t depth, 
                  BezierCurve* border0 = nullptr, BezierCurve* border1 = nullptr, BezierCurve* border2 = nullptr, BezierCurve* border3 = nullptr)
        {
          while (true) 
          {
            typename CatmullClarkPatch::Type ty = patch.type();

            if (unlikely(final(patch,ty,depth)))
            {
              if (ty & CatmullClarkRing::TYPE_REGULAR) { 
                RegularPatch(patch,border0,border1,border2,border3).eval(uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale); 
                PATCH_DEBUG_SUBDIVISION(234423,c,c,-1);
                return;
              } else {
                IrregularFillPatch(patch,border0,border1,border2,border3).eval(uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale); 
                PATCH_DEBUG_SUBDIVISION(34534,c,-1,c);
                return;
              }
            }
            else if (ty & CatmullClarkRing::TYPE_REGULAR_CREASES) { 
              assert(depth > 0); 
              RegularPatch(patch,border0,border1,border2,border3).eval(uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale); 
              PATCH_DEBUG_SUBDIVISION(43524,c,c,-1);
              return;
            }
#if PATCH_USE_GREGORY == 2
            else if (ty & CatmullClarkRing::TYPE_GREGORY_CREASES) { 
              assert(depth > 0); 
              GregoryPatch(patch,border0,border1,border2,border3).eval(uv.x,uv.y,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale); 
              PATCH_DEBUG_SUBDIVISION(23498,c,-1,c);
              return;
            }
#endif
            else
            {
              array_t<CatmullClarkPatch,4> patches; 
              patch.subdivide(patches); // FIXME: only have to generate one of the patches
              
              const float u = uv.x, v = uv.y;
              if (v < 0.5f) {
                if (u < 0.5f) { patch = patches[0]; uv = Vec2f(2.0f*u,2.0f*v); dscale *= 2.0f; }
                else          { patch = patches[1]; uv = Vec2f(2.0f*u-1.0f,2.0f*v); dscale *= 2.0f; }
              } else {
                if (u > 0.5f) { patch = patches[2]; uv = Vec2f(2.0f*u-1.0f,2.0f*v-1.0f); dscale *= 2.0f; }
                else          { patch = patches[3]; uv = Vec2f(2.0f*u,2.0f*v-1.0f); dscale *= 2.0f; }
              }
              depth++;
            }
          }
        }
        
        void eval(const GeneralCatmullClarkPatch& patch, const Vec2f& uv, const size_t depth) 
        {  
          /* convert into standard quad patch if possible */
          if (likely(patch.isQuadPatch())) 
          {
            CatmullClarkPatch qpatch; patch.init(qpatch);
            return eval(qpatch,uv,1.0f,depth); 
          }
          
          /* subdivide patch */
          unsigned N;
          array_t<CatmullClarkPatch,GeneralCatmullClarkPatch::SIZE> patches; 
          patch.subdivide(patches,N); // FIXME: only have to generate one of the patches
          
          /* parametrization for quads */
          if (N == 4) 
            eval_general_quad(patch,patches,uv,depth);
          
          /* parametrization for arbitrary polygons */
          else 
          {
            const unsigned l = (unsigned) floor(0.5f*uv.x); const float u = 2.0f*frac(0.5f*uv.x)-0.5f; 
            const unsigned h = (unsigned) floor(0.5f*uv.y); const float v = 2.0f*frac(0.5f*uv.y)-0.5f; 
            const unsigned i = 4*h+l; assert(i<N);
            if (i >= N) return;

#if PATCH_USE_GREGORY == 2
            BezierCurve borders[2]; patch.getLimitBorder(borders,i);
            BezierCurve border0l,border0r; borders[0].subdivide(border0l,border0r);
            BezierCurve border2l,border2r; borders[1].subdivide(border2l,border2r);
            eval(patches[i],Vec2f(u,v),1.0f,depth+1, &border0l, nullptr, nullptr, &border2r);
#else
            eval(patches[i],Vec2f(u,v),1.0f,depth+1);
#endif
          }
        }
        
      private:
        Vertex* const P;
        Vertex* const dPdu;
        Vertex* const dPdv;
        Vertex* const ddPdudu;
        Vertex* const ddPdvdv;
        Vertex* const ddPdudv;
      };
  }
}
