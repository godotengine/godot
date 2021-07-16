// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "patch.h"
#include "feature_adaptive_eval.h"

namespace embree
{
  namespace isa
  {
    template<typename Vertex, typename Vertex_t = Vertex>
      struct PatchEval
      {
      public:
        
        typedef PatchT<Vertex,Vertex_t> Patch;
        typedef typename Patch::Ref Ref;
        typedef CatmullClarkPatchT<Vertex,Vertex_t> CatmullClarkPatch;
        
        PatchEval (SharedLazyTessellationCache::CacheEntry& entry, size_t commitCounter, 
                   const HalfEdge* edge, const char* vertices, size_t stride, const float u, const float v, 
                   Vertex* P, Vertex* dPdu, Vertex* dPdv, Vertex* ddPdudu, Vertex* ddPdvdv, Vertex* ddPdudv)
        : P(P), dPdu(dPdu), dPdv(dPdv), ddPdudu(ddPdudu), ddPdvdv(ddPdvdv), ddPdudv(ddPdudv)
        {
          /* conservative time for the very first allocation */
          auto time = SharedLazyTessellationCache::sharedLazyTessellationCache.getTime(commitCounter);

          Ref patch = SharedLazyTessellationCache::lookup(entry,commitCounter,[&] () {
              auto alloc = [&](size_t bytes) { return SharedLazyTessellationCache::malloc(bytes); };
              return Patch::create(alloc,edge,vertices,stride);
            },true);

          auto curTime = SharedLazyTessellationCache::sharedLazyTessellationCache.getTime(commitCounter);
          const bool allAllocationsValid = SharedLazyTessellationCache::validTime(time,curTime);

          if (patch && allAllocationsValid &&  eval(patch,u,v,1.0f,0)) {
            SharedLazyTessellationCache::unlock();
            return;
          }
          SharedLazyTessellationCache::unlock();
          FeatureAdaptiveEval<Vertex,Vertex_t>(edge,vertices,stride,u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv);
          PATCH_DEBUG_SUBDIVISION(edge,c,-1,-1);
        }
        
        __forceinline bool eval_quad(const typename Patch::SubdividedQuadPatch* This, const float u, const float v, const float dscale, const size_t depth)
        {
          if (v < 0.5f) {
            if (u < 0.5f) return eval(This->child[0],2.0f*u,2.0f*v,2.0f*dscale,depth+1);
            else          return eval(This->child[1],2.0f*u-1.0f,2.0f*v,2.0f*dscale,depth+1);
          } else {
            if (u > 0.5f) return eval(This->child[2],2.0f*u-1.0f,2.0f*v-1.0f,2.0f*dscale,depth+1);
            else          return eval(This->child[3],2.0f*u,2.0f*v-1.0f,2.0f*dscale,depth+1);
          }
        }
        
        bool eval_general(const typename Patch::SubdividedGeneralPatch* This, const float U, const float V, const size_t depth)
        {
          const unsigned l = (unsigned) floor(0.5f*U); const float u = 2.0f*frac(0.5f*U)-0.5f; 
          const unsigned h = (unsigned) floor(0.5f*V); const float v = 2.0f*frac(0.5f*V)-0.5f; 
          const unsigned i = 4*h+l; assert(i<This->N);
          return eval(This->child[i],u,v,1.0f,depth+1);
        }
        
        bool eval(Ref This, const float& u, const float& v, const float dscale, const size_t depth) 
        {
          if (!This) return false;
          //PRINT(depth);
          //PRINT2(u,v);
          
          switch (This.type()) 
          {
          case Patch::BILINEAR_PATCH: {
            //PRINT("bilinear");
            ((typename Patch::BilinearPatch*)This.object())->patch.eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale); 
            PATCH_DEBUG_SUBDIVISION(This,-1,c,c);
            return true;
          }
          case Patch::BSPLINE_PATCH: {
            //PRINT("bspline");
            ((typename Patch::BSplinePatch*)This.object())->patch.eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale);
            PATCH_DEBUG_SUBDIVISION(This,-1,c,-1);
            return true;
          }
          case Patch::BEZIER_PATCH: {
            //PRINT("bezier");
            ((typename Patch::BezierPatch*)This.object())->patch.eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale);
            PATCH_DEBUG_SUBDIVISION(This,-1,c,-1);
            return true;
          }
          case Patch::GREGORY_PATCH: {
            //PRINT("gregory");
            ((typename Patch::GregoryPatch*)This.object())->patch.eval(u,v,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv,dscale); 
            PATCH_DEBUG_SUBDIVISION(This,-1,-1,c);
            return true;
          }
          case Patch::SUBDIVIDED_QUAD_PATCH: {
            //PRINT("subdivided quad");
            return eval_quad(((typename Patch::SubdividedQuadPatch*)This.object()),u,v,dscale,depth);
          }
          case Patch::SUBDIVIDED_GENERAL_PATCH: { 
            //PRINT("general_patch");
            assert(dscale == 1.0f); 
            return eval_general(((typename Patch::SubdividedGeneralPatch*)This.object()),u,v,depth); 
          }
          case Patch::EVAL_PATCH: { 
            //PRINT("eval_patch");
            CatmullClarkPatch patch; patch.deserialize(This.object());
            FeatureAdaptiveEval<Vertex,Vertex_t>(patch,u,v,dscale,depth,P,dPdu,dPdv,ddPdudu,ddPdvdv,ddPdudv);
            return true;
          }
          default: 
            assert(false); 
            return false;
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
  
