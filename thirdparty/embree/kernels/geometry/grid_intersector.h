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

#pragma once

#include "grid_soa.h"
#include "grid_soa_intersector1.h"
#include "grid_soa_intersector_packet.h"
#include "../common/ray.h"

namespace embree
{
  namespace isa
  {
    template<typename T>
      class SubdivPatch1Precalculations : public T
    { 
    public:
      __forceinline SubdivPatch1Precalculations (const Ray& ray, const void* ptr)
        : T(ray,ptr) {}
    };

    template<int K, typename T>
      class SubdivPatch1PrecalculationsK : public T
    { 
    public:
      __forceinline SubdivPatch1PrecalculationsK (const vbool<K>& valid, RayK<K>& ray)
        : T(valid,ray) {}
    };

    class Grid1Intersector1
    {
    public:
      typedef GridSOA Primitive;
      typedef Grid1Precalculations<GridSOAIntersector1::Precalculations> Precalculations;

      /*! Intersect a ray with the primitive. */
      static __forceinline void intersect(Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive* prim, size_t ty, size_t& lazy_node) 
      {
        GridSOAIntersector1::intersect(pre,ray,context,prim,lazy_node);
      }
      static __forceinline void intersect(Precalculations& pre, RayHit& ray, IntersectContext* context, size_t ty0, const Primitive* prim, size_t ty, size_t& lazy_node) {
        intersect(pre,ray,context,prim,ty,lazy_node);
      }
      
      /*! Test if the ray is occluded by the primitive */
      static __forceinline bool occluded(Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive* prim, size_t ty, size_t& lazy_node)
      {
        GridSOAIntersector1::occluded(pre,ray,context,prim,lazy_node);
      }
      static __forceinline bool occluded(Precalculations& pre, Ray& ray, IntersectContext* context, size_t ty0, const Primitive* prim, size_t ty, size_t& lazy_node) {
        return occluded(pre,ray,context,prim,ty,lazy_node);
      }
    };

    template <int K>
      struct GridIntersectorK
    {
      typedef GridSOA Primitive;
      typedef SubdivPatch1PrecalculationsK<K,typename GridSOAIntersectorK<K>::Precalculations> Precalculations;
      
      
      static __forceinline void intersect(const vbool<K>& valid, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t ty, size_t& lazy_node)
      {
        GridSOAIntersectorK<K>::intersect(valid,pre,ray,context,prim,lazy_node);
      }
      
      static __forceinline vbool<K> occluded(const vbool<K>& valid, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t ty, size_t& lazy_node)
      {
        GridSOAIntersectorK<K>::occluded(valid,pre,ray,context,prim,lazy_node);
      }
      
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t ty, size_t& lazy_node)
      {
        GridSOAIntersectorK<K>::intersect(pre,ray,k,context,prim,lazy_node);
      }
      
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t ty, size_t& lazy_node)
      {
        GridSOAIntersectorK<K>::occluded(pre,ray,k,context,prim,lazy_node);
      }
    };

    typedef Grid1IntersectorK<4>  SubdivPatch1Intersector4;
    typedef Grid1IntersectorK<8>  SubdivPatch1Intersector8;
    typedef Grid1IntersectorK<16> SubdivPatch1Intersector16;

  }
}
