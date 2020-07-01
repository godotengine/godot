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
#include "../common/ray.h"
#include "triangle_intersector_pluecker.h"

namespace embree
{
  namespace isa
  {
    template<int K>
      struct MapUV0
    {
      const float* const grid_uv;
      size_t ofs00, ofs01, ofs10, ofs11;
      
      __forceinline MapUV0(const float* const grid_uv, size_t ofs00, size_t ofs01, size_t ofs10, size_t ofs11)
        : grid_uv(grid_uv), ofs00(ofs00), ofs01(ofs01), ofs10(ofs10), ofs11(ofs11) {}
      
      __forceinline void operator() (vfloat<K>& u, vfloat<K>& v) const {
        const vfloat<K> uv00(grid_uv[ofs00]);
        const vfloat<K> uv01(grid_uv[ofs01]);
        const vfloat<K> uv10(grid_uv[ofs10]);
        const vfloat<K> uv11(grid_uv[ofs11]);
        const Vec2vf<K> uv0 = GridSOA::decodeUV(uv00);
        const Vec2vf<K> uv1 = GridSOA::decodeUV(uv01);
        const Vec2vf<K> uv2 = GridSOA::decodeUV(uv10);
        const Vec2vf<K> uv = madd(u,uv1,madd(v,uv2,(1.0f-u-v)*uv0));
        u = uv[0]; v = uv[1];
      }
    };
    
    template<int K>
      struct MapUV1
    {
      const float* const grid_uv;
      size_t ofs00, ofs01, ofs10, ofs11;
      
      __forceinline MapUV1(const float* const grid_uv, size_t ofs00, size_t ofs01, size_t ofs10, size_t ofs11)
        : grid_uv(grid_uv), ofs00(ofs00), ofs01(ofs01), ofs10(ofs10), ofs11(ofs11) {}
      
      __forceinline void operator() (vfloat<K>& u, vfloat<K>& v) const {
        const vfloat<K> uv00(grid_uv[ofs00]);
        const vfloat<K> uv01(grid_uv[ofs01]);
        const vfloat<K> uv10(grid_uv[ofs10]);
        const vfloat<K> uv11(grid_uv[ofs11]);
        const Vec2vf<K> uv0 = GridSOA::decodeUV(uv10);
        const Vec2vf<K> uv1 = GridSOA::decodeUV(uv01);
        const Vec2vf<K> uv2 = GridSOA::decodeUV(uv11);
        const Vec2vf<K> uv = madd(u,uv1,madd(v,uv2,(1.0f-u-v)*uv0));
        u = uv[0]; v = uv[1];
      }
    };
    
    template<int K>
      class GridSOAIntersectorK
    {
    public:
      typedef void Primitive;

      class Precalculations
      {
#if defined(__AVX__)
        static const int M = 8;
#else
        static const int M = 4;
#endif

      public:
        __forceinline Precalculations (const vbool<K>& valid, const RayK<K>& ray)
          : grid(nullptr), intersector(valid,ray) {}

      public:
        GridSOA* grid;
        PlueckerIntersectorK<M,K> intersector; // FIXME: use quad intersector
      };

      /*! Intersect a ray with the primitive. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        const size_t dim_offset    = pre.grid->dim_offset;
        const size_t line_offset   = pre.grid->width;
        const float* const grid_x  = pre.grid->decodeLeaf(0,prim);
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;

        const size_t max_x = pre.grid->width  == 2 ? 1 : 2;
        const size_t max_y = pre.grid->height == 2 ? 1 : 2;
        for (size_t y=0; y<max_y; y++)        
        {
          for (size_t x=0; x<max_x; x++)
          {
            const size_t ofs00 = (y+0)*line_offset+(x+0);
            const size_t ofs01 = (y+0)*line_offset+(x+1);
            const size_t ofs10 = (y+1)*line_offset+(x+0);
            const size_t ofs11 = (y+1)*line_offset+(x+1);
            const Vec3vf<K> p00(grid_x[ofs00],grid_y[ofs00],grid_z[ofs00]);
            const Vec3vf<K> p01(grid_x[ofs01],grid_y[ofs01],grid_z[ofs01]);
            const Vec3vf<K> p10(grid_x[ofs10],grid_y[ofs10],grid_z[ofs10]);
            const Vec3vf<K> p11(grid_x[ofs11],grid_y[ofs11],grid_z[ofs11]);

            pre.intersector.intersectK(valid_i,ray,p00,p01,p10,MapUV0<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),IntersectKEpilogMU<1,K,true>(ray,context,pre.grid->geomID(),pre.grid->primID()));
            pre.intersector.intersectK(valid_i,ray,p10,p01,p11,MapUV1<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),IntersectKEpilogMU<1,K,true>(ray,context,pre.grid->geomID(),pre.grid->primID()));
          }
        }
      }

      /*! Test if the ray is occluded by the primitive */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        const size_t dim_offset    = pre.grid->dim_offset;
        const size_t line_offset   = pre.grid->width;
        const float* const grid_x  = pre.grid->decodeLeaf(0,prim);
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;

        vbool<K> valid = valid_i;
        const size_t max_x = pre.grid->width  == 2 ? 1 : 2;
        const size_t max_y = pre.grid->height == 2 ? 1 : 2;
        for (size_t y=0; y<max_y; y++)        
        {
          for (size_t x=0; x<max_x; x++)
          {
            const size_t ofs00 = (y+0)*line_offset+(x+0);
            const size_t ofs01 = (y+0)*line_offset+(x+1);
            const size_t ofs10 = (y+1)*line_offset+(x+0);
            const size_t ofs11 = (y+1)*line_offset+(x+1);
            const Vec3vf<K> p00(grid_x[ofs00],grid_y[ofs00],grid_z[ofs00]);
            const Vec3vf<K> p01(grid_x[ofs01],grid_y[ofs01],grid_z[ofs01]);
            const Vec3vf<K> p10(grid_x[ofs10],grid_y[ofs10],grid_z[ofs10]);
            const Vec3vf<K> p11(grid_x[ofs11],grid_y[ofs11],grid_z[ofs11]);

            pre.intersector.intersectK(valid,ray,p00,p01,p10,MapUV0<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),OccludedKEpilogMU<1,K,true>(valid,ray,context,pre.grid->geomID(),pre.grid->primID()));
            if (none(valid)) break;
            pre.intersector.intersectK(valid,ray,p10,p01,p11,MapUV1<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),OccludedKEpilogMU<1,K,true>(valid,ray,context,pre.grid->geomID(),pre.grid->primID()));
            if (none(valid)) break;
          }
        }
        return !valid;
      }

      template<typename Loader>
        static __forceinline void intersect(RayHitK<K>& ray, size_t k,
                                            IntersectContext* context,
                                            const float* const grid_x,
                                            const size_t line_offset,
                                            const size_t lines,
                                            Precalculations& pre)
      {
        typedef typename Loader::vfloat vfloat;
        const size_t dim_offset    = pre.grid->dim_offset;
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;
        Vec3<vfloat> v0, v1, v2; Loader::gather(grid_x,grid_y,grid_z,line_offset,lines,v0,v1,v2);
        pre.intersector.intersect(ray,k,v0,v1,v2,GridSOA::MapUV<Loader>(grid_uv,line_offset,lines),Intersect1KEpilogMU<Loader::M,K,true>(ray,k,context,pre.grid->geomID(),pre.grid->primID()));
      };

      template<typename Loader>
        static __forceinline bool occluded(RayK<K>& ray, size_t k,
                                           IntersectContext* context,
                                           const float* const grid_x,
                                           const size_t line_offset,
                                           const size_t lines,
                                           Precalculations& pre)
      {
        typedef typename Loader::vfloat vfloat;
        const size_t dim_offset    = pre.grid->dim_offset;
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;
        Vec3<vfloat> v0, v1, v2; Loader::gather(grid_x,grid_y,grid_z,line_offset,lines,v0,v1,v2);
        return pre.intersector.intersect(ray,k,v0,v1,v2,GridSOA::MapUV<Loader>(grid_uv,line_offset,lines),Occluded1KEpilogMU<Loader::M,K,true>(ray,k,context,pre.grid->geomID(),pre.grid->primID()));
      }

      /*! Intersect a ray with the primitive. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        const size_t line_offset   = pre.grid->width;
        const size_t lines         = pre.grid->height;
        const float* const grid_x  = pre.grid->decodeLeaf(0,prim);
#if defined(__AVX__)
        intersect<GridSOA::Gather3x3>( ray, k, context, grid_x, line_offset, lines, pre);
#else
        intersect<GridSOA::Gather2x3>(ray, k, context, grid_x            , line_offset, lines, pre);
        if (likely(lines > 2))
          intersect<GridSOA::Gather2x3>(ray, k, context, grid_x+line_offset, line_offset, lines, pre);
#endif
      }

      /*! Test if the ray is occluded by the primitive */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        const size_t line_offset   = pre.grid->width;
        const size_t lines         = pre.grid->height;
        const float* const grid_x  = pre.grid->decodeLeaf(0,prim);

#if defined(__AVX__)
        return occluded<GridSOA::Gather3x3>( ray, k, context, grid_x, line_offset, lines, pre);
#else
        if (occluded<GridSOA::Gather2x3>(ray, k, context, grid_x            , line_offset, lines, pre)) return true;
        if (likely(lines > 2))
          if (occluded<GridSOA::Gather2x3>(ray, k, context, grid_x+line_offset, line_offset, lines, pre)) return true;
#endif
        return false;
      }
    };

    template<int K>
    class GridSOAMBIntersectorK
    {
    public:
      typedef void Primitive;
      typedef typename GridSOAIntersectorK<K>::Precalculations Precalculations;

      /*! Intersect a ray with the primitive. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        vfloat<K> vftime;
        vint<K> vitime = getTimeSegment(ray.time(), vfloat<K>((float)(pre.grid->time_steps-1)), vftime);

        vbool<K> valid1 = valid_i;
        while (any(valid1)) {
          const size_t j = bsf(movemask(valid1));
          const int itime = vitime[j];
          const vbool<K> valid2 = valid1 & (itime == vitime);
          valid1 = valid1 & !valid2;
          intersect(valid2,pre,ray,vftime,itime,context,prim,lazy_node);
        }
      }

      /*! Intersect a ray with the primitive. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, const vfloat<K>& ftime, int itime, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        const size_t grid_offset   = pre.grid->gridBytes >> 2;
        const size_t dim_offset    = pre.grid->dim_offset;
        const size_t line_offset   = pre.grid->width;
        const float* const grid_x  = pre.grid->decodeLeaf(itime,prim);
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;

        const size_t max_x = pre.grid->width  == 2 ? 1 : 2;
        const size_t max_y = pre.grid->height == 2 ? 1 : 2;
        for (size_t y=0; y<max_y; y++)        
        {
          for (size_t x=0; x<max_x; x++)
          {
            size_t ofs00 = (y+0)*line_offset+(x+0);
            size_t ofs01 = (y+0)*line_offset+(x+1);
            size_t ofs10 = (y+1)*line_offset+(x+0);
            size_t ofs11 = (y+1)*line_offset+(x+1);
            const Vec3vf<K> a00(grid_x[ofs00],grid_y[ofs00],grid_z[ofs00]);
            const Vec3vf<K> a01(grid_x[ofs01],grid_y[ofs01],grid_z[ofs01]);
            const Vec3vf<K> a10(grid_x[ofs10],grid_y[ofs10],grid_z[ofs10]);
            const Vec3vf<K> a11(grid_x[ofs11],grid_y[ofs11],grid_z[ofs11]);
            ofs00 += grid_offset;
            ofs01 += grid_offset;
            ofs10 += grid_offset;
            ofs11 += grid_offset;
            const Vec3vf<K> b00(grid_x[ofs00],grid_y[ofs00],grid_z[ofs00]);
            const Vec3vf<K> b01(grid_x[ofs01],grid_y[ofs01],grid_z[ofs01]);
            const Vec3vf<K> b10(grid_x[ofs10],grid_y[ofs10],grid_z[ofs10]);
            const Vec3vf<K> b11(grid_x[ofs11],grid_y[ofs11],grid_z[ofs11]);
            const Vec3vf<K> p00 = lerp(a00,b00,ftime);
            const Vec3vf<K> p01 = lerp(a01,b01,ftime);
            const Vec3vf<K> p10 = lerp(a10,b10,ftime);
            const Vec3vf<K> p11 = lerp(a11,b11,ftime);

            pre.intersector.intersectK(valid_i,ray,p00,p01,p10,MapUV0<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),IntersectKEpilogMU<1,K,true>(ray,context,pre.grid->geomID(),pre.grid->primID()));
            pre.intersector.intersectK(valid_i,ray,p10,p01,p11,MapUV1<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),IntersectKEpilogMU<1,K,true>(ray,context,pre.grid->geomID(),pre.grid->primID()));
          }
        }
      }

      /*! Test if the ray is occluded by the primitive */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        vfloat<K> vftime;
        vint<K> vitime = getTimeSegment(ray.time(), vfloat<K>((float)(pre.grid->time_steps-1)), vftime);

        vbool<K> valid_o = valid_i;
        vbool<K> valid1 = valid_i;
        while (any(valid1)) {
          const int j = int(bsf(movemask(valid1)));
          const int itime = vitime[j];
          const vbool<K> valid2 = valid1 & (itime == vitime);
          valid1 = valid1 & !valid2;
          valid_o &= !valid2 | occluded(valid2,pre,ray,vftime,itime,context,prim,lazy_node);
        }
        return !valid_o;
      }

      /*! Test if the ray is occluded by the primitive */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, const vfloat<K>& ftime, int itime, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        const size_t grid_offset   = pre.grid->gridBytes >> 2;
        const size_t dim_offset    = pre.grid->dim_offset;
        const size_t line_offset   = pre.grid->width;
        const float* const grid_x  = pre.grid->decodeLeaf(itime,prim);
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;

        vbool<K> valid = valid_i;
        const size_t max_x = pre.grid->width  == 2 ? 1 : 2;
        const size_t max_y = pre.grid->height == 2 ? 1 : 2;
        for (size_t y=0; y<max_y; y++)        
        {
          for (size_t x=0; x<max_x; x++)
          {
            size_t ofs00 = (y+0)*line_offset+(x+0);
            size_t ofs01 = (y+0)*line_offset+(x+1);
            size_t ofs10 = (y+1)*line_offset+(x+0);
            size_t ofs11 = (y+1)*line_offset+(x+1);
            const Vec3vf<K> a00(grid_x[ofs00],grid_y[ofs00],grid_z[ofs00]);
            const Vec3vf<K> a01(grid_x[ofs01],grid_y[ofs01],grid_z[ofs01]);
            const Vec3vf<K> a10(grid_x[ofs10],grid_y[ofs10],grid_z[ofs10]);
            const Vec3vf<K> a11(grid_x[ofs11],grid_y[ofs11],grid_z[ofs11]);
            ofs00 += grid_offset;
            ofs01 += grid_offset;
            ofs10 += grid_offset;
            ofs11 += grid_offset;
            const Vec3vf<K> b00(grid_x[ofs00],grid_y[ofs00],grid_z[ofs00]);
            const Vec3vf<K> b01(grid_x[ofs01],grid_y[ofs01],grid_z[ofs01]);
            const Vec3vf<K> b10(grid_x[ofs10],grid_y[ofs10],grid_z[ofs10]);
            const Vec3vf<K> b11(grid_x[ofs11],grid_y[ofs11],grid_z[ofs11]);
            const Vec3vf<K> p00 = lerp(a00,b00,ftime);
            const Vec3vf<K> p01 = lerp(a01,b01,ftime);
            const Vec3vf<K> p10 = lerp(a10,b10,ftime);
            const Vec3vf<K> p11 = lerp(a11,b11,ftime);

            pre.intersector.intersectK(valid,ray,p00,p01,p10,MapUV0<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),OccludedKEpilogMU<1,K,true>(valid,ray,context,pre.grid->geomID(),pre.grid->primID()));
            if (none(valid)) break;
            pre.intersector.intersectK(valid,ray,p10,p01,p11,MapUV1<K>(grid_uv,ofs00,ofs01,ofs10,ofs11),OccludedKEpilogMU<1,K,true>(valid,ray,context,pre.grid->geomID(),pre.grid->primID()));
            if (none(valid)) break;
          }
        }
        return valid;
      }

      template<typename Loader>
        static __forceinline void intersect(RayHitK<K>& ray, size_t k,
                                            const float ftime,
                                            IntersectContext* context,
                                            const float* const grid_x,
                                            const size_t line_offset,
                                            const size_t lines,
                                            Precalculations& pre)
      {
        typedef typename Loader::vfloat vfloat;
        const size_t grid_offset   = pre.grid->gridBytes >> 2;
        const size_t dim_offset    = pre.grid->dim_offset;
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;

        Vec3<vfloat> a0, a1, a2;
        Loader::gather(grid_x,grid_y,grid_z,line_offset,lines,a0,a1,a2);

        Vec3<vfloat> b0, b1, b2;
        Loader::gather(grid_x+grid_offset,grid_y+grid_offset,grid_z+grid_offset,line_offset,lines,b0,b1,b2);

        Vec3<vfloat> v0 = lerp(a0,b0,vfloat(ftime));
        Vec3<vfloat> v1 = lerp(a1,b1,vfloat(ftime));
        Vec3<vfloat> v2 = lerp(a2,b2,vfloat(ftime));

        pre.intersector.intersect(ray,k,v0,v1,v2,GridSOA::MapUV<Loader>(grid_uv,line_offset,lines),Intersect1KEpilogMU<Loader::M,K,true>(ray,k,context,pre.grid->geomID(),pre.grid->primID()));
      };

      template<typename Loader>
        static __forceinline bool occluded(RayK<K>& ray, size_t k,
                                           const float ftime,
                                           IntersectContext* context,
                                           const float* const grid_x,
                                           const size_t line_offset,
                                           const size_t lines,
                                           Precalculations& pre)
      {
        typedef typename Loader::vfloat vfloat;
        const size_t grid_offset   = pre.grid->gridBytes >> 2;
        const size_t dim_offset    = pre.grid->dim_offset;
        const float* const grid_y  = grid_x + 1 * dim_offset;
        const float* const grid_z  = grid_x + 2 * dim_offset;
        const float* const grid_uv = grid_x + 3 * dim_offset;

        Vec3<vfloat> a0, a1, a2;
        Loader::gather(grid_x,grid_y,grid_z,line_offset,lines,a0,a1,a2);

        Vec3<vfloat> b0, b1, b2;
        Loader::gather(grid_x+grid_offset,grid_y+grid_offset,grid_z+grid_offset,line_offset,lines,b0,b1,b2);

        Vec3<vfloat> v0 = lerp(a0,b0,vfloat(ftime));
        Vec3<vfloat> v1 = lerp(a1,b1,vfloat(ftime));
        Vec3<vfloat> v2 = lerp(a2,b2,vfloat(ftime));

        return pre.intersector.intersect(ray,k,v0,v1,v2,GridSOA::MapUV<Loader>(grid_uv,line_offset,lines),Occluded1KEpilogMU<Loader::M,K,true>(ray,k,context,pre.grid->geomID(),pre.grid->primID()));
      }

      /*! Intersect a ray with the primitive. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      { 
        float ftime;
        int itime = getTimeSegment(ray.time()[k], float(pre.grid->time_steps-1), ftime);

        const size_t line_offset   = pre.grid->width;
        const size_t lines         = pre.grid->height;
        const float* const grid_x  = pre.grid->decodeLeaf(itime,prim);

#if defined(__AVX__)
        intersect<GridSOA::Gather3x3>( ray, k, ftime, context, grid_x, line_offset, lines, pre);
#else
        intersect<GridSOA::Gather2x3>(ray, k, ftime, context, grid_x, line_offset, lines, pre);
        if (likely(lines > 2))
          intersect<GridSOA::Gather2x3>(ray, k, ftime, context, grid_x+line_offset, line_offset, lines, pre);
#endif
      }

      /*! Test if the ray is occluded by the primitive */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t& lazy_node)
      {
        float ftime;
        int itime = getTimeSegment(ray.time()[k], float(pre.grid->time_steps-1), ftime);

        const size_t line_offset   = pre.grid->width;
        const size_t lines         = pre.grid->height;
        const float* const grid_x  = pre.grid->decodeLeaf(itime,prim);

#if defined(__AVX__)
        return occluded<GridSOA::Gather3x3>( ray, k, ftime, context, grid_x, line_offset, lines, pre);
#else
        if (occluded<GridSOA::Gather2x3>(ray, k, ftime, context, grid_x, line_offset, lines, pre)) return true;
        if (likely(lines > 2))
          if (occluded<GridSOA::Gather2x3>(ray, k, ftime, context, grid_x+line_offset, line_offset, lines, pre)) return true;
#endif
        return false;
      }
    };
  }
}
