// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "../common/scene_subdiv_mesh.h"
#include "../bvh/bvh.h"
#include "../subdiv/tessellation.h"
#include "../subdiv/tessellation_cache.h"
#include "subdivpatch1.h"

namespace embree
{
  namespace isa
  {
    class GridSOA
    {
    public:

      /*! GridSOA constructor */
      GridSOA(const SubdivPatch1Base* patches, const unsigned time_steps,
              const unsigned x0, const unsigned x1, const unsigned y0, const unsigned y1, const unsigned swidth, const unsigned sheight,
              const SubdivMesh* const geom, const size_t totalBvhBytes, const size_t gridBytes, BBox3fa* bounds_o = nullptr);

      /*! Subgrid creation */
      template<typename Allocator>
        static GridSOA* create(const SubdivPatch1Base* patches, const unsigned time_steps,
                               unsigned x0, unsigned x1, unsigned y0, unsigned y1, 
                               const Scene* scene, Allocator& alloc, BBox3fa* bounds_o = nullptr)
      {
        const unsigned width = x1-x0+1;  
        const unsigned height = y1-y0+1; 
        const GridRange range(0,width-1,0,height-1);
        size_t bvhBytes = 0;
        if (time_steps == 1) 
          bvhBytes = getBVHBytes(range,sizeof(BVH4::AABBNode),0);
        else {
          bvhBytes = (time_steps-1)*getBVHBytes(range,sizeof(BVH4::AABBNodeMB),0);
          bvhBytes += getTemporalBVHBytes(make_range(0,int(time_steps-1)),sizeof(BVH4::AABBNodeMB4D));
        }
        const size_t gridBytes = 4*size_t(width)*size_t(height)*sizeof(float);  
        size_t rootBytes = time_steps*sizeof(BVH4::NodeRef);
#if !defined(__64BIT__)
        rootBytes += 4; // We read 2 elements behind the grid. As we store at least 8 root bytes after the grid we are fine in 64 bit mode. But in 32 bit mode we have to do additional padding.
#endif
        void* data = alloc(offsetof(GridSOA,data)+bvhBytes+time_steps*gridBytes+rootBytes);
        assert(data);
        return new (data) GridSOA(patches,time_steps,x0,x1,y0,y1,patches->grid_u_res,patches->grid_v_res,scene->get<SubdivMesh>(patches->geomID()),bvhBytes,gridBytes,bounds_o);
      }

      /*! Grid creation */
      template<typename Allocator>
        static GridSOA* create(const SubdivPatch1Base* const patches, const unsigned time_steps,
                               const Scene* scene, const Allocator& alloc, BBox3fa* bounds_o = nullptr) 
      {
        return create(patches,time_steps,0,patches->grid_u_res-1,0,patches->grid_v_res-1,scene,alloc,bounds_o);
      }

       /*! returns reference to root */
      __forceinline       BVH4::NodeRef& root(size_t t = 0)       { return (BVH4::NodeRef&)data[rootOffset + t*sizeof(BVH4::NodeRef)]; }
      __forceinline const BVH4::NodeRef& root(size_t t = 0) const { return (BVH4::NodeRef&)data[rootOffset + t*sizeof(BVH4::NodeRef)]; }

      /*! returns pointer to BVH array */
      __forceinline       char* bvhData()       { return &data[0]; }
      __forceinline const char* bvhData() const { return &data[0]; }

      /*! returns pointer to Grid array */
      __forceinline       float* gridData(size_t t = 0)       { return (float*) &data[gridOffset + t*gridBytes]; }
      __forceinline const float* gridData(size_t t = 0) const { return (float*) &data[gridOffset + t*gridBytes]; }
      
      __forceinline void* encodeLeaf(size_t u, size_t v) {
        return (void*) (16*(v * width + u + 1)); // +1 to not create empty leaf
      }
      __forceinline float* decodeLeaf(size_t t, const void* ptr) {
        return gridData(t) + (((size_t) (ptr) >> 4) - 1);
      }

      /*! returns the size of the BVH over the grid in bytes */
      static size_t getBVHBytes(const GridRange& range, const size_t nodeBytes, const size_t leafBytes);

      /*! returns the size of the temporal BVH over the time range BVHs */
      static size_t getTemporalBVHBytes(const range<int> time_range, const size_t nodeBytes);

      /*! calculates bounding box of grid range */
      __forceinline BBox3fa calculateBounds(size_t time, const GridRange& range) const
      {
        const float* const grid_array = gridData(time);
        const float* const grid_x_array = grid_array + 0 * dim_offset;
        const float* const grid_y_array = grid_array + 1 * dim_offset;
        const float* const grid_z_array = grid_array + 2 * dim_offset;
        
        /* compute the bounds just for the range! */
        BBox3fa bounds( empty );
        for (unsigned v = range.v_start; v<=range.v_end; v++) 
        {
          for (unsigned u = range.u_start; u<=range.u_end; u++)
          {
            const float x = grid_x_array[ v * width + u];
            const float y = grid_y_array[ v * width + u];
            const float z = grid_z_array[ v * width + u];
            bounds.extend( Vec3fa(x,y,z) );
          }
        }
        assert(is_finite(bounds));
        return bounds;
      }

      /*! Evaluates grid over patch and builds BVH4 tree over the grid. */
      std::pair<BVH4::NodeRef,BBox3fa> buildBVH(BBox3fa* bounds_o);
      
      /*! Create BVH4 tree over grid. */
      std::pair<BVH4::NodeRef,BBox3fa> buildBVH(const GridRange& range, size_t& allocator);

      /*! Evaluates grid over patch and builds MSMBlur BVH4 tree over the grid. */
      std::pair<BVH4::NodeRef,LBBox3fa> buildMSMBlurBVH(const range<int> time_range, BBox3fa* bounds_o);
      
      /*! Create MBlur BVH4 tree over grid. */
      std::pair<BVH4::NodeRef,LBBox3fa> buildMBlurBVH(size_t time, const GridRange& range, size_t& allocator);

      /*! Create MSMBlur BVH4 tree over grid. */
      std::pair<BVH4::NodeRef,LBBox3fa> buildMSMBlurBVH(const range<int> time_range, size_t& allocator, BBox3fa* bounds_o);

      template<typename Loader>
        struct MapUV
      {
        typedef typename Loader::vfloat vfloat;
        const float* const grid_uv;
        size_t line_offset;
        size_t lines;

        __forceinline MapUV(const float* const grid_uv, size_t line_offset, const size_t lines)
          : grid_uv(grid_uv), line_offset(line_offset), lines(lines) {}

        __forceinline void operator() (vfloat& u, vfloat& v, Vec3<vfloat>& Ng) const {
          const Vec3<vfloat> tri_v012_uv = Loader::gather(grid_uv,line_offset,lines);	
          const Vec2<vfloat> uv0 = GridSOA::decodeUV(tri_v012_uv[0]);
          const Vec2<vfloat> uv1 = GridSOA::decodeUV(tri_v012_uv[1]);
          const Vec2<vfloat> uv2 = GridSOA::decodeUV(tri_v012_uv[2]);        
          const Vec2<vfloat> uv = u * uv1 + v * uv2 + (1.0f-u-v) * uv0;        
          u = uv[0];v = uv[1]; 
        }
      };

      struct Gather2x3
      {
        enum { M = 4 };
        typedef vbool4 vbool;
        typedef vint4 vint;
        typedef vfloat4 vfloat;
        
        static __forceinline const Vec3vf4 gather(const float* const grid, const size_t line_offset, const size_t lines)
        {
          vfloat4 r0 = vfloat4::loadu(grid + 0*line_offset);
          vfloat4 r1 = vfloat4::loadu(grid + 1*line_offset); // this accesses 2 elements too much in case of 2x2 grid, but this is ok as we ensure enough padding after the grid
          if (unlikely(line_offset == 2))
          {
            r0 = shuffle<0,1,1,1>(r0);
            r1 = shuffle<0,1,1,1>(r1);
          }
          return Vec3vf4(unpacklo(r0,r1),       // r00, r10, r01, r11
                         shuffle<1,1,2,2>(r0),  // r01, r01, r02, r02
                         shuffle<0,1,1,2>(r1)); // r10, r11, r11, r12
        }

        static __forceinline void gather(const float* const grid_x, 
                                         const float* const grid_y, 
                                         const float* const grid_z, 
                                         const size_t line_offset,
                                         const size_t lines,
                                         Vec3vf4& v0_o,
                                         Vec3vf4& v1_o,
                                         Vec3vf4& v2_o)
        {
          const Vec3vf4 tri_v012_x = gather(grid_x,line_offset,lines);
          const Vec3vf4 tri_v012_y = gather(grid_y,line_offset,lines);
          const Vec3vf4 tri_v012_z = gather(grid_z,line_offset,lines);
          v0_o = Vec3vf4(tri_v012_x[0],tri_v012_y[0],tri_v012_z[0]);
          v1_o = Vec3vf4(tri_v012_x[1],tri_v012_y[1],tri_v012_z[1]);
          v2_o = Vec3vf4(tri_v012_x[2],tri_v012_y[2],tri_v012_z[2]);
        }
      };
      
#if defined (__AVX__)
      struct Gather3x3
      {
        enum { M = 8 };
        typedef vbool8 vbool;
        typedef vint8 vint;
        typedef vfloat8 vfloat;
        
        static __forceinline const Vec3vf8 gather(const float* const grid, const size_t line_offset, const size_t lines)
        {
          vfloat4 ra = vfloat4::loadu(grid + 0*line_offset);
          vfloat4 rb = vfloat4::loadu(grid + 1*line_offset); // this accesses 2 elements too much in case of 2x2 grid, but this is ok as we ensure enough padding after the grid
          vfloat4 rc;
          if (likely(lines > 2)) 
            rc = vfloat4::loadu(grid + 2*line_offset);
          else                   
            rc = rb;

          if (unlikely(line_offset == 2))
          {
            ra = shuffle<0,1,1,1>(ra);
            rb = shuffle<0,1,1,1>(rb);
            rc = shuffle<0,1,1,1>(rc);
          }
          
          const vfloat8 r0 = vfloat8(ra,rb);
          const vfloat8 r1 = vfloat8(rb,rc);
          return Vec3vf8(unpacklo(r0,r1),         // r00, r10, r01, r11, r10, r20, r11, r21
                         shuffle<1,1,2,2>(r0),    // r01, r01, r02, r02, r11, r11, r12, r12
                         shuffle<0,1,1,2>(r1));   // r10, r11, r11, r12, r20, r21, r21, r22
        }

        static __forceinline void gather(const float* const grid_x, 
                                         const float* const grid_y, 
                                         const float* const grid_z, 
                                         const size_t line_offset,
                                         const size_t lines,
                                         Vec3vf8& v0_o,
                                         Vec3vf8& v1_o,
                                         Vec3vf8& v2_o)
        {
          const Vec3vf8 tri_v012_x = gather(grid_x,line_offset,lines);
          const Vec3vf8 tri_v012_y = gather(grid_y,line_offset,lines);
          const Vec3vf8 tri_v012_z = gather(grid_z,line_offset,lines);
          v0_o = Vec3vf8(tri_v012_x[0],tri_v012_y[0],tri_v012_z[0]);
          v1_o = Vec3vf8(tri_v012_x[1],tri_v012_y[1],tri_v012_z[1]);
          v2_o = Vec3vf8(tri_v012_x[2],tri_v012_y[2],tri_v012_z[2]);
        }
      };
#endif

      template<typename vfloat>
      static __forceinline Vec2<vfloat> decodeUV(const vfloat& uv)
      {
        typedef typename vfloat::Int vint;
        const vint iu  = asInt(uv) & 0xffff;
        const vint iv  = srl(asInt(uv),16);
	const vfloat u = (vfloat)iu * vfloat(8.0f/0x10000);
	const vfloat v = (vfloat)iv * vfloat(8.0f/0x10000);
	return Vec2<vfloat>(u,v);
      }
      
      __forceinline unsigned int geomID() const  {
        return _geomID;
      } 
      
      __forceinline unsigned int primID() const  {
        return _primID;
      } 

    public:
      BVH4::NodeRef troot;
#if !defined(__64BIT__)
      unsigned align1;
#endif
      unsigned time_steps;
      unsigned width;

      unsigned height;
      unsigned dim_offset;
      unsigned _geomID;
      unsigned _primID;

      unsigned align2;
      unsigned gridOffset;
      unsigned gridBytes;
      unsigned rootOffset;

      char data[1];        //!< after the struct we first store the BVH, then the grid, and finally the roots
    };
  }
}
