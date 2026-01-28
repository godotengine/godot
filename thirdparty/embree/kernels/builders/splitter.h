// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(RTHWIF_STANDALONE)
#include "../common/scene.h"
#endif

#include "../builders/primref.h"

namespace embree
{
  namespace isa
  {
    template<size_t N>
    __forceinline void splitPolygon(const BBox3fa& bounds, 
                                    const size_t dim, 
                                    const float pos, 
                                    const Vec3fa (&v)[N+1],
                                    BBox3fa& left_o, 
                                    BBox3fa& right_o)
    {
      BBox3fa left = empty, right = empty;
      /* clip triangle to left and right box by processing all edges */
      for (size_t i=0; i<N; i++)
      {
        const Vec3fa &v0 = v[i]; 
        const Vec3fa &v1 = v[i+1]; 
        const float v0d = v0[dim];
        const float v1d = v1[dim];
        
        if (v0d <= pos) left. extend(v0); // this point is on left side
        if (v0d >= pos) right.extend(v0); // this point is on right side
        
        if ((v0d < pos && pos < v1d) || (v1d < pos && pos < v0d)) // the edge crosses the splitting location
        {
          assert((v1d-v0d) != 0.0f);
          const float inv_length = 1.0f/(v1d-v0d);
          const Vec3fa c = madd(Vec3fa((pos-v0d)*inv_length),v1-v0,v0);
          left.extend(c);
          right.extend(c);
        }
      }
      
      /* clip against current bounds */
      left_o  = intersect(left,bounds);
      right_o = intersect(right,bounds);
    }
    
    template<size_t N>
    __forceinline void splitPolygon(const BBox3fa& bounds, 
                                    const size_t dim, 
                                    const float pos, 
                                    const Vec3fa (&v)[N+1],
                                    const Vec3fa (&inv_length)[N],
                                    BBox3fa& left_o, 
                                    BBox3fa& right_o)
    {
      BBox3fa left = empty, right = empty;
      /* clip triangle to left and right box by processing all edges */
      for (size_t i=0; i<N; i++)
      {
        const Vec3fa &v0 = v[i]; 
        const Vec3fa &v1 = v[i+1]; 
        const float v0d = v0[dim];
        const float v1d = v1[dim];
        
        if (v0d <= pos) left. extend(v0); // this point is on left side
        if (v0d >= pos) right.extend(v0); // this point is on right side
        
        if ((v0d < pos && pos < v1d) || (v1d < pos && pos < v0d)) // the edge crosses the splitting location
        {
          assert((v1d-v0d) != 0.0f);
          const Vec3fa c = madd(Vec3fa((pos-v0d)*inv_length[i][dim]),v1-v0,v0);
          left.extend(c);
          right.extend(c);
        }
      }
      
      /* clip against current bounds */
      left_o  = intersect(left,bounds);
      right_o = intersect(right,bounds);
    }
    
    template<size_t N>
      __forceinline void splitPolygon(const PrimRef& prim, 
                                      const size_t dim, 
                                      const float pos, 
                                      const Vec3fa (&v)[N+1],
                                      PrimRef& left_o, 
                                      PrimRef& right_o)
    {
      BBox3fa left = empty, right = empty;
      for (size_t i=0; i<N; i++)
      {
        const Vec3fa &v0 = v[i]; 
        const Vec3fa &v1 = v[i+1]; 
        const float v0d = v0[dim];
        const float v1d = v1[dim];
        
        if (v0d <= pos) left. extend(v0); // this point is on left side
        if (v0d >= pos) right.extend(v0); // this point is on right side
        
        if ((v0d < pos && pos < v1d) || (v1d < pos && pos < v0d)) // the edge crosses the splitting location
        {
          assert((v1d-v0d) != 0.0f);
          const float inv_length = 1.0f/(v1d-v0d);
          const Vec3fa c = madd(Vec3fa((pos-v0d)*inv_length),v1-v0,v0);
          left.extend(c);
          right.extend(c);
        }
      }
      
      /* clip against current bounds */
      new (&left_o ) PrimRef(intersect(left ,prim.bounds()),prim.geomID(), prim.primID());
      new (&right_o) PrimRef(intersect(right,prim.bounds()),prim.geomID(), prim.primID());
    }

#if !defined(RTHWIF_STANDALONE)

    struct TriangleSplitter
    {
      __forceinline TriangleSplitter(const Scene* scene, const PrimRef& prim)
      {
        const unsigned int mask = 0xFFFFFFFF >> RESERVED_NUM_SPATIAL_SPLITS_GEOMID_BITS;
        const TriangleMesh* mesh = (const TriangleMesh*) scene->get(prim.geomID() & mask );  
        TriangleMesh::Triangle tri = mesh->triangle(prim.primID());
        v[0] = mesh->vertex(tri.v[0]);
        v[1] = mesh->vertex(tri.v[1]);
        v[2] = mesh->vertex(tri.v[2]);
        v[3] = mesh->vertex(tri.v[0]);
        inv_length[0] = Vec3fa(1.0f) / (v[1]-v[0]);
        inv_length[1] = Vec3fa(1.0f) / (v[2]-v[1]);
        inv_length[2] = Vec3fa(1.0f) / (v[0]-v[2]);
      }
      
      __forceinline void operator() (const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const {
        splitPolygon<3>(prim,dim,pos,v,left_o,right_o);
      }
      
      __forceinline void operator() (const BBox3fa& prim, const size_t dim, const float pos, BBox3fa& left_o, BBox3fa& right_o) const {
        splitPolygon<3>(prim,dim,pos,v,inv_length,left_o,right_o);
      }
      
    private:
      Vec3fa v[4];
      Vec3fa inv_length[3];
    };
    
    struct TriangleSplitterFactory
    {
      __forceinline TriangleSplitterFactory(const Scene* scene)
        : scene(scene) {}
      
      __forceinline TriangleSplitter operator() (const PrimRef& prim) const {
        return TriangleSplitter(scene,prim);
      }
      
    private:
      const Scene* scene;
    };
    
    struct QuadSplitter
    {
      __forceinline QuadSplitter(const Scene* scene, const PrimRef& prim)
      {
        const unsigned int mask = 0xFFFFFFFF >> RESERVED_NUM_SPATIAL_SPLITS_GEOMID_BITS;
        const QuadMesh* mesh = (const QuadMesh*) scene->get(prim.geomID() & mask );  
        QuadMesh::Quad quad = mesh->quad(prim.primID());
        v[0] = mesh->vertex(quad.v[1]);
        v[1] = mesh->vertex(quad.v[2]);
        v[2] = mesh->vertex(quad.v[3]);
        v[3] = mesh->vertex(quad.v[0]);
        v[4] = mesh->vertex(quad.v[1]);
        v[5] = mesh->vertex(quad.v[3]);
        inv_length[0] = Vec3fa(1.0f) / (v[1] - v[0]);
        inv_length[1] = Vec3fa(1.0f) / (v[2] - v[1]);
        inv_length[2] = Vec3fa(1.0f) / (v[3] - v[2]);
        inv_length[3] = Vec3fa(1.0f) / (v[4] - v[3]);
        inv_length[4] = Vec3fa(1.0f) / (v[5] - v[4]);
      }
      
      __forceinline void operator() (const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const {
        splitPolygon<5>(prim,dim,pos,v,left_o,right_o);
      }
      
      __forceinline void operator() (const BBox3fa& prim, const size_t dim, const float pos, BBox3fa& left_o, BBox3fa& right_o) const {
        splitPolygon<5>(prim,dim,pos,v,inv_length,left_o,right_o);
      }
      
    private:
      Vec3fa v[6];
      Vec3fa inv_length[5];
    };
    
    struct QuadSplitterFactory
    {
      __forceinline QuadSplitterFactory(const Scene* scene)
        : scene(scene) {}
      
      __forceinline QuadSplitter operator() (const PrimRef& prim) const {
        return QuadSplitter(scene,prim);
      }
      
    private:
      const Scene* scene;
    };


    struct DummySplitter
    {
      __forceinline DummySplitter(const Scene* scene, const PrimRef& prim)
      {
      }

      __forceinline void operator() (const PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const {
      }
      
      __forceinline void operator() (const BBox3fa& prim, const size_t dim, const float pos, BBox3fa& left_o, BBox3fa& right_o) const {
      }
      
    };
    
    struct DummySplitterFactory
    {
      __forceinline DummySplitterFactory(const Scene* scene)
        : scene(scene) {}
      
      __forceinline DummySplitter operator() (const PrimRef& prim) const {
        return DummySplitter(scene,prim);
      }
      
    private:
      const Scene* scene;
    };
#endif 
  }
}

