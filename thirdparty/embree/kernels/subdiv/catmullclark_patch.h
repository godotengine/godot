// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "catmullclark_ring.h"
#include "bezier_curve.h"

namespace embree
{
  template<typename Vertex, typename Vertex_t = Vertex>
    class __aligned(64) CatmullClarkPatchT
    {
    public:
    typedef CatmullClark1RingT<Vertex,Vertex_t> CatmullClark1Ring;
    typedef typename CatmullClark1Ring::Type Type;
    
    array_t<CatmullClark1RingT<Vertex,Vertex_t>,4> ring;
    
    public:
    __forceinline CatmullClarkPatchT () {}

    __forceinline CatmullClarkPatchT (const HalfEdge* first_half_edge, const char* vertices, size_t stride) {
      init(first_half_edge,vertices,stride);
    }
    
    __forceinline CatmullClarkPatchT (const HalfEdge* first_half_edge, const BufferView<Vec3fa>& vertices) {
      init(first_half_edge,vertices.getPtr(),vertices.getStride());
    }
    
    __forceinline void init (const HalfEdge* first_half_edge, const char* vertices, size_t stride) 
    {
      for (unsigned i=0; i<4; i++)
        ring[i].init(first_half_edge+i,vertices,stride);

      assert(verify());
    }

    __forceinline size_t bytes() const {
      return ring[0].bytes()+ring[1].bytes()+ring[2].bytes()+ring[3].bytes();
    }

    __forceinline void serialize(void* ptr, size_t& ofs) const
    {
      for (size_t i=0; i<4; i++)
        ring[i].serialize((char*)ptr,ofs);
    }

    __forceinline void deserialize(void* ptr)
    {
      size_t ofs = 0;
      for (size_t i=0; i<4; i++)
        ring[i].deserialize((char*)ptr,ofs);
    }

    __forceinline BBox3fa bounds() const
    {
      BBox3fa bounds (ring[0].bounds());
      for (size_t i=1; i<4; i++)
	bounds.extend(ring[i].bounds());
      return bounds;
    }
    
    __forceinline Type type() const 
    {
      const int ty0 = ring[0].type() ^ CatmullClark1Ring::TYPE_CREASES;
      const int ty1 = ring[1].type() ^ CatmullClark1Ring::TYPE_CREASES;
      const int ty2 = ring[2].type() ^ CatmullClark1Ring::TYPE_CREASES;
      const int ty3 = ring[3].type() ^ CatmullClark1Ring::TYPE_CREASES;
      return (Type) ((ty0 & ty1 & ty2 & ty3) ^ CatmullClark1Ring::TYPE_CREASES);
    }
    
    __forceinline bool isFinalResolution(float res) const {
      return ring[0].isFinalResolution(res) && ring[1].isFinalResolution(res) && ring[2].isFinalResolution(res) && ring[3].isFinalResolution(res);
    }
    
    static __forceinline void init_regular(const CatmullClark1RingT<Vertex,Vertex_t>& p0,
					   const CatmullClark1RingT<Vertex,Vertex_t>& p1,
					   CatmullClark1RingT<Vertex,Vertex_t>& dest0,
					   CatmullClark1RingT<Vertex,Vertex_t>& dest1) 
    {
      assert(p1.face_valence > 2);
      dest1.vertex_level = dest0.vertex_level = p0.edge_level;
      dest1.face_valence = dest0.face_valence = 4;
      dest1.edge_valence = dest0.edge_valence = 8;
      dest1.border_index = dest0.border_index = -1;
      dest1.vtx = dest0.vtx = (Vertex_t)p0.ring[0];
      dest1.vertex_crease_weight = dest0.vertex_crease_weight = 0.0f;
      
      dest1.ring[2] = dest0.ring[0] = (Vertex_t)p0.ring[1];
      dest1.ring[1] = dest0.ring[7] = (Vertex_t)p1.ring[0];
      dest1.ring[0] = dest0.ring[6] = (Vertex_t)p1.vtx;
      dest1.ring[7] = dest0.ring[5] = (Vertex_t)p1.ring[4];
      dest1.ring[6] = dest0.ring[4] = (Vertex_t)p0.ring[p0.edge_valence-1];
      dest1.ring[5] = dest0.ring[3] = (Vertex_t)p0.ring[p0.edge_valence-2];
      dest1.ring[4] = dest0.ring[2] = (Vertex_t)p0.vtx;
      dest1.ring[3] = dest0.ring[1] = (Vertex_t)p0.ring[2];
      
      dest1.crease_weight[1] = dest0.crease_weight[0] = 0.0f;
      dest1.crease_weight[0] = dest0.crease_weight[3] = p1.crease_weight[1];
      dest1.crease_weight[3] = dest0.crease_weight[2] = 0.0f;
      dest1.crease_weight[2] = dest0.crease_weight[1] = p0.crease_weight[0];
      
      if (p0.eval_unique_identifier <= p1.eval_unique_identifier)
      {
        dest0.eval_start_index = 3;
        dest1.eval_start_index = 0;
        dest0.eval_unique_identifier = p0.eval_unique_identifier;
        dest1.eval_unique_identifier = p0.eval_unique_identifier;
      }
      else
      {
        dest0.eval_start_index = 1;
        dest1.eval_start_index = 2;
        dest0.eval_unique_identifier = p1.eval_unique_identifier;
        dest1.eval_unique_identifier = p1.eval_unique_identifier;
      }
    }    
    
    static __forceinline void init_border(const CatmullClark1RingT<Vertex,Vertex_t> &p0,
                                          const CatmullClark1RingT<Vertex,Vertex_t> &p1,
                                          CatmullClark1RingT<Vertex,Vertex_t> &dest0,
                                          CatmullClark1RingT<Vertex,Vertex_t> &dest1) 
    {
      dest1.vertex_level = dest0.vertex_level = p0.edge_level;
      dest1.face_valence = dest0.face_valence = 3;
      dest1.edge_valence = dest0.edge_valence = 6;
      dest0.border_index = 2;
      dest1.border_index = 4;
      dest1.vtx  = dest0.vtx = (Vertex_t)p0.ring[0];
      dest1.vertex_crease_weight = dest0.vertex_crease_weight = 0.0f;
      
      dest1.ring[2] = dest0.ring[0] = (Vertex_t)p0.ring[1];
      dest1.ring[1] = dest0.ring[5] = (Vertex_t)p1.ring[0];
      dest1.ring[0] = dest0.ring[4] = (Vertex_t)p1.vtx;
      dest1.ring[5] = dest0.ring[3] = (Vertex_t)p0.ring[p0.border_index+1]; // dummy
      dest1.ring[4] = dest0.ring[2] = (Vertex_t)p0.vtx;
      dest1.ring[3] = dest0.ring[1] = (Vertex_t)p0.ring[2];
      
      dest1.crease_weight[1] = dest0.crease_weight[0] = 0.0f;
      dest1.crease_weight[0] = dest0.crease_weight[2] = p1.crease_weight[1];
      dest1.crease_weight[2] = dest0.crease_weight[1] = p0.crease_weight[0];
      
      if (p0.eval_unique_identifier <= p1.eval_unique_identifier)
      {
        dest0.eval_start_index = 1;
        dest1.eval_start_index = 2;
        dest0.eval_unique_identifier = p0.eval_unique_identifier;
        dest1.eval_unique_identifier = p0.eval_unique_identifier;
      }
      else
      {
        dest0.eval_start_index = 2;
        dest1.eval_start_index = 0;
        dest0.eval_unique_identifier = p1.eval_unique_identifier;
        dest1.eval_unique_identifier = p1.eval_unique_identifier;
      }
    }
    
    static __forceinline void init_regular(const Vertex_t &center, const Vertex_t center_ring[8], const unsigned int offset, CatmullClark1RingT<Vertex,Vertex_t> &dest)
    {
      dest.vertex_level = 0.0f;
      dest.face_valence = 4;
      dest.edge_valence = 8;
      dest.border_index = -1;
      dest.vtx     = (Vertex_t)center;
      dest.vertex_crease_weight = 0.0f;
      for (size_t i=0; i<8; i++) 
	dest.ring[i] = (Vertex_t)center_ring[(offset+i)%8];
      for (size_t i=0; i<4; i++) 
        dest.crease_weight[i] = 0.0f;
      
      dest.eval_start_index = (8-offset)>>1;
      if (dest.eval_start_index >= dest.face_valence) dest.eval_start_index -= dest.face_valence;
      assert( dest.eval_start_index < dest.face_valence );
      dest.eval_unique_identifier = 0;
    }
    
    __noinline void subdivide(array_t<CatmullClarkPatchT,4>& patch) const
    {
      ring[0].subdivide(patch[0].ring[0]);
      ring[1].subdivide(patch[1].ring[1]);
      ring[2].subdivide(patch[2].ring[2]);
      ring[3].subdivide(patch[3].ring[3]);
      
      patch[0].ring[0].edge_level = 0.5f*ring[0].edge_level;
      patch[0].ring[1].edge_level = 0.25f*(ring[1].edge_level+ring[3].edge_level);
      patch[0].ring[2].edge_level = 0.25f*(ring[0].edge_level+ring[2].edge_level);
      patch[0].ring[3].edge_level = 0.5f*ring[3].edge_level;
      
      patch[1].ring[0].edge_level = 0.5f*ring[0].edge_level;
      patch[1].ring[1].edge_level = 0.5f*ring[1].edge_level;
      patch[1].ring[2].edge_level = 0.25f*(ring[0].edge_level+ring[2].edge_level);
      patch[1].ring[3].edge_level = 0.25f*(ring[1].edge_level+ring[3].edge_level);
      
      patch[2].ring[0].edge_level = 0.25f*(ring[0].edge_level+ring[2].edge_level);
      patch[2].ring[1].edge_level = 0.5f*ring[1].edge_level;
      patch[2].ring[2].edge_level = 0.5f*ring[2].edge_level;
      patch[2].ring[3].edge_level = 0.25f*(ring[1].edge_level+ring[3].edge_level);
      
      patch[3].ring[0].edge_level = 0.25f*(ring[0].edge_level+ring[2].edge_level);
      patch[3].ring[1].edge_level = 0.25f*(ring[1].edge_level+ring[3].edge_level);
      patch[3].ring[2].edge_level = 0.5f*ring[2].edge_level;
      patch[3].ring[3].edge_level = 0.5f*ring[3].edge_level;
      
      const bool regular0 = ring[0].has_last_face() && ring[1].face_valence > 2;
      if (likely(regular0))
        init_regular(patch[0].ring[0],patch[1].ring[1],patch[0].ring[1],patch[1].ring[0]);
      else
        init_border(patch[0].ring[0],patch[1].ring[1],patch[0].ring[1],patch[1].ring[0]);
      
      const bool regular1 = ring[1].has_last_face() && ring[2].face_valence > 2;
      if (likely(regular1))
        init_regular(patch[1].ring[1],patch[2].ring[2],patch[1].ring[2],patch[2].ring[1]);
      else
        init_border(patch[1].ring[1],patch[2].ring[2],patch[1].ring[2],patch[2].ring[1]);
      
      const bool regular2 = ring[2].has_last_face() && ring[3].face_valence > 2;
      if (likely(regular2))
        init_regular(patch[2].ring[2],patch[3].ring[3],patch[2].ring[3],patch[3].ring[2]);
      else
        init_border(patch[2].ring[2],patch[3].ring[3],patch[2].ring[3],patch[3].ring[2]);
      
      const bool regular3 = ring[3].has_last_face() && ring[0].face_valence > 2;
      if (likely(regular3))
        init_regular(patch[3].ring[3],patch[0].ring[0],patch[3].ring[0],patch[0].ring[3]);
      else
        init_border(patch[3].ring[3],patch[0].ring[0],patch[3].ring[0],patch[0].ring[3]);
      
      Vertex_t center = (ring[0].vtx + ring[1].vtx + ring[2].vtx + ring[3].vtx) * 0.25f;

      Vertex_t center_ring[8];
      center_ring[0] = (Vertex_t)patch[3].ring[3].ring[0];
      center_ring[7] = (Vertex_t)patch[3].ring[3].vtx;
      center_ring[6] = (Vertex_t)patch[2].ring[2].ring[0];
      center_ring[5] = (Vertex_t)patch[2].ring[2].vtx;
      center_ring[4] = (Vertex_t)patch[1].ring[1].ring[0];
      center_ring[3] = (Vertex_t)patch[1].ring[1].vtx;
      center_ring[2] = (Vertex_t)patch[0].ring[0].ring[0];
      center_ring[1] = (Vertex_t)patch[0].ring[0].vtx;
      
      init_regular(center,center_ring,0,patch[0].ring[2]);
      init_regular(center,center_ring,2,patch[1].ring[3]);
      init_regular(center,center_ring,4,patch[2].ring[0]);
      init_regular(center,center_ring,6,patch[3].ring[1]);
      
      assert(patch[0].verify());
      assert(patch[1].verify());
      assert(patch[2].verify());
      assert(patch[3].verify());
    }
    
    bool verify() const {
      return ring[0].hasValidPositions() && ring[1].hasValidPositions() && ring[2].hasValidPositions() && ring[3].hasValidPositions();
    }
    
    __forceinline void init( FinalQuad& quad ) const
    {
      quad.vtx[0] = (Vertex_t)ring[0].vtx;
      quad.vtx[1] = (Vertex_t)ring[1].vtx;
      quad.vtx[2] = (Vertex_t)ring[2].vtx;
      quad.vtx[3] = (Vertex_t)ring[3].vtx;
    };
    
    friend __forceinline embree_ostream operator<<(embree_ostream o, const CatmullClarkPatchT &p)
    {
      o << "CatmullClarkPatch { " << embree_endl;
      for (size_t i=0; i<4; i++)
	o << "ring" << i << ": " << p.ring[i] << embree_endl;
      o << "}" << embree_endl;
      return o;
    }
    };
  
  typedef CatmullClarkPatchT<Vec3fa,Vec3fa_t> CatmullClarkPatch3fa;
  
  template<typename Vertex, typename Vertex_t = Vertex>
    class __aligned(64) GeneralCatmullClarkPatchT
    {
    public:
    typedef CatmullClarkPatchT<Vertex,Vertex_t> CatmullClarkPatch;
    typedef CatmullClark1RingT<Vertex,Vertex_t> CatmullClark1Ring;
    typedef BezierCurveT<Vertex> BezierCurve;

    static const unsigned SIZE = MAX_PATCH_VALENCE;
    DynamicStackArray<GeneralCatmullClark1RingT<Vertex,Vertex_t>,8,SIZE> ring;
    unsigned N;
    
    __forceinline GeneralCatmullClarkPatchT () 
    : N(0) {}
    
    GeneralCatmullClarkPatchT (const HalfEdge* h, const char* vertices, size_t stride) {
      init(h,vertices,stride);
    }

    __forceinline GeneralCatmullClarkPatchT (const HalfEdge* first_half_edge, const BufferView<Vec3fa>& vertices) {
      init(first_half_edge,vertices.getPtr(),vertices.getStride());
    }

    __forceinline void init (const HalfEdge* h, const char* vertices, size_t stride) 
    {
      unsigned int i = 0;
      const HalfEdge* edge = h; 
      do {
        ring[i].init(edge,vertices,stride);
        edge = edge->next();
        i++;
      } while ((edge != h) && (i < SIZE));
      N = i;
    }

    __forceinline unsigned size() const { 
      return N; 
    }
    
    __forceinline bool isQuadPatch() const {
      return (N == 4) && ring[0].only_quads && ring[1].only_quads && ring[2].only_quads && ring[3].only_quads;
    }

    static __forceinline void init_regular(const CatmullClark1RingT<Vertex,Vertex_t>& p0,
					   const CatmullClark1RingT<Vertex,Vertex_t>& p1,
					   CatmullClark1RingT<Vertex,Vertex_t>& dest0,
					   CatmullClark1RingT<Vertex,Vertex_t>& dest1) 
    {
      assert(p1.face_valence > 2);
      dest1.vertex_level = dest0.vertex_level = p0.edge_level;
      dest1.face_valence = dest0.face_valence = 4;
      dest1.edge_valence = dest0.edge_valence = 8;
      dest1.border_index = dest0.border_index = -1;
      dest1.vtx = dest0.vtx = (Vertex_t)p0.ring[0];
      dest1.vertex_crease_weight = dest0.vertex_crease_weight = 0.0f;
      
      dest1.ring[2] = dest0.ring[0] = (Vertex_t)p0.ring[1];
      dest1.ring[1] = dest0.ring[7] = (Vertex_t)p1.ring[0];
      dest1.ring[0] = dest0.ring[6] = (Vertex_t)p1.vtx;
      dest1.ring[7] = dest0.ring[5] = (Vertex_t)p1.ring[4];
      dest1.ring[6] = dest0.ring[4] = (Vertex_t)p0.ring[p0.edge_valence-1];
      dest1.ring[5] = dest0.ring[3] = (Vertex_t)p0.ring[p0.edge_valence-2];
      dest1.ring[4] = dest0.ring[2] = (Vertex_t)p0.vtx;
      dest1.ring[3] = dest0.ring[1] = (Vertex_t)p0.ring[2];
      
      dest1.crease_weight[1] = dest0.crease_weight[0] = 0.0f;
      dest1.crease_weight[0] = dest0.crease_weight[3] = p1.crease_weight[1];
      dest1.crease_weight[3] = dest0.crease_weight[2] = 0.0f;
      dest1.crease_weight[2] = dest0.crease_weight[1] = p0.crease_weight[0];
      
      if (p0.eval_unique_identifier <= p1.eval_unique_identifier)
      {
        dest0.eval_start_index = 3;
        dest1.eval_start_index = 0;
        dest0.eval_unique_identifier = p0.eval_unique_identifier;
        dest1.eval_unique_identifier = p0.eval_unique_identifier;
      }
      else
      {
        dest0.eval_start_index = 1;
        dest1.eval_start_index = 2;
        dest0.eval_unique_identifier = p1.eval_unique_identifier;
        dest1.eval_unique_identifier = p1.eval_unique_identifier;
      }      
    }
    
    
    static __forceinline void init_border(const CatmullClark1RingT<Vertex,Vertex_t> &p0,
                                          const CatmullClark1RingT<Vertex,Vertex_t> &p1,
                                          CatmullClark1RingT<Vertex,Vertex_t> &dest0,
                                          CatmullClark1RingT<Vertex,Vertex_t> &dest1) 
    {
      dest1.vertex_level = dest0.vertex_level = p0.edge_level;
      dest1.face_valence = dest0.face_valence = 3;
      dest1.edge_valence = dest0.edge_valence = 6;
      dest0.border_index = 2;
      dest1.border_index = 4;
      dest1.vtx  = dest0.vtx = (Vertex_t)p0.ring[0];
      dest1.vertex_crease_weight = dest0.vertex_crease_weight = 0.0f;
      
      dest1.ring[2] = dest0.ring[0] = (Vertex_t)p0.ring[1];
      dest1.ring[1] = dest0.ring[5] = (Vertex_t)p1.ring[0];
      dest1.ring[0] = dest0.ring[4] = (Vertex_t)p1.vtx;
      dest1.ring[5] = dest0.ring[3] = (Vertex_t)p0.ring[p0.border_index+1]; // dummy
      dest1.ring[4] = dest0.ring[2] = (Vertex_t)p0.vtx;
      dest1.ring[3] = dest0.ring[1] = (Vertex_t)p0.ring[2];
      
      dest1.crease_weight[1] = dest0.crease_weight[0] = 0.0f;
      dest1.crease_weight[0] = dest0.crease_weight[2] = p1.crease_weight[1];
      dest1.crease_weight[2] = dest0.crease_weight[1] = p0.crease_weight[0];
      
      if (p0.eval_unique_identifier <= p1.eval_unique_identifier)
      {
        dest0.eval_start_index = 1;
        dest1.eval_start_index = 2;
        dest0.eval_unique_identifier = p0.eval_unique_identifier;
        dest1.eval_unique_identifier = p0.eval_unique_identifier;
      }
      else
      {
        dest0.eval_start_index = 2;
        dest1.eval_start_index = 0;
        dest0.eval_unique_identifier = p1.eval_unique_identifier;
        dest1.eval_unique_identifier = p1.eval_unique_identifier;
      }
    }
    
    static __forceinline void init_regular(const Vertex_t &center, const array_t<Vertex_t,2*SIZE>& center_ring, const float vertex_level, const unsigned int N, const unsigned int offset, CatmullClark1RingT<Vertex,Vertex_t> &dest)
    {
      assert(N<(MAX_RING_FACE_VALENCE));
      assert(2*N<(MAX_RING_EDGE_VALENCE));
      dest.vertex_level = vertex_level;
      dest.face_valence = N;
      dest.edge_valence = 2*N;
      dest.border_index = -1;
      dest.vtx     = (Vertex_t)center;
      dest.vertex_crease_weight = 0.0f;
      for (unsigned i=0; i<2*N; i++) {
        dest.ring[i] = (Vertex_t)center_ring[(2*N+offset+i-1)%(2*N)];
        assert(isvalid(dest.ring[i]));
      }
      for (unsigned i=0; i<N; i++) 
        dest.crease_weight[i] = 0.0f;
      
      assert(offset <= 2*N);
      dest.eval_start_index = (2*N-offset)>>1;
      if (dest.eval_start_index >= dest.face_valence) dest.eval_start_index -= dest.face_valence;
      
      assert( dest.eval_start_index < dest.face_valence );
      dest.eval_unique_identifier = 0;
    }
    
    __noinline void subdivide(array_t<CatmullClarkPatch,SIZE>& patch, unsigned& N_o) const
    {
      N_o = N;
      assert( N );
      for (unsigned i=0; i<N; i++) {
        unsigned ip1 = (i+1)%N; // FIXME: %
        ring[i].subdivide(patch[i].ring[0]);
        patch[i]  .ring[0].edge_level = 0.5f*ring[i].edge_level;
        patch[ip1].ring[3].edge_level = 0.5f*ring[i].edge_level;
        
	assert( patch[i].ring[0].hasValidPositions() );
        
      }
      assert(N < 2*SIZE);
      Vertex_t center = Vertex_t(0.0f);
      array_t<Vertex_t,2*SIZE> center_ring;
      float center_vertex_level = 2.0f; // guarantees that irregular vertices get always isolated also for non-quads
      
      for (unsigned i=0; i<N; i++)
      {
        unsigned ip1 = (i+1)%N; // FIXME: %
        unsigned im1 = (i+N-1)%N; // FIXME: %
        bool regular = ring[i].has_last_face() && ring[ip1].face_valence > 2;
        if (likely(regular)) init_regular(patch[i].ring[0],patch[ip1].ring[0],patch[i].ring[1],patch[ip1].ring[3]); 
        else                 init_border (patch[i].ring[0],patch[ip1].ring[0],patch[i].ring[1],patch[ip1].ring[3]);
        
	assert( patch[i].ring[1].hasValidPositions() );
	assert( patch[ip1].ring[3].hasValidPositions() );
        
	float level = 0.25f*(ring[im1].edge_level+ring[ip1].edge_level);
        patch[i].ring[1].edge_level = patch[ip1].ring[2].edge_level = level;
	center_vertex_level = max(center_vertex_level,level);
        
        center += ring[i].vtx;
        center_ring[2*i+0] = (Vertex_t)patch[i].ring[0].vtx;
        center_ring[2*i+1] = (Vertex_t)patch[i].ring[0].ring[0];
      }
      center /= float(N);
      
      for (unsigned int i=0; i<N; i++) {
        init_regular(center,center_ring,center_vertex_level,N,2*i,patch[i].ring[2]);
        
	assert( patch[i].ring[2].hasValidPositions() );
      }
    }
    
    void init(CatmullClarkPatch& patch) const
    {
      assert(size() == 4);
      ring[0].convert(patch.ring[0]);
      ring[1].convert(patch.ring[1]);
      ring[2].convert(patch.ring[2]);
      ring[3].convert(patch.ring[3]);
    }
    
    static void fix_quad_ring_order (array_t<CatmullClarkPatch,GeneralCatmullClarkPatchT::SIZE>& patches)
    {
      CatmullClark1Ring patches1ring1 = patches[1].ring[1];
      patches[1].ring[1] = patches[1].ring[0]; // FIXME: optimize these assignments
      patches[1].ring[0] = patches[1].ring[3];
      patches[1].ring[3] = patches[1].ring[2];
      patches[1].ring[2] = patches1ring1;
      
      CatmullClark1Ring patches2ring2 = patches[2].ring[2];
      patches[2].ring[2] = patches[2].ring[0];
      patches[2].ring[0] = patches2ring2;
      CatmullClark1Ring patches2ring3 = patches[2].ring[3];
      patches[2].ring[3] = patches[2].ring[1];
      patches[2].ring[1] = patches2ring3;
      
      CatmullClark1Ring patches3ring3 = patches[3].ring[3];
      patches[3].ring[3] = patches[3].ring[0];
      patches[3].ring[0] = patches[3].ring[1];
      patches[3].ring[1] = patches[3].ring[2];
      patches[3].ring[2] = patches3ring3;
    }

    __forceinline void getLimitBorder(BezierCurve curves[GeneralCatmullClarkPatchT::SIZE]) const
    {
      Vertex P0 = ring[0].getLimitVertex();
      for (unsigned i=0; i<N; i++)
      {
        const unsigned i0 = i, i1 = i+1==N ? 0 : i+1;
        const Vertex P1 = madd(1.0f/3.0f,ring[i0].getLimitTangent(),P0);
        const Vertex P3 = ring[i1].getLimitVertex();
        const Vertex P2 = madd(1.0f/3.0f,ring[i1].getSecondLimitTangent(),P3);
        new (&curves[i]) BezierCurve(P0,P1,P2,P3);
        P0 = P3;
      }
    }

    __forceinline void getLimitBorder(BezierCurve curves[2], const unsigned subPatch) const
    {
      const unsigned i0 = subPatch;
      const Vertex t0_p = ring[i0].getLimitTangent();
      const Vertex t0_m = ring[i0].getSecondLimitTangent();
          
      const unsigned i1 = subPatch+1 == N ? 0 : subPatch+1;
      const Vertex t1_p = ring[i1].getLimitTangent();
      const Vertex t1_m = ring[i1].getSecondLimitTangent();
      
      const unsigned i2 = subPatch == 0 ? N-1 : subPatch-1;
      const Vertex t2_p = ring[i2].getLimitTangent();
      const Vertex t2_m = ring[i2].getSecondLimitTangent();
      
      const Vertex b00 = ring[i0].getLimitVertex();
      const Vertex b03 = ring[i1].getLimitVertex();
      const Vertex b33 = ring[i2].getLimitVertex();
      
      const Vertex b01 = madd(1.0/3.0f,t0_p,b00);
      const Vertex b11 = madd(1.0/3.0f,t0_m,b00);
      
      //const Vertex b13 = madd(1.0/3.0f,t1_p,b03);
      const Vertex b02 = madd(1.0/3.0f,t1_m,b03);
          
      const Vertex b22 = madd(1.0/3.0f,t2_p,b33);
      const Vertex b23 = madd(1.0/3.0f,t2_m,b33);
          
      new (&curves[0]) BezierCurve(b00,b01,b02,b03);
      new (&curves[1]) BezierCurve(b33,b22,b11,b00);
    }
    
    friend __forceinline embree_ostream operator<<(embree_ostream o, const GeneralCatmullClarkPatchT &p)
    {
      o << "GeneralCatmullClarkPatch { " << embree_endl;
      for (unsigned i=0; i<p.N; i++)
	o << "ring" << i << ": " << p.ring[i] << embree_endl;
      o << "}" << embree_endl;
      return o;
    }
    };
  
  typedef GeneralCatmullClarkPatchT<Vec3fa,Vec3fa_t> GeneralCatmullClarkPatch3fa;
}
