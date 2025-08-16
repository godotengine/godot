// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"

namespace embree
{
  template<int M>
  struct LineMi
  {
    /* Virtual interface to query information about the line segment type */
    struct Type : public PrimitiveType
    {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;      
    };
    static Type type;

  public:

    /* primitive supports multiple time segments */
    static const bool singleTimeSegment = false;

    /* Returns maximum number of stored line segments */
    static __forceinline size_t max_size() { return M; }

    /* Returns required number of primitive blocks for N line segments */
    static __forceinline size_t blocks(size_t N) { return (N+max_size()-1)/max_size(); }

    /* Returns required number of bytes for N line segments */
    static __forceinline size_t bytes(size_t N) { return blocks(N)*sizeof(LineMi); }

  public:

    /* Default constructor */
    __forceinline LineMi() {  }

    /* Construction from vertices and IDs */
    __forceinline LineMi(const vuint<M>& v0, unsigned short leftExists, unsigned short rightExists, const vuint<M>& geomIDs, const vuint<M>& primIDs, Geometry::GType gtype)
      : gtype((unsigned char)gtype), m((unsigned char)popcnt(vuint<M>(primIDs) != vuint<M>(-1))), sharedGeomID(geomIDs[0]), leftExists (leftExists), rightExists(rightExists), v0(v0), primIDs(primIDs)
    {
      assert(all(vuint<M>(geomID()) == geomIDs));
    }

    /* Returns a mask that tells which line segments are valid */
    __forceinline vbool<M> valid() const { return primIDs != vuint<M>(-1); }

    /* Returns if the specified line segment is valid */
    __forceinline bool valid(const size_t i) const { assert(i<M); return primIDs[i] != -1; }

    /* Returns the number of stored line segments */
    __forceinline size_t size() const { return bsf(~movemask(valid())); }

    /* Returns the geometry IDs */
    //template<class T>
    //static __forceinline T unmask(T &index) { return index & 0x3fffffff; }

    __forceinline     unsigned int geomID(unsigned int i = 0) const { return sharedGeomID; }
    //__forceinline       vuint<M> geomID()       { return unmask(geomIDs); }
    //__forceinline const vuint<M> geomID() const { return unmask(geomIDs); }
    //__forceinline unsigned int geomID(const size_t i) const { assert(i<M); return unmask(geomIDs[i]); }

    /* Returns the primitive IDs */
    __forceinline       vuint<M>& primID()       { return primIDs; }
    __forceinline const vuint<M>& primID() const { return primIDs; }
    __forceinline unsigned int primID(const size_t i) const { assert(i<M); return primIDs[i]; }

    /* gather the line segments */
    __forceinline void gather(Vec4vf<M>& p0,
                              Vec4vf<M>& p1,
                              const LineSegments* geom) const;

    __forceinline void gatheri(Vec4vf<M>& p0,
                               Vec4vf<M>& p1,
                               const LineSegments* geom,
                               const int itime) const;

    __forceinline void gather(Vec4vf<M>& p0,
                              Vec4vf<M>& p1,
                              const LineSegments* geom,
                              float time) const;

    /* gather the line segments with lateral info */
    __forceinline void gather(Vec4vf<M>& p0,
                              Vec4vf<M>& p1,
                              Vec4vf<M>& pL,
                              Vec4vf<M>& pR,
                              const LineSegments* geom) const;

    __forceinline void gatheri(Vec4vf<M>& p0,
                               Vec4vf<M>& p1,
                               Vec4vf<M>& pL,
                               Vec4vf<M>& pR,
                               const LineSegments* geom,
                               const int itime) const;

    __forceinline void gather(Vec4vf<M>& p0,
                              Vec4vf<M>& p1,
                              Vec4vf<M>& pL,
                              Vec4vf<M>& pR,
                              const LineSegments* geom,
                              float time) const;

    __forceinline void gather(Vec4vf<M>& p0,
                              Vec4vf<M>& p1,
                              vbool<M>& cL,
                              vbool<M>& cR,
                              const LineSegments* geom) const;

    __forceinline void gatheri(Vec4vf<M>& p0,
                               Vec4vf<M>& p1,
                               vbool<M>& cL,
                               vbool<M>& cR,
                               const LineSegments* geom,
                               const int itime) const;

    __forceinline void gather(Vec4vf<M>& p0,
                              Vec4vf<M>& p1,
                              vbool<M>& cL,
                              vbool<M>& cR,
                              const LineSegments* geom,
                              float time) const;

    /* Calculate the bounds of the line segments */
    __forceinline const BBox3fa bounds(const Scene* scene, size_t itime = 0) const
    {
      BBox3fa bounds = empty;
      for (size_t i=0; i<M && valid(i); i++)
      {
        const LineSegments* geom = scene->get<LineSegments>(geomID(i));
        const Vec3ff& p0 = geom->vertex(v0[i]+0,itime);
        const Vec3ff& p1 = geom->vertex(v0[i]+1,itime);
        BBox3fa b = merge(BBox3fa(p0),BBox3fa(p1));
        b = enlarge(b,Vec3fa(max(p0.w,p1.w)));
        bounds.extend(b);
      }
      return bounds;
    }

    /* Calculate the linear bounds of the primitive */
    __forceinline LBBox3fa linearBounds(const Scene* scene, size_t itime) {
      return LBBox3fa(bounds(scene,itime+0), bounds(scene,itime+1));
    }

    __forceinline LBBox3fa linearBounds(const Scene *const scene, size_t itime, size_t numTimeSteps) {
      LBBox3fa allBounds = empty;
      for (size_t i=0; i<M && valid(i); i++)
      {
        const LineSegments* geom = scene->get<LineSegments>(geomID(i));
        allBounds.extend(geom->linearBounds(primID(i), itime, numTimeSteps));
      }
      return allBounds;
    }

    __forceinline LBBox3fa linearBounds(const Scene *const scene, const BBox1f time_range) 
    {
      LBBox3fa allBounds = empty;
      for (size_t i=0; i<M && valid(i); i++)
      {
        const LineSegments* geom = scene->get<LineSegments>(geomID((unsigned int)i));
        allBounds.extend(geom->linearBounds(primID(i), time_range));
      }
      return allBounds;
    }

    /* Fill line segment from line segment list */
    template<typename PrimRefT>
    __forceinline void fill(const PrimRefT* prims, size_t& begin, size_t end, Scene* scene)
    {
      Geometry::GType gty = scene->get(prims[begin].geomID())->getType();
      vuint<M> geomID, primID;
      vuint<M> v0;
      unsigned short leftExists = 0;
      unsigned short rightExists = 0;
      const PrimRefT* prim = &prims[begin];

      for (size_t i=0; i<M; i++)
      {
        const LineSegments* geom = scene->get<LineSegments>(prim->geomID());
        if (begin<end) {
          geomID[i] = prim->geomID();
          primID[i] = prim->primID();
          v0[i] = geom->segment(prim->primID());
          leftExists |= geom->segmentLeftExists(primID[i]) << i;
          rightExists |= geom->segmentRightExists(primID[i]) << i;         
          begin++;
        } else {
          assert(i);
          if (i>0) {
            geomID[i] = geomID[i-1];
            primID[i] = -1;
            v0[i] = v0[i-1];
          }
        }
        if (begin<end) prim = &prims[begin]; // FIXME: remove this line
      }
      new (this) LineMi(v0,leftExists,rightExists,geomID,primID,gty); // FIXME: use non temporal store
    }

     template<typename BVH, typename Allocator>
      __forceinline static typename BVH::NodeRef createLeaf (BVH* bvh, const PrimRef* prims, const range<size_t>& set, const Allocator& alloc)
    {
      size_t start = set.begin();
      size_t items = LineMi::blocks(set.size());
      size_t numbytes = LineMi::bytes(set.size());
      LineMi* accel = (LineMi*) alloc.malloc1(numbytes,M*sizeof(float));
      for (size_t i=0; i<items; i++) {
        accel[i].fill(prims,start,set.end(),bvh->scene);
      }
      return bvh->encodeLeaf((char*)accel,items);
    };
    
    __forceinline LBBox3fa fillMB(const PrimRef* prims, size_t& begin, size_t end, Scene* scene, size_t itime)
    {
      fill(prims,begin,end,scene);
      return linearBounds(scene,itime);
    }

    __forceinline LBBox3fa fillMB(const PrimRefMB* prims, size_t& begin, size_t end, Scene* scene, const BBox1f time_range)
    {
      fill(prims,begin,end,scene);
      return linearBounds(scene,time_range);
    }

      template<typename BVH, typename SetMB, typename Allocator>
    __forceinline static typename BVH::NodeRecordMB4D createLeafMB(BVH* bvh, const SetMB& prims, const Allocator& alloc)
    {
      size_t start = prims.begin();
      size_t end   = prims.end();
      size_t items = LineMi::blocks(prims.size());
      size_t numbytes = LineMi::bytes(prims.size());
      LineMi* accel = (LineMi*) alloc.malloc1(numbytes,M*sizeof(float));
      const typename BVH::NodeRef node = bvh->encodeLeaf((char*)accel,items);
      
      LBBox3fa bounds = empty;
      for (size_t i=0; i<items; i++)
        bounds.extend(accel[i].fillMB(prims.prims->data(),start,end,bvh->scene,prims.time_range));
      
      return typename BVH::NodeRecordMB4D(node,bounds,prims.time_range);
    };

    /* Updates the primitive */
    __forceinline BBox3fa update(LineSegments* geom)
    {
      BBox3fa bounds = empty;
      for (size_t i=0; i<M && valid(i); i++)
      {
        const Vec3ff& p0 = geom->vertex(v0[i]+0);
        const Vec3ff& p1 = geom->vertex(v0[i]+1);
        BBox3fa b = merge(BBox3fa(p0),BBox3fa(p1));
        b = enlarge(b,Vec3fa(max(p0.w,p1.w)));
        bounds.extend(b);
      }
      return bounds;
    }

    /*! output operator */
    friend __forceinline embree_ostream operator<<(embree_ostream cout, const LineMi& line) {
      return cout << "Line" << M << "i {" << line.v0 << ", " << line.geomID() << ", " << line.primID() << "}";
    }
    
  public:
    unsigned char gtype;
    unsigned char m;
    unsigned int sharedGeomID;
    unsigned short leftExists, rightExists;
    vuint<M> v0;      // index of start vertex
  private:
    vuint<M> primIDs; // primitive ID
  };

  template<>
    __forceinline void LineMi<4>::gather(Vec4vf4& p0,
                                         Vec4vf4& p1,
                                         const LineSegments* geom) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3]));
    transpose(a0,a1,a2,a3,p0.x,p0.y,p0.z,p0.w);

    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1));
    transpose(b0,b1,b2,b3,p1.x,p1.y,p1.z,p1.w);
  }

  template<>
  __forceinline void LineMi<4>::gatheri(Vec4vf4& p0,
                                       Vec4vf4& p1,
                                       const LineSegments* geom,
                                       const int itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0],itime));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1],itime));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2],itime));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3],itime));
    transpose(a0,a1,a2,a3,p0.x,p0.y,p0.z,p0.w);

    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1,itime));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1,itime));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1,itime));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1,itime));
    transpose(b0,b1,b2,b3,p1.x,p1.y,p1.z,p1.w);
  }

  template<>
    __forceinline void LineMi<4>::gather(Vec4vf4& p0,
                                         Vec4vf4& p1,
                                         const LineSegments* geom,
                                         float time) const
  {
    float ftime;
    const int itime = geom->timeSegment(time, ftime);

    Vec4vf4 a0,a1;
    gatheri(a0,a1,geom,itime);
    Vec4vf4 b0,b1;
    gatheri(b0,b1,geom,itime+1);
    p0 = lerp(a0,b0,vfloat4(ftime));
    p1 = lerp(a1,b1,vfloat4(ftime));
  }

  template<>
    __forceinline void LineMi<4>::gather(Vec4vf4& p0,
                                         Vec4vf4& p1,
                                         vbool4&  cL,
                                         vbool4&  cR,
                                         const LineSegments* geom) const
  {
    gather(p0,p1,geom);
    cL = !vbool4(leftExists);
    cR = !vbool4(rightExists);
  }

  template<>
    __forceinline void LineMi<4>::gatheri(Vec4vf4& p0,
                                          Vec4vf4& p1,
                                          vbool4&  cL,
                                          vbool4&  cR,
                                          const LineSegments* geom,
                                          const int itime) const
  {
    gatheri(p0,p1,geom,itime);
    cL = !vbool4(leftExists);
    cR = !vbool4(rightExists);
  }

  template<>
    __forceinline void LineMi<4>::gather(Vec4vf4& p0,
                                         Vec4vf4& p1,
                                         vbool4&  cL,
                                         vbool4&  cR,
                                         const LineSegments* geom,
                                         float time) const
  {
    float ftime;
    const int itime = geom->timeSegment(time, ftime);
    
    Vec4vf4 a0,a1;
    gatheri(a0,a1,geom,itime);
    Vec4vf4 b0,b1;
    gatheri(b0,b1,geom,itime+1);
    p0 = lerp(a0,b0,vfloat4(ftime));
    p1 = lerp(a1,b1,vfloat4(ftime));
    cL = !vbool4(leftExists);
    cR = !vbool4(rightExists);
  }

  template<>
    __forceinline void LineMi<4>::gather(Vec4vf4& p0,
                                              Vec4vf4& p1,
                                              Vec4vf4& pL,
                                              Vec4vf4& pR,
                                              const LineSegments* geom) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3]));
    transpose(a0,a1,a2,a3,p0.x,p0.y,p0.z,p0.w);

    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1));
    transpose(b0,b1,b2,b3,p1.x,p1.y,p1.z,p1.w);
    
    const vfloat4 l0 = (leftExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]-1)) : vfloat4(inf);
    const vfloat4 l1 = (leftExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]-1)) : vfloat4(inf);
    const vfloat4 l2 = (leftExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]-1)) : vfloat4(inf);
    const vfloat4 l3 = (leftExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]-1)) : vfloat4(inf);
    transpose(l0,l1,l2,l3,pL.x,pL.y,pL.z,pL.w);
    
    const vfloat4 r0 = (rightExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]+2)) : vfloat4(inf);
    const vfloat4 r1 = (rightExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]+2)) : vfloat4(inf);
    const vfloat4 r2 = (rightExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]+2)) : vfloat4(inf);
    const vfloat4 r3 = (rightExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]+2)) : vfloat4(inf);
    transpose(r0,r1,r2,r3,pR.x,pR.y,pR.z,pR.w);
  }
  
  template<>
    __forceinline void LineMi<4>::gatheri(Vec4vf4& p0,
                                              Vec4vf4& p1,
                                              Vec4vf4& pL,
                                              Vec4vf4& pR,
                                              const LineSegments* geom,
                                              const int itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0],itime));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1],itime));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2],itime));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3],itime));
    transpose(a0,a1,a2,a3,p0.x,p0.y,p0.z,p0.w);
    
    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1,itime));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1,itime));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1,itime));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1,itime));
    transpose(b0,b1,b2,b3,p1.x,p1.y,p1.z,p1.w);
    
    const vfloat4 l0 = (leftExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]-1,itime)) : vfloat4(inf);
    const vfloat4 l1 = (leftExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]-1,itime)) : vfloat4(inf);
    const vfloat4 l2 = (leftExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]-1,itime)) : vfloat4(inf);
    const vfloat4 l3 = (leftExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]-1,itime)) : vfloat4(inf);
    transpose(l0,l1,l2,l3,pL.x,pL.y,pL.z,pL.w);
    
    const vfloat4 r0 = (rightExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]+2,itime)) : vfloat4(inf);
    const vfloat4 r1 = (rightExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]+2,itime)) : vfloat4(inf);
    const vfloat4 r2 = (rightExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]+2,itime)) : vfloat4(inf);
    const vfloat4 r3 = (rightExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]+2,itime)) : vfloat4(inf);
    transpose(r0,r1,r2,r3,pR.x,pR.y,pR.z,pR.w);
  }
  
  template<>
    __forceinline void LineMi<4>::gather(Vec4vf4& p0,
                                              Vec4vf4& p1,
                                              Vec4vf4& pL,
                                              Vec4vf4& pR,
                                              const LineSegments* geom,
                                              float time) const
  {
    float ftime;
    const int itime = geom->timeSegment(time, ftime);
    
    Vec4vf4 a0,a1,aL,aR;
    gatheri(a0,a1,aL,aR,geom,itime);
    Vec4vf4 b0,b1,bL,bR;
    gatheri(b0,b1,bL,bR,geom,itime+1);
    p0 = lerp(a0,b0,vfloat4(ftime));
    p1 = lerp(a1,b1,vfloat4(ftime));
    pL = lerp(aL,bL,vfloat4(ftime));
    pR = lerp(aR,bR,vfloat4(ftime));

    pL = select(vboolf4(leftExists), pL, Vec4vf4(inf));
    pR = select(vboolf4(rightExists), pR, Vec4vf4(inf));
  }

#if defined(__AVX__)

  template<>
    __forceinline void LineMi<8>::gather(Vec4vf8& p0,
                                         Vec4vf8& p1,
                                         const LineSegments* geom) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3]));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(v0[4]));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(v0[5]));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(v0[6]));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(v0[7]));
    transpose(a0,a1,a2,a3,a4,a5,a6,a7,p0.x,p0.y,p0.z,p0.w);

    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1));
    const vfloat4 b4 = vfloat4::loadu(geom->vertexPtr(v0[4]+1));
    const vfloat4 b5 = vfloat4::loadu(geom->vertexPtr(v0[5]+1));
    const vfloat4 b6 = vfloat4::loadu(geom->vertexPtr(v0[6]+1));
    const vfloat4 b7 = vfloat4::loadu(geom->vertexPtr(v0[7]+1));
    transpose(b0,b1,b2,b3,b4,b5,b6,b7,p1.x,p1.y,p1.z,p1.w);
  }

  template<>
  __forceinline void LineMi<8>::gatheri(Vec4vf8& p0,
                                       Vec4vf8& p1,
                                       const LineSegments* geom,
                                       const int itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0],itime));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1],itime));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2],itime));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3],itime));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(v0[4],itime));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(v0[5],itime));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(v0[6],itime));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(v0[7],itime));
    transpose(a0,a1,a2,a3,a4,a5,a6,a7,p0.x,p0.y,p0.z,p0.w);

    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1,itime));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1,itime));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1,itime));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1,itime));
    const vfloat4 b4 = vfloat4::loadu(geom->vertexPtr(v0[4]+1,itime));
    const vfloat4 b5 = vfloat4::loadu(geom->vertexPtr(v0[5]+1,itime));
    const vfloat4 b6 = vfloat4::loadu(geom->vertexPtr(v0[6]+1,itime));
    const vfloat4 b7 = vfloat4::loadu(geom->vertexPtr(v0[7]+1,itime));
    transpose(b0,b1,b2,b3,b4,b5,b6,b7,p1.x,p1.y,p1.z,p1.w);
  }

  template<>
    __forceinline void LineMi<8>::gather(Vec4vf8& p0,
                                         Vec4vf8& p1,
                                         const LineSegments* geom,
                                         float time) const
  {
    float ftime;
    const int itime = geom->timeSegment(time, ftime);

    Vec4vf8 a0,a1;
    gatheri(a0,a1,geom,itime);
    Vec4vf8 b0,b1;
    gatheri(b0,b1,geom,itime+1);
    p0 = lerp(a0,b0,vfloat8(ftime));
    p1 = lerp(a1,b1,vfloat8(ftime));
  }
  
  template<>
    __forceinline void LineMi<8>::gather(Vec4vf8& p0,
                                              Vec4vf8& p1,
                                              Vec4vf8& pL,
                                              Vec4vf8& pR,
                                              const LineSegments* geom) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3]));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(v0[4]));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(v0[5]));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(v0[6]));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(v0[7]));
    transpose(a0,a1,a2,a3,a4,a5,a6,a7,p0.x,p0.y,p0.z,p0.w);
    
    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1));
    const vfloat4 b4 = vfloat4::loadu(geom->vertexPtr(v0[4]+1));
    const vfloat4 b5 = vfloat4::loadu(geom->vertexPtr(v0[5]+1));
    const vfloat4 b6 = vfloat4::loadu(geom->vertexPtr(v0[6]+1));
    const vfloat4 b7 = vfloat4::loadu(geom->vertexPtr(v0[7]+1));
    transpose(b0,b1,b2,b3,b4,b5,b6,b7,p1.x,p1.y,p1.z,p1.w);
    
    const vfloat4 l0 = (leftExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]-1)) : vfloat4(inf);
    const vfloat4 l1 = (leftExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]-1)) : vfloat4(inf);
    const vfloat4 l2 = (leftExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]-1)) : vfloat4(inf);
    const vfloat4 l3 = (leftExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]-1)) : vfloat4(inf);
    const vfloat4 l4 = (leftExists & (1<<4)) ? vfloat4::loadu(geom->vertexPtr(v0[4]-1)) : vfloat4(inf);
    const vfloat4 l5 = (leftExists & (1<<5)) ? vfloat4::loadu(geom->vertexPtr(v0[5]-1)) : vfloat4(inf);
    const vfloat4 l6 = (leftExists & (1<<6)) ? vfloat4::loadu(geom->vertexPtr(v0[6]-1)) : vfloat4(inf);
    const vfloat4 l7 = (leftExists & (1<<7)) ? vfloat4::loadu(geom->vertexPtr(v0[7]-1)) : vfloat4(inf);
    transpose(l0,l1,l2,l3,l4,l5,l6,l7,pL.x,pL.y,pL.z,pL.w);
    
    const vfloat4 r0 = (rightExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]+2)) : vfloat4(inf);
    const vfloat4 r1 = (rightExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]+2)) : vfloat4(inf);
    const vfloat4 r2 = (rightExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]+2)) : vfloat4(inf);
    const vfloat4 r3 = (rightExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]+2)) : vfloat4(inf);
    const vfloat4 r4 = (rightExists & (1<<4)) ? vfloat4::loadu(geom->vertexPtr(v0[4]+2)) : vfloat4(inf);
    const vfloat4 r5 = (rightExists & (1<<5)) ? vfloat4::loadu(geom->vertexPtr(v0[5]+2)) : vfloat4(inf);
    const vfloat4 r6 = (rightExists & (1<<6)) ? vfloat4::loadu(geom->vertexPtr(v0[6]+2)) : vfloat4(inf);
    const vfloat4 r7 = (rightExists & (1<<7)) ? vfloat4::loadu(geom->vertexPtr(v0[7]+2)) : vfloat4(inf);
    transpose(r0,r1,r2,r3,r4,r5,r6,r7,pR.x,pR.y,pR.z,pR.w);
  }
  
  template<>
    __forceinline void LineMi<8>::gatheri(Vec4vf8& p0,
                                              Vec4vf8& p1,
                                              Vec4vf8& pL,
                                              Vec4vf8& pR,
                                              const LineSegments* geom,
                                              const int itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(v0[0],itime));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(v0[1],itime));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(v0[2],itime));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(v0[3],itime));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(v0[4],itime));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(v0[5],itime));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(v0[6],itime));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(v0[7],itime));
    transpose(a0,a1,a2,a3,a4,a5,a6,a7,p0.x,p0.y,p0.z,p0.w);
    
    const vfloat4 b0 = vfloat4::loadu(geom->vertexPtr(v0[0]+1,itime));
    const vfloat4 b1 = vfloat4::loadu(geom->vertexPtr(v0[1]+1,itime));
    const vfloat4 b2 = vfloat4::loadu(geom->vertexPtr(v0[2]+1,itime));
    const vfloat4 b3 = vfloat4::loadu(geom->vertexPtr(v0[3]+1,itime));
    const vfloat4 b4 = vfloat4::loadu(geom->vertexPtr(v0[4]+1,itime));
    const vfloat4 b5 = vfloat4::loadu(geom->vertexPtr(v0[5]+1,itime));
    const vfloat4 b6 = vfloat4::loadu(geom->vertexPtr(v0[6]+1,itime));
    const vfloat4 b7 = vfloat4::loadu(geom->vertexPtr(v0[7]+1,itime));
    transpose(b0,b1,b2,b3,b4,b5,b6,b7,p1.x,p1.y,p1.z,p1.w);
    
    const vfloat4 l0 = (leftExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]-1,itime)) : vfloat4(inf);
    const vfloat4 l1 = (leftExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]-1,itime)) : vfloat4(inf);
    const vfloat4 l2 = (leftExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]-1,itime)) : vfloat4(inf);
    const vfloat4 l3 = (leftExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]-1,itime)) : vfloat4(inf);
    const vfloat4 l4 = (leftExists & (1<<4)) ? vfloat4::loadu(geom->vertexPtr(v0[4]-1,itime)) : vfloat4(inf);
    const vfloat4 l5 = (leftExists & (1<<5)) ? vfloat4::loadu(geom->vertexPtr(v0[5]-1,itime)) : vfloat4(inf);
    const vfloat4 l6 = (leftExists & (1<<6)) ? vfloat4::loadu(geom->vertexPtr(v0[6]-1,itime)) : vfloat4(inf);
    const vfloat4 l7 = (leftExists & (1<<7)) ? vfloat4::loadu(geom->vertexPtr(v0[7]-1,itime)) : vfloat4(inf);
    transpose(l0,l1,l2,l3,l4,l5,l6,l7,pL.x,pL.y,pL.z,pL.w);
    
    const vfloat4 r0 = (rightExists & (1<<0)) ? vfloat4::loadu(geom->vertexPtr(v0[0]+2,itime)) : vfloat4(inf);
    const vfloat4 r1 = (rightExists & (1<<1)) ? vfloat4::loadu(geom->vertexPtr(v0[1]+2,itime)) : vfloat4(inf);
    const vfloat4 r2 = (rightExists & (1<<2)) ? vfloat4::loadu(geom->vertexPtr(v0[2]+2,itime)) : vfloat4(inf);
    const vfloat4 r3 = (rightExists & (1<<3)) ? vfloat4::loadu(geom->vertexPtr(v0[3]+2,itime)) : vfloat4(inf);
    const vfloat4 r4 = (rightExists & (1<<4)) ? vfloat4::loadu(geom->vertexPtr(v0[4]+2,itime)) : vfloat4(inf);
    const vfloat4 r5 = (rightExists & (1<<5)) ? vfloat4::loadu(geom->vertexPtr(v0[5]+2,itime)) : vfloat4(inf);
    const vfloat4 r6 = (rightExists & (1<<6)) ? vfloat4::loadu(geom->vertexPtr(v0[6]+2,itime)) : vfloat4(inf);
    const vfloat4 r7 = (rightExists & (1<<7)) ? vfloat4::loadu(geom->vertexPtr(v0[7]+2,itime)) : vfloat4(inf);
    transpose(r0,r1,r2,r3,r4,r5,r6,r7,pR.x,pR.y,pR.z,pR.w);
  }
  
  template<>
    __forceinline void LineMi<8>::gather(Vec4vf8& p0,
                                              Vec4vf8& p1,
                                              Vec4vf8& pL,
                                              Vec4vf8& pR,
                                              const LineSegments* geom,
                                              float time) const
  {
    float ftime;
    const int itime = geom->timeSegment(time, ftime);
    
    Vec4vf8 a0,a1,aL,aR;
    gatheri(a0,a1,aL,aR,geom,itime);
    Vec4vf8 b0,b1,bL,bR;
    gatheri(b0,b1,bL,bR,geom,itime+1);
    p0 = lerp(a0,b0,vfloat8(ftime));
    p1 = lerp(a1,b1,vfloat8(ftime));
    pL = lerp(aL,bL,vfloat8(ftime));
    pR = lerp(aR,bR,vfloat8(ftime));
    
    pL = select(vboolf4(leftExists), pL, Vec4vf8(inf));
    pR = select(vboolf4(rightExists), pR, Vec4vf8(inf));
  }

  template<>
    __forceinline void LineMi<8>::gather(Vec4vf8& p0,
                                         Vec4vf8& p1,
                                         vbool8& cL,
                                         vbool8& cR,
                                         const LineSegments* geom) const
  {
    gather(p0,p1,geom);
    cL = !vbool8(leftExists);
    cR = !vbool8(rightExists);
  }
  
  template<>
    __forceinline void LineMi<8>::gatheri(Vec4vf8& p0,
                                              Vec4vf8& p1,
                                              vbool8& cL,
                                              vbool8& cR,
                                              const LineSegments* geom,
                                              const int itime) const
  {
    gatheri(p0,p1,geom,itime);
    cL = !vbool8(leftExists);
    cR = !vbool8(rightExists);
  }
  
  template<>
    __forceinline void LineMi<8>::gather(Vec4vf8& p0,
                                              Vec4vf8& p1,
                                              vbool8& cL,
                                              vbool8& cR,
                                              const LineSegments* geom,
                                              float time) const
  {
    float ftime;
    const int itime = geom->timeSegment(time, ftime);
    
    Vec4vf8 a0,a1;
    gatheri(a0,a1,geom,itime);
    Vec4vf8 b0,b1;
    gatheri(b0,b1,geom,itime+1);
    p0 = lerp(a0,b0,vfloat8(ftime));
    p1 = lerp(a1,b1,vfloat8(ftime));
    cL = !vbool8(leftExists);
    cR = !vbool8(rightExists);
  }
  
#endif
  
  template<int M>
  typename LineMi<M>::Type LineMi<M>::type;

  typedef LineMi<4> Line4i;
  typedef LineMi<8> Line8i;
}
