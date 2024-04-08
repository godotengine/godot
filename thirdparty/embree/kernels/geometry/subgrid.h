// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "../common/scene_grid_mesh.h"
#include "../bvh/bvh.h"

namespace embree
{
    /* Stores M quads from an indexed face set */
      struct SubGrid
      {
        /* Virtual interface to query information about the quad type */
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

        /* Returns maximum number of stored quads */
        static __forceinline size_t max_size() { return 1; }

        /* Returns required number of primitive blocks for N primitives */
        static __forceinline size_t blocks(size_t N) { return (N+max_size()-1)/max_size(); }

      public:

        /* Default constructor */
        __forceinline SubGrid() {  }

        /* Construction from vertices and IDs */
        __forceinline SubGrid(const unsigned int x,
                              const unsigned int y,
                              const unsigned int geomID,
                              const unsigned int primID)
          : _x(x), _y(y), _geomID(geomID), _primID(primID)
        {
        }

        __forceinline bool invalid3x3X() const { return (unsigned int)_x & (1<<15); }
        __forceinline bool invalid3x3Y() const { return (unsigned int)_y & (1<<15); }

        /* Gather the quads */
        __forceinline void gather(Vec3vf4& p0,
                                  Vec3vf4& p1,
                                  Vec3vf4& p2,
                                  Vec3vf4& p3,
                                  const GridMesh* const mesh,
                                  const GridMesh::Grid &g) const
        {
          /* first quad always valid */
          const size_t vtxID00 = g.startVtxID + x() + y() * g.lineVtxOffset;
          const size_t vtxID01 = vtxID00 + 1;
          const vfloat4 vtx00  = vfloat4::loadu(mesh->vertexPtr(vtxID00));
          const vfloat4 vtx01  = vfloat4::loadu(mesh->vertexPtr(vtxID01));
          const size_t vtxID10 = vtxID00 + g.lineVtxOffset;
          const size_t vtxID11 = vtxID01 + g.lineVtxOffset;
          const vfloat4 vtx10  = vfloat4::loadu(mesh->vertexPtr(vtxID10));
          const vfloat4 vtx11  = vfloat4::loadu(mesh->vertexPtr(vtxID11));

          /* deltaX => vtx02, vtx12 */
          const size_t deltaX  = invalid3x3X() ? 0 : 1;
          const size_t vtxID02 = vtxID01 + deltaX;       
          const vfloat4 vtx02  = vfloat4::loadu(mesh->vertexPtr(vtxID02));
          const size_t vtxID12 = vtxID11 + deltaX;       
          const vfloat4 vtx12  = vfloat4::loadu(mesh->vertexPtr(vtxID12));

          /* deltaY => vtx20, vtx21 */
          const size_t deltaY  = invalid3x3Y() ? 0 : g.lineVtxOffset;
          const size_t vtxID20 = vtxID10 + deltaY;
          const size_t vtxID21 = vtxID11 + deltaY;
          const vfloat4 vtx20  = vfloat4::loadu(mesh->vertexPtr(vtxID20));
          const vfloat4 vtx21  = vfloat4::loadu(mesh->vertexPtr(vtxID21));

          /* deltaX/deltaY => vtx22 */
          const size_t vtxID22 = vtxID11 + deltaX + deltaY;       
          const vfloat4 vtx22  = vfloat4::loadu(mesh->vertexPtr(vtxID22));

          transpose(vtx00,vtx01,vtx11,vtx10,p0.x,p0.y,p0.z);
          transpose(vtx01,vtx02,vtx12,vtx11,p1.x,p1.y,p1.z);
          transpose(vtx11,vtx12,vtx22,vtx21,p2.x,p2.y,p2.z);
          transpose(vtx10,vtx11,vtx21,vtx20,p3.x,p3.y,p3.z);                    
        }

        template<typename T>
        __forceinline vfloat4 getVertexMB(const GridMesh* const mesh, const size_t offset, const size_t itime, const float ftime) const
        {
          const T v0 = T::loadu(mesh->vertexPtr(offset,itime+0));
          const T v1 = T::loadu(mesh->vertexPtr(offset,itime+1));
          return lerp(v0,v1,ftime);
        }

        /* Gather the quads */
        __forceinline void gatherMB(Vec3vf4& p0,
                                    Vec3vf4& p1,
                                    Vec3vf4& p2,
                                    Vec3vf4& p3,
                                    const GridMesh* const mesh,
                                    const GridMesh::Grid &g,
                                    const size_t itime, 
                                    const float ftime) const
        {
          /* first quad always valid */
          const size_t vtxID00 = g.startVtxID + x() + y() * g.lineVtxOffset;
          const size_t vtxID01 = vtxID00 + 1;
          const vfloat4 vtx00  = getVertexMB<vfloat4>(mesh,vtxID00,itime,ftime);
          const vfloat4 vtx01  = getVertexMB<vfloat4>(mesh,vtxID01,itime,ftime);
          const size_t vtxID10 = vtxID00 + g.lineVtxOffset;
          const size_t vtxID11 = vtxID01 + g.lineVtxOffset;
          const vfloat4 vtx10  = getVertexMB<vfloat4>(mesh,vtxID10,itime,ftime);
          const vfloat4 vtx11  = getVertexMB<vfloat4>(mesh,vtxID11,itime,ftime);

          /* deltaX => vtx02, vtx12 */
          const size_t deltaX  = invalid3x3X() ? 0 : 1;
          const size_t vtxID02 = vtxID01 + deltaX;       
          const vfloat4 vtx02  = getVertexMB<vfloat4>(mesh,vtxID02,itime,ftime);
          const size_t vtxID12 = vtxID11 + deltaX;       
          const vfloat4 vtx12  = getVertexMB<vfloat4>(mesh,vtxID12,itime,ftime);

          /* deltaY => vtx20, vtx21 */
          const size_t deltaY  = invalid3x3Y() ? 0 : g.lineVtxOffset;
          const size_t vtxID20 = vtxID10 + deltaY;
          const size_t vtxID21 = vtxID11 + deltaY;
          const vfloat4 vtx20  = getVertexMB<vfloat4>(mesh,vtxID20,itime,ftime);
          const vfloat4 vtx21  = getVertexMB<vfloat4>(mesh,vtxID21,itime,ftime);

          /* deltaX/deltaY => vtx22 */
          const size_t vtxID22 = vtxID11 + deltaX + deltaY;       
          const vfloat4 vtx22  = getVertexMB<vfloat4>(mesh,vtxID22,itime,ftime);

          transpose(vtx00,vtx01,vtx11,vtx10,p0.x,p0.y,p0.z);
          transpose(vtx01,vtx02,vtx12,vtx11,p1.x,p1.y,p1.z);
          transpose(vtx11,vtx12,vtx22,vtx21,p2.x,p2.y,p2.z);
          transpose(vtx10,vtx11,vtx21,vtx20,p3.x,p3.y,p3.z);                    
        }



        /* Gather the quads */
        __forceinline void gather(Vec3vf4& p0,
                                  Vec3vf4& p1,
                                  Vec3vf4& p2,
                                  Vec3vf4& p3,
                                  const Scene *const scene) const
        {
          const GridMesh* const mesh = scene->get<GridMesh>(geomID());
          const GridMesh::Grid &g    = mesh->grid(primID());
          gather(p0,p1,p2,p3,mesh,g);
        }

        /* Gather the quads in the motion blur case */
        __forceinline void gatherMB(Vec3vf4& p0,
                                    Vec3vf4& p1,
                                    Vec3vf4& p2,
                                    Vec3vf4& p3,
                                    const Scene *const scene,
                                    const size_t itime, 
                                    const float ftime) const
        {
          const GridMesh* const mesh = scene->get<GridMesh>(geomID());
          const GridMesh::Grid &g    = mesh->grid(primID());
          gatherMB(p0,p1,p2,p3,mesh,g,itime,ftime);
        }

        /* Gather the quads */
        __forceinline void gather(Vec3fa vtx[16], const Scene *const scene) const
        {
          const GridMesh* mesh     = scene->get<GridMesh>(geomID());
          const GridMesh::Grid &g  = mesh->grid(primID());

          /* first quad always valid */
          const size_t vtxID00 = g.startVtxID + x() + y() * g.lineVtxOffset;
          const size_t vtxID01 = vtxID00 + 1;
          const Vec3fa vtx00  = Vec3fa::loadu(mesh->vertexPtr(vtxID00));
          const Vec3fa vtx01  = Vec3fa::loadu(mesh->vertexPtr(vtxID01));
          const size_t vtxID10 = vtxID00 + g.lineVtxOffset;
          const size_t vtxID11 = vtxID01 + g.lineVtxOffset;
          const Vec3fa vtx10  = Vec3fa::loadu(mesh->vertexPtr(vtxID10));
          const Vec3fa vtx11  = Vec3fa::loadu(mesh->vertexPtr(vtxID11));

          /* deltaX => vtx02, vtx12 */
          const size_t deltaX  = invalid3x3X() ? 0 : 1;
          const size_t vtxID02 = vtxID01 + deltaX;       
          const Vec3fa vtx02  = Vec3fa::loadu(mesh->vertexPtr(vtxID02));
          const size_t vtxID12 = vtxID11 + deltaX;       
          const Vec3fa vtx12  = Vec3fa::loadu(mesh->vertexPtr(vtxID12));

          /* deltaY => vtx20, vtx21 */
          const size_t deltaY  = invalid3x3Y() ? 0 : g.lineVtxOffset;
          const size_t vtxID20 = vtxID10 + deltaY;
          const size_t vtxID21 = vtxID11 + deltaY;
          const Vec3fa vtx20  = Vec3fa::loadu(mesh->vertexPtr(vtxID20));
          const Vec3fa vtx21  = Vec3fa::loadu(mesh->vertexPtr(vtxID21));

          /* deltaX/deltaY => vtx22 */
          const size_t vtxID22 = vtxID11 + deltaX + deltaY;       
          const Vec3fa vtx22  = Vec3fa::loadu(mesh->vertexPtr(vtxID22));

          vtx[ 0] = vtx00; vtx[ 1] = vtx01; vtx[ 2] = vtx11; vtx[ 3] = vtx10;
          vtx[ 4] = vtx01; vtx[ 5] = vtx02; vtx[ 6] = vtx12; vtx[ 7] = vtx11;
          vtx[ 8] = vtx10; vtx[ 9] = vtx11; vtx[10] = vtx21; vtx[11] = vtx20;
          vtx[12] = vtx11; vtx[13] = vtx12; vtx[14] = vtx22; vtx[15] = vtx21;
        }

        /* Gather the quads */
        __forceinline void gatherMB(vfloat4 vtx[16], const Scene *const scene, const size_t itime, const float ftime) const
        {
          const GridMesh* mesh     = scene->get<GridMesh>(geomID());
          const GridMesh::Grid &g  = mesh->grid(primID());

          /* first quad always valid */
          const size_t vtxID00 = g.startVtxID + x() + y() * g.lineVtxOffset;
          const size_t vtxID01 = vtxID00 + 1;
          const vfloat4 vtx00  = getVertexMB<vfloat4>(mesh,vtxID00,itime,ftime);
          const vfloat4 vtx01  = getVertexMB<vfloat4>(mesh,vtxID01,itime,ftime);
          const size_t vtxID10 = vtxID00 + g.lineVtxOffset;
          const size_t vtxID11 = vtxID01 + g.lineVtxOffset;
          const vfloat4 vtx10  = getVertexMB<vfloat4>(mesh,vtxID10,itime,ftime);
          const vfloat4 vtx11  = getVertexMB<vfloat4>(mesh,vtxID11,itime,ftime);

          /* deltaX => vtx02, vtx12 */
          const size_t deltaX  = invalid3x3X() ? 0 : 1;
          const size_t vtxID02 = vtxID01 + deltaX;       
          const vfloat4 vtx02  = getVertexMB<vfloat4>(mesh,vtxID02,itime,ftime);
          const size_t vtxID12 = vtxID11 + deltaX;       
          const vfloat4 vtx12  = getVertexMB<vfloat4>(mesh,vtxID12,itime,ftime);

          /* deltaY => vtx20, vtx21 */
          const size_t deltaY  = invalid3x3Y() ? 0 : g.lineVtxOffset;
          const size_t vtxID20 = vtxID10 + deltaY;
          const size_t vtxID21 = vtxID11 + deltaY;
          const vfloat4 vtx20  = getVertexMB<vfloat4>(mesh,vtxID20,itime,ftime);
          const vfloat4 vtx21  = getVertexMB<vfloat4>(mesh,vtxID21,itime,ftime);

          /* deltaX/deltaY => vtx22 */
          const size_t vtxID22 = vtxID11 + deltaX + deltaY;       
          const vfloat4 vtx22  = getVertexMB<vfloat4>(mesh,vtxID22,itime,ftime);

          vtx[ 0] = vtx00; vtx[ 1] = vtx01; vtx[ 2] = vtx11; vtx[ 3] = vtx10;
          vtx[ 4] = vtx01; vtx[ 5] = vtx02; vtx[ 6] = vtx12; vtx[ 7] = vtx11;
          vtx[ 8] = vtx10; vtx[ 9] = vtx11; vtx[10] = vtx21; vtx[11] = vtx20;
          vtx[12] = vtx11; vtx[13] = vtx12; vtx[14] = vtx22; vtx[15] = vtx21;
        }        
          

        /* Calculate the bounds of the subgrid */
        __forceinline const BBox3fa bounds(const Scene *const scene, const size_t itime=0) const
        {
          BBox3fa bounds = empty;
          FATAL("not implemented yet");
          return bounds;
        }

        /* Calculate the linear bounds of the primitive */
        __forceinline LBBox3fa linearBounds(const Scene* const scene, const size_t itime)
        {
          return LBBox3fa(bounds(scene,itime+0),bounds(scene,itime+1));
        }

        __forceinline LBBox3fa linearBounds(const Scene *const scene, size_t itime, size_t numTimeSteps)
        {
          LBBox3fa allBounds = empty;
          FATAL("not implemented yet");
          return allBounds;
        }

        __forceinline LBBox3fa linearBounds(const Scene *const scene, const BBox1f time_range)
        {
          LBBox3fa allBounds = empty;
          FATAL("not implemented yet");
          return allBounds;
        }


        friend embree_ostream operator<<(embree_ostream cout, const SubGrid& sg) {
          return cout << "SubGrid " << " ( x = " << sg.x() << ", y = " << sg.y() << ", geomID = " << sg.geomID() << ", primID = " << sg.primID() << ", invalid3x3X() " << (int)sg.invalid3x3X() << ", invalid3x3Y() " << (int)sg.invalid3x3Y();
        }

        __forceinline unsigned int geomID() const { return _geomID; }
        __forceinline unsigned int primID() const { return _primID; }
        __forceinline unsigned int x() const { return (unsigned int)_x & 0x7fff; }
        __forceinline unsigned int y() const { return (unsigned int)_y & 0x7fff; }

      private:
        unsigned short _x;
        unsigned short _y;
        unsigned int _geomID;    // geometry ID of mesh
        unsigned int _primID;    // primitive ID of primitive inside mesh
      };

      struct SubGridID {
        unsigned short x;
        unsigned short y;
        unsigned int primID;
        
        __forceinline SubGridID() {}
        __forceinline SubGridID(const unsigned int x, const unsigned int y, const unsigned int primID) :
        x(x), y(y), primID(primID) {}
        
      };
      
      /* QuantizedBaseNode as large subgrid leaf */
      template<int N>
      struct SubGridQBVHN
      {
        /* Virtual interface to query information about the quad type */
        struct Type : public PrimitiveType
        {
          const char* name() const;
          size_t sizeActive(const char* This) const;
          size_t sizeTotal(const char* This) const;
          size_t getBytes(const char* This) const;
        };
        static Type type;

      public:

        __forceinline size_t size() const
        {
          for (size_t i=0;i<N;i++)
            if (primID(i) == -1) return i;
          return N;
        }

      __forceinline void clear() {
        for (size_t i=0;i<N;i++)
          subgridIDs[i] = SubGridID(0,0,(unsigned int)-1);
        qnode.clear();
      }

        /* Default constructor */
        __forceinline SubGridQBVHN() {  }

        /* Construction from vertices and IDs */
        __forceinline SubGridQBVHN(const unsigned int x[N],
                                   const unsigned int y[N],
                                   const unsigned int primID[N],
                                   const BBox3fa * const subGridBounds,
                                   const unsigned int geomID,
                                   const unsigned int items)
        {
          clear();
          _geomID = geomID;

          __aligned(64) typename BVHN<N>::AABBNode node;
          node.clear();          
          for (size_t i=0;i<items;i++)
          {
            subgridIDs[i] = SubGridID(x[i],y[i],primID[i]);
            node.setBounds(i,subGridBounds[i]);
          }
          qnode.init_dim(node);
        }

        __forceinline unsigned int geomID() const { return _geomID; }
        __forceinline unsigned int primID(const size_t i) const { assert(i < N); return subgridIDs[i].primID; }
        __forceinline unsigned int x(const size_t i) const { assert(i < N); return subgridIDs[i].x; }
        __forceinline unsigned int y(const size_t i) const { assert(i < N); return subgridIDs[i].y; }

        __forceinline SubGrid subgrid(const size_t i) const {
          assert(i < N);
          assert(primID(i) != -1);
          return SubGrid(x(i),y(i),geomID(),primID(i));
        }

      public:
        SubGridID subgridIDs[N];

        typename BVHN<N>::QuantizedBaseNode qnode;

        unsigned int _geomID;    // geometry ID of mesh


        friend embree_ostream operator<<(embree_ostream cout, const SubGridQBVHN& sg) {
          cout << "SubGridQBVHN " << embree_endl;
          for (size_t i=0;i<N;i++)
            cout << i << " ( x = " << sg.subgridIDs[i].x << ", y = " << sg.subgridIDs[i].y << ", primID = " << sg.subgridIDs[i].primID << " )" << embree_endl;
          cout << "geomID " << sg._geomID << embree_endl;
          cout << "lowerX " << sg.qnode.dequantizeLowerX() << embree_endl;
          cout << "upperX " << sg.qnode.dequantizeUpperX() << embree_endl;
          cout << "lowerY " << sg.qnode.dequantizeLowerY() << embree_endl;
          cout << "upperY " << sg.qnode.dequantizeUpperY() << embree_endl;
          cout << "lowerZ " << sg.qnode.dequantizeLowerZ() << embree_endl;
          cout << "upperZ " << sg.qnode.dequantizeUpperZ() << embree_endl;
          return cout;
        }

      };

      template<int N>
        typename SubGridQBVHN<N>::Type SubGridQBVHN<N>::type;

      typedef SubGridQBVHN<4> SubGridQBVH4;
      typedef SubGridQBVHN<8> SubGridQBVH8;


      


      /* QuantizedBaseNode as large subgrid leaf */
      template<int N>
      struct SubGridMBQBVHN
      {
        /* Virtual interface to query information about the quad type */
        struct Type : public PrimitiveType
        {
          const char* name() const;
          size_t sizeActive(const char* This) const;
          size_t sizeTotal(const char* This) const;
          size_t getBytes(const char* This) const;
        };
        static Type type;

      public:

        __forceinline size_t size() const
        {
          for (size_t i=0;i<N;i++)
            if (primID(i) == -1) return i;
          return N;
        }

      __forceinline void clear() {
        for (size_t i=0;i<N;i++)
          subgridIDs[i] = SubGridID(0,0,(unsigned int)-1);
        qnode.clear();
      }

        /* Default constructor */
        __forceinline SubGridMBQBVHN() {  }

        /* Construction from vertices and IDs */
        __forceinline SubGridMBQBVHN(const unsigned int x[N],
                                     const unsigned int y[N],
                                     const unsigned int primID[N],
                                     const BBox3fa * const subGridBounds0,
                                     const BBox3fa * const subGridBounds1,
                                     const unsigned int geomID,
                                     const float toffset,
                                     const float tscale,
                                     const unsigned int items)
        {
          clear();
          _geomID = geomID;
          time_offset = toffset;
          time_scale  = tscale;

          __aligned(64) typename BVHN<N>::AABBNode node0,node1;
          node0.clear();          
          node1.clear();          
          for (size_t i=0;i<items;i++)
          {
            subgridIDs[i] = SubGridID(x[i],y[i],primID[i]);
            node0.setBounds(i,subGridBounds0[i]);
            node1.setBounds(i,subGridBounds1[i]);
          }
          qnode.node0.init_dim(node0);
          qnode.node1.init_dim(node1);
        }

        __forceinline unsigned int geomID() const { return _geomID; }
        __forceinline unsigned int primID(const size_t i) const { assert(i < N); return subgridIDs[i].primID; }
        __forceinline unsigned int x(const size_t i) const { assert(i < N); return subgridIDs[i].x; }
        __forceinline unsigned int y(const size_t i) const { assert(i < N); return subgridIDs[i].y; }

        __forceinline SubGrid subgrid(const size_t i) const {
          assert(i < N);
          assert(primID(i) != -1);
          return SubGrid(x(i),y(i),geomID(),primID(i));
        }

        __forceinline float adjustTime(const float t) const { return time_scale * (t-time_offset); }

        template<int K>
        __forceinline vfloat<K> adjustTime(const vfloat<K> &t) const { return time_scale * (t-time_offset); }

      public:
        SubGridID subgridIDs[N];

        typename BVHN<N>::QuantizedBaseNodeMB qnode;

        float time_offset;
        float time_scale;
        unsigned int _geomID;    // geometry ID of mesh


        friend embree_ostream operator<<(embree_ostream cout, const SubGridMBQBVHN& sg) {
          cout << "SubGridMBQBVHN " << embree_endl;
          for (size_t i=0;i<N;i++)
            cout << i << " ( x = " << sg.subgridIDs[i].x << ", y = " << sg.subgridIDs[i].y << ", primID = " << sg.subgridIDs[i].primID << " )" << embree_endl;
          cout << "geomID      " << sg._geomID << embree_endl;
          cout << "time_offset " << sg.time_offset << embree_endl;
          cout << "time_scale  " << sg.time_scale << embree_endl;         
          cout << "lowerX " << sg.qnode.node0.dequantizeLowerX() << embree_endl;
          cout << "upperX " << sg.qnode.node0.dequantizeUpperX() << embree_endl;
          cout << "lowerY " << sg.qnode.node0.dequantizeLowerY() << embree_endl;
          cout << "upperY " << sg.qnode.node0.dequantizeUpperY() << embree_endl;
          cout << "lowerZ " << sg.qnode.node0.dequantizeLowerZ() << embree_endl;
          cout << "upperZ " << sg.qnode.node0.dequantizeUpperZ() << embree_endl;
          cout << "lowerX " << sg.qnode.node1.dequantizeLowerX() << embree_endl;
          cout << "upperX " << sg.qnode.node1.dequantizeUpperX() << embree_endl;
          cout << "lowerY " << sg.qnode.node1.dequantizeLowerY() << embree_endl;
          cout << "upperY " << sg.qnode.node1.dequantizeUpperY() << embree_endl;
          cout << "lowerZ " << sg.qnode.node1.dequantizeLowerZ() << embree_endl;
          cout << "upperZ " << sg.qnode.node1.dequantizeUpperZ() << embree_endl;
          return cout;
        }

      };
}
