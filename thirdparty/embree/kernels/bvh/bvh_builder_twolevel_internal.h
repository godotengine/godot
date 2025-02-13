// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"
#include "../geometry/triangle.h"
#include "../geometry/trianglev.h"
#include "../geometry/trianglei.h"
#include "../geometry/quadv.h"
#include "../geometry/quadi.h"
#include "../geometry/object.h"
#include "../geometry/instance.h"
#include "../geometry/instance_array.h"

namespace embree
{
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4MeshBuilderMortonGeneral,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4MeshBuilderSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4MeshRefitSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4vMeshBuilderMortonGeneral,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4vMeshBuilderSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4vMeshRefitSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4iMeshBuilderMortonGeneral,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4iMeshBuilderSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Triangle4iMeshRefitSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Quad4vMeshBuilderMortonGeneral,void* COMMA QuadMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Quad4vMeshBuilderSAH,void* COMMA QuadMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4Quad4vMeshRefitSAH,void* COMMA QuadMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4VirtualMeshBuilderMortonGeneral,void* COMMA UserGeometry* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4VirtualMeshBuilderSAH,void* COMMA UserGeometry* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4VirtualMeshRefitSAH,void* COMMA UserGeometry* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4InstanceMeshBuilderMortonGeneral,void* COMMA Instance* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4InstanceMeshBuilderSAH,void* COMMA Instance* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4InstanceMeshRefitSAH,void* COMMA Instance* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t)
  DECLARE_ISA_FUNCTION(Builder*,BVH4InstanceArrayMeshBuilderMortonGeneral,void* COMMA InstanceArray* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4InstanceArrayMeshBuilderSAH,void* COMMA InstanceArray* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH4InstanceArrayMeshRefitSAH,void* COMMA InstanceArray* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t)
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4MeshBuilderMortonGeneral,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4MeshBuilderSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4MeshRefitSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4vMeshBuilderMortonGeneral,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4vMeshBuilderSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4vMeshRefitSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4iMeshBuilderMortonGeneral,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4iMeshBuilderSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Triangle4iMeshRefitSAH,void* COMMA TriangleMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Quad4vMeshBuilderMortonGeneral,void* COMMA QuadMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Quad4vMeshBuilderSAH,void* COMMA QuadMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8Quad4vMeshRefitSAH,void* COMMA QuadMesh* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8VirtualMeshBuilderMortonGeneral,void* COMMA UserGeometry* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8VirtualMeshBuilderSAH,void* COMMA UserGeometry* COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8VirtualMeshRefitSAH,void* COMMA UserGeometry* COMMA unsigned int COMMA size_t); 
  DECLARE_ISA_FUNCTION(Builder*,BVH8InstanceMeshBuilderMortonGeneral,void* COMMA Instance* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8InstanceMeshBuilderSAH,void* COMMA Instance* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8InstanceMeshRefitSAH,void* COMMA Instance* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t) 
  DECLARE_ISA_FUNCTION(Builder*,BVH8InstanceArrayMeshBuilderMortonGeneral,void* COMMA InstanceArray* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8InstanceArrayMeshBuilderSAH,void* COMMA InstanceArray* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t);
  DECLARE_ISA_FUNCTION(Builder*,BVH8InstanceArrayMeshRefitSAH,void* COMMA InstanceArray* COMMA Geometry::GTypeMask COMMA unsigned int COMMA size_t) 
  
  namespace isa
  {

    namespace __internal_two_level_builder__ {

      template<int N, typename Mesh, typename Primitive>
      struct MortonBuilder {};
      template<>
      struct MortonBuilder<4,TriangleMesh,Triangle4> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4MeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<4,TriangleMesh,Triangle4v> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4vMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<4,TriangleMesh,Triangle4i> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4iMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<4,QuadMesh,Quad4v> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, QuadMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Quad4vMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<4,UserGeometry,Object> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, UserGeometry* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4VirtualMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<4,Instance,InstancePrimitive> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, Instance* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH4InstanceMeshBuilderMortonGeneral(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct MortonBuilder<4,InstanceArray,InstanceArrayPrimitive> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, InstanceArray* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH4InstanceArrayMeshBuilderMortonGeneral(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,TriangleMesh,Triangle4> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4MeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,TriangleMesh,Triangle4v> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4vMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,TriangleMesh,Triangle4i> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4iMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,QuadMesh,Quad4v> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, QuadMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Quad4vMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,UserGeometry,Object> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, UserGeometry* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8VirtualMeshBuilderMortonGeneral(bvh,mesh,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,Instance,InstancePrimitive> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, Instance* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH8InstanceMeshBuilderMortonGeneral(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct MortonBuilder<8,InstanceArray,InstanceArrayPrimitive> {
        MortonBuilder () {}
        Builder* operator () (void* bvh, InstanceArray* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH8InstanceArrayMeshBuilderMortonGeneral(bvh,mesh,gtype,geomID,0);}
      };

      template<int N, typename Mesh, typename Primitive>
      struct SAHBuilder {};
      template<>
      struct SAHBuilder<4,TriangleMesh,Triangle4> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4MeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<4,TriangleMesh,Triangle4v> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4vMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<4,TriangleMesh,Triangle4i> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4iMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<4,QuadMesh,Quad4v> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, QuadMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Quad4vMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<4,UserGeometry,Object> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, UserGeometry* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4VirtualMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<4,Instance,InstancePrimitive> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, Instance* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH4InstanceMeshBuilderSAH(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct SAHBuilder<4,InstanceArray,InstanceArrayPrimitive> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, InstanceArray* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH4InstanceArrayMeshBuilderSAH(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,TriangleMesh,Triangle4> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4MeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,TriangleMesh,Triangle4v> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4vMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,TriangleMesh,Triangle4i> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4iMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,QuadMesh,Quad4v> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, QuadMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Quad4vMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,UserGeometry,Object> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, UserGeometry* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8VirtualMeshBuilderSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,Instance,InstancePrimitive> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, Instance* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH8InstanceMeshBuilderSAH(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct SAHBuilder<8,InstanceArray,InstanceArrayPrimitive> {
        SAHBuilder () {}
        Builder* operator () (void* bvh, InstanceArray* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH8InstanceArrayMeshBuilderSAH(bvh,mesh,gtype,geomID,0);}
      };

      template<int N, typename Mesh, typename Primitive>
      struct RefitBuilder {};
      template<>
      struct RefitBuilder<4,TriangleMesh,Triangle4> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4MeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<4,TriangleMesh,Triangle4v> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4vMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<4,TriangleMesh,Triangle4i> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Triangle4iMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<4,QuadMesh,Quad4v> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, QuadMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4Quad4vMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<4,UserGeometry,Object> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, UserGeometry* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH4VirtualMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<4,Instance,InstancePrimitive> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, Instance* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH4InstanceMeshRefitSAH(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct RefitBuilder<4,InstanceArray,InstanceArrayPrimitive> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, InstanceArray* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH4InstanceArrayMeshRefitSAH(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,TriangleMesh,Triangle4> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4MeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,TriangleMesh,Triangle4v> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4vMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,TriangleMesh,Triangle4i> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, TriangleMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Triangle4iMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,QuadMesh,Quad4v> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, QuadMesh* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8Quad4vMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,UserGeometry,Object> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, UserGeometry* mesh, size_t geomID, Geometry::GTypeMask /*gtype*/) { return BVH8VirtualMeshRefitSAH(bvh,mesh,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,Instance,InstancePrimitive> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, Instance* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH8InstanceMeshRefitSAH(bvh,mesh,gtype,geomID,0);}
      };
      template<>
      struct RefitBuilder<8,InstanceArray,InstanceArrayPrimitive> {
        RefitBuilder () {}
        Builder* operator () (void* bvh, InstanceArray* mesh, size_t geomID, Geometry::GTypeMask gtype) { return BVH8InstanceArrayMeshRefitSAH(bvh,mesh,gtype,geomID,0);}
      };

      template<int N, typename Mesh, typename Primitive>
      struct MeshBuilder {
        MeshBuilder () {}
        void operator () (void* bvh, Mesh* mesh, size_t geomID, Geometry::GTypeMask gtype, bool useMortonBuilder, Builder*& builder) {
          if(useMortonBuilder) {
            builder = MortonBuilder<N,Mesh,Primitive>()(bvh,mesh,geomID,gtype);
            return;
          }
          switch (mesh->quality) {
            case RTC_BUILD_QUALITY_LOW:    builder = MortonBuilder<N,Mesh,Primitive>()(bvh,mesh,geomID,gtype); break;
            case RTC_BUILD_QUALITY_MEDIUM:
            case RTC_BUILD_QUALITY_HIGH:   builder = SAHBuilder<N,Mesh,Primitive>()(bvh,mesh,geomID,gtype); break;
            case RTC_BUILD_QUALITY_REFIT:  builder = RefitBuilder<N,Mesh,Primitive>()(bvh,mesh,geomID,gtype); break;
            default: throw_RTCError(RTC_ERROR_UNKNOWN,"invalid build quality");
          }
        }
      };
    }
  }
}