// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_factory.h"

namespace embree
{
  /*! BVH4 instantiations */
  class BVH4Factory : public BVHFactory
  {
  public:
    BVH4Factory(int bfeatures, int ifeatures);

  public:
    Accel* BVH4OBBVirtualCurve4i(Scene* scene, IntersectVariant ivariant);
    Accel* BVH4OBBVirtualCurve4v(Scene* scene, IntersectVariant ivariant);
    Accel* BVH4OBBVirtualCurve8i(Scene* scene, IntersectVariant ivariant);
    Accel* BVH4OBBVirtualCurve4iMB(Scene* scene, IntersectVariant ivariant);
    Accel* BVH4OBBVirtualCurve8iMB(Scene* scene, IntersectVariant ivariant);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector4i);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector8i);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector4v);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector8v);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector4iMB);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector8iMB);
        
    Accel* BVH4Triangle4   (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH4Triangle4v  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::ROBUST);
    Accel* BVH4Triangle4i  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH4Triangle4vMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH4Triangle4iMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);

    Accel* BVH4Quad4v  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH4Quad4i  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH4Quad4iMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);

    Accel* BVH4QuantizedTriangle4i(Scene* scene);
    Accel* BVH4QuantizedQuad4i(Scene* scene);
 
    Accel* BVH4SubdivPatch1(Scene* scene);
    Accel* BVH4SubdivPatch1MB(Scene* scene);

    Accel* BVH4UserGeometry(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC);
    Accel* BVH4UserGeometryMB(Scene* scene);

    Accel* BVH4Instance(Scene* scene, bool isExpensive, BuildVariant bvariant = BuildVariant::STATIC);
    Accel* BVH4InstanceMB(Scene* scene, bool isExpensive);

    Accel* BVH4InstanceArray(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC);
    Accel* BVH4InstanceArrayMB(Scene* scene);

    Accel* BVH4Grid(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH4GridMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);

  private:
    void selectBuilders(int features);
    void selectIntersectors(int features);
    
  private:
    Accel::Intersectors BVH4OBBVirtualCurveIntersectors(BVH4* bvh, VirtualCurveIntersector* leafIntersector, IntersectVariant ivariant);
    Accel::Intersectors BVH4OBBVirtualCurveIntersectorsMB(BVH4* bvh, VirtualCurveIntersector* leafIntersector, IntersectVariant ivariant);
    
    Accel::Intersectors BVH4Triangle4Intersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4Triangle4vIntersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4Triangle4iIntersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4Triangle4iMBIntersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4Triangle4vMBIntersectors(BVH4* bvh, IntersectVariant ivariant);

    Accel::Intersectors BVH4Quad4vIntersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4Quad4iIntersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4Quad4iMBIntersectors(BVH4* bvh, IntersectVariant ivariant);

    Accel::Intersectors QBVH4Quad4iIntersectors(BVH4* bvh);
    Accel::Intersectors QBVH4Triangle4iIntersectors(BVH4* bvh);

    Accel::Intersectors BVH4UserGeometryIntersectors(BVH4* bvh);
    Accel::Intersectors BVH4UserGeometryMBIntersectors(BVH4* bvh);

    Accel::Intersectors BVH4InstanceIntersectors(BVH4* bvh);
    Accel::Intersectors BVH4InstanceMBIntersectors(BVH4* bvh);

    Accel::Intersectors BVH4InstanceArrayIntersectors(BVH4* bvh);
    Accel::Intersectors BVH4InstanceArrayMBIntersectors(BVH4* bvh);

    Accel::Intersectors BVH4SubdivPatch1Intersectors(BVH4* bvh);
    Accel::Intersectors BVH4SubdivPatch1MBIntersectors(BVH4* bvh);

    Accel::Intersectors BVH4GridIntersectors(BVH4* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH4GridMBIntersectors(BVH4* bvh, IntersectVariant ivariant);
    
  private:

    DEFINE_SYMBOL2(Accel::Collider,BVH4ColliderUserGeom);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4OBBVirtualCurveIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4OBBVirtualCurveIntersector1MB);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4OBBVirtualCurveIntersectorRobust1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4OBBVirtualCurveIntersectorRobust1MB);
    
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4Intersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4iIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4vIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4iIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4vMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4iMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4vMBIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Triangle4iMBIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Quad4vIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Quad4iIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Quad4vIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Quad4iIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Quad4iMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4Quad4iMBIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,QBVH4Triangle4iIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,QBVH4Quad4iIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4SubdivPatch1Intersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4SubdivPatch1MBIntersector1);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4VirtualIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4VirtualMBIntersector1);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4InstanceIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4InstanceMBIntersector1);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4InstanceArrayIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4InstanceArrayMBIntersector1);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH4GridIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4GridMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH4GridIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4OBBVirtualCurveIntersector4Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4OBBVirtualCurveIntersector4HybridMB);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4OBBVirtualCurveIntersectorRobust4Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4OBBVirtualCurveIntersectorRobust4HybridMB);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4Intersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4Intersector4HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4iIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4vIntersector4HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4iIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4vMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4iMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4vMBIntersector4HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Triangle4iMBIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4vIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4vIntersector4HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4iIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4vIntersector4HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4iIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4iMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4Quad4iMBIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4SubdivPatch1Intersector4);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4SubdivPatch1MBIntersector4);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4VirtualIntersector4Chunk);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4VirtualMBIntersector4Chunk);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4InstanceIntersector4Chunk);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4InstanceMBIntersector4Chunk);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4InstanceArrayIntersector4Chunk);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4InstanceArrayMBIntersector4Chunk);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH4GridIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4GridMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH4GridIntersector4HybridPluecker);

    // ==============

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4OBBVirtualCurveIntersector8Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4OBBVirtualCurveIntersector8HybridMB);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4OBBVirtualCurveIntersectorRobust8Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4OBBVirtualCurveIntersectorRobust8HybridMB);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4Intersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4Intersector8HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4iIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4vIntersector8HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4iIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4vMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4iMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4vMBIntersector8HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Triangle4iMBIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4vIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4vIntersector8HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4iIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4vIntersector8HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4iIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4iMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4Quad4iMBIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4SubdivPatch1Intersector8);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4SubdivPatch1MBIntersector8);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4VirtualIntersector8Chunk);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4VirtualMBIntersector8Chunk);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4InstanceIntersector8Chunk);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4InstanceMBIntersector8Chunk);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4InstanceArrayIntersector8Chunk);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4InstanceArrayMBIntersector8Chunk);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH4GridIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4GridMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH4GridIntersector8HybridPluecker);

    // ==============

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4OBBVirtualCurveIntersector16Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4OBBVirtualCurveIntersector16HybridMB);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4OBBVirtualCurveIntersectorRobust16Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4OBBVirtualCurveIntersectorRobust16HybridMB);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4Intersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4Intersector16HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4iIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4vIntersector16HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4iIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4vMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4iMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4vMBIntersector16HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Triangle4iMBIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4vIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4vIntersector16HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4iIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4vIntersector16HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4iIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4iMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4Quad4iMBIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4SubdivPatch1Intersector16);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4SubdivPatch1MBIntersector16);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4VirtualIntersector16Chunk);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4VirtualMBIntersector16Chunk);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4InstanceIntersector16Chunk);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4InstanceMBIntersector16Chunk);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4InstanceArrayIntersector16Chunk);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4InstanceArrayMBIntersector16Chunk);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH4GridIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4GridMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH4GridIntersector16HybridPluecker);

    // SAH scene builders
  private:
    DEFINE_ISA_FUNCTION(Builder*,BVH4Curve4vBuilder_OBB_New,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Curve4iBuilder_OBB_New,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4OBBCurve4iMBBuilder_OBB,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Curve8iBuilder_OBB_New,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4OBBCurve8iMBBuilder_OBB,void* COMMA Scene* COMMA size_t);

    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4SceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4vSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4iMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4vMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4QuantizedTriangle4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    
    DEFINE_ISA_FUNCTION(Builder*,BVH4Quad4vSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Quad4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Quad4iMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4QuantizedQuad4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    
    DEFINE_ISA_FUNCTION(Builder*,BVH4SubdivPatch1BuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4SubdivPatch1MBBuilderSAH,void* COMMA Scene* COMMA size_t);
    
    DEFINE_ISA_FUNCTION(Builder*,BVH4VirtualSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4VirtualMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);

    DEFINE_ISA_FUNCTION(Builder*,BVH4InstanceSceneBuilderSAH,void* COMMA Scene* COMMA Geometry::GTypeMask);
    DEFINE_ISA_FUNCTION(Builder*,BVH4InstanceMBSceneBuilderSAH,void* COMMA Scene* COMMA Geometry::GTypeMask);

    DEFINE_ISA_FUNCTION(Builder*,BVH4InstanceArraySceneBuilderSAH,void* COMMA Scene* COMMA Geometry::GTypeMask);
    DEFINE_ISA_FUNCTION(Builder*,BVH4InstanceArrayMBSceneBuilderSAH,void* COMMA Scene* COMMA Geometry::GTypeMask);

    DEFINE_ISA_FUNCTION(Builder*,BVH4GridSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4GridMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);

    // spatial scene builder
  private:
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4SceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4vSceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Triangle4iSceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH4Quad4vSceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);
    
    // twolevel scene builders
  private:
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelTriangle4MeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelTriangle4vMeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelTriangle4iMeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelQuadMeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelVirtualSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelInstanceSAH,void* COMMA Scene* COMMA Geometry::GTypeMask COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH4BuilderTwoLevelInstanceArraySAH,void* COMMA Scene* COMMA Geometry::GTypeMask COMMA bool);
  };
}
