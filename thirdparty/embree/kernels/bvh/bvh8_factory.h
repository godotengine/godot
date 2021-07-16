// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_factory.h"

namespace embree
{
  /*! BVH8 instantiations */
  class BVH8Factory : public BVHFactory
  {
  public:
    BVH8Factory(int bfeatures, int ifeatures);

  public:
    Accel* BVH8OBBVirtualCurve8v(Scene* scene, IntersectVariant ivariant);
    Accel* BVH8OBBVirtualCurve8iMB(Scene* scene, IntersectVariant ivariant);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector8v);
    DEFINE_SYMBOL2(VirtualCurveIntersector*,VirtualCurveIntersector8iMB);
    
    Accel* BVH8Triangle4   (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8Triangle4v  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8Triangle4i  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8Triangle4vMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8Triangle4iMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);

    Accel* BVH8Quad4v  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8Quad4i  (Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8Quad4iMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);

    Accel* BVH8QuantizedTriangle4i(Scene* scene);
    Accel* BVH8QuantizedTriangle4(Scene* scene);
    Accel* BVH8QuantizedQuad4i(Scene* scene);

    Accel* BVH8UserGeometry(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC);
    Accel* BVH8UserGeometryMB(Scene* scene);

    Accel* BVH8Instance(Scene* scene, bool isExpensive, BuildVariant bvariant = BuildVariant::STATIC);
    Accel* BVH8InstanceMB(Scene* scene, bool isExpensive);

    Accel* BVH8Grid(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);
    Accel* BVH8GridMB(Scene* scene, BuildVariant bvariant = BuildVariant::STATIC, IntersectVariant ivariant = IntersectVariant::FAST);

  private:
    void selectBuilders(int features);
    void selectIntersectors(int features);

  private:
    Accel::Intersectors BVH8OBBVirtualCurveIntersectors(BVH8* bvh, VirtualCurveIntersector* leafIntersector, IntersectVariant ivariant);
    Accel::Intersectors BVH8OBBVirtualCurveIntersectorsMB(BVH8* bvh, VirtualCurveIntersector* leafIntersector, IntersectVariant ivariant);
    
    Accel::Intersectors BVH8Triangle4Intersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8Triangle4vIntersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8Triangle4iIntersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8Triangle4iMBIntersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8Triangle4vMBIntersectors(BVH8* bvh, IntersectVariant ivariant);

    Accel::Intersectors BVH8Quad4vIntersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8Quad4iIntersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8Quad4iMBIntersectors(BVH8* bvh, IntersectVariant ivariant);

    Accel::Intersectors QBVH8Triangle4iIntersectors(BVH8* bvh);
    Accel::Intersectors QBVH8Triangle4Intersectors(BVH8* bvh);
    Accel::Intersectors QBVH8Quad4iIntersectors(BVH8* bvh);

    Accel::Intersectors BVH8UserGeometryIntersectors(BVH8* bvh);
    Accel::Intersectors BVH8UserGeometryMBIntersectors(BVH8* bvh);

    Accel::Intersectors BVH8InstanceIntersectors(BVH8* bvh);
    Accel::Intersectors BVH8InstanceMBIntersectors(BVH8* bvh);

    Accel::Intersectors BVH8GridIntersectors(BVH8* bvh, IntersectVariant ivariant);
    Accel::Intersectors BVH8GridMBIntersectors(BVH8* bvh, IntersectVariant ivariant);

  private:
    DEFINE_SYMBOL2(Accel::Collider,BVH8ColliderUserGeom);
    
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8OBBVirtualCurveIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8OBBVirtualCurveIntersector1MB);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8OBBVirtualCurveIntersectorRobust1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8OBBVirtualCurveIntersectorRobust1MB);
    
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4Intersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4iIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4vIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4iIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4vMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4iMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4vMBIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4iMBIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Triangle4vIntersector1Woop);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Quad4vIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Quad4iIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Quad4vIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Quad4iIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Quad4iMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8Quad4iMBIntersector1Pluecker);

    DEFINE_SYMBOL2(Accel::Intersector1,QBVH8Triangle4iIntersector1Pluecker);
    DEFINE_SYMBOL2(Accel::Intersector1,QBVH8Triangle4Intersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,QBVH8Quad4iIntersector1Pluecker);
    
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8VirtualIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8VirtualMBIntersector1);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH8InstanceIntersector1);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8InstanceMBIntersector1);

    DEFINE_SYMBOL2(Accel::Intersector1,BVH8GridIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8GridMBIntersector1Moeller);
    DEFINE_SYMBOL2(Accel::Intersector1,BVH8GridIntersector1Pluecker);
    
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8OBBVirtualCurveIntersector4Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8OBBVirtualCurveIntersector4HybridMB);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8OBBVirtualCurveIntersectorRobust4Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8OBBVirtualCurveIntersectorRobust4HybridMB);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4Intersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4Intersector4HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4iIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4vIntersector4HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4iIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4vMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4iMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4vMBIntersector4HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Triangle4iMBIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4vIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4vIntersector4HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4iIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4vIntersector4HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4iIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4iMBIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8Quad4iMBIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH8VirtualIntersector4Chunk);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8VirtualMBIntersector4Chunk);

    DEFINE_SYMBOL2(Accel::Intersector4,BVH8InstanceIntersector4Chunk);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8InstanceMBIntersector4Chunk);
    
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8GridIntersector4HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector4,BVH8GridIntersector4HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8OBBVirtualCurveIntersector8Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8OBBVirtualCurveIntersector8HybridMB);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8OBBVirtualCurveIntersectorRobust8Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8OBBVirtualCurveIntersectorRobust8HybridMB);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4Intersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4Intersector8HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4iIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4vIntersector8HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4iIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4vMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4iMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4vMBIntersector8HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Triangle4iMBIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4vIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4vIntersector8HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4iIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4vIntersector8HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4iIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4iMBIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8Quad4iMBIntersector8HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8VirtualIntersector8Chunk);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8VirtualMBIntersector8Chunk);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8InstanceIntersector8Chunk);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8InstanceMBIntersector8Chunk);

    DEFINE_SYMBOL2(Accel::Intersector8,BVH8GridIntersector8HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector8,BVH8GridIntersector8HybridPluecker);
   
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8OBBVirtualCurveIntersector16Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8OBBVirtualCurveIntersector16HybridMB);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8OBBVirtualCurveIntersectorRobust16Hybrid);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8OBBVirtualCurveIntersectorRobust16HybridMB);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4Intersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4Intersector16HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4iIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4vIntersector16HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4iIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4vMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4iMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4vMBIntersector16HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Triangle4iMBIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4vIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4vIntersector16HybridMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4iIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4vIntersector16HybridPluecker);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4iIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4iMBIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8Quad4iMBIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH8VirtualIntersector16Chunk);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8VirtualMBIntersector16Chunk);

    DEFINE_SYMBOL2(Accel::Intersector16,BVH8InstanceIntersector16Chunk);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8InstanceMBIntersector16Chunk);
   
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8GridIntersector16HybridMoeller);
    DEFINE_SYMBOL2(Accel::Intersector16,BVH8GridIntersector16HybridPluecker);

    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8IntersectorStreamPacketFallback);

    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Triangle4IntersectorStreamMoeller);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Triangle4IntersectorStreamMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Triangle4iIntersectorStreamMoeller);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Triangle4vIntersectorStreamPluecker);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Triangle4iIntersectorStreamPluecker);

    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Quad4vIntersectorStreamMoeller);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Quad4vIntersectorStreamMoellerNoFilter);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Quad4iIntersectorStreamMoeller);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Quad4vIntersectorStreamPluecker);
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8Quad4iIntersectorStreamPluecker);

    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8VirtualIntersectorStream);
    
    DEFINE_SYMBOL2(Accel::IntersectorN,BVH8InstanceIntersectorStream);

    // SAH scene builders
  private:
    DEFINE_ISA_FUNCTION(Builder*,BVH8Curve8vBuilder_OBB_New,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8OBBCurve8iMBBuilder_OBB,void* COMMA Scene* COMMA size_t);
 
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4SceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4vSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4iMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4vMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8QuantizedTriangle4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8QuantizedTriangle4SceneBuilderSAH,void* COMMA Scene* COMMA size_t);
 
    DEFINE_ISA_FUNCTION(Builder*,BVH8Quad4vSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Quad4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Quad4iMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8QuantizedQuad4iSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    
    DEFINE_ISA_FUNCTION(Builder*,BVH8VirtualSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8VirtualMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);

    DEFINE_ISA_FUNCTION(Builder*,BVH8InstanceSceneBuilderSAH,void* COMMA Scene* COMMA Geometry::GTypeMask);
    DEFINE_ISA_FUNCTION(Builder*,BVH8InstanceMBSceneBuilderSAH,void* COMMA Scene* COMMA Geometry::GTypeMask);

    DEFINE_ISA_FUNCTION(Builder*,BVH8GridSceneBuilderSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8GridMBSceneBuilderSAH,void* COMMA Scene* COMMA size_t);

    // SAH spatial scene builders
  private:
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4SceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Triangle4vSceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);
    DEFINE_ISA_FUNCTION(Builder*,BVH8Quad4vSceneBuilderFastSpatialSAH,void* COMMA Scene* COMMA size_t);

    // twolevel scene builders
  private:
    DEFINE_ISA_FUNCTION(Builder*,BVH8BuilderTwoLevelTriangle4MeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH8BuilderTwoLevelTriangle4vMeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH8BuilderTwoLevelTriangle4iMeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH8BuilderTwoLevelQuadMeshSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH8BuilderTwoLevelVirtualSAH,void* COMMA Scene* COMMA bool);
    DEFINE_ISA_FUNCTION(Builder*,BVH8BuilderTwoLevelInstanceSAH,void* COMMA Scene* COMMA Geometry::GTypeMask COMMA bool);
  };
}
