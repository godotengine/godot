# Requires C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Root
set(JOLT_PHYSICS_ROOT ${PHYSICS_REPO_ROOT}/Jolt)

# Source files
set(JOLT_PHYSICS_SRC_FILES
	${JOLT_PHYSICS_ROOT}/AABBTree/AABBTreeBuilder.cpp
	${JOLT_PHYSICS_ROOT}/AABBTree/AABBTreeBuilder.h
	${JOLT_PHYSICS_ROOT}/AABBTree/AABBTreeToBuffer.h
	${JOLT_PHYSICS_ROOT}/AABBTree/NodeCodec/NodeCodecQuadTreeHalfFloat.h
	${JOLT_PHYSICS_ROOT}/AABBTree/TriangleCodec/TriangleCodecIndexed8BitPackSOA4Flags.h
	${JOLT_PHYSICS_ROOT}/ConfigurationString.h
	${JOLT_PHYSICS_ROOT}/Core/ARMNeon.h
	${JOLT_PHYSICS_ROOT}/Core/Array.h
	${JOLT_PHYSICS_ROOT}/Core/Atomics.h
	${JOLT_PHYSICS_ROOT}/Core/ByteBuffer.h
	${JOLT_PHYSICS_ROOT}/Core/Color.cpp
	${JOLT_PHYSICS_ROOT}/Core/Color.h
	${JOLT_PHYSICS_ROOT}/Core/Core.h
	${JOLT_PHYSICS_ROOT}/Core/Factory.cpp
	${JOLT_PHYSICS_ROOT}/Core/Factory.h
	${JOLT_PHYSICS_ROOT}/Core/FixedSizeFreeList.h
	${JOLT_PHYSICS_ROOT}/Core/FixedSizeFreeList.inl
	${JOLT_PHYSICS_ROOT}/Core/FPControlWord.h
	${JOLT_PHYSICS_ROOT}/Core/FPException.h
	${JOLT_PHYSICS_ROOT}/Core/FPFlushDenormals.h
	${JOLT_PHYSICS_ROOT}/Core/HashCombine.h
	${JOLT_PHYSICS_ROOT}/Core/InsertionSort.h
	${JOLT_PHYSICS_ROOT}/Core/IssueReporting.cpp
	${JOLT_PHYSICS_ROOT}/Core/IssueReporting.h
	${JOLT_PHYSICS_ROOT}/Core/JobSystem.h
	${JOLT_PHYSICS_ROOT}/Core/JobSystem.inl
	${JOLT_PHYSICS_ROOT}/Core/JobSystemSingleThreaded.cpp
	${JOLT_PHYSICS_ROOT}/Core/JobSystemSingleThreaded.h
	${JOLT_PHYSICS_ROOT}/Core/JobSystemThreadPool.cpp
	${JOLT_PHYSICS_ROOT}/Core/JobSystemThreadPool.h
	${JOLT_PHYSICS_ROOT}/Core/JobSystemWithBarrier.cpp
	${JOLT_PHYSICS_ROOT}/Core/JobSystemWithBarrier.h
	${JOLT_PHYSICS_ROOT}/Core/LinearCurve.cpp
	${JOLT_PHYSICS_ROOT}/Core/LinearCurve.h
	${JOLT_PHYSICS_ROOT}/Core/LockFreeHashMap.h
	${JOLT_PHYSICS_ROOT}/Core/LockFreeHashMap.inl
	${JOLT_PHYSICS_ROOT}/Core/Memory.cpp
	${JOLT_PHYSICS_ROOT}/Core/Memory.h
	${JOLT_PHYSICS_ROOT}/Core/Mutex.h
	${JOLT_PHYSICS_ROOT}/Core/MutexArray.h
	${JOLT_PHYSICS_ROOT}/Core/NonCopyable.h
	${JOLT_PHYSICS_ROOT}/Core/Profiler.cpp
	${JOLT_PHYSICS_ROOT}/Core/Profiler.h
	${JOLT_PHYSICS_ROOT}/Core/Profiler.inl
	${JOLT_PHYSICS_ROOT}/Core/QuickSort.h
	${JOLT_PHYSICS_ROOT}/Core/Reference.h
	${JOLT_PHYSICS_ROOT}/Core/Result.h
	${JOLT_PHYSICS_ROOT}/Core/RTTI.cpp
	${JOLT_PHYSICS_ROOT}/Core/RTTI.h
	${JOLT_PHYSICS_ROOT}/Core/ScopeExit.h
	${JOLT_PHYSICS_ROOT}/Core/Semaphore.cpp
	${JOLT_PHYSICS_ROOT}/Core/Semaphore.h
	${JOLT_PHYSICS_ROOT}/Core/StaticArray.h
	${JOLT_PHYSICS_ROOT}/Core/STLAlignedAllocator.h
	${JOLT_PHYSICS_ROOT}/Core/STLAllocator.h
	${JOLT_PHYSICS_ROOT}/Core/STLTempAllocator.h
	${JOLT_PHYSICS_ROOT}/Core/StreamIn.h
	${JOLT_PHYSICS_ROOT}/Core/StreamOut.h
	${JOLT_PHYSICS_ROOT}/Core/StreamUtils.h
	${JOLT_PHYSICS_ROOT}/Core/StreamWrapper.h
	${JOLT_PHYSICS_ROOT}/Core/StridedPtr.h
	${JOLT_PHYSICS_ROOT}/Core/StringTools.cpp
	${JOLT_PHYSICS_ROOT}/Core/StringTools.h
	${JOLT_PHYSICS_ROOT}/Core/TempAllocator.h
	${JOLT_PHYSICS_ROOT}/Core/TickCounter.cpp
	${JOLT_PHYSICS_ROOT}/Core/TickCounter.h
	${JOLT_PHYSICS_ROOT}/Core/UnorderedMap.h
	${JOLT_PHYSICS_ROOT}/Core/UnorderedSet.h
	${JOLT_PHYSICS_ROOT}/Geometry/AABox.h
	${JOLT_PHYSICS_ROOT}/Geometry/AABox4.h
	${JOLT_PHYSICS_ROOT}/Geometry/ClipPoly.h
	${JOLT_PHYSICS_ROOT}/Geometry/ClosestPoint.h
	${JOLT_PHYSICS_ROOT}/Geometry/ConvexHullBuilder.cpp
	${JOLT_PHYSICS_ROOT}/Geometry/ConvexHullBuilder.h
	${JOLT_PHYSICS_ROOT}/Geometry/ConvexHullBuilder2D.cpp
	${JOLT_PHYSICS_ROOT}/Geometry/ConvexHullBuilder2D.h
	${JOLT_PHYSICS_ROOT}/Geometry/ConvexSupport.h
	${JOLT_PHYSICS_ROOT}/Geometry/Ellipse.h
	${JOLT_PHYSICS_ROOT}/Geometry/EPAConvexHullBuilder.h
	${JOLT_PHYSICS_ROOT}/Geometry/EPAPenetrationDepth.h
	${JOLT_PHYSICS_ROOT}/Geometry/GJKClosestPoint.h
	${JOLT_PHYSICS_ROOT}/Geometry/IndexedTriangle.h
	${JOLT_PHYSICS_ROOT}/Geometry/Indexify.cpp
	${JOLT_PHYSICS_ROOT}/Geometry/Indexify.h
	${JOLT_PHYSICS_ROOT}/Geometry/MortonCode.h
	${JOLT_PHYSICS_ROOT}/Geometry/OrientedBox.cpp
	${JOLT_PHYSICS_ROOT}/Geometry/OrientedBox.h
	${JOLT_PHYSICS_ROOT}/Geometry/Plane.h
	${JOLT_PHYSICS_ROOT}/Geometry/RayAABox.h
	${JOLT_PHYSICS_ROOT}/Geometry/RayCapsule.h
	${JOLT_PHYSICS_ROOT}/Geometry/RayCylinder.h
	${JOLT_PHYSICS_ROOT}/Geometry/RaySphere.h
	${JOLT_PHYSICS_ROOT}/Geometry/RayTriangle.h
	${JOLT_PHYSICS_ROOT}/Geometry/Sphere.h
	${JOLT_PHYSICS_ROOT}/Geometry/Triangle.h
	${JOLT_PHYSICS_ROOT}/Jolt.cmake
	${JOLT_PHYSICS_ROOT}/Jolt.h
	${JOLT_PHYSICS_ROOT}/Math/DMat44.h
	${JOLT_PHYSICS_ROOT}/Math/DMat44.inl
	${JOLT_PHYSICS_ROOT}/Math/Double3.h
	${JOLT_PHYSICS_ROOT}/Math/DVec3.h
	${JOLT_PHYSICS_ROOT}/Math/DVec3.inl
	${JOLT_PHYSICS_ROOT}/Math/DynMatrix.h
	${JOLT_PHYSICS_ROOT}/Math/EigenValueSymmetric.h
	${JOLT_PHYSICS_ROOT}/Math/FindRoot.h
	${JOLT_PHYSICS_ROOT}/Math/Float2.h
	${JOLT_PHYSICS_ROOT}/Math/Float3.h
	${JOLT_PHYSICS_ROOT}/Math/Float4.h
	${JOLT_PHYSICS_ROOT}/Math/GaussianElimination.h
	${JOLT_PHYSICS_ROOT}/Math/HalfFloat.h
	${JOLT_PHYSICS_ROOT}/Math/Mat44.h
	${JOLT_PHYSICS_ROOT}/Math/Mat44.inl
	${JOLT_PHYSICS_ROOT}/Math/Math.h
	${JOLT_PHYSICS_ROOT}/Math/MathTypes.h
	${JOLT_PHYSICS_ROOT}/Math/Matrix.h
	${JOLT_PHYSICS_ROOT}/Math/Quat.h
	${JOLT_PHYSICS_ROOT}/Math/Quat.inl
	${JOLT_PHYSICS_ROOT}/Math/Real.h
	${JOLT_PHYSICS_ROOT}/Math/Swizzle.h
	${JOLT_PHYSICS_ROOT}/Math/Trigonometry.h
	${JOLT_PHYSICS_ROOT}/Math/UVec4.h
	${JOLT_PHYSICS_ROOT}/Math/UVec4.inl
	${JOLT_PHYSICS_ROOT}/Math/Vec3.cpp
	${JOLT_PHYSICS_ROOT}/Math/Vec3.h
	${JOLT_PHYSICS_ROOT}/Math/Vec3.inl
	${JOLT_PHYSICS_ROOT}/Math/Vec4.h
	${JOLT_PHYSICS_ROOT}/Math/Vec4.inl
	${JOLT_PHYSICS_ROOT}/Math/Vector.h
	${JOLT_PHYSICS_ROOT}/ObjectStream/SerializableObject.cpp
	${JOLT_PHYSICS_ROOT}/ObjectStream/SerializableObject.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/AllowedDOFs.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/Body.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Body/Body.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/Body.inl
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyAccess.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyActivationListener.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyCreationSettings.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyCreationSettings.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyFilter.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyID.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyInterface.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyInterface.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyLock.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyLockInterface.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyLockMulti.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyManager.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyManager.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyPair.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/BodyType.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/MassProperties.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Body/MassProperties.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/MotionProperties.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Body/MotionProperties.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/MotionProperties.inl
	${JOLT_PHYSICS_ROOT}/Physics/Body/MotionQuality.h
	${JOLT_PHYSICS_ROOT}/Physics/Body/MotionType.h
	${JOLT_PHYSICS_ROOT}/Physics/Character/Character.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Character/Character.h
	${JOLT_PHYSICS_ROOT}/Physics/Character/CharacterBase.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Character/CharacterBase.h
	${JOLT_PHYSICS_ROOT}/Physics/Character/CharacterVirtual.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Character/CharacterVirtual.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/AABoxCast.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ActiveEdgeMode.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ActiveEdges.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BackFaceMode.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhase.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhase.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseBruteForce.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseBruteForce.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseLayer.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceMask.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseQuadTree.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseQuadTree.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/BroadPhaseQuery.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterMask.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/QuadTree.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/BroadPhase/QuadTree.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CastConvexVsTriangles.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CastConvexVsTriangles.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CastResult.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CastSphereVsTriangles.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CastSphereVsTriangles.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollectFacesMode.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideConvexVsTriangles.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideConvexVsTriangles.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollidePointResult.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideSoftBodyVertexIterator.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideSoftBodyVerticesVsTriangles.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideSphereVsTriangles.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollideSphereVsTriangles.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollisionCollector.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollisionCollectorImpl.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollisionDispatch.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollisionDispatch.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollisionGroup.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/CollisionGroup.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ContactListener.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/EstimateCollisionResponse.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/EstimateCollisionResponse.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/GroupFilter.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/GroupFilter.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/GroupFilterTable.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/GroupFilterTable.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/InternalEdgeRemovingCollector.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ManifoldBetweenTwoFaces.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ManifoldBetweenTwoFaces.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/NarrowPhaseQuery.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/NarrowPhaseQuery.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/NarrowPhaseStats.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/NarrowPhaseStats.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ObjectLayer.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ObjectLayerPairFilterMask.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ObjectLayerPairFilterTable.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/PhysicsMaterial.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/PhysicsMaterial.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/PhysicsMaterialSimple.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/PhysicsMaterialSimple.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/RayCast.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/BoxShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/BoxShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CapsuleShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CapsuleShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CompoundShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CompoundShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CompoundShapeVisitors.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ConvexHullShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ConvexHullShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ConvexShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ConvexShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CylinderShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/CylinderShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/DecoratedShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/DecoratedShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/EmptyShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/EmptyShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/GetTrianglesContext.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/HeightFieldShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/HeightFieldShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/MeshShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/MeshShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/MutableCompoundShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/MutableCompoundShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/OffsetCenterOfMassShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/OffsetCenterOfMassShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/PlaneShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/PlaneShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/PolyhedronSubmergedVolumeCalculator.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/RotatedTranslatedShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/RotatedTranslatedShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ScaledShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ScaledShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/ScaleHelpers.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/Shape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/Shape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/SphereShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/SphereShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/StaticCompoundShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/StaticCompoundShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/SubShapeID.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/SubShapeIDPair.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TaperedCapsuleShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TaperedCapsuleShape.gliffy
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TaperedCapsuleShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TaperedCylinderShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TaperedCylinderShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TriangleShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/Shape/TriangleShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ShapeCast.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/ShapeFilter.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/SortReverseAndStore.h
	${JOLT_PHYSICS_ROOT}/Physics/Collision/TransformedShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Collision/TransformedShape.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/CalculateSolverSteps.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConeConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConeConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/Constraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/Constraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintManager.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintManager.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/AngleConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/AxisConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/DualAxisConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/GearConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/HingeRotationConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/IndependentAxisConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/PointConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/RackAndPinionConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/RotationEulerConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/RotationQuatConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/SpringPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ConstraintPart/SwingTwistConstraintPart.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ContactConstraintManager.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/ContactConstraintManager.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/DistanceConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/DistanceConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/FixedConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/FixedConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/GearConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/GearConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/HingeConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/HingeConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/MotorSettings.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/MotorSettings.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PathConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PathConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PathConstraintPath.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PathConstraintPath.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PathConstraintPathHermite.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PathConstraintPathHermite.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PointConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PointConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PulleyConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/PulleyConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/RackAndPinionConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/RackAndPinionConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SixDOFConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SixDOFConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SliderConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SliderConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SpringSettings.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SpringSettings.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SwingTwistConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/SwingTwistConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/TwoBodyConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Constraints/TwoBodyConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/DeterminismLog.cpp
	${JOLT_PHYSICS_ROOT}/Physics/DeterminismLog.h
	${JOLT_PHYSICS_ROOT}/Physics/EActivation.h
	${JOLT_PHYSICS_ROOT}/Physics/EPhysicsUpdateError.h
	${JOLT_PHYSICS_ROOT}/Physics/IslandBuilder.cpp
	${JOLT_PHYSICS_ROOT}/Physics/IslandBuilder.h
	${JOLT_PHYSICS_ROOT}/Physics/LargeIslandSplitter.cpp
	${JOLT_PHYSICS_ROOT}/Physics/LargeIslandSplitter.h
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsLock.h
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsScene.cpp
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsScene.h
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsSettings.h
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsStepListener.h
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsSystem.cpp
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsSystem.h
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsUpdateContext.cpp
	${JOLT_PHYSICS_ROOT}/Physics/PhysicsUpdateContext.h
	${JOLT_PHYSICS_ROOT}/Physics/Ragdoll/Ragdoll.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Ragdoll/Ragdoll.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyContactListener.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyCreationSettings.cpp
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyCreationSettings.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyManifold.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyMotionProperties.cpp
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyMotionProperties.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyShape.cpp
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyShape.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodySharedSettings.cpp
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodySharedSettings.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyUpdateContext.h
	${JOLT_PHYSICS_ROOT}/Physics/SoftBody/SoftBodyVertex.h
	${JOLT_PHYSICS_ROOT}/Physics/StateRecorder.h
	${JOLT_PHYSICS_ROOT}/Physics/StateRecorderImpl.cpp
	${JOLT_PHYSICS_ROOT}/Physics/StateRecorderImpl.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/MotorcycleController.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/MotorcycleController.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/TrackedVehicleController.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/TrackedVehicleController.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleAntiRollBar.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleAntiRollBar.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleCollisionTester.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleCollisionTester.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleConstraint.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleConstraint.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleController.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleController.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleDifferential.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleDifferential.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleEngine.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleEngine.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleTrack.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleTrack.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleTransmission.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/VehicleTransmission.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/Wheel.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/Wheel.h
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/WheeledVehicleController.cpp
	${JOLT_PHYSICS_ROOT}/Physics/Vehicle/WheeledVehicleController.h
	${JOLT_PHYSICS_ROOT}/RegisterTypes.cpp
	${JOLT_PHYSICS_ROOT}/RegisterTypes.h
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRenderer.cpp
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRenderer.h
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRendererPlayback.cpp
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRendererPlayback.h
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRendererRecorder.cpp
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRendererRecorder.h
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRendererSimple.cpp
	${JOLT_PHYSICS_ROOT}/Renderer/DebugRendererSimple.h
	${JOLT_PHYSICS_ROOT}/Skeleton/SkeletalAnimation.cpp
	${JOLT_PHYSICS_ROOT}/Skeleton/SkeletalAnimation.h
	${JOLT_PHYSICS_ROOT}/Skeleton/Skeleton.cpp
	${JOLT_PHYSICS_ROOT}/Skeleton/Skeleton.h
	${JOLT_PHYSICS_ROOT}/Skeleton/SkeletonMapper.cpp
	${JOLT_PHYSICS_ROOT}/Skeleton/SkeletonMapper.h
	${JOLT_PHYSICS_ROOT}/Skeleton/SkeletonPose.cpp
	${JOLT_PHYSICS_ROOT}/Skeleton/SkeletonPose.h
	${JOLT_PHYSICS_ROOT}/TriangleGrouper/TriangleGrouper.h
	${JOLT_PHYSICS_ROOT}/TriangleGrouper/TriangleGrouperClosestCentroid.cpp
	${JOLT_PHYSICS_ROOT}/TriangleGrouper/TriangleGrouperClosestCentroid.h
	${JOLT_PHYSICS_ROOT}/TriangleGrouper/TriangleGrouperMorton.cpp
	${JOLT_PHYSICS_ROOT}/TriangleGrouper/TriangleGrouperMorton.h
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitter.cpp
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitter.h
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterBinning.cpp
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterBinning.h
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterFixedLeafSize.cpp
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterFixedLeafSize.h
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterLongestAxis.cpp
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterLongestAxis.h
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterMean.cpp
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterMean.h
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterMorton.cpp
	${JOLT_PHYSICS_ROOT}/TriangleSplitter/TriangleSplitterMorton.h
)

if (ENABLE_OBJECT_STREAM)
	set(JOLT_PHYSICS_SRC_FILES
		${JOLT_PHYSICS_SRC_FILES}
		${JOLT_PHYSICS_ROOT}/ObjectStream/GetPrimitiveTypeOfType.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStream.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStream.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamBinaryIn.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamBinaryIn.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamBinaryOut.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamBinaryOut.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamIn.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamIn.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamOut.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamOut.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamTextIn.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamTextIn.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamTextOut.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamTextOut.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/ObjectStreamTypes.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/SerializableAttribute.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/SerializableAttributeEnum.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/SerializableAttributeTyped.h
		${JOLT_PHYSICS_ROOT}/ObjectStream/TypeDeclarations.cpp
		${JOLT_PHYSICS_ROOT}/ObjectStream/TypeDeclarations.h
	)
endif()

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
	# Add natvis file
	set(JOLT_PHYSICS_SRC_FILES ${JOLT_PHYSICS_SRC_FILES} ${JOLT_PHYSICS_ROOT}/Jolt.natvis)
endif()

# Group source files
source_group(TREE ${JOLT_PHYSICS_ROOT} FILES ${JOLT_PHYSICS_SRC_FILES})

# Create Jolt lib
add_library(Jolt ${JOLT_PHYSICS_SRC_FILES})

if (BUILD_SHARED_LIBS)
	# Set default visibility to hidden
	set(CMAKE_CXX_VISIBILITY_PRESET hidden)

	if (GENERATE_DEBUG_SYMBOLS)
		if (MSVC)
			# MSVC specific option to enable PDB generation
			set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG:FASTLINK")
		else()
			# Clang/GCC option to enable debug symbol generation
			set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -g")
		endif()
	endif()

	# Set linker flags for other build types to be the same as release
	set(CMAKE_SHARED_LINKER_FLAGS_RELEASEASAN "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
	set(CMAKE_SHARED_LINKER_FLAGS_RELEASEUBSAN "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
	set(CMAKE_SHARED_LINKER_FLAGS_RELEASECOVERAGE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
	set(CMAKE_SHARED_LINKER_FLAGS_DISTRIBUTION "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

	# Public define to instruct user code to import Jolt symbols (rather than use static linking)
	target_compile_definitions(Jolt PUBLIC JPH_SHARED_LIBRARY)

	# Private define to instruct the library to export symbols for shared linking
	target_compile_definitions(Jolt PRIVATE JPH_BUILD_SHARED_LIBRARY)
endif()

# Use repository as include directory when building, install directory when installing
target_include_directories(Jolt PUBLIC
	$<BUILD_INTERFACE:${PHYSICS_REPO_ROOT}>
	$<INSTALL_INTERFACE:include/>)

# Code coverage doesn't work when using precompiled headers
if (CMAKE_GENERATOR STREQUAL "Ninja Multi-Config" AND MSVC)
	# The Ninja Multi-Config generator errors out when selectively disabling precompiled headers for certain configurations.
	# See: https://github.com/jrouwe/JoltPhysics/issues/1211
	target_precompile_headers(Jolt PRIVATE "${JOLT_PHYSICS_ROOT}/Jolt.h")
else()
	target_precompile_headers(Jolt PRIVATE "$<$<NOT:$<CONFIG:ReleaseCoverage>>:${JOLT_PHYSICS_ROOT}/Jolt.h>")
endif()

if (NOT CPP_EXCEPTIONS_ENABLED)
	# Disable use of exceptions in MSVC's STL
	target_compile_definitions(Jolt PUBLIC $<$<BOOL:${MSVC}>:_HAS_EXCEPTIONS=0>)
endif()

# Set the debug/non-debug build flags
target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:Debug>:_DEBUG>")
target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:Release,Distribution,ReleaseASAN,ReleaseUBSAN,ReleaseCoverage>:NDEBUG>")

# ASAN should use the default allocators
target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:ReleaseASAN>:JPH_DISABLE_TEMP_ALLOCATOR;JPH_DISABLE_CUSTOM_ALLOCATOR>")

# Setting floating point exceptions
if (FLOATING_POINT_EXCEPTIONS_ENABLED AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:Debug,Release>:JPH_FLOATING_POINT_EXCEPTIONS_ENABLED>")
endif()

# Setting the disable custom allocator flag
if (DISABLE_CUSTOM_ALLOCATOR)
	target_compile_definitions(Jolt PUBLIC JPH_DISABLE_CUSTOM_ALLOCATOR)
endif()

# Setting enable asserts flag
if (USE_ASSERTS)
	target_compile_definitions(Jolt PUBLIC JPH_ENABLE_ASSERTS)
endif()

# Setting double precision flag
if (DOUBLE_PRECISION)
	target_compile_definitions(Jolt PUBLIC JPH_DOUBLE_PRECISION)
endif()

# Setting to attempt cross platform determinism
if (CROSS_PLATFORM_DETERMINISTIC)
	target_compile_definitions(Jolt PUBLIC JPH_CROSS_PLATFORM_DETERMINISTIC)
endif()

# Setting to determine number of bits in ObjectLayer
if (OBJECT_LAYER_BITS)
	target_compile_definitions(Jolt PUBLIC JPH_OBJECT_LAYER_BITS=${OBJECT_LAYER_BITS})
endif()

if (USE_STD_VECTOR)
	target_compile_definitions(Jolt PUBLIC JPH_USE_STD_VECTOR)
endif()

# Setting to periodically trace broadphase stats to help determine if the broadphase layer configuration is optimal
if (TRACK_BROADPHASE_STATS)
	target_compile_definitions(Jolt PUBLIC JPH_TRACK_BROADPHASE_STATS)
endif()

# Setting to periodically trace narrowphase stats to help determine which collision queries could be optimized
if (TRACK_NARROWPHASE_STATS)
	target_compile_definitions(Jolt PUBLIC JPH_TRACK_NARROWPHASE_STATS)
endif()

# Enable the debug renderer
if (DEBUG_RENDERER_IN_DISTRIBUTION)
	target_compile_definitions(Jolt PUBLIC "JPH_DEBUG_RENDERER")
elseif (DEBUG_RENDERER_IN_DEBUG_AND_RELEASE)
	target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:Debug,Release,ReleaseASAN,ReleaseUBSAN>:JPH_DEBUG_RENDERER>")
endif()

# Enable the profiler
if (PROFILER_IN_DISTRIBUTION)
	target_compile_definitions(Jolt PUBLIC "JPH_PROFILE_ENABLED")
elseif (PROFILER_IN_DEBUG_AND_RELEASE)
	target_compile_definitions(Jolt PUBLIC "$<$<CONFIG:Debug,Release,ReleaseASAN,ReleaseUBSAN>:JPH_PROFILE_ENABLED>")
endif()

# Compile the ObjectStream class and RTTI attribute information
if (ENABLE_OBJECT_STREAM)
	target_compile_definitions(Jolt PUBLIC JPH_OBJECT_STREAM)
endif()

# Emit the instruction set definitions to ensure that child projects use the same settings even if they override the used instruction sets (a mismatch causes link errors)
function(EMIT_X86_INSTRUCTION_SET_DEFINITIONS)
	if (USE_AVX512)
		target_compile_definitions(Jolt PUBLIC JPH_USE_AVX512)
	endif()
	if (USE_AVX2)
		target_compile_definitions(Jolt PUBLIC JPH_USE_AVX2)
	endif()
	if (USE_AVX)
		target_compile_definitions(Jolt PUBLIC JPH_USE_AVX)
	endif()
	if (USE_SSE4_1)
		target_compile_definitions(Jolt PUBLIC JPH_USE_SSE4_1)
	endif()
	if (USE_SSE4_2)
		target_compile_definitions(Jolt PUBLIC JPH_USE_SSE4_2)
	endif()
	if (USE_LZCNT)
		target_compile_definitions(Jolt PUBLIC JPH_USE_LZCNT)
	endif()
	if (USE_TZCNT)
		target_compile_definitions(Jolt PUBLIC JPH_USE_TZCNT)
	endif()
	if (USE_F16C)
		target_compile_definitions(Jolt PUBLIC JPH_USE_F16C)
	endif()
	if (USE_FMADD AND NOT CROSS_PLATFORM_DETERMINISTIC)
		target_compile_definitions(Jolt PUBLIC JPH_USE_FMADD)
	endif()
endfunction()

# Add the compiler commandline flags to select the right instruction sets
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	if ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86" OR "${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
		if (USE_AVX512)
			target_compile_options(Jolt PUBLIC /arch:AVX512)
		elseif (USE_AVX2)
			target_compile_options(Jolt PUBLIC /arch:AVX2)
		elseif (USE_AVX)
			target_compile_options(Jolt PUBLIC /arch:AVX)
		endif()
		EMIT_X86_INSTRUCTION_SET_DEFINITIONS()
	endif()
else()
	if (XCODE)
		# XCode builds for multiple architectures, we can't set global flags
	elseif (CROSS_COMPILE_ARM OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
		# ARM64 uses no special commandline flags
	elseif (EMSCRIPTEN)
		if (USE_WASM_SIMD)
			# Jolt currently doesn't implement the WASM specific SIMD intrinsics so uses the SSE 4.2 intrinsics
			# See: https://emscripten.org/docs/porting/simd.html#webassembly-simd-intrinsics
			# Note that this does not require the browser to actually support SSE 4.2 it merely means that it can translate those instructions to WASM SIMD instructions
			target_compile_options(Jolt PUBLIC -msimd128 -msse4.2)
		endif()
	elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "i386")
		# x86 and x86_64
		# On 32-bit builds we need to default to using SSE instructions, the x87 FPU instructions have higher intermediate precision
		# which will cause problems in the collision detection code (the effect is similar to leaving FMA on, search for
		# JPH_PRECISE_MATH_ON for the locations where this is a problem).

		if (USE_AVX512)
			target_compile_options(Jolt PUBLIC -mavx512f -mavx512vl -mavx512dq -mavx2 -mbmi -mpopcnt -mlzcnt -mf16c)
		elseif (USE_AVX2)
			target_compile_options(Jolt PUBLIC -mavx2 -mbmi -mpopcnt -mlzcnt -mf16c)
		elseif (USE_AVX)
			target_compile_options(Jolt PUBLIC -mavx -mpopcnt)
		elseif (USE_SSE4_2)
			target_compile_options(Jolt PUBLIC -msse4.2 -mpopcnt)
		elseif (USE_SSE4_1)
			target_compile_options(Jolt PUBLIC -msse4.1)
		else()
			target_compile_options(Jolt PUBLIC -msse2)
		endif()
		if (USE_LZCNT)
			target_compile_options(Jolt PUBLIC -mlzcnt)
		endif()
		if (USE_TZCNT)
			target_compile_options(Jolt PUBLIC -mbmi)
		endif()
		if (USE_F16C)
			target_compile_options(Jolt PUBLIC -mf16c)
		endif()
		if (USE_FMADD AND NOT CROSS_PLATFORM_DETERMINISTIC)
			target_compile_options(Jolt PUBLIC -mfma)
		endif()

		if (NOT MSVC)
			target_compile_options(Jolt PUBLIC -mfpmath=sse)
		endif()

		EMIT_X86_INSTRUCTION_SET_DEFINITIONS()
	endif()
endif()

# On Unix flavors we need the pthread library
if (NOT ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows") AND NOT EMSCRIPTEN)
	target_compile_options(Jolt PUBLIC -pthread)
	target_link_options(Jolt PUBLIC -pthread)
endif()

if (EMSCRIPTEN)
	# We need more than the default 64KB stack and 16MB memory
	# Also disable warning: running limited binaryen optimizations because DWARF info requested (or indirectly required)
	target_link_options(Jolt PUBLIC -sSTACK_SIZE=1048576 -sINITIAL_MEMORY=134217728 -Wno-limited-postlink-optimizations)
endif()
