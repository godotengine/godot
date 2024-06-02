#pragma once

// NOLINTBEGIN(readability-duplicate-include): We need to be able to include certain headers
// multiple times when they're conditionally included through multiple preprocessor definitions that
// might not be perfectly mutually exclusive.

#ifdef _MSC_VER
// HACK(mihe): CMake's Visual Studio generator doesn't support system include paths
#pragma warning(push, 0)

// Pushing level 0 doesn't seem to disable the ones we've explicitly enabled
// C4245: conversion from 'type1' to 'type2', signed/unsigned mismatch
// C4365: conversion from 'type1' to 'type2', signed/unsigned mismatch
#pragma warning(disable : 4245 4365)
#endif // _MSC_VER

#include "core/extension/gdextension_interface.h"

#include "core/core_bind.h"
#include "core/os/time.h"
#include "core/templates/rid.h"
#include "core/templates/rid_owner.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/string/node_path.h"
#include "core/string/ustring.h"
#include "core/string/print_string.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include "core/config/project_settings.h"
#include "scene/resources/mesh.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/3d/physics/physics_body_3d.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering_server.h"
#include "core/object/worker_thread_pool.h"
#include "servers/extensions/physics_server_3d_extension.h"
#include "servers/extensions/physics_server_2d_extension.h"
#include "scene/gui/popup.h"
#include "scene/gui/popup_menu.h"
#include <variant>


#ifdef TOOLS_ENABLED

#include "editor/plugins/node_3d_editor_gizmos.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/editor_interface.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/editor_settings.h"
#include "core/debugger/engine_debugger.h"


#endif // TOOLS_ENABLED

#ifdef JPH_DEBUG_RENDERER

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/world3d.hpp>

#endif // JPH_DEBUG_RENDERER

#include "jolt/Jolt.h"

#include "jolt/Core/Factory.h"
#include "jolt/Core/FixedSizeFreeList.h"
#include "jolt/Core/IssueReporting.h"
#include "jolt/Core/JobSystemWithBarrier.h"
#include "jolt/Core/TempAllocator.h"
#include "jolt/Geometry/ConvexSupport.h"
#include "jolt/Geometry/GJKClosestPoint.h"
#include "jolt/Physics/Body/BodyCreationSettings.h"
#include "jolt/Physics/Body/BodyID.h"
#include "jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h"
#include "jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h"
#include "jolt/Physics/Collision/CastResult.h"
#include "jolt/Physics/Collision/CollidePointResult.h"
#include "jolt/Physics/Collision/CollideShape.h"
#include "jolt/Physics/Collision/CollisionDispatch.h"
#include "jolt/Physics/Collision/CollisionGroup.h"
#include "jolt/Physics/Collision/ContactListener.h"
#include "jolt/Physics/Collision/EstimateCollisionResponse.h"
#include "jolt/Physics/Collision/GroupFilter.h"
#include "jolt/Physics/Collision/InternalEdgeRemovingCollector.h"
#include "jolt/Physics/Collision/ManifoldBetweenTwoFaces.h"
#include "jolt/Physics/Collision/NarrowPhaseQuery.h"
#include "jolt/Physics/Collision/ObjectLayer.h"
#include "jolt/Physics/Collision/RayCast.h"
#include "jolt/Physics/Collision/Shape/BoxShape.h"
#include "jolt/Physics/Collision/Shape/CapsuleShape.h"
#include "jolt/Physics/Collision/Shape/ConvexHullShape.h"
#include "jolt/Physics/Collision/Shape/CylinderShape.h"
#include "jolt/Physics/Collision/Shape/HeightFieldShape.h"
#include "jolt/Physics/Collision/Shape/MeshShape.h"
#include "jolt/Physics/Collision/Shape/MutableCompoundShape.h"
#include "jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h"
#include "jolt/Physics/Collision/Shape/RotatedTranslatedShape.h"
#include "jolt/Physics/Collision/Shape/ScaledShape.h"
#include "jolt/Physics/Collision/Shape/SphereShape.h"
#include "jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "jolt/Physics/Constraints/FixedConstraint.h"
#include "jolt/Physics/Constraints/HingeConstraint.h"
#include "jolt/Physics/Constraints/PointConstraint.h"
#include "jolt/Physics/Constraints/SixDOFConstraint.h"
#include "jolt/Physics/Constraints/SliderConstraint.h"
#include "jolt/Physics/Constraints/SwingTwistConstraint.h"
#include "jolt/Physics/PhysicsScene.h"
#include "jolt/Physics/PhysicsSystem.h"
#include "jolt/Physics/SoftBody/SoftBodyContactListener.h"
#include "jolt/Physics/SoftBody/SoftBodyCreationSettings.h"
#include "jolt/Physics/SoftBody/SoftBodyManifold.h"
#include "jolt/Physics/SoftBody/SoftBodyMotionProperties.h"
#include "jolt/Physics/SoftBody/SoftBodySharedSettings.h"
#include "jolt/RegisterTypes.h"










