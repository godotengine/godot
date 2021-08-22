/*************************************************************************/
/*  converter.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef CONVERTER_H
#define CONVERTER_H

#include "core/core_bind.h"
#include "core/io/file_access.h"
#include "core/string/print_string.cpp"
#include "core/string/ustring.h"
#include "modules/regex/regex.h"

// TODO add Regex as required dependency

// TODO maybe cache all compiled regex

static const char *enum_renames[][2] = {
	{ "TYPE_QUAT", "TYPE_QUATERNION" },
	{ "TYPE_REAL", "TYPE_FLOAT" },
	{ "TYPE_TRANSFORM", "TYPE_TRANSFORM3D" },

	{ "MODE_OPEN_ANY", "FILE_MODE_OPEN_ANY" },
	{ "MODE_OPEN_DIR", "FILE_MODE_OPEN_DIR" },
	{ "MODE_OPEN_FILES", "FILE_MODE_OPEN_FILE" },
	{ "MODE_OPEN_FILE", "FILE_MODE_OPEN_FILE" },
	{ "MODE_SAVE_FILE", "FILE_MODE_SAVE_FILE" },
	{ nullptr, nullptr },
};

// Simple renaming functions - "function1" -> "function2"
// Do not add functions which are named same in multiple classes like "start", because this will broke other functions, also
static const char *function_renames[][2] = {
	{ nullptr, nullptr },
};

static const char *properties_renames[][2] = {
	{ nullptr, nullptr },
};

static const char *shaders_renames[][2] = {
	{ "NORMALMAP", "NORMAL_MAP" },
	{ nullptr, nullptr },
};

static const char *gdscript_keywords_renames[][2] = {
	{ "onready", "@onready" },
	{ nullptr, nullptr },
};

static const char *class_renames[][2] = {

	{ "ARVRAnchor", "XRAnchor3D" },
	{ "ARVRCamera", "XRCamera3D" },
	{ "ARVRController", "XRController3D" },
	{ "ARVRInterface", "XRInterface" },
	{ "ARVROrigin", "XROrigin3D" },
	{ "ARVRPositionalTracker", "XRPositionalTracker" },
	{ "ARVRServer", "XRServer" },
	{ "AnimatedSprite", "AnimatedSprite2D" },
	{ "AnimationTreePlayer", "AnimationTree" },
	{ "Area", "Area3D" },
	{ "BakedLightmap", "LightmapGI" },
	{ "BakedLightmapData", "LightmapGIData" },
	{ "BitmapFont", "Font" },
	{ "Bone", "Bone3D" },
	{ "BoneAttachment", "BoneAttachment3D" },
	{ "BoxShape", "BoxShape3D" },
	{ "BulletPhysicsDirectBodyState", "BulletPhysicsDirectBodyState3D" },
	{ "BulletPhysicsServer", "BulletPhysicsServer3D" },
	{ "ButtonList", "MouseButton" },
	{ "CPUParticles", "CPUParticles3D" },
	{ "CSGBox", "CSGBox3D" },
	{ "CSGCombiner", "CSGCombiner3D" },
	{ "CSGCylinder", "CSGCylinder3D" },
	{ "CSGMesh", "CSGMesh3D" },
	{ "CSGPolygon", "CSGPolygon3D" },
	{ "CSGPrimitive", "CSGPrimitive3D" },
	{ "CSGShape", "CSGShape3D" },
	{ "CSGSphere", "CSGSphere3D" },
	{ "CSGTorus", "CSGTorus3D" },
	{ "Camera", "Camera3D" },
	{ "CapsuleShape", "CapsuleShape3D" },
	{ "ClippedCamera", "ClippedCamera3D" },
	{ "CollisionObject", "CollisionObject3D" },
	{ "CollisionPolygon", "CollisionPolygon3D" },
	{ "CollisionShape", "CollisionShape3D" },
	{ "ConcavePolygonShape", "ConcavePolygonShape3D" },
	{ "ConeTwistJoint", "ConeTwistJoint3D" },
	{ "ConvexPolygonShape", "ConvexPolygonShape3D" },
	{ "CubeMesh", "BoxMesh" },
	{ "CylinderShape", "CylinderShape3D" },
	{ "DirectionalLight", "DirectionalLight3D" },
	{ "DynamicFont", "Font" },
	{ "DynamicFontData", "FontData" },
	{ "EditorSpatialGizmo", "EditorNode3DGizmo" },
	{ "EditorSpatialGizmoPlugin", "EditorNode3DGizmoPlugin" },
	{ "GIProbe", "VoxelGI" },
	{ "GIProbeData", "VoxelGIData" },
	{ "Generic6DOFJoint", "Generic6DOFJoint3D" },
	{ "GeometryInstance", "GeometryInstance3D" },
	{ "HeightMapShape", "HeightMapShape3D" },
	{ "HingeJoint", "HingeJoint3D" },
	{ "IP_Unix", "IPUnix" },
	{ "ImmediateGeometry", "ImmediateGeometry3D" },
	{ "InterpolatedCamera", "InterpolatedCamera3D" },
	{ "Joint", "Joint3D" },
	{ "KinematicBody", "CharacterBody3D" },
	{ "KinematicBody2D", "CharacterBody2D" },
	{ "KinematicCollision", "KinematicCollision3D" },
	{ "LargeTexture", "ImageTexture" }, //Missing alternative, so probably it good to choose any function
	{ "Light", "Light3D" },
	{ "Light2D", "PointLight2D" },
	{ "LineShape2D", "WorldMarginShape2D" },
	{ "Listener", "Listener3D" },
	{ "MeshInstance", "MeshInstance3D" },
	{ "MultiMeshInstance", "MultiMeshInstance3D" },
	{ "Navigation", "Node3D" }, //Missing alternative?
	{ "Navigation2D", "Node2D" },
	{ "Navigation2DServer", "NavigationServer2D" },
	{ "Navigation3D", "Node3D" },
	{ "NavigationAgent", "NavigationAgent3D" },
	{ "NavigationMeshInstance", "NavigationRegion3D" },
	{ "NavigationObstacle", "NavigationObstacle3D" },
	{ "NavigationPolygonInstance", "NavigationRegion2D" },
	{ "NavigationRegion", "NavigationRegion3D" },
	{ "NavigationServer", "NavigationServer3D" },
	{ "OmniLight", "OmniLight3D" },
	{ "PHashTranslation", "OptimizedTranslation" },
	{ "PanoramaSky", "Sky" },
	{ "Particles", "GPUParticles3D" },
	{ "Particles2D", "GPUParticles2D" },
	{ "Path", "Path3D" },
	{ "PathFollow", "PathFollow3D" },
	{ "PhysicalBone", "PhysicalBone3D" },
	{ "Physics2DDirectBodyState", "PhysicsDirectBodyState2D" },
	{ "Physics2DDirectBodyStateSW", "PhysicsDirectBodyState2DSW" },
	{ "Physics2DDirectSpaceState", "PhysicsDirectSpaceState2D" },
	{ "Physics2DServer", "PhysicsServer2D" },
	{ "Physics2DServerSW", "PhysicsServer2DSW" },
	{ "Physics2DShapeQueryParameters", "PhysicsShapeQueryParameters2D" },
	{ "Physics2DShapeQueryResult", "PhysicsShapeQueryResult2D" },
	{ "Physics2DTestMotionResult", "PhysicsTestMotionResult2D" },
	{ "PhysicsBody", "PhysicsBody3D" },
	{ "PhysicsDirectBodyState", "PhysicsDirectBodyState3D" },
	{ "PhysicsDirectSpaceState", "PhysicsDirectSpaceState3D" },
	{ "PhysicsServer", "PhysicsServer3D" },
	{ "PhysicsShapeQueryParameters", "PhysicsShapeQueryParameters3D" },
	{ "PhysicsShapeQueryResult", "PhysicsShapeQueryResult3D" },
	{ "PinJoint", "PinJoint3D" },
	{ "PlaneShape", "WorldMarginShape3D" },
	{ "PoolByteArray", "PackedByteArray" },
	{ "PoolColorArray", "PackedColorArray" },
	{ "PoolIntArray", "PackedInt32Array" },
	{ "PoolRealArray", "PackedFloat32Array" },
	{ "PoolStringArray", "PackedStringArray" },
	{ "PoolVector2Array", "PackedVector2Array" },
	{ "PoolVector3Array", "PackedVector3Array" },
	{ "PopupDialog", "Popup" },
	{ "ProceduralSky", "Sky" },
	{ "ProximityGroup", "ProximityGroup3D" },
	{ "Quat", "Quaternion" },
	{ "RayCast", "RayCast3D" },
	{ "RemoteTransform", "RemoteTransform3D" },
	{ "RigidBody", "RigidBody3D" },
	{ "Shape", "Shape3D" },
	{ "ShortCut", "Shortcut" },
	{ "Skeleton", "Skeleton3D" },
	{ "SkeletonIK", "SkeletonIK3D" },
	{ "SliderJoint", "SliderJoint3D" },
	{ "SoftBody", "SoftBody3D" },
	{ "Spatial", "Node3D" },
	{ "SpatialGizmo", "Node3DGizmo" },
	{ "SpatialMaterial", "StandardMaterial3D" },
	{ "SpatialVelocityTracker", "VelocityTracker3D" },
	{ "SphereShape", "SphereShape3D" },
	{ "SpotLight", "SpotLight3D" },
	{ "SpringArm", "SpringArm3D" },
	{ "Sprite", "Sprite2D" },
	{ "StaticBody", "StaticBody3D" },
	{ "StreamTexture", "StreamTexture2D" },
	{ "TCP_Server", "TCPServer" },
	{ "Texture", "Texture2D" },
	{ "TextureArray", "Texture2DArray" },
	{ "TextureProgress", "TextureProgressBar" },
	{ "ToolButton", "Button" },
	{ "Transform", "Transform3D" },
	{ "VehicleBody", "VehicleBody3D" },
	{ "VehicleWheel", "VehicleWheel3D" },
	{ "Viewport", "SubViewport" },
	{ "ViewportContainer", "SubViewportContainer" },
	{ "VisibilityEnabler", "VisibleOnScreenEnabler3D" },
	{ "VisibilityEnabler2D", "VisibleOnScreenEnabler2D" },
	{ "VisibilityNotifier", "VisibleOnScreenNotifier3D" },
	{ "VisibilityNotifier2D", "VisibleOnScreenNotifier2D" },
	{ "VisibilityNotifier3D", "VisibleOnScreenNotifier3D" },
	{ "VisualInstance", "VisualInstance3D" },
	{ "VisualServer", "RenderingServer" },
	{ "VisualShaderNodeScalarClamp", "VisualShaderNodeClamp" },
	{ "VisualShaderNodeScalarConstant", "VisualShaderNodeFloatConstant" },
	{ "VisualShaderNodeScalarFunc", "VisualShaderNodeFloatFunc" },
	{ "VisualShaderNodeScalarInterp", "VisualShaderNodeMix" },
	{ "VisualShaderNodeScalarOp", "VisualShaderNodeFloatOp" },
	{ "VisualShaderNodeScalarSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeScalarSwitch", "VisualShaderNodeSwitch" },
	{ "VisualShaderNodeScalarTransformMult", "VisualShaderNodeTransformOp" },
	{ "VisualShaderNodeScalarUniform", "VisualShaderNodeFloatUniform" },
	{ "VisualShaderNodeVectorClamp", "VisualShaderNodeClamp" },
	{ "VisualShaderNodeVectorInterp", "VisualShaderNodeMix" },
	{ "VisualShaderNodeVectorScalarMix", "VisualShaderNodeMix" },
	{ "VisualShaderNodeVectorScalarSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeVectorScalarStep", "VisualShaderNodeStep" },
	{ "VisualShaderNodeVectorSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "World", "World3D" },
	{ "XRAnchor", "XRAnchor3D" },
	{ "XRController", "XRController3D" },
	{ "XROrigin", "XROrigin3D" },
	{ "YSort", "Node2D" }, //Missing alternative?
	{ "Geometry", "Geometry2D" }, // Geometry class is split between Geometry2D and Geometry3D so we need to choose one
	{ "PhysicsTestMotionResult", "PhysicsTestMotionResult2D" }, // PhysicsTestMotionResult class is split between PhysicsTestMotionResult2D and PhysicsTestMotionResult3D so we need to choose one
	{ "ExternalTexture", "ImageTexture" }, // ExternalTexture is missing, so we choose ImageTexture as replamencement

	{ "NetworkedMultiplayerPeer", "MultiplayerPeer" },
	{ "WebRTCMultiplayer", "WebRTCMultiplayerPeer" },
	{ "ResourceInteractiveLoader", "ResourceLoader" },

	{ "NetworkedMultiplayerENet", "ENetMultiplayerPeer" },
	{ "JSONParseResult", "JSON" },

	{ "RayShape2D", "RayCast2D" }, // TODO looks that this class is not visible
	{ "RayShape", "RayCast3D" }, // TODO looks that this class is not visible

	{ "WindowDialog", "Window" }, // TODO not sure about it
	{ "InterpolatedCamera3D", "Camera3D" }, // InterpolatedCamera3D is missing so probably the best is to use Camera3D
	{ "ImmediateGeometry3D", "ImmediateMesh" },
	{ "Reference", "RefCounted" },

	{ "VisualShaderNodeCubeMapUniform", "VisualShaderNodeCubemapUniform" },
	{ "VisualShaderNodeCubeMap", "VisualShaderNodeCubemap" },
	{ "CubeMap", "Cubemap" },
	{ "FuncRef", "Callable" },

	{ "VisualShaderNodeTransformMult", "VisualShaderNode" }, // Not sure about it
	{ "GDScriptNativeClass", "Node3D" }, // Not sure about it
	{ "GDScriptFunctionState", "Node3D" }, // Not sure about it
	{ "ARVRInterfaceGDNative", "Node3D" }, // Not sure about it
	{ "TextFile", "Node3D" }, // TextFile was hided

	{ "CullInstance", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "RoomGroup", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "Room", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "RoomManager", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "Portal", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x

	{ "TextFile", "Node3D" },

	{ nullptr, nullptr },
};

static void rename_enums(String &file_content) {
	int current_index = 0;
	while (enum_renames[current_index][0]) {
		// File.MODE_OLD -> File.MODE_NEW
		RegEx reg = RegEx(String("\\b") + enum_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, enum_renames[current_index][1], true);
		current_index++;
	}
};
static void rename_classes(String &file_content) {
	int current_index = 0;

	// TODO for now it changes also e.g. Spatial.tscn -> Node3D.tscn which will broke some scripts
	while (class_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + class_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, class_renames[current_index][1], true);
		current_index++;
	}
};
static void rename_functions(String &file_content) {
	int current_index = 0;
	while (function_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + function_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, function_renames[current_index][1], true);
		current_index++;
	}
};
static void rename_properties(String &file_content) {
	int current_index = 0;
	while (properties_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + properties_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, properties_renames[current_index][1], true);
		current_index++;
	}
};
static void rename_shaders(String &file_content) {
	int current_index = 0;
	while (shaders_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + shaders_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, shaders_renames[current_index][1], true);
		current_index++;
	}
};

static void rename_gdscript_keywords(String &file_content) {
	int current_index = 0;
	while (gdscript_keywords_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + gdscript_keywords_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, gdscript_keywords_renames[current_index][1], true);
		current_index++;
	}
};

// Collect files which will be checked, it will not touch txt, mp4, wav etc. files
static Vector<String> check_for_files() {
	Vector<String> collected_files = Vector<String>();

	Vector<String> directories_to_check = Vector<String>();
	directories_to_check.push_back("res://");

	core_bind::Directory dir = core_bind::Directory();
	while (!directories_to_check.is_empty()) {
		String path = directories_to_check.get(directories_to_check.size() - 1); // Is there any pop_back function?
		directories_to_check.resize(directories_to_check.size() - 1); // Remove last element
		if (dir.open(path) == OK) {
			dir.list_dir_begin();
			String current_dir = dir.get_current_dir();
			String file_name = dir.get_next();

			while (file_name != "") {
				if (file_name == ".." || file_name == "." || file_name == ".") {
					file_name = dir.get_next();
					continue;
				}
				if (dir.current_is_dir()) {
					directories_to_check.append(current_dir + file_name + "/");
				} else {
					bool proper_extension = false;
					// TODO enable all files
					if (file_name.ends_with(".gd") || file_name.ends_with(".shader")) // || file_name.ends_with(".tscn") || file_name.ends_with(".tres") || file_name.ends_with(".godot"))|| file_name.ends_with(".cs"))|| file_name.ends_with(".csproj"))
						proper_extension = true;

					if (proper_extension) {
						collected_files.append(current_dir + file_name);
					}
				}
				file_name = dir.get_next();
			}
		} else {
			print_verbose("Failed to open " + path);
		}
	}
	return collected_files;
}

static void converter() {
	print_line("Starting Converting.");

	// Checking if folder contains valid Godot 3 project.
	// Project cannot be converted 2 times
	{
		String conventer_text = "; Project was converted by built-in tool to Godot 4.0";

		ERR_FAIL_COND_MSG(!FileAccess::exists("project.godot"), "Current directory doesn't contains any Godot 3 project");

		// Check if folder
		Error err = OK;
		String project_godot_content = FileAccess::get_file_as_string("project.godot", &err);

		ERR_FAIL_COND_MSG(err != OK, "Failed to read content of \"project.godot\" file.");
		ERR_FAIL_COND_MSG(project_godot_content.find(conventer_text) != -1, "Project already was converted with this tool.");

		// TODO - Re-enable this after testing

		//		FileAccess *file = FileAccess::open("project.godot", FileAccess::WRITE);
		//		ERR_FAIL_COND_MSG(!file, "Failed to open project.godot file.");

		//		file->store_string(conventer_text + "\n" + project_godot_content);
	}

	Vector<String> collected_files = check_for_files();

	uint32_t converted_files = 0;

	// Check file by file
	for (int i = 0; i < collected_files.size(); i++) {
		String file_name = collected_files[i];
		Error err = OK;
		String file_content = FileAccess::get_file_as_string(file_name, &err);
		ERR_CONTINUE_MSG(err != OK, "Failed to read content of \"" + file_name + "\".");
		uint64_t hash_before = file_content.hash64();

		if (file_name.ends_with(".gd")) {
			rename_gdscript_keywords(file_content);
			rename_classes(file_content);
			rename_enums(file_content);
			rename_functions(file_content);
			rename_properties(file_content);
		} else if (file_name.ends_with(".tscn")) {
			rename_classes(file_content);
		} else if (file_name.ends_with(".cs")) { // TODO, C# should use different methods
			rename_classes(file_content);
		} else if (file_name.ends_with(".shader")) {
			rename_shaders(file_content);
		} else if (file_name.ends_with(".csproj")) {
			//TODO
		} else if (file_name == "project.godot") {
			rename_properties(file_content);
		} else {
			ERR_PRINT(file_name + " is not supported!");
			continue;
		}
		// TODO maybe also rename files

		String changed = "NOT changed";

		uint64_t hash_after = file_content.hash64();

		// Don't need to save file without any changes
		if (hash_before != hash_after) {
			changed = "changed";
			converted_files++;

			FileAccess *file = FileAccess::open(file_name, FileAccess::WRITE);
			ERR_CONTINUE_MSG(!file, "Failed to open \"" + file_name + "\" to save data to file.");
			file->store_string(file_content);
			memdelete(file);
		}

		print_line("Processed " + itos(i + 1) + "/" + itos(collected_files.size()) + " file - " + file_name.trim_prefix("res://") + " was " + changed + ".");
	}

	print_line("Converting ended - all files(" + itos(collected_files.size()) + "), converted files(" + itos(converted_files) + "), not converted files(" + itos(collected_files.size() - converted_files) + ").");
};

#endif // CONVERTER_H
