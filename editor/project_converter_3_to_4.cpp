/*************************************************************************/
/*  project_converter_3_to_4.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "project_converter_3_to_4.h"

#include "modules/modules_enabled.gen.h"

const int ERROR_CODE = 77;

#ifdef MODULE_REGEX_ENABLED

#include "modules/regex/regex.h"

#include "core/os/time.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"

const uint64_t CONVERSION_MAX_FILE_SIZE_MB = 4;
const uint64_t CONVERSION_MAX_FILE_SIZE = 1024 * 1024 * CONVERSION_MAX_FILE_SIZE_MB;

static const char *enum_renames[][2] = {
	//// constants
	{ "TYPE_COLOR_ARRAY", "TYPE_PACKED_COLOR_ARRAY" },
	{ "TYPE_FLOAT64_ARRAY", "TYPE_PACKED_FLOAT64_ARRAY" },
	{ "TYPE_INT64_ARRAY", "TYPE_PACKED_INT64_ARRAY" },
	{ "TYPE_INT_ARRAY", "TYPE_PACKED_INT32_ARRAY" },
	{ "TYPE_QUAT", "TYPE_QUATERNION" },
	{ "TYPE_RAW_ARRAY", "TYPE_PACKED_BYTE_ARRAY" },
	{ "TYPE_REAL", "TYPE_FLOAT" },
	{ "TYPE_REAL_ARRAY", "TYPE_PACKED_FLOAT32_ARRAY" },
	{ "TYPE_STRING_ARRAY", "TYPE_PACKED_STRING_ARRAY" },
	{ "TYPE_TRANSFORM", "TYPE_TRANSFORM3D" },
	{ "TYPE_VECTOR2_ARRAY", "TYPE_PACKED_VECTOR2_ARRAY" },
	{ "TYPE_VECTOR3_ARRAY", "TYPE_PACKED_VECTOR3_ARRAY" },

	// {"FLAG_MAX", "PARTICLE_FLAG_MAX"}, // CPUParticles2D - used in more classes
	{ "ALIGN_BEGIN", "ALIGNMENT_BEGIN" }, //AspectRatioContainer
	{ "ALIGN_CENTER", "ALIGNMENT_CENTER" }, //AspectRatioContainer
	{ "ALIGN_END", "ALIGNMENT_END" }, //AspectRatioContainer
	{ "ARRAY_COMPRESS_BASE", "ARRAY_COMPRESS_FLAGS_BASE" }, // Mesh
	{ "ARVR_AR", "XR_AR" }, // XRInterface
	{ "ARVR_EXCESSIVE_MOTION", "XR_EXCESSIVE_MOTION" }, // XRInterface
	{ "ARVR_EXTERNAL", "XR_EXTERNAL" }, // XRInterface
	{ "ARVR_INSUFFICIENT_FEATURES", "XR_INSUFFICIENT_FEATURES" }, // XRInterface
	{ "ARVR_MONO", "XR_MONO" }, // XRInterface
	{ "ARVR_NONE", "XR_NONE" }, // XRInterface
	{ "ARVR_NORMAL_TRACKING", "XR_NORMAL_TRACKING" }, // XRInterface
	{ "ARVR_NOT_TRACKING", "XR_NOT_TRACKING" }, // XRInterface
	{ "ARVR_STEREO", "XR_STEREO" }, // XRInterface
	{ "ARVR_UNKNOWN_TRACKING", "XR_UNKNOWN_TRACKING" }, // XRInterface
	{ "BAKE_ERROR_INVALID_MESH", "BAKE_ERROR_MESHES_INVALID" }, // LightmapGI
	{ "BODY_MODE_CHARACTER", "BODY_MODE_DYNAMIC" }, // PhysicsServer2D
	{ "BODY_MODE_DYNAMIC_LOCKED", "BODY_MODE_DYNAMIC_LINEAR" }, // PhysicsServer3D
	{ "BUTTON_LEFT", "MOUSE_BUTTON_LEFT" }, // Globals
	{ "BUTTON_MASK_LEFT", "MOUSE_BUTTON_MASK_LEFT" }, // Globals
	{ "BUTTON_MASK_MIDDLE", "MOUSE_BUTTON_MASK_MIDDLE" }, // Globals
	{ "BUTTON_MASK_RIGHT", "MOUSE_BUTTON_MASK_RIGHT" }, // Globals
	{ "BUTTON_MASK_XBUTTON1", "MOUSE_BUTTON_MASK_XBUTTON1" }, // Globals
	{ "BUTTON_MASK_XBUTTON2", "MOUSE_BUTTON_MASK_XBUTTON2" }, // Globals
	{ "BUTTON_MIDDLE", "MOUSE_BUTTON_MIDDLE" }, // Globals
	{ "BUTTON_RIGHT", "MOUSE_BUTTON_RIGHT" }, // Globals
	{ "BUTTON_WHEEL_DOWN", "MOUSE_BUTTON_WHEEL_DOWN" }, // Globals
	{ "BUTTON_WHEEL_LEFT", "MOUSE_BUTTON_WHEEL_LEFT" }, // Globals
	{ "BUTTON_WHEEL_RIGHT", "MOUSE_BUTTON_WHEEL_RIGHT" }, // Globals
	{ "BUTTON_WHEEL_UP", "MOUSE_BUTTON_WHEEL_UP" }, // Globals
	{ "BUTTON_XBUTTON1", "MOUSE_BUTTON_XBUTTON1" }, // Globals
	{ "BUTTON_XBUTTON2", "MOUSE_BUTTON_XBUTTON2" }, // Globals
	{ "CLEAR_MODE_ONLY_NEXT_FRAME", "CLEAR_MODE_ONCE" }, // SubViewport
	{ "COMPRESS_PVRTC4", "COMPRESS_PVRTC1_4" }, // Image
	{ "CUBEMAP_BACK", "CUBEMAP_LAYER_BACK" }, // RenderingServer
	{ "CUBEMAP_BOTTOM", "CUBEMAP_LAYER_BOTTOM" }, // RenderingServer
	{ "CUBEMAP_FRONT", "CUBEMAP_LAYER_FRONT" }, // RenderingServer
	{ "CUBEMAP_LEFT", "CUBEMAP_LAYER_LEFT" }, // RenderingServer
	{ "CUBEMAP_RIGHT", "CUBEMAP_LAYER_RIGHT" }, // RenderingServer
	{ "CUBEMAP_TOP", "CUBEMAP_LAYER_TOP" }, // RenderingServer
	{ "DAMPED_STRING_DAMPING", "DAMPED_SPRING_DAMPING" }, // PhysicsServer2D
	{ "DAMPED_STRING_REST_LENGTH", "DAMPED_SPRING_REST_LENGTH" }, // PhysicsServer2D
	{ "DAMPED_STRING_STIFFNESS", "DAMPED_SPRING_STIFFNESS" }, // PhysicsServer2D
	{ "FLAG_ALIGN_Y_TO_VELOCITY", "PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY" }, // CPUParticles2D
	{ "FLAG_DISABLE_Z", "PARTICLE_FLAG_DISABLE_Z" }, // CPUParticles2D
	{ "FLAG_ROTATE_Y", "PARTICLE_FLAG_ROTATE_Y" }, // CPUParticles2D
	{ "FLAG_USE_BAKED_LIGHT", "GI_MODE_BAKED" }, // GeometryInstance3D
	{ "FORMAT_PVRTC2", "FORMAT_PVRTC1_2" }, // Image
	{ "FORMAT_PVRTC2A", "FORMAT_PVRTC1_2A" }, // Image
	{ "FORMAT_PVRTC4", "FORMAT_PVRTC1_4" }, // Image
	{ "FORMAT_PVRTC4A", "FORMAT_PVRTC1_4A" }, // Image
	{ "FUNC_FRAC", "FUNC_FRACT" }, // VisualShaderNodeVectorFunc
	{ "INSTANCE_LIGHTMAP_CAPTURE", "INSTANCE_LIGHTMAP" }, // RenderingServer
	{ "JOINT_6DOF", "JOINT_TYPE_6DOF" }, // PhysicsServer3D
	{ "JOINT_CONE_TWIST", "JOINT_TYPE_CONE_TWIST" }, // PhysicsServer3D
	{ "JOINT_DAMPED_SPRING", "JOINT_TYPE_DAMPED_SPRING" }, // PhysicsServer2D
	{ "JOINT_GROOVE", "JOINT_TYPE_GROOVE" }, // PhysicsServer2D
	{ "JOINT_HINGE", "JOINT_TYPE_HINGE" }, // PhysicsServer3D
	{ "JOINT_PIN", "JOINT_TYPE_PIN" }, // PhysicsServer2D
	{ "JOINT_SLIDER", "JOINT_TYPE_SLIDER" }, // PhysicsServer3D
	{ "KEY_CONTROL", "KEY_CTRL" }, // Globals
	{ "LOOP_PING_PONG", "LOOP_PINGPONG" }, // AudioStreamWAV
	{ "MATH_RAND", "MATH_RANDF_RANGE" }, // VisualScriptBuiltinFunc
	{ "MATH_RANDOM", "MATH_RANDI_RANGE" }, // VisualScriptBuiltinFunc
	{ "MATH_STEPIFY", "MATH_STEP_DECIMALS" }, // VisualScriptBuiltinFunc
	{ "MODE_CHARACTER", "MODE_DYNAMIC_LOCKED" }, // RigidBody2D, RigidBody3D
	{ "MODE_KINEMATIC", "FREEZE_MODE_KINEMATIC" }, // RigidDynamicBody
	{ "MODE_OPEN_ANY", "FILE_MODE_OPEN_ANY" }, // FileDialog
	{ "MODE_OPEN_DIR", "FILE_MODE_OPEN_DIR" }, // FileDialog
	{ "MODE_OPEN_FILE", "FILE_MODE_OPEN_FILE" }, // FileDialog
	{ "MODE_OPEN_FILES", "FILE_MODE_OPEN_FILES" }, // FileDialog
	{ "MODE_RIGID", "MODE_DYNAMIC" }, // RigidBody2D, RigidBody3D
	{ "MODE_SAVE_FILE", "FILE_MODE_SAVE_FILE" }, // FileDialog
	{ "MODE_STATIC", "FREEZE_MODE_STATIC" }, // RigidDynamicBody
	{ "NOTIFICATION_APP_PAUSED", "NOTIFICATION_APPLICATION_PAUSED" }, // MainLoop
	{ "NOTIFICATION_APP_RESUMED", "NOTIFICATION_APPLICATION_RESUMED" }, // MainLoop
	{ "NOTIFICATION_PATH_CHANGED", "NOTIFICATION_PATH_RENAMED" }, //Node
	{ "NOTIFICATION_WM_FOCUS_IN", "NOTIFICATION_APPLICATION_FOCUS_IN" }, // MainLoop
	{ "NOTIFICATION_WM_FOCUS_OUT", "NOTIFICATION_APPLICATION_FOCUS_OUT" }, // MainLoop
	{ "NOTIFICATION_WM_UNFOCUS_REQUEST", "NOTIFICATION_WM_WINDOW_FOCUS_OUT" }, //Node
	{ "PAUSE_MODE_INHERIT", "PROCESS_MODE_INHERIT" }, // Node
	{ "PAUSE_MODE_PROCESS", "PROCESS_MODE_ALWAYS" }, // Node
	{ "PAUSE_MODE_STOP", "PROCESS_MODE_PAUSABLE" }, // Node
	{ "RENDER_DRAW_CALLS_IN_FRAME", "RENDER_TOTAL_DRAW_CALLS_IN_FRAME" }, // Performance
	{ "RENDER_OBJECTS_IN_FRAME", "RENDER_TOTAL_OBJECTS_IN_FRAME" }, // Performance
	{ "SIDE_BOTTOM", "MARGIN_BOTTOM" }, // Globals
	{ "SIDE_LEFT", "MARGIN_LEFT" }, // Globals
	{ "SIDE_RIGHT", "MARGIN_RIGHT" }, // Globals
	{ "SIDE_TOP", "MARGIN_TOP" }, // Globals
	{ "TEXTURE_TYPE_2D_ARRAY", "TEXTURE_LAYERED_2D_ARRAY" }, // RenderingServer
	{ "TEXTURE_TYPE_CUBEMAP", "TEXTURE_LAYERED_CUBEMAP_ARRAY" }, // RenderingServer
	{ "TRACKER_LEFT_HAND", "TRACKER_HAND_LEFT" }, // XRPositionalTracker
	{ "TRACKER_RIGHT_HAND", "TRACKER_HAND_RIGHT" }, // XRPositionalTracker
	{ "TYPE_NORMALMAP", "TYPE_NORMAL_MAP" }, // VisualShaderNodeCubemap

	/// enums
	{ "AlignMode", "AlignmentMode" }, //AspectRatioContainer
	{ "AnimationProcessMode", "AnimationProcessCallback" }, // AnimationTree, AnimationPlayer
	{ "Camera2DProcessMode", "Camera2DProcessCallback" }, // Camera2D
	{ "CubeMapSide", "CubeMapLayer" }, // RenderingServer
	{ "DampedStringParam", "DampedSpringParam" }, // PhysicsServer2D
	{ "FFT_Size", "FFTSize" }, // AudioEffectPitchShift,AudioEffectSpectrumAnalyzer
	{ "PauseMode", "ProcessMode" }, // Node
	{ "TimerProcessMode", "TimerProcessCallback" }, // Timer
	{ "Tracking_status", "TrackingStatus" }, // XRInterface
	{ nullptr, nullptr },
};

static const char *gdscript_function_renames[][2] = {
	// { "_set_name", "get_tracker_name"}, // XRPositionalTracker - CameraFeed use this
	// { "_unhandled_input", "_unhandled_key_input"}, // BaseButton, ViewportContainer broke Node, FileDialog,SubViewportContainer
	// { "create_gizmo", "_create_gizmo"}, // EditorNode3DGizmoPlugin - may be used
	// { "get_dependencies", "_get_dependencies" }, // ResourceFormatLoader broke ResourceLoader
	// { "get_extents", "get_size" }, // BoxShape, RectangleShape broke Decal, VoxelGI, GPUParticlesCollisionBox, GPUParticlesCollisionSDF, GPUParticlesCollisionHeightField, GPUParticlesAttractorBox, GPUParticlesAttractorVectorField, FogVolume
	// { "get_h_offset", "get_drag_horizontal_offset"}, // Camera2D, broke PathFollow, Camera
	// { "get_mode", "get_file_mode"}, // FileDialog broke Panel, Shader, CSGPolygon, Tilemap
	// { "get_motion", "get_travel"}, // PhysicsTestMotionResult2D broke ParalaxLayer
	// { "get_name", "get_tracker_name"}, // XRPositionalTracker broke OS, Node
	// { "get_network_connected_peers", "get_peers"}, // MultiplayerAPI broke SceneTree
	// { "get_network_peer", "has_multiplayer_peer"}, // MultiplayerAPI broke SceneTree
	// { "get_network_unique_id", "get_unique_id"}, // MultiplayerAPI broke SceneTree
	// { "get_offset", "get_position_offset" }, // GraphNode broke Gradient
	// { "get_peer_port", "get_peer" }, // ENetMultiplayerPeer broke WebSocketServer
	// { "get_process_mode", "get_process_callback" }, // ClippedCamera3D broke Node, Sky
	// { "get_render_info", "get_rendering_info" }, // RenderingServer broke Viewport
	// { "get_type", "get_tracker_type"}, // XRPositionalTracker broke GLTFAccessor, GLTFLight
	// { "get_v_offset", "get_drag_vertical_offset"}, // Camera2D, broke PathFollow, Camera
	// { "has_network_peer", "has_multiplayer_peer"}, // MultiplayerAPI broke SceneTree
	// { "instance", "instantiate" }, // PackedScene, ClassDB - Broke FileSystemDock signal and also tscn files - [instance=ExtResource( 17 )] - this is implemented as custom rule
	// { "is_listening", "is_bound"}, // PacketPeerUDP broke TCPServer, UDPServer
	// { "is_refusing_new_network_connections", "is_refusing_new_connections"}, // MultiplayerAPI broke SceneTree
	// { "is_valid", "has_valid_event" }, // Shortcut broke e.g. Callable
	// { "listen", "bound"}, // PacketPeerUDP broke TCPServer, UDPServer
	// { "load", "_load"}, // ResourceFormatLoader broke ConfigFile, Image, StreamTexture2D
	// { "make_current", "set_current" }, // Camera2D broke Camera3D, Listener2D
	// { "process", "_process" }, // AnimationNode - This word is commonly used
	// { "save", "_save"}, // ResourceFormatLoader broke ConfigFile, Image, StreamTexture2D
	// { "set_autowrap", "set_autowrap_mode" }, // AcceptDialog broke Label - Cyclic Rename
	// { "set_color", "surface_set_color"}, // ImmediateMesh broke Light2D, Theme, SurfaceTool
	// { "set_event", "set_shortcut" }, // BaseButton - Cyclic Rename
	// { "set_extents", "set_size"}, // BoxShape, RectangleShape broke ReflectionProbe
	// { "set_flag", "set_particle_flag"}, // ParticlesMaterial broke Window, HingeJoint3D
	// { "set_h_offset", "set_drag_horizontal_offset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// { "set_margin", "set_offset" }, // Control broke Shape3D, AtlasTexture
	// { "set_mode", "set_mode_file_mode" }, // FileDialog broke Panel, Shader, CSGPolygon, Tilemap
	// { "set_normal", "surface_set_normal"}, // ImmediateGeometry broke SurfaceTool, WorldMarginShape2D
	// { "set_process_mode", "set_process_callback" }, // AnimationTree broke Node, Tween, Sky
	// { "set_refuse_new_network_connections", "set_refuse_new_connections"}, // MultiplayerAPI broke SceneTree
	// { "set_uv", "surface_set_uv" }, // ImmediateMesh broke Polygon2D
	// { "set_v_offset", "set_drag_vertical_offset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// {"get_points","get_points_id"},// Astar, broke Line2D, Convexpolygonshape
	// {"get_v_scroll","get_v_scroll_bar"},//ItemList, broke TextView
	{ "_about_to_show", "_about_to_popup" }, // ColorPickerButton
	{ "_get_configuration_warning", "_get_configuration_warnings" }, // Node
	{ "_set_current", "set_current" }, // Camera2D
	{ "_set_editor_description", "set_editor_description" }, // Node
	{ "_toplevel_raise_self", "_top_level_raise_self" }, // CanvasItem
	{ "_update_wrap_at", "_update_wrap_at_column" }, // TextEdit
	{ "add_animation", "add_animation_library" }, // AnimationPlayer
	{ "add_cancel", "add_cancel_button" }, // AcceptDialog
	{ "add_central_force", "apply_central_force" }, //RigidDynamicBody2D
	{ "add_child_below_node", "add_sibling" }, // Node
	{ "add_color_override", "add_theme_color_override" }, // Control
	{ "add_constant_override", "add_theme_constant_override" }, // Control
	{ "add_font_override", "add_theme_font_override" }, // Control
	{ "add_force", "apply_force" }, //RigidDynamicBody2D
	{ "add_icon_override", "add_theme_icon_override" }, // Control
	{ "add_scene_import_plugin", "add_scene_format_importer_plugin" }, //EditorPlugin
	{ "add_stylebox_override", "add_theme_stylebox_override" }, // Control
	{ "add_torque", "apply_torque" }, //RigidDynamicBody2D
	{ "apply_changes", "_apply_changes" }, // EditorPlugin
	{ "bind_child_node_to_bone", "set_bone_children" }, // Skeleton3D
	{ "body_add_force", "body_apply_force" }, // PhysicsServer2D
	{ "body_add_torque", "body_apply_torque" }, // PhysicsServer2D
	{ "bumpmap_to_normalmap", "bump_map_to_normal_map" }, // Image
	{ "can_be_hidden", "_can_be_hidden" }, // EditorNode3DGizmoPlugin
	{ "can_drop_data_fw", "_can_drop_data_fw" }, // ScriptEditor
	{ "can_generate_small_preview", "_can_generate_small_preview" }, // EditorResourcePreviewGenerator
	{ "can_instance", "can_instantiate" }, // PackedScene, Script
	{ "canvas_light_set_scale", "canvas_light_set_texture_scale" }, // RenderingServer
	{ "center_viewport_to_cursor", "center_viewport_to_caret" }, // TextEdit
	{ "clip_polygons_2d", "clip_polygons" }, // Geometry2D
	{ "clip_polyline_with_polygon_2d", "clip_polyline_with_polygon" }, //Geometry2D
	{ "commit_handle", "_commit_handle" }, // EditorNode3DGizmo
	{ "convex_hull_2d", "convex_hull" }, // Geometry2D
	{ "create_gizmo", "_create_gizmo" }, // EditorNode3DGizmoPlugin
	{ "cursor_get_blink_speed", "get_caret_blink_speed" }, // TextEdit
	{ "cursor_get_column", "get_caret_column" }, // TextEdit
	{ "cursor_get_line", "get_caret_line" }, // TextEdit
	{ "cursor_set_blink_enabled", "set_caret_blink_enabled" }, // TextEdit
	{ "cursor_set_blink_speed", "set_caret_blink_speed" }, // TextEdit
	{ "cursor_set_column", "set_caret_column" }, // TextEdit
	{ "cursor_set_line", "set_caret_line" }, // TextEdit
	{ "damped_spring_joint_create", "joint_make_damped_spring" }, // PhysicsServer2D
	{ "damped_string_joint_get_param", "damped_spring_joint_get_param" }, // PhysicsServer2D
	{ "damped_string_joint_set_param", "damped_spring_joint_set_param" }, // PhysicsServer2D
	{ "dectime", "move_toward" }, // GDScript, Math functions
	{ "delete_char_at_cursor", "delete_char_at_caret" }, // LineEdit
	{ "deselect_items", "deselect_all" }, // FileDialog
	{ "disable_plugin", "_disable_plugin" }, // EditorPlugin
	{ "drop_data_fw", "_drop_data_fw" }, // ScriptEditor
	{ "exclude_polygons_2d", "exclude_polygons" }, // Geometry2D
	{ "find_node", "find_child" }, // Node
	{ "find_scancode_from_string", "find_keycode_from_string" }, // OS
	{ "forward_canvas_draw_over_viewport", "_forward_canvas_draw_over_viewport" }, // EditorPlugin
	{ "forward_canvas_force_draw_over_viewport", "_forward_canvas_force_draw_over_viewport" }, // EditorPlugin
	{ "forward_canvas_gui_input", "_forward_canvas_gui_input" }, // EditorPlugin
	{ "forward_spatial_draw_over_viewport", "_forward_3d_draw_over_viewport" }, // EditorPlugin
	{ "forward_spatial_force_draw_over_viewport", "_forward_3d_force_draw_over_viewport" }, // EditorPlugin
	{ "forward_spatial_gui_input", "_forward_3d_gui_input" }, // EditorPlugin
	{ "generate_from_path", "_generate_from_path" }, // EditorResourcePreviewGenerator
	{ "generate_small_preview_automatically", "_generate_small_preview_automatically" }, // EditorResourcePreviewGenerator
	{ "get_action_list", "action_get_events" }, // InputMap
	{ "get_alt", "is_alt_pressed" }, // InputEventWithModifiers
	{ "get_animation_process_mode", "get_process_callback" }, // AnimationPlayer
	{ "get_applied_force", "get_constant_force" }, //RigidDynamicBody2D
	{ "get_applied_torque", "get_constant_torque" }, //RigidDynamicBody2D
	{ "get_audio_bus", "get_audio_bus_name" }, // Area3D
	{ "get_bound_child_nodes_to_bone", "get_bone_children" }, // Skeleton3D
	{ "get_camera", "get_camera_3d" }, // Viewport -> this is also convertable to get_camera_2d, broke GLTFNode
	{ "get_cancel", "get_cancel_button" }, // ConfirmationDialog
	{ "get_caption", "_get_caption" }, // AnimationNode
	{ "get_cast_to", "get_target_position" }, // RayCast2D, RayCast3D
	{ "get_child_by_name", "_get_child_by_name" }, // AnimationNode
	{ "get_child_nodes", "_get_child_nodes" }, // AnimationNode
	{ "get_closest_point_to_segment_2d", "get_closest_point_to_segment" }, // Geometry2D
	{ "get_closest_point_to_segment_uncapped_2d", "get_closest_point_to_segment_uncapped" }, // Geometry2D
	{ "get_closest_points_between_segments_2d", "get_closest_point_to_segment" }, // Geometry2D
	{ "get_collision_layer_bit", "get_collision_layer_value" }, // CSGShape3D and a lot of others like GridMap
	{ "get_collision_mask_bit", "get_collision_mask_value" }, // CSGShape3D and a lot of others like GridMap
	{ "get_color_types", "get_color_type_list" }, // Theme
	{ "get_command", "is_command_pressed" }, // InputEventWithModifiers
	{ "get_constant_types", "get_constant_type_list" }, // Theme
	{ "get_control", "is_ctrl_pressed" }, // InputEventWithModifiers
	{ "get_cull_mask_bit", "get_cull_mask_value" }, // Camera3D
	{ "get_cursor_position", "get_caret_column" }, // LineEdit
	{ "get_d", "get_distance" }, // LineShape2D
	{ "get_drag_data", "_get_drag_data" }, // Control
	{ "get_drag_data_fw", "_get_drag_data_fw" }, // ScriptEditor
	{ "get_editor_description", "_get_editor_description" }, // Node
	{ "get_editor_viewport", "get_viewport" }, // EditorPlugin
	{ "get_enabled_focus_mode", "get_focus_mode" }, // BaseButton
	{ "get_endian_swap", "is_big_endian" }, // File
	{ "get_error_string", "get_error_message" }, // JSON
	{ "get_focus_neighbour", "get_focus_neighbor" }, // Control
	{ "get_font_types", "get_font_type_list" }, // Theme
	{ "get_frame_color", "get_color" }, // ColorRect
	{ "get_global_rate_scale", "get_playback_speed_scale" }, // AudioServer
	{ "get_gravity_distance_scale", "get_gravity_point_distance_scale" }, //Area2D
	{ "get_gravity_vector", "get_gravity_direction" }, //Area2D
	{ "get_h_scrollbar", "get_h_scroll_bar" }, //ScrollContainer
	{ "get_hand", "get_tracker_hand" }, // XRPositionalTracker
	{ "get_handle_name", "_get_handle_name" }, // EditorNode3DGizmo
	{ "get_handle_value", "_get_handle_value" }, // EditorNode3DGizmo
	{ "get_icon_align", "get_icon_alignment" }, // Button
	{ "get_icon_types", "get_icon_type_list" }, // Theme
	{ "get_idle_frames", "get_process_frames" }, // Engine
	{ "get_import_options", "_get_import_options" }, // EditorImportPlugin
	{ "get_import_order", "_get_import_order" }, // EditorImportPlugin
	{ "get_importer_name", "_get_importer_name" }, // EditorImportPlugin
	{ "get_interior_ambient", "get_ambient_color" }, // ReflectionProbe
	{ "get_interior_ambient_energy", "get_ambient_color_energy" }, // ReflectionProbe
	{ "get_iterations_per_second", "get_physics_ticks_per_second" }, // Engine
	{ "get_last_mouse_speed", "get_last_mouse_velocity" }, // Input
	{ "get_layer_mask_bit", "get_layer_mask_value" }, // VisualInstance3D
	{ "get_len", "get_length" }, // File
	{ "get_max_atlas_size", "get_max_texture_size" }, // LightmapGI
	{ "get_metakey", "is_meta_pressed" }, // InputEventWithModifiers
	{ "get_mid_height", "get_height" }, // CapsuleMesh
	{ "get_motion_remainder", "get_remainder" }, // PhysicsTestMotionResult2D
	{ "get_network_connected_peers", "get_peers" }, // Multiplayer API
	{ "get_network_master", "get_multiplayer_authority" }, // Node
	{ "get_network_peer", "get_multiplayer_peer" }, // Multiplayer API
	{ "get_network_unique_id", "get_unique_id" }, // Multiplayer API
	{ "get_ok", "get_ok_button" }, // AcceptDialog
	{ "get_option_visibility", "_get_option_visibility" }, // EditorImportPlugin
	{ "get_parameter_default_value", "_get_parameter_default_value" }, // AnimationNode
	{ "get_parameter_list", "_get_parameter_list" }, // AnimationNode
	{ "get_parent_spatial", "get_parent_node_3d" }, // Node3D
	{ "get_pause_mode", "get_process_mode" }, // Node
	{ "get_physical_scancode", "get_physical_keycode" }, // InputEventKey
	{ "get_physical_scancode_with_modifiers", "get_physical_keycode_with_modifiers" }, // InputEventKey
	{ "get_plugin_icon", "_get_plugin_icon" }, // EditorPlugin
	{ "get_plugin_name", "_get_plugin_name" }, // EditorPlugin
	{ "get_preset_count", "_get_preset_count" }, // EditorImportPlugin
	{ "get_preset_name", "_get_preset_name" }, // EditorImportPlugin
	{ "get_recognized_extensions", "_get_recognized_extensions" }, // ResourceFormatLoader, EditorImportPlugin broke ResourceSaver
	{ "get_render_info", "get_rendering_info" }, // RenderingServer
	{ "get_render_targetsize", "get_render_target_size" }, // XRInterface
	{ "get_resource_type", "_get_resource_type" }, // ResourceFormatLoader
	{ "get_result", "get_data" }, //JSON
	{ "get_rpc_sender_id", "get_remote_sender_id" }, // Multiplayer API
	{ "get_save_extension", "_get_save_extension" }, // EditorImportPlugin
	{ "get_scancode", "get_keycode" }, // InputEventKey
	{ "get_scancode_string", "get_keycode_string" }, // OS
	{ "get_scancode_with_modifiers", "get_keycode_with_modifiers" }, // InputEventKey
	{ "get_shift", "is_shift_pressed" }, // InputEventWithModifiers
	{ "get_size_override", "get_size_2d_override" }, // SubViewport
	{ "get_slide_count", "get_slide_collision_count" }, // CharacterBody2D, CharacterBody3D
	{ "get_slips_on_slope", "get_slide_on_slope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "get_space_override_mode", "get_gravity_space_override_mode" }, // Area2D
	{ "get_speed", "get_velocity" }, // InputEventMouseMotion
	{ "get_stylebox_types", "get_stylebox_type_list" }, // Theme
	{ "get_surface_material", "get_surface_override_material" }, // MeshInstance3D broke ImporterMesh
	{ "get_surface_material_count", "get_surface_override_material_count" }, // MeshInstance3D
	{ "get_tab_disabled", "is_tab_disabled" }, // Tab
	{ "get_tab_hidden", "is_tab_hidden" }, // Tab
	{ "get_text_align", "get_text_alignment" }, // Button
	{ "get_theme_item_types", "get_theme_item_type_list" }, // Theme
	{ "get_timer_process_mode", "get_timer_process_callback" }, // Timer
	{ "get_translation", "get_position" }, // Node3D broke GLTFNode which is used rarely
	{ "get_use_in_baked_light", "is_baking_navigation" }, // GridMap
	{ "get_used_cells_by_id", "get_used_cells" }, // TileMap
	{ "get_v_scrollbar", "get_v_scroll_bar" }, //ScrollContainer
	{ "get_visible_name", "_get_visible_name" }, // EditorImportPlugin
	{ "get_window_layout", "_get_window_layout" }, // EditorPlugin
	{ "get_word_under_cursor", "get_word_under_caret" }, // TextEdit
	{ "get_world", "get_world_3d" }, // Viewport, Spatial
	{ "get_zfar", "get_far" }, // Camera3D broke GLTFCamera
	{ "get_znear", "get_near" }, // Camera3D broke GLTFCamera
	{ "groove_joint_create", "joint_make_groove" }, // PhysicsServer2D
	{ "handle_menu_selected", "_handle_menu_selected" }, // EditorResourcePicker
	{ "handles_type", "_handles_type" }, // ResourceFormatLoader
	{ "has_color", "has_theme_color" }, //  Control broke Theme
	{ "has_color_override", "has_theme_color_override" }, // Control broke Theme
	{ "has_constant", "has_theme_constant" }, // Control
	{ "has_constant_override", "has_theme_constant_override" }, // Control
	{ "has_filter", "_has_filter" }, // AnimationNode
	{ "has_font", "has_theme_font" }, // Control broke Theme
	{ "has_font_override", "has_theme_font_override" }, // Control
	{ "has_icon", "has_theme_icon" }, // Control broke Theme
	{ "has_icon_override", "has_theme_icon_override" }, // Control
	{ "has_main_screen", "_has_main_screen" }, // EditorPlugin
	{ "has_network_peer", "has_multiplayer_peer" }, // Multiplayer API
	{ "has_stylebox", "has_theme_stylebox" }, // Control broke Theme
	{ "has_stylebox_override", "has_theme_stylebox_override" }, // Control
	{ "http_escape", "uri_encode" }, // String
	{ "http_unescape", "uri_decode" }, // String
	{ "import_animation_from_other_importer", "_import_animation" }, //EditorSceneFormatImporter
	{ "import_scene_from_other_importer", "_import_scene" }, //EditorSceneFormatImporter
	{ "instance_set_surface_material", "instance_set_surface_override_material" }, // RenderingServer
	{ "intersect_polygons_2d", "intersect_polygons" }, // Geometry2D
	{ "intersect_polyline_with_polygon_2d", "intersect_polyline_with_polygon" }, // Geometry2D
	{ "is_a_parent_of", "is_ancestor_of" }, // Node
	{ "is_commiting_action", "is_committing_action" }, // UndoRedo
	{ "is_doubleclick", "is_double_click" }, // InputEventMouseButton
	{ "is_draw_red", "is_draw_warning" }, // EditorProperty
	{ "is_h_drag_enabled", "is_drag_horizontal_enabled" }, // Camera2D
	{ "is_handle_highlighted", "_is_handle_highlighted" }, // EditorNode3DGizmo, EditorNode3DGizmoPlugin
	{ "is_inverting_faces", "get_flip_faces" }, // CSGPrimitive3D
	{ "is_network_master", "is_multiplayer_authority" }, // Node
	{ "is_network_server", "is_server" }, // Multiplayer API
	{ "is_normalmap", "is_normal_map" }, // NoiseTexture
	{ "is_refusing_new_network_connections", "is_refusing_new_connections" }, // Multiplayer API
	{ "is_region", "is_region_enabled" }, // Sprite2D
	{ "is_scancode_unicode", "is_keycode_unicode" }, // OS
	{ "is_selectable_when_hidden", "_is_selectable_when_hidden" }, // EditorNode3DGizmoPlugin
	{ "is_set_as_toplevel", "is_set_as_top_level" }, // CanvasItem
	{ "is_shortcut", "matches_event" }, // Shortcut
	{ "is_size_override_stretch_enabled", "is_size_2d_override_stretch_enabled" }, // SubViewport
	{ "is_sort_enabled", "is_y_sort_enabled" }, // Node2D
	{ "is_static_body", "is_able_to_sleep" }, // PhysicalBone3D - TODO - not sure
	{ "is_v_drag_enabled", "is_drag_vertical_enabled" }, // Camera2D
	{ "joint_create_cone_twist", "joint_make_cone_twist" }, // PhysicsServer3D
	{ "joint_create_generic_6dof", "joint_make_generic_6dof" }, // PhysicsServer3D
	{ "joint_create_hinge", "joint_make_hinge" }, // PhysicsServer3D
	{ "joint_create_pin", "joint_make_pin" }, // PhysicsServer3D
	{ "joint_create_slider", "joint_make_slider" }, // PhysicsServer3D
	{ "line_intersects_line_2d", "line_intersects_line" }, // Geometry2D
	{ "load_from_globals", "load_from_project_settings" }, // InputMap
	{ "load_interactive", "load_threaded_request" }, // ResourceLoader - load_threaded_request is alternative, but is used differently
	{ "make_convex_from_brothers", "make_convex_from_siblings" }, // CollisionShape3D
	{ "make_visible", "_make_visible" }, // EditorPlugin
	{ "merge_polygons_2d", "merge_polygons" }, // Geometry2D
	{ "mesh_surface_get_format", "mesh_surface_get_format_attribute_stride" }, // RenderingServer
	{ "mesh_surface_update_region", "mesh_surface_update_attribute_region" }, // RenderingServer
	{ "move_to_bottom", "move_after" }, // Skeleton3D
	{ "move_to_top", "move_before" }, // Skeleton3D
	{ "multimesh_allocate", "multimesh_allocate_data" }, // RenderingServer
	{ "normalmap_to_xy", "normal_map_to_xy" }, // Image
	{ "offset_polygon_2d", "offset_polygon" }, // Geometry2D
	{ "offset_polyline_2d", "offset_polyline" }, // Geometry2D
	{ "percent_decode", "uri_decode" }, // String
	{ "percent_encode", "uri_encode" }, // String
	{ "pin_joint_create", "joint_make_pin" }, // PhysicsServer2D
	{ "popup_centered_minsize", "popup_centered_clamped" }, // Window
	{ "post_import", "_post_import" }, // EditorScenePostImport
	{ "print_stray_nodes", "print_orphan_nodes" }, // Node
	{ "property_list_changed_notify", "notify_property_list_changed" }, // Object
	{ "recognize", "_recognize" }, // ResourceFormatLoader
	{ "regen_normalmaps", "regen_normal_maps" }, // ArrayMesh
	{ "remove", "remove_at" }, // Array, broke Directory
	{ "remove_animation", "remove_animation_library" }, // AnimationPlayer
	{ "remove_color_override", "remove_theme_color_override" }, // Control
	{ "remove_constant_override", "remove_theme_constant_override" }, // Control
	{ "remove_font_override", "remove_theme_font_override" }, // Control
	{ "remove_icon_override", "remove_theme_icon_override" }, // Control
	{ "remove_scene_import_plugin", "remove_scene_format_importer_plugin" }, //EditorPlugin
	{ "remove_stylebox_override", "remove_theme_stylebox_override" }, // Control
	{ "rename_animation", "rename_animation_library" }, // AnimationPlayer
	{ "rename_dependencies", "_rename_dependencies" }, // ResourceFormatLoader
	{ "save_external_data", "_save_external_data" }, // EditorPlugin
	{ "segment_intersects_segment_2d", "segment_intersects_segment" }, // Geometry2D
	{ "set_adjustment_enable", "set_adjustment_enabled" }, // Environment
	{ "set_alt", "set_alt_pressed" }, // InputEventWithModifiers
	{ "set_anchor_and_margin", "set_anchor_and_offset" }, // Control
	{ "set_anchors_and_margins_preset", "set_anchors_and_offsets_preset" }, // Control
	{ "set_animation_process_mode", "set_process_callback" }, // AnimationPlayer
	{ "set_as_bulk_array", "set_buffer" }, // MultiMesh
	{ "set_as_normalmap", "set_as_normal_map" }, // NoiseTexture
	{ "set_as_toplevel", "set_as_top_level" }, // CanvasItem
	{ "set_audio_bus", "set_audio_bus_name" }, // Area3D
	{ "set_autowrap", "set_autowrap_mode" }, // Label broke AcceptDialog
	{ "set_cast_to", "set_target_position" }, // RayCast2D, RayCast3D
	{ "set_collision_layer_bit", "set_collision_layer_value" }, // CSGShape3D and a lot of others like GridMap
	{ "set_collision_mask_bit", "set_collision_mask_value" }, // CSGShape3D and a lot of others like GridMap
	{ "set_column_min_width", "set_column_custom_minimum_width" }, // Tree
	{ "set_command", "set_command_pressed" }, // InputEventWithModifiers
	{ "set_control", "set_ctrl_pressed" }, // InputEventWithModifiers
	{ "set_create_options", "_set_create_options" }, //  EditorResourcePicker
	{ "set_cull_mask_bit", "set_cull_mask_value" }, // Camera3D
	{ "set_cursor_position", "set_caret_column" }, // LineEdit
	{ "set_d", "set_distance" }, // WorldMarginShape2D
	{ "set_doubleclick", "set_double_click" }, // InputEventMouseButton
	{ "set_draw_red", "set_draw_warning" }, // EditorProperty
	{ "set_enabled_focus_mode", "set_focus_mode" }, // BaseButton
	{ "set_endian_swap", "set_big_endian" }, // File
	{ "set_expand_to_text_length", "set_expand_to_text_length_enabled" }, // LineEdit
	{ "set_filename", "set_scene_file_path" }, // Node, WARNING, this may be used in a lot of other places
	{ "set_focus_neighbour", "set_focus_neighbor" }, // Control
	{ "set_frame_color", "set_color" }, // ColorRect
	{ "set_global_rate_scale", "set_playback_speed_scale" }, // AudioServer
	{ "set_gravity_distance_scale", "set_gravity_point_distance_scale" }, // Area2D
	{ "set_gravity_vector", "set_gravity_direction" }, // Area2D
	{ "set_h_drag_enabled", "set_drag_horizontal_enabled" }, // Camera2D
	{ "set_icon_align", "set_icon_alignment" }, // Button
	{ "set_interior_ambient", "set_ambient_color" }, // ReflectionProbe
	{ "set_interior_ambient_energy", "set_ambient_color_energy" }, // ReflectionProbe
	{ "set_invert_faces", "set_flip_faces" }, // CSGPrimitive3D
	{ "set_is_initialized", "_is_initialized" }, // XRInterface
	{ "set_is_primary", "set_primary" }, // XRInterface
	{ "set_iterations_per_second", "set_physics_ticks_per_second" }, // Engine
	{ "set_layer_mask_bit", "set_layer_mask_value" }, // VisualInstance3D
	{ "set_margins_preset", "set_offsets_preset" }, //  Control
	{ "set_max_atlas_size", "set_max_texture_size" }, // LightmapGI
	{ "set_metakey", "set_meta_pressed" }, // InputEventWithModifiers
	{ "set_mid_height", "set_height" }, // CapsuleMesh
	{ "set_network_master", "set_multiplayer_authority" }, // Node
	{ "set_network_peer", "set_multiplayer_peer" }, // Multiplayer API
	{ "set_pause_mode", "set_process_mode" }, // Node
	{ "set_physical_scancode", "set_physical_keycode" }, // InputEventKey
	{ "set_refuse_new_network_connections", "set_refuse_new_connections" }, // Multiplayer API
	{ "set_region", "set_region_enabled" }, // Sprite2D, Sprite broke AtlasTexture
	{ "set_region_filter_clip", "set_region_filter_clip_enabled" }, // Sprite2D
	{ "set_rotate", "set_rotates" }, // PathFollow2D
	{ "set_scancode", "set_keycode" }, // InputEventKey
	{ "set_shift", "set_shift_pressed" }, // InputEventWithModifiers
	{ "set_size_override", "set_size_2d_override" }, // SubViewport broke ImageTexture
	{ "set_size_override_stretch", "set_size_2d_override_stretch" }, // SubViewport
	{ "set_slips_on_slope", "set_slide_on_slope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "set_sort_enabled", "set_y_sort_enabled" }, // Node2D
	{ "set_space_override_mode", "set_gravity_space_override_mode" }, // Area2D
	{ "set_speed", "set_velocity" }, // InputEventMouseMotion
	{ "set_ssao_edge_sharpness", "set_ssao_sharpness" }, // Environment
	{ "set_surface_material", "set_surface_override_material" }, // MeshInstance3D broke ImporterMesh
	{ "set_tab_align", "set_tab_alignment" }, //TabContainer
	{ "set_tangent", "surface_set_tangent" }, // ImmediateGeometry broke SurfaceTool
	{ "set_text_align", "set_text_alignment" }, // Button
	{ "set_timer_process_mode", "set_timer_process_callback" }, // Timer
	{ "set_tonemap_auto_exposure", "set_tonemap_auto_exposure_enabled" }, // Environment
	{ "set_translation", "set_position" }, // Node3D - this broke GLTFNode which is used rarely
	{ "set_uv2", "surface_set_uv2" }, // ImmediateMesh broke Surffacetool
	{ "set_v_drag_enabled", "set_drag_vertical_enabled" }, // Camera2D
	{ "set_valign", "set_vertical_alignment" }, // Label
	{ "set_window_layout", "_set_window_layout" }, // EditorPlugin
	{ "set_zfar", "set_far" }, // Camera3D broke GLTFCamera
	{ "set_znear", "set_near" }, // Camera3D broke GLTFCamera
	{ "shortcut_match", "is_match" }, // InputEvent
	{ "skeleton_allocate", "skeleton_allocate_data" }, // RenderingServer
	{ "surface_update_region", "surface_update_attribute_region" }, // ArrayMesh
	{ "targeting_method", "tween_method" }, // Tween
	{ "targeting_property", "tween_property" }, // Tween
	{ "track_remove_key_at_position", "track_remove_key_at_time" }, // Animation
	{ "triangulate_delaunay_2d", "triangulate_delaunay" }, // Geometry2D
	{ "unbind_child_node_from_bone", "remove_bone_child" }, // Skeleton3D
	{ "unselect", "deselect" }, // ItemList
	{ "unselect_all", "deselect_all" }, // ItemList
	{ "update_configuration_warning", "update_configuration_warnings" }, // Node
	{ "update_gizmo", "update_gizmos" }, // Node3D
	{ "viewport_set_use_arvr", "viewport_set_use_xr" }, // RenderingServer
	{ "warp_mouse_position", "warp_mouse" }, // Input

	// Builtin types
	//	{ "empty", "is_empty" }, // Array - Used as custom rule  // Be careful, this will be used everywhere
	{ "clamped", "clamp" }, // Vector2  // Be careful, this will be used everywhere
	{ "get_rotation_quat", "get_rotation_quaternion" }, // Basis
	{ "grow_margin", "grow_side" }, // Rect2
	{ "invert", "reverse" }, // Array - TODO check  // Be careful, this will be used everywhere
	{ "is_abs_path", "is_absolute_path" }, // String
	{ "is_valid_integer", "is_valid_int" }, // String
	{ "linear_interpolate", "lerp" }, // Color
	{ "to_ascii", "to_ascii_buffer" }, // String
	{ "to_utf8", "to_utf8_buffer" }, // String
	{ "to_wchar", "to_utf32_buffer" }, // String // TODO - utf32 or utf16?

	// Globals
	{ "rand_range", "randf_range" },
	{ "stepify", "snapped" },

	{ nullptr, nullptr },
};

// gdscript_function_renames clone with CamelCase
static const char *csharp_function_renames[][2] = {
	// { "_SetName", "GetTrackerName"}, // XRPositionalTracker - CameraFeed use this
	// { "_UnhandledInput", "_UnhandledKeyInput"}, // BaseButton, ViewportContainer broke Node, FileDialog,SubViewportContainer
	// { "CreateGizmo", "_CreateGizmo"}, // EditorNode3DGizmoPlugin - may be used
	// { "GetDependencies", "_GetDependencies" }, // ResourceFormatLoader broke ResourceLoader
	// { "GetExtents", "GetSize" }, // BoxShape, RectangleShape broke Decal, VoxelGI, GPUParticlesCollisionBox, GPUParticlesCollisionSDF, GPUParticlesCollisionHeightField, GPUParticlesAttractorBox, GPUParticlesAttractorVectorField, FogVolume
	// { "GetHOffset", "GetDragHorizontalOffset"}, // Camera2D, broke PathFollow, Camera
	// { "GetMode", "GetFileMode"}, // FileDialog broke Panel, Shader, CSGPolygon, Tilemap
	// { "GetMotion", "GetTravel"}, // PhysicsTestMotionResult2D broke ParalaxLayer
	// { "GetName", "GetTrackerName"}, // XRPositionalTracker broke OS, Node
	// { "GetNetworkConnectedPeers", "GetPeers"}, // MultiplayerAPI broke SceneTree
	// { "GetNetworkPeer", "HasMultiplayerPeer"}, // MultiplayerAPI broke SceneTree
	// { "GetNetworkUniqueId", "GetUniqueId"}, // MultiplayerAPI broke SceneTree
	// { "GetOffset", "GetPositionOffset" }, // GraphNode broke Gradient
	// { "GetPeerPort", "GetPeer" }, // ENetMultiplayerPeer broke WebSocketServer
	// { "GetProcessMode", "GetProcessCallback" }, // ClippedCamera3D broke Node, Sky
	// { "GetRenderInfo", "GetRenderingInfo" }, // RenderingServer broke Viewport
	// { "GetType", "GetTrackerType"}, // XRPositionalTracker broke GLTFAccessor, GLTFLight
	// { "GetVOffset", "GetDragVerticalOffset"}, // Camera2D, broke PathFollow, Camera
	// { "HasNetworkPeer", "HasMultiplayerPeer"}, // MultiplayerAPI broke SceneTree
	// { "Instance", "Instantiate" }, // PackedScene, ClassDB - Broke FileSystemDock signal and also tscn files - [instance=ExtResource( 17 )] - this is implemented as custom rule
	// { "IsListening", "IsBound"}, // PacketPeerUDP broke TCPServer, UDPServer
	// { "IsRefusingNewNetworkConnections", "IsRefusingNewConnections"}, // MultiplayerAPI broke SceneTree
	// { "IsValid", "HasValidEvent" }, // Shortcut broke e.g. Callable
	// { "Listen", "Bound"}, // PacketPeerUDP broke TCPServer, UDPServer
	// { "Load", "_Load"}, // ResourceFormatLoader broke ConfigFile, Image, StreamTexture2D
	// { "MakeCurrent", "SetCurrent" }, // Camera2D broke Camera3D, Listener2D
	// { "Process", "_Process" }, // AnimationNode - This word is commonly used
	// { "Save", "_Save"}, // ResourceFormatLoader broke ConfigFile, Image, StreamTexture2D
	// { "SetAutowrap", "SetAutowrapMode" }, // AcceptDialog broke Label - Cyclic Rename
	// { "SetColor", "SurfaceSetColor"}, // ImmediateMesh broke Light2D, Theme, SurfaceTool
	// { "SetEvent", "SetShortcut" }, // BaseButton - Cyclic Rename
	// { "SetExtents", "SetSize"}, // BoxShape, RectangleShape broke ReflectionProbe
	// { "SetFlag", "SetParticleFlag"}, // ParticlesMaterial broke Window, HingeJoint3D
	// { "SetHOffset", "SetDragHorizontalOffset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// { "SetMargin", "SetOffset" }, // Control broke Shape3D, AtlasTexture
	// { "SetMode", "SetModeFileMode" }, // FileDialog broke Panel, Shader, CSGPolygon, Tilemap
	// { "SetNormal", "SurfaceSetNormal"}, // ImmediateGeometry broke SurfaceTool, WorldMarginShape2D
	// { "SetProcessMode", "SetProcessCallback" }, // AnimationTree broke Node, Tween, Sky
	// { "SetRefuseNewNetworkConnections", "SetRefuseNewConnections"}, // MultiplayerAPI broke SceneTree
	// { "SetUv", "SurfaceSetUv" }, // ImmediateMesh broke Polygon2D
	// { "SetVOffset", "SetDragVerticalOffset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// {"GetPoints","GetPointsId"},// Astar, broke Line2D, Convexpolygonshape
	// {"GetVScroll","GetVScrollBar"},//ItemList, broke TextView
	{ "RenderingServer", "GetTabAlignment" }, // Tab
	{ "_AboutToShow", "_AboutToPopup" }, // ColorPickerButton
	{ "_GetConfigurationWarning", "_GetConfigurationWarnings" }, // Node
	{ "_SetCurrent", "SetCurrent" }, // Camera2D
	{ "_SetEditorDescription", "SetEditorDescription" }, // Node
	{ "_ToplevelRaiseSelf", "_TopLevelRaiseSelf" }, // CanvasItem
	{ "_UpdateWrapAt", "_UpdateWrapAtColumn" }, // TextEdit
	{ "AddAnimation", "AddAnimationLibrary" }, // AnimationPlayer
	{ "AddCancel", "AddCancelButton" }, // AcceptDialog
	{ "AddCentralForce", "AddConstantCentralForce" }, //RigidDynamicBody2D
	{ "AddChildBelowNode", "AddSibling" }, // Node
	{ "AddColorOverride", "AddThemeColorOverride" }, // Control
	{ "AddConstantOverride", "AddThemeConstantOverride" }, // Control
	{ "AddFontOverride", "AddThemeFontOverride" }, // Control
	{ "AddForce", "AddConstantForce" }, //RigidDynamicBody2D
	{ "AddIconOverride", "AddThemeIconOverride" }, // Control
	{ "AddSceneImportPlugin", "AddSceneFormatImporterPlugin" }, //EditorPlugin
	{ "AddStyleboxOverride", "AddThemeStyleboxOverride" }, // Control
	{ "AddTorque", "AddConstantTorque" }, //RigidDynamicBody2D
	{ "BindChildNodeToBone", "SetBoneChildren" }, // Skeleton3D
	{ "BumpmapToNormalmap", "BumpMapToNormalMap" }, // Image
	{ "CanBeHidden", "_CanBeHidden" }, // EditorNode3DGizmoPlugin
	{ "CanDropDataFw", "_CanDropDataFw" }, // ScriptEditor
	{ "CanGenerateSmallPreview", "_CanGenerateSmallPreview" }, // EditorResourcePreviewGenerator
	{ "CanInstance", "CanInstantiate" }, // PackedScene, Script
	{ "CanvasLightSetScale", "CanvasLightSetTextureScale" }, // RenderingServer
	{ "CenterViewportToCursor", "CenterViewportToCaret" }, // TextEdit
	{ "ClipPolygons2d", "ClipPolygons" }, // Geometry2D
	{ "ClipPolylineWithPolygon2d", "ClipPolylineWithPolygon" }, //Geometry2D
	{ "CommitHandle", "_CommitHandle" }, // EditorNode3DGizmo
	{ "ConvexHull2d", "ConvexHull" }, // Geometry2D
	{ "CursorGetBlinkSpeed", "GetCaretBlinkSpeed" }, // TextEdit
	{ "CursorGetColumn", "GetCaretColumn" }, // TextEdit
	{ "CursorGetLine", "GetCaretLine" }, // TextEdit
	{ "CursorSetBlinkEnabled", "SetCaretBlinkEnabled" }, // TextEdit
	{ "CursorSetBlinkSpeed", "SetCaretBlinkSpeed" }, // TextEdit
	{ "CursorSetColumn", "SetCaretColumn" }, // TextEdit
	{ "CursorSetLine", "SetCaretLine" }, // TextEdit
	{ "DampedSpringJointCreate", "JointMakeDampedSpring" }, // PhysicsServer2D
	{ "DampedStringJointGetParam", "DampedSpringJointGetParam" }, // PhysicsServer2D
	{ "DampedStringJointSetParam", "DampedSpringJointSetParam" }, // PhysicsServer2D
	{ "DeleteCharAtCursor", "DeleteCharAtCaret" }, // LineEdit
	{ "DeselectItems", "DeselectAll" }, // FileDialog
	{ "DropDataFw", "_DropDataFw" }, // ScriptEditor
	{ "ExcludePolygons2d", "ExcludePolygons" }, // Geometry2D
	{ "FindScancodeFromString", "FindKeycodeFromString" }, // OS
	{ "ForwardCanvasDrawOverViewport", "_ForwardCanvasDrawOverViewport" }, // EditorPlugin
	{ "ForwardCanvasForceDrawOverViewport", "_ForwardCanvasForceDrawOverViewport" }, // EditorPlugin
	{ "ForwardCanvasGuiInput", "_ForwardCanvasGuiInput" }, // EditorPlugin
	{ "ForwardSpatialDrawOverViewport", "_Forward3dDrawOverViewport" }, // EditorPlugin
	{ "ForwardSpatialForceDrawOverViewport", "_Forward3dForceDrawOverViewport" }, // EditorPlugin
	{ "ForwardSpatialGuiInput", "_Forward3dGuiInput" }, // EditorPlugin
	{ "GenerateFromPath", "_GenerateFromPath" }, // EditorResourcePreviewGenerator
	{ "GenerateSmallPreviewAutomatically", "_GenerateSmallPreviewAutomatically" }, // EditorResourcePreviewGenerator
	{ "GetActionList", "ActionGetEvents" }, // InputMap
	{ "GetAlt", "IsAltPressed" }, // InputEventWithModifiers
	{ "GetAnimationProcessMode", "GetProcessCallback" }, // AnimationPlayer
	{ "GetAppliedForce", "GetConstantForce" }, //RigidDynamicBody2D
	{ "GetAppliedTorque", "GetConstantTorque" }, //RigidDynamicBody2D
	{ "GetAudioBus", "GetAudioBusName" }, // Area3D
	{ "GetBoundChildNodesToBone", "GetBoneChildren" }, // Skeleton3D
	{ "GetCamera", "GetCamera3d" }, // Viewport -> this is also convertable to getCamera2d, broke GLTFNode
	{ "GetCancel", "GetCancelButton" }, // ConfirmationDialog
	{ "GetCaption", "_GetCaption" }, // AnimationNode
	{ "GetCastTo", "GetTargetPosition" }, // RayCast2D, RayCast3D
	{ "GetChildByName", "_GetChildByName" }, // AnimationNode
	{ "GetChildNodes", "_GetChildNodes" }, // AnimationNode
	{ "GetClosestPointToSegment2d", "GetClosestPointToSegment" }, // Geometry2D
	{ "GetClosestPointToSegmentUncapped2d", "GetClosestPointToSegmentUncapped" }, // Geometry2D
	{ "GetClosestPointsBetweenSegments2d", "GetClosestPointToSegment" }, // Geometry2D
	{ "GetCollisionLayerBit", "GetCollisionLayerValue" }, // CSGShape3D and a lot of others like GridMap
	{ "GetCollisionMaskBit", "GetCollisionMaskValue" }, // CSGShape3D and a lot of others like GridMap
	{ "GetColorTypes", "GetColorTypeList" }, // Theme
	{ "GetCommand", "IsCommandPressed" }, // InputEventWithModifiers
	{ "GetConstantTypes", "GetConstantTypeList" }, // Theme
	{ "GetControl", "IsCtrlPressed" }, // InputEventWithModifiers
	{ "GetCullMaskBit", "GetCullMaskValue" }, // Camera3D
	{ "GetCursorPosition", "GetCaretColumn" }, // LineEdit
	{ "GetD", "GetDistance" }, // LineShape2D
	{ "GetDragDataFw", "_GetDragDataFw" }, // ScriptEditor
	{ "GetEditorViewport", "GetViewport" }, // EditorPlugin
	{ "GetEnabledFocusMode", "GetFocusMode" }, // BaseButton
	{ "GetEndianSwap", "IsBigEndian" }, // File
	{ "GetErrorString", "GetErrorMessage" }, // JSON
	{ "GetFocusNeighbour", "GetFocusNeighbor" }, // Control
	{ "GetFontTypes", "GetFontTypeList" }, // Theme
	{ "GetFrameColor", "GetColor" }, // ColorRect
	{ "GetGlobalRateScale", "GetPlaybackSpeedScale" }, // AudioServer
	{ "GetGravityDistanceScale", "GetGravityPointDistanceScale" }, //Area2D
	{ "GetGravityVector", "GetGravityDirection" }, //Area2D
	{ "GetHScrollbar", "GetHScrollBar" }, //ScrollContainer
	{ "GetHand", "GetTrackerHand" }, // XRPositionalTracker
	{ "GetHandleName", "_GetHandleName" }, // EditorNode3DGizmo
	{ "GetHandleValue", "_GetHandleValue" }, // EditorNode3DGizmo
	{ "GetIconAlign", "GetIconAlignment" }, // Button
	{ "GetIconTypes", "GetIconTypeList" }, // Theme
	{ "GetIdleFrames", "GetProcessFrames" }, // Engine
	{ "GetImportOptions", "_GetImportOptions" }, // EditorImportPlugin
	{ "GetImportOrder", "_GetImportOrder" }, // EditorImportPlugin
	{ "GetImporterName", "_GetImporterName" }, // EditorImportPlugin
	{ "GetInteriorAmbient", "GetAmbientColor" }, // ReflectionProbe
	{ "GetInteriorAmbientEnergy", "GetAmbientColorEnergy" }, // ReflectionProbe
	{ "GetIterationsPerSecond", "GetPhysicsTicksPerSecond" }, // Engine
	{ "GetLastMouseSpeed", "GetLastMouseVelocity" }, // Input
	{ "GetLayerMaskBit", "GetLayerMaskValue" }, // VisualInstance3D
	{ "GetLen", "GetLength" }, // File
	{ "GetMaxAtlasSize", "GetMaxTextureSize" }, // LightmapGI
	{ "GetMetakey", "IsMetaPressed" }, // InputEventWithModifiers
	{ "GetMidHeight", "GetHeight" }, // CapsuleMesh
	{ "GetMotionRemainder", "GetRemainder" }, // PhysicsTestMotionResult2D
	{ "GetNetworkConnectedPeers", "GetPeers" }, // Multiplayer API
	{ "GetNetworkMaster", "GetMultiplayerAuthority" }, // Node
	{ "GetNetworkPeer", "GetMultiplayerPeer" }, // Multiplayer API
	{ "GetNetworkUniqueId", "GetUniqueId" }, // Multiplayer API
	{ "GetOk", "GetOkButton" }, // AcceptDialog
	{ "GetOptionVisibility", "_GetOptionVisibility" }, // EditorImportPlugin
	{ "GetParameterDefaultValue", "_GetParameterDefaultValue" }, // AnimationNode
	{ "GetParameterList", "_GetParameterList" }, // AnimationNode
	{ "GetParentSpatial", "GetParentNode3d" }, // Node3D
	{ "GetPhysicalScancode", "GetPhysicalKeycode" }, // InputEventKey
	{ "GetPhysicalScancodeWithModifiers", "GetPhysicalKeycodeWithModifiers" }, // InputEventKey
	{ "GetPluginIcon", "_GetPluginIcon" }, // EditorPlugin
	{ "GetPluginName", "_GetPluginName" }, // EditorPlugin
	{ "GetPresetCount", "_GetPresetCount" }, // EditorImportPlugin
	{ "GetPresetName", "_GetPresetName" }, // EditorImportPlugin
	{ "GetRecognizedExtensions", "_GetRecognizedExtensions" }, // ResourceFormatLoader, EditorImportPlugin broke ResourceSaver
	{ "GetRenderInfo", "GetRenderingInfo" }, // RenderingServer
	{ "GetRenderTargetsize", "GetRenderTargetSize" }, // XRInterface
	{ "GetResourceType", "_GetResourceType" }, // ResourceFormatLoader
	{ "GetResult", "GetData" }, //JSON
	{ "GetRpcSenderId", "GetRemoteSenderId" }, // Multiplayer API
	{ "GetSaveExtension", "_GetSaveExtension" }, // EditorImportPlugin
	{ "GetScancode", "GetKeycode" }, // InputEventKey
	{ "GetScancodeString", "GetKeycodeString" }, // OS
	{ "GetScancodeWithModifiers", "GetKeycodeWithModifiers" }, // InputEventKey
	{ "GetShift", "IsShiftPressed" }, // InputEventWithModifiers
	{ "GetSizeOverride", "GetSize2dOverride" }, // SubViewport
	{ "GetSlipsOnSlope", "GetSlideOnSlope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "GetSpaceOverrideMode", "GetGravitySpaceOverrideMode" }, // Area2D
	{ "GetSpeed", "GetVelocity" }, // InputEventMouseMotion
	{ "GetStyleboxTypes", "GetStyleboxTypeList" }, // Theme
	{ "GetSurfaceMaterial", "GetSurfaceOverrideMaterial" }, // MeshInstance3D broke ImporterMesh
	{ "GetSurfaceMaterialCount", "GetSurfaceOverrideMaterialCount" }, // MeshInstance3D
	{ "GetTabDisabled", "IsTabDisabled" }, // Tab
	{ "GetTabHidden", "IsTabHidden" }, // Tab
	{ "GetTextAlign", "GetTextAlignment" }, // Button
	{ "GetThemeItemTypes", "GetThemeItemTypeList" }, // Theme
	{ "GetTimerProcessMode", "GetTimerProcessCallback" }, // Timer
	{ "GetTranslation", "GetPosition" }, // Node3D broke GLTFNode which is used rarely
	{ "GetUseInBakedLight", "IsBakingNavigation" }, // GridMap
	{ "GetUsedCellsById", "GetUsedCells" }, // TileMap
	{ "GetVScrollbar", "GetVScrollBar" }, //ScrollContainer
	{ "GetVisibleName", "_GetVisibleName" }, // EditorImportPlugin
	{ "GetWindowLayout", "_GetWindowLayout" }, // EditorPlugin
	{ "GetWordUnderCursor", "GetWordUnderCaret" }, // TextEdit
	{ "GetWorld", "GetWorld3d" }, // Viewport, Spatial
	{ "GetZfar", "GetFar" }, // Camera3D broke GLTFCamera
	{ "GetZnear", "GetNear" }, // Camera3D broke GLTFCamera
	{ "GrooveJointCreate", "JointMakeGroove" }, // PhysicsServer2D
	{ "HandleMenuSelected", "_HandleMenuSelected" }, // EditorResourcePicker
	{ "HandlesType", "_HandlesType" }, // ResourceFormatLoader
	{ "HasColor", "HasThemeColor" }, //  Control broke Theme
	{ "HasColorOverride", "HasThemeColorOverride" }, // Control broke Theme
	{ "HasConstant", "HasThemeConstant" }, // Control
	{ "HasConstantOverride", "HasThemeConstantOverride" }, // Control
	{ "HasFilter", "_HasFilter" }, // AnimationNode
	{ "HasFont", "HasThemeFont" }, // Control broke Theme
	{ "HasFontOverride", "HasThemeFontOverride" }, // Control
	{ "HasIcon", "HasThemeIcon" }, // Control broke Theme
	{ "HasIconOverride", "HasThemeIconOverride" }, // Control
	{ "HasMainScreen", "_HasMainScreen" }, // EditorPlugin
	{ "HasNetworkPeer", "HasMultiplayerPeer" }, // Multiplayer API
	{ "HasStylebox", "HasThemeStylebox" }, // Control broke Theme
	{ "HasStyleboxOverride", "HasThemeStyleboxOverride" }, // Control
	{ "HttpEscape", "UriEncode" }, // String
	{ "HttpUnescape", "UriDecode" }, // String
	{ "ImportAnimationFromOtherImporter", "_ImportAnimation" }, //EditorSceneFormatImporter
	{ "ImportSceneFromOtherImporter", "_ImportScene" }, //EditorSceneFormatImporter
	{ "InstanceSetSurfaceMaterial", "InstanceSetSurfaceOverrideMaterial" }, // RenderingServer
	{ "IntersectPolygons2d", "IntersectPolygons" }, // Geometry2D
	{ "IntersectPolylineWithPolygon2d", "IntersectPolylineWithPolygon" }, // Geometry2D
	{ "IsAParentOf", "IsAncestorOf" }, // Node
	{ "IsCommitingAction", "IsCommittingAction" }, // UndoRedo
	{ "IsDoubleclick", "IsDoubleClick" }, // InputEventMouseButton
	{ "IsHDragEnabled", "IsDragHorizontalEnabled" }, // Camera2D
	{ "IsHandleHighlighted", "_IsHandleHighlighted" }, // EditorNode3DGizmo, EditorNode3DGizmoPlugin
	{ "IsNetworkMaster", "IsMultiplayerAuthority" }, // Node
	{ "IsNetworkServer", "IsServer" }, // Multiplayer API
	{ "IsNormalmap", "IsNormalMap" }, // NoiseTexture
	{ "IsRefusingNewNetworkConnections", "IsRefusingNewConnections" }, // Multiplayer API
	{ "IsRegion", "IsRegionEnabled" }, // Sprite2D
	{ "IsScancodeUnicode", "IsKeycodeUnicode" }, // OS
	{ "IsSelectableWhenHidden", "_IsSelectableWhenHidden" }, // EditorNode3DGizmoPlugin
	{ "IsSetAsToplevel", "IsSetAsTopLevel" }, // CanvasItem
	{ "IsShortcut", "MatchesEvent" }, // Shortcut
	{ "IsSizeOverrideStretchEnabled", "IsSize2dOverrideStretchEnabled" }, // SubViewport
	{ "IsSortEnabled", "IsYSortEnabled" }, // Node2D
	{ "IsStaticBody", "IsAbleToSleep" }, // PhysicalBone3D - TODO - not sure
	{ "IsVDragEnabled", "IsDragVerticalEnabled" }, // Camera2D
	{ "JointCreateConeTwist", "JointMakeConeTwist" }, // PhysicsServer3D
	{ "JointCreateGeneric6dof", "JointMakeGeneric6dof" }, // PhysicsServer3D
	{ "JointCreateHinge", "JointMakeHinge" }, // PhysicsServer3D
	{ "JointCreatePin", "JointMakePin" }, // PhysicsServer3D
	{ "JointCreateSlider", "JointMakeSlider" }, // PhysicsServer3D
	{ "LineIntersectsLine2d", "LineIntersectsLine" }, // Geometry2D
	{ "LoadFromGlobals", "LoadFromProjectSettings" }, // InputMap
	{ "MakeConvexFromBrothers", "MakeConvexFromSiblings" }, // CollisionShape3D
	{ "MergePolygons2d", "MergePolygons" }, // Geometry2D
	{ "MeshSurfaceGetFormat", "MeshSurfaceGetFormatAttributeStride" }, // RenderingServer
	{ "MeshSurfaceUpdateRegion", "MeshSurfaceUpdateAttributeRegion" }, // RenderingServer
	{ "MoveToBottom", "MoveAfter" }, // Skeleton3D
	{ "MoveToTop", "MoveBefore" }, // Skeleton3D
	{ "MultimeshAllocate", "MultimeshAllocateData" }, // RenderingServer
	{ "NormalmapToXy", "NormalMapToXy" }, // Image
	{ "OffsetPolygon2d", "OffsetPolygon" }, // Geometry2D
	{ "OffsetPolyline2d", "OffsetPolyline" }, // Geometry2D
	{ "PercentDecode", "UriDecode" }, // String
	{ "PercentEncode", "UriEncode" }, // String
	{ "PinJointCreate", "JointMakePin" }, // PhysicsServer2D
	{ "PopupCenteredMinsize", "PopupCenteredClamped" }, // Window
	{ "PostImport", "_PostImport" }, // EditorScenePostImport
	{ "PrintStrayNodes", "PrintOrphanNodes" }, // Node
	{ "PropertyListChangedNotify", "NotifyPropertyListChanged" }, // Object
	{ "Recognize", "_Recognize" }, // ResourceFormatLoader
	{ "RegenNormalmaps", "RegenNormalMaps" }, // ArrayMesh
	{ "Remove", "RemoveAt" }, // Array, broke Directory
	{ "RemoveAnimation", "RemoveAnimationLibrary" }, // AnimationPlayer
	{ "RemoveColorOverride", "RemoveThemeColorOverride" }, // Control
	{ "RemoveConstantOverride", "RemoveThemeConstantOverride" }, // Control
	{ "RemoveFontOverride", "RemoveThemeFontOverride" }, // Control
	{ "RemoveSceneImportPlugin", "RemoveSceneFormatImporterPlugin" }, //EditorPlugin
	{ "RemoveStyleboxOverride", "RemoveThemeStyleboxOverride" }, // Control
	{ "RenameAnimation", "RenameAnimationLibrary" }, // AnimationPlayer
	{ "RenameDependencies", "_RenameDependencies" }, // ResourceFormatLoader
	{ "SaveExternalData", "_SaveExternalData" }, // EditorPlugin
	{ "SegmentIntersectsSegment2d", "SegmentIntersectsSegment" }, // Geometry2D
	{ "SetAdjustmentEnable", "SetAdjustmentEnabled" }, // Environment
	{ "SetAlt", "SetAltPressed" }, // InputEventWithModifiers
	{ "SetAnchorAndMargin", "SetAnchorAndOffset" }, // Control
	{ "SetAnchorsAndMarginsPreset", "SetAnchorsAndOffsetsPreset" }, // Control
	{ "SetAnimationProcessMode", "SetProcessCallback" }, // AnimationPlayer
	{ "SetAsBulkArray", "SetBuffer" }, // MultiMesh
	{ "SetAsNormalmap", "SetAsNormalMap" }, // NoiseTexture
	{ "SetAsToplevel", "SetAsTopLevel" }, // CanvasItem
	{ "SetAudioBus", "SetAudioBusName" }, // Area3D
	{ "SetAutowrap", "SetAutowrapMode" }, // Label broke AcceptDialog
	{ "SetCastTo", "SetTargetPosition" }, // RayCast2D, RayCast3D
	{ "SetCollisionLayerBit", "SetCollisionLayerValue" }, // CSGShape3D and a lot of others like GridMap
	{ "SetCollisionMaskBit", "SetCollisionMaskValue" }, // CSGShape3D and a lot of others like GridMap
	{ "SetColumnMinWidth", "SetColumnCustomMinimumWidth" }, // Tree
	{ "SetCommand", "SetCommandPressed" }, // InputEventWithModifiers
	{ "SetControl", "SetCtrlPressed" }, // InputEventWithModifiers
	{ "SetCreateOptions", "_SetCreateOptions" }, //  EditorResourcePicker
	{ "SetCullMaskBit", "SetCullMaskValue" }, // Camera3D
	{ "SetCursorPosition", "SetCaretColumn" }, // LineEdit
	{ "SetD", "SetDistance" }, // WorldMarginShape2D
	{ "SetDoubleclick", "SetDoubleClick" }, // InputEventMouseButton
	{ "SetEnabledFocusMode", "SetFocusMode" }, // BaseButton
	{ "SetEndianSwap", "SetBigEndian" }, // File
	{ "SetExpandToTextLength", "SetExpandToTextLengthEnabled" }, // LineEdit
	{ "SetFocusNeighbour", "SetFocusNeighbor" }, // Control
	{ "SetFrameColor", "SetColor" }, // ColorRect
	{ "SetGlobalRateScale", "SetPlaybackSpeedScale" }, // AudioServer
	{ "SetGravityDistanceScale", "SetGravityPointDistanceScale" }, // Area2D
	{ "SetGravityVector", "SetGravityDirection" }, // Area2D
	{ "SetHDragEnabled", "SetDragHorizontalEnabled" }, // Camera2D
	{ "SetIconAlign", "SetIconAlignment" }, // Button
	{ "SetInteriorAmbient", "SetAmbientColor" }, // ReflectionProbe
	{ "SetInteriorAmbientEnergy", "SetAmbientColorEnergy" }, // ReflectionProbe
	{ "SetIsInitialized", "_IsInitialized" }, // XRInterface
	{ "SetIsPrimary", "SetPrimary" }, // XRInterface
	{ "SetIterationsPerSecond", "SetPhysicsTicksPerSecond" }, // Engine
	{ "SetLayerMaskBit", "SetLayerMaskValue" }, // VisualInstance3D
	{ "SetMarginsPreset", "SetOffsetsPreset" }, //  Control
	{ "SetMaxAtlasSize", "SetMaxTextureSize" }, // LightmapGI
	{ "SetMetakey", "SetMetaPressed" }, // InputEventWithModifiers
	{ "SetMidHeight", "SetHeight" }, // CapsuleMesh
	{ "SetNetworkMaster", "SetMultiplayerAuthority" }, // Node
	{ "SetNetworkPeer", "SetMultiplayerPeer" }, // Multiplayer API
	{ "SetPhysicalScancode", "SetPhysicalKeycode" }, // InputEventKey
	{ "SetRefuseNewNetworkConnections", "SetRefuseNewConnections" }, // Multiplayer API
	{ "SetRegion", "SetRegionEnabled" }, // Sprite2D, Sprite broke AtlasTexture
	{ "SetRegionFilterClip", "SetRegionFilterClipEnabled" }, // Sprite2D
	{ "SetRotate", "SetRotates" }, // PathFollow2D
	{ "SetScancode", "SetKeycode" }, // InputEventKey
	{ "SetShift", "SetShiftPressed" }, // InputEventWithModifiers
	{ "SetSizeOverride", "SetSize2dOverride" }, // SubViewport broke ImageTexture
	{ "SetSizeOverrideStretch", "SetSize2dOverrideStretch" }, // SubViewport
	{ "SetSlipsOnSlope", "SetSlideOnSlope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "SetSortEnabled", "SetYSortEnabled" }, // Node2D
	{ "SetSpaceOverrideMode", "SetGravitySpaceOverrideMode" }, // Area2D
	{ "SetSpeed", "SetVelocity" }, // InputEventMouseMotion
	{ "SetSsaoEdgeSharpness", "SetSsaoSharpness" }, // Environment
	{ "SetSurfaceMaterial", "SetSurfaceOverrideMaterial" }, // MeshInstance3D broke ImporterMesh
	{ "SetTabAlign", "SetTabAlignment" }, //TabContainer
	{ "SetTangent", "SurfaceSetTangent" }, // ImmediateGeometry broke SurfaceTool
	{ "SetTextAlign", "SetTextAlignment" }, // Button
	{ "SetTimerProcessMode", "SetTimerProcessCallback" }, // Timer
	{ "SetTonemapAutoExposure", "SetTonemapAutoExposureEnabled" }, // Environment
	{ "SetTranslation", "SetPosition" }, // Node3D - this broke GLTFNode which is used rarely
	{ "SetUv2", "SurfaceSetUv2" }, // ImmediateMesh broke Surffacetool
	{ "SetVDragEnabled", "SetDragVerticalEnabled" }, // Camera2D
	{ "SetValign", "SetVerticalAlignment" }, // Label
	{ "SetWindowLayout", "_SetWindowLayout" }, // EditorPlugin
	{ "SetZfar", "SetFar" }, // Camera3D broke GLTFCamera
	{ "SetZnear", "SetNear" }, // Camera3D broke GLTFCamera
	{ "ShortcutMatch", "IsMatch" }, // InputEvent
	{ "SkeletonAllocate", "SkeletonAllocateData" }, // RenderingServer
	{ "SurfaceUpdateRegion", "SurfaceUpdateAttributeRegion" }, // ArrayMesh
	{ "TargetingMethod", "TweenMethod" }, // Tween
	{ "TargetingProperty", "TweenProperty" }, // Tween
	{ "TrackRemoveKeyAtPosition", "TrackRemoveKeyAtTime" }, // Animation
	{ "TriangulateDelaunay2d", "TriangulateDelaunay" }, // Geometry2D
	{ "UnbindChildNodeFromBone", "RemoveBoneChild" }, // Skeleton3D
	{ "Unselect", "Deselect" }, // ItemList
	{ "UnselectAll", "DeselectAll" }, // ItemList
	{ "UpdateConfigurationWarning", "UpdateConfigurationWarnings" }, // Node
	{ "UpdateGizmo", "UpdateGizmos" }, // Node3D
	{ "ViewportSetUseArvr", "ViewportSetUseXr" }, // RenderingServer
	{ "WarpMousePosition", "WarpMouse" }, // Input

	// Builtin types
	//	{ "Empty", "IsEmpty" }, // Array - Used as custom rule  // Be careful, this will be used everywhere
	{ "Clamped", "Clamp" }, // Vector2  // Be careful, this will be used everywhere
	{ "GetRotationQuat", "GetRotationQuaternion" }, // Basis
	{ "GrowMargin", "GrowSide" }, // Rect2
	{ "Invert", "Reverse" }, // Array - TODO check  // Be careful, this will be used everywhere
	{ "IsAbsPath", "IsAbsolutePath" }, // String
	{ "IsValidInteger", "IsValidInt" }, // String
	{ "LinearInterpolate", "Lerp" }, // Color
	{ "ToAscii", "ToAsciiBuffer" }, // String
	{ "ToUtf8", "ToUtf8Buffer" }, // String
	{ "ToWchar", "ToUtf32Buffer" }, // String // TODO - utf32 or utf16?

	// Globals
	{ "RandRange", "RandfRange" },
	{ "Stepify", "Snapped" },

	{ nullptr, nullptr },
};

// Some needs to be disabled, because users can use this names as variables
static const char *gdscript_properties_renames[][2] = {
	//	// { "d", "distance" }, //WorldMarginShape2D - TODO, looks that polish letters ą ę are treaten as space, not as letter, so `będą` are renamed to `będistanceą`
	//	// {"alt","alt_pressed"}, // This may broke a lot of comments and user variables
	//	// {"command","command_pressed"},// This may broke a lot of comments and user variables
	//	// {"control","ctrl_pressed"},// This may broke a lot of comments and user variables
	//	// {"extends","size"}, // BoxShape3D, LightmapGI broke ReflectionProbe
	//	// {"meta","meta_pressed"},// This may broke a lot of comments and user variables
	//	// {"pause_mode","process_mode"}, // Node - Cyclic rename, look for others
	//	// {"rotate","rotates"}, // PathFollow2D - probably function exists with same name
	//	// {"shift","shift_pressed"},// This may broke a lot of comments and user variables
	//	{ "autowrap", "autowrap_mode" }, // Label
	//	{ "cast_to", "target_position" }, // RayCast2D, RayCast3D
	//	{ "doubleclick", "double_click" }, // InputEventMouseButton
	//	{ "group", "button_group" }, // BaseButton
	//	{ "process_mode", "process_callback" }, // AnimationTree, Camera2D
	//	{ "scancode", "keycode" }, // InputEventKey
	//	{ "toplevel", "top_level" }, // Node
	//	{ "window_title", "title" }, // Window
	//	{ "wrap_enabled", "wrap_mode" }, // TextEdit
	//	{ "zfar", "far" }, // Camera3D
	//	{ "znear", "near" }, // Camera3D
	//	{ "filename", "scene_file_path" }, // Node
	{ "as_normalmap", "as_normal_map" }, // NoiseTexture
	{ "bbcode_text", "text" }, // RichTextLabel
	{ "caret_moving_by_right_click", "caret_move_on_right_click" }, // TextEdit
	{ "caret_position", "caret_column" }, // LineEdit
	{ "check_vadjust", "check_v_adjust" }, // Theme
	{ "close_h_ofs", "close_h_offset" }, // Theme
	{ "close_v_ofs", "close_v_offset" }, // Theme
	{ "commentfocus", "comment_focus" }, // Theme
	{ "drag_margin_bottom", "drag_bottom_margin" }, // Camera2D
	{ "drag_margin_h_enabled", "drag_horizontal_enabled" }, // Camera2D
	{ "drag_margin_left", "drag_left_margin" }, // Camera2D
	{ "drag_margin_right", "drag_right_margin" }, // Camera2D
	{ "drag_margin_top", "drag_top_margin" }, // Camera2D
	{ "drag_margin_v_enabled", "drag_vertical_enabled" }, // Camera2D
	{ "enabled_focus_mode", "focus_mode" }, // BaseButton - Removed
	{ "extra_spacing_bottom", "spacing_bottom" }, // Font
	{ "extra_spacing_top", "spacing_top" }, // Font
	{ "focus_neighbour_bottom", "focus_neighbor_bottom" }, // Control
	{ "focus_neighbour_left", "focus_neighbor_left" }, // Control
	{ "focus_neighbour_right", "focus_neighbor_right" }, // Control
	{ "focus_neighbour_top", "focus_neighbor_top" }, // Control
	{ "global_rate_scale", "playback_speed_scale" }, // AudioServer
	{ "gravity_distance_scale", "gravity_point_distance_scale" }, // Area2D
	{ "gravity_vec", "gravity_direction" }, // Area2D
	{ "hseparation", "h_separation" }, // Theme
	{ "iterations_per_second", "physics_ticks_per_second" }, // Engine
	{ "margin_bottom", "offset_bottom" }, // Control broke NinePatchRect, StyleBox
	{ "margin_left", "offset_left" }, // Control broke NinePatchRect, StyleBox
	{ "margin_right", "offset_right" }, // Control broke NinePatchRect, StyleBox
	{ "margin_top", "offset_top" }, // Control broke NinePatchRect, StyleBox
	{ "mid_height", "height" }, // CapsuleMesh
	{ "offset_h", "drag_horizontal_offset" }, // Camera2D
	{ "offset_v", "drag_vertical_offset" }, // Camera2D
	{ "ofs", "offset" }, // Theme
	{ "out_of_range_mode", "max_polyphony" }, // AudioStreamPlayer3D
	{ "pause_mode", "process_mode" }, // Node
	{ "physical_scancode", "physical_keycode" }, // InputEventKey
	{ "popup_exclusive", "exclusive" }, // Window
	{ "refuse_new_network_connections", "refuse_new_connections" }, // MultiplayerAPI
	{ "region_filter_clip", "region_filter_clip_enabled" }, // Sprite2D
	{ "selectedframe", "selected_frame" }, // Theme
	{ "size_override_stretch", "size_2d_override_stretch" }, // SubViewport
	{ "slips_on_slope", "slide_on_slope" }, // SeparationRayShape2D
	{ "ss_reflections_depth_tolerance", "ssr_depth_tolerance" }, // Environment
	{ "ss_reflections_enabled", "ssr_enabled" }, // Environment
	{ "ss_reflections_fade_in", "ssr_fade_in" }, // Environment
	{ "ss_reflections_fade_out", "ssr_fade_out" }, // Environment
	{ "ss_reflections_max_steps", "ssr_max_steps" }, // Environment
	{ "state_machine_selectedframe", "state_machine_selected_frame" }, // Theme
	{ "syntax_highlighting", "syntax_highlighter" }, // TextEdit
	{ "tab_align", "tab_alignment" }, // TabContainer
	{ "table_hseparation", "table_h_separation" }, // Theme
	{ "table_vseparation", "table_v_separation" }, // Theme
	{ "translation", "position" }, // Node3D - broke GLTFNode
	{ "vseparation", "v_separation" }, // Theme

	{ nullptr, nullptr },
};

// Some needs to be disabled, because users can use this names as variables
static const char *csharp_properties_renames[][2] = {
	//	// { "D", "Distance" }, //WorldMarginShape2D - TODO, looks that polish letters ą ę are treaten as space, not as letter, so `będą` are renamed to `będistanceą`
	//	// {"Alt","AltPressed"}, // This may broke a lot of comments and user variables
	//	// {"Command","CommandPressed"},// This may broke a lot of comments and user variables
	//	// {"Control","CtrlPressed"},// This may broke a lot of comments and user variables
	//	// {"Extends","Size"}, // BoxShape3D, LightmapGI broke ReflectionProbe
	//	// {"Meta","MetaPressed"},// This may broke a lot of comments and user variables
	//	// {"PauseMode","ProcessMode"}, // Node - Cyclic rename, look for others
	//	// {"Rotate","Rotates"}, // PathFollow2D - probably function exists with same name
	//	// {"Shift","ShiftPressed"},// This may broke a lot of comments and user variables
	//	{ "Autowrap", "AutowrapMode" }, // Label
	//	{ "CastTo", "TargetPosition" }, // RayCast2D, RayCast3D
	//	{ "Doubleclick", "DoubleClick" }, // InputEventMouseButton
	//	{ "Group", "ButtonGroup" }, // BaseButton
	//	{ "ProcessMode", "ProcessCallback" }, // AnimationTree, Camera2D
	//	{ "Scancode", "Keycode" }, // InputEventKey
	//	{ "Toplevel", "TopLevel" }, // Node
	//	{ "WindowTitle", "Title" }, // Window
	//	{ "WrapEnabled", "WrapMode" }, // TextEdit
	//	{ "Zfar", "Far" }, // Camera3D
	//	{ "Znear", "Near" }, // Camera3D
	{ "AsNormalmap", "AsNormalMap" }, // NoiseTexture
	{ "BbcodeText", "Text" }, // RichTextLabel
	{ "CaretMovingByRightClick", "CaretMoveOnRightClick" }, // TextEdit
	{ "CaretPosition", "CaretColumn" }, // LineEdit
	{ "CheckVadjust", "CheckVAdjust" }, // Theme
	{ "CloseHOfs", "CloseHOffset" }, // Theme
	{ "CloseVOfs", "CloseVOffset" }, // Theme
	{ "Commentfocus", "CommentFocus" }, // Theme
	{ "DragMarginBottom", "DragBottomMargin" }, // Camera2D
	{ "DragMarginHEnabled", "DragHorizontalEnabled" }, // Camera2D
	{ "DragMarginLeft", "DragLeftMargin" }, // Camera2D
	{ "DragMarginRight", "DragRightMargin" }, // Camera2D
	{ "DragMarginTop", "DragTopMargin" }, // Camera2D
	{ "DragMarginVEnabled", "DragVerticalEnabled" }, // Camera2D
	{ "EnabledFocusMode", "FocusMode" }, // BaseButton - Removed
	{ "ExtraSpacingBottom", "SpacingBottom" }, // Font
	{ "ExtraSpacingTop", "SpacingTop" }, // Font
	{ "FocusNeighbourBottom", "FocusNeighborBottom" }, // Control
	{ "FocusNeighbourLeft", "FocusNeighborLeft" }, // Control
	{ "FocusNeighbourRight", "FocusNeighborRight" }, // Control
	{ "FocusNeighbourTop", "FocusNeighborTop" }, // Control
	{ "GlobalRateScale", "PlaybackSpeedScale" }, // AudioServer
	{ "GravityDistanceScale", "GravityPointDistanceScale" }, // Area2D
	{ "GravityVec", "GravityDirection" }, // Area2D
	{ "Hseparation", "HSeparation" }, // Theme
	{ "IterationsPerSecond", "PhysicsTicksPerSecond" }, // Engine
	{ "MarginBottom", "OffsetBottom" }, // Control broke NinePatchRect, StyleBox
	{ "MarginLeft", "OffsetLeft" }, // Control broke NinePatchRect, StyleBox
	{ "MarginRight", "OffsetRight" }, // Control broke NinePatchRect, StyleBox
	{ "MarginTop", "OffsetTop" }, // Control broke NinePatchRect, StyleBox
	{ "MidHeight", "Height" }, // CapsuleMesh
	{ "OffsetH", "DragHorizontalOffset" }, // Camera2D
	{ "OffsetV", "DragVerticalOffset" }, // Camera2D
	{ "Ofs", "Offset" }, // Theme
	{ "OutOfRangeMode", "MaxPolyphony" }, // AudioStreamPlayer3D
	{ "PauseMode", "ProcessMode" }, // Node
	{ "PhysicalScancode", "PhysicalKeycode" }, // InputEventKey
	{ "PopupExclusive", "Exclusive" }, // Window
	{ "RefuseNewNetworkConnections", "RefuseNewConnections" }, // MultiplayerAPI
	{ "RegionFilterClip", "RegionFilterClipEnabled" }, // Sprite2D
	{ "Selectedframe", "SelectedFrame" }, // Theme
	{ "SizeOverrideStretch", "Size2dOverrideStretch" }, // SubViewport
	{ "SlipsOnSlope", "SlideOnSlope" }, // SeparationRayShape2D
	{ "SsReflectionsDepthTolerance", "SsrDepthTolerance" }, // Environment
	{ "SsReflectionsEnabled", "SsrEnabled" }, // Environment
	{ "SsReflectionsFadeIn", "SsrFadeIn" }, // Environment
	{ "SsReflectionsFadeOut", "SsrFadeOut" }, // Environment
	{ "SsReflectionsMaxSteps", "SsrMaxSteps" }, // Environment
	{ "StateMachineSelectedframe", "StateMachineSelectedFrame" }, // Theme
	{ "SyntaxHighlighting", "SyntaxHighlighter" }, // TextEdit
	{ "TabAlign", "TabAlignment" }, // TabContainer
	{ "TableHseparation", "TableHSeparation" }, // Theme
	{ "TableVseparation", "TableVSeparation" }, // Theme
	{ "Translation", "Position" }, // Node3D - broke GLTFNode
	{ "Vseparation", "VSeparation" }, // Theme

	{ nullptr, nullptr },
};

static const char *gdscript_signals_renames[][2] = {
	//  {"instantiate","instance"}, // FileSystemDock
	// { "hide", "hidden" }, // CanvasItem - function with same name exists
	// { "tween_all_completed","loop_finished"}, // Tween - TODO, not sure
	// {"changed","settings_changed"}, // EditorSettings
	{ "about_to_show", "about_to_popup" }, // Popup
	{ "button_release", "button_released" }, // XRController3D
	{ "network_peer_connected", "peer_connected" }, // MultiplayerAPI
	{ "network_peer_disconnected", "peer_disconnected" }, // MultiplayerAPI
	{ "network_peer_packet", "peer_packet" }, // MultiplayerAPI
	{ "node_unselected", "node_deselected" }, // GraphEdit
	{ "offset_changed", "position_offset_changed" }, // GraphNode
	{ "settings_changed", "changed" }, // TileMap broke EditorSettings
	{ "skeleton_updated", "pose_updated" }, //
	{ "tab_close", "tab_closed" }, // TextEdit
	{ "tab_hover", "tab_hovered" }, // TextEdit
	{ "text_entered", "text_submitted" }, // LineEdit
	{ "tween_completed", "finished" }, // Tween
	{ "tween_step", "step_finished" }, // Tween

	{ nullptr, nullptr },
};

static const char *csharp_signals_renames[][2] = {
	//  {"Instantiate","Instance"}, // FileSystemDock
	// { "Hide", "Hidden" }, // CanvasItem - function with same name exists
	// { "TweenAllCompleted","LoopFinished"}, // Tween - TODO, not sure
	// {"Changed","SettingsChanged"}, // EditorSettings
	{ "AboutToShow", "AboutToPopup" }, // Popup
	{ "ButtonRelease", "ButtonReleased" }, // XRController3D
	{ "NetworkPeerConnected", "PeerConnected" }, // MultiplayerAPI
	{ "NetworkPeerDisconnected", "PeerDisconnected" }, // MultiplayerAPI
	{ "NetworkPeerPacket", "PeerPacket" }, // MultiplayerAPI
	{ "NodeUnselected", "NodeDeselected" }, // GraphEdit
	{ "OffsetChanged", "PositionOffsetChanged" }, // GraphNode
	{ "SettingsChanged", "Changed" }, // TileMap broke EditorSettings
	{ "SkeletonUpdated", "PoseUpdated" }, //
	{ "TabClose", "TabClosed" }, // TextEdit
	{ "TabHover", "TabHovered" }, // TextEdit
	{ "TextEntered", "TextSubmitted" }, // LineEdit
	{ "TweenCompleted", "Finished" }, // Tween
	{ "TweenStep", "StepFinished" }, // Tween

	{ nullptr, nullptr },

};

static const char *project_settings_renames[][2] = {
	{ "audio/channel_disable_threshold_db", "audio/buses/channel_disable_threshold_db" },
	{ "audio/channel_disable_time", "audio/buses/channel_disable_time" },
	{ "audio/default_bus_layout", "audio/buses/default_bus_layout" },
	{ "audio/driver", "audio/driver/driver" },
	{ "audio/enable_audio_input", "audio/driver/enable_input" },
	{ "audio/mix_rate", "audio/driver/mix_rate" },
	{ "audio/output_latency", "audio/driver/output_latency" },
	{ "audio/output_latency.web", "audio/driver/output_latency.web" },
	{ "audio/video_delay_compensation_ms", "audio/video/video_delay_compensation_ms" },
	{ "display/window/vsync/use_vsync", "display/window/vsync/vsync_mode" },
	{ "editor/main_run_args", "editor/run/main_run_args" },
	{ "gui/common/swap_ok_cancel", "gui/common/swap_cancel_ok" },
	{ "network/limits/debugger_stdout/max_chars_per_second", "network/limits/debugger/max_chars_per_second" },
	{ "network/limits/debugger_stdout/max_errors_per_second", "network/limits/debugger/max_errors_per_second" },
	{ "network/limits/debugger_stdout/max_messages_per_frame", "network/limits/debugger/max_queued_messages" },
	{ "network/limits/debugger_stdout/max_warnings_per_second", "network/limits/debugger/max_warnings_per_second" },
	{ "network/ssl/certificates", "network/ssl/certificate_bundle_override" },
	{ "physics/2d/thread_model", "physics/2d/run_on_thread" }, // TODO not sure
	{ "rendering/environment/default_clear_color", "rendering/environment/defaults/default_clear_color" },
	{ "rendering/environment/default_environment", "rendering/environment/defaults/default_environment" },
	{ "rendering/quality/depth_prepass/disable_for_vendors", "rendering/driver/depth_prepass/disable_for_vendors" },
	{ "rendering/quality/depth_prepass/enable", "rendering/driver/depth_prepass/enable" },
	{ "rendering/quality/shading/force_blinn_over_ggx", "rendering/shading/overrides/force_blinn_over_ggx" },
	{ "rendering/quality/shading/force_blinn_over_ggx.mobile", "rendering/shading/overrides/force_blinn_over_ggx.mobile" },
	{ "rendering/quality/shading/force_lambert_over_burley", "rendering/shading/overrides/force_lambert_over_burley" },
	{ "rendering/quality/shading/force_lambert_over_burley.mobile", "rendering/shading/overrides/force_lambert_over_burley.mobile" },
	{ "rendering/quality/shading/force_vertex_shading", "rendering/shading/overrides/force_vertex_shading" },
	{ "rendering/quality/shading/force_vertex_shading.mobile", "rendering/shading/overrides/force_vertex_shading.mobile" },
	{ "rendering/quality/shadow_atlas/quadrant_0_subdiv", "rendering/shadows/shadow_atlas/quadrant_0_subdiv" },
	{ "rendering/quality/shadow_atlas/quadrant_1_subdiv", "rendering/shadows/shadow_atlas/quadrant_1_subdiv" },
	{ "rendering/quality/shadow_atlas/quadrant_2_subdiv", "rendering/shadows/shadow_atlas/quadrant_2_subdiv" },
	{ "rendering/quality/shadow_atlas/quadrant_3_subdiv", "rendering/shadows/shadow_atlas/quadrant_3_subdiv" },
	{ "rendering/quality/shadow_atlas/size", "rendering/shadows/shadow_atlas/size" },
	{ "rendering/quality/shadow_atlas/size.mobile", "rendering/shadows/shadow_atlas/size.mobile" },
	{ "rendering/vram_compression/import_bptc", "rendering/textures/vram_compression/import_bptc" },
	{ "rendering/vram_compression/import_etc", "rendering/textures/vram_compression/import_etc" },
	{ "rendering/vram_compression/import_etc2", "rendering/textures/vram_compression/import_etc2" },
	{ "rendering/vram_compression/import_pvrtc", "rendering/textures/vram_compression/import_pvrtc" },
	{ "rendering/vram_compression/import_s3tc", "rendering/textures/vram_compression/import_s3tc" },

	{ nullptr, nullptr },
};

static const char *builtin_types_renames[][2] = {
	{ "PoolByteArray", "PackedByteArray" },
	{ "PoolColorArray", "PackedColorArray" },
	{ "PoolIntArray", "PackedInt32Array" },
	{ "PoolRealArray", "PackedFloat32Array" },
	{ "PoolStringArray", "PackedStringArray" },
	{ "PoolVector2Array", "PackedVector2Array" },
	{ "PoolVector3Array", "PackedVector3Array" },
	{ "Quat", "Quaternion" },
	{ "Transform", "Transform3D" },

	{ nullptr, nullptr },
};

static const char *shaders_renames[][2] = {
	{ "ALPHA_SCISSOR", "ALPHA_SCISSOR_THRESHOLD" },
	{ "NORMALMAP", "NORMAL_MAP" },
	{ "NORMALMAP_DEPTH", "NORMAL_MAP_DEPTH" },
	{ "TRANSMISSION", "SSS_TRANSMITTANCE_COLOR" },
	{ nullptr, nullptr },
};

static const char *class_renames[][2] = {
	// { "BulletPhysicsDirectBodyState", "BulletPhysicsDirectBodyState3D" }, // Class is not visible in ClassDB
	// { "BulletPhysicsServer", "BulletPhysicsServer3D" }, // Class is not visible in ClassDB
	// { "GDScriptFunctionState", "Node3D" }, // TODO - not sure to which should be changed
	// { "GDScriptNativeClass", "Node3D" }, // TODO - not sure to which should be changed
	// { "InputDefault",""}, // TODO ?
	// { "Physics2DDirectBodyStateSW", "GodotPhysicsDirectBodyState2D" }, // Class is not visible in ClassDB
	// { "Physics2DShapeQueryResult", "PhysicsShapeQueryResult2D" }, // Class is not visible in ClassDB
	// { "PhysicsShapeQueryResult", "PhysicsShapeQueryResult3D" }, // Class is not visible in ClassDB
	// { "NativeScript","NativeExtension"}, ??
	{ "ARVRAnchor", "XRAnchor3D" },
	{ "ARVRCamera", "XRCamera3D" },
	{ "ARVRController", "XRController3D" },
	{ "ARVRInterface", "XRInterface" },
	{ "ARVRInterfaceGDNative", "Node3D" },
	{ "ARVROrigin", "XROrigin3D" },
	{ "ARVRPositionalTracker", "XRPositionalTracker" },
	{ "ARVRServer", "XRServer" },
	{ "AStar", "AStar3D" },
	{ "AnimatedSprite", "AnimatedSprite2D" },
	{ "AnimationTreePlayer", "AnimationTree" },
	{ "Area", "Area3D" }, // Be careful, this will be used everywhere
	{ "AudioStreamOGGVorbis", "AudioStreamOggVorbis" },
	{ "AudioStreamRandomPitch", "AudioStreamRandomizer" },
	{ "AudioStreamSample", "AudioStreamWAV" },
	{ "BakedLightmap", "LightmapGI" },
	{ "BakedLightmapData", "LightmapGIData" },
	{ "BitmapFont", "FontFile" },
	{ "BoneAttachment", "BoneAttachment3D" },
	{ "BoxShape", "BoxShape3D" },
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
	{ "Camera", "Camera3D" }, // Be careful, this will be used everywhere
	{ "CapsuleShape", "CapsuleShape3D" },
	{ "ClippedCamera", "Camera3D" },
	{ "CollisionObject", "CollisionObject3D" },
	{ "CollisionPolygon", "CollisionPolygon3D" },
	{ "CollisionShape", "CollisionShape3D" },
	{ "ConcavePolygonShape", "ConcavePolygonShape3D" },
	{ "ConeTwistJoint", "ConeTwistJoint3D" },
	{ "ConvexPolygonShape", "ConvexPolygonShape3D" },
	{ "CubeMap", "Cubemap" },
	{ "CubeMesh", "BoxMesh" },
	{ "CylinderShape", "CylinderShape3D" },
	{ "DirectionalLight", "DirectionalLight3D" },
	{ "DynamicFont", "FontFile" },
	{ "DynamicFontData", "FontFile" },
	{ "EditorNavigationMeshGenerator", "NavigationMeshGenerator" },
	{ "EditorSceneImporter", "EditorSceneFormatImporter" },
	{ "EditorSceneImporterFBX", "EditorSceneFormatImporterFBX" },
	{ "EditorSceneImporterGLTF", "EditorSceneFormatImporterGLTF" },
	{ "EditorSpatialGizmo", "EditorNode3DGizmo" },
	{ "EditorSpatialGizmoPlugin", "EditorNode3DGizmoPlugin" },
	{ "ExternalTexture", "ImageTexture" },
	{ "FuncRef", "Callable" },
	{ "GIProbe", "VoxelGI" },
	{ "GIProbeData", "VoxelGIData" },
	{ "Generic6DOFJoint", "Generic6DOFJoint3D" },
	{ "Geometry", "Geometry2D" }, // Geometry class is split between Geometry2D and Geometry3D so we need to choose one
	{ "GeometryInstance", "GeometryInstance3D" },
	{ "GradientTexture", "GradientTexture2D" },
	{ "HeightMapShape", "HeightMapShape3D" },
	{ "HingeJoint", "HingeJoint3D" },
	{ "IP_Unix", "IPUnix" },
	{ "ImmediateGeometry", "ImmediateMesh" },
	{ "ImmediateGeometry3D", "ImmediateMesh" },
	{ "InterpolatedCamera", "Camera3D" },
	{ "InterpolatedCamera3D", "Camera3D" },
	{ "JSONParseResult", "JSON" },
	{ "Joint", "Joint3D" },
	{ "KinematicBody", "CharacterBody3D" },
	{ "KinematicBody2D", "CharacterBody2D" },
	{ "KinematicCollision", "KinematicCollision3D" },
	{ "LargeTexture", "ImageTexture" },
	{ "Light", "Light3D" },
	{ "Light2D", "PointLight2D" },
	{ "LineShape2D", "WorldBoundaryShape2D" },
	{ "Listener", "AudioListener3D" },
	{ "Listener2D", "AudioListener2D" },
	{ "MeshInstance", "MeshInstance3D" },
	{ "MultiMeshInstance", "MultiMeshInstance3D" },
	{ "MultiplayerPeerGDNative", "MultiplayerPeerExtension" },
	{ "Navigation", "Node3D" },
	{ "Navigation2D", "Node2D" },
	{ "Navigation2DServer", "NavigationServer2D" },
	{ "Navigation3D", "Node3D" },
	{ "NavigationAgent", "NavigationAgent3D" },
	{ "NavigationMeshInstance", "NavigationRegion3D" },
	{ "NavigationObstacle", "NavigationObstacle3D" },
	{ "NavigationPolygonInstance", "NavigationRegion2D" },
	{ "NavigationRegion", "NavigationRegion3D" },
	{ "NavigationServer", "NavigationServer3D" },
	{ "NetworkedMultiplayerENet", "ENetMultiplayerPeer" },
	{ "NetworkedMultiplayerPeer", "MultiplayerPeer" },
	{ "Occluder", "OccluderInstance3D" },
	{ "OmniLight", "OmniLight3D" },
	{ "PHashTranslation", "OptimizedTranslation" },
	{ "PacketPeerGDNative", "PacketPeerExtension" },
	{ "PanoramaSky", "Sky" },
	{ "Particles", "GPUParticles3D" }, // Be careful, this will be used everywhere
	{ "Particles2D", "GPUParticles2D" },
	{ "Path", "Path3D" }, // Be careful, this will be used everywhere
	{ "PathFollow", "PathFollow3D" },
	{ "PhysicalBone", "PhysicalBone3D" },
	{ "Physics2DDirectBodyState", "PhysicsDirectBodyState2D" },
	{ "Physics2DDirectSpaceState", "PhysicsDirectSpaceState2D" },
	{ "Physics2DServer", "PhysicsServer2D" },
	{ "Physics2DServerSW", "GodotPhysicsServer2D" },
	{ "Physics2DShapeQueryParameters", "PhysicsShapeQueryParameters2D" },
	{ "Physics2DTestMotionResult", "PhysicsTestMotionResult2D" },
	{ "PhysicsBody", "PhysicsBody3D" },
	{ "PhysicsDirectBodyState", "PhysicsDirectBodyState3D" },
	{ "PhysicsDirectSpaceState", "PhysicsDirectSpaceState3D" },
	{ "PhysicsServer", "PhysicsServer3D" },
	{ "PhysicsShapeQueryParameters", "PhysicsShapeQueryParameters3D" },
	{ "PhysicsTestMotionResult", "PhysicsTestMotionResult3D" },
	{ "PinJoint", "PinJoint3D" },
	{ "PlaneShape", "WorldBoundaryShape3D" },
	{ "PopupDialog", "Popup" },
	{ "ProceduralSky", "Sky" },
	{ "RayCast", "RayCast3D" },
	{ "RayShape", "SeparationRayShape3D" },
	{ "RayShape2D", "SeparationRayShape2D" },
	{ "Reference", "RefCounted" }, // Be careful, this will be used everywhere
	{ "RemoteTransform", "RemoteTransform3D" },
	{ "ResourceInteractiveLoader", "ResourceLoader" },
	{ "RigidBody", "RigidDynamicBody3D" },
	{ "RigidBody2D", "RigidDynamicBody2D" },
	{ "SceneTreeTween", "Tween" },
	{ "Shape", "Shape3D" }, // Be careful, this will be used everywhere
	{ "ShortCut", "Shortcut" },
	{ "Skeleton", "Skeleton3D" },
	{ "SkeletonIK", "SkeletonIK3D" },
	{ "SliderJoint", "SliderJoint3D" },
	{ "SoftBody", "SoftDynamicBody3D" },
	{ "Spatial", "Node3D" },
	{ "SpatialGizmo", "Node3DGizmo" },
	{ "SpatialMaterial", "StandardMaterial3D" },
	{ "SpatialVelocityTracker", "VelocityTracker3D" },
	{ "SphereShape", "SphereShape3D" },
	{ "SpotLight", "SpotLight3D" },
	{ "SpringArm", "SpringArm3D" },
	{ "Sprite", "Sprite2D" },
	{ "StaticBody", "StaticBody3D" },
	{ "StreamCubemap", "CompressedCubemap" },
	{ "StreamCubemapArray", "CompressedCubemapArray" },
	{ "StreamPeerGDNative", "StreamPeerExtension" },
	{ "StreamTexture", "CompressedTexture2D" },
	{ "StreamTexture2D", "CompressedTexture2D" },
	{ "StreamTexture2DArray", "CompressedTexture2DArray" },
	{ "StreamTextureLayered", "CompressedTextureLayered" },
	{ "TCP_Server", "TCPServer" },
	{ "Tabs", "TabBar" }, // Be careful, this will be used everywhere
	{ "TextFile", "Node3D" },
	{ "Texture", "Texture2D" }, // May broke TextureRect
	{ "TextureArray", "Texture2DArray" },
	{ "TextureProgress", "TextureProgressBar" },
	{ "ToolButton", "Button" },
	{ "VehicleBody", "VehicleBody3D" },
	{ "VehicleWheel", "VehicleWheel3D" },
	{ "VideoPlayer", "VideoStreamPlayer" },
	{ "Viewport", "SubViewport" },
	{ "ViewportContainer", "SubViewportContainer" },
	{ "VisibilityEnabler", "VisibleOnScreenEnabler3D" },
	{ "VisibilityEnabler2D", "VisibleOnScreenEnabler2D" },
	{ "VisibilityNotifier", "VisibleOnScreenNotifier3D" },
	{ "VisibilityNotifier2D", "VisibleOnScreenNotifier2D" },
	{ "VisibilityNotifier3D", "VisibleOnScreenNotifier3D" },
	{ "VisualInstance", "VisualInstance3D" },
	{ "VisualServer", "RenderingServer" },
	{ "VisualShaderNodeCubeMap", "VisualShaderNodeCubemap" },
	{ "VisualShaderNodeCubeMapUniform", "VisualShaderNodeCubemapUniform" },
	{ "VisualShaderNodeScalarClamp", "VisualShaderNodeClamp" },
	{ "VisualShaderNodeScalarConstant", "VisualShaderNodeFloatConstant" },
	{ "VisualShaderNodeScalarFunc", "VisualShaderNodeFloatFunc" },
	{ "VisualShaderNodeScalarInterp", "VisualShaderNodeMix" },
	{ "VisualShaderNodeScalarOp", "VisualShaderNodeFloatOp" },
	{ "VisualShaderNodeScalarSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeScalarSwitch", "VisualShaderNodeSwitch" },
	{ "VisualShaderNodeScalarTransformMult", "VisualShaderNodeTransformOp" },
	{ "VisualShaderNodeScalarUniform", "VisualShaderNodeFloatUniform" },
	{ "VisualShaderNodeTransformMult", "VisualShaderNode" },
	{ "VisualShaderNodeVectorClamp", "VisualShaderNodeClamp" },
	{ "VisualShaderNodeVectorInterp", "VisualShaderNodeMix" },
	{ "VisualShaderNodeVectorScalarMix", "VisualShaderNodeMix" },
	{ "VisualShaderNodeVectorScalarSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeVectorScalarStep", "VisualShaderNodeStep" },
	{ "VisualShaderNodeVectorSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "WebRTCDataChannelGDNative", "WebRTCDataChannelExtension" },
	{ "WebRTCMultiplayer", "WebRTCMultiplayerPeer" },
	{ "WebRTCPeerConnectionGDNative", "WebRTCPeerConnectionExtension" },
	{ "WindowDialog", "Window" },
	{ "World", "World3D" }, // Be careful, this will be used everywhere
	{ "XRAnchor", "XRAnchor3D" },
	{ "XRController", "XRController3D" },
	{ "XROrigin", "XROrigin3D" },
	{ "YSort", "Node2D" },

	{ "CullInstance", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "RoomGroup", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "Room", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "RoomManager", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "Portal", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x

	{ nullptr, nullptr },
};

// TODO - this colors needs to be validated(not all are valid)
static const char *colors_renames[][2] = {
	{ "aliceblue", "ALICE_BLUE" },
	{ "antiquewhite", "ANTIQUE_WHITE" },
	{ "aqua", "AQUA" },
	{ "aquamarine", "AQUAMARINE" },
	{ "azure", "AZURE" },
	{ "beige", "BEIGE" },
	{ "bisque", "BISQUE" },
	{ "black", "BLACK" },
	{ "blanchedalmond", "BLANCHED_ALMOND" },
	{ "blue", "BLUE" },
	{ "blueviolet", "BLUE_VIOLET" },
	{ "brown", "BROWN" },
	{ "burlywood", "BURLYWOOD" },
	{ "cadetblue", "CADET_BLUE" },
	{ "chartreuse", "CHARTREUSE" },
	{ "chocolate", "CHOCOLATE" },
	{ "coral", "CORAL" },
	{ "cornflowerblue", "CORNFLOWER_BLUE" },
	{ "cornsilk", "CORNSILK" },
	{ "crimson", "CRIMSON" },
	{ "cyan", "CYAN" },
	{ "darkblue", "DARK_BLUE" },
	{ "darkcyan", "DARK_CYAN" },
	{ "darkgoldenrod", "DARK_GOLDENROD" },
	{ "darkgray", "DARK_GRAY" },
	{ "darkgreen", "DARK_GREEN" },
	{ "darkkhaki", "DARK_KHAKI" },
	{ "darkmagenta", "DARK_MAGENTA" },
	{ "darkolivegreen", "DARK_OLIVE_GREEN" },
	{ "darkorange", "DARK_ORANGE" },
	{ "darkorchid", "DARK_ORCHID" },
	{ "darkred", "DARK_RED" },
	{ "darksalmon", "DARK_SALMON" },
	{ "darkseagreen", "DARK_SEA_GREEN" },
	{ "darkslateblue", "DARK_SLATE_BLUE" },
	{ "darkslategray", "DARK_SLATE_GRAY" },
	{ "darkturquoise", "DARK_TURQUOISE" },
	{ "darkviolet", "DARK_VIOLET" },
	{ "deeppink", "DEEP_PINK" },
	{ "deepskyblue", "DEEP_SKY_BLUE" },
	{ "dimgray", "DIM_GRAY" },
	{ "dodgerblue", "DODGER_BLUE" },
	{ "firebrick", "FIREBRICK" },
	{ "floralwhite", "FLORAL_WHITE" },
	{ "forestgreen", "FOREST_GREEN" },
	{ "fuchsia", "FUCHSIA" },
	{ "gainsboro", "GAINSBORO" },
	{ "ghostwhite", "GHOST_WHITE" },
	{ "gold", "GOLD" },
	{ "goldenrod", "GOLDENROD" },
	{ "gray", "GRAY" },
	{ "green", "GREEN" },
	{ "greenyellow", "GREEN_YELLOW" },
	{ "honeydew", "HONEYDEW" },
	{ "hotpink", "HOT_PINK" },
	{ "indianred", "INDIAN_RED" },
	{ "indigo", "INDIGO" },
	{ "ivory", "IVORY" },
	{ "khaki", "KHAKI" },
	{ "lavender", "LAVENDER" },
	{ "lavenderblush", "LAVENDER_BLUSH" },
	{ "lawngreen", "LAWN_GREEN" },
	{ "lemonchiffon", "LEMON_CHIFFON" },
	{ "lightblue", "LIGHT_BLUE" },
	{ "lightcoral", "LIGHT_CORAL" },
	{ "lightcyan", "LIGHT_CYAN" },
	{ "lightgoldenrod", "LIGHT_GOLDENROD" },
	{ "lightgray", "LIGHT_GRAY" },
	{ "lightgreen", "LIGHT_GREEN" },
	{ "lightpink", "LIGHT_PINK" },
	{ "lightsalmon", "LIGHT_SALMON" },
	{ "lightseagreen", "LIGHT_SEA_GREEN" },
	{ "lightskyblue", "LIGHT_SKY_BLUE" },
	{ "lightslategray", "LIGHT_SLATE_GRAY" },
	{ "lightsteelblue", "LIGHT_STEEL_BLUE" },
	{ "lightyellow", "LIGHT_YELLOW" },
	{ "lime", "LIME" },
	{ "limegreen", "LIME_GREEN" },
	{ "linen", "LINEN" },
	{ "magenta", "MAGENTA" },
	{ "maroon", "MAROON" },
	{ "mediumaquamarine", "MEDIUM_AQUAMARINE" },
	{ "mediumblue", "MEDIUM_BLUE" },
	{ "mediumorchid", "MEDIUM_ORCHID" },
	{ "mediumpurple", "MEDIUM_PURPLE" },
	{ "mediumseagreen", "MEDIUM_SEA_GREEN" },
	{ "mediumslateblue", "MEDIUM_SLATE_BLUE" },
	{ "mediumspringgreen", "MEDIUM_SPRING_GREEN" },
	{ "mediumturquoise", "MEDIUM_TURQUOISE" },
	{ "mediumvioletred", "MEDIUM_VIOLET_RED" },
	{ "midnightblue", "MIDNIGHT_BLUE" },
	{ "mintcream", "MINT_CREAM" },
	{ "mistyrose", "MISTY_ROSE" },
	{ "moccasin", "MOCCASIN" },
	{ "navajowhite", "NAVAJO_WHITE" },
	{ "navyblue", "NAVY_BLUE" },
	{ "oldlace", "OLD_LACE" },
	{ "olive", "OLIVE" },
	{ "olivedrab", "OLIVE_DRAB" },
	{ "orange", "ORANGE" },
	{ "orangered", "ORANGE_RED" },
	{ "orchid", "ORCHID" },
	{ "palegoldenrod", "PALE_GOLDENROD" },
	{ "palegreen", "PALE_GREEN" },
	{ "paleturquoise", "PALE_TURQUOISE" },
	{ "palevioletred", "PALE_VIOLET_RED" },
	{ "papayawhip", "PAPAYA_WHIP" },
	{ "peachpuff", "PEACH_PUFF" },
	{ "peru", "PERU" },
	{ "pink", "PINK" },
	{ "plum", "PLUM" },
	{ "powderblue", "POWDER_BLUE" },
	{ "purple", "PURPLE" },
	{ "rebeccapurple", "REBECCA_PURPLE" },
	{ "red", "RED" },
	{ "rosybrown", "ROSY_BROWN" },
	{ "royalblue", "ROYAL_BLUE" },
	{ "saddlebrown", "SADDLE_BROWN" },
	{ "salmon", "SALMON" },
	{ "sandybrown", "SANDY_BROWN" },
	{ "seagreen", "SEA_GREEN" },
	{ "seashell", "SEASHELL" },
	{ "sienna", "SIENNA" },
	{ "silver", "SILVER" },
	{ "skyblue", "SKY_BLUE" },
	{ "slateblue", "SLATE_BLUE" },
	{ "slategray", "SLATE_GRAY" },
	{ "snow", "SNOW" },
	{ "springgreen", "SPRING_GREEN" },
	{ "steelblue", "STEEL_BLUE" },
	{ "tan", "TAN" },
	{ "teal", "TEAL" },
	{ "thistle", "THISTLE" },
	{ "tomato", "TOMATO" },
	{ "transparent", "TRANSPARENT" },
	{ "turquoise", "TURQUOISE" },
	{ "violet", "VIOLET" },
	{ "webgray", "WEB_GRAY" },
	{ "webgreen", "WEB_GREEN" },
	{ "webmaroon", "WEB_MAROON" },
	{ "webpurple", "WEB_PURPLE" },
	{ "wheat", "WHEAT" },
	{ "white", "WHITE" },
	{ "whitesmoke", "WHITE_SMOKE" },
	{ "yellow", "YELLOW" },
	{ "yellowgreen", "YELLOW_GREEN" },

	{ nullptr, nullptr },
};

class ProjectConverter3To4::RegExContainer {
public:
	RegEx reg_is_empty = RegEx("\\bempty\\(");
	RegEx reg_super = RegEx("([\t ])\\.([a-zA-Z_])");
	RegEx reg_json_to = RegEx("\\bto_json\\b");
	RegEx reg_json_parse = RegEx("([\t]{0,})([^\n]+)parse_json\\(([^\n]+)");
	RegEx reg_json_non_new = RegEx("([\t]{0,})([^\n]+)JSON\\.parse\\(([^\n]+)");
	RegEx reg_export = RegEx("export\\(([a-zA-Z0-9_]+)\\)[ ]+var[ ]+([a-zA-Z0-9_]+)");
	RegEx reg_export_advanced = RegEx("export\\(([^)^\n]+)\\)[ ]+var[ ]+([a-zA-Z0-9_]+)([^\n]+)");
	RegEx reg_setget_setget = RegEx("var[ ]+([a-zA-Z0-9_]+)([^\n]+)setget[ \t]+([a-zA-Z0-9_]+)[ \t]*,[ \t]*([a-zA-Z0-9_]+)");
	RegEx reg_setget_set = RegEx("var[ ]+([a-zA-Z0-9_]+)([^\n]+)setget[ \t]+([a-zA-Z0-9_]+)[ \t]*[,]*[^a-z^A-Z^0-9^_]*$");
	RegEx reg_setget_get = RegEx("var[ ]+([a-zA-Z0-9_]+)([^\n]+)setget[ \t]+,[ \t]*([a-zA-Z0-9_]+)[ \t]*$");
	RegEx reg_join = RegEx("([\\(\\)a-zA-Z0-9_]+)\\.join\\(([^\n^\\)]+)\\)");
	RegEx reg_mixed_tab_space = RegEx("([\t]+)([ ]+)");
	RegEx reg_image_lock = RegEx("([a-zA-Z0-9_\\.]+)\\.lock\\(\\)");
	RegEx reg_image_unlock = RegEx("([a-zA-Z0-9_\\.]+)\\.unlock\\(\\)");
	RegEx reg_os_fullscreen = RegEx("OS.window_fullscreen[= ]+([^#^\n]+)");
};

// Function responsible for converting project
int ProjectConverter3To4::convert() {
	print_line("Starting conversion.");

	RegExContainer reg_container = RegExContainer();

	ERR_FAIL_COND_V_MSG(!test_array_names(), ERROR_CODE, "Cannot start converting due to problems with data in arrays.");
	ERR_FAIL_COND_V_MSG(!test_conversion(reg_container), ERROR_CODE, "Cannot start converting due to problems with converting arrays.");

	// Checking if folder contains valid Godot 3 project.
	// Project cannot be converted 2 times
	{
		String conventer_text = "; Project was converted by built-in tool to Godot 4.0";

		ERR_FAIL_COND_V_MSG(!FileAccess::exists("project.godot"), ERROR_CODE, "Current directory doesn't contains any Godot 3 project");

		Error err = OK;
		String project_godot_content = FileAccess::get_file_as_string("project.godot", &err);

		ERR_FAIL_COND_V_MSG(err != OK, ERROR_CODE, "Failed to read content of \"project.godot\" file.");
		ERR_FAIL_COND_V_MSG(project_godot_content.find(conventer_text) != -1, ERROR_CODE, "Project already was converted with this tool.");

		Ref<FileAccess> file = FileAccess::open("project.godot", FileAccess::WRITE);
		ERR_FAIL_COND_V_MSG(file.is_null(), ERROR_CODE, "Failed to open project.godot file.");

		file->store_string(conventer_text + "\n" + project_godot_content);
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
		uint64_t file_size = file_content.size();
		print_line("Trying to convert\t" + itos(i + 1) + "/" + itos(collected_files.size()) + " file - \"" + file_name.trim_prefix("res://") + "\" with size - " + itos(file_size / 1024) + " KB");

		Vector<String> reason;
		bool is_ignored = false;
		uint64_t start_time = Time::get_singleton()->get_ticks_msec();

		if (file_name.ends_with(".shader")) {
			DirAccess::remove_file_or_error(file_name.trim_prefix("res://"));
			file_name = file_name.replace(".shader", ".gdshader");
		}

		if (file_size < CONVERSION_MAX_FILE_SIZE) {
			// TSCN must be the same work exactly same as .gd file because it may contains builtin script
			if (file_name.ends_with(".gd")) {
				rename_classes(file_content); // Using only specialized function

				rename_common(enum_renames, file_content);
				rename_enums(file_content); // Require to additional rename

				rename_common(gdscript_function_renames, file_content);
				rename_gdscript_functions(file_content, reg_container, false); // Require to additional rename

				rename_common(project_settings_renames, file_content);
				rename_gdscript_keywords(file_content);
				rename_common(gdscript_properties_renames, file_content);
				rename_common(gdscript_signals_renames, file_content);
				rename_common(shaders_renames, file_content);
				rename_common(builtin_types_renames, file_content);

				custom_rename(file_content, "\\.shader", ".gdshader");
				custom_rename(file_content, "instance", "instantiate");
			} else if (file_name.ends_with(".tscn")) {
				rename_classes(file_content); // Using only specialized function

				rename_common(enum_renames, file_content);
				rename_enums(file_content); // Require to additional rename

				rename_common(gdscript_function_renames, file_content);
				rename_gdscript_functions(file_content, reg_container, true); // Require to additional rename

				rename_common(project_settings_renames, file_content);
				rename_gdscript_keywords(file_content);
				rename_common(gdscript_properties_renames, file_content);
				rename_common(gdscript_signals_renames, file_content);
				rename_common(shaders_renames, file_content);
				rename_common(builtin_types_renames, file_content);

				custom_rename(file_content, "\\.shader", ".gdshader");
			} else if (file_name.ends_with(".cs")) { // TODO, C# should use different methods
				rename_classes(file_content); // Using only specialized function
				rename_common(csharp_function_renames, file_content);
				rename_common(builtin_types_renames, file_content);
				rename_common(csharp_properties_renames, file_content);
				rename_common(csharp_signals_renames, file_content);
				rename_csharp_functions(file_content);
				rename_csharp_attributes(file_content);
				custom_rename(file_content, "public class ", "public partial class ");
			} else if (file_name.ends_with(".gdshader") || file_name.ends_with(".shader")) {
				rename_common(shaders_renames, file_content);
			} else if (file_name.ends_with("tres")) {
				rename_classes(file_content); // Using only specialized function

				rename_common(shaders_renames, file_content);
				rename_common(builtin_types_renames, file_content);

				custom_rename(file_content, "\\.shader", ".gdshader");
			} else if (file_name.ends_with("project.godot")) {
				rename_common(project_settings_renames, file_content);
				rename_common(builtin_types_renames, file_content);
			} else if (file_name.ends_with(".csproj")) {
				// TODO
			} else {
				ERR_PRINT(file_name + " is not supported!");
				continue;
			}
		} else {
			reason.append("    ERROR: File has exceeded the maximum size allowed - " + itos(CONVERSION_MAX_FILE_SIZE_MB) + " MB");
			is_ignored = true;
		}

		uint64_t end_time = Time::get_singleton()->get_ticks_msec();

		if (!is_ignored) {
			uint64_t hash_after = file_content.hash64();
			// Don't need to save file without any changes
			// Save if this is a shader, because it was renamed
			if (hash_before != hash_after || file_name.ends_with(".gdshader")) {
				converted_files++;

				Ref<FileAccess> file = FileAccess::open(file_name, FileAccess::WRITE);
				ERR_CONTINUE_MSG(file.is_null(), "Failed to open \"" + file_name + "\" to save data to file.");
				file->store_string(file_content);
				reason.append("    File was changed, conversion took " + itos(end_time - start_time) + " ms.");
			} else {
				reason.append("    File was not changed, checking took " + itos(end_time - start_time) + " ms.");
			}
		}
		for (int k = 0; k < reason.size(); k++) {
			print_line(reason[k]);
		}
	}

	print_line("Conversion ended - all files(" + itos(collected_files.size()) + "), converted files(" + itos(converted_files) + "), not converted files(" + itos(collected_files.size() - converted_files) + ").");
	return 0;
};

// Function responsible for validating project conversion.
int ProjectConverter3To4::validate_conversion() {
	print_line("Starting checking if project conversion can be done.");

	RegExContainer reg_container = RegExContainer();

	ERR_FAIL_COND_V_MSG(!test_array_names(), ERROR_CODE, "Cannot start converting due to problems with data in arrays.");
	ERR_FAIL_COND_V_MSG(!test_conversion(reg_container), ERROR_CODE, "Cannot start converting due to problems with converting arrays.");

	// Checking if folder contains valid Godot 3 project.
	// Project cannot be converted 2 times
	{
		String conventer_text = "; Project was converted by built-in tool to Godot 4.0";

		ERR_FAIL_COND_V_MSG(!FileAccess::exists("project.godot"), ERROR_CODE, "Current directory doesn't contains any Godot 3 project");

		Error err = OK;
		String project_godot_content = FileAccess::get_file_as_string("project.godot", &err);

		ERR_FAIL_COND_V_MSG(err != OK, ERROR_CODE, "Failed to read content of \"project.godot\" file.");
		ERR_FAIL_COND_V_MSG(project_godot_content.find(conventer_text) != -1, ERROR_CODE, "Project already was converted with this tool.");
	}

	Vector<String> collected_files = check_for_files();

	uint32_t converted_files = 0;

	// Check file by file
	for (int i = 0; i < collected_files.size(); i++) {
		String file_name = collected_files[i];
		Vector<String> file_content;
		uint64_t file_size = 0;
		{
			Ref<FileAccess> file = FileAccess::open(file_name, FileAccess::READ);
			ERR_CONTINUE_MSG(file.is_null(), "Failed to read content of \"" + file_name + "\".");
			while (!file->eof_reached()) {
				String line = file->get_line();
				file_size += line.size();
				file_content.append(line);
			}
		}
		print_line("Checking for conversion - " + itos(i + 1) + "/" + itos(collected_files.size()) + " file - \"" + file_name.trim_prefix("res://") + "\" with size - " + itos(file_size / 1024) + " KB");

		Vector<String> changed_elements;
		Vector<String> reason;
		bool is_ignored = false;
		uint64_t start_time = Time::get_singleton()->get_ticks_msec();

		if (file_name.ends_with(".shader")) {
			reason.append("\tFile extension will be renamed from `shader` to `gdshader`.");
		}

		if (file_size < CONVERSION_MAX_FILE_SIZE) {
			if (file_name.ends_with(".gd")) {
				changed_elements.append_array(check_for_rename_classes(file_content));

				changed_elements.append_array(check_for_rename_common(enum_renames, file_content));
				changed_elements.append_array(check_for_rename_enums(file_content));

				changed_elements.append_array(check_for_rename_common(gdscript_function_renames, file_content));
				changed_elements.append_array(check_for_rename_gdscript_functions(file_content, reg_container, false));

				changed_elements.append_array(check_for_rename_common(project_settings_renames, file_content));
				changed_elements.append_array(check_for_rename_gdscript_keywords(file_content));
				changed_elements.append_array(check_for_rename_common(gdscript_properties_renames, file_content));
				changed_elements.append_array(check_for_rename_common(gdscript_signals_renames, file_content));
				changed_elements.append_array(check_for_rename_common(shaders_renames, file_content));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, file_content));

				changed_elements.append_array(check_for_custom_rename(file_content, "instance", "instantiate"));
				changed_elements.append_array(check_for_custom_rename(file_content, "\\.shader", ".gdshader"));
			} else if (file_name.ends_with(".tscn")) {
				changed_elements.append_array(check_for_rename_classes(file_content));

				changed_elements.append_array(check_for_rename_common(enum_renames, file_content));
				changed_elements.append_array(check_for_rename_enums(file_content));

				changed_elements.append_array(check_for_rename_common(gdscript_function_renames, file_content));
				changed_elements.append_array(check_for_rename_gdscript_functions(file_content, reg_container, true));

				changed_elements.append_array(check_for_rename_common(project_settings_renames, file_content));
				changed_elements.append_array(check_for_rename_gdscript_keywords(file_content));
				changed_elements.append_array(check_for_rename_common(gdscript_properties_renames, file_content));
				changed_elements.append_array(check_for_rename_common(gdscript_signals_renames, file_content));
				changed_elements.append_array(check_for_rename_common(shaders_renames, file_content));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, file_content));

				changed_elements.append_array(check_for_custom_rename(file_content, "\\.shader", ".gdshader"));
			} else if (file_name.ends_with(".cs")) {
				changed_elements.append_array(check_for_rename_common(class_renames, file_content));
				changed_elements.append_array(check_for_rename_common(csharp_function_renames, file_content));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, file_content));
				changed_elements.append_array(check_for_rename_common(csharp_properties_renames, file_content));
				changed_elements.append_array(check_for_rename_common(csharp_signals_renames, file_content));
				changed_elements.append_array(check_for_rename_csharp_functions(file_content));
				changed_elements.append_array(check_for_rename_csharp_attributes(file_content));
				changed_elements.append_array(check_for_custom_rename(file_content, "public class ", "public partial class "));
			} else if (file_name.ends_with(".gdshader") || file_name.ends_with(".shader")) {
				changed_elements.append_array(check_for_rename_common(shaders_renames, file_content));
			} else if (file_name.ends_with("tres")) {
				changed_elements.append_array(check_for_rename_classes(file_content));

				changed_elements.append_array(check_for_rename_common(shaders_renames, file_content));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, file_content));

				changed_elements.append_array(check_for_custom_rename(file_content, "\\.shader", ".gdshader"));
			} else if (file_name.ends_with("project.godot")) {
				changed_elements.append_array(check_for_rename_common(project_settings_renames, file_content));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, file_content));
			} else if (file_name.ends_with(".csproj")) {
				// TODO
			} else {
				ERR_PRINT(file_name + " is not supported!");
				continue;
			}
		} else {
			reason.append("\tERROR: File has exceeded the maximum size allowed  - " + itos(CONVERSION_MAX_FILE_SIZE_MB) + " MB");
			is_ignored = true;
		}

		uint64_t end_time = Time::get_singleton()->get_ticks_msec();
		print_line("    Checking file took " + itos(end_time - start_time) + " ms.");

		for (int k = 0; k < reason.size(); k++) {
			print_line(reason[k]);
		}

		if (changed_elements.size() > 0 && !is_ignored) {
			converted_files++;

			for (int k = 0; k < changed_elements.size(); k++) {
				print_line(String("\t\t") + changed_elements[k]);
			}
		}
	}

	print_line("Checking for valid conversion ended - all files(" + itos(collected_files.size()) + "), files which would be converted(" + itos(converted_files) + "), files which would not be converted(" + itos(collected_files.size() - converted_files) + ").");
	return 0;
}

// Collect files which will be checked, it will not touch txt, mp4, wav etc. files
Vector<String> ProjectConverter3To4::check_for_files() {
	Vector<String> collected_files = Vector<String>();

	Vector<String> directories_to_check = Vector<String>();
	directories_to_check.push_back("res://");

	core_bind::Directory dir = core_bind::Directory();
	while (!directories_to_check.is_empty()) {
		String path = directories_to_check.get(directories_to_check.size() - 1); // Is there any pop_back function?
		directories_to_check.resize(directories_to_check.size() - 1); // Remove last element
		if (dir.open(path) == OK) {
			dir.set_include_hidden(true);
			dir.list_dir_begin();
			String current_dir = dir.get_current_dir();
			String file_name = dir.get_next();

			while (file_name != "") {
				if (file_name == ".git" || file_name == ".import" || file_name == ".godot") {
					file_name = dir.get_next();
					continue;
				}
				if (dir.current_is_dir()) {
					directories_to_check.append(current_dir.plus_file(file_name) + "/");
				} else {
					bool proper_extension = false;
					if (file_name.ends_with(".gd") || file_name.ends_with(".shader") || file_name.ends_with(".tscn") || file_name.ends_with(".tres") || file_name.ends_with(".godot") || file_name.ends_with(".cs") || file_name.ends_with(".csproj"))
						proper_extension = true;

					if (proper_extension) {
						collected_files.append(current_dir.plus_file(file_name));
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

bool ProjectConverter3To4::test_conversion_single_additional(String name, String expected, void (ProjectConverter3To4::*func)(String &), String what) {
	String got = name;
	(this->*func)(got);
	if (expected != got) {
		ERR_PRINT("Failed to convert " + what + " `" + name + "` to `" + expected + "`, got instead `" + got + "`");
		return false;
	}

	return true;
}

bool ProjectConverter3To4::test_conversion_single_additional_builtin(String name, String expected, void (ProjectConverter3To4::*func)(String &, const RegExContainer &, bool), String what, const RegExContainer &reg_container, bool builtin_script) {
	String got = name;
	(this->*func)(got, reg_container, builtin_script);
	if (expected != got) {
		ERR_PRINT("Failed to convert " + what + " `" + name + "` to `" + expected + "`, got instead `" + got + "`");
		return false;
	}

	return true;
}

bool ProjectConverter3To4::test_conversion_single_normal(String name, String expected, const char *array[][2], String what) {
	String got = name;
	rename_common(array, got);
	if (expected != got) {
		ERR_PRINT("Failed to convert " + what + " `" + name + "` to `" + expected + "`, got instead `" + got + "`");
		return false;
	}
	return true;
}

// Validate if conversions are proper
bool ProjectConverter3To4::test_conversion(const RegExContainer &reg_container) {
	bool valid = true;

	valid = valid & test_conversion_single_normal("Spatial", "Node3D", class_renames, "class");

	valid = valid & test_conversion_single_normal("TYPE_REAL", "TYPE_FLOAT", enum_renames, "enum");

	valid = valid & test_conversion_single_normal("can_instance", "can_instantiate", gdscript_function_renames, "gdscript function");

	valid = valid & test_conversion_single_normal("CanInstance", "CanInstantiate", csharp_function_renames, "csharp function");

	valid = valid & test_conversion_single_normal("translation", "position", gdscript_properties_renames, "gdscript property");

	valid = valid & test_conversion_single_normal("Translation", "Position", csharp_properties_renames, "csharp property");

	valid = valid & test_conversion_single_normal("NORMALMAP", "NORMAL_MAP", shaders_renames, "shader");

	valid = valid & test_conversion_single_normal("text_entered", "text_submitted", gdscript_signals_renames, "gdscript signal");

	valid = valid & test_conversion_single_normal("TextEntered", "TextSubmitted", csharp_signals_renames, "csharp signal");

	valid = valid & test_conversion_single_normal("audio/channel_disable_threshold_db", "audio/buses/channel_disable_threshold_db", project_settings_renames, "project setting");

	valid = valid & test_conversion_single_normal("Transform", "Transform3D", builtin_types_renames, "builtin type");

	// Custom Renames

	valid = valid & test_conversion_single_additional("(Connect(A,B,C,D,E,F,G) != OK):", "(Connect(A,new Callable(B,C),D,E,F,G) != OK):", &ProjectConverter3To4::rename_csharp_functions, "custom rename csharp");
	valid = valid & test_conversion_single_additional("(Disconnect(A,B,C) != OK):", "(Disconnect(A,new Callable(B,C)) != OK):", &ProjectConverter3To4::rename_csharp_functions, "custom rename csharp");
	valid = valid & test_conversion_single_additional("(IsConnected(A,B,C) != OK):", "(IsConnected(A,new Callable(B,C)) != OK):", &ProjectConverter3To4::rename_csharp_functions, "custom rename");

	valid = valid & test_conversion_single_additional("[Remote]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[RemoteSync]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[Sync]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[Slave]", "[RPC]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[Puppet]", "[RPC]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[PuppetSync]", "[RPC(CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[Master]", "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using Multiplayer.GetRemoteSenderId()\n[RPC]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");
	valid = valid & test_conversion_single_additional("[MasterSync]", "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using Multiplayer.GetRemoteSenderId()\n[RPC(CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp");

	valid = valid & test_conversion_single_additional_builtin("OS.window_fullscreen = Settings.fullscreen", "ProjectSettings.set(\"display/window/size/fullscreen\", Settings.fullscreen)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("OS.window_fullscreen = Settings.fullscreen", "ProjectSettings.set(\\\"display/window/size/fullscreen\\\", Settings.fullscreen)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, true);
	valid = valid & test_conversion_single_additional_builtin("OS.get_window_safe_area()", "DisplayServer.get_display_safe_area()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("\tvar aa = roman(r.move_and_slide( a, b, c, d, e, f )) # Roman", "\tr.set_velocity(a)\n\tr.set_up_direction(b)\n\tr.set_floor_stop_on_slope_enabled(c)\n\tr.set_max_slides(d)\n\tr.set_floor_max_angle(e)\n\t# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `f`\n\tr.move_and_slide()\n\tvar aa = roman(r.velocity) # Roman", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\tvar aa = roman(r.move_and_slide_with_snap( a, g, b, c, d, e, f )) # Roman", "\tr.set_velocity(a)\n\t# TODOConverter40 looks that snap in Godot 4.0 is float, not vector like in Godot 3 - previous value `g`\n\tr.set_up_direction(b)\n\tr.set_floor_stop_on_slope_enabled(c)\n\tr.set_max_slides(d)\n\tr.set_floor_max_angle(e)\n\t# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `f`\n\tr.move_and_slide()\n\tvar aa = roman(r.velocity) # Roman", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("list_dir_begin( a , b )", "list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("list_dir_begin( a )", "list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("list_dir_begin( )", "list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("sort_custom( a , b )", "sort_custom(Callable(a,b))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("func c(var a, var b)", "func c(a, b)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("draw_line(1, 2, 3, 4, 5)", "draw_line(1,2,3,4)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("\timage.lock()", "\tfalse # image.lock() # TODOConverter40, image no longer require locking, `false` helps to not broke one line if/else, so can be freely removed", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\timage.unlock()", "\tfalse # image.unlock() # TODOConverter40, image no longer require locking, `false` helps to not broke one line if/else, so can be freely removed", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\troman.image.unlock()", "\tfalse # roman.image.unlock() # TODOConverter40, image no longer require locking, `false` helps to not broke one line if/else, so can be freely removed", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\tmtx.lock()", "\tmtx.lock()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\tmutex.unlock()", "\tmutex.unlock()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional("\nonready", "\n@onready", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("onready", "@onready", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional(" onready", " onready", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\nexport", "\n@export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\texport", "\t@export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\texport_dialog", "\texport_dialog", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("export", "@export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional(" export", " export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("tool", "@tool", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n    tool", "\n    tool", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\ntool", "\n\n@tool", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\nremote func", "\n\n@rpc(any_peer) func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\nremotesync func", "\n\n@rpc(any_peer, call_local) func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\nsync func", "\n\n@rpc(any_peer, call_local) func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\nslave func", "\n\n@rpc func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\npuppet func", "\n\n@rpc func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\npuppetsync func", "\n\n@rpc(call_local) func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\nmaster func", "\n\nThe master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using get_multiplayer().get_remote_sender_id()\n@rpc func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");
	valid = valid & test_conversion_single_additional("\n\nmastersync func", "\n\nThe master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using get_multiplayer().get_remote_sender_id()\n@rpc(call_local) func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword");

	valid = valid & test_conversion_single_additional_builtin("var size : Vector2 = Vector2() setget set_function , get_function", "var size : Vector2 = Vector2() :\n	get:\n		return size # TODOConverter40 Copy here content of get_function\n	set(mod_value):\n		mod_value  # TODOConverter40 Copy here content of set_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("var size : Vector2 = Vector2() setget set_function , ", "var size : Vector2 = Vector2() :\n	get:\n		return size # TODOConverter40 Non existent get function \n	set(mod_value):\n		mod_value  # TODOConverter40 Copy here content of set_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("var size : Vector2 = Vector2() setget set_function", "var size : Vector2 = Vector2() :\n	get:\n		return size # TODOConverter40 Non existent get function \n	set(mod_value):\n		mod_value  # TODOConverter40 Copy here content of set_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("var size : Vector2 = Vector2() setget  , get_function", "var size : Vector2 = Vector2() :\n	get:\n		return size # TODOConverter40 Copy here content of get_function \n	set(mod_value):\n		mod_value  # TODOConverter40  Non existent set function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("get_node(@", "get_node(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("yield(this, \"timeout\")", "await this.timeout", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("yield(this, \\\"timeout\\\")", "await this.timeout", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, true);

	valid = valid & test_conversion_single_additional_builtin(" Transform.xform(Vector3(a,b,c)) ", " Transform * Vector3(a,b,c) ", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin(" Transform.xform_inv(Vector3(a,b,c)) ", " Vector3(a,b,c) * Transform ", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("export(float) var lifetime = 3.0", "export var lifetime: float = 3.0", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("export(String, 'AnonymousPro', 'CourierPrime') var _font_name = 'AnonymousPro'", "export var _font_name = 'AnonymousPro' # (String, 'AnonymousPro', 'CourierPrime')", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false); // TODO, this is only a workaround
	valid = valid & test_conversion_single_additional_builtin("export(PackedScene) var mob_scene", "export var mob_scene: PackedScene", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("var d = parse_json(roman(sfs))", "var test_json_conv = JSON.new()\ntest_json_conv.parse(roman(sfs))\nvar d = test_json_conv.get_data()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("to_json( AA ) szon", "JSON.new().stringify( AA ) szon", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("s to_json", "s JSON.new().stringify", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("AF to_json2", "AF to_json2", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("var rr = JSON.parse(a)", "var test_json_conv = JSON.new()\ntest_json_conv.parse(a)\nvar rr = test_json_conv.get_data()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("empty()", "is_empty()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin(".empty", ".empty", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin(").roman(", ").roman(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\t.roman(", "\tsuper.roman(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin(" .roman(", " super.roman(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin(".1", ".1", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin(" .1", " .1", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("'.'", "'.'", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("'.a'", "'.a'", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\t._input(_event)", "\tsuper._input(_event)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("(connect(A,B,C) != OK):", "(connect(A,Callable(B,C)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(connect(A,B,C,D) != OK):", "(connect(A,Callable(B,C).bind(D)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(connect(A,B,C,[D]) != OK):", "(connect(A,Callable(B,C).bind(D)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(connect(A,B,C,D,E) != OK):", "(connect(A,Callable(B,C).bind(D),E) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("(start(A,B) != OK):", "(start(Callable(A,B)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("func start(A,B):", "func start(A,B):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(start(A,B,C,D,E,F,G) != OK):", "(start(Callable(A,B).bind(C),D,E,F,G) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("disconnect(A,B,C) != OK):", "disconnect(A,Callable(B,C)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("is_connected(A,B,C) != OK):", "is_connected(A,Callable(B,C)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("is_connected(A,B,C))", "is_connected(A,Callable(B,C)))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("(tween_method(A,B,C,D,E).foo())", "(tween_method(Callable(A,B),C,D,E).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(tween_method(A,B,C,D,E,[F,G]).foo())", "(tween_method(Callable(A,B).bind(F,G),C,D,E).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(tween_callback(A,B).foo())", "(tween_callback(Callable(A,B)).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("(tween_callback(A,B,[C,D]).foo())", "(tween_callback(Callable(A,B).bind(C,D)).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("func _init(p_x:int)->void:", "func _init(p_x:int):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("q_PackedDataContainer._iter_init(variable1)", "q_PackedDataContainer._iter_init(variable1)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("assert(speed < 20, str(randi()%10))", "assert(speed < 20) #,str(randi()%10))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("assert(speed < 2)", "assert(speed < 2)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("assert(false, \"Missing type --\" + str(argument.type) + \"--, needs to be added to project\")", "assert(false) #,\"Missing type --\" + str(argument.type) + \"--, needs to be added to project\")", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("create_from_image(aa, bb)", "create_from_image(aa) #,bb", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("q_ImageTexture.create_from_image(variable1, variable2)", "q_ImageTexture.create_from_image(variable1) #,variable2", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("set_cell_item(a, b, c, d ,e) # AA", "set_cell_item( Vector3(a,b,c) ,d,e) # AA", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("set_cell_item(a, b)", "set_cell_item(a, b)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("get_cell_item_orientation(a, b,c)", "get_cell_item_orientation(Vector3i(a,b,c))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("get_cell_item(a, b,c)", "get_cell_item(Vector3i(a,b,c))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("map_to_world(a, b,c)", "map_to_world(Vector3i(a,b,c))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("PackedStringArray(req_godot).join('.')", "'.'.join(PackedStringArray(req_godot))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("=PackedStringArray(req_godot).join('.')", "='.'.join(PackedStringArray(req_godot))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("    aa", "    aa", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\taa", "\taa", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("\t    aa", "\taa", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("    \taa", "    \taa", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional_builtin("apply_force(position, impulse)", "apply_force(impulse, position)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid & test_conversion_single_additional_builtin("apply_impulse(position, impulse)", "apply_impulse(impulse, position)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid & test_conversion_single_additional("AAA Color.white AF", "AAA Color.WHITE AF", &ProjectConverter3To4::rename_enums, "custom rename");

	// Custom rule conversion
	{
		String from = "instance";
		String to = "instantiate";
		String name = "AA.instance()";
		String got = "AA.instance()";
		String expected = "AA.instantiate()";
		custom_rename(got, from, to);
		if (got != expected) {
			ERR_PRINT("Failed to convert custom rename `" + name + "` to `" + expected + "`, got instead `" + got + "`");
		}
		valid = valid & (got == expected);
	}

	// get_object_of_execution
	{
		{ String base = "var roman = kieliszek.";
	String expected = "kieliszek.";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from get_object_of_execution `" + base + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
{
	String base = "r.";
	String expected = "r.";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from get_object_of_execution `" + base + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
{
	String base = "mortadela(";
	String expected = "";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from get_object_of_execution `" + base + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
}
// get_starting_space
{
	String base = "\t\t\t var roman = kieliszek.";
	String expected = "\t\t\t";
	String got = get_starting_space(base);
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from get_starting_space `" + base + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
// Parse Arguments
{
	String line = "( )";
	Vector<String> got_vector = parse_arguments(line);
	String got = "";
	String expected = "";
	for (String &part : got_vector) {
		got += part + "|||";
	}
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from parse_arguments `" + line + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
{
	String line = "(a , b , c)";
	Vector<String> got_vector = parse_arguments(line);
	String got = "";
	String expected = "a|||b|||c|||";
	for (String &part : got_vector) {
		got += part + "|||";
	}
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from parse_arguments `" + line + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
{
	String line = "(a , \"b,\" , c)";
	Vector<String> got_vector = parse_arguments(line);
	String got = "";
	String expected = "a|||\"b,\"|||c|||";
	for (String &part : got_vector) {
		got += part + "|||";
	}
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from parse_arguments `" + line + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}
{
	String line = "(a , \"(,),,,,\" , c)";
	Vector<String> got_vector = parse_arguments(line);
	String got = "";
	String expected = "a|||\"(,),,,,\"|||c|||";
	for (String &part : got_vector) {
		got += part + "|||";
	}
	if (got != expected) {
		ERR_PRINT("Failed to get proper data from parse_arguments `" + line + "` should return `" + expected + "`(" + itos(expected.size()) + "), got instead `" + got + "`(" + itos(got.size()) + ")");
	}
	valid = valid & (got == expected);
}

return valid;
}

// Validate in all arrays if names don't do cyclic renames `Node` -> `Node2D` | `Node2D` -> `2DNode`
bool ProjectConverter3To4::test_array_names() {
	bool valid = true;
	Vector<String> names = Vector<String>();

	// Validate if all classes are valid
	{
		int current_index = 0;
		while (class_renames[current_index][0]) {
			const String old_class = class_renames[current_index][0];
			const String new_class = class_renames[current_index][1];

			// Light2D, Texture, Viewport are special classes(probably virtual ones)
			if (ClassDB::class_exists(StringName(old_class)) && old_class != "Light2D" && old_class != "Texture" && old_class != "Viewport") {
				ERR_PRINT(String("Class `") + old_class + "` exists in Godot 4.0, so cannot be renamed to something else.");
				valid = false; // This probably should be only a warning, but not 100% sure - this would need to be added to CI
			}

			// Callable is special class, to which normal classes may be renamed
			if (!ClassDB::class_exists(StringName(new_class)) && new_class != "Callable") {
				ERR_PRINT(String("Class `") + new_class + "` doesn't exists in Godot 4.0, so cannot be used in conversion.");
				valid = false; // This probably should be only a warning, but not 100% sure - this would need to be added to CI
			}
			current_index++;
		}
	}

	//	// TODO To be able to fully work, it needs https://github.com/godotengine/godot/pull/49053
	//	// TODO this needs to be changed to hashset when available https://github.com/godotengine/godot-proposals/issues/867, to speedup searchng
	//	{
	//		OrderedHashMap<String, bool> all_functions;

	//		List<StringName> classes_list;
	//		ClassDB::get_class_list(&classes_list);
	//		for (StringName &name_of_class : classes_list) {
	//			List<MethodInfo> method_list;
	//			ClassDB::get_method_list(name_of_class, &method_list, true);
	//			for (MethodInfo &function_data : method_list) {
	//				if (!all_functions.has(function_data.name)) {
	//					all_functions.insert(function_data.name, false);
	//				}
	//			}
	//		}

	//		for (int type = Variant::Type::NIL + 1; type < Variant::Type::VARIANT_MAX; type++) {
	//			List<MethodInfo> method_list;
	//			Variant::get_method_list_by_type(&method_list, Variant::Type(type));
	//			for (MethodInfo &function_data : method_list) {
	//				if (!all_functions.has(function_data.name)) {
	//					all_functions.insert(function_data.name, false);
	//				}
	//			}
	//		}

	//		int current_element = 0;
	//		while (gdscript_function_renames[current_element][0] != nullptr) {
	//			if (!all_functions.has(gdscript_function_renames[current_element][1])) {
	//				ERR_PRINT(String("Missing gdscript function in pair (") + gdscript_function_renames[current_element][0] + " - ===> " + gdscript_function_renames[current_element][1] + " <===)");
	//				valid = false;
	//			}
	//			//			// DEBUG, disable below after tests
	//			//			if (all_functions.has(gdscript_function_renames[current_element][0])) {
	//			//				String used_in_classes = "";
	//			//
	//			//				for (StringName &name_of_class : classes_list) {
	//			//					List<MethodInfo> method_list;
	//			//					ClassDB::get_method_list(name_of_class, &method_list, true);
	//			//					for (MethodInfo &function_data : method_list) {
	//			//						if (function_data.name == gdscript_function_renames[current_element][0]) {
	//			//							used_in_classes += String(name_of_class) + ", ";
	//			//						}
	//			//					}
	//			//				}
	//			//				for (int type = Variant::Type::NIL + 1; type < Variant::Type::VARIANT_MAX; type++) {
	//			//					List<MethodInfo> method_list;
	//			//					Variant::get_method_list_by_type(&method_list, Variant::Type(type));
	//			//					for (MethodInfo &function_data : method_list) {
	//			//						if (function_data.name == gdscript_function_renames[current_element][0]) {
	//			//							used_in_classes += Variant::get_type_name(Variant::Type(type)) + ", ";
	//			//						}
	//			//					}
	//			//				}
	//			//				used_in_classes = used_in_classes.trim_suffix(", ");
	//			//
	//			//				WARN_PRINT(String("Godot contains function which will be renamed in pair ( ===> ") + gdscript_function_renames[current_element][0] + " <=== - " + gdscript_function_renames[current_element][1] + ") in class " + used_in_classes + " - check for possible invalid rule.");
	//			//			}
	//			current_element++;
	//		}
	//	}

	valid = valid & test_single_array(enum_renames);
	valid = valid & test_single_array(class_renames, true);
	valid = valid & test_single_array(gdscript_function_renames, true);
	valid = valid & test_single_array(csharp_function_renames, true);
	valid = valid & test_single_array(gdscript_properties_renames);
	valid = valid & test_single_array(csharp_properties_renames);
	valid = valid & test_single_array(shaders_renames);
	valid = valid & test_single_array(gdscript_signals_renames);
	valid = valid & test_single_array(project_settings_renames);
	valid = valid & test_single_array(builtin_types_renames);
	valid = valid & test_single_array(colors_renames);

	return valid;
}

// Validate in one array if names don't do cyclic renames `Node` -> `Node2D` | `Node2D` -> `2DNode`
// Also checks if in name contains spaces at the end or beginning
bool ProjectConverter3To4::test_single_array(const char *array[][2], bool ignore_second_check) {
	bool valid = true;
	int current_index = 0;
	Vector<String> names = Vector<String>();

	while (array[current_index][0]) {
		if (String(array[current_index][0]).begins_with(" ") || String(array[current_index][0]).ends_with(" ")) {
			{
				ERR_PRINT(String("Entry \"") + array[current_index][0] + "\" ends or stars with space.");
				valid = false;
			}
		}
		if (names.has(array[current_index][0])) {
			ERR_PRINT(String("Found duplicated things, pair ( -> ") + array[current_index][0] + " , " + array[current_index][1] + ")");
			valid = false;
		}
		names.append(array[current_index][0]);

		if (String(array[current_index][1]).begins_with(" ") || String(array[current_index][1]).ends_with(" ")) {
			{
				ERR_PRINT(String("Entry \"") + array[current_index][1] + "\" ends or stars with space.");
				valid = false;
			}
		}
		if (names.has(array[current_index][1])) {
			ERR_PRINT(String("Found duplicated things, pair (") + array[current_index][0] + " , ->" + array[current_index][1] + ")");
			valid = false;
		}
		if (!ignore_second_check) {
			names.append(array[current_index][1]);
		}
		current_index++;
	}
	return valid;
};

// Returns arguments from given function execution, this cannot be really done as regex
// `abc(d,e(f,g),h)` -> [d], [e(f,g)], [h]
Vector<String> ProjectConverter3To4::parse_arguments(const String &line) {
	Vector<String> parts;
	int string_size = line.length();
	int current_index = 0;
	int start_part = 0; // Index of beginning of start par
	int parts_counter = 0;
	char32_t previous_character = '\0';
	bool is_inside_string = false; // if true, it ignore this 3 characters ( , ) inside string

	if (line.count("(") != line.count(")")) {
		ERR_PRINT("Bug: substring should have equal number of open and close parenthess - `" + line + "`");
		return parts;
	}

	while (current_index < string_size) {
		char32_t character = line.get(current_index);
		switch (character) {
			case '(': {
				parts_counter++;
				if (parts_counter == 1 && !is_inside_string) {
					start_part = current_index;
				}
				break;
			};
			case ')': {
				parts_counter--;
				if (parts_counter == 0 && !is_inside_string) {
					parts.append(line.substr(start_part + 1, current_index - start_part - 1));
					start_part = current_index;
				}
				break;
			};
			case ',': {
				if (parts_counter == 1 && !is_inside_string) {
					parts.append(line.substr(start_part + 1, current_index - start_part - 1));
					start_part = current_index;
				}
				break;
			};
			case '"': {
				if (previous_character != '\\')
					is_inside_string = !is_inside_string;
			}
		}
		current_index++;
		previous_character = character;
	}

	Vector<String> clean_parts;
	for (String &part : parts) {
		part = part.strip_edges();
		if (!part.is_empty()) {
			clean_parts.append(part);
		}
	}

	return clean_parts;
}

// Finds latest parenthess owned by function
// `function(abc(a,b),DD)):` finds this parenthess `function(abc(a,b),DD => ) <= ):`
int ProjectConverter3To4::get_end_parenthess(const String &line) const {
	int current_index = 0;
	int current_state = 0;
	while (line.length() > current_index) {
		char32_t character = line.get(current_index);
		if (character == '(') {
			current_state++;
		}
		if (character == ')') {
			current_state--;
			if (current_state == 0) {
				return current_index;
			}
		}
		current_index++;
	}
	return -1;
}

// Connects arguments from vector to one string
// Needed when after processing e.g. 2 arguments, later arguments are not changed in any way
String ProjectConverter3To4::connect_arguments(const Vector<String> &arguments, int from, int to) const {
	if (to == -1) {
		to = arguments.size();
	}

	String value;
	if (arguments.size() > 0 && from != 0 && from < to) {
		value = ",";
	}

	for (int i = from; i < to; i++) {
		value += arguments[i];
		if (i != to - 1) {
			value += ',';
		}
	}
	return value;
}

// Return spaces or tabs which starts line e.g. `\t\tmove_this` will return `\t\t`
String ProjectConverter3To4::get_starting_space(const String &line) const {
	String empty_space;
	int current_character = 0;

	if (line.is_empty()) {
		return empty_space;
	}

	if (line[0] == ' ') {
		while (current_character < line.size()) {
			if (line[current_character] == ' ') {
				empty_space += ' ';
				current_character++;
			} else {
				break;
			}
		}
	}
	if (line[0] == '\t') {
		while (current_character < line.size()) {
			if (line[current_character] == '\t') {
				empty_space += '\t';
				current_character++;
			} else {
				break;
			}
		}
	}
	return empty_space;
}

// Return object which execute specific function
// e.g. in `var roman = kieliszek.funkcja()` to this function is passed everything before function which we want to check
// so it is `var roman = kieliszek.` and this function return `kieliszek.`
String ProjectConverter3To4::get_object_of_execution(const String &line) const {
	int end = line.size() - 1; // Last one is \0
	int start = end - 1;

	while (start >= 0) {
		char32_t character = line[start];
		if ((character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') || character == '.' || character == '_') {
			if (start == 0) {
				break;
			}
			start--;
			continue;
		} else {
			start++; // Found invalid character, needs to be ignored
			break;
		}
	}
	return line.substr(start, (end - start));
}

void ProjectConverter3To4::rename_enums(String &file_content) {
	int current_index = 0;

	// Rename colors
	if (file_content.find("Color.") != -1) {
		while (colors_renames[current_index][0]) {
			RegEx reg = RegEx(String("\\bColor.") + colors_renames[current_index][0] + "\\b");
			CRASH_COND(!reg.is_valid());
			file_content = reg.sub(file_content, String("Color.") + colors_renames[current_index][1], true);
			current_index++;
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_rename_enums(Vector<String> &file_content) {
	int current_index = 0;

	Vector<String> found_things;

	// Rename colors
	if (file_content.find("Color.") != -1) {
		while (colors_renames[current_index][0]) {
			RegEx reg = RegEx(String("\\bColor.") + colors_renames[current_index][0] + "\\b");
			CRASH_COND(!reg.is_valid());

			int current_line = 1;
			for (String &line : file_content) {
				TypedArray<RegExMatch> reg_match = reg.search_all(line);
				if (reg_match.size() > 0) {
					found_things.append(line_formatter(current_line, colors_renames[current_index][0], colors_renames[current_index][1], line));
				}
				current_line++;
			}
			current_index++;
		}
	}

	return found_things;
}

void ProjectConverter3To4::rename_classes(String &file_content) {
	int current_index = 0;

	// TODO Maybe it is better way to not rename gd, tscn and other files which are named as classes
	while (class_renames[current_index][0]) {
		// Begin renaming workaround `Resource.gd` -> `RefCounter.gd`
		RegEx reg_before = RegEx(String("\\b") + class_renames[current_index][0] + ".tscn\\b");
		CRASH_COND(!reg_before.is_valid());
		file_content = reg_before.sub(file_content, "TEMP_RENAMED_CLASS.tscn", true);
		RegEx reg_before2 = RegEx(String("\\b") + class_renames[current_index][0] + ".gd\\b");
		CRASH_COND(!reg_before2.is_valid());
		file_content = reg_before2.sub(file_content, "TEMP_RENAMED_CLASS.gd", true);
		RegEx reg_before3 = RegEx(String("\\b") + class_renames[current_index][0] + ".shader\\b");
		CRASH_COND(!reg_before3.is_valid());
		file_content = reg_before3.sub(file_content, "TEMP_RENAMED_CLASS.gd", true);
		// End

		RegEx reg = RegEx(String("\\b") + class_renames[current_index][0] + "\\b");
		CRASH_COND(!reg.is_valid());
		file_content = reg.sub(file_content, class_renames[current_index][1], true);

		// Begin renaming workaround `Resource.gd` -> `RefCounter.gd`
		RegEx reg_after = RegEx("\\bTEMP_RENAMED_CLASS.tscn\\b");
		CRASH_COND(!reg_after.is_valid());
		file_content = reg_after.sub(file_content, String(class_renames[current_index][0]) + ".tscn", true);
		RegEx reg_after2 = RegEx("\\bTEMP_RENAMED_CLASS.gd\\b");
		CRASH_COND(!reg_after2.is_valid());
		file_content = reg_after2.sub(file_content, String(class_renames[current_index][0]) + ".gd", true);
		RegEx reg_after3 = RegEx("\\bTEMP_RENAMED_CLASS.gd\\b");
		CRASH_COND(!reg_after3.is_valid());
		file_content = reg_after3.sub(file_content, String(class_renames[current_index][0]) + ".shader", true);
		// End

		current_index++;
	}

	// OS.get_ticks_msec -> Time.get_ticks_msec
	RegEx reg_time1 = RegEx("OS.get_ticks_msec");
	CRASH_COND(!reg_time1.is_valid());
	file_content = reg_time1.sub(file_content, "Time.get_ticks_msec", true);
	RegEx reg_time2 = RegEx("OS.get_ticks_usec");
	CRASH_COND(!reg_time2.is_valid());
	file_content = reg_time2.sub(file_content, "Time.get_ticks_usec", true);
};

Vector<String> ProjectConverter3To4::check_for_rename_classes(Vector<String> &file_content) {
	int current_index = 0;

	Vector<String> found_things;

	while (class_renames[current_index][0]) {
		RegEx reg_before = RegEx(String("\\b") + class_renames[current_index][0] + ".tscn\\b");
		CRASH_COND(!reg_before.is_valid());
		RegEx reg_before2 = RegEx(String("\\b") + class_renames[current_index][0] + ".gd\\b");
		CRASH_COND(!reg_before2.is_valid());

		RegEx reg = RegEx(String("\\b") + class_renames[current_index][0] + "\\b");
		CRASH_COND(!reg.is_valid());

		int current_line = 1;
		for (String &line : file_content) {
			line = reg_before.sub(line, "TEMP_RENAMED_CLASS.tscn", true);
			line = reg_before2.sub(line, "TEMP_RENAMED_CLASS.gd", true);

			TypedArray<RegExMatch> reg_match = reg.search_all(line);
			if (reg_match.size() > 0) {
				found_things.append(line_formatter(current_line, class_renames[current_index][0], class_renames[current_index][1], line));
			}
			current_line++;
		}
		current_index++;
	}

	// TODO OS -> TIME
	int current_line = 1;
	RegEx reg_time1 = RegEx("OS.get_ticks_msec");
	CRASH_COND(!reg_time1.is_valid());
	RegEx reg_time2 = RegEx("OS.get_ticks_usec");
	CRASH_COND(!reg_time2.is_valid());
	for (String &line : file_content) {
		String old = line;

		line = reg_time1.sub(line, "Time.get_ticks_msec", true);
		line = reg_time2.sub(line, "Time.get_ticks_usec", true);

		if (old != line) {
			found_things.append(simple_line_formatter(current_line, old, line));
		}
		current_line++;
	}
	return found_things;
}

void ProjectConverter3To4::rename_gdscript_functions(String &file_content, const RegExContainer &reg_container, bool builtin) {
	Vector<String> lines = file_content.split("\n");

	for (String &line : lines) {
		process_gdscript_line(line, reg_container, builtin);
	}

	// Collect vector to string
	file_content = "";
	for (int i = 0; i < lines.size(); i++) {
		file_content += lines[i];

		if (i != lines.size() - 1) {
			file_content += "\n";
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_rename_gdscript_functions(Vector<String> &file_content, const RegExContainer &reg_container, bool builtin) {
	int current_line = 1;

	Vector<String> found_things;

	for (String &line : file_content) {
		String old_line = line;
		process_gdscript_line(line, reg_container, builtin);
		if (old_line != line) {
			found_things.append(simple_line_formatter(current_line, old_line, line));
		}
	}

	return found_things;
}
void ProjectConverter3To4::process_gdscript_line(String &line, const RegExContainer &reg_container, bool builtin) {
	if (line.find("mtx") == -1 && line.find("mutex") == -1 && line.find("Mutex") == -1) {
		line = reg_container.reg_image_lock.sub(line, "false # $1.lock() # TODOConverter40, image no longer require locking, `false` helps to not broke one line if/else, so can be freely removed", true);
		line = reg_container.reg_image_unlock.sub(line, "false # $1.unlock() # TODOConverter40, image no longer require locking, `false` helps to not broke one line if/else, so can be freely removed", true);
	}

	// Mixed use of spaces and tabs - tabs as first - TODO, this probably is problem problem, but not sure
	line = reg_container.reg_mixed_tab_space.sub(line, "$1", true);

	// PackedStringArray(req_godot).join('.') -> '.'.join(PackedStringArray(req_godot))       PoolStringArray
	line = reg_container.reg_join.sub(line, "$2.join($1)", true);

	// -- empty() -> is_empty()       Pool*Array
	line = reg_container.reg_is_empty.sub(line, "is_empty(", true);

	// -- \t.func() -> \tsuper.func()       Object
	line = reg_container.reg_super.sub(line, "$1super.$2", true); // TODO, not sure if possible, but for now this brake String text e.g. "Choosen .gitignore" -> "Choosen super.gitignore"

	// -- JSON.parse(a) -> JSON.new().parse(a) etc.    JSON
	line = reg_container.reg_json_non_new.sub(line, "$1var test_json_conv = JSON.new()\n$1test_json_conv.parse($3\n$1$2test_json_conv.get_data()", true);

	// -- to_json(a) -> JSON.new().stringify(a)     Object
	line = reg_container.reg_json_to.sub(line, "JSON.new().stringify", true);

	// -- parse_json(a) -> JSON.get_data() etc.    Object
	line = reg_container.reg_json_parse.sub(line, "$1var test_json_conv = JSON.new()\n$1test_json_conv.parse($3\n$1$2test_json_conv.get_data()", true);

	// -- get_node(@ -> get_node(       Node
	line = line.replace("get_node(@", "get_node(");

	// export(float) var lifetime = 3.0 -> export var lifetime: float = 3.0     GDScript
	line = reg_container.reg_export.sub(line, "export var $2: $1");

	// export(String, 'AnonymousPro', 'CourierPrime') var _font_name = 'AnonymousPro' -> export var _font_name = 'AnonymousPro' #(String, 'AnonymousPro', 'CourierPrime')   GDScript
	line = reg_container.reg_export_advanced.sub(line, "export var $2$3 # ($1)");

	// Setget Setget
	line = reg_container.reg_setget_setget.sub(line, "var $1$2:\n\tget:\n\t\treturn $1 # TODOConverter40 Copy here content of $4\n\tset(mod_value):\n\t\tmod_value  # TODOConverter40 Copy here content of $3", true);

	// Setget set
	line = reg_container.reg_setget_set.sub(line, "var $1$2:\n\tget:\n\t\treturn $1 # TODOConverter40 Non existent get function \n\tset(mod_value):\n\t\tmod_value  # TODOConverter40 Copy here content of $3", true);

	// Setget get
	line = reg_container.reg_setget_get.sub(line, "var $1$2:\n\tget:\n\t\treturn $1 # TODOConverter40 Copy here content of $3 \n\tset(mod_value):\n\t\tmod_value  # TODOConverter40  Non existent set function", true);

	// OS.window_fullscreen = true -> ProjectSettings.set("display/window/size/fullscreen",true)
	if (builtin) {
		line = reg_container.reg_os_fullscreen.sub(line, "ProjectSettings.set(\\\"display/window/size/fullscreen\\\", $1)", true);
	} else {
		line = reg_container.reg_os_fullscreen.sub(line, "ProjectSettings.set(\"display/window/size/fullscreen\", $1)", true);
	}

	// -- r.move_and_slide( a, b, c, d, e )  ->  r.set_velocity(a) ... r.move_and_slide()         KinematicBody
	if (line.find("move_and_slide(") != -1) {
		int start = line.find("move_and_slide(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			String base_obj = get_object_of_execution(line.substr(0, start));
			String starting_space = get_starting_space(line);

			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() >= 1) {
				String line_new;

				// motion_velocity
				line_new += starting_space + base_obj + "set_velocity(" + parts[0] + ")\n";

				// up_direction
				if (parts.size() >= 2) {
					line_new += starting_space + base_obj + "set_up_direction(" + parts[1] + ")\n";
				}

				// stop_on_slope
				if (parts.size() >= 3) {
					line_new += starting_space + base_obj + "set_floor_stop_on_slope_enabled(" + parts[2] + ")\n";
				}

				// max_slides
				if (parts.size() >= 4) {
					line_new += starting_space + base_obj + "set_max_slides(" + parts[3] + ")\n";
				}

				// floor_max_angle
				if (parts.size() >= 5) {
					line_new += starting_space + base_obj + "set_floor_max_angle(" + parts[4] + ")\n";
				}

				// infiinite_interia
				if (parts.size() >= 6) {
					line_new += starting_space + "# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `" + parts[5] + "`\n";
				}

				line_new += starting_space + base_obj + "move_and_slide()\n";
				line = line_new + line.substr(0, start) + "velocity" + line.substr(end + start);
			}
		}
	}

	// -- r.move_and_slide_with_snap( a, b, c, d, e )  ->  r.set_velocity(a) ... r.move_and_slide()         KinematicBody
	if (line.find("move_and_slide_with_snap(") != -1) {
		int start = line.find("move_and_slide_with_snap(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			String base_obj = get_object_of_execution(line.substr(0, start));
			String starting_space = get_starting_space(line);

			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() >= 1) {
				String line_new;

				// motion_velocity
				line_new += starting_space + base_obj + "set_velocity(" + parts[0] + ")\n";

				// snap
				if (parts.size() >= 2) {
					line_new += starting_space + "# TODOConverter40 looks that snap in Godot 4.0 is float, not vector like in Godot 3 - previous value `" + parts[1] + "`\n";
				}

				// up_direction
				if (parts.size() >= 3) {
					line_new += starting_space + base_obj + "set_up_direction(" + parts[2] + ")\n";
				}

				// stop_on_slope
				if (parts.size() >= 4) {
					line_new += starting_space + base_obj + "set_floor_stop_on_slope_enabled(" + parts[3] + ")\n";
				}

				// max_slides
				if (parts.size() >= 5) {
					line_new += starting_space + base_obj + "set_max_slides(" + parts[4] + ")\n";
				}

				// floor_max_angle
				if (parts.size() >= 6) {
					line_new += starting_space + base_obj + "set_floor_max_angle(" + parts[5] + ")\n";
				}

				// infiinite_interia
				if (parts.size() >= 7) {
					line_new += starting_space + "# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `" + parts[6] + "`\n";
				}

				line_new += starting_space + base_obj + "move_and_slide()\n";
				line = line_new + line.substr(0, start) + "velocity" + line.substr(end + start); // move_and_slide used to return velocity
			}
		}
	}

	// -- sort_custom( a , b )  ->  sort_custom(Callable( a , b ))            Object
	if (line.find("sort_custom(") != -1) {
		int start = line.find("sort_custom(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "sort_custom(Callable(" + parts[0] + "," + parts[1] + "))" + line.substr(end + start);
			}
		}
	}

	// -- list_dir_begin( )  ->  list_dir_begin()            Object
	if (line.find("list_dir_begin(") != -1) {
		int start = line.find("list_dir_begin(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			line = line.substr(0, start) + "list_dir_begin() " + line.substr(end + start) + "# TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547";
		}
	}

	// -- draw_line(1,2,3,4,5) -> draw_line(1,2,3,4)            CanvasItem
	if (line.find("draw_line(") != -1) {
		int start = line.find("draw_line(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 5) {
				line = line.substr(0, start) + "draw_line(" + parts[0] + "," + parts[1] + "," + parts[2] + "," + parts[3] + ")" + line.substr(end + start);
			}
		}
	}

	// -- func c(var a, var b) -> func c(a, b)
	if (line.find("func ") != -1 && line.find("var ") != -1) {
		int start = line.find("func ");
		start = line.substr(start).find("(") + start;
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));

			String start_string = line.substr(0, start) + "(";
			for (int i = 0; i < parts.size(); i++) {
				start_string += parts[i].strip_edges().trim_prefix("var ");
				if (i != parts.size() - 1) {
					start_string += ", ";
				}
			}
			line = start_string + ")" + line.substr(end + start);
		}
	}

	// -- yield(this, \"timeout\") -> await this.timeout         GDScript
	if (line.find("yield(") != -1) {
		int start = line.find("yield(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				if (builtin) {
					line = line.substr(0, start) + "await " + parts[0] + "." + parts[1].replace("\\\"", "").replace("\\'", "").replace(" ", "") + line.substr(end + start);
				} else {
					line = line.substr(0, start) + "await " + parts[0] + "." + parts[1].replace("\"", "").replace("\'", "").replace(" ", "") + line.substr(end + start);
				}
			}
		}
	}

	// -- parse_json( AA ) -> TODO       Object
	if (line.find("parse_json(") != -1) {
		int start = line.find("parse_json(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			line = line.substr(0, start) + "JSON.new().stringify(" + connect_arguments(parts, 0) + ")" + line.substr(end + start);
		}
	}

	// -- .xform(Vector3(a,b,c)) -> * Vector3(a,b,c)            Transform
	if (line.find(".xform(") != -1) {
		int start = line.find(".xform(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 1) {
				line = line.substr(0, start) + " * " + parts[0] + line.substr(end + start);
			}
		}
	}

	// -- .xform_inv(Vector3(a,b,c)) -> * Vector3(a,b,c)       Transform
	if (line.find(".xform_inv(") != -1) {
		int start = line.find(".xform_inv(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			String object_exec = get_object_of_execution(line.substr(0, start));
			if (line.find(object_exec + ".xform") != -1) {
				int start2 = line.find(object_exec + ".xform");
				Vector<String> parts = parse_arguments(line.substr(start, end));
				if (parts.size() == 1) {
					line = line.substr(0, start2) + parts[0] + " * " + object_exec + line.substr(end + start);
				}
			}
		}
	}

	// -- "(connect(A,B,C,D,E) != OK):", "(connect(A,Callable(B,C).bind(D),E)      Object
	if (line.find("connect(") != -1) {
		int start = line.find("connect(");
		// Protection from disconnect
		if (start == 0 || line.get(start - 1) != 's') {
			int end = get_end_parenthess(line.substr(start)) + 1;
			if (end > -1) {
				Vector<String> parts = parse_arguments(line.substr(start, end));
				if (parts.size() == 3) {
					line = line.substr(0, start) + "connect(" + parts[0] + ",Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
				} else if (parts.size() >= 4) {
					line = line.substr(0, start) + "connect(" + parts[0] + ",Callable(" + parts[1] + "," + parts[2] + ").bind(" + parts[3].lstrip("[").rstrip("]") + ")" + connect_arguments(parts, 4) + ")" + line.substr(end + start);
				}
			}
		}
	}
	// -- disconnect(a,b,c) -> disconnect(a,Callable(b,c))      Object
	if (line.find("disconnect(") != -1) {
		int start = line.find("disconnect(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "disconnect(" + parts[0] + ",Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	// -- is_connected(a,b,c) -> is_connected(a,Callable(b,c))      Object
	if (line.find("is_connected(") != -1) {
		int start = line.find("is_connected(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "is_connected(" + parts[0] + ",Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	// -- "(tween_method(A,B,C,D,E) != OK):", "(tween_method(Callable(A,B),C,D,E)      Object
	// -- "(tween_method(A,B,C,D,E,[F,G]) != OK):", "(tween_method(Callable(A,B).bind(F,G),C,D,E)      Object
	if (line.find("tween_method(") != -1) {
		int start = line.find("tween_method(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 5) {
				line = line.substr(0, start) + "tween_method(Callable(" + parts[0] + "," + parts[1] + ")," + parts[2] + "," + parts[3] + "," + parts[4] + ")" + line.substr(end + start);
			} else if (parts.size() >= 6) {
				line = line.substr(0, start) + "tween_method(Callable(" + parts[0] + "," + parts[1] + ").bind(" + connect_arguments(parts, 5).substr(1).lstrip("[").rstrip("]") + ")," + parts[2] + "," + parts[3] + "," + parts[4] + ")" + line.substr(end + start);
			}
		}
	}
	// -- "(tween_callback(A,B,[C,D]) != OK):", "(connect(Callable(A,B).bind(C,D))      Object
	if (line.find("tween_callback(") != -1) {
		int start = line.find("tween_callback(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "tween_callback(Callable(" + parts[0] + "," + parts[1] + "))" + line.substr(end + start);
			} else if (parts.size() >= 3) {
				line = line.substr(0, start) + "tween_callback(Callable(" + parts[0] + "," + parts[1] + ").bind(" + connect_arguments(parts, 2).substr(1).lstrip("[").rstrip("]") + "))" + line.substr(end + start);
			}
		}
	}
	// -- start(a,b) -> start(Callable(a,b))      Thread
	// -- start(a,b,c,d) -> start(Callable(a,b).bind(c),d)      Thread
	if (line.find("start(") != -1) {
		int start = line.find("start(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		// Protection from 'func start'
		if (!line.begins_with("func ")) {
			if (end > -1) {
				Vector<String> parts = parse_arguments(line.substr(start, end));
				if (parts.size() == 2) {
					line = line.substr(0, start) + "start(Callable(" + parts[0] + "," + parts[1] + "))" + line.substr(end + start);
				} else if (parts.size() >= 3) {
					line = line.substr(0, start) + "start(Callable(" + parts[0] + "," + parts[1] + ").bind(" + parts[2] + ")" + connect_arguments(parts, 3) + ")" + line.substr(end + start);
				}
			}
		}
	}
	// -- func _init(p_x:int)->void:  -> func _init(p_x:int):    Object # https://github.com/godotengine/godot/issues/50589
	if (line.find(" _init(") != -1) {
		int start = line.find(" _init(");
		int end = line.rfind(":") + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			line = line.substr(0, start) + " _init(" + connect_arguments(parts, 0) + "):" + line.substr(end + start);
		}
	}
	//  assert(speed < 20, str(randi()%10))  ->  assert(speed < 20) #,str(randi()%10))    GDScript - GDScript bug constant message
	if (line.find("assert(") != -1) {
		int start = line.find("assert(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "assert(" + parts[0] + ") " + line.substr(end + start) + "#," + parts[1] + ")";
			}
		}
	}
	//  create_from_image(aa, bb)  ->   create_from_image(aa) #, bb   ImageTexture
	if (line.find("create_from_image(") != -1) {
		int start = line.find("create_from_image(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "create_from_image(" + parts[0] + ") " + "#," + parts[1] + line.substr(end + start);
			}
		}
	}
	//  set_cell_item(a, b, c, d ,e)  ->   set_cell_item(Vector3(a, b, c), d ,e)
	if (line.find("set_cell_item(") != -1) {
		int start = line.find("set_cell_item(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() > 2) {
				line = line.substr(0, start) + "set_cell_item( Vector3(" + parts[0] + "," + parts[1] + "," + parts[2] + ") " + connect_arguments(parts, 3) + ")" + line.substr(end + start);
			}
		}
	}
	//  get_cell_item(a, b, c)  ->   get_cell_item(Vector3i(a, b, c))
	if (line.find("get_cell_item(") != -1) {
		int start = line.find("get_cell_item(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "get_cell_item(Vector3i(" + parts[0] + "," + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	//  get_cell_item_orientation(a, b, c)  ->   get_cell_item_orientation(Vector3i(a, b, c))
	if (line.find("get_cell_item_orientation(") != -1) {
		int start = line.find("get_cell_item_orientation(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "get_cell_item_orientation(Vector3i(" + parts[0] + "," + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	//  apply_impulse(A, B)  ->   apply_impulse(B, A)
	if (line.find("apply_impulse(") != -1) {
		int start = line.find("apply_impulse(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "apply_impulse(" + parts[1] + ", " + parts[0] + ")" + line.substr(end + start);
			}
		}
	}
	//  apply_force(A, B)  ->   apply_force(B, A)
	if (line.find("apply_force(") != -1) {
		int start = line.find("apply_force(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "apply_force(" + parts[1] + ", " + parts[0] + ")" + line.substr(end + start);
			}
		}
	}
	//  map_to_world(a, b, c)  ->   map_to_world(Vector3i(a, b, c))
	if (line.find("map_to_world(") != -1) {
		int start = line.find("map_to_world(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "map_to_world(Vector3i(" + parts[0] + "," + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	//  OS.get_window_safe_area()  ->   DisplayServer.get_display_safe_area()
	if (line.find("OS.get_window_safe_area(") != -1) {
		int start = line.find("OS.get_window_safe_area(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 0) {
				line = line.substr(0, start) + "DisplayServer.get_display_safe_area()" + line.substr(end + start);
			}
		}
	}
}

void ProjectConverter3To4::process_csharp_line(String &line) {
	// TODO maybe this can be changed to normal rule
	line = line.replace("OS.GetWindowSafeArea()", "DisplayServer.ScreenGetUsableRect()");

	// -- Connect(,,,things) -> Connect(,Callable(,),things)      Object
	if (line.find("Connect(") != -1) {
		int start = line.find("Connect(");
		// Protection from disconnect
		if (start == 0 || line.get(start - 1) != 's') {
			int end = get_end_parenthess(line.substr(start)) + 1;
			if (end > -1) {
				Vector<String> parts = parse_arguments(line.substr(start, end));
				if (parts.size() >= 3) {
					line = line.substr(0, start) + "Connect(" + parts[0] + ",new Callable(" + parts[1] + "," + parts[2] + ")" + connect_arguments(parts, 3) + ")" + line.substr(end + start);
				}
			}
		}
	}
	// -- Disconnect(a,b,c) -> Disconnect(a,Callable(b,c))      Object
	if (line.find("Disconnect(") != -1) {
		int start = line.find("Disconnect(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "Disconnect(" + parts[0] + ",new Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	// -- IsConnected(a,b,c) -> IsConnected(a,Callable(b,c))      Object
	if (line.find("IsConnected(") != -1) {
		int start = line.find("IsConnected(");
		int end = get_end_parenthess(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "IsConnected(" + parts[0] + ",new Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
}

void ProjectConverter3To4::rename_csharp_functions(String &file_content) {
	Vector<String> lines = file_content.split("\n");

	for (String &line : lines) {
		process_csharp_line(line);
	}

	// Collect vector to string
	file_content = "";
	for (int i = 0; i < lines.size(); i++) {
		file_content += lines[i];

		if (i != lines.size() - 1) {
			file_content += "\n";
		}
	}
};

// This is almost 1:1 copy of function which rename gdscript functions
Vector<String> ProjectConverter3To4::check_for_rename_csharp_functions(Vector<String> &file_content) {
	int current_line = 1;

	Vector<String> found_things;

	for (String &line : file_content) {
		String old_line = line;
		process_csharp_line(line);
		if (old_line != line) {
			found_things.append(simple_line_formatter(current_line, old_line, line));
		}
	}

	return found_things;
}

void ProjectConverter3To4::rename_csharp_attributes(String &file_content) {
	// -- [Remote] -> [RPC(MultiplayerAPI.RPCMode.AnyPeer)]
	{
		RegEx reg_remote = RegEx("\\[Remote(Attribute)?(\\(\\))?\\]");
		CRASH_COND(!reg_remote.is_valid());
		file_content = reg_remote.sub(file_content, "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", true);
	}
	// -- [RemoteSync] -> [RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]
	{
		RegEx reg_remotesync = RegEx("\\[(Remote)?Sync(Attribute)?(\\(\\))?\\]");
		CRASH_COND(!reg_remotesync.is_valid());
		file_content = reg_remotesync.sub(file_content, "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", true);
	}
	// -- [Puppet] -> [RPC]
	{
		RegEx reg_puppet = RegEx("\\[(Puppet|Slave)(Attribute)?(\\(\\))?\\]");
		CRASH_COND(!reg_puppet.is_valid());
		file_content = reg_puppet.sub(file_content, "[RPC]", true);
	}
	// -- [PuppetSync] -> [RPC(CallLocal = true)]
	{
		RegEx reg_puppetsync = RegEx("\\[PuppetSync(Attribute)?(\\(\\))?\\]");
		CRASH_COND(!reg_puppetsync.is_valid());
		file_content = reg_puppetsync.sub(file_content, "[RPC(CallLocal = true)]", true);
	}
	String error_message = "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using Multiplayer.GetRemoteSenderId()\n";
	// -- [Master] -> [RPC]
	{
		RegEx reg_remote = RegEx("\\[Master(Attribute)?(\\(\\))?\\]");
		CRASH_COND(!reg_remote.is_valid());
		file_content = reg_remote.sub(file_content, error_message + "[RPC]", true);
	}
	// -- [MasterSync] -> [RPC(CallLocal = true)]
	{
		RegEx reg_remote = RegEx("\\[MasterSync(Attribute)?(\\(\\))?\\]");
		CRASH_COND(!reg_remote.is_valid());
		file_content = reg_remote.sub(file_content, error_message + "[RPC(CallLocal = true)]", true);
	}
}

Vector<String> ProjectConverter3To4::check_for_rename_csharp_attributes(Vector<String> &file_content) {
	int current_line = 1;

	Vector<String> found_things;

	for (String &line : file_content) {
		String old;
		old = line;
		{
			RegEx regex = RegEx("\\[Remote(Attribute)?(\\(\\))?\\]");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "[Remote]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", line));
		}
		old = line;
		{
			RegEx regex = RegEx("\\[(Remote)?Sync(Attribute)?(\\(\\))?\\]");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "[RemoteSync]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", line));
		}
		old = line;
		{
			RegEx regex = RegEx("\\[Puppet(Attribute)?(\\(\\))?\\]");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "[RPC]", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "[Puppet]", "[RPC]", line));
		}
		old = line;
		{
			RegEx regex = RegEx("\\[(Puppet|Slave)Sync(Attribute)?(\\(\\))?\\]");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "[RPC(CallLocal = true)]", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "[PuppetSync]", "[RPC(CallLocal = true)]", line));
		}
		old = line;
		{
			RegEx regex = RegEx("\\[Master(Attribute)?(\\(\\))?\\]");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "[RPC]", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "[Master]", "[RPC]", line));
		}
		old = line;
		{
			RegEx regex = RegEx("\\[MasterSync(Attribute)?(\\(\\))?\\]");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "[RPC(CallLocal = true)]", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "[MasterSync]", "[RPC(CallLocal = true)]", line));
		}

		current_line++;
	}

	return found_things;
}

void ProjectConverter3To4::rename_gdscript_keywords(String &file_content) {
	{
		RegEx reg_first = RegEx("([\n]+)tool");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@tool", true);
		RegEx reg_second = RegEx("^tool");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@tool", true);
	}
	{
		RegEx reg_first = RegEx("([\n\t]+)export\\b");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@export", true);
		RegEx reg_second = RegEx("^export");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@export", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)onready");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@onready", true);
		RegEx reg_second = RegEx("^onready");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@onready", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)remote func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@rpc(any_peer) func", true);
		RegEx reg_second = RegEx("^remote func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@rpc(any_peer) func", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)remotesync func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@rpc(any_peer, call_local) func", true);
		RegEx reg_second = RegEx("^remotesync func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@rpc(any_peer, call_local) func", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)sync func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@rpc(any_peer, call_local) func", true);
		RegEx reg_second = RegEx("^sync func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@rpc(any_peer, call_local) func", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)slave func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@rpc func", true);
		RegEx reg_second = RegEx("^slave func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@rpc func", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)puppet func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@rpc func", true);
		RegEx reg_second = RegEx("^puppet func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@rpc func", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)puppetsync func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1@rpc(call_local) func", true);
		RegEx reg_second = RegEx("^puppetsync func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, "@rpc(call_local) func", true);
	}
	String error_message = "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using get_multiplayer().get_remote_sender_id()\n";
	{
		RegEx reg_first = RegEx("([\n]+)master func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1" + error_message + "@rpc func", true);
		RegEx reg_second = RegEx("^master func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, error_message + "@rpc func", true);
	}
	{
		RegEx reg_first = RegEx("([\n]+)mastersync func");
		CRASH_COND(!reg_first.is_valid());
		file_content = reg_first.sub(file_content, "$1" + error_message + "@rpc(call_local) func", true);
		RegEx reg_second = RegEx("^mastersync func");
		CRASH_COND(!reg_second.is_valid());
		file_content = reg_second.sub(file_content, error_message + "@rpc(call_local) func", true);
	}
}

Vector<String> ProjectConverter3To4::check_for_rename_gdscript_keywords(Vector<String> &file_content) {
	Vector<String> found_things;

	int current_line = 1;

	for (String &line : file_content) {
		String old;
		old = line;
		{
			RegEx reg_first = RegEx("^tool");
			CRASH_COND(!reg_first.is_valid());
			line = reg_first.sub(line, "@tool", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "tool", "@tool", line));
		}
		old = line;
		{
			RegEx reg_first = RegEx("([\t]+)export\\b");
			CRASH_COND(!reg_first.is_valid());
			line = reg_first.sub(line, "$1@export", true);
			RegEx reg_second = RegEx("^export");
			CRASH_COND(!reg_second.is_valid());
			line = reg_second.sub(line, "@export", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "export", "@export", line));
		}
		old = line;
		{
			RegEx reg_first = RegEx("^onready");
			CRASH_COND(!reg_first.is_valid());
			line = reg_first.sub(line, "@onready", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "onready", "@onready", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^remote func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc(any_peer) func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "remote func", "@rpc(any_peer) func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^remotesync func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc(any_peer, call_local)) func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "remotesync func", "@rpc(any_peer, call_local)) func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^sync func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc(any_peer, call_local)) func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "sync func", "@rpc(any_peer, call_local)) func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^slave func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "slave func", "@rpc func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^puppet func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "puppet func", "@rpc func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^puppetsync func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc(call_local) func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "puppetsync func", "@rpc(call_local) func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^master func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "master func", "@rpc func", line));
		}
		old = line;
		{
			RegEx regex = RegEx("^mastersync func");
			CRASH_COND(!regex.is_valid());
			line = regex.sub(line, "@rpc(call_local) func", true);
		}
		if (old != line) {
			found_things.append(line_formatter(current_line, "mastersync func", "@rpc(call_local) func", line));
		}
		old = line;

		current_line++;
	}

	return found_things;
}

void ProjectConverter3To4::custom_rename(String &file_content, String from, String to) {
	RegEx reg = RegEx(String("\\b") + from + "\\b");
	CRASH_COND(!reg.is_valid());
	file_content = reg.sub(file_content, to, true);
};

Vector<String> ProjectConverter3To4::check_for_custom_rename(Vector<String> &file_content, String from, String to) {
	Vector<String> found_things;

	RegEx reg = RegEx(String("\\b") + from + "\\b");
	CRASH_COND(!reg.is_valid());

	int current_line = 1;
	for (String &line : file_content) {
		TypedArray<RegExMatch> reg_match = reg.search_all(line);
		if (reg_match.size() > 0) {
			found_things.append(line_formatter(current_line, from.replace("\\.", "."), to, line)); // Without replacing it will print "\.shader" instead ".shader"
		}
		current_line++;
	}
	return found_things;
}

void ProjectConverter3To4::rename_common(const char *array[][2], String &file_content) {
	int current_index = 0;
	while (array[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + array[current_index][0] + "\\b");
		CRASH_COND(!reg.is_valid());
		file_content = reg.sub(file_content, array[current_index][1], true);
		current_index++;
	}
}

// Common renaming,
Vector<String> ProjectConverter3To4::check_for_rename_common(const char *array[][2], Vector<String> &file_content) {
	int current_index = 0;

	Vector<String> found_things;

	while (array[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + array[current_index][0] + "\\b");
		CRASH_COND(!reg.is_valid());

		int current_line = 1;
		for (String &line : file_content) {
			TypedArray<RegExMatch> reg_match = reg.search_all(line);
			if (reg_match.size() > 0) {
				found_things.append(line_formatter(current_line, array[current_index][0], array[current_index][1], line));
			}
			current_line++;
		}
		current_index++;
	}
	return found_things;
}

// Formats data to print them into user console when trying to convert data
String ProjectConverter3To4::line_formatter(int current_line, String from, String to, String line) {
	if (from.size() > 200) {
		from = from.substr(0, 197) + "...";
	}
	if (to.size() > 200) {
		to = to.substr(0, 197) + "...";
	}
	if (line.size() > 400) {
		line = line.substr(0, 397) + "...";
	}
	return String("Line (") + itos(current_line) + ") " + from.replace("\r", "").replace("\n", "") + " -> " + to.replace("\r", "").replace("\n", "") + "  -  LINE \"\"\" " + line.replace("\r", "").replace("\n", "").strip_edges() + " \"\"\"";
}

String ProjectConverter3To4::simple_line_formatter(int current_line, String old_line, String line) {
	if (old_line.size() > 1000) {
		old_line = old_line.substr(0, 997) + "...";
	}
	if (line.size() > 1000) {
		line = line.substr(0, 997) + "...";
	}
	return String("Line (") + itos(current_line) + ") - FULL LINES - \"\"\"" + old_line.replace("\r", "").replace("\n", "").strip_edges() + "\"\"\"  =====>  \"\"\" " + line.replace("\r", "").replace("\n", "").strip_edges() + " \"\"\"";
}

#else // No regex.

int ProjectConverter3To4::convert() {
	ERR_FAIL_V_MSG(ERROR_CODE, "Can't run converter for Godot 3.x projects as RegEx module is disabled.");
}

int ProjectConverter3To4::validate_conversion() {
	ERR_FAIL_V_MSG(ERROR_CODE, "Can't validate conversion for Godot 3.x projects as RegEx module is disabled.");
}

#endif // MODULE_REGEX_ENABLED
