/**************************************************************************/
/*  project_converter_3_to_4.cpp                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "project_converter_3_to_4.h"

#include "modules/modules_enabled.gen.h"

const int ERROR_CODE = 77;

#ifdef MODULE_REGEX_ENABLED

#include "modules/regex/regex.h"

#include "core/io/dir_access.h"
#include "core/os/time.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/templates/local_vector.h"

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
	{ "BODY_MODE_CHARACTER", "BODY_MODE_RIGID_LINEAR" }, // PhysicsServer
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
	{ "CONNECT_ONESHOT", "CONNECT_ONE_SHOT" }, // Object
	{ "CONTAINER_PROPERTY_EDITOR_BOTTOM", "CONTAINER_INSPECTOR_BOTTOM" }, // EditorPlugin
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
	{ "MODE_KINEMATIC", "FREEZE_MODE_KINEMATIC" }, // RigidBody
	{ "MODE_OPEN_ANY", "FILE_MODE_OPEN_ANY" }, // FileDialog
	{ "MODE_OPEN_DIR", "FILE_MODE_OPEN_DIR" }, // FileDialog
	{ "MODE_OPEN_FILE", "FILE_MODE_OPEN_FILE" }, // FileDialog
	{ "MODE_OPEN_FILES", "FILE_MODE_OPEN_FILES" }, // FileDialog
	{ "MODE_SAVE_FILE", "FILE_MODE_SAVE_FILE" }, // FileDialog
	{ "MODE_STATIC", "FREEZE_MODE_STATIC" }, // RigidBody
	{ "NOTIFICATION_APP_PAUSED", "NOTIFICATION_APPLICATION_PAUSED" }, // MainLoop
	{ "NOTIFICATION_APP_RESUMED", "NOTIFICATION_APPLICATION_RESUMED" }, // MainLoop
	{ "NOTIFICATION_INSTANCED", "NOTIFICATION_SCENE_INSTANTIATED" }, // Node
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
	// { "set_flag", "set_particle_flag"}, // ParticleProcessMaterial broke Window, HingeJoint3D
	// { "set_h_offset", "set_drag_horizontal_offset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// { "set_margin", "set_offset" }, // Control broke Shape3D, AtlasTexture
	// { "set_mode", "set_mode_file_mode" }, // FileDialog broke Panel, Shader, CSGPolygon, Tilemap
	// { "set_normal", "surface_set_normal"}, // ImmediateGeometry broke SurfaceTool, WorldMarginShape2D
	// { "set_offset", "set_progress" }, // PathFollow2D, PathFollow3D - Too common
	// { "set_process_mode", "set_process_callback" }, // AnimationTree broke Node, Tween, Sky
	// { "set_refuse_new_network_connections", "set_refuse_new_connections"}, // MultiplayerAPI broke SceneTree
	// { "set_tooltip", "set_tooltip_text" }, // Control, breaks TreeItem, at least for now.
	// { "set_uv", "surface_set_uv" }, // ImmediateMesh broke Polygon2D
	// { "set_v_offset", "set_drag_vertical_offset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// {"get_points","get_points_id"},// Astar, broke Line2D, Convexpolygonshape
	// {"get_v_scroll","get_v_scroll_bar"},//ItemList, broke TextView
	// { "get_stylebox", "get_theme_stylebox" }, // Control - Will rename the method in Theme as well, skipping
	{ "_about_to_show", "_about_to_popup" }, // ColorPickerButton
	{ "_get_configuration_warning", "_get_configuration_warnings" }, // Node
	{ "_set_current", "set_current" }, // Camera2D
	{ "_set_editor_description", "set_editor_description" }, // Node
	{ "_toplevel_raise_self", "_top_level_raise_self" }, // CanvasItem
	{ "_update_wrap_at", "_update_wrap_at_column" }, // TextEdit
	{ "add_animation", "add_animation_library" }, // AnimationPlayer
	{ "add_cancel", "add_cancel_button" }, // AcceptDialog
	{ "add_central_force", "apply_central_force" }, //RigidBody2D
	{ "add_child_below_node", "add_sibling" }, // Node
	{ "add_color_override", "add_theme_color_override" }, // Control
	{ "add_constant_override", "add_theme_constant_override" }, // Control
	{ "add_font_override", "add_theme_font_override" }, // Control
	{ "add_force", "apply_force" }, //RigidBody2D
	{ "add_icon_override", "add_theme_icon_override" }, // Control
	{ "add_scene_import_plugin", "add_scene_format_importer_plugin" }, //EditorPlugin
	{ "add_spatial_gizmo_plugin", "add_node_3d_gizmo_plugin" }, // EditorPlugin
	{ "add_stylebox_override", "add_theme_stylebox_override" }, // Control
	{ "add_torque", "apply_torque" }, //RigidBody2D
	{ "agent_set_neighbor_dist", "agent_set_neighbor_distance" }, // NavigationServer2D, NavigationServer3D
	{ "apply_changes", "_apply_changes" }, // EditorPlugin
	{ "body_add_force", "body_apply_force" }, // PhysicsServer2D
	{ "body_add_torque", "body_apply_torque" }, // PhysicsServer2D
	{ "bumpmap_to_normalmap", "bump_map_to_normal_map" }, // Image
	{ "can_be_hidden", "_can_be_hidden" }, // EditorNode3DGizmoPlugin
	{ "can_drop_data", "_can_drop_data" }, // Control
	{ "can_generate_small_preview", "_can_generate_small_preview" }, // EditorResourcePreviewGenerator
	{ "can_instance", "can_instantiate" }, // PackedScene, Script
	{ "canvas_light_set_scale", "canvas_light_set_texture_scale" }, // RenderingServer
	{ "center_viewport_to_cursor", "center_viewport_to_caret" }, // TextEdit
	{ "change_scene", "change_scene_to_file" }, // SceneTree
	{ "change_scene_to", "change_scene_to_packed" }, // SceneTree
	{ "clip_polygons_2d", "clip_polygons" }, // Geometry2D
	{ "clip_polyline_with_polygon_2d", "clip_polyline_with_polygon" }, //Geometry2D
	{ "commit_handle", "_commit_handle" }, // EditorNode3DGizmo
	{ "convex_hull_2d", "convex_hull" }, // Geometry2D
	{ "create_gizmo", "_create_gizmo" }, // EditorNode3DGizmoPlugin
	{ "cursor_get_blink_speed", "get_caret_blink_interval" }, // TextEdit
	{ "cursor_get_column", "get_caret_column" }, // TextEdit
	{ "cursor_get_line", "get_caret_line" }, // TextEdit
	{ "cursor_set_blink_enabled", "set_caret_blink_enabled" }, // TextEdit
	{ "cursor_set_blink_speed", "set_caret_blink_interval" }, // TextEdit
	{ "cursor_set_column", "set_caret_column" }, // TextEdit
	{ "cursor_set_line", "set_caret_line" }, // TextEdit
	{ "damped_spring_joint_create", "joint_make_damped_spring" }, // PhysicsServer2D
	{ "damped_string_joint_get_param", "damped_spring_joint_get_param" }, // PhysicsServer2D
	{ "damped_string_joint_set_param", "damped_spring_joint_set_param" }, // PhysicsServer2D
	{ "dectime", "move_toward" }, // GDScript, Math functions
	{ "delete_char_at_cursor", "delete_char_at_caret" }, // LineEdit
	{ "deselect_items", "deselect_all" }, // FileDialog
	{ "disable_plugin", "_disable_plugin" }, // EditorPlugin
	{ "drop_data", "_drop_data" }, // Control
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
	{ "get_applied_force", "get_constant_force" }, //RigidBody2D
	{ "get_applied_torque", "get_constant_torque" }, //RigidBody2D
	{ "get_audio_bus", "get_audio_bus_name" }, // Area3D
	{ "get_bound_child_nodes_to_bone", "get_bone_children" }, // Skeleton3D
	{ "get_camera", "get_camera_3d" }, // Viewport -> this is also convertible to get_camera_2d, broke GLTFNode
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
	{ "get_command", "is_command_or_control_pressed" }, // InputEventWithModifiers
	{ "get_constant_types", "get_constant_type_list" }, // Theme
	{ "get_control", "is_ctrl_pressed" }, // InputEventWithModifiers
	{ "get_cull_mask_bit", "get_cull_mask_value" }, // Camera3D
	{ "get_cursor_position", "get_caret_column" }, // LineEdit
	{ "get_d", "get_distance" }, // LineShape2D
	{ "get_depth_bias_enable", "get_depth_bias_enabled" }, // RDPipelineRasterizationState
	{ "get_drag_data", "_get_drag_data" }, // Control
	{ "get_editor_viewport", "get_editor_main_screen" }, // EditorPlugin
	{ "get_enabled_focus_mode", "get_focus_mode" }, // BaseButton
	{ "get_endian_swap", "is_big_endian" }, // File
	{ "get_error_string", "get_error_message" }, // JSON
	{ "get_filename", "get_scene_file_path" }, // Node, WARNING, this may be used in a lot of other places
	{ "get_focus_neighbour", "get_focus_neighbor" }, // Control
	{ "get_follow_smoothing", "get_position_smoothing_speed" }, // Camera2D
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
	{ "get_neighbor_dist", "get_neighbor_distance" }, // NavigationAgent2D, NavigationAgent3D
	{ "get_network_connected_peers", "get_peers" }, // Multiplayer API
	{ "get_network_master", "get_multiplayer_authority" }, // Node
	{ "get_network_peer", "get_multiplayer_peer" }, // Multiplayer API
	{ "get_network_unique_id", "get_unique_id" }, // Multiplayer API
	{ "get_ok", "get_ok_button" }, // AcceptDialog
	{ "get_oneshot", "get_one_shot" }, // AnimatedTexture
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
	{ "get_reverb_bus", "set_reverb_bus_name" }, // Area3D
	{ "get_rpc_sender_id", "get_remote_sender_id" }, // Multiplayer API
	{ "get_save_extension", "_get_save_extension" }, // EditorImportPlugin
	{ "get_scancode", "get_keycode" }, // InputEventKey
	{ "get_scancode_string", "get_keycode_string" }, // OS
	{ "get_scancode_with_modifiers", "get_keycode_with_modifiers" }, // InputEventKey
	{ "get_selected_path", "get_current_directory" }, // EditorInterface
	{ "get_shift", "is_shift_pressed" }, // InputEventWithModifiers
	{ "get_size_override", "get_size_2d_override" }, // SubViewport
	{ "get_slide_count", "get_slide_collision_count" }, // CharacterBody2D, CharacterBody3D
	{ "get_slips_on_slope", "get_slide_on_slope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "get_space_override_mode", "get_gravity_space_override_mode" }, // Area2D
	{ "get_spatial_node", "get_node_3d" }, // EditorNode3DGizmo
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
	{ "get_unit_db", "get_volume_db" }, // AudioStreamPlayer3D
	{ "get_unit_offset", "get_progress_ratio" }, // PathFollow2D, PathFollow3D
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
	{ "import_scene_from_other_importer", "_import_scene" }, //EditorSceneFormatImporter
	{ "instance_set_surface_material", "instance_set_surface_override_material" }, // RenderingServer
	{ "interpolate", "sample" }, // Curve, Curve2D, Curve3D, Gradient
	{ "intersect_polygons_2d", "intersect_polygons" }, // Geometry2D
	{ "intersect_polyline_with_polygon_2d", "intersect_polyline_with_polygon" }, // Geometry2D
	{ "is_a_parent_of", "is_ancestor_of" }, // Node
	{ "is_commiting_action", "is_committing_action" }, // UndoRedo
	{ "is_doubleclick", "is_double_click" }, // InputEventMouseButton
	{ "is_draw_red", "is_draw_warning" }, // EditorProperty
	{ "is_follow_smoothing_enabled", "is_position_smoothing_enabled" }, // Camera2D
	{ "is_h_drag_enabled", "is_drag_horizontal_enabled" }, // Camera2D
	{ "is_handle_highlighted", "_is_handle_highlighted" }, // EditorNode3DGizmo, EditorNode3DGizmoPlugin
	{ "is_inverting_faces", "get_flip_faces" }, // CSGPrimitive3D
	{ "is_network_master", "is_multiplayer_authority" }, // Node
	{ "is_network_server", "is_server" }, // Multiplayer API
	{ "is_normalmap", "is_normal_map" }, // NoiseTexture
	{ "is_refusing_new_network_connections", "is_refusing_new_connections" }, // Multiplayer API
	{ "is_region", "is_region_enabled" }, // Sprite2D
	{ "is_rotating", "is_ignoring_rotation" }, // Camera2D
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
	{ "raise", "move_to_front" }, // CanvasItem
	{ "recognize", "_recognize" }, // ResourceFormatLoader
	{ "regen_normalmaps", "regen_normal_maps" }, // ArrayMesh
	{ "remove", "remove_at" }, // Array, broke Directory
	{ "remove_animation", "remove_animation_library" }, // AnimationPlayer
	{ "remove_color_override", "remove_theme_color_override" }, // Control
	{ "remove_constant_override", "remove_theme_constant_override" }, // Control
	{ "remove_font_override", "remove_theme_font_override" }, // Control
	{ "remove_icon_override", "remove_theme_icon_override" }, // Control
	{ "remove_scene_import_plugin", "remove_scene_format_importer_plugin" }, //EditorPlugin
	{ "remove_spatial_gizmo_plugin", "remove_node_3d_gizmo_plugin" }, // EditorPlugin
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
	{ "set_command", "set_meta_pressed" }, // InputEventWithModifiers
	{ "set_control", "set_ctrl_pressed" }, // InputEventWithModifiers
	{ "set_create_options", "_set_create_options" }, //  EditorResourcePicker
	{ "set_cull_mask_bit", "set_cull_mask_value" }, // Camera3D
	{ "set_cursor_position", "set_caret_column" }, // LineEdit
	{ "set_d", "set_distance" }, // WorldMarginShape2D
	{ "set_depth_bias_enable", "set_depth_bias_enabled" }, // RDPipelineRasterizationState
	{ "set_doubleclick", "set_double_click" }, // InputEventMouseButton
	{ "set_draw_red", "set_draw_warning" }, // EditorProperty
	{ "set_enable_follow_smoothing", "set_position_smoothing_enabled" }, // Camera2D
	{ "set_enabled_focus_mode", "set_focus_mode" }, // BaseButton
	{ "set_endian_swap", "set_big_endian" }, // File
	{ "set_expand_to_text_length", "set_expand_to_text_length_enabled" }, // LineEdit
	{ "set_filename", "set_scene_file_path" }, // Node, WARNING, this may be used in a lot of other places
	{ "set_focus_neighbour", "set_focus_neighbor" }, // Control
	{ "set_follow_smoothing", "set_position_smoothing_speed" }, // Camera2D
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
	{ "set_neighbor_dist", "set_neighbor_distance" }, // NavigationAgent2D, NavigationAgent3D
	{ "set_network_master", "set_multiplayer_authority" }, // Node
	{ "set_network_peer", "set_multiplayer_peer" }, // Multiplayer API
	{ "set_oneshot", "set_one_shot" }, // AnimatedTexture
	{ "set_pause_mode", "set_process_mode" }, // Node
	{ "set_physical_scancode", "set_physical_keycode" }, // InputEventKey
	{ "set_proximity_fade", "set_proximity_fade_enabled" }, // Material
	{ "set_refuse_new_network_connections", "set_refuse_new_connections" }, // Multiplayer API
	{ "set_region", "set_region_enabled" }, // Sprite2D, Sprite broke AtlasTexture
	{ "set_region_filter_clip", "set_region_filter_clip_enabled" }, // Sprite2D
	{ "set_reverb_bus", "set_reverb_bus_name" }, // Area3D
	{ "set_rotate", "set_rotates" }, // PathFollow2D
	{ "set_scancode", "set_keycode" }, // InputEventKey
	{ "set_shift", "set_shift_pressed" }, // InputEventWithModifiers
	{ "set_size_override", "set_size_2d_override" }, // SubViewport broke ImageTexture
	{ "set_size_override_stretch", "set_size_2d_override_stretch" }, // SubViewport
	{ "set_slips_on_slope", "set_slide_on_slope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "set_sort_enabled", "set_y_sort_enabled" }, // Node2D
	{ "set_space_override_mode", "set_gravity_space_override_mode" }, // Area2D
	{ "set_spatial_node", "set_node_3d" }, // EditorNode3DGizmo
	{ "set_speed", "set_velocity" }, // InputEventMouseMotion
	{ "set_ssao_edge_sharpness", "set_ssao_sharpness" }, // Environment
	{ "set_surface_material", "set_surface_override_material" }, // MeshInstance3D broke ImporterMesh
	{ "set_tab_align", "set_tab_alignment" }, //TabContainer
	{ "set_tangent", "surface_set_tangent" }, // ImmediateGeometry broke SurfaceTool
	{ "set_text_align", "set_text_alignment" }, // Button
	{ "set_timer_process_mode", "set_timer_process_callback" }, // Timer
	{ "set_translation", "set_position" }, // Node3D - this broke GLTFNode which is used rarely
	{ "set_unit_db", "set_volume_db" }, // AudioStreamPlayer3D
	{ "set_unit_offset", "set_progress_ratio" }, // PathFollow2D, PathFollow3D
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
	{ "unselect", "deselect" }, // ItemList
	{ "unselect_all", "deselect_all" }, // ItemList
	{ "update_configuration_warning", "update_configuration_warnings" }, // Node
	{ "update_gizmo", "update_gizmos" }, // Node3D
	{ "viewport_set_use_arvr", "viewport_set_use_xr" }, // RenderingServer
	{ "warp_mouse_position", "warp_mouse" }, // Input
	{ "world_to_map", "local_to_map" }, // TileMap, GridMap
	{ "set_shader_param", "set_shader_parameter" }, // ShaderMaterial
	{ "get_shader_param", "get_shader_parameter" }, // ShaderMaterial
	{ "set_uniform_name", "set_parameter_name" }, // ParameterRef
	{ "get_uniform_name", "get_parameter_name" }, // ParameterRef

	// Builtin types
	// Remember to add them to builtin_types_excluded_functions variable, because for now this functions cannot be listed
	//	{ "empty", "is_empty" }, // Array - Used as custom rule  // Be careful, this will be used everywhere
	{ "clamped", "clamp" }, // Vector2  // Be careful, this will be used everywhere
	{ "get_rotation_quat", "get_rotation_quaternion" }, // Basis
	{ "grow_margin", "grow_side" }, // Rect2
	{ "invert", "reverse" }, // Array - TODO check  // Be careful, this will be used everywhere
	{ "is_abs_path", "is_absolute_path" }, // String
	{ "is_valid_integer", "is_valid_int" }, // String
	{ "linear_interpolate", "lerp" }, // Color
	{ "find_last", "rfind" }, // Array, String
	{ "to_ascii", "to_ascii_buffer" }, // String
	{ "to_utf8", "to_utf8_buffer" }, // String
	{ "to_wchar", "to_utf32_buffer" }, // String // TODO - utf32 or utf16?

	// @GlobalScope
	// Remember to add them to builtin_types_excluded_functions variable, because for now this functions cannot be listed
	{ "bytes2var", "bytes_to_var" },
	{ "bytes2var_with_objects", "bytes_to_var_with_objects" },
	{ "db2linear", "db_to_linear" },
	{ "deg2rad", "deg_to_rad" },
	{ "linear2db", "linear_to_db" },
	{ "rad2deg", "rad_to_deg" },
	{ "rand_range", "randf_range" },
	{ "range_lerp", "remap" },
	{ "stepify", "snapped" },
	{ "str2var", "str_to_var" },
	{ "var2str", "var_to_str" },
	{ "var2bytes", "var_to_bytes" },
	{ "var2bytes_with_objects", "var_to_bytes_with_objects" },

	// @GDScript
	// Remember to add them to builtin_types_excluded_functions variable, because for now this functions cannot be listed
	{ "dict2inst", "dict_to_inst" },
	{ "inst2dict", "inst_to_dict" },

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
	// { "SetFlag", "SetParticleFlag"}, // ParticleProcessMaterial broke Window, HingeJoint3D
	// { "SetHOffset", "SetDragHorizontalOffset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// { "SetMargin", "SetOffset" }, // Control broke Shape3D, AtlasTexture
	// { "SetMode", "SetModeFileMode" }, // FileDialog broke Panel, Shader, CSGPolygon, Tilemap
	// { "SetNormal", "SurfaceSetNormal"}, // ImmediateGeometry broke SurfaceTool, WorldMarginShape2D
	// { "SetOffset", "SetProgress" }, // PathFollow2D, PathFollow3D - Too common
	// { "SetProcessMode", "SetProcessCallback" }, // AnimationTree broke Node, Tween, Sky
	// { "SetRefuseNewNetworkConnections", "SetRefuseNewConnections"}, // MultiplayerAPI broke SceneTree
	// { "SetTooltip", "SetTooltipText" }, // Control, breaks TreeItem, at least for now.
	// { "SetUv", "SurfaceSetUv" }, // ImmediateMesh broke Polygon2D
	// { "SetVOffset", "SetDragVerticalOffset" }, // Camera2D broke Camera3D, PathFollow3D, PathFollow2D
	// {"GetPoints","GetPointsId"},// Astar, broke Line2D, Convexpolygonshape
	// {"GetVScroll","GetVScrollBar"},//ItemList, broke TextView
	// { "GetStylebox", "GetThemeStylebox" }, // Control - Will rename the method in Theme as well, skipping
	{ "AddSpatialGizmoPlugin", "AddNode3dGizmoPlugin" }, // EditorPlugin
	{ "RenderingServer", "GetTabAlignment" }, // Tab
	{ "_AboutToShow", "_AboutToPopup" }, // ColorPickerButton
	{ "_GetConfigurationWarning", "_GetConfigurationWarnings" }, // Node
	{ "_SetCurrent", "SetCurrent" }, // Camera2D
	{ "_SetEditorDescription", "SetEditorDescription" }, // Node
	{ "_SetPlaying", "SetPlaying" }, // AnimatedSprite3D
	{ "_ToplevelRaiseSelf", "_TopLevelRaiseSelf" }, // CanvasItem
	{ "_UpdateWrapAt", "_UpdateWrapAtColumn" }, // TextEdit
	{ "AddAnimation", "AddAnimationLibrary" }, // AnimationPlayer
	{ "AddCancel", "AddCancelButton" }, // AcceptDialog
	{ "AddCentralForce", "AddConstantCentralForce" }, //RigidBody2D
	{ "AddChildBelowNode", "AddSibling" }, // Node
	{ "AddColorOverride", "AddThemeColorOverride" }, // Control
	{ "AddConstantOverride", "AddThemeConstantOverride" }, // Control
	{ "AddFontOverride", "AddThemeFontOverride" }, // Control
	{ "AddForce", "AddConstantForce" }, //RigidBody2D
	{ "AddIconOverride", "AddThemeIconOverride" }, // Control
	{ "AddSceneImportPlugin", "AddSceneFormatImporterPlugin" }, //EditorPlugin
	{ "AddStyleboxOverride", "AddThemeStyleboxOverride" }, // Control
	{ "AddTorque", "AddConstantTorque" }, //RigidBody2D
	{ "AgentSetNeighborDist", "AgentSetNeighborDistance" }, // NavigationServer2D, NavigationServer3D
	{ "BindChildNodeToBone", "SetBoneChildren" }, // Skeleton3D
	{ "BumpmapToNormalmap", "BumpMapToNormalMap" }, // Image
	{ "CanBeHidden", "_CanBeHidden" }, // EditorNode3DGizmoPlugin
	{ "CanDropData", "_CanDropData" }, // Control
	{ "CanDropDataFw", "_CanDropDataFw" }, // ScriptEditor
	{ "CanGenerateSmallPreview", "_CanGenerateSmallPreview" }, // EditorResourcePreviewGenerator
	{ "CanInstance", "CanInstantiate" }, // PackedScene, Script
	{ "CanvasLightSetScale", "CanvasLightSetTextureScale" }, // RenderingServer
	{ "CenterViewportToCursor", "CenterViewportToCaret" }, // TextEdit
	{ "ChangeScene", "ChangeSceneToFile" }, // SceneTree
	{ "ChangeSceneTo", "ChangeSceneToPacked" }, // SceneTree
	{ "ClipPolygons2d", "ClipPolygons" }, // Geometry2D
	{ "ClipPolylineWithPolygon2d", "ClipPolylineWithPolygon" }, //Geometry2D
	{ "CommitHandle", "_CommitHandle" }, // EditorNode3DGizmo
	{ "ConvexHull2d", "ConvexHull" }, // Geometry2D
	{ "CursorGetBlinkSpeed", "GetCaretBlinkInterval" }, // TextEdit
	{ "CursorGetColumn", "GetCaretColumn" }, // TextEdit
	{ "CursorGetLine", "GetCaretLine" }, // TextEdit
	{ "CursorSetBlinkEnabled", "SetCaretBlinkEnabled" }, // TextEdit
	{ "CursorSetBlinkSpeed", "SetCaretBlinkInterval" }, // TextEdit
	{ "CursorSetColumn", "SetCaretColumn" }, // TextEdit
	{ "CursorSetLine", "SetCaretLine" }, // TextEdit
	{ "DampedSpringJointCreate", "JointMakeDampedSpring" }, // PhysicsServer2D
	{ "DampedStringJointGetParam", "DampedSpringJointGetParam" }, // PhysicsServer2D
	{ "DampedStringJointSetParam", "DampedSpringJointSetParam" }, // PhysicsServer2D
	{ "DeleteCharAtCursor", "DeleteCharAtCaret" }, // LineEdit
	{ "DeselectItems", "DeselectAll" }, // FileDialog
	{ "DropData", "_DropData" }, // Control
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
	{ "GetAppliedForce", "GetConstantForce" }, //RigidBody2D
	{ "GetAppliedTorque", "GetConstantTorque" }, //RigidBody2D
	{ "GetAudioBus", "GetAudioBusName" }, // Area3D
	{ "GetBoundChildNodesToBone", "GetBoneChildren" }, // Skeleton3D
	{ "GetCamera", "GetCamera3d" }, // Viewport -> this is also convertible to getCamera2d, broke GLTFNode
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
	{ "GetDepthBiasEnable", "GetDepthBiasEnabled" }, // RDPipelineRasterizationState
	{ "GetDragDataFw", "_GetDragDataFw" }, // ScriptEditor
	{ "GetEditorViewport", "GetViewport" }, // EditorPlugin
	{ "GetEnabledFocusMode", "GetFocusMode" }, // BaseButton
	{ "GetEndianSwap", "IsBigEndian" }, // File
	{ "GetErrorString", "GetErrorMessage" }, // JSON
	{ "GetFocusNeighbour", "GetFocusNeighbor" }, // Control
	{ "GetFollowSmoothing", "GetFollowSmoothingSpeed" }, // Camera2D
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
	{ "GetNeighborDist", "GetNeighborDistance" }, // NavigationAgent2D, NavigationAgent3D
	{ "GetNetworkConnectedPeers", "GetPeers" }, // Multiplayer API
	{ "GetNetworkMaster", "GetMultiplayerAuthority" }, // Node
	{ "GetNetworkPeer", "GetMultiplayerPeer" }, // Multiplayer API
	{ "GetNetworkUniqueId", "GetUniqueId" }, // Multiplayer API
	{ "GetOneshot", "GetOneShot" }, // AnimatedTexture
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
	{ "GetReverbBus", "GetReverbBusName" }, // Area3D
	{ "GetRpcSenderId", "GetRemoteSenderId" }, // Multiplayer API
	{ "GetSaveExtension", "_GetSaveExtension" }, // EditorImportPlugin
	{ "GetScancode", "GetKeycode" }, // InputEventKey
	{ "GetScancodeString", "GetKeycodeString" }, // OS
	{ "GetScancodeWithModifiers", "GetKeycodeWithModifiers" }, // InputEventKey
	{ "GetShift", "IsShiftPressed" }, // InputEventWithModifiers
	{ "GetSizeOverride", "GetSize2dOverride" }, // SubViewport
	{ "GetSlipsOnSlope", "GetSlideOnSlope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "GetSpaceOverrideMode", "GetGravitySpaceOverrideMode" }, // Area2D
	{ "GetSpatialNode", "GetNode3d" }, // EditorNode3DGizmo
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
	{ "GetUnitDb", "GetVolumeDb" }, // AudioStreamPlayer3D
	{ "GetUnitOffset", "GetProgressRatio" }, // PathFollow2D, PathFollow3D
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
	{ "IsFollowSmoothingEnabled", "IsPositionSmoothingEnabled" }, // Camera2D
	{ "IsHDragEnabled", "IsDragHorizontalEnabled" }, // Camera2D
	{ "IsHandleHighlighted", "_IsHandleHighlighted" }, // EditorNode3DGizmo, EditorNode3DGizmoPlugin
	{ "IsNetworkMaster", "IsMultiplayerAuthority" }, // Node
	{ "IsNetworkServer", "IsServer" }, // Multiplayer API
	{ "IsNormalmap", "IsNormalMap" }, // NoiseTexture
	{ "IsRefusingNewNetworkConnections", "IsRefusingNewConnections" }, // Multiplayer API
	{ "IsRegion", "IsRegionEnabled" }, // Sprite2D
	{ "IsRotating", "IsIgnoringRotation" }, // Camera2D
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
	{ "RemoveSpatialGizmoPlugin", "RemoveNode3dGizmoPlugin" }, // EditorPlugin
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
	{ "SetDepthBiasEnable", "SetDepthBiasEnabled" }, // RDPipelineRasterizationState
	{ "SetDoubleclick", "SetDoubleClick" }, // InputEventMouseButton
	{ "SetEnableFollowSmoothing", "SetFollowSmoothingEnabled" }, // Camera2D
	{ "SetEnabledFocusMode", "SetFocusMode" }, // BaseButton
	{ "SetEndianSwap", "SetBigEndian" }, // File
	{ "SetExpandToTextLength", "SetExpandToTextLengthEnabled" }, // LineEdit
	{ "SetFocusNeighbour", "SetFocusNeighbor" }, // Control
	{ "SetFollowSmoothing", "SetFollowSmoothingSpeed" }, // Camera2D
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
	{ "SetNeighborDist", "SetNeighborDistance" }, // NavigationAgent2D, NavigationAgent3D
	{ "SetNetworkMaster", "SetMultiplayerAuthority" }, // Node
	{ "SetNetworkPeer", "SetMultiplayerPeer" }, // Multiplayer API
	{ "SetOneshot", "SetOneShot" }, // AnimatedTexture
	{ "SetPhysicalScancode", "SetPhysicalKeycode" }, // InputEventKey
	{ "SetProximityFade", "SetProximityFadeEnabled" }, // Material
	{ "SetRefuseNewNetworkConnections", "SetRefuseNewConnections" }, // Multiplayer API
	{ "SetRegion", "SetRegionEnabled" }, // Sprite2D, Sprite broke AtlasTexture
	{ "SetRegionFilterClip", "SetRegionFilterClipEnabled" }, // Sprite2D
	{ "SetReverbBus", "SetReverbBusName" }, // Area3D
	{ "SetRotate", "SetRotates" }, // PathFollow2D
	{ "SetScancode", "SetKeycode" }, // InputEventKey
	{ "SetShift", "SetShiftPressed" }, // InputEventWithModifiers
	{ "SetSizeOverride", "SetSize2dOverride" }, // SubViewport broke ImageTexture
	{ "SetSizeOverrideStretch", "SetSize2dOverrideStretch" }, // SubViewport
	{ "SetSlipsOnSlope", "SetSlideOnSlope" }, // SeparationRayShape2D, SeparationRayShape3D
	{ "SetSortEnabled", "SetYSortEnabled" }, // Node2D
	{ "SetSpaceOverrideMode", "SetGravitySpaceOverrideMode" }, // Area2D
	{ "SetSpatialNode", "SetNode3d" }, // EditorNode3DGizmo
	{ "SetSpeed", "SetVelocity" }, // InputEventMouseMotion
	{ "SetSsaoEdgeSharpness", "SetSsaoSharpness" }, // Environment
	{ "SetSurfaceMaterial", "SetSurfaceOverrideMaterial" }, // MeshInstance3D broke ImporterMesh
	{ "SetTabAlign", "SetTabAlignment" }, //TabContainer
	{ "SetTangent", "SurfaceSetTangent" }, // ImmediateGeometry broke SurfaceTool
	{ "SetTextAlign", "SetTextAlignment" }, // Button
	{ "SetTimerProcessMode", "SetTimerProcessCallback" }, // Timer
	{ "SetTonemapAutoExposure", "SetTonemapAutoExposureEnabled" }, // Environment
	{ "SetTranslation", "SetPosition" }, // Node3D - this broke GLTFNode which is used rarely
	{ "SetUnitDb", "SetVolumeDb" }, // AudioStreamPlayer3D
	{ "SetUnitOffset", "SetProgressRatio" }, // PathFollow2D, PathFollow3D
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
	{ "WorldToMap", "LocalToMap" }, // TileMap, GridMap
	{ "SetShaderParam", "SetShaderParameter" }, // ShaderMaterial
	{ "GetShaderParam", "GetShaderParameter" }, // ShaderMaterial
	{ "SetUniformName", "SetParameterName" }, // ParameterRef
	{ "GetUniformName", "GetParameterName" }, // ParameterRef

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

	// @GlobalScope
	{ "Bytes2Var", "BytesToVar" },
	{ "Bytes2VarWithObjects", "BytesToVarWithObjects" },
	{ "Db2Linear", "DbToLinear" },
	{ "Deg2Rad", "DegToRad" },
	{ "Linear2Db", "LinearToDb" },
	{ "Rad2Deg", "RadToDeg" },
	{ "RandRange", "RandfRange" },
	{ "RangeLerp", "Remap" },
	{ "Stepify", "Snapped" },
	{ "Str2Var", "StrToVar" },
	{ "Var2Str", "VarToStr" },
	{ "Var2Bytes", "VarToBytes" },
	{ "Var2BytesWithObjects", "VarToBytesWithObjects" },

	// @GDScript
	{ "Dict2Inst", "DictToInst" },
	{ "Inst2Dict", "InstToDict" },

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
	// 	// {"offset","progress"}, // PathFollow2D, PathFollow3D - Name is way too vague
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
	//	{ "pressed", "button_pressed" }, // BaseButton - Will also rename the signal, skipping for now
	{ "as_normalmap", "as_normal_map" }, // NoiseTexture
	{ "bbcode_text", "text" }, // RichTextLabel
	{ "bg", "panel" }, // Theme
	{ "bg_focus", "focus" }, // Theme
	{ "caret_blink_speed", "caret_blink_interval" }, // TextEdit, LineEdit
	{ "caret_moving_by_right_click", "caret_move_on_right_click" }, // TextEdit
	{ "caret_position", "caret_column" }, // LineEdit
	{ "check_vadjust", "check_v_offset" }, // Theme
	{ "close_h_ofs", "close_h_offset" }, // Theme
	{ "close_v_ofs", "close_v_offset" }, // Theme
	{ "commentfocus", "comment_focus" }, // Theme
	{ "contacts_reported", "max_contacts_reported" }, // RigidBody
	{ "depth_bias_enable", "depth_bias_enabled" }, // RDPipelineRasterizationState
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
	{ "follow_viewport_enable", "follow_viewport_enabled" }, // CanvasItem
	{ "file_icon_modulate", "file_icon_color" }, // Theme
	{ "files_disabled", "file_disabled_color" }, // Theme
	{ "folder_icon_modulate", "folder_icon_color" }, // Theme
	{ "global_rate_scale", "playback_speed_scale" }, // AudioServer
	{ "gravity_distance_scale", "gravity_point_distance_scale" }, // Area2D
	{ "gravity_vec", "gravity_direction" }, // Area2D
	{ "hint_tooltip", "tooltip_text" }, // Control
	{ "hseparation", "h_separation" }, // Theme
	{ "icon_align", "icon_alignment" }, // Button
	{ "iterations_per_second", "physics_ticks_per_second" }, // Engine
	{ "invert_enable", "invert_enabled" }, // Polygon2D
	{ "margin_bottom", "offset_bottom" }, // Control broke NinePatchRect, StyleBox
	{ "margin_left", "offset_left" }, // Control broke NinePatchRect, StyleBox
	{ "margin_right", "offset_right" }, // Control broke NinePatchRect, StyleBox
	{ "margin_top", "offset_top" }, // Control broke NinePatchRect, StyleBox
	{ "mid_height", "height" }, // CapsuleMesh
	{ "neighbor_dist", "neighbor_distance" }, // NavigationAgent2D, NavigationAgent3D
	{ "offset_h", "drag_horizontal_offset" }, // Camera2D
	{ "offset_v", "drag_vertical_offset" }, // Camera2D
	{ "off", "unchecked" }, // Theme
	{ "off_disabled", "unchecked_disabled" }, // Theme
	{ "ofs", "offset" }, // Theme
	{ "on", "checked" }, // Theme
	{ "on_disabled", "checked_disabled" }, // Theme
	{ "oneshot", "one_shot" }, // AnimatedTexture
	{ "out_of_range_mode", "max_polyphony" }, // AudioStreamPlayer3D
	{ "pause_mode", "process_mode" }, // Node
	{ "physical_scancode", "physical_keycode" }, // InputEventKey
	{ "popup_exclusive", "exclusive" }, // Window
	{ "proximity_fade_enable", "proximity_fade_enabled" }, // Material
	{ "rect_position", "position" }, // Control
	{ "rect_global_position", "global_position" }, // Control
	{ "rect_size", "size" }, // Control
	{ "rect_min_size", "custom_minimum_size" }, // Control
	{ "rect_rotation", "rotation" }, // Control
	{ "rect_scale", "scale" }, // Control
	{ "rect_pivot_offset", "pivot_offset" }, // Control
	{ "rect_clip_content", "clip_contents" }, // Control
	{ "refuse_new_network_connections", "refuse_new_connections" }, // MultiplayerAPI
	{ "region_filter_clip", "region_filter_clip_enabled" }, // Sprite2D
	{ "reverb_bus_enable", "reverb_bus_enabled" }, // Area3D
	{ "selectedframe", "selected_frame" }, // Theme
	{ "size_override_stretch", "size_2d_override_stretch" }, // SubViewport
	{ "slips_on_slope", "slide_on_slope" }, // SeparationRayShape2D
	{ "smoothing_enabled", "follow_smoothing_enabled" }, // Camera2D
	{ "smoothing_speed", "position_smoothing_speed" }, // Camera2D
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
	{ "unit_db", "volume_db" }, // AudioStreamPlayer3D
	{ "unit_offset", "progress_ratio" }, // PathFollow2D, PathFollow3D
	{ "vseparation", "v_separation" }, // Theme
	{ "frames", "sprite_frames" }, // AnimatedSprite2D, AnimatedSprite3D

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
	// 	// {"Offset","Progress"}, // PathFollow2D, PathFollow3D - Name is way too vague
	//	// {"Shift","ShiftPressed"},// This may broke a lot of comments and user variables
	//	{ "Autowrap", "AutowrapMode" }, // Label
	//	{ "CastTo", "TargetPosition" }, // RayCast2D, RayCast3D
	//	{ "Doubleclick", "DoubleClick" }, // InputEventMouseButton
	//	{ "Group", "ButtonGroup" }, // BaseButton
	//  { "PercentVisible, "ShowPercentage}, // ProgressBar, conflicts with Label and RichTextLabel, but may be a worth it.
	//	{ "ProcessMode", "ProcessCallback" }, // AnimationTree, Camera2D
	//	{ "Scancode", "Keycode" }, // InputEventKey
	//	{ "Toplevel", "TopLevel" }, // Node
	//	{ "WindowTitle", "Title" }, // Window
	//	{ "WrapEnabled", "WrapMode" }, // TextEdit
	//	{ "Zfar", "Far" }, // Camera3D
	//	{ "Znear", "Near" }, // Camera3D
	//	{ "Pressed", "ButtonPressed" }, // BaseButton - Will also rename the signal, skipping for now
	{ "AsNormalmap", "AsNormalMap" }, // NoiseTexture
	{ "BbcodeText", "Text" }, // RichTextLabel
	{ "CaretBlinkSpeed", "CaretBlinkInterval" }, // TextEdit, LineEdit
	{ "CaretMovingByRightClick", "CaretMoveOnRightClick" }, // TextEdit
	{ "CaretPosition", "CaretColumn" }, // LineEdit
	{ "CheckVadjust", "CheckVAdjust" }, // Theme
	{ "CloseHOfs", "CloseHOffset" }, // Theme
	{ "CloseVOfs", "CloseVOffset" }, // Theme
	{ "Commentfocus", "CommentFocus" }, // Theme
	{ "DepthBiasEnable", "DepthBiasEnabled" }, // RDPipelineRasterizationState
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
	{ "FollowViewportEnable", "FollowViewportEnabled" }, // CanvasItem
	{ "GlobalRateScale", "PlaybackSpeedScale" }, // AudioServer
	{ "GravityDistanceScale", "GravityPointDistanceScale" }, // Area2D
	{ "GravityVec", "GravityDirection" }, // Area2D
	{ "HintTooltip", "TooltipText" }, // Control
	{ "Hseparation", "HSeparation" }, // Theme
	{ "IconAlign", "IconAlignment" }, // Button
	{ "IterationsPerSecond", "PhysicsTicksPerSecond" }, // Engine
	{ "InvertEnable", "InvertEnabled" }, // Polygon2D
	{ "MarginBottom", "OffsetBottom" }, // Control broke NinePatchRect, StyleBox
	{ "MarginLeft", "OffsetLeft" }, // Control broke NinePatchRect, StyleBox
	{ "MarginRight", "OffsetRight" }, // Control broke NinePatchRect, StyleBox
	{ "MarginTop", "OffsetTop" }, // Control broke NinePatchRect, StyleBox
	{ "MidHeight", "Height" }, // CapsuleMesh
	{ "NeighborDist", "NeighborDistance" }, // NavigationAgent2D, NavigationAgent3D
	{ "OffsetH", "DragHorizontalOffset" }, // Camera2D
	{ "OffsetV", "DragVerticalOffset" }, // Camera2D
	{ "Ofs", "Offset" }, // Theme
	{ "Oneshot", "OneShot" }, // AnimatedTexture
	{ "OutOfRangeMode", "MaxPolyphony" }, // AudioStreamPlayer3D
	{ "PauseMode", "ProcessMode" }, // Node
	{ "PhysicalScancode", "PhysicalKeycode" }, // InputEventKey
	{ "PopupExclusive", "Exclusive" }, // Window
	{ "ProximityFadeEnable", "ProximityFadeEnabled" }, // Material
	{ "RectPosition", "Position" }, // Control
	{ "RectGlobalPosition", "GlobalPosition" }, // Control
	{ "RectSize", "Size" }, // Control
	{ "RectMinSize", "CustomMinimumSize" }, // Control
	{ "RectRotation", "Rotation" }, // Control
	{ "RectScale", "Scale" }, // Control
	{ "RectPivotOffset", "PivotOffset" }, // Control
	{ "RectClipContent", "ClipContents" }, // Control
	{ "RefuseNewNetworkConnections", "RefuseNewConnections" }, // MultiplayerAPI
	{ "RegionFilterClip", "RegionFilterClipEnabled" }, // Sprite2D
	{ "ReverbBusEnable", "ReverbBusEnabled" }, // Area3D
	{ "Selectedframe", "SelectedFrame" }, // Theme
	{ "SizeOverrideStretch", "Size2dOverrideStretch" }, // SubViewport
	{ "SlipsOnSlope", "SlideOnSlope" }, // SeparationRayShape2D
	{ "SmoothingEnabled", "FollowSmoothingEnabled" }, // Camera2D
	{ "SmoothingSpeed", "FollowSmoothingSpeed" }, // Camera2D
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
	{ "UnitDb", "VolumeDb" }, // AudioStreamPlayer3D
	{ "UnitOffset", "ProgressRatio" }, // PathFollow2D, PathFollow3D
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
	{ "cancelled", "canceled" }, // AcceptDialog
	{ "item_double_clicked", "item_icon_double_clicked" }, // Tree
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
	{ "network/ssl/certificates", "network/tls/certificate_bundle_override" },
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
	{ "rendering/quality/shadow_atlas/quadrant_0_subdiv", "rendering/lights_and_shadows/shadow_atlas/quadrant_0_subdiv" },
	{ "rendering/quality/shadow_atlas/quadrant_1_subdiv", "rendering/lights_and_shadows/shadow_atlas/quadrant_1_subdiv" },
	{ "rendering/quality/shadow_atlas/quadrant_2_subdiv", "rendering/lights_and_shadows/shadow_atlas/quadrant_2_subdiv" },
	{ "rendering/quality/shadow_atlas/quadrant_3_subdiv", "rendering/lights_and_shadows/shadow_atlas/quadrant_3_subdiv" },
	{ "rendering/quality/shadow_atlas/size", "rendering/lights_and_shadows/shadow_atlas/size" },
	{ "rendering/quality/shadow_atlas/size.mobile", "rendering/lights_and_shadows/shadow_atlas/size.mobile" },
	{ "rendering/vram_compression/import_bptc", "rendering/textures/vram_compression/import_bptc" },
	{ "rendering/vram_compression/import_etc", "rendering/textures/vram_compression/import_etc" },
	{ "rendering/vram_compression/import_etc2", "rendering/textures/vram_compression/import_etc2" },
	{ "rendering/vram_compression/import_pvrtc", "rendering/textures/vram_compression/import_pvrtc" },
	{ "rendering/vram_compression/import_s3tc", "rendering/textures/vram_compression/import_s3tc" },
	{ "window/size/width", "window/size/viewport_width" },
	{ "window/size/height", "window/size/viewport_height" },
	{ "window/size/test_width", "window/size/window_width_override" },
	{ "window/size/test_height", "window/size/window_height_override" },

	{ nullptr, nullptr },
};

static const char *input_map_renames[][2] = {
	{ ",\"alt\":", ",\"alt_pressed\":" },
	{ ",\"shift\":", ",\"shift_pressed\":" },
	{ ",\"control\":", ",\"ctrl_pressed\":" },
	{ ",\"meta\":", ",\"meta_pressed\":" },
	{ ",\"scancode\":", ",\"keycode\":" },
	{ ",\"physical_scancode\":", ",\"physical_keycode\":" },
	{ ",\"doubleclick\":", ",\"double_click\":" },

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
	{ "CAMERA_MATRIX", "INV_VIEW_MATRIX" },
	{ "INV_CAMERA_MATRIX", "VIEW_MATRIX" },
	{ "NORMALMAP", "NORMAL_MAP" },
	{ "NORMALMAP_DEPTH", "NORMAL_MAP_DEPTH" },
	{ "TRANSMISSION", "BACKLIGHT" },
	{ "WORLD_MATRIX", "MODEL_MATRIX" },
	{ "depth_draw_alpha_prepass", "depth_draw_opaque" },
	{ "hint_albedo", "source_color" },
	{ "hint_aniso", "hint_anisotropy" },
	{ "hint_black", "hint_default_black" },
	{ "hint_black_albedo", "hint_default_black" },
	{ "hint_color", "source_color" },
	{ "hint_white", "hint_default_white" },
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
	// { "NativeScript","GDExtension"}, ??
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
	{ "ParticlesMaterial", "ParticleProcessMaterial" },
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
	{ "Position2D", "Marker2D" },
	{ "Position3D", "Marker3D" },
	{ "ProceduralSky", "Sky" },
	{ "RayCast", "RayCast3D" },
	{ "RayShape", "SeparationRayShape3D" },
	{ "RayShape2D", "SeparationRayShape2D" },
	{ "Reference", "RefCounted" }, // Be careful, this will be used everywhere
	{ "RemoteTransform", "RemoteTransform3D" },
	{ "ResourceInteractiveLoader", "ResourceLoader" },
	{ "RigidBody", "RigidBody3D" },
	{ "SceneTreeTween", "Tween" },
	{ "Shape", "Shape3D" }, // Be careful, this will be used everywhere
	{ "ShortCut", "Shortcut" },
	{ "Skeleton", "Skeleton3D" },
	{ "SkeletonIK", "SkeletonIK3D" },
	{ "SliderJoint", "SliderJoint3D" },
	{ "SoftBody", "SoftBody3D" },
	{ "Spatial", "Node3D" },
	{ "SpatialGizmo", "Node3DGizmo" },
	{ "SpatialMaterial", "StandardMaterial3D" },
	{ "SphereShape", "SphereShape3D" },
	{ "SpotLight", "SpotLight3D" },
	{ "SpringArm", "SpringArm3D" },
	{ "Sprite", "Sprite2D" },
	{ "StaticBody", "StaticBody3D" },
	{ "StreamCubemap", "CompressedCubemap" },
	{ "StreamCubemapArray", "CompressedCubemapArray" },
	{ "StreamPeerGDNative", "StreamPeerExtension" },
	{ "StreamPeerSSL", "StreamPeerTLS" },
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
	{ "VisualShaderNodeScalarClamp", "VisualShaderNodeClamp" },
	{ "VisualShaderNodeScalarConstant", "VisualShaderNodeFloatConstant" },
	{ "VisualShaderNodeScalarFunc", "VisualShaderNodeFloatFunc" },
	{ "VisualShaderNodeScalarInterp", "VisualShaderNodeMix" },
	{ "VisualShaderNodeScalarOp", "VisualShaderNodeFloatOp" },
	{ "VisualShaderNodeScalarSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeScalarSwitch", "VisualShaderNodeSwitch" },
	{ "VisualShaderNodeScalarTransformMult", "VisualShaderNodeTransformOp" },
	{ "VisualShaderNodeTransformMult", "VisualShaderNode" },
	{ "VisualShaderNodeVectorClamp", "VisualShaderNodeClamp" },
	{ "VisualShaderNodeVectorInterp", "VisualShaderNodeMix" },
	{ "VisualShaderNodeVectorScalarMix", "VisualShaderNodeMix" },
	{ "VisualShaderNodeVectorScalarSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeVectorScalarStep", "VisualShaderNodeStep" },
	{ "VisualShaderNodeVectorSmoothStep", "VisualShaderNodeSmoothStep" },
	{ "VisualShaderNodeBooleanUniform", "VisualShaderNodeBooleanParameter" },
	{ "VisualShaderNodeColorUniform", "VisualShaderNodeColorParameter" },
	{ "VisualShaderNodeScalarUniform", "VisualShaderNodeFloatParameter" },
	{ "VisualShaderNodeCubemapUniform", "VisualShaderNodeCubemapParameter" },
	{ "VisualShaderNodeTextureUniform", "VisualShaderNodeTexture2DParameter" },
	{ "VisualShaderNodeTextureUniformTriplanar", "VisualShaderNodeTextureParameterTriplanar" },
	{ "VisualShaderNodeTransformUniform", "VisualShaderNodeTransformParameter" },
	{ "VisualShaderNodeVec3Uniform", "VisualShaderNodeVec3Parameter" },
	{ "VisualShaderNodeUniform", "VisualShaderNodeParameter" },
	{ "VisualShaderNodeUniformRef", "VisualShaderNodeParameterRef" },
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

static const char *color_renames[][2] = {
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

// Find "OS.set_property(x)", capturing x into $1.
static String make_regex_gds_os_property_set(String name_set) {
	return String("\\bOS\\.") + name_set + "\\s*\\((.*)\\)";
}
// Find "OS.property = x", capturing x into $1 or $2.
static String make_regex_gds_os_property_assign(String name) {
	return String("\\bOS\\.") + name + "\\s*=\\s*([^#]+)";
}
// Find "OS.property" OR "OS.get_property()" / "OS.is_property()".
static String make_regex_gds_os_property_get(String name, String get) {
	return String("\\bOS\\.(") + get + "_)?" + name + "(\\s*\\(\\s*\\))?";
}

class ProjectConverter3To4::RegExContainer {
public:
	// Custom GDScript.
	RegEx reg_is_empty = RegEx("\\bempty\\(");
	RegEx reg_super = RegEx("([\t ])\\.([a-zA-Z_])");
	RegEx reg_json_to = RegEx("\\bto_json\\b");
	RegEx reg_json_parse = RegEx("([\t ]{0,})([^\n]+)parse_json\\(([^\n]+)");
	RegEx reg_json_non_new = RegEx("([\t ]{0,})([^\n]+)JSON\\.parse\\(([^\n]+)");
	RegEx reg_json_print = RegEx("\\bJSON\\b\\.print\\(");
	RegEx reg_export = RegEx("export\\(([a-zA-Z0-9_]+)\\)[ ]+var[ ]+([a-zA-Z0-9_]+)");
	RegEx reg_export_advanced = RegEx("export\\(([^)^\n]+)\\)[ ]+var[ ]+([a-zA-Z0-9_]+)([^\n]+)");
	RegEx reg_setget_setget = RegEx("var[ ]+([a-zA-Z0-9_]+)([^\n]+)setget[ \t]+([a-zA-Z0-9_]+)[ \t]*,[ \t]*([a-zA-Z0-9_]+)");
	RegEx reg_setget_set = RegEx("var[ ]+([a-zA-Z0-9_]+)([^\n]+)setget[ \t]+([a-zA-Z0-9_]+)[ \t]*[,]*[^a-z^A-Z^0-9^_]*$");
	RegEx reg_setget_get = RegEx("var[ ]+([a-zA-Z0-9_]+)([^\n]+)setget[ \t]+,[ \t]*([a-zA-Z0-9_]+)[ \t]*$");
	RegEx reg_join = RegEx("([\\(\\)a-zA-Z0-9_]+)\\.join\\(([^\n^\\)]+)\\)");
	RegEx reg_image_lock = RegEx("([a-zA-Z0-9_\\.]+)\\.lock\\(\\)");
	RegEx reg_image_unlock = RegEx("([a-zA-Z0-9_\\.]+)\\.unlock\\(\\)");
	RegEx reg_instantiate = RegEx("\\.instance\\(([^\\)]*)\\)");
	// Simple OS properties with getters/setters.
	RegEx reg_os_current_screen = RegEx("\\bOS\\.(set_|get_)?current_screen\\b");
	RegEx reg_os_min_window_size = RegEx("\\bOS\\.(set_|get_)?min_window_size\\b");
	RegEx reg_os_max_window_size = RegEx("\\bOS\\.(set_|get_)?max_window_size\\b");
	RegEx reg_os_window_position = RegEx("\\bOS\\.(set_|get_)?window_position\\b");
	RegEx reg_os_window_size = RegEx("\\bOS\\.(set_|get_)?window_size\\b");
	RegEx reg_os_getset_screen_orient = RegEx("\\bOS\\.(s|g)et_screen_orientation\\b");
	// OS property getters/setters for non trivial replacements.
	RegEx reg_os_set_window_resizable = RegEx(make_regex_gds_os_property_set("set_window_resizable"));
	RegEx reg_os_assign_window_resizable = RegEx(make_regex_gds_os_property_assign("window_resizable"));
	RegEx reg_os_is_window_resizable = RegEx(make_regex_gds_os_property_get("window_resizable", "is"));
	RegEx reg_os_set_fullscreen = RegEx(make_regex_gds_os_property_set("set_window_fullscreen"));
	RegEx reg_os_assign_fullscreen = RegEx(make_regex_gds_os_property_assign("window_fullscreen"));
	RegEx reg_os_is_fullscreen = RegEx(make_regex_gds_os_property_get("window_fullscreen", "is"));
	RegEx reg_os_set_maximized = RegEx(make_regex_gds_os_property_set("set_window_maximized"));
	RegEx reg_os_assign_maximized = RegEx(make_regex_gds_os_property_assign("window_maximized"));
	RegEx reg_os_is_maximized = RegEx(make_regex_gds_os_property_get("window_maximized", "is"));
	RegEx reg_os_set_minimized = RegEx(make_regex_gds_os_property_set("set_window_minimized"));
	RegEx reg_os_assign_minimized = RegEx(make_regex_gds_os_property_assign("window_minimized"));
	RegEx reg_os_is_minimized = RegEx(make_regex_gds_os_property_get("window_minimized", "is"));
	RegEx reg_os_set_vsync = RegEx(make_regex_gds_os_property_set("set_use_vsync"));
	RegEx reg_os_assign_vsync = RegEx(make_regex_gds_os_property_assign("vsync_enabled"));
	RegEx reg_os_is_vsync = RegEx(make_regex_gds_os_property_get("vsync_enabled", "is"));
	// OS properties specific cases & specific replacements.
	RegEx reg_os_assign_screen_orient = RegEx("^(\\s*)OS\\.screen_orientation\\s*=\\s*([^#]+)"); // $1 - indent, $2 - value
	RegEx reg_os_set_always_on_top = RegEx(make_regex_gds_os_property_set("set_window_always_on_top"));
	RegEx reg_os_is_always_on_top = RegEx("\\bOS\\.is_window_always_on_top\\s*\\(.*\\)");
	RegEx reg_os_set_borderless = RegEx(make_regex_gds_os_property_set("set_borderless_window"));
	RegEx reg_os_get_borderless = RegEx("\\bOS\\.get_borderless_window\\s*\\(\\s*\\)");
	RegEx reg_os_screen_orient_enum = RegEx("\\bOS\\.SCREEN_ORIENTATION_(\\w+)\\b"); // $1 - constant suffix

	// GDScript keywords.
	RegEx keyword_gdscript_tool = RegEx("^tool");
	RegEx keyword_gdscript_export_single = RegEx("^export");
	RegEx keyword_gdscript_export_mutli = RegEx("([\t]+)export\\b");
	RegEx keyword_gdscript_onready = RegEx("^onready");
	RegEx keyword_gdscript_remote = RegEx("^remote func");
	RegEx keyword_gdscript_remotesync = RegEx("^remotesync func");
	RegEx keyword_gdscript_sync = RegEx("^sync func");
	RegEx keyword_gdscript_slave = RegEx("^slave func");
	RegEx keyword_gdscript_puppet = RegEx("^puppet func");
	RegEx keyword_gdscript_puppetsync = RegEx("^puppetsync func");
	RegEx keyword_gdscript_master = RegEx("^master func");
	RegEx keyword_gdscript_mastersync = RegEx("^mastersync func");

	// CSharp keywords.
	RegEx keyword_csharp_remote = RegEx("\\[Remote(Attribute)?(\\(\\))?\\]");
	RegEx keyword_csharp_remotesync = RegEx("\\[(Remote)?Sync(Attribute)?(\\(\\))?\\]");
	RegEx keyword_csharp_puppet = RegEx("\\[(Puppet|Slave)(Attribute)?(\\(\\))?\\]");
	RegEx keyword_csharp_puppetsync = RegEx("\\[PuppetSync(Attribute)?(\\(\\))?\\]");
	RegEx keyword_csharp_master = RegEx("\\[Master(Attribute)?(\\(\\))?\\]");
	RegEx keyword_csharp_mastersync = RegEx("\\[MasterSync(Attribute)?(\\(\\))?\\]");

	// Colors.
	LocalVector<RegEx *> color_regexes;
	LocalVector<String> color_renamed;

	// Classes.
	LocalVector<RegEx *> class_tscn_regexes;
	LocalVector<RegEx *> class_gd_regexes;
	LocalVector<RegEx *> class_shader_regexes;

	LocalVector<RegEx *> class_regexes;

	RegEx class_temp_tscn = RegEx("\\bTEMP_RENAMED_CLASS.tscn\\b");
	RegEx class_temp_gd = RegEx("\\bTEMP_RENAMED_CLASS.gd\\b");
	RegEx class_temp_shader = RegEx("\\bTEMP_RENAMED_CLASS.shader\\b");

	LocalVector<String> class_temp_tscn_renames;
	LocalVector<String> class_temp_gd_renames;
	LocalVector<String> class_temp_shader_renames;

	// Common.
	LocalVector<RegEx *> enum_regexes;
	LocalVector<RegEx *> gdscript_function_regexes;
	LocalVector<RegEx *> project_settings_regexes;
	LocalVector<RegEx *> input_map_regexes;
	LocalVector<RegEx *> gdscript_properties_regexes;
	LocalVector<RegEx *> gdscript_signals_regexes;
	LocalVector<RegEx *> shaders_regexes;
	LocalVector<RegEx *> builtin_types_regexes;
	LocalVector<RegEx *> csharp_function_regexes;
	LocalVector<RegEx *> csharp_properties_regexes;
	LocalVector<RegEx *> csharp_signal_regexes;

	RegExContainer() {
		// Common.
		{
			// Enum.
			for (unsigned int current_index = 0; enum_renames[current_index][0]; current_index++) {
				enum_regexes.push_back(memnew(RegEx(String("\\b") + enum_renames[current_index][0] + "\\b")));
			}
			// GDScript functions.
			for (unsigned int current_index = 0; gdscript_function_renames[current_index][0]; current_index++) {
				gdscript_function_regexes.push_back(memnew(RegEx(String("\\b") + gdscript_function_renames[current_index][0] + "\\b")));
			}
			// Project Settings.
			for (unsigned int current_index = 0; project_settings_renames[current_index][0]; current_index++) {
				project_settings_regexes.push_back(memnew(RegEx(String("\\b") + project_settings_renames[current_index][0] + "\\b")));
			}
			// Input Map.
			for (unsigned int current_index = 0; input_map_renames[current_index][0]; current_index++) {
				input_map_regexes.push_back(memnew(RegEx(String("\\b") + input_map_renames[current_index][0] + "\\b")));
			}
			// GDScript properties.
			for (unsigned int current_index = 0; gdscript_properties_renames[current_index][0]; current_index++) {
				gdscript_properties_regexes.push_back(memnew(RegEx(String("\\b") + gdscript_properties_renames[current_index][0] + "\\b")));
			}
			// GDScript Signals.
			for (unsigned int current_index = 0; gdscript_signals_renames[current_index][0]; current_index++) {
				gdscript_signals_regexes.push_back(memnew(RegEx(String("\\b") + gdscript_signals_renames[current_index][0] + "\\b")));
			}
			// Shaders.
			for (unsigned int current_index = 0; shaders_renames[current_index][0]; current_index++) {
				shaders_regexes.push_back(memnew(RegEx(String("\\b") + shaders_renames[current_index][0] + "\\b")));
			}
			// Builtin types.
			for (unsigned int current_index = 0; builtin_types_renames[current_index][0]; current_index++) {
				builtin_types_regexes.push_back(memnew(RegEx(String("\\b") + builtin_types_renames[current_index][0] + "\\b")));
			}
			// CSharp function renames.
			for (unsigned int current_index = 0; csharp_function_renames[current_index][0]; current_index++) {
				csharp_function_regexes.push_back(memnew(RegEx(String("\\b") + csharp_function_renames[current_index][0] + "\\b")));
			}
			// CSharp properties renames.
			for (unsigned int current_index = 0; csharp_properties_renames[current_index][0]; current_index++) {
				csharp_properties_regexes.push_back(memnew(RegEx(String("\\b") + csharp_properties_renames[current_index][0] + "\\b")));
			}
			// CSharp signals renames.
			for (unsigned int current_index = 0; csharp_signals_renames[current_index][0]; current_index++) {
				csharp_signal_regexes.push_back(memnew(RegEx(String("\\b") + csharp_signals_renames[current_index][0] + "\\b")));
			}
		}

		// Colors.
		{
			for (unsigned int current_index = 0; color_renames[current_index][0]; current_index++) {
				color_regexes.push_back(memnew(RegEx(String("\\bColor.") + color_renames[current_index][0] + "\\b")));
				color_renamed.push_back(String("Color.") + color_renames[current_index][1]);
			}
		}
		// Classes.
		{
			for (unsigned int current_index = 0; class_renames[current_index][0]; current_index++) {
				class_tscn_regexes.push_back(memnew(RegEx(String("\\b") + class_renames[current_index][0] + ".tscn\\b")));
				class_gd_regexes.push_back(memnew(RegEx(String("\\b") + class_renames[current_index][0] + ".gd\\b")));
				class_shader_regexes.push_back(memnew(RegEx(String("\\b") + class_renames[current_index][0] + ".shader\\b")));

				class_regexes.push_back(memnew(RegEx(String("\\b") + class_renames[current_index][0] + "\\b")));

				class_temp_tscn_renames.push_back(String(class_renames[current_index][0]) + ".tscn");
				class_temp_gd_renames.push_back(String(class_renames[current_index][0]) + ".gd");
				class_temp_shader_renames.push_back(String(class_renames[current_index][0]) + ".shader");
			}
		}
	}
	~RegExContainer() {
		for (RegEx *regex : color_regexes) {
			memdelete(regex);
		}
		for (unsigned int i = 0; i < class_tscn_regexes.size(); i++) {
			memdelete(class_tscn_regexes[i]);
			memdelete(class_gd_regexes[i]);
			memdelete(class_shader_regexes[i]);
			memdelete(class_regexes[i]);
		}
		for (RegEx *regex : enum_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : gdscript_function_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : project_settings_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : input_map_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : gdscript_properties_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : gdscript_signals_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : shaders_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : builtin_types_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : csharp_function_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : csharp_properties_regexes) {
			memdelete(regex);
		}
		for (RegEx *regex : csharp_signal_regexes) {
			memdelete(regex);
		}
	}
};

ProjectConverter3To4::ProjectConverter3To4(int p_maximum_file_size_kb, int p_maximum_line_length) {
	maximum_file_size = p_maximum_file_size_kb * 1024;
	maximum_line_length = p_maximum_line_length;
}

// Function responsible for converting project.
int ProjectConverter3To4::convert() {
	print_line("Starting conversion.");
	uint64_t conversion_start_time = Time::get_singleton()->get_ticks_msec();

	RegExContainer reg_container = RegExContainer();

	int cached_maximum_line_length = maximum_line_length;
	maximum_line_length = 10000; // Use only for tests bigger value, to not break them.

	ERR_FAIL_COND_V_MSG(!test_array_names(), ERROR_CODE, "Cannot start converting due to problems with data in arrays.");
	ERR_FAIL_COND_V_MSG(!test_conversion(reg_container), ERROR_CODE, "Cannot start converting due to problems with converting arrays.");

	maximum_line_length = cached_maximum_line_length;

	// Checking if folder contains valid Godot 3 project.
	// Project should not be converted more than once.
	{
		String converter_text = "; Project was converted by built-in tool to Godot 4.0";

		ERR_FAIL_COND_V_MSG(!FileAccess::exists("project.godot"), ERROR_CODE, "Current working directory doesn't contain a \"project.godot\" file for a Godot 3 project.");

		Error err = OK;
		String project_godot_content = FileAccess::get_file_as_string("project.godot", &err);

		ERR_FAIL_COND_V_MSG(err != OK, ERROR_CODE, "Unable to read \"project.godot\".");
		ERR_FAIL_COND_V_MSG(project_godot_content.contains(converter_text), ERROR_CODE, "Project was already converted with this tool.");

		Ref<FileAccess> file = FileAccess::open("project.godot", FileAccess::WRITE);
		ERR_FAIL_COND_V_MSG(file.is_null(), ERROR_CODE, "Unable to open \"project.godot\".");

		file->store_string(converter_text + "\n" + project_godot_content);
	}

	Vector<String> collected_files = check_for_files();

	uint32_t converted_files = 0;

	// Check file by file.
	for (int i = 0; i < collected_files.size(); i++) {
		String file_name = collected_files[i];
		Vector<String> lines;
		uint32_t ignored_lines = 0;
		{
			Ref<FileAccess> file = FileAccess::open(file_name, FileAccess::READ);
			ERR_CONTINUE_MSG(file.is_null(), vformat("Unable to read content of \"%s\".", file_name));
			while (!file->eof_reached()) {
				String line = file->get_line();
				lines.append(line);
			}
		}
		String file_content_before = collect_string_from_vector(lines);
		uint64_t hash_before = file_content_before.hash();
		uint64_t file_size = file_content_before.size();
		print_line(vformat("Trying to convert\t%d/%d file - \"%s\" with size - %d KB", i + 1, collected_files.size(), file_name.trim_prefix("res://"), file_size / 1024));

		Vector<String> reason;
		bool is_ignored = false;
		uint64_t start_time = Time::get_singleton()->get_ticks_msec();

		if (file_name.ends_with(".shader")) {
			DirAccess::remove_file_or_error(file_name.trim_prefix("res://"));
			file_name = file_name.replace(".shader", ".gdshader");
		}

		if (file_size < uint64_t(maximum_file_size)) {
			// ".tscn" must work exactly the same as ".gd" files because they may contain built-in Scripts.
			if (file_name.ends_with(".gd")) {
				rename_classes(lines, reg_container); // Using only specialized function.

				rename_common(enum_renames, reg_container.enum_regexes, lines);
				rename_colors(lines, reg_container); // Require to additional rename.

				rename_common(gdscript_function_renames, reg_container.gdscript_function_regexes, lines);
				rename_gdscript_functions(lines, reg_container, false); // Require to additional rename.

				rename_common(project_settings_renames, reg_container.project_settings_regexes, lines);
				rename_gdscript_keywords(lines, reg_container);
				rename_common(gdscript_properties_renames, reg_container.gdscript_properties_regexes, lines);
				rename_common(gdscript_signals_renames, reg_container.gdscript_signals_regexes, lines);
				rename_common(shaders_renames, reg_container.shaders_regexes, lines);
				rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines);

				custom_rename(lines, "\\.shader", ".gdshader");
			} else if (file_name.ends_with(".tscn")) {
				rename_classes(lines, reg_container); // Using only specialized function.

				rename_common(enum_renames, reg_container.enum_regexes, lines);
				rename_colors(lines, reg_container); // Require to do additional renames.

				rename_common(gdscript_function_renames, reg_container.gdscript_function_regexes, lines);
				rename_gdscript_functions(lines, reg_container, true); // Require to do additional renames.

				rename_common(project_settings_renames, reg_container.project_settings_regexes, lines);
				rename_gdscript_keywords(lines, reg_container);
				rename_common(gdscript_properties_renames, reg_container.gdscript_properties_regexes, lines);
				rename_common(gdscript_signals_renames, reg_container.gdscript_signals_regexes, lines);
				rename_common(shaders_renames, reg_container.shaders_regexes, lines);
				rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines);

				custom_rename(lines, "\\.shader", ".gdshader");
			} else if (file_name.ends_with(".cs")) { // TODO, C# should use different methods.
				rename_classes(lines, reg_container); // Using only specialized function.
				rename_common(csharp_function_renames, reg_container.csharp_function_regexes, lines);
				rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines);
				rename_common(csharp_properties_renames, reg_container.csharp_properties_regexes, lines);
				rename_common(csharp_signals_renames, reg_container.csharp_signal_regexes, lines);
				rename_csharp_functions(lines, reg_container);
				rename_csharp_attributes(lines, reg_container);
				custom_rename(lines, "public class ", "public partial class ");
			} else if (file_name.ends_with(".gdshader") || file_name.ends_with(".shader")) {
				rename_common(shaders_renames, reg_container.shaders_regexes, lines);
			} else if (file_name.ends_with("tres")) {
				rename_classes(lines, reg_container); // Using only specialized function.

				rename_common(shaders_renames, reg_container.shaders_regexes, lines);
				rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines);

				custom_rename(lines, "\\.shader", ".gdshader");
			} else if (file_name.ends_with("project.godot")) {
				rename_common(project_settings_renames, reg_container.project_settings_regexes, lines);
				rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines);
				rename_common(input_map_renames, reg_container.input_map_regexes, lines);
			} else if (file_name.ends_with(".csproj")) {
				// TODO
			} else {
				ERR_PRINT(file_name + " is not supported!");
				continue;
			}

			for (String &line : lines) {
				if (uint64_t(line.length()) > maximum_line_length) {
					ignored_lines += 1;
				}
			}
		} else {
			reason.append(vformat("    ERROR: File has exceeded the maximum size allowed - %d KB", maximum_file_size / 1024));
			is_ignored = true;
		}

		uint64_t end_time = Time::get_singleton()->get_ticks_msec();
		if (is_ignored) {
			String end_message = vformat("    Checking file took %d ms.", end_time - start_time);
			print_line(end_message);
		} else {
			String file_content_after = collect_string_from_vector(lines);
			uint64_t hash_after = file_content_after.hash64();
			// Don't need to save file without any changes.
			// Save if this is a shader, because it was renamed.
			if (hash_before != hash_after || file_name.ends_with(".gdshader")) {
				converted_files++;

				Ref<FileAccess> file = FileAccess::open(file_name, FileAccess::WRITE);
				ERR_CONTINUE_MSG(file.is_null(), vformat("Unable to apply changes to \"%s\", no writing access.", file_name));
				file->store_string(file_content_after);
				reason.append(vformat("    File was changed, conversion took %d ms.", end_time - start_time));
			} else {
				reason.append(vformat("    File was left unchanged, checking took %d ms.", end_time - start_time));
			}
			if (ignored_lines != 0) {
				reason.append(vformat("    Ignored %d lines, because their length exceeds maximum allowed characters - %d.", ignored_lines, maximum_line_length));
			}
		}
		for (int k = 0; k < reason.size(); k++) {
			print_line(reason[k]);
		}
	}
	print_line(vformat("Conversion ended - all files(%d), converted files: (%d), not converted files: (%d).", collected_files.size(), converted_files, collected_files.size() - converted_files));
	uint64_t conversion_end_time = Time::get_singleton()->get_ticks_msec();
	print_line(vformat("Conversion of all files took %10.3f seconds.", (conversion_end_time - conversion_start_time) / 1000.0));
	return 0;
};

// Function responsible for validating project conversion.
int ProjectConverter3To4::validate_conversion() {
	print_line("Starting checking if project conversion can be done.");
	uint64_t conversion_start_time = Time::get_singleton()->get_ticks_msec();

	RegExContainer reg_container = RegExContainer();

	int cached_maximum_line_length = maximum_line_length;
	maximum_line_length = 10000; // To avoid breaking the tests, only use this for the their larger value.

	ERR_FAIL_COND_V_MSG(!test_array_names(), ERROR_CODE, "Cannot start converting due to problems with data in arrays.");
	ERR_FAIL_COND_V_MSG(!test_conversion(reg_container), ERROR_CODE, "Cannot start converting due to problems with converting arrays.");

	maximum_line_length = cached_maximum_line_length;

	// Checking if folder contains valid Godot 3 project.
	// Project should not be converted more than once.
	{
		String conventer_text = "; Project was converted by built-in tool to Godot 4.0";

		ERR_FAIL_COND_V_MSG(!FileAccess::exists("project.godot"), ERROR_CODE, "Current directory doesn't contains any Godot 3 project");

		Error err = OK;
		String project_godot_content = FileAccess::get_file_as_string("project.godot", &err);

		ERR_FAIL_COND_V_MSG(err != OK, ERROR_CODE, "Failed to read content of \"project.godot\" file.");
		ERR_FAIL_COND_V_MSG(project_godot_content.contains(conventer_text), ERROR_CODE, "Project already was converted with this tool.");
	}

	Vector<String> collected_files = check_for_files();

	uint32_t converted_files = 0;

	// Check file by file.
	for (int i = 0; i < collected_files.size(); i++) {
		String file_name = collected_files[i];
		Vector<String> lines;
		uint32_t ignored_lines = 0;
		uint64_t file_size = 0;
		{
			Ref<FileAccess> file = FileAccess::open(file_name, FileAccess::READ);
			ERR_CONTINUE_MSG(file.is_null(), vformat("Unable to read content of \"%s\".", file_name));
			while (!file->eof_reached()) {
				String line = file->get_line();
				file_size += line.size();
				lines.append(line);
			}
		}
		print_line(vformat("Checking for conversion - %d/%d file - \"%s\" with size - %d KB", i + 1, collected_files.size(), file_name.trim_prefix("res://"), file_size / 1024));

		Vector<String> changed_elements;
		Vector<String> reason;
		bool is_ignored = false;
		uint64_t start_time = Time::get_singleton()->get_ticks_msec();

		if (file_name.ends_with(".shader")) {
			reason.append("\tFile extension will be renamed from \"shader\" to \"gdshader\".");
		}

		if (file_size < uint64_t(maximum_file_size)) {
			if (file_name.ends_with(".gd")) {
				changed_elements.append_array(check_for_rename_classes(lines, reg_container));

				changed_elements.append_array(check_for_rename_common(enum_renames, reg_container.enum_regexes, lines));
				changed_elements.append_array(check_for_rename_colors(lines, reg_container));

				changed_elements.append_array(check_for_rename_common(gdscript_function_renames, reg_container.gdscript_function_regexes, lines));
				changed_elements.append_array(check_for_rename_gdscript_functions(lines, reg_container, false));

				changed_elements.append_array(check_for_rename_common(project_settings_renames, reg_container.project_settings_regexes, lines));
				changed_elements.append_array(check_for_rename_gdscript_keywords(lines, reg_container));
				changed_elements.append_array(check_for_rename_common(gdscript_properties_renames, reg_container.gdscript_properties_regexes, lines));
				changed_elements.append_array(check_for_rename_common(gdscript_signals_renames, reg_container.gdscript_signals_regexes, lines));
				changed_elements.append_array(check_for_rename_common(shaders_renames, reg_container.shaders_regexes, lines));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines));

				changed_elements.append_array(check_for_custom_rename(lines, "\\.shader", ".gdshader"));
			} else if (file_name.ends_with(".tscn")) {
				changed_elements.append_array(check_for_rename_classes(lines, reg_container));

				changed_elements.append_array(check_for_rename_common(enum_renames, reg_container.enum_regexes, lines));
				changed_elements.append_array(check_for_rename_colors(lines, reg_container));

				changed_elements.append_array(check_for_rename_common(gdscript_function_renames, reg_container.gdscript_function_regexes, lines));
				changed_elements.append_array(check_for_rename_gdscript_functions(lines, reg_container, true));

				changed_elements.append_array(check_for_rename_common(project_settings_renames, reg_container.project_settings_regexes, lines));
				changed_elements.append_array(check_for_rename_gdscript_keywords(lines, reg_container));
				changed_elements.append_array(check_for_rename_common(gdscript_properties_renames, reg_container.gdscript_properties_regexes, lines));
				changed_elements.append_array(check_for_rename_common(gdscript_signals_renames, reg_container.gdscript_signals_regexes, lines));
				changed_elements.append_array(check_for_rename_common(shaders_renames, reg_container.shaders_regexes, lines));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines));

				changed_elements.append_array(check_for_custom_rename(lines, "\\.shader", ".gdshader"));
			} else if (file_name.ends_with(".cs")) {
				changed_elements.append_array(check_for_rename_classes(lines, reg_container));
				changed_elements.append_array(check_for_rename_common(csharp_function_renames, reg_container.csharp_function_regexes, lines));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines));
				changed_elements.append_array(check_for_rename_common(csharp_properties_renames, reg_container.csharp_properties_regexes, lines));
				changed_elements.append_array(check_for_rename_common(csharp_signals_renames, reg_container.csharp_signal_regexes, lines));
				changed_elements.append_array(check_for_rename_csharp_functions(lines, reg_container));
				changed_elements.append_array(check_for_rename_csharp_attributes(lines, reg_container));
				changed_elements.append_array(check_for_custom_rename(lines, "public class ", "public partial class "));
			} else if (file_name.ends_with(".gdshader") || file_name.ends_with(".shader")) {
				changed_elements.append_array(check_for_rename_common(shaders_renames, reg_container.shaders_regexes, lines));
			} else if (file_name.ends_with("tres")) {
				changed_elements.append_array(check_for_rename_classes(lines, reg_container));

				changed_elements.append_array(check_for_rename_common(shaders_renames, reg_container.shaders_regexes, lines));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines));

				changed_elements.append_array(check_for_custom_rename(lines, "\\.shader", ".gdshader"));
			} else if (file_name.ends_with("project.godot")) {
				changed_elements.append_array(check_for_rename_common(project_settings_renames, reg_container.project_settings_regexes, lines));
				changed_elements.append_array(check_for_rename_common(builtin_types_renames, reg_container.builtin_types_regexes, lines));
				changed_elements.append_array(check_for_rename_common(input_map_renames, reg_container.input_map_regexes, lines));
			} else if (file_name.ends_with(".csproj")) {
				// TODO
			} else {
				ERR_PRINT(vformat("\"%s\", is not supported!", file_name));
				continue;
			}

			for (String &line : lines) {
				if (uint64_t(line.length()) > maximum_line_length) {
					ignored_lines += 1;
				}
			}
		} else {
			reason.append(vformat("\tERROR: File has exceeded the maximum size allowed  - %d KB.", maximum_file_size / 1024));
			is_ignored = true;
		}

		uint64_t end_time = Time::get_singleton()->get_ticks_msec();
		String end_message = vformat("    Checking file took %10.3f ms.", (end_time - start_time) / 1000.0);
		if (ignored_lines != 0) {
			end_message += vformat(" Ignored %d lines, because their length exceeds maximum allowed characters - %d.", ignored_lines, maximum_line_length);
		}
		print_line(end_message);

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

	print_line(vformat("Checking for valid conversion ended - all files(%d), files which would be converted(%d), files which would not be converted(%d).", collected_files.size(), converted_files, collected_files.size() - converted_files));
	uint64_t conversion_end_time = Time::get_singleton()->get_ticks_msec();
	print_line(vformat("Conversion of all files took %10.3f seconds.", (conversion_end_time - conversion_start_time) / 1000.0));
	return 0;
}

// Collect files which will be checked, excluding ".txt", ".mp4", ".wav" etc. files.
Vector<String> ProjectConverter3To4::check_for_files() {
	Vector<String> collected_files = Vector<String>();

	Vector<String> directories_to_check = Vector<String>();
	directories_to_check.push_back("res://");

	while (!directories_to_check.is_empty()) {
		String path = directories_to_check.get(directories_to_check.size() - 1); // Is there any pop_back function?
		directories_to_check.resize(directories_to_check.size() - 1); // Remove last element

		Ref<DirAccess> dir = DirAccess::open(path);
		if (dir.is_valid()) {
			dir->set_include_hidden(true);
			dir->list_dir_begin();
			String current_dir = dir->get_current_dir();
			String file_name = dir->_get_next();

			while (file_name != "") {
				if (file_name == ".git" || file_name == ".import" || file_name == ".godot") {
					file_name = dir->_get_next();
					continue;
				}
				if (dir->current_is_dir()) {
					directories_to_check.append(current_dir.path_join(file_name) + "/");
				} else {
					bool proper_extension = false;
					if (file_name.ends_with(".gd") || file_name.ends_with(".shader") || file_name.ends_with(".tscn") || file_name.ends_with(".tres") || file_name.ends_with(".godot") || file_name.ends_with(".cs") || file_name.ends_with(".csproj"))
						proper_extension = true;

					if (proper_extension) {
						collected_files.append(current_dir.path_join(file_name));
					}
				}
				file_name = dir->_get_next();
			}
		} else {
			print_verbose("Failed to open " + path);
		}
	}
	return collected_files;
}

// Test expected results of gdscript
bool ProjectConverter3To4::test_conversion_gdscript_builtin(String name, String expected, void (ProjectConverter3To4::*func)(Vector<String> &, const RegExContainer &, bool), String what, const RegExContainer &reg_container, bool builtin_script) {
	Vector<String> got = name.split("\n");
	(this->*func)(got, reg_container, builtin_script);
	String got_str = collect_string_from_vector(got);
	ERR_FAIL_COND_V_MSG(expected != got_str, false, vformat("Failed to convert %s \"%s\" to \"%s\", got instead \"%s\"", what, name, expected, got_str));

	return true;
}

bool ProjectConverter3To4::test_conversion_with_regex(String name, String expected, void (ProjectConverter3To4::*func)(Vector<String> &, const RegExContainer &), String what, const RegExContainer &reg_container) {
	Vector<String> got = name.split("\n");
	(this->*func)(got, reg_container);
	String got_str = collect_string_from_vector(got);
	ERR_FAIL_COND_V_MSG(expected != got_str, false, vformat("Failed to convert %s \"%s\" to \"%s\", got instead \"%s\"", what, name, expected, got_str));

	return true;
}

bool ProjectConverter3To4::test_conversion_basic(String name, String expected, const char *array[][2], LocalVector<RegEx *> &regex_cache, String what) {
	Vector<String> got = name.split("\n");
	rename_common(array, regex_cache, got);
	String got_str = collect_string_from_vector(got);
	ERR_FAIL_COND_V_MSG(expected != got_str, false, vformat("Failed to convert %s \"%s\" to \"%s\", got instead \"%s\"", what, name, expected, got_str));

	return true;
}

// Validate if conversions are proper.
bool ProjectConverter3To4::test_conversion(RegExContainer &reg_container) {
	bool valid = true;

	valid = valid && test_conversion_basic("TYPE_REAL", "TYPE_FLOAT", enum_renames, reg_container.enum_regexes, "enum");

	valid = valid && test_conversion_basic("can_instance", "can_instantiate", gdscript_function_renames, reg_container.gdscript_function_regexes, "gdscript function");

	valid = valid && test_conversion_basic("CanInstance", "CanInstantiate", csharp_function_renames, reg_container.csharp_function_regexes, "csharp function");

	valid = valid && test_conversion_basic("translation", "position", gdscript_properties_renames, reg_container.gdscript_properties_regexes, "gdscript property");

	valid = valid && test_conversion_basic("Translation", "Position", csharp_properties_renames, reg_container.csharp_properties_regexes, "csharp property");

	valid = valid && test_conversion_basic("NORMALMAP", "NORMAL_MAP", shaders_renames, reg_container.shaders_regexes, "shader");

	valid = valid && test_conversion_basic("text_entered", "text_submitted", gdscript_signals_renames, reg_container.gdscript_signals_regexes, "gdscript signal");

	valid = valid && test_conversion_basic("TextEntered", "TextSubmitted", csharp_signals_renames, reg_container.csharp_signal_regexes, "csharp signal");

	valid = valid && test_conversion_basic("audio/channel_disable_threshold_db", "audio/buses/channel_disable_threshold_db", project_settings_renames, reg_container.project_settings_regexes, "project setting");

	valid = valid && test_conversion_basic("\"device\":-1,\"alt\":false,\"shift\":false,\"control\":false,\"meta\":false,\"doubleclick\":false,\"scancode\":0,\"physical_scancode\":16777254,\"script\":null", "\"device\":-1,\"alt_pressed\":false,\"shift_pressed\":false,\"ctrl_pressed\":false,\"meta_pressed\":false,\"double_click\":false,\"keycode\":0,\"physical_keycode\":16777254,\"script\":null", input_map_renames, reg_container.input_map_regexes, "input map");

	valid = valid && test_conversion_basic("Transform", "Transform3D", builtin_types_renames, reg_container.builtin_types_regexes, "builtin type");

	// Custom Renames.

	valid = valid && test_conversion_with_regex("(Connect(A,B,C,D,E,F,G) != OK):", "(Connect(A,new Callable(B,C),D,E,F,G) != OK):", &ProjectConverter3To4::rename_csharp_functions, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("(Disconnect(A,B,C) != OK):", "(Disconnect(A,new Callable(B,C)) != OK):", &ProjectConverter3To4::rename_csharp_functions, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("(IsConnected(A,B,C) != OK):", "(IsConnected(A,new Callable(B,C)) != OK):", &ProjectConverter3To4::rename_csharp_functions, "custom rename", reg_container);

	valid = valid && test_conversion_with_regex("[Remote]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[RemoteSync]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[Sync]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[Slave]", "[RPC]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[Puppet]", "[RPC]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[PuppetSync]", "[RPC(CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[Master]", "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using Multiplayer.GetRemoteSenderId()\n[RPC]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);
	valid = valid && test_conversion_with_regex("[MasterSync]", "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using Multiplayer.GetRemoteSenderId()\n[RPC(CallLocal = true)]", &ProjectConverter3To4::rename_csharp_attributes, "custom rename csharp", reg_container);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.window_resizable: pass", "\tif (not get_window().unresizable): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tif OS.is_window_resizable(): pass", "\tif (not get_window().unresizable): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_window_resizable(Settings.resizable)", "\tget_window().unresizable = not (Settings.resizable)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.window_resizable = Settings.resizable", "\tget_window().unresizable = not (Settings.resizable)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.window_fullscreen: pass", "\tif ((get_window().mode == Window.MODE_EXCLUSIVE_FULLSCREEN) or (get_window().mode == Window.MODE_FULLSCREEN)): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tif OS.is_window_fullscreen(): pass", "\tif ((get_window().mode == Window.MODE_EXCLUSIVE_FULLSCREEN) or (get_window().mode == Window.MODE_FULLSCREEN)): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_window_fullscreen(Settings.fullscreen)", "\tget_window().mode = Window.MODE_EXCLUSIVE_FULLSCREEN if (Settings.fullscreen) else Window.MODE_WINDOWED", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.window_fullscreen = Settings.fullscreen", "\tget_window().mode = Window.MODE_EXCLUSIVE_FULLSCREEN if (Settings.fullscreen) else Window.MODE_WINDOWED", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.window_maximized: pass", "\tif (get_window().mode == Window.MODE_MAXIMIZED): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tif OS.is_window_maximized(): pass", "\tif (get_window().mode == Window.MODE_MAXIMIZED): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_window_maximized(Settings.maximized)", "\tget_window().mode = Window.MODE_MAXIMIZED if (Settings.maximized) else Window.MODE_WINDOWED", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.window_maximized = Settings.maximized", "\tget_window().mode = Window.MODE_MAXIMIZED if (Settings.maximized) else Window.MODE_WINDOWED", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.window_minimized: pass", "\tif (get_window().mode == Window.MODE_MINIMIZED): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tif OS.is_window_minimized(): pass", "\tif (get_window().mode == Window.MODE_MINIMIZED): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_window_minimized(Settings.minimized)", "\tget_window().mode = Window.MODE_MINIMIZED if (Settings.minimized) else Window.MODE_WINDOWED", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.window_minimized = Settings.minimized", "\tget_window().mode = Window.MODE_MINIMIZED if (Settings.minimized) else Window.MODE_WINDOWED", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.vsync_enabled: pass", "\tif (DisplayServer.window_get_vsync_mode() != DisplayServer.VSYNC_DISABLED): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tif OS.is_vsync_enabled(): pass", "\tif (DisplayServer.window_get_vsync_mode() != DisplayServer.VSYNC_DISABLED): pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_use_vsync(Settings.vsync)", "\tDisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_ENABLED if (Settings.vsync) else DisplayServer.VSYNC_DISABLED)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.vsync_enabled = Settings.vsync", "\tDisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_ENABLED if (Settings.vsync) else DisplayServer.VSYNC_DISABLED)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.screen_orientation = OS.SCREEN_ORIENTATION_VERTICAL: pass", "\tif DisplayServer.screen_get_orientation() = DisplayServer.SCREEN_VERTICAL: pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tif OS.get_screen_orientation() = OS.SCREEN_ORIENTATION_LANDSCAPE: pass", "\tif DisplayServer.screen_get_orientation() = DisplayServer.SCREEN_LANDSCAPE: pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_screen_orientation(Settings.orient)", "\tDisplayServer.screen_set_orientation(Settings.orient)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.screen_orientation = Settings.orient", "\tDisplayServer.screen_set_orientation(Settings.orient)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.is_window_always_on_top(): pass", "\tif get_window().always_on_top: pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_window_always_on_top(Settings.alwaystop)", "\tget_window().always_on_top = (Settings.alwaystop)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tif OS.get_borderless_window(): pass", "\tif get_window().borderless: pass", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tOS.set_borderless_window(Settings.borderless)", "\tget_window().borderless = (Settings.borderless)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\tvar aa = roman(r.move_and_slide( a, b, c, d, e, f )) # Roman", "\tr.set_velocity(a)\n\tr.set_up_direction(b)\n\tr.set_floor_stop_on_slope_enabled(c)\n\tr.set_max_slides(d)\n\tr.set_floor_max_angle(e)\n\t# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `f`\n\tr.move_and_slide()\n\tvar aa = roman(r.velocity) # Roman", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tmove_and_slide( a, b, c, d, e, f ) # Roman", "\tset_velocity(a)\n\tset_up_direction(b)\n\tset_floor_stop_on_slope_enabled(c)\n\tset_max_slides(d)\n\tset_floor_max_angle(e)\n\t# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `f`\n\tmove_and_slide() # Roman", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tvar aa = roman(r.move_and_slide_with_snap( a, g, b, c, d, e, f )) # Roman", "\tr.set_velocity(a)\n\t# TODOConverter40 looks that snap in Godot 4.0 is float, not vector like in Godot 3 - previous value `g`\n\tr.set_up_direction(b)\n\tr.set_floor_stop_on_slope_enabled(c)\n\tr.set_max_slides(d)\n\tr.set_floor_max_angle(e)\n\t# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `f`\n\tr.move_and_slide()\n\tvar aa = roman(r.velocity) # Roman", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tmove_and_slide_with_snap( a, g, b, c, d, e, f ) # Roman", "\tset_velocity(a)\n\t# TODOConverter40 looks that snap in Godot 4.0 is float, not vector like in Godot 3 - previous value `g`\n\tset_up_direction(b)\n\tset_floor_stop_on_slope_enabled(c)\n\tset_max_slides(d)\n\tset_floor_max_angle(e)\n\t# TODOConverter40 infinite_inertia were removed in Godot 4.0 - previous value `f`\n\tmove_and_slide() # Roman", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("list_dir_begin( a , b )", "list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("list_dir_begin( a )", "list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("list_dir_begin( )", "list_dir_begin() # TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("sort_custom( a , b )", "sort_custom(Callable(a,b))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("func c(var a, var b)", "func c(a, b)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("draw_line(1, 2, 3, 4, 5)", "draw_line(1,2,3,4)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("\timage.lock()", "\tfalse # image.lock() # TODOConverter40, Image no longer requires locking, `false` helps to not break one line if/else, so it can freely be removed", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\timage.unlock()", "\tfalse # image.unlock() # TODOConverter40, Image no longer requires locking, `false` helps to not break one line if/else, so it can freely be removed", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\troman.image.unlock()", "\tfalse # roman.image.unlock() # TODOConverter40, Image no longer requires locking, `false` helps to not break one line if/else, so it can freely be removed", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tmtx.lock()", "\tmtx.lock()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\tmutex.unlock()", "\tmutex.unlock()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_with_regex("extends CSGBox", "extends CSGBox3D", &ProjectConverter3To4::rename_classes, "classes", reg_container);
	valid = valid && test_conversion_with_regex("CSGBox", "CSGBox3D", &ProjectConverter3To4::rename_classes, "classes", reg_container);
	valid = valid && test_conversion_with_regex("Spatial", "Node3D", &ProjectConverter3To4::rename_classes, "classes", reg_container);
	valid = valid && test_conversion_with_regex("Spatial.tscn", "Spatial.tscn", &ProjectConverter3To4::rename_classes, "classes", reg_container);
	valid = valid && test_conversion_with_regex("Spatial.gd", "Spatial.gd", &ProjectConverter3To4::rename_classes, "classes", reg_container);
	valid = valid && test_conversion_with_regex("Spatial.shader", "Spatial.shader", &ProjectConverter3To4::rename_classes, "classes", reg_container);
	valid = valid && test_conversion_with_regex("Spatial.other", "Node3D.other", &ProjectConverter3To4::rename_classes, "classes", reg_container);

	valid = valid && test_conversion_with_regex("\nonready", "\n@onready", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("onready", "@onready", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex(" onready", " onready", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\nexport", "\n@export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\texport", "\t@export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\texport_dialog", "\texport_dialog", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("export", "@export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex(" export", " export", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("tool", "@tool", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n    tool", "\n    tool", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\ntool", "\n\n@tool", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\nremote func", "\n\n@rpc(\"any_peer\") func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\nremotesync func", "\n\n@rpc(\"any_peer\", \"call_local\") func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\nsync func", "\n\n@rpc(\"any_peer\", \"call_local\") func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\nslave func", "\n\n@rpc func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\npuppet func", "\n\n@rpc func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\npuppetsync func", "\n\n@rpc(\"call_local\") func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\nmaster func", "\n\nThe master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using get_multiplayer().get_remote_sender_id()\n@rpc func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);
	valid = valid && test_conversion_with_regex("\n\nmastersync func", "\n\nThe master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using get_multiplayer().get_remote_sender_id()\n@rpc(\"call_local\") func", &ProjectConverter3To4::rename_gdscript_keywords, "gdscript keyword", reg_container);

	valid = valid && test_conversion_gdscript_builtin("var size : Vector2 = Vector2() setget set_function , get_function", "var size : Vector2 = Vector2() : get = get_function, set = set_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("var size : Vector2 = Vector2() setget set_function , ", "var size : Vector2 = Vector2() : set = set_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("var size : Vector2 = Vector2() setget set_function", "var size : Vector2 = Vector2() : set = set_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("var size : Vector2 = Vector2() setget  , get_function", "var size : Vector2 = Vector2() : get = get_function", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("get_node(@", "get_node(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("yield(this, \"timeout\")", "await this.timeout", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("yield(this, \\\"timeout\\\")", "await this.timeout", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, true);

	valid = valid && test_conversion_gdscript_builtin(" Transform.xform(Vector3(a,b,c)) ", " Transform * Vector3(a,b,c) ", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin(" Transform.xform_inv(Vector3(a,b,c)) ", " Vector3(a,b,c) * Transform ", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("export(float) var lifetime = 3.0", "export var lifetime: float = 3.0", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("export(String, 'AnonymousPro', 'CourierPrime') var _font_name = 'AnonymousPro'", "export var _font_name = 'AnonymousPro' # (String, 'AnonymousPro', 'CourierPrime')", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false); // TODO, this is only a workaround
	valid = valid && test_conversion_gdscript_builtin("export(PackedScene) var mob_scene", "export var mob_scene: PackedScene", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("var d = parse_json(roman(sfs))", "var test_json_conv = JSON.new()\ntest_json_conv.parse(roman(sfs))\nvar d = test_json_conv.get_data()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("to_json( AA ) szon", "JSON.new().stringify( AA ) szon", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("s to_json", "s JSON.new().stringify", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("AF to_json2", "AF to_json2", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("var rr = JSON.parse(a)", "var test_json_conv = JSON.new()\ntest_json_conv.parse(a)\nvar rr = test_json_conv.get_data()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("empty()", "is_empty()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin(".empty", ".empty", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin(").roman(", ").roman(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\t.roman(", "\tsuper.roman(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin(" .roman(", " super.roman(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin(".1", ".1", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin(" .1", " .1", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("'.'", "'.'", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("'.a'", "'.a'", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("\t._input(_event)", "\tsuper._input(_event)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("(connect(A,B,C) != OK):", "(connect(A,Callable(B,C)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(connect(A,B,C,D) != OK):", "(connect(A,Callable(B,C).bind(D)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(connect(A,B,C,[D]) != OK):", "(connect(A,Callable(B,C).bind(D)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(connect(A,B,C,[D,E]) != OK):", "(connect(A,Callable(B,C).bind(D,E)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(connect(A,B,C,[D,E],F) != OK):", "(connect(A,Callable(B,C).bind(D,E),F) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(connect(A,B,C,D,E) != OK):", "(connect(A,Callable(B,C).bind(D),E) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("(start(A,B) != OK):", "(start(Callable(A,B)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("func start(A,B):", "func start(A,B):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(start(A,B,C,D,E,F,G) != OK):", "(start(Callable(A,B).bind(C),D,E,F,G) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("disconnect(A,B,C) != OK):", "disconnect(A,Callable(B,C)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("is_connected(A,B,C) != OK):", "is_connected(A,Callable(B,C)) != OK):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("is_connected(A,B,C))", "is_connected(A,Callable(B,C)))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("(tween_method(A,B,C,D,E).foo())", "(tween_method(Callable(A,B),C,D,E).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(tween_method(A,B,C,D,E,[F,G]).foo())", "(tween_method(Callable(A,B).bind(F,G),C,D,E).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(tween_callback(A,B).foo())", "(tween_callback(Callable(A,B)).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("(tween_callback(A,B,[C,D]).foo())", "(tween_callback(Callable(A,B).bind(C,D)).foo())", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("func _init(", "func _init(", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("func _init(p_x:int)->void:", "func _init(p_x:int):", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("q_PackedDataContainer._iter_init(variable1)", "q_PackedDataContainer._iter_init(variable1)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("assert(speed < 20, str(randi()%10))", "assert(speed < 20) #,str(randi()%10))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("assert(speed < 2)", "assert(speed < 2)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("assert(false, \"Missing type --\" + str(argument.type) + \"--, needs to be added to project\")", "assert(false) #,\"Missing type --\" + str(argument.type) + \"--, needs to be added to project\")", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("create_from_image(aa, bb)", "create_from_image(aa) #,bb", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("q_ImageTexture.create_from_image(variable1, variable2)", "q_ImageTexture.create_from_image(variable1) #,variable2", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("set_cell_item(a, b, c, d ,e) # AA", "set_cell_item( Vector3(a,b,c) ,d,e) # AA", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("set_cell_item(a, b)", "set_cell_item(a, b)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("get_cell_item_orientation(a, b,c)", "get_cell_item_orientation(Vector3i(a,b,c))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("get_cell_item(a, b,c)", "get_cell_item(Vector3i(a,b,c))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("map_to_world(a, b,c)", "map_to_local(Vector3i(a,b,c))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("PackedStringArray(req_godot).join('.')", "'.'.join(PackedStringArray(req_godot))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("=PackedStringArray(req_godot).join('.')", "='.'.join(PackedStringArray(req_godot))", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_gdscript_builtin("apply_force(position, impulse)", "apply_force(impulse, position)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("apply_impulse(position, impulse)", "apply_impulse(impulse, position)", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("draw_rect(a,b,c,d,e).abc", "draw_rect(a,b,c,d).abc# e) TODOGODOT4 Antialiasing argument is missing", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("get_focus_owner()", "get_viewport().gui_get_focus_owner()", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("button.pressed = 1", "button.button_pressed = 1", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("button.pressed=1", "button.button_pressed=1", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);
	valid = valid && test_conversion_gdscript_builtin("button.pressed SF", "button.pressed SF", &ProjectConverter3To4::rename_gdscript_functions, "custom rename", reg_container, false);

	valid = valid && test_conversion_with_regex("AAA Color.white AF", "AAA Color.WHITE AF", &ProjectConverter3To4::rename_colors, "custom rename", reg_container);

	// Custom rule conversion
	{
		String from = "instance";
		String to = "instantiate";
		String name = "AA.instance()";
		Vector<String> got = String("AA.instance()").split("\n");
		String expected = "AA.instantiate()";
		custom_rename(got, from, to);
		String got_str = collect_string_from_vector(got);
		if (got_str != expected) {
			ERR_PRINT(vformat("Failed to convert custom rename \"%s\" to \"%s\", got \"%s\", instead.", name, expected, got_str));
		}
		valid = valid && (got_str == expected);
	}

	// get_object_of_execution
	{
		{ String base = "var roman = kieliszek.";
	String expected = "kieliszek.";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT(vformat("Failed to get proper data from get_object_of_execution. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", base, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
}
{
	String base = "r.";
	String expected = "r.";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT(vformat("Failed to get proper data from get_object_of_execution. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", base, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
}
{
	String base = "mortadela(";
	String expected = "";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT(vformat("Failed to get proper data from get_object_of_execution. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", base, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
}
{
	String base = "var node = $world/ukraine/lviv.";
	String expected = "$world/ukraine/lviv.";
	String got = get_object_of_execution(base);
	if (got != expected) {
		ERR_PRINT(vformat("Failed to get proper data from get_object_of_execution. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", base, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
}
}
// get_starting_space
{
	String base = "\t\t\t var roman = kieliszek.";
	String expected = "\t\t\t";
	String got = get_starting_space(base);
	if (got != expected) {
		ERR_PRINT(vformat("Failed to get proper data from get_object_of_execution. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", base, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
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
		ERR_PRINT(vformat("Failed to get proper data from parse_arguments. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", line, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
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
		ERR_PRINT(vformat("Failed to get proper data from parse_arguments. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", line, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
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
		ERR_PRINT(vformat("Failed to get proper data from parse_arguments. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", line, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
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
		ERR_PRINT(vformat("Failed to get proper data from parse_arguments. \"%s\" should return \"%s\"(%d), got \"%s\"(%d), instead.", line, expected, expected.size(), got, got.size()));
	}
	valid = valid && (got == expected);
}

return valid;
}

// Validate in all arrays if names don't do cyclic renames "Node" -> "Node2D" | "Node2D" -> "2DNode"
bool ProjectConverter3To4::test_array_names() {
	bool valid = true;
	Vector<String> names = Vector<String>();

	// Validate if all classes are valid.
	{
		for (unsigned int current_index = 0; class_renames[current_index][0]; current_index++) {
			const String old_class = class_renames[current_index][0];
			const String new_class = class_renames[current_index][1];

			// Light2D, Texture, Viewport are special classes(probably virtual ones).
			if (ClassDB::class_exists(StringName(old_class)) && old_class != "Light2D" && old_class != "Texture" && old_class != "Viewport") {
				ERR_PRINT(vformat("Class \"%s\" exists in Godot 4.0, so it cannot be renamed to something else.", old_class));
				valid = false; // This probably should be only a warning, but not 100% sure - this would need to be added to CI.
			}

			// Callable is special class, to which normal classes may be renamed.
			if (!ClassDB::class_exists(StringName(new_class)) && new_class != "Callable") {
				ERR_PRINT(vformat("Class \"%s\" does not exist in Godot 4.0, so it cannot be used in the conversion.", old_class));
				valid = false; // This probably should be only a warning, but not 100% sure - this would need to be added to CI.
			}
		}
	}

	{
		HashSet<String> all_functions;

		// List of excluded functions from builtin types and global namespace, because currently it is not possible to get list of functions from them.
		// This will be available when https://github.com/godotengine/godot/pull/49053 or similar will be included into Godot.
		static const char *builtin_types_excluded_functions[] = { "dict_to_inst", "inst_to_dict", "bytes_to_var", "bytes_to_var_with_objects", "db_to_linear", "deg_to_rad", "linear_to_db", "rad_to_deg", "randf_range", "snapped", "str_to_var", "var_to_str", "var_to_bytes", "var_to_bytes_with_objects", "move_toward", "uri_encode", "uri_decode", "remove_at", "get_rotation_quaternion", "clamp", "grow_side", "is_absolute_path", "is_valid_int", "lerp", "to_ascii_buffer", "to_utf8_buffer", "to_utf32_buffer", "snapped", "remap", "rfind", nullptr };
		for (int current_index = 0; builtin_types_excluded_functions[current_index]; current_index++) {
			all_functions.insert(builtin_types_excluded_functions[current_index]);
		}

		//			for (int type = Variant::Type::NIL + 1; type < Variant::Type::VARIANT_MAX; type++) {
		//				List<MethodInfo> method_list;
		//				Variant::get_method_list_by_type(&method_list, Variant::Type(type));
		//				for (MethodInfo &function_data : method_list) {
		//					if (!all_functions.has(function_data.name)) {
		//						all_functions.insert(function_data.name);
		//					}
		//				}
		//			}

		List<StringName> classes_list;
		ClassDB::get_class_list(&classes_list);
		for (StringName &name_of_class : classes_list) {
			List<MethodInfo> method_list;
			ClassDB::get_method_list(name_of_class, &method_list, true);
			for (MethodInfo &function_data : method_list) {
				if (!all_functions.has(function_data.name)) {
					all_functions.insert(function_data.name);
				}
			}
		}

		int current_element = 0;
		while (gdscript_function_renames[current_element][0] != nullptr) {
			String name_3_x = gdscript_function_renames[current_element][0];
			String name_4_0 = gdscript_function_renames[current_element][1];
			if (!all_functions.has(gdscript_function_renames[current_element][1])) {
				ERR_PRINT(vformat("Missing GDScript function in pair (%s - ===> %s <===)", name_3_x, name_4_0));
				valid = false;
			}
			current_element++;
		}
	}
	if (!valid) {
		ERR_PRINT("Found function which is used in the converter, but it cannot be found in Godot 4. Rename this element or remove its entry if it's obsolete.");
	}

	valid = valid && test_single_array(enum_renames);
	valid = valid && test_single_array(class_renames, true);
	valid = valid && test_single_array(gdscript_function_renames, true);
	valid = valid && test_single_array(csharp_function_renames, true);
	valid = valid && test_single_array(gdscript_properties_renames, true);
	valid = valid && test_single_array(csharp_properties_renames, true);
	valid = valid && test_single_array(shaders_renames, true);
	valid = valid && test_single_array(gdscript_signals_renames);
	valid = valid && test_single_array(project_settings_renames);
	valid = valid && test_single_array(input_map_renames);
	valid = valid && test_single_array(builtin_types_renames);
	valid = valid && test_single_array(color_renames);

	return valid;
}

// Validates the array to prevent cyclic renames, such as `Node` -> `Node2D`, then `Node2D` -> `2DNode`.
// Also checks if names contain leading or trailing spaces.
bool ProjectConverter3To4::test_single_array(const char *p_array[][2], bool p_ignore_4_0_name) {
	bool valid = true;
	Vector<String> names = Vector<String>();

	for (unsigned int current_index = 0; p_array[current_index][0]; current_index++) {
		String name_3_x = p_array[current_index][0];
		String name_4_0 = p_array[current_index][1];
		if (name_3_x != name_3_x.strip_edges()) {
			ERR_PRINT(vformat("Invalid Entry \"%s\" contains leading or trailing spaces.", name_3_x));
			valid = false;
		}
		if (names.has(name_3_x)) {
			ERR_PRINT(vformat("Found duplicated entry, pair ( -> %s , %s)", name_3_x, name_4_0));
			valid = false;
		}
		names.append(name_3_x);

		if (name_4_0 != name_4_0.strip_edges()) {
			ERR_PRINT(vformat("Invalid Entry \"%s\" contains leading or trailing spaces.", name_3_x));
			valid = false;
		}
		if (names.has(name_4_0)) {
			ERR_PRINT(vformat("Found duplicated entry, pair ( -> %s , %s)", name_3_x, name_4_0));
			valid = false;
		}
		if (!p_ignore_4_0_name) {
			names.append(name_4_0);
		}
	}
	return valid;
};

// Returns arguments from given function execution, this cannot be really done as regex.
// `abc(d,e(f,g),h)` -> [d], [e(f,g)], [h]
Vector<String> ProjectConverter3To4::parse_arguments(const String &line) {
	Vector<String> parts;
	int string_size = line.length();
	int start_part = 0; // Index of beginning of start part.
	int parts_counter = 0;
	char32_t previous_character = '\0';
	bool is_inside_string = false; // If true, it ignores these 3 characters ( , ) inside string.

	ERR_FAIL_COND_V_MSG(line.count("(") != line.count(")"), parts, vformat("Converter internal bug: substring should have equal number of open and close parentheses in line - \"%s\".", line));

	for (int current_index = 0; current_index < string_size; current_index++) {
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
			case '[': {
				parts_counter++;
				if (parts_counter == 1 && !is_inside_string) {
					start_part = current_index;
				}
				break;
			};
			case ']': {
				parts_counter--;
				if (parts_counter == 0 && !is_inside_string) {
					parts.append(line.substr(start_part, current_index - start_part));
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

// Finds latest parenthesis owned by function.
// `function(abc(a,b),DD)):` finds this parenthess `function(abc(a,b),DD => ) <= ):`
int ProjectConverter3To4::get_end_parenthesis(const String &line) const {
	int current_state = 0;
	for (int current_index = 0; line.length() > current_index; current_index++) {
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
	}
	return -1;
}

// Merges multiple arguments into a single String.
// Needed when after processing e.g. 2 arguments, later arguments are not changed in any way.
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

// Returns the indentation (spaces and tabs) at the start of the line e.g. `\t\tmove_this` returns `\t\t`.
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

// Returns the object that’s executing the function in the line.
// e.g. Passing the line "var roman = kieliszek.funkcja()" to this function returns "kieliszek".
String ProjectConverter3To4::get_object_of_execution(const String &line) const {
	int end = line.size() - 1; // Last one is \0
	int variable_start = end - 1;
	int start = end - 1;

	bool is_possibly_nodepath = false;
	bool is_valid_nodepath = false;

	while (start >= 0) {
		char32_t character = line[start];
		bool is_variable_char = (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') || character == '.' || character == '_';
		bool is_nodepath_start = character == '$';
		bool is_nodepath_sep = character == '/';
		if (is_variable_char || is_nodepath_start || is_nodepath_sep) {
			if (start == 0) {
				break;
			} else if (is_nodepath_sep) {
				// Freeze variable_start, try to fetch more chars since this might be a Node path literal.
				is_possibly_nodepath = true;
			} else if (is_nodepath_start) {
				// Found $, this is a Node path literal.
				is_valid_nodepath = true;
				break;
			}
			if (!is_possibly_nodepath) {
				variable_start--;
			}
			start--;
			continue;
		} else {
			// Abandon all hope, this is neither a variable nor a Node path literal.
			variable_start++; // Found invalid character, needs to be ignored.
			break;
		}
	}
	if (is_valid_nodepath) {
		variable_start = start;
	}
	return line.substr(variable_start, (end - variable_start));
}

void ProjectConverter3To4::rename_colors(Vector<String> &lines, const RegExContainer &reg_container) {
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			if (line.contains("Color.")) {
				for (unsigned int current_index = 0; color_renames[current_index][0]; current_index++) {
					line = reg_container.color_regexes[current_index]->sub(line, reg_container.color_renamed[current_index], true);
				}
			}
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_rename_colors(Vector<String> &lines, const RegExContainer &reg_container) {
	Vector<String> found_renames;

	int current_line = 1;
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			if (line.contains("Color.")) {
				for (unsigned int current_index = 0; color_renames[current_index][0]; current_index++) {
					TypedArray<RegExMatch> reg_match = reg_container.color_regexes[current_index]->search_all(line);
					if (reg_match.size() > 0) {
						found_renames.append(line_formatter(current_line, color_renames[current_index][0], color_renames[current_index][1], line));
					}
				}
			}
		}
		current_line++;
	}

	return found_renames;
}

void ProjectConverter3To4::rename_classes(Vector<String> &lines, const RegExContainer &reg_container) {
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			for (unsigned int current_index = 0; class_renames[current_index][0]; current_index++) {
				if (line.contains(class_renames[current_index][0])) {
					bool found_ignored_items = false;
					// Renaming Spatial.tscn to TEMP_RENAMED_CLASS.tscn.
					if (line.contains(String(class_renames[current_index][0]) + ".")) {
						found_ignored_items = true;
						line = reg_container.class_tscn_regexes[current_index]->sub(line, "TEMP_RENAMED_CLASS.tscn", true);
						line = reg_container.class_gd_regexes[current_index]->sub(line, "TEMP_RENAMED_CLASS.gd", true);
						line = reg_container.class_shader_regexes[current_index]->sub(line, "TEMP_RENAMED_CLASS.shader", true);
					}

					// Causal renaming Spatial -> Node3D.
					line = reg_container.class_regexes[current_index]->sub(line, class_renames[current_index][1], true);

					// Restore Spatial.tscn from TEMP_RENAMED_CLASS.tscn.
					if (found_ignored_items) {
						line = reg_container.class_temp_tscn.sub(line, reg_container.class_temp_tscn_renames[current_index], true);
						line = reg_container.class_temp_gd.sub(line, reg_container.class_temp_gd_renames[current_index], true);
						line = reg_container.class_temp_shader.sub(line, reg_container.class_temp_shader_renames[current_index], true);
					}
				}
			}
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_rename_classes(Vector<String> &lines, const RegExContainer &reg_container) {
	Vector<String> found_renames;

	int current_line = 1;

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			for (unsigned int current_index = 0; class_renames[current_index][0]; current_index++) {
				if (line.contains(class_renames[current_index][0])) {
					String old_line = line;
					bool found_ignored_items = false;
					// Renaming Spatial.tscn to TEMP_RENAMED_CLASS.tscn.
					if (line.contains(String(class_renames[current_index][0]) + ".")) {
						found_ignored_items = true;
						line = reg_container.class_tscn_regexes[current_index]->sub(line, "TEMP_RENAMED_CLASS.tscn", true);
						line = reg_container.class_gd_regexes[current_index]->sub(line, "TEMP_RENAMED_CLASS.gd", true);
						line = reg_container.class_shader_regexes[current_index]->sub(line, "TEMP_RENAMED_CLASS.shader", true);
					}

					// Causal renaming Spatial -> Node3D.
					TypedArray<RegExMatch> reg_match = reg_container.class_regexes[current_index]->search_all(line);
					if (reg_match.size() > 0) {
						found_renames.append(line_formatter(current_line, class_renames[current_index][0], class_renames[current_index][1], old_line));
					}

					// Restore Spatial.tscn from TEMP_RENAMED_CLASS.tscn.
					if (found_ignored_items) {
						line = reg_container.class_temp_tscn.sub(line, reg_container.class_temp_tscn_renames[current_index], true);
						line = reg_container.class_temp_gd.sub(line, reg_container.class_temp_gd_renames[current_index], true);
						line = reg_container.class_temp_shader.sub(line, reg_container.class_temp_shader_renames[current_index], true);
					}
				}
			}
		}
		current_line++;
	}
	return found_renames;
}

void ProjectConverter3To4::rename_gdscript_functions(Vector<String> &lines, const RegExContainer &reg_container, bool builtin) {
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			process_gdscript_line(line, reg_container, builtin);
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_rename_gdscript_functions(Vector<String> &lines, const RegExContainer &reg_container, bool builtin) {
	int current_line = 1;

	Vector<String> found_renames;

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			String old_line = line;
			process_gdscript_line(line, reg_container, builtin);
			if (old_line != line) {
				found_renames.append(simple_line_formatter(current_line, old_line, line));
			}
		}
	}

	return found_renames;
}

// TODO, this function should run only on all ".gd" files and also on lines in ".tscn" files which are parts of built-in Scripts.
void ProjectConverter3To4::process_gdscript_line(String &line, const RegExContainer &reg_container, bool builtin) {
	// In this and other functions, reg.sub() is used only after checking lines with str.contains().
	// With longer lines, doing so can sometimes be significantly faster.

	if ((line.contains(".lock") || line.contains(".unlock")) && !line.contains("mtx") && !line.contains("mutex") && !line.contains("Mutex")) {
		line = reg_container.reg_image_lock.sub(line, "false # $1.lock() # TODOConverter40, Image no longer requires locking, `false` helps to not break one line if/else, so it can freely be removed", true);
		line = reg_container.reg_image_unlock.sub(line, "false # $1.unlock() # TODOConverter40, Image no longer requires locking, `false` helps to not break one line if/else, so it can freely be removed", true);
	}

	// PackedStringArray(req_godot).join('.') -> '.'.join(PackedStringArray(req_godot))       PoolStringArray
	if (line.contains(".join")) {
		line = reg_container.reg_join.sub(line, "$2.join($1)", true);
	}

	// -- empty() -> is_empty()       Pool*Array
	if (line.contains("empty")) {
		line = reg_container.reg_is_empty.sub(line, "is_empty(", true);
	}

	// -- \t.func() -> \tsuper.func()       Object
	if (line.contains("(") && line.contains(".")) {
		line = reg_container.reg_super.sub(line, "$1super.$2", true); // TODO, not sure if possible, but for now this broke String text e.g. "Chosen .gitignore" -> "Chosen super.gitignore"
	}

	// -- JSON.parse(a) -> JSON.new().parse(a) etc.    JSON
	if (line.contains("parse")) {
		line = reg_container.reg_json_non_new.sub(line, "$1var test_json_conv = JSON.new()\n$1test_json_conv.parse($3\n$1$2test_json_conv.get_data()", true);
	}

	// -- to_json(a) -> JSON.new().stringify(a)     Object
	if (line.contains("to_json")) {
		line = reg_container.reg_json_to.sub(line, "JSON.new().stringify", true);
	}
	// -- parse_json(a) -> JSON.get_data() etc.    Object
	if (line.contains("parse_json")) {
		line = reg_container.reg_json_parse.sub(line, "$1var test_json_conv = JSON.new()\n$1test_json_conv.parse($3\n$1$2test_json_conv.get_data()", true);
	}
	// -- JSON.print( -> JSON.stringify(
	if (line.contains("JSON.print(")) {
		line = reg_container.reg_json_print.sub(line, "JSON.stringify(", true);
	}

	// -- get_node(@ -> get_node(       Node
	if (line.contains("get_node")) {
		line = line.replace("get_node(@", "get_node(");
	}

	// export(float) var lifetime = 3.0 -> export var lifetime: float = 3.0     GDScript
	if (line.contains("export")) {
		line = reg_container.reg_export.sub(line, "export var $2: $1");
	}

	// export(String, 'AnonymousPro', 'CourierPrime') var _font_name = 'AnonymousPro' -> export var _font_name = 'AnonymousPro' #(String, 'AnonymousPro', 'CourierPrime')   GDScript
	if (line.contains("export")) {
		line = reg_container.reg_export_advanced.sub(line, "export var $2$3 # ($1)");
	}

	// Setget Setget
	if (line.contains("setget")) {
		line = reg_container.reg_setget_setget.sub(line, "var $1$2: get = $4, set = $3", true);
	}

	// Setget set
	if (line.contains("setget")) {
		line = reg_container.reg_setget_set.sub(line, "var $1$2: set = $3", true);
	}

	// Setget get
	if (line.contains("setget")) {
		line = reg_container.reg_setget_get.sub(line, "var $1$2: get = $3", true);
	}

	if (line.contains("window_resizable")) {
		// OS.set_window_resizable(a) -> get_window().unresizable = not (a)
		line = reg_container.reg_os_set_window_resizable.sub(line, "get_window().unresizable = not ($1)", true);
		// OS.window_resizable = a -> same
		line = reg_container.reg_os_assign_window_resizable.sub(line, "get_window().unresizable = not ($1)", true);
		// OS.[is_]window_resizable() -> (not get_window().unresizable)
		line = reg_container.reg_os_is_window_resizable.sub(line, "(not get_window().unresizable)", true);
	}

	if (line.contains("window_fullscreen")) {
		// OS.window_fullscreen(a) -> get_window().mode = Window.MODE_EXCLUSIVE_FULLSCREEN if (a) else Window.MODE_WINDOWED
		line = reg_container.reg_os_set_fullscreen.sub(line, "get_window().mode = Window.MODE_EXCLUSIVE_FULLSCREEN if ($1) else Window.MODE_WINDOWED", true);
		// window_fullscreen = a -> same
		line = reg_container.reg_os_assign_fullscreen.sub(line, "get_window().mode = Window.MODE_EXCLUSIVE_FULLSCREEN if ($1) else Window.MODE_WINDOWED", true);
		// OS.[is_]window_fullscreen() -> ((get_window().mode == Window.MODE_EXCLUSIVE_FULLSCREEN) or (get_window().mode == Window.MODE_FULLSCREEN))
		line = reg_container.reg_os_is_fullscreen.sub(line, "((get_window().mode == Window.MODE_EXCLUSIVE_FULLSCREEN) or (get_window().mode == Window.MODE_FULLSCREEN))", true);
	}

	if (line.contains("window_maximized")) {
		// OS.window_maximized(a) -> get_window().mode = Window.MODE_MAXIMIZED if (a) else Window.MODE_WINDOWED
		line = reg_container.reg_os_set_maximized.sub(line, "get_window().mode = Window.MODE_MAXIMIZED if ($1) else Window.MODE_WINDOWED", true);
		// window_maximized = a -> same
		line = reg_container.reg_os_assign_maximized.sub(line, "get_window().mode = Window.MODE_MAXIMIZED if ($1) else Window.MODE_WINDOWED", true);
		// OS.[is_]window_maximized() -> (get_window().mode == Window.MODE_MAXIMIZED)
		line = reg_container.reg_os_is_maximized.sub(line, "(get_window().mode == Window.MODE_MAXIMIZED)", true);
	}

	if (line.contains("window_minimized")) {
		// OS.window_minimized(a) -> get_window().mode = Window.MODE_MINIMIZED if (a) else Window.MODE_WINDOWED
		line = reg_container.reg_os_set_minimized.sub(line, "get_window().mode = Window.MODE_MINIMIZED if ($1) else Window.MODE_WINDOWED", true);
		// window_minimized = a -> same
		line = reg_container.reg_os_assign_minimized.sub(line, "get_window().mode = Window.MODE_MINIMIZED if ($1) else Window.MODE_WINDOWED", true);
		// OS.[is_]window_minimized() -> (get_window().mode == Window.MODE_MINIMIZED)
		line = reg_container.reg_os_is_minimized.sub(line, "(get_window().mode == Window.MODE_MINIMIZED)", true);
	}

	if (line.contains("set_use_vsync")) {
		// OS.set_use_vsync(a) -> get_window().window_set_vsync_mode(DisplayServer.VSYNC_ENABLED if (a) else DisplayServer.VSYNC_DISABLED)
		line = reg_container.reg_os_set_vsync.sub(line, "DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_ENABLED if ($1) else DisplayServer.VSYNC_DISABLED)", true);
	}
	if (line.contains("vsync_enabled")) {
		// vsync_enabled = a -> get_window().window_set_vsync_mode(DisplayServer.VSYNC_ENABLED if (a) else DisplayServer.VSYNC_DISABLED)
		line = reg_container.reg_os_assign_vsync.sub(line, "DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_ENABLED if ($1) else DisplayServer.VSYNC_DISABLED)", true);
		// OS.[is_]vsync_enabled() -> (DisplayServer.window_get_vsync_mode() != DisplayServer.VSYNC_DISABLED)
		line = reg_container.reg_os_is_vsync.sub(line, "(DisplayServer.window_get_vsync_mode() != DisplayServer.VSYNC_DISABLED)", true);
	}

	if (line.contains("OS.screen_orientation")) { // keep "OS." at start
		// OS.screen_orientation = a -> DisplayServer.screen_set_orientation(a)
		line = reg_container.reg_os_assign_screen_orient.sub(line, "$1DisplayServer.screen_set_orientation($2)", true); // assignment
		line = line.replace("OS.screen_orientation", "DisplayServer.screen_get_orientation()"); // value access
	}

	if (line.contains("_window_always_on_top")) {
		// OS.set_window_always_on_top(a) -> get_window().always_on_top = (a)
		line = reg_container.reg_os_set_always_on_top.sub(line, "get_window().always_on_top = ($1)", true);
		// OS.is_window_always_on_top() -> get_window().always_on_top
		line = reg_container.reg_os_is_always_on_top.sub(line, "get_window().always_on_top", true);
	}

	if (line.contains("et_borderless_window")) {
		// OS.set_borderless_window(a) -> get_window().borderless = (a)
		line = reg_container.reg_os_set_borderless.sub(line, "get_window().borderless = ($1)", true);
		// OS.get_borderless_window() -> get_window().borderless
		line = reg_container.reg_os_get_borderless.sub(line, "get_window().borderless", true);
	}

	// OS.SCREEN_ORIENTATION_* -> DisplayServer.SCREEN_*
	if (line.contains("OS.SCREEN_ORIENTATION_")) {
		line = reg_container.reg_os_screen_orient_enum.sub(line, "DisplayServer.SCREEN_$1", true);
	}

	// OS -> Window simple replacements with optional set/get.
	if (line.contains("current_screen")) {
		line = reg_container.reg_os_current_screen.sub(line, "get_window().$1current_screen", true);
	}
	if (line.contains("min_window_size")) {
		line = reg_container.reg_os_min_window_size.sub(line, "get_window().$1min_size", true);
	}
	if (line.contains("max_window_size")) {
		line = reg_container.reg_os_max_window_size.sub(line, "get_window().$1max_size", true);
	}
	if (line.contains("window_position")) {
		line = reg_container.reg_os_window_position.sub(line, "get_window().$1position", true);
	}
	if (line.contains("window_size")) {
		line = reg_container.reg_os_window_size.sub(line, "get_window().$1size", true);
	}
	if (line.contains("et_screen_orientation")) {
		line = reg_container.reg_os_getset_screen_orient.sub(line, "DisplayServer.screen_$1et_orientation", true);
	}

	// Instantiate
	if (line.contains("instance")) {
		line = reg_container.reg_instantiate.sub(line, ".instantiate($1)", true);
	}

	// -- r.move_and_slide( a, b, c, d, e )  ->  r.set_velocity(a) ... r.move_and_slide()         KinematicBody
	if (line.contains(("move_and_slide("))) {
		int start = line.find("move_and_slide(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
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

				line_new += starting_space + base_obj + "move_and_slide()";

				if (!line.begins_with(starting_space + "move_and_slide")) {
					line = line_new + "\n" + line.substr(0, start) + "velocity" + line.substr(end + start);
				} else {
					line = line_new + line.substr(end + start);
				}
			}
		}
	}

	// -- r.move_and_slide_with_snap( a, b, c, d, e )  ->  r.set_velocity(a) ... r.move_and_slide()         KinematicBody
	if (line.contains("move_and_slide_with_snap(")) {
		int start = line.find("move_and_slide_with_snap(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
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

				line_new += starting_space + base_obj + "move_and_slide()";

				if (!line.begins_with(starting_space + "move_and_slide_with_snap")) {
					line = line_new + "\n" + line.substr(0, start) + "velocity" + line.substr(end + start);
				} else {
					line = line_new + line.substr(end + start);
				}
			}
		}
	}

	// -- sort_custom( a , b )  ->  sort_custom(Callable( a , b ))            Object
	if (line.contains("sort_custom(")) {
		int start = line.find("sort_custom(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "sort_custom(Callable(" + parts[0] + "," + parts[1] + "))" + line.substr(end + start);
			}
		}
	}

	// -- list_dir_begin( )  ->  list_dir_begin()            Object
	if (line.contains("list_dir_begin(")) {
		int start = line.find("list_dir_begin(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			line = line.substr(0, start) + "list_dir_begin() " + line.substr(end + start) + "# TODOGODOT4 fill missing arguments https://github.com/godotengine/godot/pull/40547";
		}
	}

	// -- draw_line(1,2,3,4,5) -> draw_line(1,2,3,4)            CanvasItem
	if (line.contains("draw_line(")) {
		int start = line.find("draw_line(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 5) {
				line = line.substr(0, start) + "draw_line(" + parts[0] + "," + parts[1] + "," + parts[2] + "," + parts[3] + ")" + line.substr(end + start);
			}
		}
	}

	// -- func c(var a, var b) -> func c(a, b)
	if (line.contains("func ") && line.contains("var ")) {
		int start = line.find("func ");
		start = line.substr(start).find("(") + start;
		int end = get_end_parenthesis(line.substr(start)) + 1;
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
	if (line.contains("yield(")) {
		int start = line.find("yield(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
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
	if (line.contains("parse_json(")) {
		int start = line.find("parse_json(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			line = line.substr(0, start) + "JSON.new().stringify(" + connect_arguments(parts, 0) + ")" + line.substr(end + start);
		}
	}

	// -- .xform(Vector3(a,b,c)) -> * Vector3(a,b,c)            Transform
	if (line.contains(".xform(")) {
		int start = line.find(".xform(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 1) {
				line = line.substr(0, start) + " * " + parts[0] + line.substr(end + start);
			}
		}
	}

	// -- .xform_inv(Vector3(a,b,c)) -> * Vector3(a,b,c)       Transform
	if (line.contains(".xform_inv(")) {
		int start = line.find(".xform_inv(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			String object_exec = get_object_of_execution(line.substr(0, start));
			if (line.contains(object_exec + ".xform")) {
				int start2 = line.find(object_exec + ".xform");
				Vector<String> parts = parse_arguments(line.substr(start, end));
				if (parts.size() == 1) {
					line = line.substr(0, start2) + parts[0] + " * " + object_exec + line.substr(end + start);
				}
			}
		}
	}

	// -- "(connect(A,B,C,D,E) != OK):", "(connect(A,Callable(B,C).bind(D),E)      Object
	if (line.contains("connect(")) {
		int start = line.find("connect(");
		// Protection from disconnect
		if (start == 0 || line.get(start - 1) != 's') {
			int end = get_end_parenthesis(line.substr(start)) + 1;
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
	if (line.contains("disconnect(")) {
		int start = line.find("disconnect(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "disconnect(" + parts[0] + ",Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	// -- is_connected(a,b,c) -> is_connected(a,Callable(b,c))      Object
	if (line.contains("is_connected(")) {
		int start = line.find("is_connected(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "is_connected(" + parts[0] + ",Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	// -- "(tween_method(A,B,C,D,E) != OK):", "(tween_method(Callable(A,B),C,D,E)      Object
	// -- "(tween_method(A,B,C,D,E,[F,G]) != OK):", "(tween_method(Callable(A,B).bind(F,G),C,D,E)      Object
	if (line.contains("tween_method(")) {
		int start = line.find("tween_method(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
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
	if (line.contains("tween_callback(")) {
		int start = line.find("tween_callback(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
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
	if (line.contains("start(")) {
		int start = line.find("start(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
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
	if (line.contains(" _init(")) {
		int start = line.find(" _init(");
		if (line.contains(":")) {
			int end = line.rfind(":") + 1;
			if (end > -1) {
				Vector<String> parts = parse_arguments(line.substr(start, end));
				line = line.substr(0, start) + " _init(" + connect_arguments(parts, 0) + "):" + line.substr(end + start);
			}
		}
	}
	//  assert(speed < 20, str(randi()%10))  ->  assert(speed < 20) #,str(randi()%10))    GDScript - GDScript bug constant message
	if (line.contains("assert(")) {
		int start = line.find("assert(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "assert(" + parts[0] + ") " + line.substr(end + start) + "#," + parts[1] + ")";
			}
		}
	}
	//  create_from_image(aa, bb)  ->   create_from_image(aa) #, bb   ImageTexture
	if (line.contains("create_from_image(")) {
		int start = line.find("create_from_image(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "create_from_image(" + parts[0] + ") " + "#," + parts[1] + line.substr(end + start);
			}
		}
	}
	//  set_cell_item(a, b, c, d ,e)  ->   set_cell_item(Vector3(a, b, c), d ,e)
	if (line.contains("set_cell_item(")) {
		int start = line.find("set_cell_item(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() > 2) {
				line = line.substr(0, start) + "set_cell_item( Vector3(" + parts[0] + "," + parts[1] + "," + parts[2] + ") " + connect_arguments(parts, 3) + ")" + line.substr(end + start);
			}
		}
	}
	//  get_cell_item(a, b, c)  ->   get_cell_item(Vector3i(a, b, c))
	if (line.contains("get_cell_item(")) {
		int start = line.find("get_cell_item(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "get_cell_item(Vector3i(" + parts[0] + "," + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	//  get_cell_item_orientation(a, b, c)  ->   get_cell_item_orientation(Vector3i(a, b, c))
	if (line.contains("get_cell_item_orientation(")) {
		int start = line.find("get_cell_item_orientation(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "get_cell_item_orientation(Vector3i(" + parts[0] + "," + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	//  apply_impulse(A, B)  ->   apply_impulse(B, A)
	if (line.contains("apply_impulse(")) {
		int start = line.find("apply_impulse(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "apply_impulse(" + parts[1] + ", " + parts[0] + ")" + line.substr(end + start);
			}
		}
	}
	//  apply_force(A, B)  ->   apply_force(B, A)
	if (line.contains("apply_force(")) {
		int start = line.find("apply_force(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 2) {
				line = line.substr(0, start) + "apply_force(" + parts[1] + ", " + parts[0] + ")" + line.substr(end + start);
			}
		}
	}
	//  map_to_world(a, b, c)  ->   map_to_local(Vector3i(a, b, c))
	if (line.contains("map_to_world(")) {
		int start = line.find("map_to_world(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "map_to_local(Vector3i(" + parts[0] + "," + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			} else if (parts.size() == 1) {
				line = line.substr(0, start) + "map_to_local(" + parts[0] + ")" + line.substr(end + start);
			}
		}
	}

	//  set_rotating(true)  ->   set_ignore_rotation(false)
	if (line.contains("set_rotating(")) {
		int start = line.find("set_rotating(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 1) {
				String opposite = parts[0] == "true" ? "false" : "true";
				line = line.substr(0, start) + "set_ignore_rotation(" + opposite + ")";
			}
		}
	}

	//  OS.get_window_safe_area()  ->   DisplayServer.get_display_safe_area()
	if (line.contains("OS.get_window_safe_area(")) {
		int start = line.find("OS.get_window_safe_area(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 0) {
				line = line.substr(0, start) + "DisplayServer.get_display_safe_area()" + line.substr(end + start);
			}
		}
	}
	//  draw_rect(a,b,c,d,e)  ->   draw_rect(a,b,c,d)#e) TODOGODOT4 Antialiasing argument is missing
	if (line.contains("draw_rect(")) {
		int start = line.find("draw_rect(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 5) {
				line = line.substr(0, start) + "draw_rect(" + parts[0] + "," + parts[1] + "," + parts[2] + "," + parts[3] + ")" + line.substr(end + start) + "# " + parts[4] + ") TODOGODOT4 Antialiasing argument is missing";
			}
		}
	}
	// get_focus_owner() -> get_viewport().gui_get_focus_owner()
	if (line.contains("get_focus_owner()")) {
		line = line.replace("get_focus_owner()", "get_viewport().gui_get_focus_owner()");
	}

	// button.pressed = 1 -> button.button_pressed = 1
	if (line.contains(".pressed")) {
		int start = line.find(".pressed");
		bool foundNextEqual = false;
		String line_to_check = line.substr(start + String(".pressed").length());
		for (int current_index = 0; line_to_check.length() > current_index; current_index++) {
			char32_t chr = line_to_check.get(current_index);
			if (chr == '\t' || chr == ' ') {
				continue;
			} else if (chr == '=') {
				foundNextEqual = true;
			} else {
				break;
			}
		}
		if (foundNextEqual) {
			line = line.substr(0, start) + ".button_pressed" + line.substr(start + String(".pressed").length());
		}
	}

	// rotating = true  ->   ignore_rotation = false # reversed "rotating" for Camera2D
	if (line.contains("rotating")) {
		int start = line.find("rotating");
		bool foundNextEqual = false;
		String line_to_check = line.substr(start + String("rotating").length());
		String assigned_value;
		for (int current_index = 0; line_to_check.length() > current_index; current_index++) {
			char32_t chr = line_to_check.get(current_index);
			if (chr == '\t' || chr == ' ') {
				continue;
			} else if (chr == '=') {
				foundNextEqual = true;
				assigned_value = line.right(current_index).strip_edges();
				assigned_value = assigned_value == "true" ? "false" : "true";
			} else {
				break;
			}
		}
		if (foundNextEqual) {
			line = line.substr(0, start) + "ignore_rotation =" + assigned_value + " # reversed \"rotating\" for Camera2D";
		}
	}

	// OS -> Time functions
	if (line.contains("OS.get_ticks_msec")) {
		line = line.replace("OS.get_ticks_msec", "Time.get_ticks_msec");
	}
	if (line.contains("OS.get_ticks_usec")) {
		line = line.replace("OS.get_ticks_usec", "Time.get_ticks_usec");
	}
	if (line.contains("OS.get_unix_time")) {
		line = line.replace("OS.get_unix_time", "Time.get_unix_time_from_system");
	}
	if (line.contains("OS.get_datetime")) {
		line = line.replace("OS.get_datetime", "Time.get_datetime_dict_from_system");
	}

	// OS -> DisplayServer
	if (line.contains("OS.get_display_cutouts")) {
		line = line.replace("OS.get_display_cutouts", "DisplayServer.get_display_cutouts");
	}
	if (line.contains("OS.get_screen_count")) {
		line = line.replace("OS.get_screen_count", "DisplayServer.get_screen_count");
	}
	if (line.contains("OS.get_screen_dpi")) {
		line = line.replace("OS.get_screen_dpi", "DisplayServer.screen_get_dpi");
	}
	if (line.contains("OS.get_screen_max_scale")) {
		line = line.replace("OS.get_screen_max_scale", "DisplayServer.screen_get_max_scale");
	}
	if (line.contains("OS.get_screen_position")) {
		line = line.replace("OS.get_screen_position", "DisplayServer.screen_get_position");
	}
	if (line.contains("OS.get_screen_refresh_rate")) {
		line = line.replace("OS.get_screen_refresh_rate", "DisplayServer.screen_get_refresh_rate");
	}
	if (line.contains("OS.get_screen_scale")) {
		line = line.replace("OS.get_screen_scale", "DisplayServer.screen_get_scale");
	}
	if (line.contains("OS.get_screen_size")) {
		line = line.replace("OS.get_screen_size", "DisplayServer.screen_get_size");
	}
	if (line.contains("OS.set_icon")) {
		line = line.replace("OS.set_icon", "DisplayServer.set_icon");
	}
	if (line.contains("OS.set_native_icon")) {
		line = line.replace("OS.set_native_icon", "DisplayServer.set_native_icon");
	}

	// OS -> Window
	if (line.contains("OS.window_borderless")) {
		line = line.replace("OS.window_borderless", "get_window().borderless");
	}
	if (line.contains("OS.get_real_window_size")) {
		line = line.replace("OS.get_real_window_size", "get_window().get_size_with_decorations");
	}
	if (line.contains("OS.is_window_focused")) {
		line = line.replace("OS.is_window_focused", "get_window().has_focus");
	}
	if (line.contains("OS.move_window_to_foreground")) {
		line = line.replace("OS.move_window_to_foreground", "get_window().move_to_foreground");
	}
	if (line.contains("OS.request_attention")) {
		line = line.replace("OS.request_attention", "get_window().request_attention");
	}
	if (line.contains("OS.set_window_title")) {
		line = line.replace("OS.set_window_title", "get_window().set_title");
	}

	// get_tree().set_input_as_handled() -> get_viewport().set_input_as_handled()
	if (line.contains("get_tree().set_input_as_handled()")) {
		line = line.replace("get_tree().set_input_as_handled()", "get_viewport().set_input_as_handled()");
	}

	// Fix the simple case of using _unhandled_key_input
	// func _unhandled_key_input(event: InputEventKey) -> _unhandled_key_input(event: InputEvent)
	if (line.contains("_unhandled_key_input(event: InputEventKey)")) {
		line = line.replace("_unhandled_key_input(event: InputEventKey)", "_unhandled_key_input(event: InputEvent)");
	}
}

void ProjectConverter3To4::process_csharp_line(String &line, const RegExContainer &reg_container) {
	line = line.replace("OS.GetWindowSafeArea()", "DisplayServer.ScreenGetUsableRect()");

	// GetTree().SetInputAsHandled() -> GetViewport().SetInputAsHandled()
	if (line.contains("GetTree().SetInputAsHandled()")) {
		line = line.replace("GetTree().SetInputAsHandled()", "GetViewport().SetInputAsHandled()");
	}

	// Fix the simple case of using _UnhandledKeyInput
	// func _UnhandledKeyInput(InputEventKey @event) -> _UnhandledKeyInput(InputEvent @event)
	if (line.contains("_UnhandledKeyInput(InputEventKey @event)")) {
		line = line.replace("_UnhandledKeyInput(InputEventKey @event)", "_UnhandledKeyInput(InputEvent @event)");
	}

	// -- Connect(,,,things) -> Connect(,Callable(,),things)      Object
	if (line.contains("Connect(")) {
		int start = line.find("Connect(");
		// Protection from disconnect
		if (start == 0 || line.get(start - 1) != 's') {
			int end = get_end_parenthesis(line.substr(start)) + 1;
			if (end > -1) {
				Vector<String> parts = parse_arguments(line.substr(start, end));
				if (parts.size() >= 3) {
					line = line.substr(0, start) + "Connect(" + parts[0] + ",new Callable(" + parts[1] + "," + parts[2] + ")" + connect_arguments(parts, 3) + ")" + line.substr(end + start);
				}
			}
		}
	}
	// -- Disconnect(a,b,c) -> Disconnect(a,Callable(b,c))      Object
	if (line.contains("Disconnect(")) {
		int start = line.find("Disconnect(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "Disconnect(" + parts[0] + ",new Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
	// -- IsConnected(a,b,c) -> IsConnected(a,Callable(b,c))      Object
	if (line.contains("IsConnected(")) {
		int start = line.find("IsConnected(");
		int end = get_end_parenthesis(line.substr(start)) + 1;
		if (end > -1) {
			Vector<String> parts = parse_arguments(line.substr(start, end));
			if (parts.size() == 3) {
				line = line.substr(0, start) + "IsConnected(" + parts[0] + ",new Callable(" + parts[1] + "," + parts[2] + "))" + line.substr(end + start);
			}
		}
	}
}

void ProjectConverter3To4::rename_csharp_functions(Vector<String> &lines, const RegExContainer &reg_container) {
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			process_csharp_line(line, reg_container);
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_rename_csharp_functions(Vector<String> &lines, const RegExContainer &reg_container) {
	int current_line = 1;

	Vector<String> found_renames;

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			String old_line = line;
			process_csharp_line(line, reg_container);
			if (old_line != line) {
				found_renames.append(simple_line_formatter(current_line, old_line, line));
			}
		}
	}

	return found_renames;
}

void ProjectConverter3To4::rename_csharp_attributes(Vector<String> &lines, const RegExContainer &reg_container) {
	static String error_message = "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using Multiplayer.GetRemoteSenderId()\n";

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			line = reg_container.keyword_csharp_remote.sub(line, "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", true);
			line = reg_container.keyword_csharp_remotesync.sub(line, "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", true);
			line = reg_container.keyword_csharp_puppet.sub(line, "[RPC]", true);
			line = reg_container.keyword_csharp_puppetsync.sub(line, "[RPC(CallLocal = true)]", true);
			line = reg_container.keyword_csharp_master.sub(line, error_message + "[RPC]", true);
			line = reg_container.keyword_csharp_mastersync.sub(line, error_message + "[RPC(CallLocal = true)]", true);
		}
	}
}

Vector<String> ProjectConverter3To4::check_for_rename_csharp_attributes(Vector<String> &lines, const RegExContainer &reg_container) {
	int current_line = 1;

	Vector<String> found_renames;

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			String old;
			old = line;
			line = reg_container.keyword_csharp_remote.sub(line, "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", true);
			if (old != line) {
				found_renames.append(line_formatter(current_line, "[Remote]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer)]", line));
			}

			old = line;
			line = reg_container.keyword_csharp_remotesync.sub(line, "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", true);
			if (old != line) {
				found_renames.append(line_formatter(current_line, "[RemoteSync]", "[RPC(MultiplayerAPI.RPCMode.AnyPeer, CallLocal = true)]", line));
			}

			old = line;
			line = reg_container.keyword_csharp_puppet.sub(line, "[RPC]", true);
			if (old != line) {
				found_renames.append(line_formatter(current_line, "[Puppet]", "[RPC]", line));
			}

			old = line;
			line = reg_container.keyword_csharp_puppetsync.sub(line, "[RPC(CallLocal = true)]", true);
			if (old != line) {
				found_renames.append(line_formatter(current_line, "[PuppetSync]", "[RPC(CallLocal = true)]", line));
			}

			old = line;
			line = reg_container.keyword_csharp_master.sub(line, "[RPC]", true);
			if (old != line) {
				found_renames.append(line_formatter(current_line, "[Master]", "[RPC]", line));
			}

			old = line;
			line = reg_container.keyword_csharp_mastersync.sub(line, "[RPC(CallLocal = true)]", true);
			if (old != line) {
				found_renames.append(line_formatter(current_line, "[MasterSync]", "[RPC(CallLocal = true)]", line));
			}
		}
		current_line++;
	}

	return found_renames;
}

void ProjectConverter3To4::rename_gdscript_keywords(Vector<String> &lines, const RegExContainer &reg_container) {
	static String error_message = "The master and mastersync rpc behavior is not officially supported anymore. Try using another keyword or making custom logic using get_multiplayer().get_remote_sender_id()\n";

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			if (line.contains("tool")) {
				line = reg_container.keyword_gdscript_tool.sub(line, "@tool", true);
			}
			if (line.contains("export")) {
				line = reg_container.keyword_gdscript_export_single.sub(line, "@export", true);
			}
			if (line.contains("export")) {
				line = reg_container.keyword_gdscript_export_mutli.sub(line, "$1@export", true);
			}
			if (line.contains("onready")) {
				line = reg_container.keyword_gdscript_onready.sub(line, "@onready", true);
			}
			if (line.contains("remote")) {
				line = reg_container.keyword_gdscript_remote.sub(line, "@rpc(\"any_peer\") func", true);
			}
			if (line.contains("remote")) {
				line = reg_container.keyword_gdscript_remotesync.sub(line, "@rpc(\"any_peer\", \"call_local\") func", true);
			}
			if (line.contains("sync")) {
				line = reg_container.keyword_gdscript_sync.sub(line, "@rpc(\"any_peer\", \"call_local\") func", true);
			}
			if (line.contains("slave")) {
				line = reg_container.keyword_gdscript_slave.sub(line, "@rpc func", true);
			}
			if (line.contains("puppet")) {
				line = reg_container.keyword_gdscript_puppet.sub(line, "@rpc func", true);
			}
			if (line.contains("puppet")) {
				line = reg_container.keyword_gdscript_puppetsync.sub(line, "@rpc(\"call_local\") func", true);
			}
			if (line.contains("master")) {
				line = reg_container.keyword_gdscript_master.sub(line, error_message + "@rpc func", true);
			}
			if (line.contains("master")) {
				line = reg_container.keyword_gdscript_mastersync.sub(line, error_message + "@rpc(\"call_local\") func", true);
			}
		}
	}
}

Vector<String> ProjectConverter3To4::check_for_rename_gdscript_keywords(Vector<String> &lines, const RegExContainer &reg_container) {
	Vector<String> found_renames;

	int current_line = 1;
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			String old;

			if (line.contains("tool")) {
				old = line;
				line = reg_container.keyword_gdscript_tool.sub(line, "@tool", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "tool", "@tool", line));
				}
			}

			if (line.contains("export")) {
				old = line;
				line = reg_container.keyword_gdscript_export_single.sub(line, "$1@export", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "export", "@export", line));
				}
			}

			if (line.contains("export")) {
				old = line;
				line = reg_container.keyword_gdscript_export_mutli.sub(line, "@export", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "export", "@export", line));
				}
			}

			if (line.contains("onready")) {
				old = line;
				line = reg_container.keyword_gdscript_tool.sub(line, "@onready", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "onready", "@onready", line));
				}
			}

			if (line.contains("remote")) {
				old = line;
				line = reg_container.keyword_gdscript_remote.sub(line, "@rpc(\"any_peer\") func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "remote func", "@rpc(\"any_peer\") func", line));
				}
			}

			if (line.contains("remote")) {
				old = line;
				line = reg_container.keyword_gdscript_remotesync.sub(line, "@rpc(\"any_peer\", \"call_local\")) func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "remotesync func", "@rpc(\"any_peer\", \"call_local\")) func", line));
				}
			}

			if (line.contains("sync")) {
				old = line;
				line = reg_container.keyword_gdscript_sync.sub(line, "@rpc(\"any_peer\", \"call_local\")) func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "sync func", "@rpc(\"any_peer\", \"call_local\")) func", line));
				}
			}

			if (line.contains("slave")) {
				old = line;
				line = reg_container.keyword_gdscript_slave.sub(line, "@rpc func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "slave func", "@rpc func", line));
				}
			}

			if (line.contains("puppet")) {
				old = line;
				line = reg_container.keyword_gdscript_puppet.sub(line, "@rpc func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "puppet func", "@rpc func", line));
				}
			}

			if (line.contains("puppet")) {
				old = line;
				line = reg_container.keyword_gdscript_puppetsync.sub(line, "@rpc(\"call_local\") func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "puppetsync func", "@rpc(\"call_local\") func", line));
				}
			}

			if (line.contains("master")) {
				old = line;
				line = reg_container.keyword_gdscript_master.sub(line, "@rpc func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "master func", "@rpc func", line));
				}
			}

			if (line.contains("master")) {
				old = line;
				line = reg_container.keyword_gdscript_master.sub(line, "@rpc(\"call_local\") func", true);
				if (old != line) {
					found_renames.append(line_formatter(current_line, "mastersync func", "@rpc(\"call_local\") func", line));
				}
			}
		}
		current_line++;
	}

	return found_renames;
}

void ProjectConverter3To4::custom_rename(Vector<String> &lines, String from, String to) {
	RegEx reg = RegEx(String("\\b") + from + "\\b");
	CRASH_COND(!reg.is_valid());
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			line = reg.sub(line, to, true);
		}
	}
};

Vector<String> ProjectConverter3To4::check_for_custom_rename(Vector<String> &lines, String from, String to) {
	Vector<String> found_renames;

	RegEx reg = RegEx(String("\\b") + from + "\\b");
	CRASH_COND(!reg.is_valid());

	int current_line = 1;
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			TypedArray<RegExMatch> reg_match = reg.search_all(line);
			if (reg_match.size() > 0) {
				found_renames.append(line_formatter(current_line, from.replace("\\.", "."), to, line)); // Without replacing it will print "\.shader" instead ".shader".
			}
		}
		current_line++;
	}
	return found_renames;
}

void ProjectConverter3To4::rename_common(const char *array[][2], LocalVector<RegEx *> &cached_regexes, Vector<String> &lines) {
	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			for (unsigned int current_index = 0; current_index < cached_regexes.size(); current_index++) {
				if (line.contains(array[current_index][0])) {
					line = cached_regexes[current_index]->sub(line, array[current_index][1], true);
				}
			}
		}
	}
}

Vector<String> ProjectConverter3To4::check_for_rename_common(const char *array[][2], LocalVector<RegEx *> &cached_regexes, Vector<String> &lines) {
	Vector<String> found_renames;

	int current_line = 1;

	for (String &line : lines) {
		if (uint64_t(line.length()) <= maximum_line_length) {
			for (unsigned int current_index = 0; current_index < cached_regexes.size(); current_index++) {
				if (line.contains(array[current_index][0])) {
					TypedArray<RegExMatch> reg_match = cached_regexes[current_index]->search_all(line);
					if (reg_match.size() > 0) {
						found_renames.append(line_formatter(current_line, array[current_index][0], array[current_index][1], line));
					}
				}
			}
		}
		current_line++;
	}

	return found_renames;
}

// Prints full info about renamed things e.g.:
// Line (67) remove -> remove_at  -  LINE """ doubler._blacklist.remove(0) """
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

	from = from.strip_escapes();
	to = to.strip_escapes();
	line = line.replace("\r", "").replace("\n", "").strip_edges();

	return vformat("Line(%d), %s -> %s  -  LINE \"\"\" %s \"\"\"", current_line, from, to, line);
}

// Prints only full lines e.g.:
// Line (1) - FULL LINES - """yield(get_tree().create_timer(3), 'timeout')"""  =====>  """ await get_tree().create_timer(3).timeout """
String ProjectConverter3To4::simple_line_formatter(int current_line, String old_line, String new_line) {
	if (old_line.size() > 1000) {
		old_line = old_line.substr(0, 997) + "...";
	}
	if (new_line.size() > 1000) {
		new_line = new_line.substr(0, 997) + "...";
	}

	old_line = old_line.replace("\r", "").replace("\n", "").strip_edges();
	new_line = new_line.replace("\r", "").replace("\n", "").strip_edges();

	return vformat("Line (%d) - FULL LINES - \"\"\" %s \"\"\"  =====>  \"\"\" %s \"\"\"", current_line, old_line, new_line);
}

// Collects string from vector strings
String ProjectConverter3To4::collect_string_from_vector(Vector<String> &vector) {
	String string = "";
	for (int i = 0; i < vector.size(); i++) {
		string += vector[i];

		if (i != vector.size() - 1) {
			string += "\n";
		}
	}
	return string;
}

#else // No RegEx.

ProjectConverter3To4::ProjectConverter3To4(int _p_maximum_file_size_kb, int _p_maximum_line_length) {}

int ProjectConverter3To4::convert() {
	ERR_FAIL_V_MSG(ERROR_CODE, "Can't run converter for Godot 3.x projects, because RegEx module is disabled.");
}

int ProjectConverter3To4::validate_conversion() {
	ERR_FAIL_V_MSG(ERROR_CODE, "Can't validate conversion for Godot 3.x projects, because RegEx module is disabled.");
}

#endif // MODULE_REGEX_ENABLED
