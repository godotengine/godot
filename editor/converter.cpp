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

#ifdef TOOLS_ENABLED

#include "core/core_bind.h"
#include "core/io/file_access.h"
#include "core/string/print_string.cpp"
#include "core/string/ustring.h"
#include "modules/regex/regex.h"

// TODO add Regex as required dependency

// TODO maybe cache all compiled regex

// TODO Remove from _init, return type(this works in 3.x but not in 4.0)

// TODO convert connect to Callable
static const char *enum_renames[][2] = {
	{ "TYPE_QUAT", "TYPE_QUATERNION" },
	{ "TYPE_REAL", "TYPE_FLOAT" },
	{ "TYPE_TRANSFORM", "TYPE_TRANSFORM3D" },
	{ "TYPE_INT_ARRAY", "TYPE_INT64_ARRAY" },
	{ "TYPE_REAL_ARRAY", "TYPE_FLOAT64_ARRAY" },

	// {"FLAG_MAX", "PARTICLE_FLAG_MAX"}, // CPUParticles2D - used in more classes
	{ "ARRAY_COMPRESS_BASE", "ARRAY_COMPRESS_FLAGS_BASE" }, //Mesh
	{ "ARVR_AR", "XR_AR" }, //XRInterface
	{ "ARVR_EXCESSIVE_MOTION", "XR_EXCESSIVE_MOTION" }, //XRInterface
	{ "ARVR_EXTERNAL", "XR_EXTERNAL" }, //XRInterface
	{ "ARVR_INSUFFICIENT_FEATURES", "XR_INSUFFICIENT_FEATURES" }, //XRInterface
	{ "ARVR_MONO", "XR_MONO" }, //XRInterface
	{ "ARVR_NONE", "XR_NONE" }, //XRInterface
	{ "ARVR_NORMAL_TRACKING", "XR_NORMAL_TRACKING" }, //XRInterface
	{ "ARVR_NOT_TRACKING", "XR_NOT_TRACKING" }, //XRInterface
	{ "ARVR_STEREO", "XR_STEREO" }, //XRInterface
	{ "ARVR_UNKNOWN_TRACKING", "XR_UNKNOWN_TRACKING" }, //XRInterface
	{ "BAKE_ERROR_INVALID_MESH", "BAKE_ERROR_MESHES_INVALID" }, //LightmapGI
	{ "BODY_MODE_CHARACTER", "BODY_MODE_DYNAMIC" }, //PhysicsServer2D
	{ "BUTTON_LEFT", "MOUSE_BUTTON_LEFT" }, //Globals
	{ "BUTTON_MASK_LEFT", "MOUSE_BUTTON_MASK_LEFT" }, //Globals
	{ "BUTTON_MASK_MIDDLE", "MOUSE_BUTTON_MASK_MIDDLE" }, //Globals
	{ "BUTTON_MASK_RIGHT", "MOUSE_BUTTON_MASK_RIGHT" }, //Globals
	{ "BUTTON_MASK_XBUTTON1", "MOUSE_BUTTON_MASK_XBUTTON1" }, //Globals
	{ "BUTTON_MASK_XBUTTON2", "MOUSE_BUTTON_MASK_XBUTTON2" }, //Globals
	{ "BUTTON_MIDDLE", "MOUSE_BUTTON_MIDDLE" }, //Globals
	{ "BUTTON_RIGHT", "MOUSE_BUTTON_RIGHT" }, //Globals
	{ "BUTTON_WHEEL_DOWN", "MOUSE_BUTTON_WHEEL_DOWN" }, //Globals
	{ "BUTTON_WHEEL_LEFT", "MOUSE_BUTTON_WHEEL_LEFT" }, //Globals
	{ "BUTTON_WHEEL_RIGHT", "MOUSE_BUTTON_WHEEL_RIGHT" }, //Globals
	{ "BUTTON_WHEEL_UP", "MOUSE_BUTTON_WHEEL_UP" }, //Globals
	{ "BUTTON_XBUTTON1", "MOUSE_BUTTON_XBUTTON1" }, //Globals
	{ "BUTTON_XBUTTON2", "MOUSE_BUTTON_XBUTTON2" }, //Globals
	{ "COMPRESS_PVRTC4", "COMPRESS_PVRTC1_4" }, //Image
	{ "CUBEMAP_BACK", "CUBEMAP_LAYER_BACK" }, //RenderingServer
	{ "CUBEMAP_BOTTOM", "CUBEMAP_LAYER_BOTTOM" }, //RenderingServer
	{ "CUBEMAP_FRONT", "CUBEMAP_LAYER_FRONT" }, //RenderingServer
	{ "CUBEMAP_LEFT", "CUBEMAP_LAYER_LEFT" }, //RenderingServer
	{ "CUBEMAP_RIGHT", "CUBEMAP_LAYER_RIGHT" }, //RenderingServer
	{ "CUBEMAP_TOP", "CUBEMAP_LAYER_TOP" }, //RenderingServer
	{ "DAMPED_STRING_DAMPING", "DAMPED_SPRING_DAMPING" }, //PhysicsServer2D
	{ "DAMPED_STRING_REST_LENGTH", "DAMPED_SPRING_REST_LENGTH" }, //PhysicsServer2D
	{ "DAMPED_STRING_STIFFNESS", "DAMPED_SPRING_STIFFNESS" }, //PhysicsServer2D
	{ "FLAG_ALIGN_Y_TO_VELOCITY", "PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY" }, // CPUParticles2D
	{ "FLAG_DISABLE_Z", "PARTICLE_FLAG_DISABLE_Z" }, // CPUParticles2D
	{ "FLAG_ROTATE_Y", "PARTICLE_FLAG_ROTATE_Y" }, // CPUParticles2D
	{ "FLAG_USE_BAKED_LIGHT", "GI_MODE_BAKED" }, // GeometryInstance3D
	{ "FORMAT_PVRTC2", "FORMAT_PVRTC1_2" }, //Image
	{ "FORMAT_PVRTC2A", "FORMAT_PVRTC1_2A" }, //Image
	{ "FORMAT_PVRTC4", "FORMAT_PVRTC1_4" }, //Image
	{ "FORMAT_PVRTC4A", "FORMAT_PVRTC1_4A" }, //Image
	{ "INSTANCE_LIGHTMAP_CAPTURE", "INSTANCE_LIGHTMAP" }, //RenderingServer
	{ "JOINT_6DOF", "JOINT_TYPE_6DOF" }, //PhysicsServer3D
	{ "JOINT_CONE_TWIST", "JOINT_TYPE_CONE_TWIST" }, //PhysicsServer3D
	{ "JOINT_DAMPED_SPRING", "JOINT_TYPE_DAMPED_SPRING" }, //PhysicsServer2D
	{ "JOINT_GROOVE", "JOINT_TYPE_GROOVE" }, //PhysicsServer2D
	{ "JOINT_HINGE", "JOINT_TYPE_HINGE" }, //PhysicsServer3D
	{ "JOINT_PIN", "JOINT_TYPE_PIN" }, //PhysicsServer2D
	{ "JOINT_SLIDER", "JOINT_TYPE_SLIDER" }, //PhysicsServer3D
	{ "KEY_CONTROL", "KEY_CTRL" }, // Globals
	{ "MATH_STEPIFY", "MATH_STEP_DECIMALS" }, //VisualScriptBuiltinFunc
	{ "NOTIFICATION_APP_PAUSED", "NOTIFICATION_APPLICATION_PAUSED" }, //MainLoop
	{ "NOTIFICATION_APP_RESUMED", "NOTIFICATION_APPLICATION_RESUMED" }, //MainLoop
	{ "NOTIFICATION_WM_FOCUS_IN", "NOTIFICATION_APPLICATION_FOCUS_IN" }, //MainLoop
	{ "NOTIFICATION_WM_FOCUS_OUT", "NOTIFICATION_APPLICATION_FOCUS_OUT" }, //MainLoop
	{ "RENDER_DRAW_CALLS_IN_FRAME", "RENDER_TOTAL_DRAW_CALLS_IN_FRAME" }, //Performance
	{ "RENDER_OBJECTS_IN_FRAME", "RENDER_TOTAL_OBJECTS_IN_FRAME" }, //Performance
	{ "SIDE_BOTTOM", "MARGIN_BOTTOM" }, //Globals
	{ "SIDE_LEFT", "MARGIN_LEFT" }, //Globals
	{ "SIDE_RIGHT", "MARGIN_RIGHT" }, //Globals
	{ "SIDE_TOP", "MARGIN_TOP" }, //Globals
	{ "TRACKER_LEFT_HAND", "TRACKER_HAND_LEFT" }, //XRPositionalTracker
	{ "TRACKER_RIGHT_HAND", "TRACKER_HAND_RIGHT" }, //XRPositionalTracker
	{ "TYPE_NORMALMAP", "TYPE_NORMAL_MAP" }, //VisualShaderNodeCubemap

	{ nullptr, nullptr },
};

// Simple renaming functions - "function1" -> "function2"
// Do not add functions which are named same in multiple classes like "start", because this will broke other functions, also
static const char *function_renames[][2] = {
	{ "add_cancel", "add_cancel_button" }, // AcceptDialog
	{ "get_ok", "get_ok_button" }, // AcceptDialog
	{ "track_remove_key_at_position", "track_remove_key_at_time" }, // Animation
	{ "get_audio_bus", "get_audio_bus_name" }, // Area3D
	{ "set_audio_bus", "set_audio_bus_name" }, // Area3D
	{ "regen_normalmaps", "regen_normal_maps" }, // ArrayMesh
	{ "surface_update_region", "surface_update_attribute_region" }, //ArrayMesh
	{ "set_global_rate_scale", "set_playback_speed_scale" }, // AudioServer
	// { "_unhandled_input", "_unhandled_key_input"}, // BaseButton - Used also in EditorFileDialog
	//{ "set_extents", "set_size"}, // BoxShape3D - Do not use, other classes also uses this function
	//{ "get_extents", "get_size"}, // BoxShape3D - Do not use, other classes also uses this function
	{ "get_collision_layer_bit", "get_collision_layer_value" }, // CSGShape3D and a lot of others like GridMap
	{ "get_collision_mask_bit", "get_collision_mask_value" }, // CSGShape3D and a lot of others like GridMap
	{ "set_collision_layer_bit", "set_collision_layer_value" }, // CSGShape3D and a lot of others like GridMap
	{ "set_collision_mask_bit", "set_collision_mask_value" }, // CSGShape3D and a lot of others like GridMap
	{ "get_cull_mask_bit", "get_cull_mask_value" }, // Camera3D
	{ "set_cull_mask_bit", "set_cull_mask_value" }, // Camera3D
	{ "_toplevel_raise_self", "_top_level_raise_self" }, // CanvasItem
	{ "is_set_as_toplevel", "is_set_as_top_level" }, // CanvasItem
	{ "set_as_toplevel", "set_as_top_level" }, // CanvasItem
	{ "get_mid_height", "get_height" }, // CapsuleMesh
	{ "set_mid_height", "set_height" }, // CapsuleMesh
	{ "make_convex_from_brothers", "make_convex_from_siblings" }, // CollisionShape3D
	{ "get_frame_color", "get_color" }, // ColorRect
	{ "set_frame_color", "set_color" }, // ColorRect
	{ "get_cancel", "get_cancel_button" }, // ConfirmationDialog
	{ "add_color_override", "add_theme_color_override" }, // Control
	{ "add_constant_override", "add_theme_constant_override" }, // Control
	{ "add_font_override", "add_theme_font_override" }, // Control
	{ "add_icon_override", "add_theme_icon_override" }, // Control
	{ "add_stylebox_override", "add_theme_stylebox_override" }, // Control
	{ "get_focus_neighbour", "get_focus_neighbor" }, //  Control
	{ "has_font", "has_theme_font" }, //  Control
	{ "has_font_override", "has_theme_font_override" }, //  Control
	{ "has_icon", "has_theme_icon" }, //  Control
	{ "has_icon_override", "has_theme_icon_override" }, //  Control
	{ "has_stylebox", "has_theme_stylebox" }, //
	{ "has_stylebox_override", "has_theme_stylebox_override" }, //  Control
	{ "set_anchor_and_margin", "set_anchor_and_offset" }, //  Control
	{ "set_anchors_and_margins_preset", "set_anchors_and_offsets_preset" }, //  Control
	{ "set_focus_neighbour", "set_focus_neighbor" }, //  Control
	// { "get_mode", "get_file_mode"}, // EditorFileDialog - Used elsewhere
	{ "set_adjustment_enable", "set_adjustment_enabled" }, // Environment
	{ "deselect_items", "deselect_all" }, // FileDialog
	{ "bumpmap_to_normalmap", "bump_map_to_normal_map" }, // Image
	{ "normalmap_to_xy", "normal_map_to_xy" }, // Image
	// { "set_color", "surface_set_color"}, // ImmediateMesh - Do not use, other classes also uses this function
	// { "set_normal", "surface_set_normal"}, // ImmediateMesh - Do not use, other classes also uses this function
	// { "set_tangent", "surface_set_tangent"}, // ImmediateMesh - Do not use, other classes also uses this function
	// { "set_uv2", "surface_set_uv2"}, // ImmediateMesh - Do not use, other classes also uses this function
	// { "set_uv", "surface_set_uv"}, // ImmediateMesh - Do not use, other classes also uses this function
	{ "get_physical_scancode", "get_physical_keycode" }, // InputEventKey
	{ "get_physical_scancode_with_modifiers", "get_physical_keycode_with_modifiers" }, // InputEventKey
	{ "get_scancode", "get_keycode" }, // InputEventKey
	{ "get_scancode_with_modifiers", "get_keycode_with_modifiers" }, // InputEventKey
	{ "set_physical_scancode", "set_physical_keycode" }, // InputEventKey
	{ "set_scancode", "set_keycode" }, // InputEventKey
	{ "is_doubleclick", "is_double_click" }, // InputEventMouseButton
	{ "set_doubleclick", "set_double_click" }, // InputEventMouseButton
	{ "get_alt", "is_alt_pressed" }, // InputEventWithModifiers
	{ "get_command", "is_command_pressed" }, // InputEventWithModifiers
	{ "get_control", "is_ctrl_pressed" }, // InputEventWithModifiers
	{ "get_metakey", "is_meta_pressed" }, // InputEventWithModifiers
	{ "get_shift", "is_shift_pressed" }, // InputEventWithModifiers
	{ "set_alt", "set_alt_pressed" }, // InputEventWithModifiers
	{ "set_command", "set_command_pressed" }, // InputEventWithModifiers
	{ "set_control", "set_ctrl_pressed" }, // InputEventWithModifiers
	{ "set_metakey", "set_meta_pressed" }, // InputEventWithModifiers
	{ "set_shift", "set_shift_pressed" }, // InputEventWithModifiers
	{ "load_from_globals", "load_from_project_settings" }, // InputMap
	{ "get_error_string", "get_error_message" }, // JSON
	{ "set_autowrap", "set_autowrap_mode" }, // Label
	{ "get_max_atlas_size", "get_max_texture_size" }, // LightmapGI
	{ "set_max_atlas_size", "set_max_texture_size" }, // LightmapGI
	{ "delete_char_at_cursor", "delete_char_at_caret" }, // LineEdit
	{ "set_expand_to_text_length", "set_expand_to_text_length_enabled" }, // LineEdit
	{ "get_surface_material", "get_surface_override_material" }, // MeshInstance3D
	{ "get_surface_material_count", "get_surface_override_material_count" }, // MeshInstance3D
	{ "set_surface_material", "set_surface_override_material" }, // MeshInstance3D
	{ "is_sort_enabled", "is_y_sort_enabled" }, // Node2D
	{ "set_sort_enabled", "set_y_sort_enabled" }, // Node2D
	{ "get_parent_spatial", "get_parent_node_3d" }, // Node3D
	{ "get_world", "get_world_3d" }, // Node3D
	{ "get_translation", "get_position" }, // Node3D - this broke GLTFNode which is used rarely
	{ "set_translation", "set_position" }, // Node3D - this broke GLTFNode which is used rarely
	{ "update_gizmo", "update_gizmos" }, // Node3D
	{ "_get_configuration_warning", "_get_configuration_warnings" }, // Node
	{ "add_child_below_node", "add_sibling" }, // Node
	{ "is_a_parent_of", "is_ancestor_of" }, // Node
	{ "update_configuration_warning", "update_configuration_warnings" }, // Node
	{ "is_normalmap", "is_normal_map" }, // NoiseTexture
	{ "set_as_normalmap", "set_as_normal_map" }, // NoiseTexture
	{ "property_list_changed_notify", "notify_property_list_changed" }, // Object
	{ "can_instance", "can_instantiate" }, // PackedScene, Script
	{ "instance", "instantiate" }, // PackedScene - Broke FileSystemDock signal
	// { "set_flag", "set_particle_flag"}, // ParticlesMaterial  - Do not use, other classes also uses this function
	{ "set_rotate", "set_rotates" }, // PathFollow2D
	{ "is_static_body", "is_able_to_sleep" }, // PhysicalBone3D - TODO - not sure
	{ "damped_string_joint_get_param", "damped_spring_joint_get_param" }, // PhysicsServer2D
	{ "damped_string_joint_set_param", "damped_spring_joint_set_param" }, // PhysicsServer2D
	{ "joint_create_cone_twist", "joint_make_cone_twist" }, // PhysicsServer3D
	{ "joint_create_generic_6dof", "joint_make_generic_6dof" }, //PhysicsServer3D
	{ "joint_create_hinge", "joint_make_hinge" }, //PhysicsServer3D
	{ "joint_create_pin", "joint_make_pin" }, //PhysicsServer3D
	{ "joint_create_slider", "joint_make_slider" }, // PhysicsServer3D
	{ "get_render_info", "get_rendering_info" }, // RenderingServer
	{ "instance_set_surface_material", "instance_set_surface_override_material" }, // RenderingServer
	{ "mesh_surface_get_format", "mesh_surface_get_format_attribute_stride" }, // RenderingServer
	{ "mesh_surface_update_region", "mesh_surface_update_attribute_region" }, // RenderingServer
	{ "multimesh_allocate", "multimesh_allocate_data" }, // RenderingServer
	{ "skeleton_allocate", "skeleton_allocate_data" }, // RenderingServer
	{ "get_dependencies", "_get_dependencies" }, // ResourceFormatLoader
	{ "get_recognized_extensions", "_get_recognized_extensions" }, // ResourceFormatLoader
	{ "get_resource_type", "_get_resource_type" }, // ResourceFormatLoader
	{ "handles_type", "_handles_type" }, // ResourceFormatLoader
	// { "load", "_load"}, // ResourceFormatLoader  - Do not use, other classes also uses this function
	{ "rename_dependencies", "_rename_dependencies" }, // ResourceFormatLoader
	// { "get_recognized_extensions", "_get_recognized_extensions"}, // ResourceFormatLoader - Used in Resource Saver
	// { "_get_recognized_extensions", "get_recognized_extensions"}, // ResourceSaver - Used in ResourceFormatLoader
	{ "recognize", "_recognize" }, // ResourceFormatLoader
	//{ "save", "_save"}, // ResourceFormatLoader - Do not use, other classes also uses this function
	{ "get_shortcut", "get_event" }, // Shortcut
	{ "is_shortcut", "matches_event" }, // Shortcut
	{ "set_shortcut", "set_event" }, // Shortcut
	{ "is_region", "is_region_enabled" }, // Sprite2D
	// { "set_region", "set_region_enabled"}, // Sprite2D - Do not use, used by AtlasTexture
	{ "set_region_filter_clip", "set_region_filter_clip_enabled" }, // Sprite2D
	{ "get_size_override", "get_size_2d_override" }, //SubViewport
	{ "is_size_override_stretch_enabled", "is_size_2d_override_stretch_enabled" }, //SubViewport
	{ "set_size_override", "set_size_2d_override" }, //SubViewport
	{ "set_size_override_stretch", "set_size_2d_override_stretch" }, //SubViewport
	{ "center_viewport_to_cursor", "center_viewport_to_caret" }, // TextEdit
	{ "_update_wrap_at", "_update_wrap_at_column" }, //TextEdit
	{ "cursor_get_blink_speed", "get_caret_blink_speed" }, // TextEdit
	{ "cursor_get_column", "get_caret_column" }, // TextEdit
	{ "cursor_get_line", "get_caret_line" }, // TextEdit
	{ "cursor_set_blink_enabled", "set_caret_blink_enabled" }, // TextEdit
	{ "cursor_set_blink_speed", "set_caret_blink_speed" }, // TextEdit
	{ "cursor_set_column", "set_caret_column" }, // TextEdit
	{ "cursor_set_line", "set_caret_line" }, // TextEdit
	{ "get_color_types", "get_color_type_list" }, // Theme
	{ "get_constant_types", "get_constant_type_list" }, // Theme
	{ "get_font_types", "get_font_type_list" }, // Theme
	{ "get_icon_types", "get_icon_type_list" }, // Theme
	{ "get_stylebox_types", "get_stylebox_type_list" }, // Theme
	{ "get_theme_item_types", "get_theme_item_type_list" }, // Theme
	{ "get_used_cells_by_id", "get_used_cells" }, // TileMap
	{ "get_timer_process_mode", "get_timer_process_callback" }, //Timer
	{ "set_timer_process_mode", "set_timer_process_callback" }, //Timer
	{ "is_commiting_action", "is_committing_action" }, // UndoRedo
	{ "get_d", "get_distance" }, //WorldMarginShape2D
	{ "set_d", "set_distance" }, //WorldMarginShape2D
	{ "get_tracks_orientation", "is_tracking_orientation" }, // XRPositionalTracker
	{ "get_tracks_position", "is_tracking_position" }, //XRPositionalTracker
	// { "set_autowrap_mode", "set_autowrap"}, // AcceptDialog - Used also in Label
	{ "get_caption", "_get_caption" }, //AnimationNode
	{ "get_child_by_name", "_get_child_by_name" }, //AnimationNode
	{ "get_child_nodes", "_get_child_nodes" }, //AnimationNode
	{ "get_parameter_default_value", "_get_parameter_default_value" }, //AnimationNode
	{ "get_parameter_list", "_get_parameter_list" }, //AnimationNode
	{ "has_filter", "_has_filter" }, //AnimationNode
	{ "process", "_process" }, // AnimationNode
	{ "get_animation_process_mode", "get_process_callback" }, // AnimationPlayer
	{ "set_animation_process_mode", "set_process_callback" }, //AnimationPlayer
	{ "get_global_rate_scale", "get_playback_speed_scale" }, // AudioServer
	// { "get_event", "get_shortcut"}, // BaseButton - Used also in Shortcut
	// { "set_event", "set_shortcut"}, // BaseButton - Used also in Shortcut
	{ "get_enabled_focus_mode", "get_shortcut_context" }, //BaseButton
	{ "set_enabled_focus_mode", "set_shortcut_context" }, //BaseButton
	{ "is_v_drag_enabled", "is_drag_vertical_enabled" }, // Camera2D
	{ "is_h_drag_enabled", "is_drag_horizontal_enabled" }, //Camera2D
	// { "get_h_offset", "get_drag_horizontal_offset"}, // Used in PathFollow2D
	// { "get_v_offset", "get_drag_vertical_offset"}, // Used in PathFollow2D
	{ "set_process_mode", "set_process_callback" }, // AnimationTree
	{ "get_process_mode", "get_process_callback" }, // ClippedCamera3D
	{ "get_zfar", "get_far" }, // Camera3D
	{ "get_znear", "get_near" }, // Camera3D
	{ "set_zfar", "set_far" }, //Camera3D
	{ "set_znear", "set_near" }, //Camera3D
	{ "commit_handle", "_commit_handle" }, //EditorNode3DGizmo
	{ "get_handle_name", "_get_handle_name" }, //EditorNode3DGizmo
	{ "get_handle_value", "_get_handle_value" }, //EditorNode3DGizmo
	{ "set_ssao_edge_sharpness", "set_ssao_sharpness" }, //Environment
	{ "set_tonemap_auto_exposure", "set_tonemap_auto_exposure_enabled" }, //Environment
	{ "get_endian_swap", "is_big_endian" }, // File
	{ "get_len", "get_length" }, //File
	{ "set_endian_swap", "set_big_endian" }, //File
	{ "set_font_path", "set_data_path" }, //FontData
	{ "get_font_path", "get_data_path" }, //FontData
	{ "get_action_list", "action_get_events" }, // InputMap
	{ "unselect", "deselect" }, //ItemList
	{ "unselect_all", "deselect_all" }, //ItemList
	{ "find_scancode_from_string", "find_keycode_from_string" }, // OS
	{ "get_scancode_string", "get_keycode_string" }, //OS
	{ "is_scancode_unicode", "is_keycode_unicode" }, //OS
	// { "listen", "bound"}, // PacketPeerUDP - Used in TCPServer
	// { "is_listening", "is_bound"}, // PacketPeerUDP - Used in TCPServer
	{ "damped_spring_joint_create", "joint_make_damped_spring" }, // PhysicsServer2D
	{ "groove_joint_create", "joint_make_groove" }, //PhysicsServer2D
	{ "pin_joint_create", "joint_make_pin" }, //PhysicsServer2D
	{ "has_valid_event", "is_valid" }, // RegEx - broke Shortcut, but probably regex is more used
	{ "canvas_light_set_scale", "canvas_light_set_texture_scale" }, //RenderingServer
	{ "viewport_set_use_arvr", "viewport_set_use_xr" }, //RenderingServer
	{ "get_drag_data_fw", "_get_drag_data_fw" }, //ScriptEditor
	{ "drop_data_fw", "_drop_data_fw" }, //ScriptEditor
	{ "can_drop_data_fw", "_can_drop_data_fw" }, //ScriptEditor
	{ "bind_child_node_to_bone", "set_bone_children" }, //Skeleton3D
	{ "get_bound_child_nodes_to_bone", "get_bone_children" }, //Skeleton3D
	{ "unbind_child_node_from_bone", "remove_bone_child" }, //Skeleton3D
	{ "move_to_top", "move_before" }, //Skeleton3D
	{ "move_to_bottom", "move_after" }, //Skeleton3D
	{ "get_hand", "get_tracker_hand" }, //XRPositionalTracker
	// { "get_name", "get_tracker_name"}, //XRPositionalTracker - Node use this
	// { "get_type", "get_tracker_type"}, //XRPositionalTracker - GLTFAccessor use this
	// { "_set_name", "get_tracker_name"}, // XRPositionalTracker - CameraFeed use this
	{ nullptr, nullptr },
};

static const char *properties_renames[][2] = {
	// {"Skeleton3D","Skeleton"}, // Polygon2D - this would rename also classes
	// {"rotate","rotates"}, // PathFollow2D - probably function exists with same name
	{ "Debug Shape3D", "Debug Shape" }, // RayCast3D
	{ "Emission Shape3D", "Emission Shape" }, // ParticlesMaterial
	{ "caret_moving_by_right_click", "caret_move_on_right_click" }, // TextEdit
	{ "d", "disatance" }, //WorldMarginShape2D
	{ "global_rate_scale", "playback_speed_scale" }, // AudioServer
	{ "group", "button_group" }, // BaseButton
	{ "region_filter_clip", "region_filter_clip_enabled" }, // Sprite2D
	{ "syntax_highlighting", "syntax_highlighter" }, // TextEdit
	{ "translation", "position" }, // Node3D - broke GLTFNode
	{ nullptr, nullptr },
};

static const char *signals_renames[][2] = {
	// { "hide", "hidden" }, // CanvasItem - function with same name exists
	{ "button_release", "button_released" }, // XRController3D
	{ "node_unselected", "node_deselected" }, // GraphEdit
	{ "offset_changed", "position_offset_changed" }, //GraphNode
	{ "tab_close", "tab_closed" }, //TextEdit
	{ "tab_hover", "tab_hovered" }, //TextEdit
	{ "text_entered", "text_submitted" }, // LineEdit
	{ nullptr, nullptr },
};

static const char *project_settings_renames[][2] = {
	{ "audio/channel_disable_threshold_db ", "audio/buses/channel_disable_threshold_db " },
	{ "audio/channel_disable_time ", "audio/buses/channel_disable_time " },
	{ "audio/default_bus_layout ", "audio/buses/default_bus_layout" },
	{ "audio/driver", "audio/driver/driver" },
	{ "audio/enable_audio_input ", "audio/driver/enable_input" },
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
	{ "rendering/quality/shading/force_blinn_over_ggx", "rendering/shading/overrides/force_blinn_over_ggx.mobile" },
	{ "rendering/quality/shading/force_blinn_over_ggx.mobile", "rendering/shading/overrides/force_lambert_over_burley" },
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

static const char *shaders_renames[][2] = {
	{ "NORMALMAP", "NORMAL_MAP" },
	{ "NORMALMAP_DEPTH", "NORMAL_MAP_DEPTH" },
	{ nullptr, nullptr },
};

static const char *gdscript_keywords_renames[][2] = {
	{ "onready", "@onready" },
	{ "export", "@export" },
	{ "tool", "@tool" },
	{ nullptr, nullptr },
};

static const char *class_renames[][2] = {
	{ "ARVRAnchor", "XRAnchor3D" },
	{ "ARVRCamera", "XRCamera3D" },
	{ "ARVRController", "XRController3D" },
	{ "ARVRInterface", "XRInterface" },
	{ "ARVRInterfaceGDNative", "Node3D" },
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
	{ "CubeMap", "Cubemap" },
	{ "CubeMesh", "BoxMesh" },
	{ "CylinderShape", "CylinderShape3D" },
	{ "DirectionalLight", "DirectionalLight3D" },
	{ "DynamicFont", "Font" },
	{ "DynamicFontData", "FontData" },
	{ "EditorSpatialGizmo", "EditorNode3DGizmo" },
	{ "EditorSpatialGizmoPlugin", "EditorNode3DGizmoPlugin" },
	{ "ExternalTexture", "ImageTexture" },
	{ "FuncRef", "Callable" },
	{ "GDScriptFunctionState", "Node3D" },
	{ "GDScriptNativeClass", "Node3D" },
	{ "GIProbe", "VoxelGI" },
	{ "GIProbeData", "VoxelGIData" },
	{ "Generic6DOFJoint", "Generic6DOFJoint3D" },
	{ "Geometry", "Geometry2D" }, // Geometry class is split between Geometry2D and Geometry3D so we need to choose one
	{ "GeometryInstance", "GeometryInstance3D" },
	{ "HeightMapShape", "HeightMapShape3D" },
	{ "HingeJoint", "HingeJoint3D" },
	{ "IP_Unix", "IPUnix" },
	{ "ImmediateGeometry", "ImmediateGeometry3D" },
	{ "ImmediateGeometry3D", "ImmediateMesh" },
	{ "InterpolatedCamera", "InterpolatedCamera3D" },
	{ "InterpolatedCamera3D", "Camera3D" },
	{ "JSONParseResult", "JSON" },
	{ "Joint", "Joint3D" },
	{ "KinematicBody", "CharacterBody3D" },
	{ "KinematicBody2D", "CharacterBody2D" },
	{ "KinematicCollision", "KinematicCollision3D" },
	{ "LargeTexture", "ImageTexture" },
	{ "Light", "Light3D" },
	{ "Light2D", "PointLight2D" },
	{ "LineShape2D", "WorldMarginShape2D" },
	{ "Listener", "Listener3D" },
	{ "MeshInstance", "MeshInstance3D" },
	{ "MultiMeshInstance", "MultiMeshInstance3D" },
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
	{ "PhysicsTestMotionResult", "PhysicsTestMotionResult2D" }, // PhysicsTestMotionResult class is split between PhysicsTestMotionResult2D and PhysicsTestMotionResult3D so we need to choose one
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
	{ "Reference", "RefCounted" },
	{ "RemoteTransform", "RemoteTransform3D" },
	{ "ResourceInteractiveLoader", "ResourceLoader" },
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
	{ "TextFile", "Node3D" },
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
	{ "WebRTCMultiplayer", "WebRTCMultiplayerPeer" },
	{ "WindowDialog", "Window" },
	{ "World", "World3D" },
	{ "XRAnchor", "XRAnchor3D" },
	{ "XRController", "XRController3D" },
	{ "XROrigin", "XROrigin3D" },
	{ "YSort", "Node2D" },

	{ "RayShape2D", "RayCast2D" }, // TODO looks that this class is not visible
	{ "RayShape", "RayCast3D" }, // TODO looks that this class is not visible

	{ "CullInstance", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "RoomGroup", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "Room", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "RoomManager", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x
	{ "Portal", "Node3D" }, // Probably this type needs to be added to Godot 4.0, since it is for now only available only in Godot 3.x

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

	// TODO Maybe it is better way to not rename gd, tscn and other files which are named are classes
	while (class_renames[current_index][0]) {
		RegEx reg_before = RegEx(String("\\b") + class_renames[current_index][0] + ".tscn\\b");
		file_content = reg_before.sub(file_content, "TEMP_RENAMED_CLASS.tscn", true);
		RegEx reg_before2 = RegEx(String("\\b") + class_renames[current_index][0] + ".gd\\b");
		file_content = reg_before2.sub(file_content, "TEMP_RENAMED_CLASS.gd", true);

		RegEx reg = RegEx(String("\\b") + class_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, class_renames[current_index][1], true);

		RegEx reg_after = RegEx("\\bTEMP_RENAMED_CLASS.tscn\\b");
		file_content = reg_after.sub(file_content, String(class_renames[current_index][0]) + ".tscn", true);
		RegEx reg_after2 = RegEx("\\bTEMP_RENAMED_CLASS.gd\\b");
		file_content = reg_after2.sub(file_content, String(class_renames[current_index][0]) + ".gd", true);
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

static void rename_signals(String &file_content) {
	int current_index = 0;
	while (signals_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + signals_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, signals_renames[current_index][1], true);
		current_index++;
	}
};

static void rename_project_settings(String &file_content) {
	int current_index = 0;
	while (project_settings_renames[current_index][0]) {
		RegEx reg = RegEx(String("\\b") + project_settings_renames[current_index][0] + "\\b");
		file_content = reg.sub(file_content, project_settings_renames[current_index][1], true);
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
				if (file_name == ".." || file_name == "." || file_name == "." || file_name == ".git" || file_name == ".import" || file_name == ".godot") {
					file_name = dir.get_next();
					continue;
				}
				if (dir.current_is_dir()) {
					directories_to_check.append(current_dir + file_name + "/");
				} else {
					bool proper_extension = false;
					// TODO enable all files
					if (file_name.ends_with(".gd") || file_name.ends_with(".shader") || file_name.ends_with(".tscn") || file_name.ends_with(".tres") || file_name.ends_with(".godot") || file_name.ends_with(".cs") || file_name.ends_with(".csproj"))
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

static bool validate_array(const char *array[][2], bool ignore_second_check = false) {
	bool valid = true;
	int current_index = 0;
	Vector<String> names = Vector<String>();

	while (array[current_index][0]) {
		if (names.has(array[current_index][0])) {
			ERR_PRINT(String("Found duplicated things, pair ( -> ") + array[current_index][0] + ", " + array[current_index][1] + ")");
			valid = false;
		}
		names.append(array[current_index][0]);

		if (names.has(array[current_index][1])) {
			ERR_PRINT(String("Found duplicated things, pair (") + array[current_index][0] + ", ->" + array[current_index][1] + ")");
			valid = false;
		}
		if (!ignore_second_check) {
			names.append(array[current_index][1]);
		}
		current_index++;
	}
	return valid;
};

static bool validate_names() {
	bool valid = true;
	Vector<String> names = Vector<String>();

	valid = valid && validate_array(enum_renames);
	valid = valid && validate_array(class_renames, true);
	valid = valid && validate_array(function_renames, true);
	valid = valid && validate_array(properties_renames);
	valid = valid && validate_array(shaders_renames);
	valid = valid && validate_array(gdscript_keywords_renames);
	valid = valid && validate_array(signals_renames);
	valid = valid && validate_array(project_settings_renames);

	return valid;
}

static void converter() {
	print_line("Starting Converting.");

	if (!validate_names()) {
		print_line("Cannot start converting due to problems.");
		return;
	}
	// Silence warning about unused converter function
	// It is used, because already function is executed
	if (false) {
		converter();
	}

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

		// TSCN must be the same work exactly same as .gd file because it may contains builtin script
		if (file_name.ends_with(".gd") || file_name.ends_with(".tscn")) {
			rename_project_settings(file_content);
			rename_gdscript_keywords(file_content);
			rename_classes(file_content);
			rename_enums(file_content);
			rename_functions(file_content);
			rename_properties(file_content);
			rename_signals(file_content);
		} else if (file_name.ends_with(".cs")) { // TODO, C# should use different methods
			rename_classes(file_content);
		} else if (file_name.ends_with(".shader")) {
			rename_shaders(file_content);
		} else if (file_name.ends_with(".csproj")) {
			//TODO
		} else if (file_name == "project.godot") {
			rename_project_settings(file_content);
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

#endif

#endif // CONVERTER_H
