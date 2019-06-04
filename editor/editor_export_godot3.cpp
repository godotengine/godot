/*************************************************************************/
/*  editor_export_godot3.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_export_godot3.h"

#include "drivers/nrex/regex.h"
#include "editor_node.h"
#include "io/resource_format_binary.h"
#include "io/resource_format_xml.h"
#include "scene/resources/scene_format_text.h"

static const char *globals_renames[][2] = {
	/* [application] */
	{ "application/name", "application/config/name" },
	{ "application/auto_accept_quit", "application/config/auto_accept_quit" },
	{ "application/boot_splash", "application/boot_splash/image" },
	{ "application/boot_splash_fullsize", "application/boot_splash/fullsize" },
	{ "application/icon", "application/config/icon" },
	{ "application/main_scene", "application/run/main_scene" },
	{ "application/main_loop_type", "application/run/main_loop_type" },
	{ "application/disable_stdout", "application/run/disable_stdout" },
	{ "application/disable_stderr", "application/run/disable_stderr" },
	{ "application/frame_delay_msec", "application/run/frame_delay_msec" },

	/* [debug] */
	{ "debug/profiler_max_functions", "debug/settings/profiler/max_functions" },
	{ "debug/max_remote_stdout_chars_per_second", "network/limits/debugger_stdout/max_chars_per_second" },
	{ "debug/force_fps", "debug/settings/fps/force_fps" },
	{ "debug/verbose_stdout", "debug/settings/stdout/verbose_stdout" },
	//{ "debug/max_texture_size", "debug/" },
	//{ "debug/max_texture_size_alert", "debug/" },
	//{ "debug/image_load_times", "debug/" },
	{ "debug/script_max_call_stack", "debug/settings/gdscript/max_call_stack" },
	{ "debug/collision_shape_color", "debug/shapes/collision/shape_color" },
	{ "debug/collision_contact_color", "debug/shapes/collision/contact_color" },
	{ "debug/navigation_geometry_color", "debug/shapes/navigation/geometry_color" },
	{ "debug/navigation_disabled_geometry_color", "debug/shapes/navigation/disabled_geometry_color" },
	{ "debug/collision_max_contacts_displayed", "debug/shapes/collision/max_contacts_displayed" },
	//{ "debug/indicators_enabled", "debug/" },
	{ "debug/print_fps", "debug/settings/stdout/print_fps" },
	//{ "debug/print_metrics", "debug/" },

	/* [display] */
	{ "display/driver", "display/driver/name" },
	{ "display/width", "display/window/size/width" },
	{ "display/height", "display/window/size/height" },
	{ "display/allow_hidpi", "display/window/dpi/allow_hidpi" },
	{ "display/fullscreen", "display/window/size/fullscreen" },
	{ "display/resizable", "display/window/size/resizable" },
	{ "display/borderless_window", "display/window/size/borderless" },
	{ "display/use_vsync", "display/window/vsync/use_vsync" },
	{ "display/test_width", "display/window/size/test_width" },
	{ "display/test_height", "display/window/size/test_height" },
	{ "display/use_2d_pixel_snap", "rendering/quality/2d/use_pixel_snap" },
	{ "display/keep_screen_on", "display/window/energy_saving/keep_screen_on" },
	{ "display/orientation", "display/window/handheld/orientation" },
	{ "display/emulate_touchscreen", "display/window/handheld/emulate_touchscreen" },
	{ "display/use_hidpi_theme", "gui/theme/use_hidpi" },
	{ "display/custom_theme", "gui/theme/custom" },
	{ "display/custom_theme_font", "gui/theme/custom_font" },
	{ "display/swap_ok_cancel", "gui/common/swap_ok_cancel" },
	{ "display/tooltip_delay", "gui/timers/tooltip_delay_sec" },
	{ "display/text_edit_idle_detect_sec", "gui/timers/text_edit_idle_detect_sec" },
	{ "display/stretch_mode", "display/window/stretch/mode" },
	{ "display/stretch_aspect", "display/window/stretch/aspect" },

	/* [render] */
	{ "render/thread_model", "rendering/threads/thread_model" },
	//{ "render/mipmap_policy", "" },
	//{ "render/thread_textures_prealloc", "" },
	//{ "render/shadows_enabled", "" },
	//{ "render/aabb_random_points", "" },
	{ "render/default_clear_color", "rendering/environment/default_clear_color" },
	//{ "render/room_cull_enabled", "" },
	//{ "render/light_discard_enabled", "" },

	/* [audio] */
	// partly unchanged
	//{ "audio/mixer_interp", "" },
	//{ "audio/use_chorus_reverb", "" },
	//{ "audio/stream_volume_scale", "" },
	//{ "audio/fx_volume_scale", "" },
	//{ "audio/event_voice_volume_scale", "" },
	//{ "audio/stream_buffering_ms", "" },
	//{ "audio/video_delay_compensation_ms", "" },
	//{ "audio/mixer_latency", "" },

	/* [physics] */
	{ "physics/fixed_fps", "physics/common/physics_fps" },
	{ "physics/remove_collision_helpers_at_runtime", "physics/" },
	{ "physics/sleep_threshold_linear", "physics/3d/sleep_threshold_linear" },
	{ "physics/sleep_threshold_angular", "physics/3d/sleep_threshold_angular" },
	{ "physics/time_before_sleep", "physics/3d/time_before_sleep" },
	{ "physics/default_gravity", "physics/3d/default_gravity" },
	{ "physics/default_gravity_vector", "physics/3d/default_gravity_vector" },
	{ "physics/default_linear_damp", "physics/3d/default_linear_damp" },
	{ "physics/default_angular_damp", "physics/3d/default_angular_damp" },
	{ "physics/enable_object_picking", "physics/common/enable_object_picking" },

	/* [core] */
	//{ "core/message_queue_size_kb", "" },
	//{ "core/rid_pool_prealloc", "" },
	//{ "core/thread_rid_pool_prealloc", "" },
	{ "core/packet_stream_peer_max_buffer_po2", "network/limits/packet_peer_stream/max_buffer_po2" },

	/* [rasterizer.Android] */
	//{ "rasterizer.Android/use_fragment_lighting", "" },
	//{ "rasterizer.Android/fp16_framebuffer", "" },

	/* [display.Android] */
	//{ "display.Android/driver", "" },

	/* [rasterizer.iOS] */
	//{ "rasterizer.iOS/use_fragment_lighting", "" },
	//{ "rasterizer.iOS/fp16_framebuffer", "" },

	/* [display.iOS] */
	//{ "display.iOS/driver", "" },
	//{ "display.iOS/use_cadisplaylink", "" },

	/* [rasterizer] */
	// most don't have an equivalent or are not meaningful to port
	{ "rasterizer/anisotropic_filter_level", "rendering/quality/filter/anisotropic_filter_level" },

	/* [physics_2d] */
	{ "physics_2d/thread_model", "physics/2d/thread_model" },
	//{ "physics_2d/motion_fix_enabled", "" },
	{ "physics_2d/sleep_threashold_linear", "physics/2d/sleep_threshold_linear" },
	{ "physics_2d/sleep_threshold_angular", "physics/2d/sleep_threshold_angular" },
	{ "physics_2d/time_before_sleep", "physics/2d/time_before_sleep" },
	{ "physics_2d/bp_hash_table_size", "physics/2d/bp_hash_table_size" },
	{ "physics_2d/cell_size", "physics/2d/cell_size" },
	{ "physics_2d/large_object_surface_treshold_in_cells", "physics/2d/large_object_surface_threshold_in_cells" },
	{ "physics_2d/default_gravity", "physics/2d/default_gravity" },
	{ "physics_2d/default_gravity_vector", "physics/2d/default_gravity_vector" },
	{ "physics_2d/default_linear_damp", "physics/2d/default_linear_damp" },
	{ "physics_2d/default_angular_damp", "physics/2d/default_angular_damp" },

	/* [image_loader] */
	//{ "image_loader/filter", "" },
	//{ "image_loader/gen_mipmaps", "" },
	//{ "image_loader/repeat", "" },

	/* [ssl] */
	{ "ssl/certificates", "network/ssl/certificates" },
	{ "ssl/config", "network/ssl/config" },

	/* [locale] */
	// no change

	/* [global] */
	{ "editor_active", "editor/active" },

	/* [editor] */
	{ "editor/main_run_args", "editor/main_run_args" },
	//{ "editor/import_shared_textures", "" },

	/* [gui] */
	{ "gui/incr_search_max_interval_msec", "gui/timers/incremental_search_max_interval_msec" },

	{ NULL, NULL }
};

static const char *prop_renames[][2] = {
	{ "anchor/bottom", "anchor_bottom" }, // Control
	{ "anchor/left", "anchor_left" }, // Control
	{ "anchor/right", "anchor_right" }, // Control
	{ "anchor/top", "anchor_top" }, // Control
	{ "bbcode/bbcode", "bbcode_text" }, // RichTextLabel
	{ "bbcode/enabled", "bbcode_enabled" }, // RichTextLabel
	{ "bias/bias", "bias" }, // Joints2D
	{ "caret/block_caret", "caret_block_mode" }, // TextEdit
	{ "caret/caret_blink", "caret_blink" }, // LineEdit, TextEdit
	{ "caret/caret_blink_speed", "caret_blink_speed" }, // LineEdit, TextEdit
	{ "cell/center_x", "cell_center_x" }, // GridMap
	{ "cell/center_y", "cell_center_y" }, // GridMap
	{ "cell/center_z", "cell_center_z" }, // GridMap
	{ "cell/custom_transform", "cell_custom_transform" }, // TileMap
	{ "cell/half_offset", "cell_half_offset" }, // TileMap
	{ "cell/octant_size", "cell_octant_size" }, // GridMap
	{ "cell/quadrant_size", "cell_quadrant_size" }, // TileMap
	{ "cell/scale", "cell_scale" }, // GridMap
	{ "cell/size", "cell_size" }, // GridMap, TileMap
	{ "cell/tile_origin", "cell_tile_origin" }, // TileMap
	{ "cell/y_sort", "cell_y_sort" }, // TileMap
	{ "collision/bounce", "collision_bounce" }, // TileMap
	//{ "collision/exclude_nodes", "disable_collision" }, // Joint, Joint2D // Joint2D can be converted, not Joint, handle manually
	{ "collision/friction", "collision_friction" }, // TileMap
	{ "collision/layers", "collision_layer" }, // Area, Area2D, PhysicsBody, PhysicsBody2D, TileMap
	{ "collision/margin", "collision/safe_margin" }, // PhysicsBody, PhysicsBody2D
	{ "collision/mask", "collision_mask" }, // Area, Area2D, PhysicsBody, PhysicsBody2D, TileMap
	{ "collision/use_kinematic", "collision_use_kinematic" }, // TileMap
	{ "config/amount", "amount" }, // Particles2D
	{ "config/emitting", "emitting" }, // Particles2D
	{ "config/explosiveness", "explosiveness" }, // Particles2D
	{ "config/h_frames", "h_frames" }, // Particles2D
	{ "config/lifetime", "lifetime" }, // Particles2D
	{ "config/local_space", "local_coords" }, // Particles2D
	{ "config/preprocess", "preprocess" }, // Particles2D
	{ "config/texture", "texture" }, // Particles2D
	{ "config/time_scale", "speed_scale" }, // Particles2D
	{ "config/v_frames", "v_frames" }, // Particles2D
	{ "content_margin/bottom", "content_margin_bottom" }, // StyleBox
	{ "content_margin/left", "content_margin_left" }, // StyleBox
	{ "content_margin/right", "content_margin_right" }, // StyleBox
	{ "content_margin/top", "content_margin_top" }, // StyleBox
	{ "damping/compression", "damping_compression" }, // VehicleWheel
	{ "damping/relaxation", "damping_relaxation" }, // VehicleWheel
	{ "damp_override/angular", "angular_damp" }, // PhysicsBody, PhysicsBody2D
	{ "damp_override/linear", "linear_damp" }, // PhysicsBody, PhysicsBody2D
	{ "dialog/hide_on_ok", "dialog_hide_on_ok" }, // AcceptDialog
	{ "dialog/text", "dialog_text" }, // AcceptDialog
	{ "drag_margin/bottom", "drag_margin_bottom" }, // Camera2D
	{ "drag_margin/h_enabled", "drag_margin_h_enabled" }, // Camera2D
	{ "drag_margin/left", "drag_margin_left" }, // Camera2D
	{ "drag_margin/right", "drag_margin_right" }, // Camera2D
	{ "drag_margin/top", "drag_margin_top" }, // Camera2D
	{ "drag_margin/v_enabled", "drag_margin_v_enabled" }, // Camera2D
	{ "enabler/fixed_process_parent", "physics_process_parent" }, // VisibilityEnabler2D
	{ "enabler/freeze_bodies", "freeze_bodies" }, // VisibilityEnabler, VisibilityEnabler2D
	{ "enabler/pause_animated_sprites", "pause_animated_sprites" }, // VisibilityEnabler2D
	{ "enabler/pause_animations", "pause_animations" }, // VisibilityEnabler, VisibilityEnabler2D
	{ "enabler/pause_particles", "pause_particles" }, // VisibilityEnabler2D
	{ "enabler/process_parent", "process_parent" }, // VisibilityEnabler2D
	{ "expand_margin/bottom", "expand_margin_bottom" }, // StyleBox
	{ "expand_margin/left", "expand_margin_left" }, // StyleBox
	{ "expand_margin/right", "expand_margin_right" }, // StyleBox
	{ "expand_margin/top", "expand_margin_top" }, // StyleBox
	{ "extra_spacing/bottom", "extra_spacing_bottom" }, // DynamicFont
	{ "extra_spacing/char", "extra_spacing_char" }, // DynamicFont
	{ "extra_spacing/space", "extra_spacing_space" }, // DynamicFont
	{ "extra_spacing/top", "extra_spacing_top" }, // DynamicFont
	{ "flags/alpha_cut", "alpha_cut" }, // Sprite3D
	{ "flags/double_sided", "double_sided" }, // Sprite3D
	{ "flags/shaded", "shaded" }, // Sprite3D
	{ "flags/transparent", "transparent" }, // Sprite3D
	{ "focus_neighbour/bottom", "focus_neighbour_bottom" }, // Control
	{ "focus_neighbour/left", "focus_neighbour_left" }, // Control
	{ "focus_neighbour/right", "focus_neighbour_right" }, // Control
	{ "focus_neighbour/top", "focus_neighbour_top" }, // Control
	{ "font/font", "font_data" }, // DynamicFont
	{ "font/size", "size" }, // DynamicFont
	{ "font/use_filter", "use_filter" }, // DynamicFont
	{ "font/use_mipmaps", "use_mipmaps" }, // DynamicFont
	{ "geometry/cast_shadow", "cast_shadow" }, // GeometryInstance
	{ "geometry/extra_cull_margin", "extra_cull_margin" }, // GeometryInstance
	{ "geometry/material_override", "material_override" }, // GeometryInstance
	{ "geometry/use_baked_light", "use_in_baked_light" }, // GeometryInstance
	{ "hint/tooltip", "hint_tooltip" }, // Control
	{ "input/capture_on_drag", "input_capture_on_drag" }, // CollisionObject
	{ "input/pickable", "input_pickable" }, // CollisionObject2D
	{ "input/ray_pickable", "input_ray_pickable" }, // CollisionObject
	{ "invert/border", "invert_border" }, // Polygon2D
	{ "invert/enable", "invert_enable" }, // Polygon2D
	{ "is_pressed", "pressed" }, // BaseButton
	{ "limit/bottom", "limit_bottom" }, // Camera2D
	{ "limit/left", "limit_left" }, // Camera2D
	{ "limit/right", "limit_right" }, // Camera2D
	{ "limit/top", "limit_top" }, // Camera2D
	{ "margin/bottom", "margin_bottom" }, // Control, StyleBox
	{ "margin/left", "margin_left" }, // Control, StyleBox
	{ "margin/right", "margin_right" }, // Control, StyleBox
	{ "margin/top", "margin_top" }, // Control, StyleBox
	{ "material/material", "material" }, // CanvasItem
	{ "material/use_parent", "use_parent_material" }, // CanvasItem
	{ "mesh/mesh", "mesh" }, // MeshInstance
	{ "mesh/skeleton", "skeleton" }, // MeshInstance
	//{ "mode", "fill_mode" }, // TextureProgress & others // Would break TileMap and others, handle manually
	{ "motion/brake", "brake" }, // VehicleBody
	{ "motion/engine_force", "engine_force" }, // VehicleBody
	{ "motion/mirroring", "motion_mirroring" }, // ParallaxLayer
	{ "motion/offset", "motion_offset" }, // ParallaxLayer
	{ "motion/scale", "motion_scale" }, // ParallaxLayer
	{ "motion/steering", "steering" }, // VehicleBody
	{ "occluder/light_mask", "occluder_light_mask" }, // TileMap
	{ "params/attenuation/distance_exp", "attenuation_distance_exp" },
	{ "params/attenuation/max_distance", "attenuation_max_distance" },
	{ "params/attenuation/min_distance", "attenuation_min_distance" },
	{ "params/emission_cone/attenuation_db", "emission_cone_attenuation_db" },
	{ "params/emission_cone/degrees", "emission_cone_degrees" },
	{ "params/modulate", "self_modulate" },
	{ "params/pitch_scale", "pitch_scale" },
	{ "params/scale", "texture_scale" },
	{ "params/volume_db", "volume_db" },
	{ "patch_margin/bottom", "patch_margin_bottom" }, // Patch9Frame
	{ "patch_margin/left", "patch_margin_left" }, // Patch9Frame
	{ "patch_margin/right", "patch_margin_right" }, // Patch9Frame
	{ "patch_margin/top", "patch_margin_top" }, // Patch9Frame
	{ "percent/visible", "percent_visible" }, // ProgressBar
	{ "placeholder/alpha", "placeholder_alpha" }, // LineEdit
	{ "placeholder/text", "placeholder_text" }, // LineEdit
	//{ "playback/active", "playback_active" }, // AnimationPlayer, AnimationTreePlayer // properly renamed for AnimationPlayer, but not AnimationTreePlayer, handle manually
	{ "playback/default_blend_time", "playback_default_blend_time" }, // AnimationPlayer
	{ "playback/process_mode", "playback_process_mode" }, // AnimationPlayer, AnimationTreePlayer, Tween
	{ "playback/speed", "playback_speed" }, // AnimationPlayer, Tween
	{ "playback/repeat", "playback_speed" }, // AnimationPlayer
	{ "popup/exclusive", "popup_exclusive" }, // Popup
	{ "process/pause_mode", "pause_mode" }, // Node
	{ "radial_fill/center_offset", "radial_center_offset" }, // TextureProgress
	{ "radial_fill/fill_degrees", "radial_fill_degrees" }, // TextureProgress
	{ "radial_fill/initial_angle", "radial_initial_angle" }, // TextureProgress
	{ "range/exp_edit", "exp_edit" }, // Range
	{ "range/height", "range_height" }, // Light2D
	{ "range/item_mask", "range_item_cull_mask" }, // Light2D
	{ "range/layer_max", "range_layer_max" }, // Light2D
	{ "range/layer_min", "range_layer_min" }, // Light2D
	{ "range/max", "max_value" }, // Range
	{ "range/min", "min_value" }, // Range
	{ "range/page", "page" }, // Range
	{ "range/rounded", "rounded" }, // Range
	{ "range/step", "step" }, // Range
	{ "range/value", "value" }, // Range
	{ "range/z_max", "range_z_max" }, // Light2D
	{ "range/z_min", "range_z_min" }, // Light2D
	{ "rect/min_size", "rect_min_size" }, // Control
	{ "rect/pos", "rect_position" }, // Control
	{ "rect/rotation", "rect_rotation" }, // Control
	{ "rect/scale", "rect_scale" }, // Control
	{ "rect/size", "rect_size" }, // Control
	//{ "region", "region_enabled" }, // Sprite, Sprite3D // Not renamed for Texture, handle manually
	{ "resource/name", "resource_name" }, // Resource
	{ "resource/path", "resource_path" }, // Resource
	{ "root/root", "root_node" }, // AnimationPlayer
	{ "script/script", "script" }, // Object
	{ "scroll/base_offset", "scroll_base_offset" }, // ParallaxBackground
	{ "scroll/base_scale", "scroll_base_scale" }, // ParallaxBackground
	{ "scroll/horizontal", "scroll_horizontal_enabled" }, // ScrollContainer
	{ "scroll/ignore_camera_zoom", "scroll_ignore_camera_zoom" }, // ParallaxBackground
	{ "scroll/limit_begin", "scroll_limit_begin" }, // ParallaxBackground
	{ "scroll/limit_end", "scroll_limit_end" }, // ParallaxBackground
	{ "scroll/offset", "scroll_offset" }, // ParallaxBackground
	{ "scroll/vertical", "scroll_vertical_enabled" }, // ScrollContainer
	{ "shadow/buffer_size", "shadow_buffer_size" }, // Light2D
	{ "shadow/color", "shadow_color" }, // Light2D
	{ "shadow/enabled", "shadow_enabled" }, // Light2D
	{ "shadow/item_mask", "shadow_item_cull_mask" }, // Light2D
	{ "size_flags/horizontal", "size_flags_horizontal" }, // Control // Enum order got inverted Expand,Fill -> Fill,Expand, handle manually after rename
	{ "size_flags/stretch_ratio", "size_flags_stretch_ratio" }, // Control
	{ "size_flags/vertical", "size_flags_vertical" }, // Control // Enum order got inverted Expand,Fill -> Fill,Expand, handle manually after rename
	{ "smoothing/enable", "smoothing_enabled" }, // Camera2D
	{ "smoothing/speed", "smoothing_speed" }, // Camera2D
	{ "sort/enabled", "sort_enabled" }, // YSort
	{ "split/collapsed", "collapsed" }, // SplitContainer
	{ "split/dragger_visibility", "dragger_visibility" }, // SplitContainer
	{ "split/offset", "split_offset" }, // SplitContainer
	{ "stream/audio_track", "audio_track" }, // VideoPlayer
	{ "stream/autoplay", "autoplay" }, // VideoPlayer
	{ "stream/buffering_ms", "buffering_msec" }, // VideoPlayer
	{ "stream/loop", "loop" }, // Audio*
	{ "stream/loop_restart_time", "loop_offset" }, // Audio*
	{ "stream/paused", "paused" }, // VideoPlayer
	{ "stream/pitch_scale", "pitch_scale" }, // Audio*
	{ "stream/play", "playing" }, // Audio*
	{ "stream/stream", "stream" }, // VideoPlayer
	{ "stream/volume_db", "volume_db" }, // VideoPlayer
	{ "suspension/max_force", "suspension_max_force" }, // VehicleWheel
	{ "suspension/stiffness", "suspension_stiffness" }, // VehicleWheel
	{ "suspension/travel", "suspension_travel" }, // VehicleWheel
	{ "texture/offset", "texture_offset" }, // Polygon2D
	{ "texture/over", "texture_over" }, // TextureProgress
	{ "texture/progress", "texture_progress" }, // TextureProgress
	{ "texture/rotation", "texture_rotation_degrees" }, // Polygon2D
	{ "texture/scale", "texture_scale" }, // Polygon2D
	{ "textures/click_mask", "texture_click_mask" }, // TextureButton
	{ "textures/disabled", "texture_disabled" }, // TextureButton
	{ "textures/focused", "texture_focused" }, // TextureButton
	{ "textures/hover", "texture_hover" }, // TextureButton
	{ "textures/normal", "texture_normal" }, // TextureButton
	{ "textures/pressed", "texture_pressed" }, // TextureButton
	{ "texture/texture", "texture" }, // Polygon2D
	{ "texture/under", "texture_under" }, // TextureProgress
	{ "theme/theme", "theme" }, // Control
	{ "transform/local", "transform" }, // Spatial
	{ "transform/pos", "position" }, // Node2D
	{ "transform/rotation", "rotation_degrees" }, // Spatial
	{ "transform/rotation_rad", "rotation" }, // Spatial
	{ "transform/rot", "rotation_degrees" }, // Node2D
	{ "transform/scale", "scale" }, // Node2D, Spatial
	{ "transform/translation", "translation" }, // Spatial
	{ "type/steering", "use_as_steering" }, // VehicleWheel
	{ "type/traction", "use_as_traction" }, // VehicleWheel
	{ "vars/lifetime", "lifetime" }, // Particles
	{ "velocity/angular", "angular_velocity" }, // PhysicsBody, PhysicsBody2D
	{ "velocity/linear", "linear_velocity" }, // PhysicsBody, PhysicsBody2D
	{ "visibility", "visibility_aabb" }, // Particles
	{ "visibility/behind_parent", "show_behind_parent" }, // CanvasItem
	{ "visibility/light_mask", "light_mask" }, // CanvasItem
	{ "visibility/on_top", "show_on_top" }, // CanvasItem
	//{ "visibility/opacity", "modulate" }, // CanvasItem // Can't be converted this way, handle manually
	//{ "visibility/self_opacity", "self_modulate" }, // CanvasItem // Can't be converted this way, handle manually
	{ "visibility/visible", "visible" }, // CanvasItem, Spatial
	{ "wheel/friction_slip", "wheel_friction_slip" }, // VehicleWheel
	{ "wheel/radius", "wheel_radius" }, // VehicleWheel
	{ "wheel/rest_length", "wheel_rest_length" }, // VehicleWheel
	{ "wheel/roll_influence", "wheel_roll_influence" }, // VehicleWheel
	{ "window/title", "window_title" }, // Dialogs
	{ "z/relative", "z_as_relative" }, // Node2D
	{ "z/z", "z_index" }, // Node2D
	{ NULL, NULL }
};

static const char *type_renames[][2] = {
	{ "CanvasItemMaterial", "ShaderMaterial" },
	{ "CanvasItemShader", "Shader" },
	{ "ColorFrame", "ColorRect" },
	{ "ColorRamp", "Gradient" },
	{ "FixedMaterial", "SpatialMaterial" },
	{ "Patch9Frame", "NinePatchRect" },
	{ "ReferenceFrame", "ReferenceRect" },
	{ "SampleLibrary", "Resource" },
	{ "SamplePlayer2D", "AudioStreamPlayer2D" },
	{ "SamplePlayer", "Node" },
	{ "SoundPlayer2D", "Node2D" },
	{ "SpatialSamplePlayer", "AudioStreamPlayer3D" },
	{ "SpatialStreamPlayer", "AudioStreamPlayer3D" },
	{ "StreamPlayer", "AudioStreamPlayer" },
	{ "TestCube", "MeshInstance" },
	{ "TextureFrame", "TextureRect" },
	// Only for scripts
	{ "Matrix32", "Transform2D" },
	{ "Matrix3", "Basis" },
	{ "RawArray", "PoolByteArray" },
	{ "IntArray", "PoolIntArray" },
	{ "RealArray", "PoolRealArray" },
	{ "StringArray", "PoolStringArray" },
	{ "Vector2Array", "PoolVector2Array" },
	{ "Vector3Array", "PoolVector3Array" },
	{ "ColorArray", "PoolColorArray" },
	{ NULL, NULL }
};

static const char *signal_renames[][2] = {
	{ "area_enter", "area_entered" }, // Area, Area2D
	{ "area_enter_shape", "area_shape_entered" }, // Area, Area2D
	{ "area_exit", "area_exited" }, // Area, Area2D
	{ "area_exit_shape", "area_shape_exited" }, // Area, Area2D
	{ "body_enter", "body_entered" }, // Area, Area2D, PhysicsBody, PhysicsBody2D
	{ "body_enter_shape", "body_shape_entered" }, // Area, Area2D, PhysicsBody, PhysicsBody2D
	{ "body_exit", "body_exited" }, // Area, Area2D, PhysicsBody, PhysicsBody2D
	{ "body_exit_shape", "body_shape_exited" }, // Area, Area2D, PhysicsBody, PhysicsBody2D
	{ "enter_camera", "camera_entered" }, // VisibilityNotifier
	{ "enter_screen", "screen_entered" }, // VisibilityNotifier, VisibilityNotifier2D
	{ "enter_tree", "tree_entered" }, // Node
	{ "enter_viewport", "viewport_entered" }, // VisibilityNotifier2D
	{ "exit_camera", "camera_exited" }, // VisibilityNotifier
	{ "exit_screen", "screen_exited" }, // VisibilityNotifier, VisibilityNotifier2D
	{ "exit_tree", "tree_exited" }, // Node
	{ "exit_viewport", "viewport_exited" }, // VisibilityNotifier2D
	//{ "finished", "animation_finished" }, // AnimationPlayer, AnimatedSprite, but not StreamPlayer, handle manually
	{ "fixed_frame", "physics_frame" }, // SceneTree
	{ "focus_enter", "focus_entered" }, // Control
	{ "focus_exit", "focus_exited" }, // Control
	{ "input_event", "gui_input" }, // Control // FIXME: but not CollisionObject and CollisionObject2D, it should be handled manually
	{ "item_pressed", "id_pressed" }, // PopupMenu
	{ "modal_close", "modal_closed" }, // Control
	{ "mouse_enter", "mouse_entered" }, // CollisionObject, CollisionObject2D, Control
	{ "mouse_exit", "mouse_exited" }, // CollisionObject, CollisionObject2D, Control
	{ "tween_start", "tween_started" }, // Tween
	{ "tween_complete", "tween_completed" }, // Tween
	{ NULL, NULL }
};

void EditorExportGodot3::_find_files(EditorFileSystemDirectory *p_dir, List<String> *r_files) {

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_find_files(p_dir->get_subdir(i), r_files);
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {

		r_files->push_back(p_dir->get_file_path(i));
	}
}

void EditorExportGodot3::_rename_properties(const String &p_type, List<ExportData::PropertyData> *p_props) {

	// We need specific hacks to fix compatibility breakage in the tracks of Animations
	bool fix_animation_tracks = (p_type == "Animation");
	String found_track_number = "";

	// Anchors/margins changed in 3.0 from always-positive to relative to their ratio anchor,
	// so we need to flip the sign of margins based on their anchor mode.
	int flip_margin_left = false;
	int flip_margin_right = false;
	int flip_margin_top = false;
	int flip_margin_bottom = false;

	for (List<ExportData::PropertyData>::Element *E = p_props->front(); E; E = E->next()) {

		/* Fixes for 2D rotations */

		// 2D rotations are now clockwise to match the downward Y base
		// Do this before the renaming, as afterwards we can't distinguish
		// between 2D and 3D rotations_degrees
		if (E->get().name == "transform/rot") {
			E->get().value = (real_t)E->get().value * -1.0;
		}

		// To fix 2D rotations in the properties of Animation tracks (see below),
		// we need to locate stuff like this:
		// tracks/0/path = NodePath("Sprite:transform/rot")
		// And then modify the 'values' key of 'tracks/0/keys'.
		// This is going to be hacky.
		// We'll assume that we get properties in the correct order, so that the path will come before the keys
		// Otherwise we'd have to keep a stack of the track keys we found to later compare them to track paths
		// that match rotation_deg...
		if (fix_animation_tracks) {
			String prop_name = E->get().name;
			if (prop_name.begins_with("tracks/") && prop_name.ends_with("/path")) {
				String path_value = E->get().value;

				// Check if it's a rotation and save the track number to fix its assigned values
				if (path_value.find("transform/rot") != -1) {
					// We found a track 'path' with a "transform/rot" NodePath, its 'keys' need to be fixed
					found_track_number = prop_name.get_slice("/", 1);
					print_line("Found Animation track with 2D rotations: " + prop_name + " = " + path_value);
				}

				// In animation tracks, NodePaths can refer to properties that need to be renamed
				int sep = path_value.find(":");
				if (sep != -1) {
					String track_nodepath = path_value.substr(0, sep);
					String track_prop = path_value.substr(sep + 1, path_value.length());
					if (prop_rename_map.has(track_prop)) {
						track_prop = prop_rename_map[track_prop];
					}

					// "[self_]opacity" was removed, and is replaced by the alpha component of "[self_]modulate"
					// "modulate" may already exist, but we posit that the "opacity" value is more important
					// Thankfully in NodePaths we can access the alpha property directly
					if (track_prop == "visibility/opacity") {
						track_prop = "modulate:a";
					} else if (track_prop == "visibility/self_opacity") {
						track_prop = "self_modulate:a";
					}

					E->get().value = NodePath(track_nodepath + ":" + track_prop);
				}
			} else if (found_track_number != "" && prop_name == "tracks/" + found_track_number + "/keys") {
				// Bingo! We found keys matching the track number we had spotted
				print_line("Fixing sign of 2D rotations in animation track " + found_track_number);
				Dictionary track_keys = E->get().value;
				if (track_keys.has("values")) {
					Array values = track_keys["values"];
					for (int i = 0; i < values.size(); i++) {
						values[i] = (real_t)values[i] * -1.0;
					}
					track_keys["values"] = values;
					E->get().value = track_keys;
					found_track_number = "";
				} else {
					print_line("Tried to change rotation in Animation tracks, but no value set found.");
				}
			}
		}

		/* Do the actual renaming */

		if (prop_rename_map.has(E->get().name)) {
			E->get().name = prop_rename_map[E->get().name];
		}

		/* Hardcoded fixups for properties that changed definition in 3.0 */

		// Anchors changed from Begin,End,Ratio,Center to only a ratio
		if (E->get().name.begins_with("anchor_")) {
			String side = E->get().name.substr(7, E->get().name.length() - 1);
			int prop_value = (int)E->get().value;
			switch (prop_value) {
				case 0: { // Begin
					E->get().value = 0.0;
				} break;
				case 1: { // End
					E->get().value = 1.0;
					// Flip corresponding margin's sign
					if (side == "left")
						flip_margin_left = true;
					else if (side == "right")
						flip_margin_right = true;
					else if (side == "top")
						flip_margin_top = true;
					else if (side == "bottom")
						flip_margin_bottom = true;
				} break;
				case 2: { // Ratio
					E->get().value = 0.0;
					print_line("WARNING: Property '" + E->get().name + "' with value 'Ratio' cannot be converted to the format used in Godot 3. Convert it to 'Begin' or 'End' to avoid losing the corresponding margin value.");
				} break;
				case 3: { // Center
					E->get().value = 0.5;
					// Flip corresponding margin's sign
					if (side == "left")
						flip_margin_left = true;
					else if (side == "right")
						flip_margin_right = true;
					else if (side == "top")
						flip_margin_top = true;
					else if (side == "bottom")
						flip_margin_bottom = true;
				} break;
			}
		}

		// Size flags enum changed ordering from "Expand,Fill" to "Fill,Expand,..."
		// So we swap 1 (Expand) and 2 (Fill), keep 0 (none) and 3 (Expand + Fill)
		if (E->get().name == "size_flags_horizontal" || E->get().name == "size_flags_vertical") {
			int prop_value = (int)E->get().value;
			switch (prop_value) {
				case 1: // Expand -> Fill
					E->get().value = 2;
				case 2: // Fill -> Expand
					E->get().value = 1;
				default: // none or both, keep
					break;
			}
		}

		// "[self_]opacity" was removed, and is replaced by the alpha component of "[self_]modulate"
		// "modulate" may already exist, but we posit that the "opacity" value is more important
		if (E->get().name == "visibility/opacity" || E->get().name == "visibility/self_opacity") {
			if (E->get().name == "visibility/self_opacity") {
				E->get().name = "self_modulate";
			} else {
				E->get().name = "modulate";
			}
			E->get().value = Color(1.0, 1.0, 1.0, (float)E->get().value);
		}

		// AnimationPlayer's "playback/active" was renamed to "playback_active", but not AnimationTreePlayer's
		if (p_type == "AnimationPlayer" && E->get().name == "playback/active") {
			E->get().name = "playback_active";
		}

		// Joint2D's "collision/exclude_nodes" was renamed to "disable_collision", but not Joint's
		if (p_type == "Joint2D" && E->get().name == "collision/exclude_nodes") {
			E->get().name = "disable_collision";
		}

		// TextureProgress' "mode" was renamed to "fill_mode", but not that of other nodes like TileMap
		if (p_type == "TextureProgress" && E->get().name == "mode") {
			E->get().name = "fill_mode";
		}

		// Sprite and Sprite3D's "region" was renamed to "region_enabled", but not Texture's
		if ((p_type == "Sprite" || p_type == "Sprite3D") && E->get().name == "region") {
			E->get().name = "region_enabled";
		}

		// "click_on_pressed" was renamed to "action_mode" and is now a enum
		if (E->get().name == "click_on_press") {
			E->get().name = "action_mode";
			if (E->get().value) {
				E->get().value = 0; // ACTION_MODE_BUTTON_PRESS
			} else {
				E->get().value = 1; // ACTION_MODE_BUTTON_RELEASE
			}
		}
	}

	// Flip margins based on the previously fixed anchor modes
	if (flip_margin_left || flip_margin_right || flip_margin_top || flip_margin_bottom) {
		// Loop again and fix the margins
		for (List<ExportData::PropertyData>::Element *E = p_props->front(); E; E = E->next()) {
			if (!E->get().name.begins_with("margin_")) {
				continue;
			}
			if ((flip_margin_left && E->get().name == "margin_left") ||
					(flip_margin_right && E->get().name == "margin_right") ||
					(flip_margin_top && E->get().name == "margin_top") ||
					(flip_margin_bottom && E->get().name == "margin_bottom")) {
				E->get().value = (real_t)E->get().value * -1.0;
			}
		}
	}
}

void EditorExportGodot3::_add_new_properties(const String &p_type, List<ExportData::PropertyData> *p_props) {
	bool add_mouse_filter = false;

	bool ignore_mouse = false;
	bool stop_mouse = false;

	for (List<ExportData::PropertyData>::Element *E = p_props->front(); E; E = E->next()) {
		String prop_name = E->get().name;
		if (prop_name == "focus/ignore_mouse" || prop_name == "focus/stop_mouse") {
			add_mouse_filter = true;

			if (prop_name == "focus/ignore_mouse") {
				ignore_mouse = E->get().value;
			} else if (prop_name == "focus/stop_mouse") {
				stop_mouse = E->get().value;
			}
		}
	}

	if (add_mouse_filter) {
		ExportData::PropertyData pdata;
		pdata.name = "mouse_filter";

		if (ignore_mouse && stop_mouse) {
			pdata.value = 1; // MOUSE_FILTER_PASS
		} else if (ignore_mouse && !stop_mouse) {
			pdata.value = 2; // MOUSE_FILTER_IGNORE
		} else {
			pdata.value = 0; // MOUSE_FILTER_STOP
		}

		p_props->push_back(pdata);
	}
}

void EditorExportGodot3::_convert_resources(ExportData &resource) {

	for (int i = 0; i < resource.resources.size(); i++) {

		_add_new_properties(resource.resources[i].type, &resource.resources[i].properties);
		_rename_properties(resource.resources[i].type, &resource.resources[i].properties);

		if (type_rename_map.has(resource.resources[i].type)) {
			resource.resources[i].type = type_rename_map[resource.resources[i].type];
		}
	}

	for (int i = 0; i < resource.nodes.size(); i++) {

		_add_new_properties(resource.nodes[i].type, &resource.nodes[i].properties);
		_rename_properties(resource.nodes[i].type, &resource.nodes[i].properties);

		if (type_rename_map.has(resource.nodes[i].type)) {
			resource.nodes[i].type = type_rename_map[resource.nodes[i].type];
		}
	}

	for (int i = 0; i < resource.connections.size(); i++) {

		if (signal_rename_map.has(resource.connections[i].signal)) {
			resource.connections[i].signal = signal_rename_map[resource.connections[i].signal];
		}

		/* Manual handling for signals which need to be conditionally renamed based on their Node's type */

		// AnimationPlayer and AnimatedSprite's "finished" signal was renamed to "animation_finished",
		// but not that of StreamPlayer. Since node information is missing from the connection data
		// (we only have the NodePath), we'll have to compare against the nodes array to find out.
		if (resource.connections[i].signal == "finished") {
			String from = resource.connections[i].from;
			// NodePath "from" is relative to root node, can be direct child (no '/') or further down
			int slice_count = from.get_slice_count("/");
			String parent = ".";
			String nodename = from;
			if (slice_count > 1) {
				parent = from.get_slice("/", slice_count - 2);
				nodename = from.get_slice("/", slice_count - 1);
			}

			for (int j = 0; j < resource.nodes.size(); j++) {
				if (resource.nodes[j].name == nodename && resource.nodes[j].parent == parent) {
					if (resource.nodes[j].type == "AnimationPlayer" || resource.nodes[j].type == "AnimatedSprite") {
						resource.connections[i].signal = "animation_finished";
						break;
					}
				}
			}
		}
	}
}

void EditorExportGodot3::_unpack_packed_scene(ExportData &resource) {

	Dictionary d;
	for (List<ExportData::PropertyData>::Element *E = resource.resources[resource.resources.size() - 1].properties.front(); E; E = E->next()) {
		if (E->get().name == "_bundled") {
			d = E->get().value;
		}
	}

	ERR_FAIL_COND(d.empty());

	ERR_FAIL_COND(!d.has("names"));
	ERR_FAIL_COND(!d.has("variants"));
	ERR_FAIL_COND(!d.has("node_count"));
	ERR_FAIL_COND(!d.has("nodes"));
	ERR_FAIL_COND(!d.has("conn_count"));
	ERR_FAIL_COND(!d.has("conns"));

	Vector<String> names;

	DVector<String> snames = d["names"];
	if (snames.size()) {

		int namecount = snames.size();
		names.resize(namecount);
		DVector<String>::Read r = snames.read();
		for (int i = 0; i < names.size(); i++)
			names[i] = r[i];
	}

	Array variants = d["variants"];

	resource.nodes.resize(d["node_count"]);

	int nc = resource.nodes.size();
	if (nc) {
		DVector<int> snodes = d["nodes"];
		DVector<int>::Read r = snodes.read();
		int idx = 0;
		for (int i = 0; i < nc; i++) {

			int parent = r[idx++];
			int owner = r[idx++];
			int type = r[idx++];
			int name = r[idx++];
			int instance = r[idx++];

			ExportData::NodeData &node_data = resource.nodes[i];

			node_data.text_data = false;
			node_data.name = names[name];
			if (type == 0x7FFFFFFF) {
				node_data.instanced = true;
			} else {
				node_data.instanced = false;
				node_data.type = names[type];
			}

			node_data.parent_int = parent;
			node_data.owner_int = owner;
			if (instance >= 0) {
				node_data.instance_is_placeholder = instance & SceneState::FLAG_INSTANCE_IS_PLACEHOLDER;
				node_data.instance = variants[instance & SceneState::FLAG_MASK];
			}

			int prop_count = r[idx++];

			for (int j = 0; j < prop_count; j++) {

				int prop_name = r[idx++];
				int prop_value = r[idx++];

				ExportData::PropertyData pdata;
				pdata.name = names[prop_name];
				pdata.value = variants[prop_value];
				node_data.properties.push_back(pdata);
			}

			int group_count = r[idx++];
			for (int j = 0; j < group_count; j++) {

				int group_name = r[idx++];
				node_data.groups.push_back(names[group_name]);
			}
		}
	}

	int cc = d["conn_count"];

	if (cc) {

		DVector<int> sconns = d["conns"];
		DVector<int>::Read r = sconns.read();
		int idx = 0;
		for (int i = 0; i < cc; i++) {

			ExportData::Connection conn;

			conn.from_int = r[idx++];
			conn.to_int = r[idx++];
			conn.signal = names[r[idx++]];
			conn.method = names[r[idx++]];
			conn.flags = r[idx++];
			int bindcount = r[idx++];

			for (int j = 0; j < bindcount; j++) {

				conn.binds.push_back(variants[r[idx++]]);
			}

			resource.connections.push_back(conn);
		}
	}

	Array np;
	if (d.has("node_paths")) {
		np = d["node_paths"];
	}

	for (int i = 0; i < np.size(); i++) {
		resource.node_paths.push_back(np[i]);
	}

	Array ei;
	if (d.has("editable_instances")) {
		ei = d["editable_instances"];
		for (int i = 0; i < ei.size(); i++) {
			resource.editables.push_back(ei[i]);
		}
	}

	if (d.has("base_scene")) {
		resource.base_scene = variants[d["base_scene"]];
	}

	resource.resources.resize(resource.resources.size() - 1); //erase packed
}

void EditorExportGodot3::_pack_packed_scene(ExportData &resource) {

	pack_names.clear();
	pack_values.clear();

	Dictionary d;

	d["node_count"] = resource.nodes.size();

	Vector<int> node_data;

	for (int i = 0; i < resource.nodes.size(); i++) {

		const ExportData::NodeData &node = resource.nodes[i];

		node_data.push_back(node.parent_int);
		node_data.push_back(node.owner_int);
		if (node.instanced) {
			node_data.push_back(0x7FFFFFFF);
		} else {
			int name = _pack_name(node.type);
			node_data.push_back(name);
		}

		node_data.push_back(_pack_name(node.name));
		int instance = -1;
		if (node.instance != String()) {
			instance = _pack_value(node.instance);
			if (node.instance_is_placeholder) {
				instance |= SceneState::FLAG_INSTANCE_IS_PLACEHOLDER;
			}
		}
		node_data.push_back(instance);

		node_data.push_back(node.properties.size());

		for (int j = 0; j < node.properties.size(); j++) {
			node_data.push_back(_pack_name(node.properties[j].name));
			node_data.push_back(_pack_value(node.properties[j].value));
		}

		node_data.push_back(node.groups.size());

		for (int j = 0; j < node.groups.size(); j++) {

			node_data.push_back(_pack_name(node.groups[j]));
		}
	}

	d["nodes"] = node_data;

	d["conn_count"] = resource.connections.size();

	Vector<int> connections;

	for (int i = 0; i < resource.connections.size(); i++) {
		const ExportData::Connection &conn = resource.connections[i];

		connections.push_back(conn.from_int);
		connections.push_back(conn.to_int);
		connections.push_back(_pack_name(conn.signal));
		connections.push_back(_pack_name(conn.method));
		connections.push_back(conn.flags);
		connections.push_back(conn.binds.size());
		for (int j = 0; j < conn.binds.size(); j++) {
			connections.push_back(_pack_value(conn.binds[j]));
		}
	}

	d["conns"] = connections;

	Array np;
	for (int i = 0; i < resource.node_paths.size(); i++) {
		np.push_back(resource.node_paths[i]);
	}

	d["node_paths"] = np;

	Array ei;
	for (int i = 0; i < resource.editables.size(); i++) {
		ei.push_back(resource.editables[i]);
	}

	d["editable_instances"] = ei;

	if (resource.base_scene.get_type()) {

		d["base_scene"] = _pack_value(resource.base_scene);
	}

	DVector<String> names;
	names.resize(pack_names.size());
	{
		DVector<String>::Write w = names.write();
		for (Map<String, int>::Element *E = pack_names.front(); E; E = E->next()) {
			w[E->get()] = E->key();
		}
	}

	d["names"] = names;

	Array values;
	values.resize(pack_values.size());

	const Variant *K = NULL;
	while ((K = pack_values.next(K))) {

		int index = pack_values[*K];
		values[index] = *K;
	}

	d["variants"] = values;

	ExportData::ResourceData packed_scene;
	packed_scene.type = "PackedScene";
	packed_scene.index = -1;
	ExportData::PropertyData pd;
	pd.name = "_bundled";
	pd.value = d;
	packed_scene.properties.push_back(pd);

	resource.resources.push_back(packed_scene);
	resource.nodes.clear();
	resource.connections.clear();
	resource.editables.clear();
	resource.node_paths.clear();
	;
	resource.base_scene = Variant();
}

static String rtosfix(double p_value) {

	if (p_value == 0.0)
		return "0"; //avoid negative zero (-0) being written, which may annoy git, svn, etc. for changes when they don't exist.
	else
		return rtoss(p_value);
}

Error EditorExportGodot3::_get_property_as_text(const Variant &p_variant, String &p_string) {

	switch (p_variant.get_type()) {

		case Variant::NIL: {
			p_string += ("null");
		} break;
		case Variant::BOOL: {

			p_string += (p_variant.operator bool() ? "true" : "false");
		} break;
		case Variant::INT: {

			p_string += (itos(p_variant.operator int()));
		} break;
		case Variant::REAL: {

			String s = rtosfix(p_variant.operator real_t());
			if (s.find(".") == -1 && s.find("e") == -1)
				s += ".0";
			p_string += (s);
		} break;
		case Variant::STRING: {

			String str = p_variant;
			if (str.begins_with("@RESLOCAL:")) {
				p_string += "SubResource( " + str.get_slice(":", 1) + " )";
			} else if (str.begins_with("@RESEXTERNAL:")) {
				p_string += "ExtResource( " + str.get_slice(":", 1) + " )";
			} else {

				// Call _replace_resource in case it's a path to a scene/resource
				str = "\"" + _replace_resource(str).c_escape_multiline() + "\"";
				p_string += (str);
			}

		} break;
		case Variant::VECTOR2: {

			Vector2 v = p_variant;
			p_string += ("Vector2( " + rtosfix(v.x) + ", " + rtosfix(v.y) + " )");
		} break;
		case Variant::RECT2: {

			Rect2 aabb = p_variant;
			p_string += ("Rect2( " + rtosfix(aabb.pos.x) + ", " + rtosfix(aabb.pos.y) + ", " + rtosfix(aabb.size.x) + ", " + rtosfix(aabb.size.y) + " )");

		} break;
		case Variant::VECTOR3: {

			Vector3 v = p_variant;
			p_string += ("Vector3( " + rtosfix(v.x) + ", " + rtosfix(v.y) + ", " + rtosfix(v.z) + " )");
		} break;
		case Variant::PLANE: {

			Plane p = p_variant;
			p_string += ("Plane( " + rtosfix(p.normal.x) + ", " + rtosfix(p.normal.y) + ", " + rtosfix(p.normal.z) + ", " + rtosfix(p.d) + " )");

		} break;
		case Variant::_AABB: {

			Rect3 aabb = p_variant;
			p_string += ("Rect3( " + rtosfix(aabb.pos.x) + ", " + rtosfix(aabb.pos.y) + ", " + rtosfix(aabb.pos.z) + ", " + rtosfix(aabb.size.x) + ", " + rtosfix(aabb.size.y) + ", " + rtosfix(aabb.size.z) + " )");

		} break;
		case Variant::QUAT: {

			Quat quat = p_variant;
			p_string += ("Quat( " + rtosfix(quat.x) + ", " + rtosfix(quat.y) + ", " + rtosfix(quat.z) + ", " + rtosfix(quat.w) + " )");

		} break;
		case Variant::MATRIX32: {

			String s = "Transform2D( ";
			Matrix32 m3 = p_variant;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 2; j++) {

					if (i != 0 || j != 0)
						s += ", ";
					s += rtosfix(m3.elements[i][j]);
				}
			}

			p_string += (s + " )");

		} break;
		case Variant::MATRIX3: {

			String s = "Basis( ";
			Matrix3 m3 = p_variant;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {

					if (i != 0 || j != 0)
						s += ", ";
					s += rtosfix(m3.elements[i][j]);
				}
			}

			p_string += (s + " )");

		} break;
		case Variant::TRANSFORM: {

			String s = "Transform( ";
			Transform t = p_variant;
			Matrix3 &m3 = t.basis;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {

					if (i != 0 || j != 0)
						s += ", ";
					s += rtosfix(m3.elements[i][j]);
				}
			}

			s = s + ", " + rtosfix(t.origin.x) + ", " + rtosfix(t.origin.y) + ", " + rtosfix(t.origin.z);

			p_string += (s + " )");
		} break;

		// misc types
		case Variant::COLOR: {

			Color c = p_variant;
			p_string += ("Color( " + rtosfix(c.r) + ", " + rtosfix(c.g) + ", " + rtosfix(c.b) + ", " + rtosfix(c.a) + " )");

		} break;
		case Variant::IMAGE: {

			Image img = p_variant;

			if (img.empty()) {
				p_string += ("Image()");
				break;
			}

			String imgstr = "Image()";
			p_string += imgstr; //do not convert this for now

			/*imgstr+=itos(img.get_width());
			imgstr+=", "+itos(img.get_height());
			imgstr+=", "+String(img.get_mipmaps()?"true":"false");
			imgstr+=", "+Image::get_format_name(img.get_format());

			String s;

			DVector<uint8_t> data = img.get_data();
			int len = data.size();
			DVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();
			for (int i=0;i<len;i++) {

				if (i>0)
					s+=", ";
				s+=itos(ptr[i]);
			}

			imgstr+=", ";
			p_string+=(imgstr);
			p_string+=(s);
			p_string+=(" )");*/
		} break;
		case Variant::NODE_PATH: {

			String str = p_variant;

			str = "NodePath(\"" + str.c_escape() + "\")";
			p_string += (str);

		} break;

		case Variant::OBJECT: {

			//should never arrive here!
			ERR_FAIL_V(ERR_BUG);
		} break;
		case Variant::INPUT_EVENT: {

			String str = "InputEvent(";

			InputEvent ev = p_variant;
			switch (ev.type) {
				case InputEvent::KEY: {

					str += "KEY," + itos(ev.key.scancode);
					String mod;
					if (ev.key.mod.alt)
						mod += "A";
					if (ev.key.mod.shift)
						mod += "S";
					if (ev.key.mod.control)
						mod += "C";
					if (ev.key.mod.meta)
						mod += "M";

					if (mod != String())
						str += "," + mod;
				} break;
				case InputEvent::MOUSE_BUTTON: {

					str += "MBUTTON," + itos(ev.mouse_button.button_index);
				} break;
				case InputEvent::JOYSTICK_BUTTON: {
					str += "JBUTTON," + itos(ev.joy_button.button_index);

				} break;
				case InputEvent::JOYSTICK_MOTION: {
					str += "JAXIS," + itos(ev.joy_motion.axis) + "," + itos(ev.joy_motion.axis_value);
				} break;
				case InputEvent::NONE: {
					str += "NONE";
				} break;
				default: {}
			}

			str += ")";

			p_string += (str); //will be added later

		} break;
		case Variant::DICTIONARY: {

			Dictionary dict = p_variant;

			List<Variant> keys;
			dict.get_key_list(&keys);
			keys.sort();

			p_string += ("{\n");
			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

				/*
				if (!_check_type(dict[E->get()]))
					continue;
				*/
				_get_property_as_text(E->get(), p_string);
				p_string += (": ");
				_get_property_as_text(dict[E->get()], p_string);
				if (E->next())
					p_string += (",\n");
			}

			p_string += ("\n}");

		} break;
		case Variant::ARRAY: {

			p_string += ("[ ");
			Array array = p_variant;
			int len = array.size();
			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");
				_get_property_as_text(array[i], p_string);
			}
			p_string += (" ]");

		} break;

		case Variant::RAW_ARRAY: {

			p_string += ("PoolByteArray( ");
			String s;
			DVector<uint8_t> data = p_variant;
			int len = data.size();
			DVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr = r.ptr();
			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");

				p_string += (itos(ptr[i]));
			}

			p_string += (" )");

		} break;
		case Variant::INT_ARRAY: {

			p_string += ("PoolIntArray( ");
			DVector<int> data = p_variant;
			int len = data.size();
			DVector<int>::Read r = data.read();
			const int *ptr = r.ptr();

			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");

				p_string += (itos(ptr[i]));
			}

			p_string += (" )");

		} break;
		case Variant::REAL_ARRAY: {

			p_string += ("PoolRealArray( ");
			DVector<real_t> data = p_variant;
			int len = data.size();
			DVector<real_t>::Read r = data.read();
			const real_t *ptr = r.ptr();

			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");
				p_string += (rtosfix(ptr[i]));
			}

			p_string += (" )");

		} break;
		case Variant::STRING_ARRAY: {

			p_string += ("PoolStringArray( ");
			DVector<String> data = p_variant;
			int len = data.size();
			DVector<String>::Read r = data.read();
			const String *ptr = r.ptr();
			String s;
			//write_string("\n");

			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");
				String str = ptr[i];
				p_string += ("\"" + str.c_escape() + "\"");
			}

			p_string += (" )");

		} break;
		case Variant::VECTOR2_ARRAY: {

			p_string += ("PoolVector2Array( ");
			DVector<Vector2> data = p_variant;
			int len = data.size();
			DVector<Vector2>::Read r = data.read();
			const Vector2 *ptr = r.ptr();

			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");
				p_string += (rtosfix(ptr[i].x) + ", " + rtosfix(ptr[i].y));
			}

			p_string += (" )");

		} break;
		case Variant::VECTOR3_ARRAY: {

			p_string += ("PoolVector3Array( ");
			DVector<Vector3> data = p_variant;
			int len = data.size();
			DVector<Vector3>::Read r = data.read();
			const Vector3 *ptr = r.ptr();

			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");
				p_string += (rtosfix(ptr[i].x) + ", " + rtosfix(ptr[i].y) + ", " + rtosfix(ptr[i].z));
			}

			p_string += (" )");

		} break;
		case Variant::COLOR_ARRAY: {

			p_string += ("PoolColorArray( ");

			DVector<Color> data = p_variant;
			int len = data.size();
			DVector<Color>::Read r = data.read();
			const Color *ptr = r.ptr();

			for (int i = 0; i < len; i++) {

				if (i > 0)
					p_string += (", ");

				p_string += (rtosfix(ptr[i].r) + ", " + rtosfix(ptr[i].g) + ", " + rtosfix(ptr[i].b) + ", " + rtosfix(ptr[i].a));
			}
			p_string += (" )");

		} break;
		default: {}
	}

	return OK;
}

static String _valprop(const String &p_name) {

	// Escape and quote strings with extended ASCII or further Unicode characters
	// as well as '"', '=' or ' ' (32)
	const CharType *cstr = p_name.c_str();
	for (int i = 0; cstr[i]; i++) {
		if (cstr[i] == '=' || cstr[i] == '"' || cstr[i] < 33 || cstr[i] > 126) {
			return "\"" + p_name.c_escape_multiline() + "\"";
		}
	}
	// Keep as is
	return p_name;
}

void EditorExportGodot3::_save_text(const String &p_path, ExportData &resource) {

	FileAccessRef f = FileAccess::open(p_path, FileAccess::WRITE);

	if (resource.nodes.size()) {
		f->store_line("[gd_scene load_steps=" + itos(resource.nodes.size() + resource.resources.size()) + " format=2]\n");
	} else {
		f->store_line("[gd_resource type=\"" + resource.resources[resource.resources.size() - 1].type + "\" load_steps=" + itos(resource.resources.size()) + " format=2]\n");
	}

	for (Map<int, ExportData::Dependency>::Element *E = resource.dependencies.front(); E; E = E->next()) {

		f->store_line("[ext_resource path=\"" + resource_replace_map[E->get().path] + "\" type=\"" + E->get().type + "\" id=" + itos(E->key()) + "]");
	}

	for (int i = 0; i < resource.resources.size(); i++) {

		if (resource.nodes.size() || i < resource.resources.size() - 1) {

			f->store_line("\n[sub_resource type=\"" + resource.resources[i].type + "\" id=" + itos(resource.resources[i].index) + "]\n");
		} else {
			f->store_line("\n[resource]\n");
		}

		for (List<ExportData::PropertyData>::Element *E = resource.resources[i].properties.front(); E; E = E->next()) {

			String prop;
			_get_property_as_text(E->get().value, prop);
			f->store_line(_valprop(E->get().name) + " = " + prop);
		}
	}

	for (int i = 0; i < resource.nodes.size(); i++) {

		String node_txt = "\n[node";

		if (resource.nodes[i].name != String()) {
			node_txt += " name=\"" + String(resource.nodes[i].name).c_escape() + "\"";
		}

		if (resource.nodes[i].owner != NodePath()) {
			node_txt += " owner=\"" + String(resource.nodes[i].owner).c_escape() + "\"";
		}

		if (resource.nodes[i].type != String()) {
			node_txt += " type=\"" + resource.nodes[i].type + "\"";
		}

		if (resource.nodes[i].parent != NodePath()) {
			node_txt += " parent=\"" + String(resource.nodes[i].parent).c_escape() + "\"";
		}

		if (resource.nodes[i].instance != String()) {
			String prop;
			_get_property_as_text(resource.nodes[i].instance, prop);
			node_txt += " instance=" + prop + "";
		}

		if (!resource.nodes[i].groups.empty()) {
			node_txt += " groups=[\n";
			for (int j = 0; j < resource.nodes[i].groups.size(); j++) {
				node_txt += "\"" + resource.nodes[i].groups[j] + "\",\n";
			}
			node_txt += "]";
		}

		node_txt += "]\n";
		f->store_line(node_txt);

		for (List<ExportData::PropertyData>::Element *E = resource.nodes[i].properties.front(); E; E = E->next()) {

			String prop;
			_get_property_as_text(E->get().value, prop);
			f->store_line(_valprop(E->get().name) + " = " + prop);
		}
	}

	for (int i = 0; i < resource.connections.size(); i++) {

		String binds_array;
		_get_property_as_text(resource.connections[i].binds, binds_array);

		f->store_line("\n[connection signal=\"" + resource.connections[i].signal + "\" from=\"" + String(resource.connections[i].from).c_escape() + "\" to=\"" + String(resource.connections[i].to).c_escape() + "\" method=\"" + resource.connections[i].method + "\" binds=" + binds_array + "]");
	}

	for (int i = 0; i < resource.editables.size(); i++) {

		f->store_line("[editable path=\"" + String(resource.editables[i]).c_escape() + "\"]");
	}
}

enum {

	//numbering must be different from variant, in case new variant types are added (variant must be always contiguous for jumptable optimization)
	VARIANT_NIL = 1,
	VARIANT_BOOL = 2,
	VARIANT_INT = 3,
	VARIANT_REAL = 4,
	VARIANT_STRING = 5,
	VARIANT_VECTOR2 = 10,
	VARIANT_RECT2 = 11,
	VARIANT_VECTOR3 = 12,
	VARIANT_PLANE = 13,
	VARIANT_QUAT = 14,
	VARIANT_AABB = 15,
	VARIANT_MATRIX3 = 16,
	VARIANT_TRANSFORM = 17,
	VARIANT_MATRIX32 = 18,
	VARIANT_COLOR = 20,
	VARIANT_IMAGE = 21,
	VARIANT_NODE_PATH = 22,
	VARIANT_RID = 23,
	VARIANT_OBJECT = 24,
	VARIANT_INPUT_EVENT = 25,
	VARIANT_DICTIONARY = 26,
	VARIANT_ARRAY = 30,
	VARIANT_RAW_ARRAY = 31,
	VARIANT_INT_ARRAY = 32,
	VARIANT_REAL_ARRAY = 33,
	VARIANT_STRING_ARRAY = 34,
	VARIANT_VECTOR3_ARRAY = 35,
	VARIANT_COLOR_ARRAY = 36,
	VARIANT_VECTOR2_ARRAY = 37,
	VARIANT_INT64 = 40,
	VARIANT_DOUBLE = 41,

	IMAGE_ENCODING_EMPTY = 0,
	IMAGE_ENCODING_RAW = 1,
	IMAGE_ENCODING_LOSSLESS = 2,
	IMAGE_ENCODING_LOSSY = 3,

	OBJECT_EMPTY = 0,
	OBJECT_EXTERNAL_RESOURCE = 1,
	OBJECT_INTERNAL_RESOURCE = 2,
	OBJECT_EXTERNAL_RESOURCE_INDEX = 3,
	//version 2: added 64 bits support for float and int
	FORMAT_VERSION = 2,
	FORMAT_VERSION_CAN_RENAME_DEPS = 1

};

enum {
	IMAGE_FORMAT_L8, //luminance
	IMAGE_FORMAT_LA8, //luminance-alpha
	IMAGE_FORMAT_R8,
	IMAGE_FORMAT_RG8,
	IMAGE_FORMAT_RGB8,
	IMAGE_FORMAT_RGBA8,
	IMAGE_FORMAT_RGB565, //16 bit
	IMAGE_FORMAT_RGBA4444,
	IMAGE_FORMAT_RGBA5551,
	IMAGE_FORMAT_RF, //float
	IMAGE_FORMAT_RGF,
	IMAGE_FORMAT_RGBF,
	IMAGE_FORMAT_RGBAF,
	IMAGE_FORMAT_RH, //half float
	IMAGE_FORMAT_RGH,
	IMAGE_FORMAT_RGBH,
	IMAGE_FORMAT_RGBAH,
	IMAGE_FORMAT_DXT1, //s3tc bc1
	IMAGE_FORMAT_DXT3, //bc2
	IMAGE_FORMAT_DXT5, //bc3
	IMAGE_FORMAT_ATI1, //bc4
	IMAGE_FORMAT_ATI2, //bc5
	IMAGE_FORMAT_BPTC_RGBA, //btpc bc6h
	IMAGE_FORMAT_BPTC_RGBF, //float /
	IMAGE_FORMAT_BPTC_RGBFU, //unsigned float
	IMAGE_FORMAT_PVRTC2, //pvrtc
	IMAGE_FORMAT_PVRTC2A,
	IMAGE_FORMAT_PVRTC4,
	IMAGE_FORMAT_PVRTC4A,
	IMAGE_FORMAT_ETC, //etc1
	IMAGE_FORMAT_ETC2_R11, //etc2
	IMAGE_FORMAT_ETC2_R11S, //signed, NOT srgb.
	IMAGE_FORMAT_ETC2_RG11,
	IMAGE_FORMAT_ETC2_RG11S,
	IMAGE_FORMAT_ETC2_RGB8,
	IMAGE_FORMAT_ETC2_RGBA8,
	IMAGE_FORMAT_ETC2_RGB8A1,

};

static void _pad_buffer(int p_bytes, FileAccess *f) {

	int extra = 4 - (p_bytes % 4);
	if (extra < 4) {
		for (int i = 0; i < extra; i++)
			f->store_8(0); //pad to 32
	}
}

static void save_unicode_string(const String &p_string, FileAccess *f, bool p_hi_bit = false) {

	CharString utf8 = p_string.utf8();
	f->store_32(uint32_t(utf8.length() + 1) | (p_hi_bit ? 0x80000000 : 0));
	f->store_buffer((const uint8_t *)utf8.get_data(), utf8.length() + 1);
}

void EditorExportGodot3::_save_binary_property(const Variant &p_property, FileAccess *f) {

	switch (p_property.get_type()) {

		case Variant::NIL: {

			f->store_32(VARIANT_NIL);
			// don't store anything
		} break;
		case Variant::BOOL: {

			f->store_32(VARIANT_BOOL);
			bool val = p_property;
			f->store_32(val);
		} break;
		case Variant::INT: {

			f->store_32(VARIANT_INT);
			int val = p_property;
			f->store_32(int32_t(val));

		} break;
		case Variant::REAL: {

			f->store_32(VARIANT_REAL);
			f->store_real(p_property);

		} break;
		case Variant::STRING: {

			String str = p_property;
			if (str.begins_with("@RESLOCAL:")) {
				f->store_32(VARIANT_OBJECT);
				f->store_32(OBJECT_INTERNAL_RESOURCE);
				f->store_32(str.get_slice(":", 1).to_int());
			} else if (str.begins_with("@RESEXTERNAL:")) {
				f->store_32(VARIANT_OBJECT);
				f->store_32(OBJECT_EXTERNAL_RESOURCE_INDEX);
				f->store_32(str.get_slice(":", 1).to_int());
			} else {

				f->store_32(VARIANT_STRING);
				save_unicode_string(str, f);
			}
		} break;
		case Variant::VECTOR2: {

			f->store_32(VARIANT_VECTOR2);
			Vector2 val = p_property;
			f->store_real(val.x);
			f->store_real(val.y);

		} break;
		case Variant::RECT2: {

			f->store_32(VARIANT_RECT2);
			Rect2 val = p_property;
			f->store_real(val.pos.x);
			f->store_real(val.pos.y);
			f->store_real(val.size.x);
			f->store_real(val.size.y);

		} break;
		case Variant::VECTOR3: {

			f->store_32(VARIANT_VECTOR3);
			Vector3 val = p_property;
			f->store_real(val.x);
			f->store_real(val.y);
			f->store_real(val.z);

		} break;
		case Variant::PLANE: {

			f->store_32(VARIANT_PLANE);
			Plane val = p_property;
			f->store_real(val.normal.x);
			f->store_real(val.normal.y);
			f->store_real(val.normal.z);
			f->store_real(val.d);

		} break;
		case Variant::QUAT: {

			f->store_32(VARIANT_QUAT);
			Quat val = p_property;
			f->store_real(val.x);
			f->store_real(val.y);
			f->store_real(val.z);
			f->store_real(val.w);

		} break;
		case Variant::_AABB: {

			f->store_32(VARIANT_AABB);
			Rect3 val = p_property;
			f->store_real(val.pos.x);
			f->store_real(val.pos.y);
			f->store_real(val.pos.z);
			f->store_real(val.size.x);
			f->store_real(val.size.y);
			f->store_real(val.size.z);

		} break;
		case Variant::MATRIX32: {

			f->store_32(VARIANT_MATRIX32);
			Matrix32 val = p_property;
			f->store_real(val.elements[0].x);
			f->store_real(val.elements[0].y);
			f->store_real(val.elements[1].x);
			f->store_real(val.elements[1].y);
			f->store_real(val.elements[2].x);
			f->store_real(val.elements[2].y);

		} break;
		case Variant::MATRIX3: {

			f->store_32(VARIANT_MATRIX3);
			Matrix3 val = p_property;
			f->store_real(val.elements[0].x);
			f->store_real(val.elements[0].y);
			f->store_real(val.elements[0].z);
			f->store_real(val.elements[1].x);
			f->store_real(val.elements[1].y);
			f->store_real(val.elements[1].z);
			f->store_real(val.elements[2].x);
			f->store_real(val.elements[2].y);
			f->store_real(val.elements[2].z);

		} break;
		case Variant::TRANSFORM: {

			f->store_32(VARIANT_TRANSFORM);
			Transform val = p_property;
			f->store_real(val.basis.elements[0].x);
			f->store_real(val.basis.elements[0].y);
			f->store_real(val.basis.elements[0].z);
			f->store_real(val.basis.elements[1].x);
			f->store_real(val.basis.elements[1].y);
			f->store_real(val.basis.elements[1].z);
			f->store_real(val.basis.elements[2].x);
			f->store_real(val.basis.elements[2].y);
			f->store_real(val.basis.elements[2].z);
			f->store_real(val.origin.x);
			f->store_real(val.origin.y);
			f->store_real(val.origin.z);

		} break;
		case Variant::COLOR: {

			f->store_32(VARIANT_COLOR);
			Color val = p_property;
			f->store_real(val.r);
			f->store_real(val.g);
			f->store_real(val.b);
			f->store_real(val.a);

		} break;
		case Variant::IMAGE: {

			f->store_32(VARIANT_IMAGE);
			Image val = p_property;
			if (val.empty()) {
				f->store_32(IMAGE_ENCODING_EMPTY);
				break;
			}

			f->store_32(IMAGE_ENCODING_RAW);

			f->store_32(val.get_width());
			f->store_32(val.get_height());
			f->store_32(val.get_mipmaps() ? 1 : 0);
			switch (val.get_format()) {
				case Image::FORMAT_GRAYSCALE:
					f->store_32(IMAGE_FORMAT_L8);
					break; ///< one byte per pixel: f->store_32(IMAGE_FORMAT_ ); break; 0-255
				case Image::FORMAT_INTENSITY:
					f->store_32(IMAGE_FORMAT_L8);
					break; ///< one byte per pixel: f->store_32(IMAGE_FORMAT_ ); break; 0-255
				case Image::FORMAT_GRAYSCALE_ALPHA:
					f->store_32(IMAGE_FORMAT_LA8);
					break; ///< two bytes per pixel: f->store_32(IMAGE_FORMAT_ ); break; 0-255. alpha 0-255
				case Image::FORMAT_RGB:
					f->store_32(IMAGE_FORMAT_RGB8);
					break; ///< one byte R: f->store_32(IMAGE_FORMAT_ ); break; one byte G: f->store_32(IMAGE_FORMAT_ ); break; one byte B
				case Image::FORMAT_RGBA:
					f->store_32(IMAGE_FORMAT_RGBA8);
					break; ///< one byte R: f->store_32(IMAGE_FORMAT_ ); break; one byte G: f->store_32(IMAGE_FORMAT_ ); break; one byte B: f->store_32(IMAGE_FORMAT_ ); break; one byte A
				case Image::FORMAT_BC1:
					f->store_32(IMAGE_FORMAT_DXT1);
					break; // DXT1
				case Image::FORMAT_BC2:
					f->store_32(IMAGE_FORMAT_DXT3);
					break; // DXT3
				case Image::FORMAT_BC3:
					f->store_32(IMAGE_FORMAT_DXT5);
					break; // DXT5
				case Image::FORMAT_BC4:
					f->store_32(IMAGE_FORMAT_ATI1);
					break; // ATI1
				case Image::FORMAT_BC5:
					f->store_32(IMAGE_FORMAT_ATI2);
					break; // ATI2
				case Image::FORMAT_PVRTC2: f->store_32(IMAGE_FORMAT_PVRTC2); break;
				case Image::FORMAT_PVRTC2_ALPHA: f->store_32(IMAGE_FORMAT_PVRTC2A); break;
				case Image::FORMAT_PVRTC4: f->store_32(IMAGE_FORMAT_PVRTC4); break;
				case Image::FORMAT_PVRTC4_ALPHA: f->store_32(IMAGE_FORMAT_PVRTC4A); break;
				case Image::FORMAT_ETC:
					f->store_32(IMAGE_FORMAT_ETC);
					break; // regular ETC: f->store_32(IMAGE_FORMAT_ ); break; no transparency
				default: f->store_32(IMAGE_FORMAT_L8); break;
			}

			int dlen = val.get_data().size();
			f->store_32(dlen);
			DVector<uint8_t>::Read r = val.get_data().read();
			f->store_buffer(r.ptr(), dlen);
			_pad_buffer(dlen, f);

		} break;
		case Variant::NODE_PATH: {
			f->store_32(VARIANT_NODE_PATH);
			NodePath np = p_property;
			f->store_16(np.get_name_count());
			uint16_t snc = np.get_subname_count();
			if (np.is_absolute())
				snc |= 0x8000;
			f->store_16(snc);
			for (int i = 0; i < np.get_name_count(); i++) {
				save_unicode_string(np.get_name(i), f, true);
			}
			for (int i = 0; i < np.get_subname_count(); i++) {
				save_unicode_string(np.get_subname(i), f, true);
			}

			save_unicode_string(np.get_property(), f, true);

		} break;
		case Variant::_RID: {

			f->store_32(VARIANT_RID);
			WARN_PRINT("Can't save RIDs");
			RID val = p_property;
			f->store_32(val.get_id());
		} break;
		case Variant::OBJECT: {

			ERR_FAIL();

		} break;
		case Variant::INPUT_EVENT: {

			f->store_32(VARIANT_INPUT_EVENT);
			//InputEvent event = p_property;
			f->store_32(0); //event type none, nothing else supported for now.

		} break;
		case Variant::DICTIONARY: {

			f->store_32(VARIANT_DICTIONARY);
			Dictionary d = p_property;
			f->store_32(uint32_t(d.size()));

			List<Variant> keys;
			d.get_key_list(&keys);

			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

				/*
				if (!_check_type(dict[E->get()]))
					continue;
				*/

				_save_binary_property(E->get(), f);
				_save_binary_property(d[E->get()], f);
			}

		} break;
		case Variant::ARRAY: {

			f->store_32(VARIANT_ARRAY);
			Array a = p_property;
			f->store_32(uint32_t(a.size()));
			for (int i = 0; i < a.size(); i++) {

				_save_binary_property(a[i], f);
			}

		} break;
		case Variant::RAW_ARRAY: {

			f->store_32(VARIANT_RAW_ARRAY);
			DVector<uint8_t> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<uint8_t>::Read r = arr.read();
			f->store_buffer(r.ptr(), len);
			_pad_buffer(len, f);

		} break;
		case Variant::INT_ARRAY: {

			f->store_32(VARIANT_INT_ARRAY);
			DVector<int> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<int>::Read r = arr.read();
			for (int i = 0; i < len; i++)
				f->store_32(r[i]);

		} break;
		case Variant::REAL_ARRAY: {

			f->store_32(VARIANT_REAL_ARRAY);
			DVector<real_t> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<real_t>::Read r = arr.read();
			for (int i = 0; i < len; i++) {
				f->store_real(r[i]);
			}

		} break;
		case Variant::STRING_ARRAY: {

			f->store_32(VARIANT_STRING_ARRAY);
			DVector<String> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<String>::Read r = arr.read();
			for (int i = 0; i < len; i++) {
				save_unicode_string(r[i], f);
			}

		} break;
		case Variant::VECTOR3_ARRAY: {

			f->store_32(VARIANT_VECTOR3_ARRAY);
			DVector<Vector3> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<Vector3>::Read r = arr.read();
			for (int i = 0; i < len; i++) {
				f->store_real(r[i].x);
				f->store_real(r[i].y);
				f->store_real(r[i].z);
			}

		} break;
		case Variant::VECTOR2_ARRAY: {

			f->store_32(VARIANT_VECTOR2_ARRAY);
			DVector<Vector2> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<Vector2>::Read r = arr.read();
			for (int i = 0; i < len; i++) {
				f->store_real(r[i].x);
				f->store_real(r[i].y);
			}

		} break;
		case Variant::COLOR_ARRAY: {

			f->store_32(VARIANT_COLOR_ARRAY);
			DVector<Color> arr = p_property;
			int len = arr.size();
			f->store_32(len);
			DVector<Color>::Read r = arr.read();
			for (int i = 0; i < len; i++) {
				f->store_real(r[i].r);
				f->store_real(r[i].g);
				f->store_real(r[i].b);
				f->store_real(r[i].a);
			}

		} break;
		default: {

			ERR_EXPLAIN("Invalid variant");
			ERR_FAIL();
		}
	}
}

void EditorExportGodot3::_save_binary(const String &p_path, ExportData &resource) {

	FileAccessRef f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND(!f.operator->());

	//save header compressed
	static const uint8_t header[4] = { 'R', 'S', 'R', 'C' };
	f->store_buffer(header, 4);
	f->store_32(0);

	f->store_32(0); //64 bits file, false for now
	f->store_32(3); //major
	f->store_32(0); //minor
	f->store_32(2); //format version (2 is for 3.0)

	//f->store_32(saved_resources.size()+external_resources.size()); // load steps -not needed
	save_unicode_string(resource.resources[resource.resources.size() - 1].type, f.operator->());
	for (int i = 0; i < 16; i++)
		f->store_32(0); // unused

	f->store_32(0); //no names saved

	f->store_32(resource.dependencies.size()); //amount of external resources

	for (Map<int, ExportData::Dependency>::Element *E = resource.dependencies.front(); E; E = E->next()) {

		save_unicode_string(E->get().type, f.operator->());
		save_unicode_string(resource_replace_map[E->get().path], f.operator->());
	}

	// save internal resource table
	Vector<uint64_t> ofs_pos;
	f->store_32(resource.resources.size()); //amount of internal resources
	for (int i = 0; i < resource.resources.size(); i++) {

		save_unicode_string("local://" + itos(resource.resources[i].index), f.operator->());
		ofs_pos.push_back(f->get_pos());
		f->store_64(0);
	}

	Vector<uint64_t> ofs_table;
	//	int saved_idx=0;
	//now actually save the resources

	for (int i = 0; i < resource.resources.size(); i++) {

		ofs_table.push_back(f->get_pos());
		save_unicode_string(resource.resources[i].type, f.operator->());
		f->store_32(resource.resources[i].properties.size());

		for (List<ExportData::PropertyData>::Element *E = resource.resources[i].properties.front(); E; E = E->next()) {

			save_unicode_string(E->get().name, f.operator->(), true);
			_save_binary_property(E->get().value, f.operator->());
		}
	}

	for (int i = 0; i < ofs_table.size(); i++) {
		f->seek(ofs_pos[i]);
		f->store_64(ofs_table[i]);
	}

	f->seek_end();

	f->store_buffer((const uint8_t *)"RSRC", 4); //magic at end

	ERR_FAIL_COND(f->get_error() != OK);
}

void EditorExportGodot3::_save_config(const String &p_path) {

	// Parse existing config, convert persisting properties and store in ConfigFile
	ConfigFile new_cfg = ConfigFile();

	List<PropertyInfo> props;
	Globals::get_singleton()->get_property_list(&props);

	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

		if (!Globals::get_singleton()->has(E->get().name))
			continue;

		if (Globals::get_singleton()->is_persisting(E->get().name)) {
			String newname;
			if (globals_rename_map.has(E->get().name)) {
				newname = globals_rename_map[E->get().name];
			} else {
				newname = E->get().name;
			}

			int sep = newname.find("/");
			String section = newname.substr(0, sep);
			String subname = newname.substr(sep + 1, newname.length());
			String value;
			_get_property_as_text(Globals::get_singleton()->get(E->get().name), value);
			new_cfg.set_value(section, subname, value);
		}
	}

	String str = "{\n\"flags/filter\": " + String(GLOBAL_DEF("image_loader/filter", true) ? "true" : "false");
	str += ",\n\"flags/mipmaps\": " + String(GLOBAL_DEF("image_loader/gen_mipmaps", true) ? "true" : "false");
	str += "\n}";
	new_cfg.set_value("importer_defaults", "texture", str);

	// Write the collected ConfigFile manually - we need to use _get_property_as_text()
	// above, so we can't rely on ConfigFile.save() to properly store the raw strings.
	FileAccessRef f = FileAccess::open(p_path.plus_file("project.godot"), FileAccess::WRITE);

	List<String> sections;
	new_cfg.get_sections(&sections);

	for (List<String>::Element *E = sections.front(); E; E = E->next()) {

		f->store_line("[" + E->get() + "]\n");

		List<String> keys;
		new_cfg.get_section_keys(E->get(), &keys);

		for (List<String>::Element *F = keys.front(); F; F = F->next()) {

			f->store_line(F->get() + " = " + new_cfg.get_value(E->get(), F->get()));
		}
		f->store_line("");
	}

	f->close();
}

Error EditorExportGodot3::_convert_script(const String &p_path, const String &p_target_path, bool mark_converted_lines) {

	FileAccessRef src = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(!src.operator->(), FAILED);
	FileAccessRef dst = FileAccess::open(p_target_path, FileAccess::WRITE);
	ERR_FAIL_COND_V(!dst.operator->(), FAILED);

	String http_var = "";
	const String note = "  #-- NOTE: Automatically converted by Godot 2 to 3 converter, please review";

	while (!src->eof_reached()) {
		String line = src->get_line();
		String origline = line;

		// Convert _fixed_process( => _physics_process(
		RegEx regexp("(.*)_fixed_process\\((.*)");
		int res = regexp.find(line);
		if (res >= 0 && regexp.get_capture_count() == 3) {
			line = regexp.get_capture(1) + "_physics_process(" + regexp.get_capture(2);
		}
		regexp.clear();

		// Convert _input_event( => _gui_input(
		regexp.compile("(.*)_input_event\\((.*)");
		res = regexp.find(line);
		if (res >= 0 && regexp.get_capture_count() == 3) {
			line = regexp.get_capture(1) + "_gui_input(" + regexp.get_capture(2);
		}
		regexp.clear();

		// Try to detect a HTTPClient object
		regexp.compile("[ \t]*([a-zA-Z0-9_]*)[ ]*=[ ]*HTTPClient\\.new\\(\\)");
		res = regexp.find(line);
		if (res >= 0 && regexp.get_capture_count() == 2) {
			http_var = regexp.get_capture(1).strip_edges();
		}
		regexp.clear();

		if (http_var != "") {
			// Convert .connect( => .connect_to_host(
			regexp.compile("(.*)" + http_var + "\\.connect\\((.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + http_var + ".connect_to_host(" + regexp.get_capture(2);
			}
			regexp.clear();
		}

		// The following replacements may be needed more than once per line, hence the loop
		int count;
		int tries = 0;
		do {
			count = 0;

			// Convert all types to fix instances of renamed Nodes, or renamed core types (Pool*Array, Basis, etc.)
			for (Map<String, String>::Element *E = type_rename_map.front(); E; E = E->next()) {
				//regexp.compile("(.*[^a-zA-Z0-9_])" + E->key() + "([^a-zA-Z0-9_].*)");
				regexp.compile("(.*\\b)" + E->key() + "(\\b.*)");
				res = regexp.find(line);
				if (res >= 0 && regexp.get_capture_count() == 3) {
					line = regexp.get_capture(1) + E->get() + regexp.get_capture(2);
					count++;
				}
				regexp.clear();
			}

			// Convert _pos( => _position(
			regexp.compile("(.*)_pos\\((.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "_position(" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert .pos => .position
			regexp.compile("(.*)\\.pos([^a-zA-Z0-9_-].*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + ".position" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert _rot( => _rotation(
			regexp.compile("(.*)_rot\\((.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "_rotation(" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert _speed( => _speed_scale(
			regexp.compile("(.*)_speed\\((.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "_speed_scale(" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert KEY_RETURN => KEY_ENTER
			regexp.compile("(.*)KEY_RETURN(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "KEY_ENTER" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert get_opacity() => modulate.a
			regexp.compile("(.*)get_opacity\\(\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "modulate.a" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert set_opacity(var) => modulate.a = var
			regexp.compile("(.*)set_opacity\\((.*)\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 4) {
				line = regexp.get_capture(1) + "modulate.a = " + regexp.get_capture(2) + regexp.get_capture(3);
				count++;
			}
			regexp.clear();

			// Convert get_self_opacity() => self_modulate.a
			regexp.compile("(.*)get_self_opacity\\(\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "self_modulate.a" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert set_self_opacity(var) => self_modulate.a = var
			regexp.compile("(.*)set_self_opacity\\((.*)\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 4) {
				line = regexp.get_capture(1) + "self_modulate.a = " + regexp.get_capture(2) + regexp.get_capture(3);
				count++;
			}
			regexp.clear();

			// Convert set_hidden(var) => visible = !(var)
			regexp.compile("(.*)set_hidden\\((.*)\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 4) {
				line = regexp.get_capture(1) + "visible = !(" + regexp.get_capture(2) + ")" + regexp.get_capture(3);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.KEY => var is InputEventKey
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.KEY(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventKey" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.MOUSE_MOTION => var is InputEventMouseMotion
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.MOUSE_MOTION(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventMouseMotion" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.MOUSE_BUTTON => var is InputEventMouseButton
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.MOUSE_BUTTON(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventMouseButton" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.JOYSTICK_MOTION => var is InputEventJoypadMotion
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.JOYSTICK_MOTION(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventJoypadMotion" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.JOYSTICK_BUTTON => var is InputEventJoypadButton
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.JOYSTICK_BUTTON(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventJoypadButton" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.SCREEN_TOUCH => var is InputEventScreenTouch
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.SCREEN_TOUCH(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventScreenTouch" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert var.type == InputEvent.SCREEN_DRAG => var is InputEventScreenDrag
			regexp.compile("(.*)\\.type[ ]*==[ ]*InputEvent.SCREEN_DRAG(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + " is InputEventScreenDrag" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert move( => move_and_collide(
			regexp.compile("(.*)move\\((.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "move_and_collide(" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert is_move_and_slide_on_floor() => is_on_floor()
			regexp.compile("(.*)is_move_and_slide_on_floor\\(\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "is_on_floor()" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert is_move_and_slide_on_ceiling() => is_on_ceiling()
			regexp.compile("(.*)is_move_and_slide_on_ceiling\\(\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "is_on_ceiling()" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert is_move_and_slide_on_wall() => is_on_wall()
			regexp.compile("(.*)is_move_and_slide_on_wall\\(\\)(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "is_on_wall()" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

			// Convert <any chars but none> extends => <any chars but none> is
			// The only case where we don't want to convert it is `^extends <Node>`
			regexp.compile("(^.+ )extends(.*)");
			res = regexp.find(line);
			if (res >= 0 && regexp.get_capture_count() == 3) {
				line = regexp.get_capture(1) + "is" + regexp.get_capture(2);
				count++;
			}
			regexp.clear();

		} while (count >= 1 && tries++ < 10);

		if (mark_converted_lines && line != origline) {
			// Add explanatory comment on the changed line
			line += note;
		}

		dst->store_line(line);
	}

	return OK;
}

Error EditorExportGodot3::export_godot3(const String &p_path, bool convert_scripts, bool mark_converted_lines) {

	List<String> files;
	_find_files(EditorFileSystem::get_singleton()->get_filesystem(), &files);

	EditorProgress progress("exporting", "Exporting the project to Godot 3.0", files.size());

	//find XML resources

	resource_replace_map.clear();

	Set<String> xml_extensions;
	Set<String> binary_extensions;
	Set<String> text_extensions;

	{
		List<String> xml_exts;
		ResourceFormatLoaderXML::singleton->get_recognized_extensions(&xml_exts);
		for (List<String>::Element *E = xml_exts.front(); E; E = E->next()) {
			xml_extensions.insert(E->get());
		}
	}

	{
		List<String> binary_exts;
		ResourceFormatLoaderBinary::singleton->get_recognized_extensions(&binary_exts);
		for (List<String>::Element *E = binary_exts.front(); E; E = E->next()) {
			binary_extensions.insert(E->get());
		}
	}

	{
		List<String> text_exts;
		ResourceFormatLoaderText::singleton->get_recognized_extensions(&text_exts);
		for (List<String>::Element *E = text_exts.front(); E; E = E->next()) {
			text_extensions.insert(E->get());
		}
	}

	for (List<String>::Element *E = files.front(); E; E = E->next()) {

		String file = E->get();
		String file_local = file.replace("res://", "");

		resource_replace_map[file] = file;
		resource_replace_map[file_local] = file_local;

		if (xml_extensions.has(file.extension().to_lower())) {
			if (ResourceLoader::get_resource_type(file) == "PackedScene") {
				resource_replace_map[file] = file.basename() + ".tscn";
				resource_replace_map[file_local] = file_local.basename() + ".tscn";
			} else {
				resource_replace_map[file] = file.basename() + ".tres";
				resource_replace_map[file_local] = file_local.basename() + ".tres";
			}
		}

		// Changing all the old extensions to new Godot 3.0 extensions.
		// Refer PR #9201
		String extension = file.extension().to_lower();
		if (extension == "anm") {
			resource_replace_map[file] = file.basename() + ".anim";
			resource_replace_map[file_local] = file_local.basename() + ".anim";
		} else if (extension == "asogg") {
			resource_replace_map[file] = file.basename() + ".oggstr";
			resource_replace_map[file_local] = file_local.basename() + ".oggstr";
		} else if (extension == "atex") {
			resource_replace_map[file] = file.basename() + ".atlastex";
			resource_replace_map[file_local] = file_local.basename() + ".atlastex";
		} else if (extension == "cbm") {
			resource_replace_map[file] = file.basename() + ".cubemap";
			resource_replace_map[file_local] = file_local.basename() + ".cubemap";
		} else if (extension == "cvtex") {
			resource_replace_map[file] = file.basename() + ".curvetex";
			resource_replace_map[file_local] = file_local.basename() + ".curvetex";
		} else if (extension == "fnt") {
			resource_replace_map[file] = file.basename() + ".font";
			resource_replace_map[file_local] = file_local.basename() + ".font";
		} else if (extension == "gt") {
			resource_replace_map[file] = file.basename() + ".meshlib";
			resource_replace_map[file_local] = file_local.basename() + ".meshlib";
		} else if (extension == "ltex") {
			resource_replace_map[file] = file.basename() + ".largetex";
			resource_replace_map[file_local] = file_local.basename() + ".largetex";
		} else if (extension == "mmsh") {
			resource_replace_map[file] = file.basename() + ".multimesh";
			resource_replace_map[file_local] = file_local.basename() + ".multimesh";
		} else if (extension == "msh") {
			resource_replace_map[file] = file.basename() + ".mesh";
			resource_replace_map[file_local] = file_local.basename() + ".mesh";
		} else if (extension == "mtl") {
			resource_replace_map[file] = file.basename() + ".material";
			resource_replace_map[file_local] = file_local.basename() + ".material";
		} else if (extension == "sbx") {
			resource_replace_map[file] = file.basename() + ".stylebox";
			resource_replace_map[file_local] = file_local.basename() + ".stylebox";
		} else if (extension == "sgp") {
			resource_replace_map[file] = file.basename() + ".vshader";
			resource_replace_map[file_local] = file_local.basename() + ".vshader";
		} else if (extension == "shd") {
			resource_replace_map[file] = file.basename() + ".shader";
			resource_replace_map[file_local] = file_local.basename() + ".shader";
		} else if (extension == "shp") {
			resource_replace_map[file] = file.basename() + ".shape";
			resource_replace_map[file_local] = file_local.basename() + ".shape";
		} else if (extension == "smp") {
			resource_replace_map[file] = file.basename() + ".sample";
			resource_replace_map[file_local] = file_local.basename() + ".sample";
		} else if (extension == "tex") {
			resource_replace_map[file] = file.basename() + ".texture";
			resource_replace_map[file_local] = file_local.basename() + ".texture";
		} else if (extension == "thm") {
			resource_replace_map[file] = file.basename() + ".theme";
			resource_replace_map[file_local] = file_local.basename() + ".theme";
		} else if (extension == "wrd") {
			resource_replace_map[file] = file.basename() + ".world";
			resource_replace_map[file_local] = file_local.basename() + ".world";
		} else if (extension == "xl") {
			resource_replace_map[file] = file.basename() + ".translation";
			resource_replace_map[file_local] = file_local.basename() + ".translation";
		}
	}

	DirAccess *directory = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	if (directory->change_dir(p_path) != OK) {
		memdelete(directory);
		ERR_FAIL_V(ERR_CANT_OPEN);
	}

	int idx = 0;
	for (List<String>::Element *E = files.front(); E; E = E->next()) {

		String path = E->get();
		String extension = path.extension().to_lower();

		String target_path;
		bool repack = false;

		target_path = p_path.plus_file(path.replace("res://", ""));

		// Changing all the old extensions to new Godot 3.0 extensions.
		// Refer PR #9201
		if (extension == "fnt") {
			target_path = target_path.basename() + ".font";
		} else if (extension == "asogg") {
			target_path = target_path.basename() + ".oggstr";
		} else if (extension == "atex") {
			target_path = target_path.basename() + ".atlastex";
		} else if (extension == "cbm") {
			target_path = target_path.basename() + ".cubemap";
		} else if (extension == "cvtex") {
			target_path = target_path.basename() + ".curvetex";
		} else if (extension == "fnt") {
			target_path = target_path.basename() + ".font";
		} else if (extension == "gt") {
			target_path = target_path.basename() + ".meshlib";
		} else if (extension == "ltex") {
			target_path = target_path.basename() + ".largetex";
		} else if (extension == "mmsh") {
			target_path = target_path.basename() + ".multimesh";
		} else if (extension == "msh") {
			target_path = target_path.basename() + ".mesh";
		} else if (extension == "mtl") {
			target_path = target_path.basename() + ".material";
		} else if (extension == "sbx") {
			target_path = target_path.basename() + ".stylebox";
		} else if (extension == "sgp") {
			target_path = target_path.basename() + ".vshader";
		} else if (extension == "shd") {
			target_path = target_path.basename() + ".shader";
		} else if (extension == "shp") {
			target_path = target_path.basename() + ".shape";
		} else if (extension == "smp") {
			target_path = target_path.basename() + ".sample";
		} else if (extension == "tex") {
			target_path = target_path.basename() + ".texture";
		} else if (extension == "thm") {
			target_path = target_path.basename() + ".theme";
		} else if (extension == "wrd") {
			target_path = target_path.basename() + ".world";
		} else if (extension == "xl") {
			target_path = target_path.basename() + ".translation";
		}

		progress.step(target_path.get_file(), idx++);

		print_line("-- Exporting file: " + target_path);

		if (directory->make_dir_recursive(target_path.get_base_dir()) != OK) {
			memdelete(directory);
			ERR_FAIL_V(ERR_CANT_CREATE);
		}

		ExportData resource_data;

		Error err;
		bool cont = false;
		if (xml_extensions.has(extension)) {

			err = ResourceLoader::get_export_data(path, resource_data);
		} else if (text_extensions.has(extension)) {

			err = ResourceLoader::get_export_data(path, resource_data);
		} else if (binary_extensions.has(extension)) {

			err = ResourceLoader::get_export_data(path, resource_data);
		} else {

			if (convert_scripts && extension == "gd") {
				err = _convert_script(path, target_path, mark_converted_lines);
			} else {
				//single file, copy it
				err = directory->copy(path, target_path);
			}

			cont = true; //no longer needed to do anything, just copied the file!
		}

		if (err != OK) {
			memdelete(directory);
			ERR_FAIL_V(err);
		}

		if (cont) {
			continue;
		}

		if (resource_data.nodes.size() == 0 && resource_data.resources[resource_data.resources.size() - 1].type == "PackedScene") {
			//must unpack a PackedScene
			_unpack_packed_scene(resource_data);
			repack = true;
		}

		_convert_resources(resource_data);

		if (repack) {

			_pack_packed_scene(resource_data);
		}

		if (xml_extensions.has(extension)) {

			String save_path = resource_replace_map[target_path];
			_save_text(save_path, resource_data);
		} else if (text_extensions.has(extension)) {
			_save_text(target_path, resource_data);
		} else if (binary_extensions.has(extension)) {
			_save_binary(target_path, resource_data);
		}
	}

	memdelete(directory);
	_save_config(p_path);

	return OK;
}

EditorExportGodot3::EditorExportGodot3() {

	int idx = 0;
	while (globals_renames[idx][0] != NULL) {

		globals_rename_map[globals_renames[idx][0]] = globals_renames[idx][1];
		idx++;
	}

	idx = 0;
	while (prop_renames[idx][0] != NULL) {

		prop_rename_map[prop_renames[idx][0]] = prop_renames[idx][1];
		idx++;
	}

	idx = 0;
	while (type_renames[idx][0] != NULL) {

		type_rename_map[type_renames[idx][0]] = type_renames[idx][1];
		idx++;
	}

	idx = 0;
	while (signal_renames[idx][0] != NULL) {

		signal_rename_map[signal_renames[idx][0]] = signal_renames[idx][1];
		idx++;
	}
}
