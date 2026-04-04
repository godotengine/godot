/**************************************************************************/
/*  viewport.hpp                                                          */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioListener2D;
class AudioListener3D;
class Camera2D;
class Camera3D;
class Control;
class InputEvent;
class Texture2D;
class ViewportTexture;
class Window;
class World2D;
class World3D;

class Viewport : public Node {
	GDEXTENSION_CLASS(Viewport, Node)

public:
	enum PositionalShadowAtlasQuadrantSubdiv {
		SHADOW_ATLAS_QUADRANT_SUBDIV_DISABLED = 0,
		SHADOW_ATLAS_QUADRANT_SUBDIV_1 = 1,
		SHADOW_ATLAS_QUADRANT_SUBDIV_4 = 2,
		SHADOW_ATLAS_QUADRANT_SUBDIV_16 = 3,
		SHADOW_ATLAS_QUADRANT_SUBDIV_64 = 4,
		SHADOW_ATLAS_QUADRANT_SUBDIV_256 = 5,
		SHADOW_ATLAS_QUADRANT_SUBDIV_1024 = 6,
		SHADOW_ATLAS_QUADRANT_SUBDIV_MAX = 7,
	};

	enum Scaling3DMode {
		SCALING_3D_MODE_BILINEAR = 0,
		SCALING_3D_MODE_FSR = 1,
		SCALING_3D_MODE_FSR2 = 2,
		SCALING_3D_MODE_METALFX_SPATIAL = 3,
		SCALING_3D_MODE_METALFX_TEMPORAL = 4,
		SCALING_3D_MODE_MAX = 5,
	};

	enum MSAA {
		MSAA_DISABLED = 0,
		MSAA_2X = 1,
		MSAA_4X = 2,
		MSAA_8X = 3,
		MSAA_MAX = 4,
	};

	enum AnisotropicFiltering {
		ANISOTROPY_DISABLED = 0,
		ANISOTROPY_2X = 1,
		ANISOTROPY_4X = 2,
		ANISOTROPY_8X = 3,
		ANISOTROPY_16X = 4,
		ANISOTROPY_MAX = 5,
	};

	enum ScreenSpaceAA {
		SCREEN_SPACE_AA_DISABLED = 0,
		SCREEN_SPACE_AA_FXAA = 1,
		SCREEN_SPACE_AA_SMAA = 2,
		SCREEN_SPACE_AA_MAX = 3,
	};

	enum RenderInfo {
		RENDER_INFO_OBJECTS_IN_FRAME = 0,
		RENDER_INFO_PRIMITIVES_IN_FRAME = 1,
		RENDER_INFO_DRAW_CALLS_IN_FRAME = 2,
		RENDER_INFO_MAX = 3,
	};

	enum RenderInfoType {
		RENDER_INFO_TYPE_VISIBLE = 0,
		RENDER_INFO_TYPE_SHADOW = 1,
		RENDER_INFO_TYPE_CANVAS = 2,
		RENDER_INFO_TYPE_MAX = 3,
	};

	enum DebugDraw {
		DEBUG_DRAW_DISABLED = 0,
		DEBUG_DRAW_UNSHADED = 1,
		DEBUG_DRAW_LIGHTING = 2,
		DEBUG_DRAW_OVERDRAW = 3,
		DEBUG_DRAW_WIREFRAME = 4,
		DEBUG_DRAW_NORMAL_BUFFER = 5,
		DEBUG_DRAW_VOXEL_GI_ALBEDO = 6,
		DEBUG_DRAW_VOXEL_GI_LIGHTING = 7,
		DEBUG_DRAW_VOXEL_GI_EMISSION = 8,
		DEBUG_DRAW_SHADOW_ATLAS = 9,
		DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS = 10,
		DEBUG_DRAW_SCENE_LUMINANCE = 11,
		DEBUG_DRAW_SSAO = 12,
		DEBUG_DRAW_SSIL = 13,
		DEBUG_DRAW_PSSM_SPLITS = 14,
		DEBUG_DRAW_DECAL_ATLAS = 15,
		DEBUG_DRAW_SDFGI = 16,
		DEBUG_DRAW_SDFGI_PROBES = 17,
		DEBUG_DRAW_GI_BUFFER = 18,
		DEBUG_DRAW_DISABLE_LOD = 19,
		DEBUG_DRAW_CLUSTER_OMNI_LIGHTS = 20,
		DEBUG_DRAW_CLUSTER_SPOT_LIGHTS = 21,
		DEBUG_DRAW_CLUSTER_DECALS = 22,
		DEBUG_DRAW_CLUSTER_REFLECTION_PROBES = 23,
		DEBUG_DRAW_OCCLUDERS = 24,
		DEBUG_DRAW_MOTION_VECTORS = 25,
		DEBUG_DRAW_INTERNAL_BUFFER = 26,
	};

	enum DefaultCanvasItemTextureFilter {
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST = 0,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR = 1,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS = 2,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS = 3,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_MAX = 4,
	};

	enum DefaultCanvasItemTextureRepeat {
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED = 0,
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_ENABLED = 1,
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MIRROR = 2,
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MAX = 3,
	};

	enum SDFOversize {
		SDF_OVERSIZE_100_PERCENT = 0,
		SDF_OVERSIZE_120_PERCENT = 1,
		SDF_OVERSIZE_150_PERCENT = 2,
		SDF_OVERSIZE_200_PERCENT = 3,
		SDF_OVERSIZE_MAX = 4,
	};

	enum SDFScale {
		SDF_SCALE_100_PERCENT = 0,
		SDF_SCALE_50_PERCENT = 1,
		SDF_SCALE_25_PERCENT = 2,
		SDF_SCALE_MAX = 3,
	};

	enum VRSMode {
		VRS_DISABLED = 0,
		VRS_TEXTURE = 1,
		VRS_XR = 2,
		VRS_MAX = 3,
	};

	enum VRSUpdateMode {
		VRS_UPDATE_DISABLED = 0,
		VRS_UPDATE_ONCE = 1,
		VRS_UPDATE_ALWAYS = 2,
		VRS_UPDATE_MAX = 3,
	};

	void set_world_2d(const Ref<World2D> &p_world_2d);
	Ref<World2D> get_world_2d() const;
	Ref<World2D> find_world_2d() const;
	void set_canvas_transform(const Transform2D &p_xform);
	Transform2D get_canvas_transform() const;
	void set_global_canvas_transform(const Transform2D &p_xform);
	Transform2D get_global_canvas_transform() const;
	Transform2D get_stretch_transform() const;
	Transform2D get_final_transform() const;
	Transform2D get_screen_transform() const;
	Rect2 get_visible_rect() const;
	void set_transparent_background(bool p_enable);
	bool has_transparent_background() const;
	void set_use_hdr_2d(bool p_enable);
	bool is_using_hdr_2d() const;
	void set_msaa_2d(Viewport::MSAA p_msaa);
	Viewport::MSAA get_msaa_2d() const;
	void set_msaa_3d(Viewport::MSAA p_msaa);
	Viewport::MSAA get_msaa_3d() const;
	void set_screen_space_aa(Viewport::ScreenSpaceAA p_screen_space_aa);
	Viewport::ScreenSpaceAA get_screen_space_aa() const;
	void set_use_taa(bool p_enable);
	bool is_using_taa() const;
	void set_use_debanding(bool p_enable);
	bool is_using_debanding() const;
	void set_use_occlusion_culling(bool p_enable);
	bool is_using_occlusion_culling() const;
	void set_debug_draw(Viewport::DebugDraw p_debug_draw);
	Viewport::DebugDraw get_debug_draw() const;
	void set_use_oversampling(bool p_enable);
	bool is_using_oversampling() const;
	void set_oversampling_override(float p_oversampling);
	float get_oversampling_override() const;
	float get_oversampling() const;
	int32_t get_render_info(Viewport::RenderInfoType p_type, Viewport::RenderInfo p_info);
	Ref<ViewportTexture> get_texture() const;
	void set_physics_object_picking(bool p_enable);
	bool get_physics_object_picking();
	void set_physics_object_picking_sort(bool p_enable);
	bool get_physics_object_picking_sort();
	void set_physics_object_picking_first_only(bool p_enable);
	bool get_physics_object_picking_first_only();
	RID get_viewport_rid() const;
	void push_text_input(const String &p_text);
	void push_input(const Ref<InputEvent> &p_event, bool p_in_local_coords = false);
	void push_unhandled_input(const Ref<InputEvent> &p_event, bool p_in_local_coords = false);
	void notify_mouse_entered();
	void notify_mouse_exited();
	Vector2 get_mouse_position() const;
	void warp_mouse(const Vector2 &p_position);
	void update_mouse_cursor_state();
	void gui_cancel_drag();
	Variant gui_get_drag_data() const;
	String gui_get_drag_description() const;
	void gui_set_drag_description(const String &p_description);
	bool gui_is_dragging() const;
	bool gui_is_drag_successful() const;
	void gui_release_focus();
	Control *gui_get_focus_owner() const;
	Control *gui_get_hovered_control() const;
	void set_disable_input(bool p_disable);
	bool is_input_disabled() const;
	void set_positional_shadow_atlas_size(int32_t p_size);
	int32_t get_positional_shadow_atlas_size() const;
	void set_positional_shadow_atlas_16_bits(bool p_enable);
	bool get_positional_shadow_atlas_16_bits() const;
	void set_snap_controls_to_pixels(bool p_enabled);
	bool is_snap_controls_to_pixels_enabled() const;
	void set_snap_2d_transforms_to_pixel(bool p_enabled);
	bool is_snap_2d_transforms_to_pixel_enabled() const;
	void set_snap_2d_vertices_to_pixel(bool p_enabled);
	bool is_snap_2d_vertices_to_pixel_enabled() const;
	void set_positional_shadow_atlas_quadrant_subdiv(int32_t p_quadrant, Viewport::PositionalShadowAtlasQuadrantSubdiv p_subdiv);
	Viewport::PositionalShadowAtlasQuadrantSubdiv get_positional_shadow_atlas_quadrant_subdiv(int32_t p_quadrant) const;
	void set_input_as_handled();
	bool is_input_handled() const;
	void set_handle_input_locally(bool p_enable);
	bool is_handling_input_locally() const;
	void set_default_canvas_item_texture_filter(Viewport::DefaultCanvasItemTextureFilter p_mode);
	Viewport::DefaultCanvasItemTextureFilter get_default_canvas_item_texture_filter() const;
	void set_embedding_subwindows(bool p_enable);
	bool is_embedding_subwindows() const;
	TypedArray<Window> get_embedded_subwindows() const;
	void set_drag_threshold(int32_t p_threshold);
	int32_t get_drag_threshold() const;
	void set_canvas_cull_mask(uint32_t p_mask);
	uint32_t get_canvas_cull_mask() const;
	void set_canvas_cull_mask_bit(uint32_t p_layer, bool p_enable);
	bool get_canvas_cull_mask_bit(uint32_t p_layer) const;
	void set_default_canvas_item_texture_repeat(Viewport::DefaultCanvasItemTextureRepeat p_mode);
	Viewport::DefaultCanvasItemTextureRepeat get_default_canvas_item_texture_repeat() const;
	void set_sdf_oversize(Viewport::SDFOversize p_oversize);
	Viewport::SDFOversize get_sdf_oversize() const;
	void set_sdf_scale(Viewport::SDFScale p_scale);
	Viewport::SDFScale get_sdf_scale() const;
	void set_mesh_lod_threshold(float p_pixels);
	float get_mesh_lod_threshold() const;
	void set_as_audio_listener_2d(bool p_enable);
	bool is_audio_listener_2d() const;
	AudioListener2D *get_audio_listener_2d() const;
	Camera2D *get_camera_2d() const;
	void set_world_3d(const Ref<World3D> &p_world_3d);
	Ref<World3D> get_world_3d() const;
	Ref<World3D> find_world_3d() const;
	void set_use_own_world_3d(bool p_enable);
	bool is_using_own_world_3d() const;
	AudioListener3D *get_audio_listener_3d() const;
	Camera3D *get_camera_3d() const;
	void set_as_audio_listener_3d(bool p_enable);
	bool is_audio_listener_3d() const;
	void set_disable_3d(bool p_disable);
	bool is_3d_disabled() const;
	void set_use_xr(bool p_use);
	bool is_using_xr();
	void set_scaling_3d_mode(Viewport::Scaling3DMode p_scaling_3d_mode);
	Viewport::Scaling3DMode get_scaling_3d_mode() const;
	void set_scaling_3d_scale(float p_scale);
	float get_scaling_3d_scale() const;
	void set_fsr_sharpness(float p_fsr_sharpness);
	float get_fsr_sharpness() const;
	void set_texture_mipmap_bias(float p_texture_mipmap_bias);
	float get_texture_mipmap_bias() const;
	void set_anisotropic_filtering_level(Viewport::AnisotropicFiltering p_anisotropic_filtering_level);
	Viewport::AnisotropicFiltering get_anisotropic_filtering_level() const;
	void set_vrs_mode(Viewport::VRSMode p_mode);
	Viewport::VRSMode get_vrs_mode() const;
	void set_vrs_update_mode(Viewport::VRSUpdateMode p_mode);
	Viewport::VRSUpdateMode get_vrs_update_mode() const;
	void set_vrs_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_vrs_texture() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Viewport::PositionalShadowAtlasQuadrantSubdiv);
VARIANT_ENUM_CAST(Viewport::Scaling3DMode);
VARIANT_ENUM_CAST(Viewport::MSAA);
VARIANT_ENUM_CAST(Viewport::AnisotropicFiltering);
VARIANT_ENUM_CAST(Viewport::ScreenSpaceAA);
VARIANT_ENUM_CAST(Viewport::RenderInfo);
VARIANT_ENUM_CAST(Viewport::RenderInfoType);
VARIANT_ENUM_CAST(Viewport::DebugDraw);
VARIANT_ENUM_CAST(Viewport::DefaultCanvasItemTextureFilter);
VARIANT_ENUM_CAST(Viewport::DefaultCanvasItemTextureRepeat);
VARIANT_ENUM_CAST(Viewport::SDFOversize);
VARIANT_ENUM_CAST(Viewport::SDFScale);
VARIANT_ENUM_CAST(Viewport::VRSMode);
VARIANT_ENUM_CAST(Viewport::VRSUpdateMode);

