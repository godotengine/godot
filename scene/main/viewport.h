/*************************************************************************/
/*  viewport.h                                                           */
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

#ifndef VIEWPORT_H
#define VIEWPORT_H

#include "scene/main/node.h"
#include "scene/resources/texture.h"

#ifndef _3D_DISABLED
class Camera3D;
class CollisionObject3D;
class AudioListener3D;
class World3D;
#endif // _3D_DISABLED

class AudioListener2D;
class Camera2D;
class CanvasItem;
class CanvasLayer;
class Control;
class Label;
class SceneTreeTimer;
class Viewport;
class Window;
class World2D;

class ViewportTexture : public Texture2D {
	GDCLASS(ViewportTexture, Texture2D);

	NodePath path;

	friend class Viewport;
	Viewport *vp = nullptr;

	mutable RID proxy_ph;
	mutable RID proxy;

protected:
	static void _bind_methods();

public:
	void set_viewport_path_in_scene(const NodePath &p_path);
	NodePath get_viewport_path_in_scene() const;

	virtual void setup_local_to_scene() override;

	virtual int get_width() const override;
	virtual int get_height() const override;
	virtual Size2 get_size() const override;
	virtual RID get_rid() const override;

	virtual bool has_alpha() const override;

	virtual Ref<Image> get_image() const override;

	ViewportTexture();
	~ViewportTexture();
};

class Viewport : public Node {
	GDCLASS(Viewport, Node);

public:
	enum ShadowAtlasQuadrantSubdiv {
		SHADOW_ATLAS_QUADRANT_SUBDIV_DISABLED,
		SHADOW_ATLAS_QUADRANT_SUBDIV_1,
		SHADOW_ATLAS_QUADRANT_SUBDIV_4,
		SHADOW_ATLAS_QUADRANT_SUBDIV_16,
		SHADOW_ATLAS_QUADRANT_SUBDIV_64,
		SHADOW_ATLAS_QUADRANT_SUBDIV_256,
		SHADOW_ATLAS_QUADRANT_SUBDIV_1024,
		SHADOW_ATLAS_QUADRANT_SUBDIV_MAX,
	};

	enum MSAA {
		MSAA_DISABLED,
		MSAA_2X,
		MSAA_4X,
		MSAA_8X,
		// 16x MSAA is not supported due to its high cost and driver bugs.
		MSAA_MAX
	};

	enum ScreenSpaceAA {
		SCREEN_SPACE_AA_DISABLED,
		SCREEN_SPACE_AA_FXAA,
		SCREEN_SPACE_AA_MAX
	};

	enum RenderInfo {
		RENDER_INFO_OBJECTS_IN_FRAME,
		RENDER_INFO_PRIMITIVES_IN_FRAME,
		RENDER_INFO_DRAW_CALLS_IN_FRAME,
		RENDER_INFO_MAX
	};

	enum RenderInfoType {
		RENDER_INFO_TYPE_VISIBLE,
		RENDER_INFO_TYPE_SHADOW,
		RENDER_INFO_TYPE_MAX
	};

	enum DebugDraw {
		DEBUG_DRAW_DISABLED,
		DEBUG_DRAW_UNSHADED,
		DEBUG_DRAW_LIGHTING,
		DEBUG_DRAW_OVERDRAW,
		DEBUG_DRAW_WIREFRAME,
		DEBUG_DRAW_NORMAL_BUFFER,
		DEBUG_DRAW_VOXEL_GI_ALBEDO,
		DEBUG_DRAW_VOXEL_GI_LIGHTING,
		DEBUG_DRAW_VOXEL_GI_EMISSION,
		DEBUG_DRAW_SHADOW_ATLAS,
		DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS,
		DEBUG_DRAW_SCENE_LUMINANCE,
		DEBUG_DRAW_SSAO,
		DEBUG_DRAW_SSIL,
		DEBUG_DRAW_PSSM_SPLITS,
		DEBUG_DRAW_DECAL_ATLAS,
		DEBUG_DRAW_SDFGI,
		DEBUG_DRAW_SDFGI_PROBES,
		DEBUG_DRAW_GI_BUFFER,
		DEBUG_DRAW_DISABLE_LOD,
		DEBUG_DRAW_CLUSTER_OMNI_LIGHTS,
		DEBUG_DRAW_CLUSTER_SPOT_LIGHTS,
		DEBUG_DRAW_CLUSTER_DECALS,
		DEBUG_DRAW_CLUSTER_REFLECTION_PROBES,
		DEBUG_DRAW_OCCLUDERS,
	};

	enum DefaultCanvasItemTextureFilter {
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS,
		DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_MAX
	};

	enum DefaultCanvasItemTextureRepeat {
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED,
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_ENABLED,
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MIRROR,
		DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MAX,
	};

	enum SDFOversize {
		SDF_OVERSIZE_100_PERCENT,
		SDF_OVERSIZE_120_PERCENT,
		SDF_OVERSIZE_150_PERCENT,
		SDF_OVERSIZE_200_PERCENT,
		SDF_OVERSIZE_MAX
	};

	enum SDFScale {
		SDF_SCALE_100_PERCENT,
		SDF_SCALE_50_PERCENT,
		SDF_SCALE_25_PERCENT,
		SDF_SCALE_MAX
	};

	enum {
		SUBWINDOW_CANVAS_LAYER = 1024
	};

private:
	friend class ViewportTexture;

	Viewport *parent = nullptr;

	AudioListener2D *audio_listener_2d = nullptr;
	Camera2D *camera_2d = nullptr;
	Set<CanvasLayer *> canvas_layers;

	RID viewport;
	RID current_canvas;
	RID subwindow_canvas;

	bool is_audio_listener_2d_enabled = false;
	RID internal_audio_listener_2d;

	bool override_canvas_transform = false;

	Transform2D canvas_transform_override;
	Transform2D canvas_transform;
	Transform2D global_canvas_transform;
	Transform2D stretch_transform;

	Size2i size = Size2i(512, 512);
	Size2i size_2d_override;
	bool size_allocated = false;

	RID contact_2d_debug;
	RID contact_3d_debug_multimesh;
	RID contact_3d_debug_instance;

	Rect2 last_vp_rect;

	bool transparent_bg = false;
	bool filter;
	bool gen_mipmaps = false;

	bool snap_controls_to_pixels = true;
	bool snap_2d_transforms_to_pixel = false;
	bool snap_2d_vertices_to_pixel = false;

	bool physics_object_picking = false;
	List<Ref<InputEvent>> physics_picking_events;
	ObjectID physics_object_capture;
	ObjectID physics_object_over;
	Transform3D physics_last_object_transform;
	Transform3D physics_last_camera_transform;
	ObjectID physics_last_id;
	bool physics_has_last_mousepos = false;
	Vector2 physics_last_mousepos = Vector2(INFINITY, INFINITY);
	struct {
		bool alt = false;
		bool control = false;
		bool shift = false;
		bool meta = false;
		int mouse_mask = 0;

	} physics_last_mouse_state;

	bool handle_input_locally = true;
	bool local_input_handled = false;

	// Collider to frame
	Map<ObjectID, uint64_t> physics_2d_mouseover;
	// Collider & shape to frame
	Map<Pair<ObjectID, int>, uint64_t, PairSort<ObjectID, int>> physics_2d_shape_mouseover;
	// Cleans up colliders corresponding to old frames or all of them.
	void _cleanup_mouseover_colliders(bool p_clean_all_frames, bool p_paused_only, uint64_t p_frame_reference = 0);

	Ref<World2D> world_2d;

	Rect2i to_screen_rect;
	StringName input_group;
	StringName gui_input_group;
	StringName unhandled_input_group;
	StringName unhandled_key_input_group;

	void _update_audio_listener_2d();

	bool disable_3d = false;

	void _propagate_viewport_notification(Node *p_node, int p_what);

	void _update_global_transform();

	RID texture_rid;

	DebugDraw debug_draw = DEBUG_DRAW_DISABLED;

	int shadow_atlas_size = 2048;
	bool shadow_atlas_16_bits = true;
	ShadowAtlasQuadrantSubdiv shadow_atlas_quadrant_subdiv[4];

	MSAA msaa = MSAA_DISABLED;
	ScreenSpaceAA screen_space_aa = SCREEN_SPACE_AA_DISABLED;
	bool use_debanding = false;
	float lod_threshold = 1.0;
	bool use_occlusion_culling = false;

	Ref<ViewportTexture> default_texture;
	Set<ViewportTexture *> viewport_textures;

	SDFOversize sdf_oversize = SDF_OVERSIZE_120_PERCENT;
	SDFScale sdf_scale = SDF_SCALE_50_PERCENT;

	enum SubWindowDrag {
		SUB_WINDOW_DRAG_DISABLED,
		SUB_WINDOW_DRAG_MOVE,
		SUB_WINDOW_DRAG_CLOSE,
		SUB_WINDOW_DRAG_RESIZE,
	};

	enum SubWindowResize {
		SUB_WINDOW_RESIZE_DISABLED,
		SUB_WINDOW_RESIZE_TOP_LEFT,
		SUB_WINDOW_RESIZE_TOP,
		SUB_WINDOW_RESIZE_TOP_RIGHT,
		SUB_WINDOW_RESIZE_LEFT,
		SUB_WINDOW_RESIZE_RIGHT,
		SUB_WINDOW_RESIZE_BOTTOM_LEFT,
		SUB_WINDOW_RESIZE_BOTTOM,
		SUB_WINDOW_RESIZE_BOTTOM_RIGHT,
		SUB_WINDOW_RESIZE_MAX
	};

	struct SubWindow {
		Window *window = nullptr;
		RID canvas_item;
	};

	struct GUI {
		// info used when this is a window

		bool forced_mouse_focus = false; //used for menu buttons
		bool key_event_accepted = false;
		Control *mouse_focus = nullptr;
		Control *last_mouse_focus = nullptr;
		Control *mouse_click_grabber = nullptr;
		int mouse_focus_mask = 0;
		Control *key_focus = nullptr;
		Control *mouse_over = nullptr;
		Control *drag_mouse_over = nullptr;
		Vector2 drag_mouse_over_pos;
		Control *tooltip_control = nullptr;
		Window *tooltip_popup = nullptr;
		Label *tooltip_label = nullptr;
		Point2 tooltip_pos;
		Point2 last_mouse_pos;
		Point2 drag_accum;
		bool drag_attempted = false;
		Variant drag_data;
		ObjectID drag_preview_id;
		Ref<SceneTreeTimer> tooltip_timer;
		double tooltip_delay = 0.0;
		Transform2D focus_inv_xform;
		bool roots_order_dirty = false;
		List<Control *> roots;
		int canvas_sort_index = 0; //for sorting items with canvas as root
		bool dragging = false;
		bool embed_subwindows_hint = false;
		bool embedding_subwindows = false;

		Window *subwindow_focused = nullptr;
		SubWindowDrag subwindow_drag = SUB_WINDOW_DRAG_DISABLED;
		Vector2 subwindow_drag_from;
		Vector2 subwindow_drag_pos;
		Rect2i subwindow_drag_close_rect;
		bool subwindow_drag_close_inside = false;
		SubWindowResize subwindow_resize_mode;
		Rect2i subwindow_resize_from_rect;

		Vector<SubWindow> sub_windows;
	} gui;

	DefaultCanvasItemTextureFilter default_canvas_item_texture_filter = DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
	DefaultCanvasItemTextureRepeat default_canvas_item_texture_repeat = DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;

	bool disable_input = false;

	void _gui_call_input(Control *p_control, const Ref<InputEvent> &p_input);
	void _gui_call_notification(Control *p_control, int p_what);

	void _gui_sort_roots();
	Control *_gui_find_control_at_pos(CanvasItem *p_node, const Point2 &p_global, const Transform2D &p_xform, Transform2D &r_inv_xform);

	void _gui_input_event(Ref<InputEvent> p_event);

	_FORCE_INLINE_ Transform2D _get_input_pre_xform() const;

	Ref<InputEvent> _make_input_local(const Ref<InputEvent> &ev);

	friend class Control;

	List<Control *>::Element *_gui_add_root_control(Control *p_control);

	void _gui_remove_root_control(List<Control *>::Element *RI);

	String _gui_get_tooltip(Control *p_control, const Vector2 &p_pos, Control **r_tooltip_owner = nullptr);
	void _gui_cancel_tooltip();
	void _gui_show_tooltip();

	void _gui_remove_control(Control *p_control);
	void _gui_hide_control(Control *p_control);

	void _gui_force_drag(Control *p_base, const Variant &p_data, Control *p_control);
	void _gui_set_drag_preview(Control *p_base, Control *p_control);
	Control *_gui_get_drag_preview();

	void _gui_remove_focus_for_window(Node *p_window);
	void _gui_remove_focus();
	void _gui_unfocus_control(Control *p_control);
	bool _gui_control_has_focus(const Control *p_control);
	void _gui_control_grab_focus(Control *p_control);
	void _gui_grab_click_focus(Control *p_control);
	void _post_gui_grab_click_focus();
	void _gui_accept_event();

	Control *_gui_get_focus_owner();

	bool _gui_drop(Control *p_at_control, Point2 p_at_pos, bool p_just_check);

	friend class AudioListener2D;
	void _audio_listener_2d_set(AudioListener2D *p_listener);
	void _audio_listener_2d_remove(AudioListener2D *p_listener);

	friend class Camera2D;
	void _camera_2d_set(Camera2D *p_camera_2d);

	friend class CanvasLayer;
	void _canvas_layer_add(CanvasLayer *p_canvas_layer);
	void _canvas_layer_remove(CanvasLayer *p_canvas_layer);

	void _drop_mouse_focus();
	void _drop_physics_mouseover(bool p_paused_only = false);

	void _update_canvas_items(Node *p_node);

	void _gui_set_root_order_dirty();

	friend class Window;

	void _sub_window_update_order();
	void _sub_window_register(Window *p_window);
	void _sub_window_update(Window *p_window);
	void _sub_window_grab_focus(Window *p_window);
	void _sub_window_remove(Window *p_window);
	bool _sub_windows_forward_input(const Ref<InputEvent> &p_event);
	SubWindowResize _sub_window_get_resize_margin(Window *p_subwindow, const Point2 &p_point);

	virtual bool _can_consume_input_events() const { return true; }
	uint64_t event_count = 0;

protected:
	void _set_size(const Size2i &p_size, const Size2i &p_size_2d_override, const Rect2i &p_to_screen_rect, const Transform2D &p_stretch_transform, bool p_allocated);

	Size2i _get_size() const;
	Size2i _get_size_2d_override() const;
	bool _is_size_allocated() const;

	void _notification(int p_what);
	void _process_picking();
	static void _bind_methods();

public:
	uint64_t get_processed_events_count() const { return event_count; }

	AudioListener2D *get_audio_listener_2d() const;
	Camera2D *get_camera_2d() const;
	void set_as_audio_listener_2d(bool p_enable);
	bool is_audio_listener_2d() const;

	void update_canvas_items();

	Rect2 get_visible_rect() const;
	RID get_viewport_rid() const;

	void set_world_2d(const Ref<World2D> &p_world_2d);
	Ref<World2D> get_world_2d() const;
	Ref<World2D> find_world_2d() const;

	void enable_canvas_transform_override(bool p_enable);
	bool is_canvas_transform_override_enbled() const;

	void set_canvas_transform_override(const Transform2D &p_transform);
	Transform2D get_canvas_transform_override() const;

	void set_canvas_transform(const Transform2D &p_transform);
	Transform2D get_canvas_transform() const;

	void set_global_canvas_transform(const Transform2D &p_transform);
	Transform2D get_global_canvas_transform() const;

	Transform2D get_final_transform() const;

	void set_transparent_background(bool p_enable);
	bool has_transparent_background() const;

	Ref<ViewportTexture> get_texture() const;

	void set_shadow_atlas_size(int p_size);
	int get_shadow_atlas_size() const;

	void set_shadow_atlas_16_bits(bool p_16_bits);
	bool get_shadow_atlas_16_bits() const;

	void set_shadow_atlas_quadrant_subdiv(int p_quadrant, ShadowAtlasQuadrantSubdiv p_subdiv);
	ShadowAtlasQuadrantSubdiv get_shadow_atlas_quadrant_subdiv(int p_quadrant) const;

	void set_msaa(MSAA p_msaa);
	MSAA get_msaa() const;

	void set_screen_space_aa(ScreenSpaceAA p_screen_space_aa);
	ScreenSpaceAA get_screen_space_aa() const;

	void set_use_debanding(bool p_use_debanding);
	bool is_using_debanding() const;

	void set_lod_threshold(float p_pixels);
	float get_lod_threshold() const;

	void set_use_occlusion_culling(bool p_us_occlusion_culling);
	bool is_using_occlusion_culling() const;

	Vector2 get_camera_coords(const Vector2 &p_viewport_coords) const;
	Vector2 get_camera_rect_size() const;

	void push_text_input(const String &p_text);
	void push_input(const Ref<InputEvent> &p_event, bool p_local_coords = false);
	void push_unhandled_input(const Ref<InputEvent> &p_event, bool p_local_coords = false);

	void set_disable_input(bool p_disable);
	bool is_input_disabled() const;

	Vector2 get_mouse_position() const;
	void warp_mouse(const Vector2 &p_pos);

	void set_physics_object_picking(bool p_enable);
	bool get_physics_object_picking();

	Variant gui_get_drag_data() const;

	void gui_reset_canvas_sort_index();
	int gui_get_canvas_sort_index();

	TypedArray<String> get_configuration_warnings() const override;

	void set_debug_draw(DebugDraw p_debug_draw);
	DebugDraw get_debug_draw() const;

	int get_render_info(RenderInfoType p_type, RenderInfo p_info);

	void set_snap_controls_to_pixels(bool p_enable);
	bool is_snap_controls_to_pixels_enabled() const;

	void set_snap_2d_transforms_to_pixel(bool p_enable);
	bool is_snap_2d_transforms_to_pixel_enabled() const;

	void set_snap_2d_vertices_to_pixel(bool p_enable);
	bool is_snap_2d_vertices_to_pixel_enabled() const;

	void set_input_as_handled();
	bool is_input_handled() const;

	void set_handle_input_locally(bool p_enable);
	bool is_handling_input_locally() const;

	bool gui_is_dragging() const;

	Control *gui_find_control(const Point2 &p_global);

	void set_sdf_oversize(SDFOversize p_sdf_oversize);
	SDFOversize get_sdf_oversize() const;

	void set_sdf_scale(SDFScale p_sdf_scale);
	SDFScale get_sdf_scale() const;

	void set_default_canvas_item_texture_filter(DefaultCanvasItemTextureFilter p_filter);
	DefaultCanvasItemTextureFilter get_default_canvas_item_texture_filter() const;

	void set_default_canvas_item_texture_repeat(DefaultCanvasItemTextureRepeat p_repeat);
	DefaultCanvasItemTextureRepeat get_default_canvas_item_texture_repeat() const;

	virtual DisplayServer::WindowID get_window_id() const = 0;

	void set_embed_subwindows_hint(bool p_embed);
	bool get_embed_subwindows_hint() const;
	bool is_embedding_subwindows() const;

	Viewport *get_parent_viewport() const;
	Window *get_base_window() const;

	void pass_mouse_focus_to(Viewport *p_viewport, Control *p_control);

#ifndef _3D_DISABLED
	bool use_xr = false;
	float scale_3d = 1.0;
	friend class AudioListener3D;
	AudioListener3D *audio_listener_3d = nullptr;
	Set<AudioListener3D *> audio_listener_3d_set;
	bool is_audio_listener_3d_enabled = false;
	RID internal_audio_listener_3d;
	AudioListener3D *get_audio_listener_3d() const;
	void set_as_audio_listener_3d(bool p_enable);
	bool is_audio_listener_3d() const;
	void _update_audio_listener_3d();
	void _listener_transform_3d_changed_notify();
	void _audio_listener_3d_set(AudioListener3D *p_listener);
	bool _audio_listener_3d_add(AudioListener3D *p_listener); //true if first
	void _audio_listener_3d_remove(AudioListener3D *p_listener);
	void _audio_listener_3d_make_next_current(AudioListener3D *p_exclude);

	void _collision_object_3d_input_event(CollisionObject3D *p_object, Camera3D *p_camera, const Ref<InputEvent> &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape);

	struct Camera3DOverrideData {
		Transform3D transform;
		enum Projection {
			PROJECTION_PERSPECTIVE,
			PROJECTION_ORTHOGONAL
		};
		Projection projection = Projection::PROJECTION_PERSPECTIVE;
		real_t fov = 0.0;
		real_t size = 0.0;
		real_t z_near = 0.0;
		real_t z_far = 0.0;
		RID rid;

		operator bool() const {
			return rid != RID();
		}
	} camera_3d_override;

	friend class Camera3D;
	Camera3D *camera_3d = nullptr;
	Set<Camera3D *> camera_3d_set;
	Camera3D *get_camera_3d() const;
	void _camera_3d_transform_changed_notify();
	void _camera_3d_set(Camera3D *p_camera);
	bool _camera_3d_add(Camera3D *p_camera); //true if first
	void _camera_3d_remove(Camera3D *p_camera);
	void _camera_3d_make_next_current(Camera3D *p_exclude);

	void enable_camera_3d_override(bool p_enable);
	bool is_camera_3d_override_enabled() const;

	void set_camera_3d_override_transform(const Transform3D &p_transform);
	Transform3D get_camera_3d_override_transform() const;

	void set_camera_3d_override_perspective(real_t p_fovy_degrees, real_t p_z_near, real_t p_z_far);
	void set_camera_3d_override_orthogonal(real_t p_size, real_t p_z_near, real_t p_z_far);

	void set_disable_3d(bool p_disable);
	bool is_3d_disabled() const;

	Ref<World3D> world_3d;
	Ref<World3D> own_world_3d;
	void set_world_3d(const Ref<World3D> &p_world_3d);
	Ref<World3D> get_world_3d() const;
	Ref<World3D> find_world_3d() const;
	void _own_world_3d_changed();
	void set_use_own_world_3d(bool p_world_3d);
	bool is_using_own_world_3d() const;
	void _propagate_enter_world_3d(Node *p_node);
	void _propagate_exit_world_3d(Node *p_node);

	void set_use_xr(bool p_use_xr);
	bool is_using_xr();

	void set_scale_3d(float p_scale_3d);
	float get_scale_3d() const;
#endif // _3D_DISABLED

	Viewport();
	~Viewport();
};

class SubViewport : public Viewport {
	GDCLASS(SubViewport, Viewport);

public:
	enum ClearMode {
		CLEAR_MODE_ALWAYS,
		CLEAR_MODE_NEVER,
		CLEAR_MODE_ONCE
	};

	enum UpdateMode {
		UPDATE_DISABLED,
		UPDATE_ONCE, //then goes to disabled
		UPDATE_WHEN_VISIBLE, // default
		UPDATE_WHEN_PARENT_VISIBLE,
		UPDATE_ALWAYS
	};

private:
	UpdateMode update_mode = UPDATE_WHEN_VISIBLE;
	ClearMode clear_mode = CLEAR_MODE_ALWAYS;
	bool size_2d_override_stretch = false;

protected:
	static void _bind_methods();
	virtual DisplayServer::WindowID get_window_id() const override;
	Transform2D _stretch_transform();
	void _notification(int p_what);

public:
	void set_size(const Size2i &p_size);
	Size2i get_size() const;

	void set_size_2d_override(const Size2i &p_size);
	Size2i get_size_2d_override() const;

	void set_size_2d_override_stretch(bool p_enable);
	bool is_size_2d_override_stretch_enabled() const;

	void set_update_mode(UpdateMode p_mode);
	UpdateMode get_update_mode() const;

	void set_clear_mode(ClearMode p_mode);
	ClearMode get_clear_mode() const;

	SubViewport();
	~SubViewport();
};
VARIANT_ENUM_CAST(SubViewport::UpdateMode);
VARIANT_ENUM_CAST(Viewport::ShadowAtlasQuadrantSubdiv);
VARIANT_ENUM_CAST(Viewport::MSAA);
VARIANT_ENUM_CAST(Viewport::ScreenSpaceAA);
VARIANT_ENUM_CAST(Viewport::DebugDraw);
VARIANT_ENUM_CAST(Viewport::SDFScale);
VARIANT_ENUM_CAST(Viewport::SDFOversize);
VARIANT_ENUM_CAST(SubViewport::ClearMode);
VARIANT_ENUM_CAST(Viewport::RenderInfo);
VARIANT_ENUM_CAST(Viewport::RenderInfoType);
VARIANT_ENUM_CAST(Viewport::DefaultCanvasItemTextureFilter);
VARIANT_ENUM_CAST(Viewport::DefaultCanvasItemTextureRepeat);

#endif
