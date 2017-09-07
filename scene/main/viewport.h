
/*************************************************************************/
/*  viewport.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "math_2d.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"
#include "scene/resources/world_2d.h"
#include "servers/visual_server.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Camera;
class Camera2D;
class Listener;
class Control;
class CanvasItem;
class Panel;
class Label;
class Timer;
class Viewport;

class ViewportTexture : public Texture {

	GDCLASS(ViewportTexture, Texture);

	NodePath path;

	friend class Viewport;
	Viewport *vp;

protected:
	static void _bind_methods();

public:
	void set_viewport_path_in_scene(const NodePath &p_path);
	NodePath get_viewport_path_in_scene() const;

	virtual void setup_local_to_scene();

	virtual int get_width() const;
	virtual int get_height() const;
	virtual Size2 get_size() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	virtual Ref<Image> get_data() const;

	ViewportTexture();
	~ViewportTexture();
};

class Viewport : public Node {

	GDCLASS(Viewport, Node);

public:
	enum UpdateMode {
		UPDATE_DISABLED,
		UPDATE_ONCE, //then goes to disabled
		UPDATE_WHEN_VISIBLE, // default
		UPDATE_ALWAYS
	};

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
		MSAA_16X,
	};

	enum Usage {
		USAGE_2D,
		USAGE_2D_NO_SAMPLING,
		USAGE_3D,
		USAGE_3D_NO_EFFECTS,
	};

	enum RenderInfo {

		RENDER_INFO_OBJECTS_IN_FRAME,
		RENDER_INFO_VERTICES_IN_FRAME,
		RENDER_INFO_MATERIAL_CHANGES_IN_FRAME,
		RENDER_INFO_SHADER_CHANGES_IN_FRAME,
		RENDER_INFO_SURFACE_CHANGES_IN_FRAME,
		RENDER_INFO_DRAW_CALLS_IN_FRAME,
		RENDER_INFO_MAX
	};

	enum DebugDraw {
		DEBUG_DRAW_DISABLED,
		DEBUG_DRAW_UNSHADED,
		DEBUG_DRAW_OVERDRAW,
		DEBUG_DRAW_WIREFRAME,
	};

	enum ClearMode {

		CLEAR_MODE_ALWAYS,
		CLEAR_MODE_NEVER,
		CLEAR_MODE_ONLY_NEXT_FRAME
	};

private:
	friend class ViewportTexture;

	Viewport *parent;

	Listener *listener;
	Set<Listener *> listeners;

	bool arvr;

	Camera *camera;
	Set<Camera *> cameras;

	RID viewport;
	RID current_canvas;

	bool audio_listener;
	RID internal_listener;

	bool audio_listener_2d;
	RID internal_listener_2d;

	Transform2D canvas_transform;
	Transform2D global_canvas_transform;
	Transform2D stretch_transform;

	Size2 size;
	Rect2 to_screen_rect;

	RID contact_2d_debug;
	RID contact_3d_debug_multimesh;
	RID contact_3d_debug_instance;

	bool size_override;
	bool size_override_stretch;
	Size2 size_override_size;
	Size2 size_override_margin;

	Rect2 last_vp_rect;

	bool transparent_bg;
	bool vflip;
	ClearMode clear_mode;
	bool filter;
	bool gen_mipmaps;

	bool snap_controls_to_pixels;

	bool physics_object_picking;
	List<Ref<InputEvent> > physics_picking_events;
	ObjectID physics_object_capture;
	ObjectID physics_object_over;
	Vector2 physics_last_mousepos;
	void _test_new_mouseover(ObjectID new_collider);
	Map<ObjectID, uint64_t> physics_2d_mouseover;

	void _update_rect();

	void _parent_resized();
	void _parent_draw();
	void _parent_visibility_changed();

	Ref<World2D> world_2d;
	Ref<World> world;
	Ref<World> own_world;

	StringName input_group;
	StringName gui_input_group;
	StringName unhandled_input_group;
	StringName unhandled_key_input_group;

	void _update_listener();
	void _update_listener_2d();

	void _propagate_enter_world(Node *p_node);
	void _propagate_exit_world(Node *p_node);
	void _propagate_viewport_notification(Node *p_node, int p_what);

	void _update_stretch_transform();
	void _update_global_transform();

	bool disable_3d;
	UpdateMode update_mode;
	RID texture_rid;
	uint32_t texture_flags;

	DebugDraw debug_draw;

	Usage usage;

	int shadow_atlas_size;
	ShadowAtlasQuadrantSubdiv shadow_atlas_quadrant_subdiv[4];

	MSAA msaa;
	bool hdr;

	Ref<ViewportTexture> default_texture;
	Set<ViewportTexture *> viewport_textures;

	struct GUI {
		// info used when this is a window

		bool key_event_accepted;
		Control *mouse_focus;
		int mouse_focus_button;
		Control *key_focus;
		Control *mouse_over;
		Control *tooltip;
		Panel *tooltip_popup;
		Label *tooltip_label;
		Point2 tooltip_pos;
		Point2 last_mouse_pos;
		Point2 drag_accum;
		bool drag_attempted;
		Variant drag_data;
		Control *drag_preview;
		float tooltip_timer;
		float tooltip_delay;
		List<Control *> modal_stack;
		unsigned int cancelled_input_ID;
		Transform2D focus_inv_xform;
		bool subwindow_order_dirty;
		List<Control *> subwindows;
		bool roots_order_dirty;
		List<Control *> roots;
		int canvas_sort_index; //for sorting items with canvas as root

		GUI();
	} gui;

	bool disable_input;

	void _gui_call_input(Control *p_control, const Ref<InputEvent> &p_input);
	void _gui_sort_subwindows();
	void _gui_sort_roots();
	void _gui_sort_modal_stack();
	Control *_gui_find_control(const Point2 &p_global);
	Control *_gui_find_control_at_pos(CanvasItem *p_node, const Point2 &p_global, const Transform2D &p_xform, Transform2D &r_inv_xform);

	void _gui_input_event(Ref<InputEvent> p_event);

	void update_worlds();

	_FORCE_INLINE_ Transform2D _get_input_pre_xform() const;

	void _vp_enter_tree();
	void _vp_exit_tree();

	void _vp_input(const Ref<InputEvent> &p_ev);
	void _vp_input_text(const String &p_text);
	void _vp_unhandled_input(const Ref<InputEvent> &p_ev);
	Ref<InputEvent> _make_input_local(const Ref<InputEvent> &ev);

	friend class Control;

	List<Control *>::Element *_gui_add_root_control(Control *p_control);
	List<Control *>::Element *_gui_add_subwindow_control(Control *p_control);

	void _gui_set_subwindow_order_dirty();
	void _gui_set_root_order_dirty();

	void _gui_remove_modal_control(List<Control *>::Element *MI);
	void _gui_remove_from_modal_stack(List<Control *>::Element *MI, ObjectID p_prev_focus_owner);
	void _gui_remove_root_control(List<Control *>::Element *RI);
	void _gui_remove_subwindow_control(List<Control *>::Element *SI);

	void _gui_cancel_tooltip();
	void _gui_show_tooltip();

	void _gui_remove_control(Control *p_control);
	void _gui_hid_control(Control *p_control);

	void _gui_force_drag(Control *p_base, const Variant &p_data, Control *p_control);
	void _gui_set_drag_preview(Control *p_base, Control *p_control);

	bool _gui_is_modal_on_top(const Control *p_control);
	List<Control *>::Element *_gui_show_modal(Control *p_control);

	void _gui_remove_focus();
	void _gui_unfocus_control(Control *p_control);
	bool _gui_control_has_focus(const Control *p_control);
	void _gui_control_grab_focus(Control *p_control);
	void _gui_grab_click_focus(Control *p_control);
	void _gui_accept_event();

	Control *_gui_get_focus_owner();

	Vector2 _get_window_offset() const;

	bool _gui_drop(Control *p_at_control, Point2 p_at_pos, bool p_just_check);

	friend class Listener;
	void _listener_transform_changed_notify();
	void _listener_set(Listener *p_listener);
	bool _listener_add(Listener *p_listener); //true if first
	void _listener_remove(Listener *p_listener);
	void _listener_make_next_current(Listener *p_exclude);

	friend class Camera;
	void _camera_transform_changed_notify();
	void _camera_set(Camera *p_camera);
	bool _camera_add(Camera *p_camera); //true if first
	void _camera_remove(Camera *p_camera);
	void _camera_make_next_current(Camera *p_exclude);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	Listener *get_listener() const;
	Camera *get_camera() const;

	void set_use_arvr(bool p_use_arvr);
	bool use_arvr();

	void set_as_audio_listener(bool p_enable);
	bool is_audio_listener() const;

	void set_as_audio_listener_2d(bool p_enable);
	bool is_audio_listener_2d() const;

	void set_size(const Size2 &p_size);

	Size2 get_size() const;
	Rect2 get_visible_rect() const;
	RID get_viewport_rid() const;

	void set_world(const Ref<World> &p_world);
	void set_world_2d(const Ref<World2D> &p_world_2d);
	Ref<World> get_world() const;
	Ref<World> find_world() const;

	Ref<World2D> get_world_2d() const;
	Ref<World2D> find_world_2d() const;

	void set_canvas_transform(const Transform2D &p_transform);
	Transform2D get_canvas_transform() const;

	void set_global_canvas_transform(const Transform2D &p_transform);
	Transform2D get_global_canvas_transform() const;

	Transform2D get_final_transform() const;

	void set_transparent_background(bool p_enable);
	bool has_transparent_background() const;

	void set_size_override(bool p_enable, const Size2 &p_size = Size2(-1, -1), const Vector2 &p_margin = Vector2());
	Size2 get_size_override() const;

	bool is_size_override_enabled() const;
	void set_size_override_stretch(bool p_enable);
	bool is_size_override_stretch_enabled() const;

	void set_vflip(bool p_enable);
	bool get_vflip() const;

	void set_clear_mode(ClearMode p_mode);
	ClearMode get_clear_mode() const;

	void set_update_mode(UpdateMode p_mode);
	UpdateMode get_update_mode() const;
	Ref<ViewportTexture> get_texture() const;

	void set_shadow_atlas_size(int p_size);
	int get_shadow_atlas_size() const;

	void set_shadow_atlas_quadrant_subdiv(int p_quadrant, ShadowAtlasQuadrantSubdiv p_subdiv);
	ShadowAtlasQuadrantSubdiv get_shadow_atlas_quadrant_subdiv(int p_quadrant) const;

	void set_msaa(MSAA p_msaa);
	MSAA get_msaa() const;

	void set_hdr(bool p_hdr);
	bool get_hdr() const;

	Vector2 get_camera_coords(const Vector2 &p_viewport_coords) const;
	Vector2 get_camera_rect_size() const;

	void set_use_own_world(bool p_world);
	bool is_using_own_world() const;

	void input(const Ref<InputEvent> &p_event);
	void unhandled_input(const Ref<InputEvent> &p_event);

	void set_disable_input(bool p_disable);
	bool is_input_disabled() const;

	void set_disable_3d(bool p_disable);
	bool is_3d_disabled() const;

	void set_attach_to_screen_rect(const Rect2 &p_rect);
	Rect2 get_attach_to_screen_rect() const;

	Vector2 get_mouse_position() const;
	void warp_mouse(const Vector2 &p_pos);

	void set_physics_object_picking(bool p_enable);
	bool get_physics_object_picking();

	bool gui_has_modal_stack() const;

	Variant gui_get_drag_data() const;
	Control *get_modal_stack_top() const;

	void gui_reset_canvas_sort_index();
	int gui_get_canvas_sort_index();

	virtual String get_configuration_warning() const;

	void set_usage(Usage p_usage);
	Usage get_usage() const;

	void set_debug_draw(DebugDraw p_debug_draw);
	DebugDraw get_debug_draw() const;

	int get_render_info(RenderInfo p_info);

	void set_snap_controls_to_pixels(bool p_enable);
	bool is_snap_controls_to_pixels_enabled() const;

	Viewport();
	~Viewport();
};

VARIANT_ENUM_CAST(Viewport::UpdateMode);
VARIANT_ENUM_CAST(Viewport::ShadowAtlasQuadrantSubdiv);
VARIANT_ENUM_CAST(Viewport::MSAA);
VARIANT_ENUM_CAST(Viewport::Usage);
VARIANT_ENUM_CAST(Viewport::DebugDraw);
VARIANT_ENUM_CAST(Viewport::ClearMode);
VARIANT_ENUM_CAST(Viewport::RenderInfo);

#endif
