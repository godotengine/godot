/*************************************************************************/
/*  viewport.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "servers/visual_server.h"
#include "scene/resources/world_2d.h"
#include "math_2d.h"
#include "scene/resources/texture.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Camera;
class Viewport;

class RenderTargetTexture : public Texture {

	OBJ_TYPE( RenderTargetTexture, Texture );

	int flags;
friend class Viewport;
	Viewport *vp;


public:


	virtual int get_width() const;
	virtual int get_height() const;
	virtual Size2 get_size() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	RenderTargetTexture(Viewport *p_vp=NULL);

};

class Viewport : public Node {

	OBJ_TYPE( Viewport, Node );
public:

	enum RenderTargetUpdateMode {
		RENDER_TARGET_UPDATE_DISABLED,
		RENDER_TARGET_UPDATE_ONCE, //then goes to disabled
		RENDER_TARGET_UPDATE_WHEN_VISIBLE, // default
		RENDER_TARGET_UPDATE_ALWAYS
	};

private:

friend class RenderTargetTexture;
	Viewport *parent;

	Camera *camera;
	Set<Camera*> cameras;

	RID viewport;
	RID canvas_item;
	RID current_canvas;

	bool audio_listener;
	RID listener;

	bool audio_listener_2d;
	RID listener_2d;

	Matrix32 canvas_transform;
	Matrix32 global_canvas_transform;
	Matrix32 stretch_transform;

	Rect2 rect;
	Rect2 to_screen_rect;


	bool size_override;
	bool size_override_stretch;
	Size2 size_override_size;
	Size2 size_override_margin;

	Rect2 last_vp_rect;

	bool transparent_bg;
	bool render_target_vflip;
	bool render_target_filter;
	bool render_target_gen_mipmaps;

	bool physics_object_picking;
	List<InputEvent> physics_picking_events;
	ObjectID physics_object_capture;
	ObjectID physics_object_over;
	Vector2 physics_last_mousepos;
	void _test_new_mouseover(ObjectID new_collider);

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


	void _update_stretch_transform();
	void _update_global_transform();

	bool render_target;
	RenderTargetUpdateMode render_target_update_mode;
	RID render_target_texture_rid;
	Ref<RenderTargetTexture> render_target_texture;


	void update_worlds();

	_FORCE_INLINE_ Matrix32 _get_input_pre_xform() const;

	void _vp_enter_tree();
	void _vp_exit_tree();

	void _vp_input(const InputEvent& p_ev);
	void _vp_unhandled_input(const InputEvent& p_ev);
	void _make_input_local(InputEvent& ev);

friend class Camera;
	void _camera_transform_changed_notify();
	void _set_camera(Camera* p_camera);

protected:	
	void _notification(int p_what);
	static void _bind_methods();
public:


	Camera* get_camera() const;

	void set_as_audio_listener(bool p_enable);
	bool is_audio_listener() const;

	void set_as_audio_listener_2d(bool p_enable);
	bool is_audio_listener_2d() const;

	void set_rect(const Rect2& p_rect);	
	Rect2 get_rect() const;
	Rect2 get_visible_rect() const;
	RID get_viewport() const;

	void set_world(const Ref<World>& p_world);
	Ref<World> get_world() const;
	Ref<World> find_world() const;

	Ref<World2D> find_world_2d() const;


	void set_canvas_transform(const Matrix32& p_transform);
	Matrix32 get_canvas_transform() const;

	void set_global_canvas_transform(const Matrix32& p_transform);
	Matrix32 get_global_canvas_transform() const;

	Matrix32 get_final_transform() const;

	void set_transparent_background(bool p_enable);
	bool has_transparent_background() const;


	void set_size_override(bool p_enable,const Size2& p_size=Size2(-1,-1),const Vector2& p_margin=Vector2());
	Size2 get_size_override() const;
	bool is_size_override_enabled() const;
	void set_size_override_stretch(bool p_enable);
	bool is_size_override_stretch_enabled() const;

	void set_as_render_target(bool p_enable);
	bool is_set_as_render_target() const;

	void set_render_target_vflip(bool p_enable);
	bool get_render_target_vflip() const;

	void set_render_target_filter(bool p_enable);
	bool get_render_target_filter() const;

	void set_render_target_gen_mipmaps(bool p_enable);
	bool get_render_target_gen_mipmaps() const;

	void set_render_target_update_mode(RenderTargetUpdateMode p_mode);
	RenderTargetUpdateMode get_render_target_update_mode() const;
	Ref<RenderTargetTexture> get_render_target_texture() const;


	Vector2 get_camera_coords(const Vector2& p_viewport_coords) const;
	Vector2 get_camera_rect_size() const;

	void queue_screen_capture();
	Image get_screen_capture() const;

	void set_use_own_world(bool p_world);
	bool is_using_own_world() const;

	void input(const InputEvent& p_event);
	void unhandled_input(const InputEvent& p_event);

	void set_render_target_to_screen_rect(const Rect2& p_rect);
	Rect2 get_render_target_to_screen_rect() const;

	void set_physics_object_picking(bool p_enable);
	bool get_physics_object_picking();

	Viewport();	
	~Viewport();

};

VARIANT_ENUM_CAST(Viewport::RenderTargetUpdateMode);
#endif
