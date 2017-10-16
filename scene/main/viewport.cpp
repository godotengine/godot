/*************************************************************************/
/*  viewport.cpp                                                         */
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
#include "viewport.h"

#include "os/input.h"
#include "os/os.h"
#include "project_settings.h"
#include "scene/2d/collision_object_2d.h"
#include "scene/3d/camera.h"
#include "scene/3d/collision_object.h"
#include "scene/3d/listener.h"
#include "scene/3d/scenario_fx.h"
#include "scene/3d/spatial.h"
#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/main/timer.h"
#include "scene/resources/mesh.h"
#include "scene/scene_string_names.h"
#include "servers/physics_2d_server.h"

void ViewportTexture::setup_local_to_scene() {

	if (vp) {
		vp->viewport_textures.erase(this);
	}

	vp = NULL;

	Node *local_scene = get_local_scene();
	if (!local_scene) {
		return;
	}

	Node *vpn = local_scene->get_node(path);
	ERR_EXPLAIN("ViewportTexture: Path to node is invalid");
	ERR_FAIL_COND(!vpn);

	vp = Object::cast_to<Viewport>(vpn);

	ERR_EXPLAIN("ViewportTexture: Path to node does not point to a viewport");
	ERR_FAIL_COND(!vp);

	vp->viewport_textures.insert(this);
}

void ViewportTexture::set_viewport_path_in_scene(const NodePath &p_path) {

	if (path == p_path)
		return;

	path = p_path;

	if (get_local_scene()) {
		setup_local_to_scene();
	}
}

NodePath ViewportTexture::get_viewport_path_in_scene() const {

	return path;
}

int ViewportTexture::get_width() const {

	ERR_FAIL_COND_V(!vp, 0);
	return vp->size.width;
}
int ViewportTexture::get_height() const {

	ERR_FAIL_COND_V(!vp, 0);
	return vp->size.height;
}
Size2 ViewportTexture::get_size() const {

	ERR_FAIL_COND_V(!vp, Size2());
	return vp->size;
}
RID ViewportTexture::get_rid() const {

	ERR_FAIL_COND_V(!vp, RID());
	return vp->texture_rid;
}

bool ViewportTexture::has_alpha() const {

	return false;
}
Ref<Image> ViewportTexture::get_data() const {

	ERR_FAIL_COND_V(!vp, Ref<Image>());
	return VS::get_singleton()->texture_get_data(vp->texture_rid);
}
void ViewportTexture::set_flags(uint32_t p_flags) {

	if (!vp)
		return;

	vp->texture_flags = p_flags;
	VS::get_singleton()->texture_set_flags(vp->texture_rid, p_flags);
}

uint32_t ViewportTexture::get_flags() const {

	if (!vp)
		return 0;

	return vp->texture_flags;
}

void ViewportTexture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_viewport_path_in_scene", "path"), &ViewportTexture::set_viewport_path_in_scene);
	ClassDB::bind_method(D_METHOD("get_viewport_path_in_scene"), &ViewportTexture::get_viewport_path_in_scene);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "viewport_path"), "set_viewport_path_in_scene", "get_viewport_path_in_scene");
}

ViewportTexture::ViewportTexture() {

	vp = NULL;
	set_local_to_scene(true);
}

ViewportTexture::~ViewportTexture() {

	if (vp) {
		vp->viewport_textures.erase(this);
	}
}

/////////////////////////////////////

class TooltipPanel : public Panel {

	GDCLASS(TooltipPanel, Panel)
public:
	TooltipPanel(){};
};

class TooltipLabel : public Label {

	GDCLASS(TooltipLabel, Label)
public:
	TooltipLabel(){};
};

Viewport::GUI::GUI() {

	mouse_focus = NULL;
	mouse_focus_button = -1;
	key_focus = NULL;
	mouse_over = NULL;

	cancelled_input_ID = 0;
	tooltip = NULL;
	tooltip_popup = NULL;
	tooltip_label = NULL;
	subwindow_order_dirty = false;
}

/////////////////////////////////////
void Viewport::_update_stretch_transform() {

	if (size_override_stretch && size_override) {

		//print_line("sive override size "+size_override_size);
		//print_line("rect size "+size);
		stretch_transform = Transform2D();
		Size2 scale = size / (size_override_size + size_override_margin * 2);
		stretch_transform.scale(scale);
		stretch_transform.elements[2] = size_override_margin * scale;

	} else {

		stretch_transform = Transform2D();
	}

	_update_global_transform();
}

void Viewport::_update_rect() {

	if (!is_inside_tree())
		return;

	/*if (!render_target && parent_control) {

		Control *c = parent_control;

		rect.pos=Point2();
		rect.size=c->get_size();
	}*/
	/*
	VisualServer::ViewportRect vr;
	vr.x=rect.pos.x;
	vr.y=rect.pos.y;

	if (render_target) {
		vr.x=0;
		vr.y=0;
	}
	vr.width=rect.size.width;
	vr.height=rect.size.height;

	VisualServer::get_singleton()->viewport_set_rect(viewport,vr);
	last_vp_rect=rect;

	if (canvas_item.is_valid()) {
		VisualServer::get_singleton()->canvas_item_set_custom_rect(canvas_item,true,rect);
	}

	emit_signal("size_changed");
	texture->emit_changed();
*/
}

void Viewport::_parent_resized() {

	_update_rect();
}

void Viewport::_parent_draw() {
}

void Viewport::_parent_visibility_changed() {

	/*
	if (parent_control) {

		Control *c = parent_control;
		VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,c->is_visible_in_tree());

		_update_listener();
		_update_listener_2d();
	}
*/
}

void Viewport::_vp_enter_tree() {

	/*	if (parent_control) {

		Control *cparent=parent_control;
		RID parent_ci = cparent->get_canvas_item();
		ERR_FAIL_COND(!parent_ci.is_valid());
		canvas_item = VisualServer::get_singleton()->canvas_item_create();

		VisualServer::get_singleton()->canvas_item_set_parent(canvas_item,parent_ci);
		VisualServer::get_singleton()->canvas_item_set_visible(canvas_item,false);
		//VisualServer::get_singleton()->canvas_item_attach_viewport(canvas_item,viewport);
		parent_control->connect("resized",this,"_parent_resized");
		parent_control->connect("visibility_changed",this,"_parent_visibility_changed");
	} else if (!parent){

		//VisualServer::get_singleton()->viewport_attach_to_screen(viewport,0);

	}
*/
}

void Viewport::_vp_exit_tree() {

	/*
	if (parent_control) {

		parent_control->disconnect("resized",this,"_parent_resized");
	}

	if (parent_control) {

		parent_control->disconnect("visibility_changed",this,"_parent_visibility_changed");
	}

	if (canvas_item.is_valid()) {

		VisualServer::get_singleton()->free(canvas_item);
		canvas_item=RID();

	}

	if (!parent) {

		VisualServer::get_singleton()->viewport_detach(viewport);

	}
*/
}

void Viewport::update_worlds() {

	if (!is_inside_tree())
		return;

	Rect2 abstracted_rect = Rect2(Vector2(), get_visible_rect().size);
	Rect2 xformed_rect = (global_canvas_transform * canvas_transform).affine_inverse().xform(abstracted_rect);
	find_world_2d()->_update_viewport(this, xformed_rect);
	find_world_2d()->_update();

	find_world()->_update(get_tree()->get_frame());
}

void Viewport::_test_new_mouseover(ObjectID new_collider) {
#ifndef _3D_DISABLED
	if (new_collider != physics_object_over) {

		if (physics_object_over) {

			CollisionObject *co = Object::cast_to<CollisionObject>(ObjectDB::get_instance(physics_object_over));
			if (co) {
				co->_mouse_exit();
			}
		}

		if (new_collider) {

			CollisionObject *co = Object::cast_to<CollisionObject>(ObjectDB::get_instance(new_collider));
			if (co) {
				co->_mouse_enter();
			}
		}

		physics_object_over = new_collider;
	}
#endif
}

void Viewport::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (get_parent()) {
				parent = get_parent()->get_viewport();
				VisualServer::get_singleton()->viewport_set_parent_viewport(viewport, parent->get_viewport_rid());
			} else {
				parent = NULL;
			}

			current_canvas = find_world_2d()->get_canvas();
			VisualServer::get_singleton()->viewport_set_scenario(viewport, find_world()->get_scenario());
			VisualServer::get_singleton()->viewport_attach_canvas(viewport, current_canvas);

			_update_listener();
			_update_listener_2d();
			_update_rect();

			find_world_2d()->_register_viewport(this, Rect2());

			add_to_group("_viewports");
			if (get_tree()->is_debugging_collisions_hint()) {
				//2D
				Physics2DServer::get_singleton()->space_set_debug_contacts(find_world_2d()->get_space(), get_tree()->get_collision_debug_contact_count());
				contact_2d_debug = VisualServer::get_singleton()->canvas_item_create();
				VisualServer::get_singleton()->canvas_item_set_parent(contact_2d_debug, find_world_2d()->get_canvas());
				//3D
				PhysicsServer::get_singleton()->space_set_debug_contacts(find_world()->get_space(), get_tree()->get_collision_debug_contact_count());
				contact_3d_debug_multimesh = VisualServer::get_singleton()->multimesh_create();
				VisualServer::get_singleton()->multimesh_allocate(contact_3d_debug_multimesh, get_tree()->get_collision_debug_contact_count(), VS::MULTIMESH_TRANSFORM_3D, VS::MULTIMESH_COLOR_8BIT);
				VisualServer::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh, 0);
				VisualServer::get_singleton()->multimesh_set_mesh(contact_3d_debug_multimesh, get_tree()->get_debug_contact_mesh()->get_rid());
				contact_3d_debug_instance = VisualServer::get_singleton()->instance_create();
				VisualServer::get_singleton()->instance_set_base(contact_3d_debug_instance, contact_3d_debug_multimesh);
				VisualServer::get_singleton()->instance_set_scenario(contact_3d_debug_instance, find_world()->get_scenario());
				//VisualServer::get_singleton()->instance_geometry_set_flag(contact_3d_debug_instance, VS::INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS, true);
			}

			VS::get_singleton()->viewport_set_active(viewport, true);
		} break;
		case NOTIFICATION_READY: {
#ifndef _3D_DISABLED
			if (listeners.size() && !listener) {
				Listener *first = NULL;
				for (Set<Listener *>::Element *E = listeners.front(); E; E = E->next()) {

					if (first == NULL || first->is_greater_than(E->get())) {
						first = E->get();
					}
				}

				if (first)
					first->make_current();
			}

			if (cameras.size() && !camera) {
				//there are cameras but no current camera, pick first in tree and make it current
				Camera *first = NULL;
				for (Set<Camera *>::Element *E = cameras.front(); E; E = E->next()) {

					if (first == NULL || first->is_greater_than(E->get())) {
						first = E->get();
					}
				}

				if (first)
					first->make_current();
			}
#endif
		} break;
		case NOTIFICATION_EXIT_TREE: {

			_gui_cancel_tooltip();
			if (world_2d.is_valid())
				world_2d->_remove_viewport(this);

			/*
			if (!render_target)
				_vp_exit_tree();
			*/

			VisualServer::get_singleton()->viewport_set_scenario(viewport, RID());
			//			SpatialSoundServer::get_singleton()->listener_set_space(internal_listener, RID());
			VisualServer::get_singleton()->viewport_remove_canvas(viewport, current_canvas);
			if (contact_2d_debug.is_valid()) {
				VisualServer::get_singleton()->free(contact_2d_debug);
				contact_2d_debug = RID();
			}

			if (contact_3d_debug_multimesh.is_valid()) {
				VisualServer::get_singleton()->free(contact_3d_debug_multimesh);
				VisualServer::get_singleton()->free(contact_3d_debug_instance);
				contact_3d_debug_instance = RID();
				contact_3d_debug_multimesh = RID();
			}

			remove_from_group("_viewports");

			VS::get_singleton()->viewport_set_active(viewport, false);

		} break;
		case NOTIFICATION_PHYSICS_PROCESS: {

			if (gui.tooltip_timer >= 0) {
				gui.tooltip_timer -= get_physics_process_delta_time();
				if (gui.tooltip_timer < 0) {
					_gui_show_tooltip();
				}
			}

			if (get_tree()->is_debugging_collisions_hint() && contact_2d_debug.is_valid()) {

				VisualServer::get_singleton()->canvas_item_clear(contact_2d_debug);
				VisualServer::get_singleton()->canvas_item_set_draw_index(contact_2d_debug, 0xFFFFF); //very high index

				Vector<Vector2> points = Physics2DServer::get_singleton()->space_get_contacts(find_world_2d()->get_space());
				int point_count = Physics2DServer::get_singleton()->space_get_contact_count(find_world_2d()->get_space());
				Color ccol = get_tree()->get_debug_collision_contact_color();

				for (int i = 0; i < point_count; i++) {

					VisualServer::get_singleton()->canvas_item_add_rect(contact_2d_debug, Rect2(points[i] - Vector2(2, 2), Vector2(5, 5)), ccol);
				}
			}

			if (get_tree()->is_debugging_collisions_hint() && contact_3d_debug_multimesh.is_valid()) {

				Vector<Vector3> points = PhysicsServer::get_singleton()->space_get_contacts(find_world()->get_space());
				int point_count = PhysicsServer::get_singleton()->space_get_contact_count(find_world()->get_space());

				VS::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh, point_count);
			}

			if (physics_object_picking && (to_screen_rect == Rect2() || Input::get_singleton()->get_mouse_mode() != Input::MOUSE_MODE_CAPTURED)) {

				Vector2 last_pos(1e20, 1e20);
				CollisionObject *last_object = NULL;
				ObjectID last_id = 0;
				PhysicsDirectSpaceState::RayResult result;
				Physics2DDirectSpaceState *ss2d = Physics2DServer::get_singleton()->space_get_direct_state(find_world_2d()->get_space());

				bool motion_tested = false;

				while (physics_picking_events.size()) {

					Ref<InputEvent> ev = physics_picking_events.front()->get();
					physics_picking_events.pop_front();

					Vector2 pos;

					Ref<InputEventMouseMotion> mm = ev;

					if (mm.is_valid()) {

						pos = mm->get_position();
						motion_tested = true;
						physics_last_mousepos = pos;
					}

					Ref<InputEventMouseButton> mb = ev;

					if (mb.is_valid()) {
						pos = mb->get_position();
					}

					Ref<InputEventScreenDrag> sd = ev;

					if (sd.is_valid()) {
						pos = sd->get_position();
					}

					Ref<InputEventScreenTouch> st = ev;

					if (st.is_valid()) {
						pos = st->get_position();
					}

					if (ss2d) {
						//send to 2D

						uint64_t frame = get_tree()->get_frame();

						Vector2 point = get_canvas_transform().affine_inverse().xform(pos);
						Physics2DDirectSpaceState::ShapeResult res[64];
						int rc = ss2d->intersect_point(point, res, 64, Set<RID>(), 0xFFFFFFFF, 0xFFFFFFFF, true);
						for (int i = 0; i < rc; i++) {

							if (res[i].collider_id && res[i].collider) {
								CollisionObject2D *co = Object::cast_to<CollisionObject2D>(res[i].collider);
								if (co) {

									Map<ObjectID, uint64_t>::Element *E = physics_2d_mouseover.find(res[i].collider_id);
									if (!E) {
										E = physics_2d_mouseover.insert(res[i].collider_id, frame);
										co->_mouse_enter();
									} else {
										E->get() = frame;
									}

									co->_input_event(this, ev, res[i].shape);
								}
							}
						}

						List<Map<ObjectID, uint64_t>::Element *> to_erase;

						for (Map<ObjectID, uint64_t>::Element *E = physics_2d_mouseover.front(); E; E = E->next()) {
							if (E->get() != frame) {
								Object *o = ObjectDB::get_instance(E->key());
								if (o) {

									CollisionObject2D *co = Object::cast_to<CollisionObject2D>(o);
									if (co) {
										co->_mouse_exit();
									}
								}
								to_erase.push_back(E);
							}
						}

						while (to_erase.size()) {
							physics_2d_mouseover.erase(to_erase.front()->get());
							to_erase.pop_front();
						}
					}

#ifndef _3D_DISABLED
					bool captured = false;

					if (physics_object_capture != 0) {

						CollisionObject *co = Object::cast_to<CollisionObject>(ObjectDB::get_instance(physics_object_capture));
						if (co) {
							co->_input_event(camera, ev, Vector3(), Vector3(), 0);
							captured = true;
							if (mb.is_valid() && mb->get_button_index() == 1 && !mb->is_pressed()) {
								physics_object_capture = 0;
							}

						} else {
							physics_object_capture = 0;
						}
					}

					if (captured) {
						//none
					} else if (pos == last_pos) {

						if (last_id) {
							if (ObjectDB::get_instance(last_id) && last_object) {
								//good, exists
								last_object->_input_event(camera, ev, result.position, result.normal, result.shape);
								if (last_object->get_capture_input_on_drag() && mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
									physics_object_capture = last_id;
								}
							}
						}
					} else {

						if (camera) {

							Vector3 from = camera->project_ray_origin(pos);
							Vector3 dir = camera->project_ray_normal(pos);

							PhysicsDirectSpaceState *space = PhysicsServer::get_singleton()->space_get_direct_state(find_world()->get_space());
							if (space) {

								bool col = space->intersect_ray(from, from + dir * 10000, result, Set<RID>(), 0xFFFFFFFF, 0xFFFFFFFF, true);
								ObjectID new_collider = 0;
								if (col) {

									CollisionObject *co = Object::cast_to<CollisionObject>(result.collider);
									if (co) {

										co->_input_event(camera, ev, result.position, result.normal, result.shape);
										last_object = co;
										last_id = result.collider_id;
										new_collider = last_id;
										if (co->get_capture_input_on_drag() && mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
											physics_object_capture = last_id;
										}
									}
								}

								if (mm.is_valid()) {
									_test_new_mouseover(new_collider);
								}
							}

							last_pos = pos;
						}
					}
				}

				if (!motion_tested && camera && physics_last_mousepos != Vector2(1e20, 1e20)) {

					//test anyway for mouseenter/exit because objects might move
					Vector3 from = camera->project_ray_origin(physics_last_mousepos);
					Vector3 dir = camera->project_ray_normal(physics_last_mousepos);

					PhysicsDirectSpaceState *space = PhysicsServer::get_singleton()->space_get_direct_state(find_world()->get_space());
					if (space) {

						bool col = space->intersect_ray(from, from + dir * 10000, result, Set<RID>(), 0xFFFFFFFF, 0xFFFFFFFF, true);
						ObjectID new_collider = 0;
						if (col) {
							CollisionObject *co = Object::cast_to<CollisionObject>(result.collider);
							if (co) {
								new_collider = result.collider_id;
							}
						}

						_test_new_mouseover(new_collider);
					}
#endif
				}
			}

		} break;
	}
}

RID Viewport::get_viewport_rid() const {

	return viewport;
}

void Viewport::set_use_arvr(bool p_use_arvr) {
	arvr = p_use_arvr;

	VS::get_singleton()->viewport_set_use_arvr(viewport, arvr);
}

bool Viewport::use_arvr() {
	return arvr;
}

void Viewport::set_size(const Size2 &p_size) {

	if (size == p_size.floor())
		return;
	size = p_size.floor();
	VS::get_singleton()->viewport_set_size(viewport, size.width, size.height);

	_update_rect();
	_update_stretch_transform();

	emit_signal("size_changed");
}

Rect2 Viewport::get_visible_rect() const {

	Rect2 r;

	if (size == Size2()) {

		r = Rect2(Point2(), Size2(OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height));
	} else {

		r = Rect2(Point2(), size);
	}

	if (size_override) {
		r.size = size_override_size;
	}

	return r;
}

Size2 Viewport::get_size() const {

	return size;
}

void Viewport::_update_listener() {
	/*
	if (is_inside_tree() && audio_listener && (camera || listener) && (!get_parent() || (Object::cast_to<Control>(get_parent()) && Object::cast_to<Control>(get_parent())->is_visible_in_tree())))  {
		SpatialSoundServer::get_singleton()->listener_set_space(internal_listener, find_world()->get_sound_space());
	} else {
		SpatialSoundServer::get_singleton()->listener_set_space(internal_listener, RID());
	}
*/
}

void Viewport::_update_listener_2d() {

	/*
	if (is_inside_tree() && audio_listener && (!get_parent() || (Object::cast_to<Control>(get_parent()) && Object::cast_to<Control>(get_parent())->is_visible_in_tree())))
		SpatialSound2DServer::get_singleton()->listener_set_space(internal_listener_2d, find_world_2d()->get_sound_space());
	else
		SpatialSound2DServer::get_singleton()->listener_set_space(internal_listener_2d, RID());
*/
}

void Viewport::set_as_audio_listener(bool p_enable) {

	if (p_enable == audio_listener)
		return;

	audio_listener = p_enable;
	_update_listener();
}

bool Viewport::is_audio_listener() const {

	return audio_listener;
}

void Viewport::set_as_audio_listener_2d(bool p_enable) {

	if (p_enable == audio_listener_2d)
		return;

	audio_listener_2d = p_enable;

	_update_listener_2d();
}

bool Viewport::is_audio_listener_2d() const {

	return audio_listener_2d;
}

void Viewport::set_canvas_transform(const Transform2D &p_transform) {

	canvas_transform = p_transform;
	VisualServer::get_singleton()->viewport_set_canvas_transform(viewport, find_world_2d()->get_canvas(), canvas_transform);

	Transform2D xform = (global_canvas_transform * canvas_transform).affine_inverse();
	Size2 ss = get_visible_rect().size;
	/*SpatialSound2DServer::get_singleton()->listener_set_transform(internal_listener_2d, Transform2D(0, xform.xform(ss*0.5)));
	Vector2 ss2 = ss*xform.get_scale();
	float panrange = MAX(ss2.x,ss2.y);

	SpatialSound2DServer::get_singleton()->listener_set_param(internal_listener_2d, SpatialSound2DServer::LISTENER_PARAM_PAN_RANGE, panrange);
*/
}

Transform2D Viewport::get_canvas_transform() const {

	return canvas_transform;
}

void Viewport::_update_global_transform() {

	Transform2D sxform = stretch_transform * global_canvas_transform;

	VisualServer::get_singleton()->viewport_set_global_canvas_transform(viewport, sxform);

	Transform2D xform = (sxform * canvas_transform).affine_inverse();
	Size2 ss = get_visible_rect().size;
	/*SpatialSound2DServer::get_singleton()->listener_set_transform(internal_listener_2d, Transform2D(0, xform.xform(ss*0.5)));
	Vector2 ss2 = ss*xform.get_scale();
	float panrange = MAX(ss2.x,ss2.y);

	SpatialSound2DServer::get_singleton()->listener_set_param(internal_listener_2d, SpatialSound2DServer::LISTENER_PARAM_PAN_RANGE, panrange);
*/
}

void Viewport::set_global_canvas_transform(const Transform2D &p_transform) {

	global_canvas_transform = p_transform;

	_update_global_transform();
}

Transform2D Viewport::get_global_canvas_transform() const {

	return global_canvas_transform;
}

void Viewport::_listener_transform_changed_notify() {

#ifndef _3D_DISABLED
//if (listener)
//		SpatialSoundServer::get_singleton()->listener_set_transform(internal_listener, listener->get_listener_transform());
#endif
}

void Viewport::_listener_set(Listener *p_listener) {

#ifndef _3D_DISABLED

	if (listener == p_listener)
		return;

	listener = p_listener;

	_update_listener();
	_listener_transform_changed_notify();
#endif
}

bool Viewport::_listener_add(Listener *p_listener) {

	listeners.insert(p_listener);
	return listeners.size() == 1;
}

void Viewport::_listener_remove(Listener *p_listener) {

	listeners.erase(p_listener);
	if (listener == p_listener) {
		listener = NULL;
	}
}

#ifndef _3D_DISABLED
void Viewport::_listener_make_next_current(Listener *p_exclude) {

	if (listeners.size() > 0) {
		for (Set<Listener *>::Element *E = listeners.front(); E; E = E->next()) {

			if (p_exclude == E->get())
				continue;
			if (!E->get()->is_inside_tree())
				continue;
			if (listener != NULL)
				return;

			E->get()->make_current();
		}
	} else {
		// Attempt to reset listener to the camera position
		if (camera != NULL) {
			_update_listener();
			_camera_transform_changed_notify();
		}
	}
}
#endif

void Viewport::_camera_transform_changed_notify() {

#ifndef _3D_DISABLED
// If there is an active listener in the scene, it takes priority over the camera
//	if (camera && !listener)
//		SpatialSoundServer::get_singleton()->listener_set_transform(internal_listener, camera->get_camera_transform());
#endif
}

void Viewport::_camera_set(Camera *p_camera) {

#ifndef _3D_DISABLED

	if (camera == p_camera)
		return;

	if (camera && find_world().is_valid()) {
		camera->notification(Camera::NOTIFICATION_LOST_CURRENT);
	}
	camera = p_camera;
	if (camera)
		VisualServer::get_singleton()->viewport_attach_camera(viewport, camera->get_camera());
	else
		VisualServer::get_singleton()->viewport_attach_camera(viewport, RID());

	if (camera && find_world().is_valid()) {
		camera->notification(Camera::NOTIFICATION_BECAME_CURRENT);
	}

	_update_listener();
	_camera_transform_changed_notify();
#endif
}

bool Viewport::_camera_add(Camera *p_camera) {

	cameras.insert(p_camera);
	return cameras.size() == 1;
}

void Viewport::_camera_remove(Camera *p_camera) {

	cameras.erase(p_camera);
	if (camera == p_camera) {
		camera = NULL;
	}
}

#ifndef _3D_DISABLED
void Viewport::_camera_make_next_current(Camera *p_exclude) {

	for (Set<Camera *>::Element *E = cameras.front(); E; E = E->next()) {

		if (p_exclude == E->get())
			continue;
		if (!E->get()->is_inside_tree())
			continue;
		if (camera != NULL)
			return;

		E->get()->make_current();
	}
}
#endif

void Viewport::set_transparent_background(bool p_enable) {

	transparent_bg = p_enable;
	VS::get_singleton()->viewport_set_transparent_background(viewport, p_enable);
}

bool Viewport::has_transparent_background() const {

	return transparent_bg;
}

void Viewport::set_world_2d(const Ref<World2D> &p_world_2d) {
	if (world_2d == p_world_2d)
		return;

	if (parent && parent->find_world_2d() == p_world_2d) {
		WARN_PRINT("Unable to use parent world as world_2d");
		return;
	}

	if (is_inside_tree()) {
		find_world_2d()->_remove_viewport(this);
		VisualServer::get_singleton()->viewport_remove_canvas(viewport, current_canvas);
	}

	if (p_world_2d.is_valid())
		world_2d = p_world_2d;
	else {
		WARN_PRINT("Invalid world");
		world_2d = Ref<World2D>(memnew(World2D));
	}

	_update_listener_2d();

	if (is_inside_tree()) {
		current_canvas = find_world_2d()->get_canvas();
		VisualServer::get_singleton()->viewport_attach_canvas(viewport, current_canvas);
		find_world_2d()->_register_viewport(this, Rect2());
	}
}

Ref<World2D> Viewport::find_world_2d() const {

	if (world_2d.is_valid())
		return world_2d;
	else if (parent)
		return parent->find_world_2d();
	else
		return Ref<World2D>();
}

void Viewport::_propagate_enter_world(Node *p_node) {

	if (p_node != this) {

		if (!p_node->is_inside_tree()) //may not have entered scene yet
			return;

		if (Object::cast_to<Spatial>(p_node) || Object::cast_to<WorldEnvironment>(p_node)) {

			p_node->notification(Spatial::NOTIFICATION_ENTER_WORLD);
		} else {
			Viewport *v = Object::cast_to<Viewport>(p_node);
			if (v) {

				if (v->world.is_valid())
					return;
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		_propagate_enter_world(p_node->get_child(i));
	}
}

void Viewport::_propagate_viewport_notification(Node *p_node, int p_what) {

	p_node->notification(p_what);
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *c = p_node->get_child(i);
		if (Object::cast_to<Viewport>(c))
			continue;
		_propagate_viewport_notification(c, p_what);
	}
}

void Viewport::_propagate_exit_world(Node *p_node) {

	if (p_node != this) {

		if (!p_node->is_inside_tree()) //may have exited scene already
			return;

		if (Object::cast_to<Spatial>(p_node) || Object::cast_to<WorldEnvironment>(p_node)) {

			p_node->notification(Spatial::NOTIFICATION_EXIT_WORLD);
		} else {
			Viewport *v = Object::cast_to<Viewport>(p_node);
			if (v) {

				if (v->world.is_valid())
					return;
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		_propagate_exit_world(p_node->get_child(i));
	}
}

void Viewport::set_world(const Ref<World> &p_world) {

	if (world == p_world)
		return;

	if (is_inside_tree())
		_propagate_exit_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_LOST_CURRENT);
#endif

	world = p_world;

	if (is_inside_tree())
		_propagate_enter_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_BECAME_CURRENT);
#endif

	//propagate exit

	if (is_inside_tree()) {
		VisualServer::get_singleton()->viewport_set_scenario(viewport, find_world()->get_scenario());
	}

	_update_listener();
}

Ref<World> Viewport::get_world() const {

	return world;
}

Ref<World2D> Viewport::get_world_2d() const {

	return world_2d;
}

Ref<World> Viewport::find_world() const {

	if (own_world.is_valid())
		return own_world;
	else if (world.is_valid())
		return world;
	else if (parent)
		return parent->find_world();
	else
		return Ref<World>();
}

Listener *Viewport::get_listener() const {

	return listener;
}

Camera *Viewport::get_camera() const {

	return camera;
}

Transform2D Viewport::get_final_transform() const {

	return stretch_transform * global_canvas_transform;
}

void Viewport::set_size_override(bool p_enable, const Size2 &p_size, const Vector2 &p_margin) {

	if (size_override == p_enable && p_size == size_override_size)
		return;

	size_override = p_enable;
	if (p_size.x >= 0 || p_size.y >= 0) {
		size_override_size = p_size;
	}
	size_override_margin = p_margin;
	_update_rect();
	_update_stretch_transform();
	emit_signal("size_changed");
}

Size2 Viewport::get_size_override() const {

	return size_override_size;
}
bool Viewport::is_size_override_enabled() const {

	return size_override;
}
void Viewport::set_size_override_stretch(bool p_enable) {

	if (p_enable == size_override_stretch)
		return;

	size_override_stretch = p_enable;
	if (size_override) {
		_update_rect();
	}

	_update_stretch_transform();
}

bool Viewport::is_size_override_stretch_enabled() const {

	return size_override_stretch;
}

void Viewport::set_update_mode(UpdateMode p_mode) {

	update_mode = p_mode;
	VS::get_singleton()->viewport_set_update_mode(viewport, VS::ViewportUpdateMode(p_mode));
}
Viewport::UpdateMode Viewport::get_update_mode() const {

	return update_mode;
}

Ref<ViewportTexture> Viewport::get_texture() const {

	return default_texture;
}

void Viewport::set_vflip(bool p_enable) {

	vflip = p_enable;
	VisualServer::get_singleton()->viewport_set_vflip(viewport, p_enable);
}

bool Viewport::get_vflip() const {

	return vflip;
}

void Viewport::set_clear_mode(ClearMode p_mode) {

	clear_mode = p_mode;
	VS::get_singleton()->viewport_set_clear_mode(viewport, VS::ViewportClearMode(p_mode));
}

Viewport::ClearMode Viewport::get_clear_mode() const {

	return clear_mode;
}

void Viewport::set_shadow_atlas_size(int p_size) {

	if (shadow_atlas_size == p_size)
		return;

	shadow_atlas_size = p_size;
	VS::get_singleton()->viewport_set_shadow_atlas_size(viewport, p_size);
}

int Viewport::get_shadow_atlas_size() const {

	return shadow_atlas_size;
}

void Viewport::set_shadow_atlas_quadrant_subdiv(int p_quadrant, ShadowAtlasQuadrantSubdiv p_subdiv) {

	ERR_FAIL_INDEX(p_quadrant, 4);
	ERR_FAIL_INDEX(p_subdiv, SHADOW_ATLAS_QUADRANT_SUBDIV_MAX);

	if (shadow_atlas_quadrant_subdiv[p_quadrant] == p_subdiv)
		return;

	shadow_atlas_quadrant_subdiv[p_quadrant] = p_subdiv;
	static const int subdiv[SHADOW_ATLAS_QUADRANT_SUBDIV_MAX] = { 0, 1, 4, 16, 64, 256, 1024 };

	VS::get_singleton()->viewport_set_shadow_atlas_quadrant_subdivision(viewport, p_quadrant, subdiv[p_subdiv]);
}
Viewport::ShadowAtlasQuadrantSubdiv Viewport::get_shadow_atlas_quadrant_subdiv(int p_quadrant) const {

	ERR_FAIL_INDEX_V(p_quadrant, 4, SHADOW_ATLAS_QUADRANT_SUBDIV_DISABLED);
	return shadow_atlas_quadrant_subdiv[p_quadrant];
}

Transform2D Viewport::_get_input_pre_xform() const {

	Transform2D pre_xf;

	if (to_screen_rect != Rect2()) {

		pre_xf.elements[2] = -to_screen_rect.position;
		pre_xf.scale(size / to_screen_rect.size);
	}

	return pre_xf;
}

Vector2 Viewport::_get_window_offset() const {

	/*
	if (parent_control) {
		return (parent_control->get_viewport()->get_final_transform() * parent_control->get_global_transform_with_canvas()).get_origin();
	}
	*/

	return Vector2();
}

Ref<InputEvent> Viewport::_make_input_local(const Ref<InputEvent> &ev) {

	Vector2 vp_ofs = _get_window_offset();
	Transform2D ai = get_final_transform().affine_inverse() * _get_input_pre_xform();

	return ev->xformed_by(ai, -vp_ofs);
}

void Viewport::_vp_input_text(const String &p_text) {

	if (gui.key_focus) {
		gui.key_focus->call("set_text", p_text);
	}
}

void Viewport::_vp_input(const Ref<InputEvent> &p_ev) {

	if (disable_input)
		return;

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
		return;
	}
#endif

	if (to_screen_rect == Rect2())
		return; //if render target, can't get input events

	//this one handles system input, p_ev are in system coordinates
	//they are converted to viewport coordinates

	Ref<InputEvent> ev = _make_input_local(p_ev);
	input(ev);
}

void Viewport::_vp_unhandled_input(const Ref<InputEvent> &p_ev) {

	if (disable_input)
		return;
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
		return;
	}
#endif

	/*
	if (parent_control && !parent_control->is_visible_in_tree())
		return;
	*/

	if (to_screen_rect == Rect2())
		return; //if render target, can't get input events

	//this one handles system input, p_ev are in system coordinates
	//they are converted to viewport coordinates

	Ref<InputEvent> ev = _make_input_local(p_ev);
	unhandled_input(ev);
}

Vector2 Viewport::get_mouse_position() const {

	return (get_final_transform().affine_inverse() * _get_input_pre_xform()).xform(Input::get_singleton()->get_mouse_position() - _get_window_offset());
}

void Viewport::warp_mouse(const Vector2 &p_pos) {

	Vector2 gpos = (get_final_transform().affine_inverse() * _get_input_pre_xform()).affine_inverse().xform(p_pos);
	Input::get_singleton()->warp_mouse_position(gpos);
}

void Viewport::_gui_sort_subwindows() {

	if (!gui.subwindow_order_dirty)
		return;

	gui.modal_stack.sort_custom<Control::CComparator>();
	gui.subwindows.sort_custom<Control::CComparator>();

	gui.subwindow_order_dirty = false;
}

void Viewport::_gui_sort_modal_stack() {

	gui.modal_stack.sort_custom<Control::CComparator>();
}

void Viewport::_gui_sort_roots() {

	if (!gui.roots_order_dirty)
		return;

	gui.roots.sort_custom<Control::CComparator>();

	gui.roots_order_dirty = false;
}

void Viewport::_gui_cancel_tooltip() {

	gui.tooltip = NULL;
	gui.tooltip_timer = -1;
	if (gui.tooltip_popup) {
		gui.tooltip_popup->queue_delete();
		gui.tooltip_popup = NULL;
	}
}

void Viewport::_gui_show_tooltip() {

	if (!gui.tooltip) {
		return;
	}

	String tooltip = gui.tooltip->get_tooltip(gui.tooltip->get_global_transform().xform_inv(gui.tooltip_pos));
	if (tooltip.length() == 0)
		return; // bye

	if (gui.tooltip_popup) {
		memdelete(gui.tooltip_popup);
		gui.tooltip_popup = NULL;
	}

	if (!gui.tooltip) {
		return;
	}

	Control *rp = gui.tooltip->get_root_parent_control();
	if (!rp)
		return;

	gui.tooltip_popup = memnew(TooltipPanel);

	rp->add_child(gui.tooltip_popup);
	gui.tooltip_popup->force_parent_owned();
	gui.tooltip_label = memnew(TooltipLabel);
	gui.tooltip_popup->add_child(gui.tooltip_label);
	gui.tooltip_popup->set_as_toplevel(true);
	gui.tooltip_popup->hide();

	Ref<StyleBox> ttp = gui.tooltip_label->get_stylebox("panel", "TooltipPanel");

	gui.tooltip_label->set_anchor_and_margin(MARGIN_LEFT, Control::ANCHOR_BEGIN, ttp->get_margin(MARGIN_LEFT));
	gui.tooltip_label->set_anchor_and_margin(MARGIN_TOP, Control::ANCHOR_BEGIN, ttp->get_margin(MARGIN_TOP));
	gui.tooltip_label->set_anchor_and_margin(MARGIN_RIGHT, Control::ANCHOR_END, -ttp->get_margin(MARGIN_RIGHT));
	gui.tooltip_label->set_anchor_and_margin(MARGIN_BOTTOM, Control::ANCHOR_END, -ttp->get_margin(MARGIN_BOTTOM));
	gui.tooltip_label->set_text(tooltip);
	Rect2 r(gui.tooltip_pos + Point2(10, 10), gui.tooltip_label->get_combined_minimum_size() + ttp->get_minimum_size());
	Rect2 vr = gui.tooltip_label->get_viewport_rect();
	if (r.size.x + r.position.x > vr.size.x)
		r.position.x = vr.size.x - r.size.x;
	else if (r.position.x < 0)
		r.position.x = 0;

	if (r.size.y + r.position.y > vr.size.y)
		r.position.y = vr.size.y - r.size.y;
	else if (r.position.y < 0)
		r.position.y = 0;

	gui.tooltip_popup->set_global_position(r.position);
	gui.tooltip_popup->set_size(r.size);

	gui.tooltip_popup->raise();
	gui.tooltip_popup->show();
}

void Viewport::_gui_call_input(Control *p_control, const Ref<InputEvent> &p_input) {

	//_block();

	Ref<InputEvent> ev = p_input;

	//mouse wheel events can't be stopped
	Ref<InputEventMouseButton> mb = p_input;

	bool cant_stop_me_now = (mb.is_valid() &&
							 (mb->get_button_index() == BUTTON_WHEEL_DOWN ||
									 mb->get_button_index() == BUTTON_WHEEL_UP ||
									 mb->get_button_index() == BUTTON_WHEEL_LEFT ||
									 mb->get_button_index() == BUTTON_WHEEL_RIGHT));

	bool ismouse = ev.is_valid() || Object::cast_to<InputEventMouseMotion>(*p_input) != NULL;

	CanvasItem *ci = p_control;
	while (ci) {

		Control *control = Object::cast_to<Control>(ci);
		if (control) {
			control->call_multilevel(SceneStringNames::get_singleton()->_gui_input, ev);
			if (gui.key_event_accepted)
				break;
			if (!control->is_inside_tree())
				break;
			control->emit_signal(SceneStringNames::get_singleton()->gui_input, ev);
			if (!control->is_inside_tree() || control->is_set_as_toplevel())
				break;
			if (gui.key_event_accepted)
				break;
			if (!cant_stop_me_now && control->data.mouse_filter == Control::MOUSE_FILTER_STOP && ismouse)
				break;
		}

		if (ci->is_set_as_toplevel())
			break;

		ev = ev->xformed_by(ci->get_transform()); //transform event upwards
		ci = ci->get_parent_item();
	}

	//_unblock();
}

Control *Viewport::_gui_find_control(const Point2 &p_global) {

	_gui_sort_subwindows();

	for (List<Control *>::Element *E = gui.subwindows.back(); E; E = E->prev()) {

		Control *sw = E->get();
		if (!sw->is_visible_in_tree())
			continue;

		Transform2D xform;
		CanvasItem *pci = sw->get_parent_item();
		if (pci)
			xform = pci->get_global_transform_with_canvas();
		else
			xform = sw->get_canvas_transform();

		Control *ret = _gui_find_control_at_pos(sw, p_global, xform, gui.focus_inv_xform);
		if (ret)
			return ret;
	}

	_gui_sort_roots();

	for (List<Control *>::Element *E = gui.roots.back(); E; E = E->prev()) {

		Control *sw = E->get();
		if (!sw->is_visible_in_tree())
			continue;

		Transform2D xform;
		CanvasItem *pci = sw->get_parent_item();
		if (pci)
			xform = pci->get_global_transform_with_canvas();
		else
			xform = sw->get_canvas_transform();

		Control *ret = _gui_find_control_at_pos(sw, p_global, xform, gui.focus_inv_xform);
		if (ret)
			return ret;
	}

	return NULL;
}

Control *Viewport::_gui_find_control_at_pos(CanvasItem *p_node, const Point2 &p_global, const Transform2D &p_xform, Transform2D &r_inv_xform) {

	if (Object::cast_to<Viewport>(p_node))
		return NULL;

	Control *c = Object::cast_to<Control>(p_node);

	if (c) {
		//print_line("at "+String(c->get_path())+" POS "+c->get_position()+" bt "+p_xform);
	}

	//subwindows first!!

	if (!p_node->is_visible()) {
		//return _find_next_visible_control_at_pos(p_node,p_global,r_inv_xform);
		return NULL; //canvas item hidden, discard
	}

	Transform2D matrix = p_xform * p_node->get_transform();
	// matrix.basis_determinant() == 0.0f implies that node does not exist on scene
	if (matrix.basis_determinant() == 0.0f)
		return NULL;

	if (!c || !c->clips_input() || c->has_point(matrix.affine_inverse().xform(p_global))) {

		for (int i = p_node->get_child_count() - 1; i >= 0; i--) {

			if (p_node == gui.tooltip_popup)
				continue;

			CanvasItem *ci = Object::cast_to<CanvasItem>(p_node->get_child(i));
			if (!ci || ci->is_set_as_toplevel())
				continue;

			Control *ret = _gui_find_control_at_pos(ci, p_global, matrix, r_inv_xform);
			if (ret)
				return ret;
		}
	}

	if (!c)
		return NULL;

	matrix.affine_invert();

	//conditions for considering this as a valid control for return
	if (c->data.mouse_filter != Control::MOUSE_FILTER_IGNORE && c->has_point(matrix.xform(p_global)) && (!gui.drag_preview || (c != gui.drag_preview && !gui.drag_preview->is_a_parent_of(c)))) {
		r_inv_xform = matrix;
		return c;
	} else
		return NULL;
}

bool Viewport::_gui_drop(Control *p_at_control, Point2 p_at_pos, bool p_just_check) {

	{ //attempt grab, try parent controls too
		CanvasItem *ci = p_at_control;
		while (ci) {

			Control *control = Object::cast_to<Control>(ci);
			if (control) {

				if (control->can_drop_data(p_at_pos, gui.drag_data)) {
					if (!p_just_check) {
						control->drop_data(p_at_pos, gui.drag_data);
					}

					return true;
				}

				if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP)
					break;
			}

			p_at_pos = ci->get_transform().xform(p_at_pos);

			if (ci->is_set_as_toplevel())
				break;

			ci = ci->get_parent_item();
		}
	}

	return false;
}

void Viewport::_gui_input_event(Ref<InputEvent> p_event) {

	if (p_event->get_id() == gui.cancelled_input_ID) {
		return;
	}
	//?
	/*
	if (!is_visible()) {
		return; //simple and plain
	}
	*/

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {

		gui.key_event_accepted = false;

		Point2 mpos = mb->get_position();
		if (mb->is_pressed()) {

			Size2 pos = mpos;
			if (gui.mouse_focus && mb->get_button_index() != gui.mouse_focus_button) {

				//do not steal mouse focus and stuff

			} else {

				_gui_sort_modal_stack();
				while (!gui.modal_stack.empty()) {

					Control *top = gui.modal_stack.back()->get();
					Vector2 pos = top->get_global_transform_with_canvas().affine_inverse().xform(mpos);
					if (!top->has_point(pos)) {

						if (top->data.modal_exclusive || top->data.modal_frame == Engine::get_singleton()->get_frames_drawn()) {
							//cancel event, sorry, modal exclusive EATS UP ALL
							//alternative, you can't pop out a window the same frame it was made modal (fixes many issues)
							get_tree()->set_input_as_handled();
							return; // no one gets the event if exclusive NO ONE
						}

						top->notification(Control::NOTIFICATION_MODAL_CLOSE);
						top->_modal_stack_remove();
						top->hide();
					} else {
						break;
					}
				}

				//Matrix32 parent_xform;

				/*
				if (data.parent_canvas_item)
					parent_xform=data.parent_canvas_item->get_global_transform();
				*/

				gui.mouse_focus = _gui_find_control(pos);
				//print_line("has mf "+itos(gui.mouse_focus!=NULL));
				gui.mouse_focus_button = mb->get_button_index();

				if (!gui.mouse_focus) {
					return;
				}

				if (mb->get_button_index() == BUTTON_LEFT) {
					gui.drag_accum = Vector2();
					gui.drag_attempted = false;
				}
			}

			mb = mb->xformed_by(Transform2D()); // make a copy of the event

			mb->set_global_position(pos);

			pos = gui.focus_inv_xform.xform(pos);

			mb->set_position(pos);

#ifdef DEBUG_ENABLED
			if (ScriptDebugger::get_singleton()) {

				Array arr;
				arr.push_back(gui.mouse_focus->get_path());
				arr.push_back(gui.mouse_focus->get_class());
				ScriptDebugger::get_singleton()->send_message("click_ctrl", arr);
			}

/*if (bool(GLOBAL_DEF("debug/print_clicked_control",false))) {

					print_line(String(gui.mouse_focus->get_path())+" - "+pos);
				}*/
#endif

			if (mb->get_button_index() == BUTTON_LEFT) { //assign focus
				CanvasItem *ci = gui.mouse_focus;
				while (ci) {

					Control *control = Object::cast_to<Control>(ci);
					if (control) {
						if (control->get_focus_mode() != Control::FOCUS_NONE) {
							if (control != gui.key_focus) {
								control->grab_focus();
							}
							break;
						}

						if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP)
							break;
					}

					if (ci->is_set_as_toplevel())
						break;

					ci = ci->get_parent_item();
				}
			}

			if (gui.mouse_focus->can_process()) {
				_gui_call_input(gui.mouse_focus, mb);
			}

			get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, "windows", "_cancel_input_ID", mb->get_id());
			get_tree()->set_input_as_handled();

			if (gui.drag_data.get_type() != Variant::NIL && mb->get_button_index() == BUTTON_LEFT) {

				//alternate drop use (when using force_drag(), as proposed by #5342
				if (gui.mouse_focus) {
					_gui_drop(gui.mouse_focus, pos, false);
				}

				gui.drag_data = Variant();

				if (gui.drag_preview) {
					memdelete(gui.drag_preview);
					gui.drag_preview = NULL;
				}
				_propagate_viewport_notification(this, NOTIFICATION_DRAG_END);
				//change mouse accordingly
			}

			_gui_cancel_tooltip();
			//gui.tooltip_popup->hide();

		} else {

			if (gui.drag_data.get_type() != Variant::NIL && mb->get_button_index() == BUTTON_LEFT) {

				if (gui.mouse_over) {
					Size2 pos = mpos;
					pos = gui.focus_inv_xform.xform(pos);

					_gui_drop(gui.mouse_over, pos, false);
				}

				if (gui.drag_preview && mb->get_button_index() == BUTTON_LEFT) {
					memdelete(gui.drag_preview);
					gui.drag_preview = NULL;
				}

				gui.drag_data = Variant();
				_propagate_viewport_notification(this, NOTIFICATION_DRAG_END);
				//change mouse accordingly
			}

			if (!gui.mouse_focus) {
				//release event is only sent if a mouse focus (previously pressed button) exists
				return;
			}

			Size2 pos = mpos;

			mb = mb->xformed_by(Transform2D()); //make a copy
			mb->set_global_position(pos);
			pos = gui.focus_inv_xform.xform(pos);
			mb->set_position(pos);

			if (gui.mouse_focus->can_process()) {
				_gui_call_input(gui.mouse_focus, mb);
			}

			if (mb->get_button_index() == gui.mouse_focus_button) {
				gui.mouse_focus = NULL;
				gui.mouse_focus_button = -1;
			}

			/*if (gui.drag_data.get_type()!=Variant::NIL && mb->get_button_index()==BUTTON_LEFT) {
				_propagate_viewport_notification(this,NOTIFICATION_DRAG_END);
				gui.drag_data=Variant(); //always clear
			}*/

			get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, "windows", "_cancel_input_ID", mb->get_id());
			get_tree()->set_input_as_handled();
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {

		gui.key_event_accepted = false;
		Point2 mpos = mm->get_position();

		gui.last_mouse_pos = mpos;

		Control *over = NULL;

		// D&D
		if (!gui.drag_attempted && gui.mouse_focus && mm->get_button_mask() & BUTTON_MASK_LEFT) {

			gui.drag_accum += mm->get_relative();
			float len = gui.drag_accum.length();
			if (len > 10) {

				{ //attempt grab, try parent controls too
					CanvasItem *ci = gui.mouse_focus;
					while (ci) {

						Control *control = Object::cast_to<Control>(ci);
						if (control) {

							gui.drag_data = control->get_drag_data(control->get_global_transform_with_canvas().affine_inverse().xform(mpos) - gui.drag_accum);
							if (gui.drag_data.get_type() != Variant::NIL) {

								gui.mouse_focus = NULL;
							}

							if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP)
								break;
						}

						if (ci->is_set_as_toplevel())
							break;

						ci = ci->get_parent_item();
					}
				}

				gui.drag_attempted = true;
				if (gui.drag_data.get_type() != Variant::NIL) {

					_propagate_viewport_notification(this, NOTIFICATION_DRAG_BEGIN);
				}
			}
		}

		if (gui.mouse_focus) {
			over = gui.mouse_focus;
			//recompute focus_inv_xform again here

		} else {

			over = _gui_find_control(mpos);
		}

		if (gui.drag_data.get_type() == Variant::NIL && over && !gui.modal_stack.empty()) {

			Control *top = gui.modal_stack.back()->get();
			if (over != top && !top->is_a_parent_of(over)) {
				over = NULL; //nothing can be found outside the modal stack
			}
		}

		if (over != gui.mouse_over) {

			if (gui.mouse_over)
				gui.mouse_over->notification(Control::NOTIFICATION_MOUSE_EXIT);

			_gui_cancel_tooltip();

			if (over)
				over->notification(Control::NOTIFICATION_MOUSE_ENTER);
		}

		gui.mouse_over = over;

		if (gui.drag_preview) {
			gui.drag_preview->set_position(mpos);
		}

		if (!over) {
			OS::get_singleton()->set_cursor_shape(OS::CURSOR_ARROW);
			return;
		}

		Transform2D localizer = over->get_global_transform_with_canvas().affine_inverse();
		Size2 pos = localizer.xform(mpos);
		Vector2 speed = localizer.basis_xform(mm->get_speed());
		Vector2 rel = localizer.basis_xform(mm->get_relative());

		mm = mm->xformed_by(Transform2D()); //make a copy

		mm->set_global_position(mpos);
		mm->set_speed(speed);
		mm->set_relative(rel);

		if (mm->get_button_mask() == 0) {
			//nothing pressed

			bool can_tooltip = true;

			if (!gui.modal_stack.empty()) {
				if (gui.modal_stack.back()->get() != over && !gui.modal_stack.back()->get()->is_a_parent_of(over))
					can_tooltip = false;
			}

			bool is_tooltip_shown = false;

			if (gui.tooltip_popup) {
				if (can_tooltip) {
					String tooltip = over->get_tooltip(gui.tooltip->get_global_transform().xform_inv(mpos));

					if (tooltip.length() == 0)
						_gui_cancel_tooltip();
					else if (tooltip == gui.tooltip_label->get_text())
						is_tooltip_shown = true;
				} else
					_gui_cancel_tooltip();
			}

			if (can_tooltip && !is_tooltip_shown) {

				gui.tooltip = over;
				gui.tooltip_pos = mpos; //(parent_xform * get_transform()).affine_inverse().xform(pos);
				gui.tooltip_timer = gui.tooltip_delay;
			}
		}

		//pos = gui.focus_inv_xform.xform(pos);

		mm->set_position(pos);

		Control::CursorShape cursor_shape = over->get_cursor_shape(pos);
		OS::get_singleton()->set_cursor_shape((OS::CursorShape)cursor_shape);

		if (over->can_process()) {
			_gui_call_input(over, mm);
		}

		get_tree()->set_input_as_handled();

		if (gui.drag_data.get_type() != Variant::NIL && mm->get_button_mask() & BUTTON_MASK_LEFT) {

			bool can_drop = _gui_drop(over, pos, true);

			if (!can_drop) {
				OS::get_singleton()->set_cursor_shape(OS::CURSOR_FORBIDDEN);
			} else {
				OS::get_singleton()->set_cursor_shape(OS::CURSOR_CAN_DROP);
			}
			//change mouse accordingly i guess
		}
	}

	Ref<InputEventScreenTouch> touch_event = p_event;
	if (touch_event.is_valid()) {

		Size2 pos = touch_event->get_position();
		if (touch_event->is_pressed()) {

			Control *over = _gui_find_control(pos);
			if (over) {

				if (!gui.modal_stack.empty()) {

					Control *top = gui.modal_stack.back()->get();
					if (over != top && !top->is_a_parent_of(over)) {

						return;
					}
				}
				if (over->can_process()) {

					touch_event = touch_event->xformed_by(Transform2D()); //make a copy
					if (over == gui.mouse_focus) {
						pos = gui.focus_inv_xform.xform(pos);
					} else {
						pos = over->get_global_transform_with_canvas().affine_inverse().xform(pos);
					}
					touch_event->set_position(pos);
					_gui_call_input(over, touch_event);
				}
				get_tree()->set_input_as_handled();
				return;
			}
		} else if (gui.mouse_focus) {

			if (gui.mouse_focus->can_process()) {

				touch_event = touch_event->xformed_by(Transform2D()); //make a copy
				touch_event->set_position(gui.focus_inv_xform.xform(pos));

				_gui_call_input(gui.mouse_focus, touch_event);
			}
			get_tree()->set_input_as_handled();
			return;
		}
	}

	Ref<InputEventScreenDrag> drag_event = p_event;
	if (drag_event.is_valid()) {

		Control *over = gui.mouse_focus;
		if (!over) {
			over = _gui_find_control(drag_event->get_position());
		}
		if (over) {

			if (!gui.modal_stack.empty()) {

				Control *top = gui.modal_stack.back()->get();
				if (over != top && !top->is_a_parent_of(over)) {

					return;
				}
			}
			if (over->can_process()) {

				Transform2D localizer = over->get_global_transform_with_canvas().affine_inverse();
				Size2 pos = localizer.xform(drag_event->get_position());
				Vector2 speed = localizer.basis_xform(drag_event->get_speed());
				Vector2 rel = localizer.basis_xform(drag_event->get_relative());

				drag_event = drag_event->xformed_by(Transform2D()); //make a copy

				drag_event->set_speed(speed);
				drag_event->set_relative(rel);
				drag_event->set_position(pos);

				_gui_call_input(over, drag_event);
			}

			get_tree()->set_input_as_handled();
			return;
		}
	}

	if (mm.is_null() && mb.is_null() && p_event->is_action_type()) {

		if (gui.key_focus && !gui.key_focus->is_visible_in_tree()) {
			gui.key_focus->release_focus();
		}

		if (gui.key_focus) {

			gui.key_event_accepted = false;
			if (gui.key_focus->can_process()) {
				gui.key_focus->call_multilevel(SceneStringNames::get_singleton()->_gui_input, p_event);
				if (gui.key_focus) //maybe lost it
					gui.key_focus->emit_signal(SceneStringNames::get_singleton()->gui_input, p_event);
			}

			if (gui.key_event_accepted) {

				get_tree()->set_input_as_handled();
				return;
			}
		}

		if (p_event->is_pressed() && p_event->is_action("ui_cancel") && !gui.modal_stack.empty()) {

			_gui_sort_modal_stack();
			Control *top = gui.modal_stack.back()->get();
			if (!top->data.modal_exclusive) {

				top->notification(Control::NOTIFICATION_MODAL_CLOSE);
				top->_modal_stack_remove();
				top->hide();
			}
		}

		Control *from = gui.key_focus ? gui.key_focus : NULL; //hmm

		//keyboard focus
		//if (from && p_event->is_pressed() && !p_event->get_alt() && !p_event->get_metakey() && !p_event->key->get_command()) {

		Ref<InputEventKey> k = p_event;
		//need to check for mods, otherwise any combination of alt/ctrl/shift+<up/down/left/righ/etc> is handled here when it shouldn't be.
		bool mods = k.is_valid() && (k->get_control() || k->get_alt() || k->get_shift() || k->get_metakey());

		if (from && p_event->is_pressed()) {
			Control *next = NULL;

			if (p_event->is_action("ui_focus_next")) {

				next = from->find_next_valid_focus();
			}

			if (p_event->is_action("ui_focus_prev")) {

				next = from->find_prev_valid_focus();
			}

			if (!mods && p_event->is_action("ui_up")) {

				next = from->_get_focus_neighbour(MARGIN_TOP);
			}

			if (!mods && p_event->is_action("ui_left")) {

				next = from->_get_focus_neighbour(MARGIN_LEFT);
			}

			if (!mods && p_event->is_action("ui_right")) {

				next = from->_get_focus_neighbour(MARGIN_RIGHT);
			}

			if (!mods && p_event->is_action("ui_down")) {

				next = from->_get_focus_neighbour(MARGIN_BOTTOM);
			}

			if (next) {
				next->grab_focus();
				get_tree()->set_input_as_handled();
			}
		}
	}
}

List<Control *>::Element *Viewport::_gui_add_root_control(Control *p_control) {

	gui.roots_order_dirty = true;
	return gui.roots.push_back(p_control);
}

List<Control *>::Element *Viewport::_gui_add_subwindow_control(Control *p_control) {

	gui.subwindow_order_dirty = true;
	return gui.subwindows.push_back(p_control);
}

void Viewport::_gui_set_subwindow_order_dirty() {
	gui.subwindow_order_dirty = true;
}

void Viewport::_gui_set_root_order_dirty() {
	gui.roots_order_dirty = true;
}

void Viewport::_gui_remove_modal_control(List<Control *>::Element *MI) {

	gui.modal_stack.erase(MI);
}

void Viewport::_gui_remove_from_modal_stack(List<Control *>::Element *MI, ObjectID p_prev_focus_owner) {

	//transfer the focus stack to the next

	List<Control *>::Element *next = MI->next();

	gui.modal_stack.erase(MI);
	MI = NULL;

	if (p_prev_focus_owner) {

		// for previous window in stack, pass the focus so it feels more
		// natural

		if (!next) { //top of stack

			Object *pfo = ObjectDB::get_instance(p_prev_focus_owner);
			Control *pfoc = Object::cast_to<Control>(pfo);
			if (!pfoc)
				return;

			if (!pfoc->is_inside_tree() || !pfoc->is_visible_in_tree())
				return;
			pfoc->grab_focus();
		} else {

			next->get()->_modal_set_prev_focus_owner(p_prev_focus_owner);
		}
	}
}

void Viewport::_gui_force_drag(Control *p_base, const Variant &p_data, Control *p_control) {

	ERR_EXPLAIN("Drag data must be a value");
	ERR_FAIL_COND(p_data.get_type() == Variant::NIL);

	gui.drag_data = p_data;
	gui.mouse_focus = NULL;

	if (p_control) {
		_gui_set_drag_preview(p_base, p_control);
	}
}

void Viewport::_gui_set_drag_preview(Control *p_base, Control *p_control) {

	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND(!Object::cast_to<Control>((Object *)p_control));
	ERR_FAIL_COND(p_control->is_inside_tree());
	ERR_FAIL_COND(p_control->get_parent() != NULL);

	if (gui.drag_preview) {
		memdelete(gui.drag_preview);
	}
	p_control->set_as_toplevel(true);
	p_control->set_position(gui.last_mouse_pos);
	p_base->get_root_parent_control()->add_child(p_control); //add as child of viewport
	p_control->raise();
	if (gui.drag_preview) {
		memdelete(gui.drag_preview);
	}
	gui.drag_preview = p_control;
}

void Viewport::_gui_remove_root_control(List<Control *>::Element *RI) {

	gui.roots.erase(RI);
}

void Viewport::_gui_remove_subwindow_control(List<Control *>::Element *SI) {

	gui.subwindows.erase(SI);
}

void Viewport::_gui_unfocus_control(Control *p_control) {

	if (gui.key_focus == p_control) {
		gui.key_focus->release_focus();
	}
}

void Viewport::_gui_hid_control(Control *p_control) {

	if (gui.mouse_focus == p_control) {
		gui.mouse_focus = NULL;
	}

	/* ???
	if (data.window==p_control) {
		window->drag_data=Variant();
		if (window->drag_preview) {
			memdelete( window->drag_preview);
			window->drag_preview=NULL;
		}
	}
	*/

	if (gui.key_focus == p_control)
		gui.key_focus = NULL;
	if (gui.mouse_over == p_control)
		gui.mouse_over = NULL;
	if (gui.tooltip == p_control)
		gui.tooltip = NULL;
	if (gui.tooltip == p_control) {
		gui.tooltip = NULL;
		_gui_cancel_tooltip();
	}
}

void Viewport::_gui_remove_control(Control *p_control) {

	if (gui.mouse_focus == p_control)
		gui.mouse_focus = NULL;
	if (gui.key_focus == p_control)
		gui.key_focus = NULL;
	if (gui.mouse_over == p_control)
		gui.mouse_over = NULL;
	if (gui.tooltip == p_control)
		gui.tooltip = NULL;
	if (gui.tooltip_popup == p_control) {
		_gui_cancel_tooltip();
	}
}

void Viewport::_gui_remove_focus() {

	if (gui.key_focus) {
		Node *f = gui.key_focus;
		gui.key_focus = NULL;
		f->notification(Control::NOTIFICATION_FOCUS_EXIT, true);
	}
}

bool Viewport::_gui_is_modal_on_top(const Control *p_control) {

	return (gui.modal_stack.size() && gui.modal_stack.back()->get() == p_control);
}

bool Viewport::_gui_control_has_focus(const Control *p_control) {

	return gui.key_focus == p_control;
}

void Viewport::_gui_control_grab_focus(Control *p_control) {

	//no need for change
	if (gui.key_focus && gui.key_focus == p_control)
		return;

	get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, "_viewports", "_gui_remove_focus");
	gui.key_focus = p_control;
	p_control->notification(Control::NOTIFICATION_FOCUS_ENTER);
	p_control->update();
}

void Viewport::_gui_accept_event() {

	gui.key_event_accepted = true;
	if (is_inside_tree())
		get_tree()->set_input_as_handled();
}

List<Control *>::Element *Viewport::_gui_show_modal(Control *p_control) {

	gui.modal_stack.push_back(p_control);
	if (gui.key_focus)
		p_control->_modal_set_prev_focus_owner(gui.key_focus->get_instance_id());
	else
		p_control->_modal_set_prev_focus_owner(0);

	return gui.modal_stack.back();
}

Control *Viewport::_gui_get_focus_owner() {

	return gui.key_focus;
}

void Viewport::_gui_grab_click_focus(Control *p_control) {

	if (gui.mouse_focus) {

		if (gui.mouse_focus == p_control)
			return;
		Ref<InputEventMouseButton> mb;
		mb.instance();

		//send unclic

		Point2 click = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);
		mb->set_position(click);
		mb->set_button_index(gui.mouse_focus_button);
		mb->set_pressed(false);
		gui.mouse_focus->call_deferred(SceneStringNames::get_singleton()->_gui_input, mb);

		gui.mouse_focus = p_control;
		gui.focus_inv_xform = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse();
		click = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);
		mb->set_position(click);
		mb->set_button_index(gui.mouse_focus_button);
		mb->set_pressed(true);
		gui.mouse_focus->call_deferred(SceneStringNames::get_singleton()->_gui_input, mb);
	}
}

///////////////////////////////

void Viewport::input(const Ref<InputEvent> &p_event) {

	ERR_FAIL_COND(!is_inside_tree());

	if (!get_tree()->is_input_handled()) {
		get_tree()->_call_input_pause(input_group, "_input", p_event); //not a bug, must happen before GUI, order is _input -> gui input -> _unhandled input
	}

	if (!get_tree()->is_input_handled()) {
		_gui_input_event(p_event);
	}
	//get_tree()->call_group(SceneTree::GROUP_CALL_REVERSE|SceneTree::GROUP_CALL_REALTIME|SceneTree::GROUP_CALL_MULIILEVEL,gui_input_group,"_gui_input",p_event); //special one for GUI, as controls use their own process check
}

void Viewport::unhandled_input(const Ref<InputEvent> &p_event) {

	ERR_FAIL_COND(!is_inside_tree());

	get_tree()->_call_input_pause(unhandled_input_group, "_unhandled_input", p_event);
	//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"unhandled_input","_unhandled_input",ev);
	if (!get_tree()->input_handled && Object::cast_to<InputEventKey>(*p_event) != NULL) {
		get_tree()->_call_input_pause(unhandled_key_input_group, "_unhandled_key_input", p_event);
		//call_group(GROUP_CALL_REVERSE|GROUP_CALL_REALTIME|GROUP_CALL_MULIILEVEL,"unhandled_key_input","_unhandled_key_input",ev);
	}

	if (physics_object_picking && !get_tree()->input_handled) {

		if (Input::get_singleton()->get_mouse_mode() != Input::MOUSE_MODE_CAPTURED &&
				(Object::cast_to<InputEventMouseButton>(*p_event) ||
						Object::cast_to<InputEventMouseMotion>(*p_event) ||
						Object::cast_to<InputEventScreenDrag>(*p_event) ||
						Object::cast_to<InputEventScreenTouch>(*p_event))) {
			physics_picking_events.push_back(p_event);
		}
	}
}

void Viewport::set_use_own_world(bool p_world) {

	if (p_world == own_world.is_valid())
		return;

	if (is_inside_tree())
		_propagate_exit_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_LOST_CURRENT);
#endif

	if (!p_world)
		own_world = Ref<World>();
	else
		own_world = Ref<World>(memnew(World));

	if (is_inside_tree())
		_propagate_enter_world(this);

#ifndef _3D_DISABLED
	if (find_world().is_valid() && camera)
		camera->notification(Camera::NOTIFICATION_BECAME_CURRENT);
#endif

	//propagate exit

	if (is_inside_tree()) {
		VisualServer::get_singleton()->viewport_set_scenario(viewport, find_world()->get_scenario());
	}

	_update_listener();
}

bool Viewport::is_using_own_world() const {

	return own_world.is_valid();
}

void Viewport::set_attach_to_screen_rect(const Rect2 &p_rect) {

	VS::get_singleton()->viewport_attach_to_screen(viewport, p_rect);
	to_screen_rect = p_rect;
}

Rect2 Viewport::get_attach_to_screen_rect() const {

	return to_screen_rect;
}

void Viewport::set_physics_object_picking(bool p_enable) {

	physics_object_picking = p_enable;
	set_physics_process(physics_object_picking);
	if (!physics_object_picking)
		physics_picking_events.clear();
}

Vector2 Viewport::get_camera_coords(const Vector2 &p_viewport_coords) const {

	Transform2D xf = get_final_transform();
	return xf.xform(p_viewport_coords);
}

Vector2 Viewport::get_camera_rect_size() const {

	return size;
}

bool Viewport::get_physics_object_picking() {

	return physics_object_picking;
}

bool Viewport::gui_has_modal_stack() const {

	return gui.modal_stack.size();
}

void Viewport::set_disable_input(bool p_disable) {
	disable_input = p_disable;
}

bool Viewport::is_input_disabled() const {

	return disable_input;
}

void Viewport::set_disable_3d(bool p_disable) {
	disable_3d = p_disable;
	VS::get_singleton()->viewport_set_disable_3d(viewport, p_disable);
}

bool Viewport::is_3d_disabled() const {

	return disable_3d;
}

Variant Viewport::gui_get_drag_data() const {
	return gui.drag_data;
}

Control *Viewport::get_modal_stack_top() const {
	return gui.modal_stack.size() ? gui.modal_stack.back()->get() : NULL;
}

String Viewport::get_configuration_warning() const {

	/*if (get_parent() && !Object::cast_to<Control>(get_parent()) && !render_target) {

		return TTR("This viewport is not set as render target. If you intend for it to display its contents directly to the screen, make it a child of a Control so it can obtain a size. Otherwise, make it a RenderTarget and assign its internal texture to some node for display.");
	}*/

	return String();
}

void Viewport::gui_reset_canvas_sort_index() {
	gui.canvas_sort_index = 0;
}
int Viewport::gui_get_canvas_sort_index() {

	return gui.canvas_sort_index++;
}

void Viewport::set_msaa(MSAA p_msaa) {

	ERR_FAIL_INDEX(p_msaa, 5);
	if (msaa == p_msaa)
		return;
	msaa = p_msaa;
	VS::get_singleton()->viewport_set_msaa(viewport, VS::ViewportMSAA(p_msaa));
}

Viewport::MSAA Viewport::get_msaa() const {

	return msaa;
}

void Viewport::set_hdr(bool p_hdr) {

	if (hdr == p_hdr)
		return;

	hdr = p_hdr;
	VS::get_singleton()->viewport_set_hdr(viewport, p_hdr);
}

bool Viewport::get_hdr() const {

	return hdr;
}

void Viewport::set_usage(Usage p_usage) {

	usage = p_usage;
	VS::get_singleton()->viewport_set_usage(viewport, VS::ViewportUsage(p_usage));
}

Viewport::Usage Viewport::get_usage() const {
	return usage;
}

void Viewport::set_debug_draw(DebugDraw p_debug_draw) {

	debug_draw = p_debug_draw;
	VS::get_singleton()->viewport_set_debug_draw(viewport, VS::ViewportDebugDraw(p_debug_draw));
}

Viewport::DebugDraw Viewport::get_debug_draw() const {

	return debug_draw;
}

int Viewport::get_render_info(RenderInfo p_info) {

	return VS::get_singleton()->viewport_get_render_info(viewport, VS::ViewportRenderInfo(p_info));
}

void Viewport::set_snap_controls_to_pixels(bool p_enable) {

	snap_controls_to_pixels = p_enable;
}

bool Viewport::is_snap_controls_to_pixels_enabled() const {

	return snap_controls_to_pixels;
}

void Viewport::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_use_arvr", "use"), &Viewport::set_use_arvr);
	ClassDB::bind_method(D_METHOD("use_arvr"), &Viewport::use_arvr);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &Viewport::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &Viewport::get_size);
	ClassDB::bind_method(D_METHOD("set_world_2d", "world_2d"), &Viewport::set_world_2d);
	ClassDB::bind_method(D_METHOD("get_world_2d"), &Viewport::get_world_2d);
	ClassDB::bind_method(D_METHOD("find_world_2d"), &Viewport::find_world_2d);
	ClassDB::bind_method(D_METHOD("set_world", "world"), &Viewport::set_world);
	ClassDB::bind_method(D_METHOD("get_world"), &Viewport::get_world);
	ClassDB::bind_method(D_METHOD("find_world"), &Viewport::find_world);

	ClassDB::bind_method(D_METHOD("set_canvas_transform", "xform"), &Viewport::set_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_canvas_transform"), &Viewport::get_canvas_transform);

	ClassDB::bind_method(D_METHOD("set_global_canvas_transform", "xform"), &Viewport::set_global_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_global_canvas_transform"), &Viewport::get_global_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_final_transform"), &Viewport::get_final_transform);

	ClassDB::bind_method(D_METHOD("get_visible_rect"), &Viewport::get_visible_rect);
	ClassDB::bind_method(D_METHOD("set_transparent_background", "enable"), &Viewport::set_transparent_background);
	ClassDB::bind_method(D_METHOD("has_transparent_background"), &Viewport::has_transparent_background);

	ClassDB::bind_method(D_METHOD("_parent_visibility_changed"), &Viewport::_parent_visibility_changed);

	ClassDB::bind_method(D_METHOD("_parent_resized"), &Viewport::_parent_resized);
	ClassDB::bind_method(D_METHOD("_vp_input"), &Viewport::_vp_input);
	ClassDB::bind_method(D_METHOD("_vp_input_text", "text"), &Viewport::_vp_input_text);
	ClassDB::bind_method(D_METHOD("_vp_unhandled_input"), &Viewport::_vp_unhandled_input);

	ClassDB::bind_method(D_METHOD("set_size_override", "enable", "size", "margin"), &Viewport::set_size_override, DEFVAL(Size2(-1, -1)), DEFVAL(Size2(0, 0)));
	ClassDB::bind_method(D_METHOD("get_size_override"), &Viewport::get_size_override);
	ClassDB::bind_method(D_METHOD("is_size_override_enabled"), &Viewport::is_size_override_enabled);
	ClassDB::bind_method(D_METHOD("set_size_override_stretch", "enabled"), &Viewport::set_size_override_stretch);
	ClassDB::bind_method(D_METHOD("is_size_override_stretch_enabled"), &Viewport::is_size_override_stretch_enabled);

	ClassDB::bind_method(D_METHOD("set_vflip", "enable"), &Viewport::set_vflip);
	ClassDB::bind_method(D_METHOD("get_vflip"), &Viewport::get_vflip);

	ClassDB::bind_method(D_METHOD("set_clear_mode", "mode"), &Viewport::set_clear_mode);
	ClassDB::bind_method(D_METHOD("get_clear_mode"), &Viewport::get_clear_mode);

	ClassDB::bind_method(D_METHOD("set_update_mode", "mode"), &Viewport::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &Viewport::get_update_mode);

	ClassDB::bind_method(D_METHOD("set_msaa", "msaa"), &Viewport::set_msaa);
	ClassDB::bind_method(D_METHOD("get_msaa"), &Viewport::get_msaa);

	ClassDB::bind_method(D_METHOD("set_hdr", "enable"), &Viewport::set_hdr);
	ClassDB::bind_method(D_METHOD("get_hdr"), &Viewport::get_hdr);

	ClassDB::bind_method(D_METHOD("set_usage", "usage"), &Viewport::set_usage);
	ClassDB::bind_method(D_METHOD("get_usage"), &Viewport::get_usage);

	ClassDB::bind_method(D_METHOD("set_debug_draw", "debug_draw"), &Viewport::set_debug_draw);
	ClassDB::bind_method(D_METHOD("get_debug_draw"), &Viewport::get_debug_draw);

	ClassDB::bind_method(D_METHOD("get_render_info", "info"), &Viewport::get_render_info);

	ClassDB::bind_method(D_METHOD("get_texture"), &Viewport::get_texture);

	ClassDB::bind_method(D_METHOD("set_physics_object_picking", "enable"), &Viewport::set_physics_object_picking);
	ClassDB::bind_method(D_METHOD("get_physics_object_picking"), &Viewport::get_physics_object_picking);

	ClassDB::bind_method(D_METHOD("get_viewport_rid"), &Viewport::get_viewport_rid);
	ClassDB::bind_method(D_METHOD("input", "local_event"), &Viewport::input);
	ClassDB::bind_method(D_METHOD("unhandled_input", "local_event"), &Viewport::unhandled_input);

	ClassDB::bind_method(D_METHOD("update_worlds"), &Viewport::update_worlds);

	ClassDB::bind_method(D_METHOD("set_use_own_world", "enable"), &Viewport::set_use_own_world);
	ClassDB::bind_method(D_METHOD("is_using_own_world"), &Viewport::is_using_own_world);

	ClassDB::bind_method(D_METHOD("get_camera"), &Viewport::get_camera);

	ClassDB::bind_method(D_METHOD("set_as_audio_listener", "enable"), &Viewport::set_as_audio_listener);
	ClassDB::bind_method(D_METHOD("is_audio_listener"), &Viewport::is_audio_listener);

	ClassDB::bind_method(D_METHOD("set_as_audio_listener_2d", "enable"), &Viewport::set_as_audio_listener_2d);
	ClassDB::bind_method(D_METHOD("is_audio_listener_2d"), &Viewport::is_audio_listener_2d);
	ClassDB::bind_method(D_METHOD("set_attach_to_screen_rect", "rect"), &Viewport::set_attach_to_screen_rect);

	ClassDB::bind_method(D_METHOD("get_mouse_position"), &Viewport::get_mouse_position);
	ClassDB::bind_method(D_METHOD("warp_mouse", "to_position"), &Viewport::warp_mouse);

	ClassDB::bind_method(D_METHOD("gui_has_modal_stack"), &Viewport::gui_has_modal_stack);
	ClassDB::bind_method(D_METHOD("gui_get_drag_data"), &Viewport::gui_get_drag_data);

	ClassDB::bind_method(D_METHOD("set_disable_input", "disable"), &Viewport::set_disable_input);
	ClassDB::bind_method(D_METHOD("is_input_disabled"), &Viewport::is_input_disabled);

	ClassDB::bind_method(D_METHOD("set_disable_3d", "disable"), &Viewport::set_disable_3d);
	ClassDB::bind_method(D_METHOD("is_3d_disabled"), &Viewport::is_3d_disabled);

	ClassDB::bind_method(D_METHOD("_gui_show_tooltip"), &Viewport::_gui_show_tooltip);
	ClassDB::bind_method(D_METHOD("_gui_remove_focus"), &Viewport::_gui_remove_focus);

	ClassDB::bind_method(D_METHOD("set_shadow_atlas_size", "size"), &Viewport::set_shadow_atlas_size);
	ClassDB::bind_method(D_METHOD("get_shadow_atlas_size"), &Viewport::get_shadow_atlas_size);

	ClassDB::bind_method(D_METHOD("set_snap_controls_to_pixels", "enabled"), &Viewport::set_snap_controls_to_pixels);
	ClassDB::bind_method(D_METHOD("is_snap_controls_to_pixels_enabled"), &Viewport::is_snap_controls_to_pixels_enabled);

	ClassDB::bind_method(D_METHOD("set_shadow_atlas_quadrant_subdiv", "quadrant", "subdiv"), &Viewport::set_shadow_atlas_quadrant_subdiv);
	ClassDB::bind_method(D_METHOD("get_shadow_atlas_quadrant_subdiv", "quadrant"), &Viewport::get_shadow_atlas_quadrant_subdiv);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "arvr"), "set_use_arvr", "use_arvr");

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "own_world"), "set_use_own_world", "is_using_own_world");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "world", PROPERTY_HINT_RESOURCE_TYPE, "World"), "set_world", "get_world");
	//ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"world_2d",PROPERTY_HINT_RESOURCE_TYPE,"World2D"), "set_world_2d", "get_world_2d") ;
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transparent_bg"), "set_transparent_background", "has_transparent_background");
	ADD_GROUP("Rendering", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "msaa", PROPERTY_HINT_ENUM, "Disabled,2x,4x,8x,16x"), "set_msaa", "get_msaa");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hdr"), "set_hdr", "get_hdr");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_3d"), "set_disable_3d", "is_3d_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "usage", PROPERTY_HINT_ENUM, "2D,2D No-Sampling,3D,3D No-Effects"), "set_usage", "get_usage");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_draw", PROPERTY_HINT_ENUM, "Disabled,Unshaded,Overdraw,Wireframe"), "set_debug_draw", "get_debug_draw");
	ADD_GROUP("Render Target", "render_target_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "render_target_v_flip"), "set_vflip", "get_vflip");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_target_clear_mode", PROPERTY_HINT_ENUM, "Always,Never,Next Frame"), "set_clear_mode", "get_clear_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_target_update_mode", PROPERTY_HINT_ENUM, "Disabled,Once,When Visible,Always"), "set_update_mode", "get_update_mode");
	ADD_GROUP("Audio Listener", "audio_listener_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_listener_enable_2d"), "set_as_audio_listener_2d", "is_audio_listener_2d");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_listener_enable_3d"), "set_as_audio_listener", "is_audio_listener");
	ADD_GROUP("Physics", "physics_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "physics_object_picking"), "set_physics_object_picking", "get_physics_object_picking");
	ADD_GROUP("GUI", "gui_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_disable_input"), "set_disable_input", "is_input_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_snap_controls_to_pixels"), "set_snap_controls_to_pixels", "is_snap_controls_to_pixels_enabled");
	ADD_GROUP("Shadow Atlas", "shadow_atlas_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_atlas_size"), "set_shadow_atlas_size", "get_shadow_atlas_size");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_0", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 0);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_1", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_2", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 2);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_3", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 3);

	ADD_SIGNAL(MethodInfo("size_changed"));

	BIND_ENUM_CONSTANT(UPDATE_DISABLED);
	BIND_ENUM_CONSTANT(UPDATE_ONCE);
	BIND_ENUM_CONSTANT(UPDATE_WHEN_VISIBLE);
	BIND_ENUM_CONSTANT(UPDATE_ALWAYS);

	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_DISABLED);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_1);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_4);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_16);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_64);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_256);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_1024);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_MAX);

	BIND_ENUM_CONSTANT(RENDER_INFO_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_VERTICES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_MATERIAL_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_SHADER_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_SURFACE_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_MAX);

	BIND_ENUM_CONSTANT(DEBUG_DRAW_DISABLED);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_UNSHADED);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_OVERDRAW);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_WIREFRAME);

	BIND_ENUM_CONSTANT(MSAA_DISABLED);
	BIND_ENUM_CONSTANT(MSAA_2X);
	BIND_ENUM_CONSTANT(MSAA_4X);
	BIND_ENUM_CONSTANT(MSAA_8X);
	BIND_ENUM_CONSTANT(MSAA_16X);

	BIND_ENUM_CONSTANT(USAGE_2D);
	BIND_ENUM_CONSTANT(USAGE_2D_NO_SAMPLING);
	BIND_ENUM_CONSTANT(USAGE_3D);
	BIND_ENUM_CONSTANT(USAGE_3D_NO_EFFECTS);

	BIND_ENUM_CONSTANT(CLEAR_MODE_ALWAYS);
	BIND_ENUM_CONSTANT(CLEAR_MODE_NEVER);
	BIND_ENUM_CONSTANT(CLEAR_MODE_ONLY_NEXT_FRAME);
}

Viewport::Viewport() {

	world_2d = Ref<World2D>(memnew(World2D));

	viewport = VisualServer::get_singleton()->viewport_create();
	texture_rid = VisualServer::get_singleton()->viewport_get_texture(viewport);
	texture_flags = 0;

	default_texture.instance();
	default_texture->vp = const_cast<Viewport *>(this);
	viewport_textures.insert(default_texture.ptr());

	//internal_listener = SpatialSoundServer::get_singleton()->listener_create();
	audio_listener = false;
	//internal_listener_2d = SpatialSound2DServer::get_singleton()->listener_create();
	audio_listener_2d = false;
	transparent_bg = false;
	parent = NULL;
	listener = NULL;
	camera = NULL;
	arvr = false;
	size_override = false;
	size_override_stretch = false;
	size_override_size = Size2(1, 1);
	gen_mipmaps = false;

	vflip = false;

	//clear=true;
	update_mode = UPDATE_WHEN_VISIBLE;

	physics_object_picking = false;
	physics_object_capture = 0;
	physics_object_over = 0;
	physics_last_mousepos = Vector2(1e20, 1e20);

	shadow_atlas_size = 0;
	for (int i = 0; i < 4; i++) {
		shadow_atlas_quadrant_subdiv[i] = SHADOW_ATLAS_QUADRANT_SUBDIV_MAX;
	}
	set_shadow_atlas_quadrant_subdiv(0, SHADOW_ATLAS_QUADRANT_SUBDIV_4);
	set_shadow_atlas_quadrant_subdiv(1, SHADOW_ATLAS_QUADRANT_SUBDIV_4);
	set_shadow_atlas_quadrant_subdiv(2, SHADOW_ATLAS_QUADRANT_SUBDIV_16);
	set_shadow_atlas_quadrant_subdiv(3, SHADOW_ATLAS_QUADRANT_SUBDIV_64);

	String id = itos(get_instance_id());
	input_group = "_vp_input" + id;
	gui_input_group = "_vp_gui_input" + id;
	unhandled_input_group = "_vp_unhandled_input" + id;
	unhandled_key_input_group = "_vp_unhandled_key_input" + id;

	disable_input = false;
	disable_3d = false;

	//window tooltip
	gui.tooltip_timer = -1;

	//gui.tooltip_timer->force_parent_owned();
	gui.tooltip_delay = GLOBAL_DEF("gui/timers/tooltip_delay_sec", 0.7);

	gui.tooltip = NULL;
	gui.tooltip_label = NULL;
	gui.drag_preview = NULL;
	gui.drag_attempted = false;
	gui.canvas_sort_index = 0;

	msaa = MSAA_DISABLED;
	hdr = false;

	usage = USAGE_3D;
	debug_draw = DEBUG_DRAW_DISABLED;
	clear_mode = CLEAR_MODE_ALWAYS;

	snap_controls_to_pixels = true;
}

Viewport::~Viewport() {

	//erase itself from viewport textures
	for (Set<ViewportTexture *>::Element *E = viewport_textures.front(); E; E = E->next()) {
		E->get()->vp = NULL;
	}
	VisualServer::get_singleton()->free(viewport);
	//SpatialSoundServer::get_singleton()->free(internal_listener);
	//SpatialSound2DServer::get_singleton()->free(internal_listener_2d);
}
