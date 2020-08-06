/*************************************************************************/
/*  viewport.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/core_string_names.h"
#include "core/debugger/engine_debugger.h"
#include "core/input/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/2d/collision_object_2d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/collision_object_3d.h"
#include "scene/3d/listener_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/timer.h"
#include "scene/main/window.h"
#include "scene/resources/mesh.h"
#include "scene/scene_string_names.h"
#include "servers/display_server.h"
#include "servers/physics_server_2d.h"

void ViewportTexture::setup_local_to_scene() {
	if (vp) {
		vp->viewport_textures.erase(this);
	}

	vp = nullptr;

	Node *local_scene = get_local_scene();
	if (!local_scene) {
		return;
	}

	Node *vpn = local_scene->get_node(path);
	ERR_FAIL_COND_MSG(!vpn, "ViewportTexture: Path to node is invalid.");

	vp = Object::cast_to<Viewport>(vpn);

	ERR_FAIL_COND_MSG(!vp, "ViewportTexture: Path to node does not point to a viewport.");

	vp->viewport_textures.insert(this);

	if (proxy_ph.is_valid()) {
		RS::get_singleton()->texture_proxy_update(proxy, vp->texture_rid);
		RS::get_singleton()->free(proxy_ph);
	} else {
		ERR_FAIL_COND(proxy.is_valid()); //should be invalid
		proxy = RS::get_singleton()->texture_proxy_create(vp->texture_rid);
	}
}

void ViewportTexture::set_viewport_path_in_scene(const NodePath &p_path) {
	if (path == p_path) {
		return;
	}

	path = p_path;

	if (get_local_scene()) {
		setup_local_to_scene();
	}
}

NodePath ViewportTexture::get_viewport_path_in_scene() const {
	return path;
}

int ViewportTexture::get_width() const {
	ERR_FAIL_COND_V_MSG(!vp, 0, "Viewport Texture must be set to use it.");
	return vp->size.width;
}

int ViewportTexture::get_height() const {
	ERR_FAIL_COND_V_MSG(!vp, 0, "Viewport Texture must be set to use it.");
	return vp->size.height;
}

Size2 ViewportTexture::get_size() const {
	ERR_FAIL_COND_V_MSG(!vp, Size2(), "Viewport Texture must be set to use it.");
	return vp->size;
}

RID ViewportTexture::get_rid() const {
	//ERR_FAIL_COND_V_MSG(!vp, RID(), "Viewport Texture must be set to use it.");
	if (proxy.is_null()) {
		proxy_ph = RS::get_singleton()->texture_2d_placeholder_create();
		proxy = RS::get_singleton()->texture_proxy_create(proxy_ph);
	}
	return proxy;
}

bool ViewportTexture::has_alpha() const {
	return false;
}

Ref<Image> ViewportTexture::get_data() const {
	ERR_FAIL_COND_V_MSG(!vp, Ref<Image>(), "Viewport Texture must be set to use it.");
	return RS::get_singleton()->texture_2d_get(vp->texture_rid);
}

void ViewportTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_viewport_path_in_scene", "path"), &ViewportTexture::set_viewport_path_in_scene);
	ClassDB::bind_method(D_METHOD("get_viewport_path_in_scene"), &ViewportTexture::get_viewport_path_in_scene);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "viewport_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "SubViewport", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT), "set_viewport_path_in_scene", "get_viewport_path_in_scene");
}

ViewportTexture::ViewportTexture() {
	vp = nullptr;
	set_local_to_scene(true);
}

ViewportTexture::~ViewportTexture() {
	if (vp) {
		vp->viewport_textures.erase(this);
	}

	if (proxy_ph.is_valid()) {
		RS::get_singleton()->free(proxy_ph);
	}
	if (proxy.is_valid()) {
		RS::get_singleton()->free(proxy);
	}
}

/////////////////////////////////////

// Aliases used to provide custom styles to tooltips in the default
// theme and editor theme.
// TooltipPanel is also used for custom tooltips, while TooltipLabel
// is only relevant for default tooltips.

class TooltipPanel : public PopupPanel {
	GDCLASS(TooltipPanel, PopupPanel);

public:
	TooltipPanel() {}
};

class TooltipLabel : public Label {
	GDCLASS(TooltipLabel, Label);

public:
	TooltipLabel() {}
};

/////////////////////////////////////

Viewport::GUI::GUI() {
	embed_subwindows_hint = false;
	embedding_subwindows = false;

	dragging = false;
	mouse_focus = nullptr;
	forced_mouse_focus = false;
	mouse_click_grabber = nullptr;
	mouse_focus_mask = 0;
	key_focus = nullptr;
	mouse_over = nullptr;
	drag_mouse_over = nullptr;

	tooltip_control = nullptr;
	tooltip_popup = nullptr;
	tooltip_label = nullptr;
}

/////////////////////////////////////

void Viewport::update_worlds() {
	if (!is_inside_tree()) {
		return;
	}

	Rect2 abstracted_rect = Rect2(Vector2(), get_visible_rect().size);
	Rect2 xformed_rect = (global_canvas_transform * canvas_transform).affine_inverse().xform(abstracted_rect);
	find_world_2d()->_update_viewport(this, xformed_rect);
	find_world_2d()->_update();

	find_world_3d()->_update(get_tree()->get_frame());
}

void Viewport::_collision_object_input_event(CollisionObject3D *p_object, Camera3D *p_camera, const Ref<InputEvent> &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape) {
	Transform object_transform = p_object->get_global_transform();
	Transform camera_transform = p_camera->get_global_transform();
	ObjectID id = p_object->get_instance_id();

	//avoid sending the fake event unnecessarily if nothing really changed in the context
	if (object_transform == physics_last_object_transform && camera_transform == physics_last_camera_transform && physics_last_id == id) {
		Ref<InputEventMouseMotion> mm = p_input_event;
		if (mm.is_valid() && mm->get_device() == InputEvent::DEVICE_ID_INTERNAL) {
			return; //discarded
		}
	}
	p_object->_input_event(camera, p_input_event, p_pos, p_normal, p_shape);
	physics_last_object_transform = object_transform;
	physics_last_camera_transform = camera_transform;
	physics_last_id = id;
}

void Viewport::_sub_window_update_order() {
	for (int i = 0; i < gui.sub_windows.size(); i++) {
		RS::get_singleton()->canvas_item_set_draw_index(gui.sub_windows[i].canvas_item, i);
	}
}

void Viewport::_sub_window_register(Window *p_window) {
	ERR_FAIL_COND(!is_inside_tree());
	for (int i = 0; i < gui.sub_windows.size(); i++) {
		ERR_FAIL_COND(gui.sub_windows[i].window == p_window);
	}

	if (gui.sub_windows.size() == 0) {
		subwindow_canvas = RS::get_singleton()->canvas_create();
		RS::get_singleton()->viewport_attach_canvas(viewport, subwindow_canvas);
		RS::get_singleton()->viewport_set_canvas_stacking(viewport, subwindow_canvas, SUBWINDOW_CANVAS_LAYER, 0);
	}
	SubWindow sw;
	sw.canvas_item = RS::get_singleton()->canvas_item_create();
	RS::get_singleton()->canvas_item_set_parent(sw.canvas_item, subwindow_canvas);
	sw.window = p_window;
	gui.sub_windows.push_back(sw);

	_sub_window_grab_focus(p_window);

	RenderingServer::get_singleton()->viewport_set_parent_viewport(p_window->viewport, viewport);
}

void Viewport::_sub_window_update(Window *p_window) {
	int index = -1;
	for (int i = 0; i < gui.sub_windows.size(); i++) {
		if (gui.sub_windows[i].window == p_window) {
			index = i;
			break;
		}
	}

	ERR_FAIL_COND(index == -1);

	const SubWindow &sw = gui.sub_windows[index];

	Transform2D pos;
	pos.set_origin(p_window->get_position());
	RS::get_singleton()->canvas_item_clear(sw.canvas_item);
	Rect2i r = Rect2i(p_window->get_position(), sw.window->get_size());

	if (!p_window->get_flag(Window::FLAG_BORDERLESS)) {
		Ref<StyleBox> panel = p_window->get_theme_stylebox("panel_window");
		panel->draw(sw.canvas_item, r);

		// Draw the title bar text.
		Ref<Font> title_font = p_window->get_theme_font("title_font");
		Color title_color = p_window->get_theme_color("title_color");
		int title_height = p_window->get_theme_constant("title_height");
		int font_height = title_font->get_height() - title_font->get_descent() * 2;
		int x = (r.size.width - title_font->get_string_size(p_window->get_title()).x) / 2;
		int y = (-title_height + font_height) / 2;

		int close_h_ofs = p_window->get_theme_constant("close_h_ofs");
		int close_v_ofs = p_window->get_theme_constant("close_v_ofs");

		title_font->draw(sw.canvas_item, r.position + Point2(x, y), p_window->get_title(), title_color, r.size.width - panel->get_minimum_size().x - close_h_ofs);

		bool hl = gui.subwindow_focused == sw.window && gui.subwindow_drag == SUB_WINDOW_DRAG_CLOSE && gui.subwindow_drag_close_inside;

		Ref<Texture2D> close_icon = p_window->get_theme_icon(hl ? "close_highlight" : "close");
		close_icon->draw(sw.canvas_item, r.position + Vector2(r.size.width - close_h_ofs, -close_v_ofs));
	}

	RS::get_singleton()->canvas_item_add_texture_rect(sw.canvas_item, r, sw.window->get_texture()->get_rid());
}

void Viewport::_sub_window_grab_focus(Window *p_window) {
	if (p_window == nullptr) {
		//release current focus
		if (gui.subwindow_focused) {
			gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
			gui.subwindow_focused = nullptr;
			gui.subwindow_drag = SUB_WINDOW_DRAG_DISABLED;
		}

		Window *this_window = Object::cast_to<Window>(this);
		if (this_window) {
			this_window->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		}

		return;
	}

	int index = -1;
	for (int i = 0; i < gui.sub_windows.size(); i++) {
		if (gui.sub_windows[i].window == p_window) {
			index = i;
			break;
		}
	}

	ERR_FAIL_COND(index == -1);

	if (p_window->get_flag(Window::FLAG_NO_FOCUS)) {
		//can only move to foreground, but no focus granted
		SubWindow sw = gui.sub_windows[index];
		gui.sub_windows.remove(index);
		gui.sub_windows.push_back(sw);
		index = gui.sub_windows.size() - 1;
		_sub_window_update_order();
		return; //i guess not...
	}

	if (gui.subwindow_focused) {
		if (gui.subwindow_focused == p_window) {
			return; //nothing to do
		}
		gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		gui.subwindow_drag = SUB_WINDOW_DRAG_DISABLED;
	} else {
		Window *this_window = Object::cast_to<Window>(this);
		if (this_window) {
			this_window->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		}
	}

	Window *old_focus = gui.subwindow_focused;

	gui.subwindow_focused = p_window;

	gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_IN);

	{ //move to foreground
		SubWindow sw = gui.sub_windows[index];
		gui.sub_windows.remove(index);
		gui.sub_windows.push_back(sw);
		index = gui.sub_windows.size() - 1;
		_sub_window_update_order();
	}

	if (old_focus) {
		_sub_window_update(old_focus);
	}

	_sub_window_update(p_window);
}

void Viewport::_sub_window_remove(Window *p_window) {
	for (int i = 0; i < gui.sub_windows.size(); i++) {
		if (gui.sub_windows[i].window == p_window) {
			RS::get_singleton()->free(gui.sub_windows[i].canvas_item);
			gui.sub_windows.remove(i);
			break;
		}
	}

	if (gui.sub_windows.size() == 0) {
		RS::get_singleton()->free(subwindow_canvas);
		subwindow_canvas = RID();
	}

	if (gui.subwindow_focused == p_window) {
		Window *parent_visible = p_window->get_parent_visible_window();

		gui.subwindow_drag = SUB_WINDOW_DRAG_DISABLED;

		gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_OUT);

		if (parent_visible && parent_visible != this) {
			gui.subwindow_focused = parent_visible;
			gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		} else {
			gui.subwindow_focused = nullptr;
			Window *this_window = Object::cast_to<Window>(this);
			if (this_window) {
				this_window->_event_callback(DisplayServer::WINDOW_EVENT_FOCUS_IN);
			}
		}
	}

	RenderingServer::get_singleton()->viewport_set_parent_viewport(p_window->viewport, p_window->parent ? p_window->parent->viewport : RID());
}

void Viewport::_own_world_3d_changed() {
	ERR_FAIL_COND(world_3d.is_null());
	ERR_FAIL_COND(own_world_3d.is_null());

	if (is_inside_tree()) {
		_propagate_exit_world(this);
	}

	own_world_3d = world_3d->duplicate();

	if (is_inside_tree()) {
		_propagate_enter_world(this);
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
	}

	_update_listener();
}

void Viewport::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			gui.embedding_subwindows = gui.embed_subwindows_hint;

			if (get_parent()) {
				parent = get_parent()->get_viewport();
				RenderingServer::get_singleton()->viewport_set_parent_viewport(viewport, parent->get_viewport_rid());
			} else {
				parent = nullptr;
			}

			current_canvas = find_world_2d()->get_canvas();
			RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
			RenderingServer::get_singleton()->viewport_attach_canvas(viewport, current_canvas);

			_update_listener();
			_update_listener_2d();

			find_world_2d()->_register_viewport(this, Rect2());

			add_to_group("_viewports");
			if (get_tree()->is_debugging_collisions_hint()) {
				//2D
				PhysicsServer2D::get_singleton()->space_set_debug_contacts(find_world_2d()->get_space(), get_tree()->get_collision_debug_contact_count());
				contact_2d_debug = RenderingServer::get_singleton()->canvas_item_create();
				RenderingServer::get_singleton()->canvas_item_set_parent(contact_2d_debug, find_world_2d()->get_canvas());
				//3D
				PhysicsServer3D::get_singleton()->space_set_debug_contacts(find_world_3d()->get_space(), get_tree()->get_collision_debug_contact_count());
				contact_3d_debug_multimesh = RenderingServer::get_singleton()->multimesh_create();
				RenderingServer::get_singleton()->multimesh_allocate(contact_3d_debug_multimesh, get_tree()->get_collision_debug_contact_count(), RS::MULTIMESH_TRANSFORM_3D, true);
				RenderingServer::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh, 0);
				RenderingServer::get_singleton()->multimesh_set_mesh(contact_3d_debug_multimesh, get_tree()->get_debug_contact_mesh()->get_rid());
				contact_3d_debug_instance = RenderingServer::get_singleton()->instance_create();
				RenderingServer::get_singleton()->instance_set_base(contact_3d_debug_instance, contact_3d_debug_multimesh);
				RenderingServer::get_singleton()->instance_set_scenario(contact_3d_debug_instance, find_world_3d()->get_scenario());
				//RenderingServer::get_singleton()->instance_geometry_set_flag(contact_3d_debug_instance, RS::INSTANCE_FLAG_VISIBLE_IN_ALL_ROOMS, true);
			}

		} break;
		case NOTIFICATION_READY: {
#ifndef _3D_DISABLED
			if (listeners.size() && !listener) {
				Listener3D *first = nullptr;
				for (Set<Listener3D *>::Element *E = listeners.front(); E; E = E->next()) {
					if (first == nullptr || first->is_greater_than(E->get())) {
						first = E->get();
					}
				}

				if (first) {
					first->make_current();
				}
			}

			if (cameras.size() && !camera) {
				//there are cameras but no current camera, pick first in tree and make it current
				Camera3D *first = nullptr;
				for (Set<Camera3D *>::Element *E = cameras.front(); E; E = E->next()) {
					if (first == nullptr || first->is_greater_than(E->get())) {
						first = E->get();
					}
				}

				if (first) {
					first->make_current();
				}
			}
#endif

			// Enable processing for tooltips, collision debugging, physics object picking, etc.
			set_process_internal(true);
			set_physics_process_internal(true);

		} break;
		case NOTIFICATION_EXIT_TREE: {
			_gui_cancel_tooltip();
			if (world_2d.is_valid()) {
				world_2d->_remove_viewport(this);
			}

			RenderingServer::get_singleton()->viewport_set_scenario(viewport, RID());
			RenderingServer::get_singleton()->viewport_remove_canvas(viewport, current_canvas);
			if (contact_2d_debug.is_valid()) {
				RenderingServer::get_singleton()->free(contact_2d_debug);
				contact_2d_debug = RID();
			}

			if (contact_3d_debug_multimesh.is_valid()) {
				RenderingServer::get_singleton()->free(contact_3d_debug_multimesh);
				RenderingServer::get_singleton()->free(contact_3d_debug_instance);
				contact_3d_debug_instance = RID();
				contact_3d_debug_multimesh = RID();
			}

			remove_from_group("_viewports");

			RS::get_singleton()->viewport_set_active(viewport, false);
			RenderingServer::get_singleton()->viewport_set_parent_viewport(viewport, RID());

		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (gui.tooltip_timer >= 0) {
				gui.tooltip_timer -= get_process_delta_time();
				if (gui.tooltip_timer < 0) {
					_gui_show_tooltip();
				}
			}

		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (get_tree()->is_debugging_collisions_hint() && contact_2d_debug.is_valid()) {
				RenderingServer::get_singleton()->canvas_item_clear(contact_2d_debug);
				RenderingServer::get_singleton()->canvas_item_set_draw_index(contact_2d_debug, 0xFFFFF); //very high index

				Vector<Vector2> points = PhysicsServer2D::get_singleton()->space_get_contacts(find_world_2d()->get_space());
				int point_count = PhysicsServer2D::get_singleton()->space_get_contact_count(find_world_2d()->get_space());
				Color ccol = get_tree()->get_debug_collision_contact_color();

				for (int i = 0; i < point_count; i++) {
					RenderingServer::get_singleton()->canvas_item_add_rect(contact_2d_debug, Rect2(points[i] - Vector2(2, 2), Vector2(5, 5)), ccol);
				}
			}

			if (get_tree()->is_debugging_collisions_hint() && contact_3d_debug_multimesh.is_valid()) {
				Vector<Vector3> points = PhysicsServer3D::get_singleton()->space_get_contacts(find_world_3d()->get_space());
				int point_count = PhysicsServer3D::get_singleton()->space_get_contact_count(find_world_3d()->get_space());

				RS::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh, point_count);
			}

			if (physics_object_picking && (to_screen_rect == Rect2i() || Input::get_singleton()->get_mouse_mode() != Input::MOUSE_MODE_CAPTURED)) {
#ifndef _3D_DISABLED
				Vector2 last_pos(1e20, 1e20);
				CollisionObject3D *last_object = nullptr;
				ObjectID last_id;
#endif
				PhysicsDirectSpaceState3D::RayResult result;
				PhysicsDirectSpaceState2D *ss2d = PhysicsServer2D::get_singleton()->space_get_direct_state(find_world_2d()->get_space());

				if (physics_has_last_mousepos) {
					// if no mouse event exists, create a motion one. This is necessary because objects or camera may have moved.
					// while this extra event is sent, it is checked if both camera and last object and last ID did not move. If nothing changed, the event is discarded to avoid flooding with unnecessary motion events every frame
					bool has_mouse_event = false;
					for (List<Ref<InputEvent>>::Element *E = physics_picking_events.front(); E; E = E->next()) {
						Ref<InputEventMouse> m = E->get();
						if (m.is_valid()) {
							has_mouse_event = true;
							break;
						}
					}

					if (!has_mouse_event) {
						Ref<InputEventMouseMotion> mm;
						mm.instance();

						mm->set_device(InputEvent::DEVICE_ID_INTERNAL);
						mm->set_global_position(physics_last_mousepos);
						mm->set_position(physics_last_mousepos);
						mm->set_alt(physics_last_mouse_state.alt);
						mm->set_shift(physics_last_mouse_state.shift);
						mm->set_control(physics_last_mouse_state.control);
						mm->set_metakey(physics_last_mouse_state.meta);
						mm->set_button_mask(physics_last_mouse_state.mouse_mask);
						physics_picking_events.push_back(mm);
					}
				}

				while (physics_picking_events.size()) {
					Ref<InputEvent> ev = physics_picking_events.front()->get();
					physics_picking_events.pop_front();

					Vector2 pos;
					bool is_mouse = false;

					Ref<InputEventMouseMotion> mm = ev;

					if (mm.is_valid()) {
						pos = mm->get_position();
						is_mouse = true;

						physics_has_last_mousepos = true;
						physics_last_mousepos = pos;
						physics_last_mouse_state.alt = mm->get_alt();
						physics_last_mouse_state.shift = mm->get_shift();
						physics_last_mouse_state.control = mm->get_control();
						physics_last_mouse_state.meta = mm->get_metakey();
						physics_last_mouse_state.mouse_mask = mm->get_button_mask();
					}

					Ref<InputEventMouseButton> mb = ev;

					if (mb.is_valid()) {
						pos = mb->get_position();
						is_mouse = true;

						physics_has_last_mousepos = true;
						physics_last_mousepos = pos;
						physics_last_mouse_state.alt = mb->get_alt();
						physics_last_mouse_state.shift = mb->get_shift();
						physics_last_mouse_state.control = mb->get_control();
						physics_last_mouse_state.meta = mb->get_metakey();

						if (mb->is_pressed()) {
							physics_last_mouse_state.mouse_mask |= (1 << (mb->get_button_index() - 1));
						} else {
							physics_last_mouse_state.mouse_mask &= ~(1 << (mb->get_button_index() - 1));

							// If touch mouse raised, assume we don't know last mouse pos until new events come
							if (mb->get_device() == InputEvent::DEVICE_ID_TOUCH_MOUSE) {
								physics_has_last_mousepos = false;
							}
						}
					}

					Ref<InputEventKey> k = ev;
					if (k.is_valid()) {
						//only for mask
						physics_last_mouse_state.alt = k->get_alt();
						physics_last_mouse_state.shift = k->get_shift();
						physics_last_mouse_state.control = k->get_control();
						physics_last_mouse_state.meta = k->get_metakey();
						continue;
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

						PhysicsDirectSpaceState2D::ShapeResult res[64];
						for (Set<CanvasLayer *>::Element *E = canvas_layers.front(); E; E = E->next()) {
							Transform2D canvas_transform;
							ObjectID canvas_layer_id;
							if (E->get()) {
								// A descendant CanvasLayer
								canvas_transform = E->get()->get_transform();
								canvas_layer_id = E->get()->get_instance_id();
							} else {
								// This Viewport's builtin canvas
								canvas_transform = get_canvas_transform();
								canvas_layer_id = ObjectID();
							}

							Vector2 point = canvas_transform.affine_inverse().xform(pos);

							int rc = ss2d->intersect_point_on_canvas(point, canvas_layer_id, res, 64, Set<RID>(), 0xFFFFFFFF, true, true, true);
							for (int i = 0; i < rc; i++) {
								if (res[i].collider_id.is_valid() && res[i].collider) {
									CollisionObject2D *co = Object::cast_to<CollisionObject2D>(res[i].collider);
									if (co) {
										bool send_event = true;
										if (is_mouse) {
											Map<ObjectID, uint64_t>::Element *F = physics_2d_mouseover.find(res[i].collider_id);

											if (!F) {
												physics_2d_mouseover.insert(res[i].collider_id, frame);
												co->_mouse_enter();
											} else {
												F->get() = frame;
												// It was already hovered, so don't send the event if it's faked
												if (mm.is_valid() && mm->get_device() == InputEvent::DEVICE_ID_INTERNAL) {
													send_event = false;
												}
											}
										}

										if (send_event) {
											co->_input_event(this, ev, res[i].shape);
										}
									}
								}
							}
						}

						if (is_mouse) {
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
					}

#ifndef _3D_DISABLED
					bool captured = false;

					if (physics_object_capture.is_valid()) {
						CollisionObject3D *co = Object::cast_to<CollisionObject3D>(ObjectDB::get_instance(physics_object_capture));
						if (co && camera) {
							_collision_object_input_event(co, camera, ev, Vector3(), Vector3(), 0);
							captured = true;
							if (mb.is_valid() && mb->get_button_index() == 1 && !mb->is_pressed()) {
								physics_object_capture = ObjectID();
							}

						} else {
							physics_object_capture = ObjectID();
						}
					}

					if (captured) {
						//none
					} else if (pos == last_pos) {
						if (last_id.is_valid()) {
							if (ObjectDB::get_instance(last_id) && last_object) {
								//good, exists
								_collision_object_input_event(last_object, camera, ev, result.position, result.normal, result.shape);
								if (last_object->get_capture_input_on_drag() && mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
									physics_object_capture = last_id;
								}
							}
						}
					} else {
						if (camera) {
							Vector3 from = camera->project_ray_origin(pos);
							Vector3 dir = camera->project_ray_normal(pos);

							PhysicsDirectSpaceState3D *space = PhysicsServer3D::get_singleton()->space_get_direct_state(find_world_3d()->get_space());
							if (space) {
								bool col = space->intersect_ray(from, from + dir * 10000, result, Set<RID>(), 0xFFFFFFFF, true, true, true);
								ObjectID new_collider;
								if (col) {
									CollisionObject3D *co = Object::cast_to<CollisionObject3D>(result.collider);
									if (co) {
										_collision_object_input_event(co, camera, ev, result.position, result.normal, result.shape);
										last_object = co;
										last_id = result.collider_id;
										new_collider = last_id;
										if (co->get_capture_input_on_drag() && mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
											physics_object_capture = last_id;
										}
									}
								}

								if (is_mouse && new_collider != physics_object_over) {
									if (physics_object_over.is_valid()) {
										CollisionObject3D *co = Object::cast_to<CollisionObject3D>(ObjectDB::get_instance(physics_object_over));
										if (co) {
											co->_mouse_exit();
										}
									}

									if (new_collider.is_valid()) {
										CollisionObject3D *co = Object::cast_to<CollisionObject3D>(ObjectDB::get_instance(new_collider));
										if (co) {
											co->_mouse_enter();
										}
									}

									physics_object_over = new_collider;
								}
							}

							last_pos = pos;
						}
					}
#endif
				}
			}

		} break;
		case NOTIFICATION_WM_MOUSE_EXIT: {
			_drop_physics_mouseover();

			// Unlike on loss of focus (NOTIFICATION_WM_WINDOW_FOCUS_OUT), do not
			// drop the gui mouseover here, as a scrollbar may be dragged while the
			// mouse is outside the window (without the window having lost focus).
			// See bug #39634
		} break;
		case NOTIFICATION_WM_WINDOW_FOCUS_OUT: {
			_drop_physics_mouseover();

			if (gui.mouse_focus && !gui.forced_mouse_focus) {
				_drop_mouse_focus();
			}
		} break;
	}
}

RID Viewport::get_viewport_rid() const {
	return viewport;
}

void Viewport::update_canvas_items() {
	if (!is_inside_tree()) {
		return;
	}

	_update_canvas_items(this);
}

void Viewport::_set_size(const Size2i &p_size, const Size2i &p_size_2d_override, const Rect2i &p_to_screen_rect, const Transform2D &p_stretch_transform, bool p_allocated) {
	if (size == p_size && size_allocated == p_allocated && stretch_transform == p_stretch_transform && p_size_2d_override == size_2d_override && to_screen_rect != p_to_screen_rect) {
		return;
	}

	size = p_size;
	size_allocated = p_allocated;
	size_2d_override = p_size_2d_override;
	stretch_transform = p_stretch_transform;
	to_screen_rect = p_to_screen_rect;

	if (p_allocated) {
		RS::get_singleton()->viewport_set_size(viewport, size.width, size.height);
	} else {
		RS::get_singleton()->viewport_set_size(viewport, 0, 0);
	}
	_update_global_transform();

	update_canvas_items();

	emit_signal("size_changed");
}

Size2i Viewport::_get_size() const {
	return size;
}

Size2i Viewport::_get_size_2d_override() const {
	return size_2d_override;
}

bool Viewport::_is_size_allocated() const {
	return size_allocated;
}

Rect2 Viewport::get_visible_rect() const {
	Rect2 r;

	if (size == Size2()) {
		r = Rect2(Point2(), DisplayServer::get_singleton()->window_get_size());
	} else {
		r = Rect2(Point2(), size);
	}

	if (size_2d_override != Size2i()) {
		r.size = size_2d_override;
	}

	return r;
}

void Viewport::_update_listener() {
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
	if (p_enable == audio_listener) {
		return;
	}

	audio_listener = p_enable;
	_update_listener();
}

bool Viewport::is_audio_listener() const {
	return audio_listener;
}

void Viewport::set_as_audio_listener_2d(bool p_enable) {
	if (p_enable == audio_listener_2d) {
		return;
	}

	audio_listener_2d = p_enable;

	_update_listener_2d();
}

bool Viewport::is_audio_listener_2d() const {
	return audio_listener_2d;
}

void Viewport::enable_canvas_transform_override(bool p_enable) {
	if (override_canvas_transform == p_enable) {
		return;
	}

	override_canvas_transform = p_enable;
	if (p_enable) {
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, find_world_2d()->get_canvas(), canvas_transform_override);
	} else {
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, find_world_2d()->get_canvas(), canvas_transform);
	}
}

bool Viewport::is_canvas_transform_override_enbled() const {
	return override_canvas_transform;
}

void Viewport::set_canvas_transform_override(const Transform2D &p_transform) {
	if (canvas_transform_override == p_transform) {
		return;
	}

	canvas_transform_override = p_transform;
	if (override_canvas_transform) {
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, find_world_2d()->get_canvas(), canvas_transform_override);
	}
}

Transform2D Viewport::get_canvas_transform_override() const {
	return canvas_transform_override;
}

void Viewport::set_canvas_transform(const Transform2D &p_transform) {
	canvas_transform = p_transform;

	if (!override_canvas_transform) {
		RenderingServer::get_singleton()->viewport_set_canvas_transform(viewport, find_world_2d()->get_canvas(), canvas_transform);
	}
}

Transform2D Viewport::get_canvas_transform() const {
	return canvas_transform;
}

void Viewport::_update_global_transform() {
	Transform2D sxform = stretch_transform * global_canvas_transform;

	RenderingServer::get_singleton()->viewport_set_global_canvas_transform(viewport, sxform);
}

void Viewport::set_global_canvas_transform(const Transform2D &p_transform) {
	global_canvas_transform = p_transform;

	_update_global_transform();
}

Transform2D Viewport::get_global_canvas_transform() const {
	return global_canvas_transform;
}

void Viewport::_listener_transform_changed_notify() {
}

void Viewport::_listener_set(Listener3D *p_listener) {
#ifndef _3D_DISABLED

	if (listener == p_listener) {
		return;
	}

	listener = p_listener;

	_update_listener();
	_listener_transform_changed_notify();
#endif
}

bool Viewport::_listener_add(Listener3D *p_listener) {
	listeners.insert(p_listener);
	return listeners.size() == 1;
}

void Viewport::_listener_remove(Listener3D *p_listener) {
	listeners.erase(p_listener);
	if (listener == p_listener) {
		listener = nullptr;
	}
}

#ifndef _3D_DISABLED
void Viewport::_listener_make_next_current(Listener3D *p_exclude) {
	if (listeners.size() > 0) {
		for (Set<Listener3D *>::Element *E = listeners.front(); E; E = E->next()) {
			if (p_exclude == E->get()) {
				continue;
			}
			if (!E->get()->is_inside_tree()) {
				continue;
			}
			if (listener != nullptr) {
				return;
			}

			E->get()->make_current();
		}
	} else {
		// Attempt to reset listener to the camera position
		if (camera != nullptr) {
			_update_listener();
			_camera_transform_changed_notify();
		}
	}
}
#endif

void Viewport::_camera_transform_changed_notify() {
#ifndef _3D_DISABLED
#endif
}

void Viewport::_camera_set(Camera3D *p_camera) {
#ifndef _3D_DISABLED

	if (camera == p_camera) {
		return;
	}

	if (camera) {
		camera->notification(Camera3D::NOTIFICATION_LOST_CURRENT);
	}

	camera = p_camera;

	if (!camera_override) {
		if (camera) {
			RenderingServer::get_singleton()->viewport_attach_camera(viewport, camera->get_camera());
		} else {
			RenderingServer::get_singleton()->viewport_attach_camera(viewport, RID());
		}
	}

	if (camera) {
		camera->notification(Camera3D::NOTIFICATION_BECAME_CURRENT);
	}

	_update_listener();
	_camera_transform_changed_notify();
#endif
}

bool Viewport::_camera_add(Camera3D *p_camera) {
	cameras.insert(p_camera);
	return cameras.size() == 1;
}

void Viewport::_camera_remove(Camera3D *p_camera) {
	cameras.erase(p_camera);
	if (camera == p_camera) {
		camera->notification(Camera3D::NOTIFICATION_LOST_CURRENT);
		camera = nullptr;
	}
}

#ifndef _3D_DISABLED
void Viewport::_camera_make_next_current(Camera3D *p_exclude) {
	for (Set<Camera3D *>::Element *E = cameras.front(); E; E = E->next()) {
		if (p_exclude == E->get()) {
			continue;
		}
		if (!E->get()->is_inside_tree()) {
			continue;
		}
		if (camera != nullptr) {
			return;
		}

		E->get()->make_current();
	}
}
#endif

void Viewport::_canvas_layer_add(CanvasLayer *p_canvas_layer) {
	canvas_layers.insert(p_canvas_layer);
}

void Viewport::_canvas_layer_remove(CanvasLayer *p_canvas_layer) {
	canvas_layers.erase(p_canvas_layer);
}

void Viewport::set_transparent_background(bool p_enable) {
	transparent_bg = p_enable;
	RS::get_singleton()->viewport_set_transparent_background(viewport, p_enable);
}

bool Viewport::has_transparent_background() const {
	return transparent_bg;
}

void Viewport::set_world_2d(const Ref<World2D> &p_world_2d) {
	if (world_2d == p_world_2d) {
		return;
	}

	if (parent && parent->find_world_2d() == p_world_2d) {
		WARN_PRINT("Unable to use parent world_3d as world_2d");
		return;
	}

	if (is_inside_tree()) {
		find_world_2d()->_remove_viewport(this);
		RenderingServer::get_singleton()->viewport_remove_canvas(viewport, current_canvas);
	}

	if (p_world_2d.is_valid()) {
		world_2d = p_world_2d;
	} else {
		WARN_PRINT("Invalid world_3d");
		world_2d = Ref<World2D>(memnew(World2D));
	}

	_update_listener_2d();

	if (is_inside_tree()) {
		current_canvas = find_world_2d()->get_canvas();
		RenderingServer::get_singleton()->viewport_attach_canvas(viewport, current_canvas);
		find_world_2d()->_register_viewport(this, Rect2());
	}
}

Ref<World2D> Viewport::find_world_2d() const {
	if (world_2d.is_valid()) {
		return world_2d;
	} else if (parent) {
		return parent->find_world_2d();
	} else {
		return Ref<World2D>();
	}
}

void Viewport::_propagate_enter_world(Node *p_node) {
	if (p_node != this) {
		if (!p_node->is_inside_tree()) { //may not have entered scene yet
			return;
		}

#ifndef _3D_DISABLED
		if (Object::cast_to<Node3D>(p_node) || Object::cast_to<WorldEnvironment>(p_node)) {
			p_node->notification(Node3D::NOTIFICATION_ENTER_WORLD);
		} else {
#endif
			Viewport *v = Object::cast_to<Viewport>(p_node);
			if (v) {
				if (v->world_3d.is_valid() || v->own_world_3d.is_valid()) {
					return;
				}
			}
#ifndef _3D_DISABLED
		}
#endif
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_propagate_enter_world(p_node->get_child(i));
	}
}

void Viewport::_propagate_viewport_notification(Node *p_node, int p_what) {
	p_node->notification(p_what);
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *c = p_node->get_child(i);
		if (Object::cast_to<Viewport>(c)) {
			continue;
		}
		_propagate_viewport_notification(c, p_what);
	}
}

void Viewport::_propagate_exit_world(Node *p_node) {
	if (p_node != this) {
		if (!p_node->is_inside_tree()) { //may have exited scene already
			return;
		}

#ifndef _3D_DISABLED
		if (Object::cast_to<Node3D>(p_node) || Object::cast_to<WorldEnvironment>(p_node)) {
			p_node->notification(Node3D::NOTIFICATION_EXIT_WORLD);
		} else {
#endif
			Viewport *v = Object::cast_to<Viewport>(p_node);
			if (v) {
				if (v->world_3d.is_valid() || v->own_world_3d.is_valid()) {
					return;
				}
			}
#ifndef _3D_DISABLED
		}
#endif
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_propagate_exit_world(p_node->get_child(i));
	}
}

void Viewport::set_world_3d(const Ref<World3D> &p_world_3d) {
	if (world_3d == p_world_3d) {
		return;
	}

	if (is_inside_tree()) {
		_propagate_exit_world(this);
	}

	if (own_world_3d.is_valid() && world_3d.is_valid()) {
		world_3d->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &Viewport::_own_world_3d_changed));
	}

	world_3d = p_world_3d;

	if (own_world_3d.is_valid()) {
		if (world_3d.is_valid()) {
			own_world_3d = world_3d->duplicate();
			world_3d->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &Viewport::_own_world_3d_changed));
		} else {
			own_world_3d = Ref<World3D>(memnew(World3D));
		}
	}

	if (is_inside_tree()) {
		_propagate_enter_world(this);
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
	}

	_update_listener();
}

Ref<World3D> Viewport::get_world_3d() const {
	return world_3d;
}

Ref<World2D> Viewport::get_world_2d() const {
	return world_2d;
}

Ref<World3D> Viewport::find_world_3d() const {
	if (own_world_3d.is_valid()) {
		return own_world_3d;
	} else if (world_3d.is_valid()) {
		return world_3d;
	} else if (parent) {
		return parent->find_world_3d();
	} else {
		return Ref<World3D>();
	}
}

Listener3D *Viewport::get_listener() const {
	return listener;
}

Camera3D *Viewport::get_camera() const {
	return camera;
}

void Viewport::enable_camera_override(bool p_enable) {
#ifndef _3D_DISABLED
	if (p_enable == camera_override) {
		return;
	}

	if (p_enable) {
		camera_override.rid = RenderingServer::get_singleton()->camera_create();
	} else {
		RenderingServer::get_singleton()->free(camera_override.rid);
		camera_override.rid = RID();
	}

	if (p_enable) {
		RenderingServer::get_singleton()->viewport_attach_camera(viewport, camera_override.rid);
	} else if (camera) {
		RenderingServer::get_singleton()->viewport_attach_camera(viewport, camera->get_camera());
	} else {
		RenderingServer::get_singleton()->viewport_attach_camera(viewport, RID());
	}
#endif
}

bool Viewport::is_camera_override_enabled() const {
	return camera_override;
}

void Viewport::set_camera_override_transform(const Transform &p_transform) {
	if (camera_override) {
		camera_override.transform = p_transform;
		RenderingServer::get_singleton()->camera_set_transform(camera_override.rid, p_transform);
	}
}

Transform Viewport::get_camera_override_transform() const {
	if (camera_override) {
		return camera_override.transform;
	}

	return Transform();
}

void Viewport::set_camera_override_perspective(float p_fovy_degrees, float p_z_near, float p_z_far) {
	if (camera_override) {
		if (camera_override.fov == p_fovy_degrees && camera_override.z_near == p_z_near &&
				camera_override.z_far == p_z_far && camera_override.projection == CameraOverrideData::PROJECTION_PERSPECTIVE) {
			return;
		}

		camera_override.fov = p_fovy_degrees;
		camera_override.z_near = p_z_near;
		camera_override.z_far = p_z_far;
		camera_override.projection = CameraOverrideData::PROJECTION_PERSPECTIVE;

		RenderingServer::get_singleton()->camera_set_perspective(camera_override.rid, camera_override.fov, camera_override.z_near, camera_override.z_far);
	}
}

void Viewport::set_camera_override_orthogonal(float p_size, float p_z_near, float p_z_far) {
	if (camera_override) {
		if (camera_override.size == p_size && camera_override.z_near == p_z_near &&
				camera_override.z_far == p_z_far && camera_override.projection == CameraOverrideData::PROJECTION_ORTHOGONAL) {
			return;
		}

		camera_override.size = p_size;
		camera_override.z_near = p_z_near;
		camera_override.z_far = p_z_far;
		camera_override.projection = CameraOverrideData::PROJECTION_ORTHOGONAL;

		RenderingServer::get_singleton()->camera_set_orthogonal(camera_override.rid, camera_override.size, camera_override.z_near, camera_override.z_far);
	}
}

Transform2D Viewport::get_final_transform() const {
	return stretch_transform * global_canvas_transform;
}

void Viewport::_update_canvas_items(Node *p_node) {
	if (p_node != this) {
		Viewport *vp = Object::cast_to<Viewport>(p_node);
		if (vp) {
			return;
		}

		CanvasItem *ci = Object::cast_to<CanvasItem>(p_node);
		if (ci) {
			ci->update();
		}
	}

	int cc = p_node->get_child_count();

	for (int i = 0; i < cc; i++) {
		_update_canvas_items(p_node->get_child(i));
	}
}

Ref<ViewportTexture> Viewport::get_texture() const {
	return default_texture;
}

void Viewport::set_shadow_atlas_size(int p_size) {
	if (shadow_atlas_size == p_size) {
		return;
	}

	shadow_atlas_size = p_size;
	RS::get_singleton()->viewport_set_shadow_atlas_size(viewport, p_size);
}

int Viewport::get_shadow_atlas_size() const {
	return shadow_atlas_size;
}

void Viewport::set_shadow_atlas_quadrant_subdiv(int p_quadrant, ShadowAtlasQuadrantSubdiv p_subdiv) {
	ERR_FAIL_INDEX(p_quadrant, 4);
	ERR_FAIL_INDEX(p_subdiv, SHADOW_ATLAS_QUADRANT_SUBDIV_MAX);

	if (shadow_atlas_quadrant_subdiv[p_quadrant] == p_subdiv) {
		return;
	}

	shadow_atlas_quadrant_subdiv[p_quadrant] = p_subdiv;
	static const int subdiv[SHADOW_ATLAS_QUADRANT_SUBDIV_MAX] = { 0, 1, 4, 16, 64, 256, 1024 };

	RS::get_singleton()->viewport_set_shadow_atlas_quadrant_subdivision(viewport, p_quadrant, subdiv[p_subdiv]);
}

Viewport::ShadowAtlasQuadrantSubdiv Viewport::get_shadow_atlas_quadrant_subdiv(int p_quadrant) const {
	ERR_FAIL_INDEX_V(p_quadrant, 4, SHADOW_ATLAS_QUADRANT_SUBDIV_DISABLED);
	return shadow_atlas_quadrant_subdiv[p_quadrant];
}

Transform2D Viewport::_get_input_pre_xform() const {
	Transform2D pre_xf;

	if (to_screen_rect.size.x != 0 && to_screen_rect.size.y != 0) {
		pre_xf.elements[2] = -to_screen_rect.position;
		pre_xf.scale(size / to_screen_rect.size);
	}

	return pre_xf;
}

Ref<InputEvent> Viewport::_make_input_local(const Ref<InputEvent> &ev) {
	Transform2D ai = get_final_transform().affine_inverse() * _get_input_pre_xform();

	return ev->xformed_by(ai);
}

Vector2 Viewport::get_mouse_position() const {
	return gui.last_mouse_pos;
}

void Viewport::warp_mouse(const Vector2 &p_pos) {
	Vector2 gpos = (get_final_transform().affine_inverse() * _get_input_pre_xform()).affine_inverse().xform(p_pos);
	Input::get_singleton()->warp_mouse_position(gpos);
}

void Viewport::_gui_sort_roots() {
	if (!gui.roots_order_dirty) {
		return;
	}

	gui.roots.sort_custom<Control::CComparator>();

	gui.roots_order_dirty = false;
}

void Viewport::_gui_cancel_tooltip() {
	gui.tooltip_control = nullptr;
	gui.tooltip_timer = -1;
	if (gui.tooltip_popup) {
		gui.tooltip_popup->queue_delete();
		gui.tooltip_popup = nullptr;
		gui.tooltip_label = nullptr;
	}
}

String Viewport::_gui_get_tooltip(Control *p_control, const Vector2 &p_pos, Control **r_tooltip_owner) {
	Vector2 pos = p_pos;
	String tooltip;

	while (p_control) {
		tooltip = p_control->get_tooltip(pos);

		if (r_tooltip_owner) {
			*r_tooltip_owner = p_control;
		}

		// If we found a tooltip, we stop here.
		if (!tooltip.empty()) {
			break;
		}

		// Otherwise, we check parent controls unless some conditions prevent it.

		if (p_control->data.mouse_filter == Control::MOUSE_FILTER_STOP) {
			break;
		}
		if (p_control->is_set_as_top_level()) {
			break;
		}

		// Transform cursor pos for parent control.
		pos = p_control->get_transform().xform(pos);

		p_control = p_control->get_parent_control();
	}

	return tooltip;
}

void Viewport::_gui_show_tooltip() {
	if (!gui.tooltip_control) {
		return;
	}

	// Get the Control under cursor and the relevant tooltip text, if any.
	Control *tooltip_owner = nullptr;
	String tooltip_text = _gui_get_tooltip(
			gui.tooltip_control,
			gui.tooltip_control->get_global_transform().xform_inv(gui.last_mouse_pos),
			&tooltip_owner);
	tooltip_text.strip_edges();
	if (tooltip_text.empty()) {
		return; // Nothing to show.
	}

	// Remove previous popup if we change something.
	if (gui.tooltip_popup) {
		memdelete(gui.tooltip_popup);
		gui.tooltip_popup = nullptr;
		gui.tooltip_label = nullptr;
	}

	if (!tooltip_owner) {
		return;
	}

	// Controls can implement `make_custom_tooltip` to provide their own tooltip.
	// This should be a Control node which will be added as child to a TooltipPanel.
	Control *base_tooltip = tooltip_owner->make_custom_tooltip(tooltip_text);

	// If no custom tooltip is given, use a default implementation.
	if (!base_tooltip) {
		gui.tooltip_label = memnew(TooltipLabel);
		gui.tooltip_label->set_text(tooltip_text);
		base_tooltip = gui.tooltip_label;
	}

	base_tooltip->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	TooltipPanel *panel = memnew(TooltipPanel);
	panel->set_transient(false);
	panel->set_flag(Window::FLAG_NO_FOCUS, true);
	panel->set_wrap_controls(true);
	panel->add_child(base_tooltip);

	gui.tooltip_popup = panel;

	tooltip_owner->add_child(gui.tooltip_popup);

	Point2 tooltip_offset = ProjectSettings::get_singleton()->get("display/mouse_cursor/tooltip_position_offset");
	Rect2 r(gui.tooltip_pos + tooltip_offset, gui.tooltip_popup->get_contents_minimum_size());

	Rect2i vr = gui.tooltip_popup->get_parent_visible_window()->get_usable_parent_rect();

	if (r.size.x + r.position.x > vr.size.x + vr.position.x) {
		r.position.x = vr.position.x + vr.size.x - r.size.x;
	} else if (r.position.x < vr.position.x) {
		r.position.x = vr.position.x;
	}

	if (r.size.y + r.position.y > vr.size.y + vr.position.y) {
		r.position.y = vr.position.y + vr.size.y - r.size.y;
	} else if (r.position.y < vr.position.y) {
		r.position.y = vr.position.y;
	}

	gui.tooltip_popup->set_position(r.position);
	gui.tooltip_popup->set_size(r.size);

	gui.tooltip_popup->show();
	gui.tooltip_popup->child_controls_changed();
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
	Ref<InputEventPanGesture> pn = p_input;
	cant_stop_me_now = pn.is_valid() || cant_stop_me_now;

	bool ismouse = ev.is_valid() || Object::cast_to<InputEventMouseMotion>(*p_input) != nullptr;

	CanvasItem *ci = p_control;
	while (ci) {
		Control *control = Object::cast_to<Control>(ci);
		if (control) {
			if (control->data.mouse_filter != Control::MOUSE_FILTER_IGNORE) {
				control->emit_signal(SceneStringNames::get_singleton()->gui_input, ev); //signal should be first, so it's possible to override an event (and then accept it)
			}
			if (gui.key_event_accepted) {
				break;
			}
			if (!control->is_inside_tree()) {
				break;
			}

			if (control->data.mouse_filter != Control::MOUSE_FILTER_IGNORE) {
				// Call both script and native methods.
				Callable::CallError error;
				Variant event = ev;
				const Variant *args[1] = { &event };
				if (control->get_script_instance()) {
					control->get_script_instance()->call(SceneStringNames::get_singleton()->_gui_input, args, 1, error);
				}
				MethodBind *method = ClassDB::get_method(control->get_class_name(), SceneStringNames::get_singleton()->_gui_input);
				if (method) {
					method->call(control, args, 1, error);
				}
			}

			if (!control->is_inside_tree() || control->is_set_as_top_level()) {
				break;
			}
			if (gui.key_event_accepted) {
				break;
			}
			if (!cant_stop_me_now && control->data.mouse_filter == Control::MOUSE_FILTER_STOP && ismouse) {
				break;
			}
		}

		if (ci->is_set_as_top_level()) {
			break;
		}

		ev = ev->xformed_by(ci->get_transform()); //transform event upwards
		ci = ci->get_parent_item();
	}

	//_unblock();
}

void Viewport::_gui_call_notification(Control *p_control, int p_what) {
	CanvasItem *ci = p_control;
	while (ci) {
		Control *control = Object::cast_to<Control>(ci);
		if (control) {
			if (control->data.mouse_filter != Control::MOUSE_FILTER_IGNORE) {
				control->notification(p_what);
			}

			if (!control->is_inside_tree()) {
				break;
			}

			if (!control->is_inside_tree() || control->is_set_as_top_level()) {
				break;
			}
			if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP) {
				break;
			}
		}

		if (ci->is_set_as_top_level()) {
			break;
		}

		ci = ci->get_parent_item();
	}

	//_unblock();
}

Control *Viewport::_gui_find_control(const Point2 &p_global) {
	//aca va subwindows
	_gui_sort_roots();

	for (List<Control *>::Element *E = gui.roots.back(); E; E = E->prev()) {
		Control *sw = E->get();
		if (!sw->is_visible_in_tree()) {
			continue;
		}

		Transform2D xform;
		CanvasItem *pci = sw->get_parent_item();
		if (pci) {
			xform = pci->get_global_transform_with_canvas();
		} else {
			xform = sw->get_canvas_transform();
		}

		Control *ret = _gui_find_control_at_pos(sw, p_global, xform, gui.focus_inv_xform);
		if (ret) {
			return ret;
		}
	}

	return nullptr;
}

Control *Viewport::_gui_find_control_at_pos(CanvasItem *p_node, const Point2 &p_global, const Transform2D &p_xform, Transform2D &r_inv_xform) {
	if (Object::cast_to<Viewport>(p_node)) {
		return nullptr;
	}

	if (!p_node->is_visible()) {
		//return _find_next_visible_control_at_pos(p_node,p_global,r_inv_xform);
		return nullptr; //canvas item hidden, discard
	}

	Transform2D matrix = p_xform * p_node->get_transform();
	// matrix.basis_determinant() == 0.0f implies that node does not exist on scene
	if (matrix.basis_determinant() == 0.0f) {
		return nullptr;
	}

	Control *c = Object::cast_to<Control>(p_node);

	if (!c || !c->clips_input() || c->has_point(matrix.affine_inverse().xform(p_global))) {
		for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
			CanvasItem *ci = Object::cast_to<CanvasItem>(p_node->get_child(i));
			if (!ci || ci->is_set_as_top_level()) {
				continue;
			}

			Control *ret = _gui_find_control_at_pos(ci, p_global, matrix, r_inv_xform);
			if (ret) {
				return ret;
			}
		}
	}

	if (!c) {
		return nullptr;
	}

	matrix.affine_invert();

	//conditions for considering this as a valid control for return
	if (c->data.mouse_filter != Control::MOUSE_FILTER_IGNORE && c->has_point(matrix.xform(p_global)) && (!gui.drag_preview || (c != gui.drag_preview && !gui.drag_preview->is_a_parent_of(c)))) {
		r_inv_xform = matrix;
		return c;
	} else {
		return nullptr;
	}
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

				if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP) {
					break;
				}
			}

			p_at_pos = ci->get_transform().xform(p_at_pos);

			if (ci->is_set_as_top_level()) {
				break;
			}

			ci = ci->get_parent_item();
		}
	}

	return false;
}

void Viewport::_gui_input_event(Ref<InputEvent> p_event) {
	ERR_FAIL_COND(p_event.is_null());

	//?
	/*
	if (!is_visible()) {
		return; //simple and plain
	}
	*/

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		gui.key_event_accepted = false;

		Control *over = nullptr;

		Point2 mpos = mb->get_position();
		if (mb->is_pressed()) {
			Size2 pos = mpos;
			if (gui.mouse_focus_mask) {
				//do not steal mouse focus and stuff while a focus mask exists
				gui.mouse_focus_mask |= 1 << (mb->get_button_index() - 1); //add the button to the mask
			} else {
				bool is_handled = false;

				if (is_handled) {
					set_input_as_handled();
					return;
				}

				//Matrix32 parent_xform;

				/*
				if (data.parent_canvas_item)
					parent_xform=data.parent_canvas_item->get_global_transform();
				*/

				gui.mouse_focus = _gui_find_control(pos);
				gui.last_mouse_focus = gui.mouse_focus;

				if (!gui.mouse_focus) {
					gui.mouse_focus_mask = 0;
					return;
				}

				gui.mouse_focus_mask = 1 << (mb->get_button_index() - 1);

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
			if (EngineDebugger::get_singleton() && gui.mouse_focus) {
				Array arr;
				arr.push_back(gui.mouse_focus->get_path());
				arr.push_back(gui.mouse_focus->get_class());
				EngineDebugger::get_singleton()->send_message("scene:click_ctrl", arr);
			}
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

						if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP) {
							break;
						}
					}

					if (ci->is_set_as_top_level()) {
						break;
					}

					ci = ci->get_parent_item();
				}
			}

			if (gui.mouse_focus && gui.mouse_focus->can_process()) {
				_gui_call_input(gui.mouse_focus, mb);
			}

			set_input_as_handled();

			if (gui.drag_data.get_type() != Variant::NIL && mb->get_button_index() == BUTTON_LEFT) {
				//alternate drop use (when using force_drag(), as proposed by #5342
				if (gui.mouse_focus) {
					_gui_drop(gui.mouse_focus, pos, false);
				}

				gui.drag_data = Variant();
				gui.dragging = false;

				if (gui.drag_preview) {
					memdelete(gui.drag_preview);
					gui.drag_preview = nullptr;
				}
				_propagate_viewport_notification(this, NOTIFICATION_DRAG_END);
				//change mouse accordingly
			}

			_gui_cancel_tooltip();
		} else {
			if (gui.drag_data.get_type() != Variant::NIL && mb->get_button_index() == BUTTON_LEFT) {
				if (gui.drag_mouse_over) {
					_gui_drop(gui.drag_mouse_over, gui.drag_mouse_over_pos, false);
				}

				if (gui.drag_preview && mb->get_button_index() == BUTTON_LEFT) {
					memdelete(gui.drag_preview);
					gui.drag_preview = nullptr;
				}

				gui.drag_data = Variant();
				gui.dragging = false;
				gui.drag_mouse_over = nullptr;
				_propagate_viewport_notification(this, NOTIFICATION_DRAG_END);
				//change mouse accordingly
			}

			gui.mouse_focus_mask &= ~(1 << (mb->get_button_index() - 1)); //remove from mask

			if (!gui.mouse_focus) {
				//release event is only sent if a mouse focus (previously pressed button) exists
				return;
			}

			Size2 pos = mpos;

			mb = mb->xformed_by(Transform2D()); //make a copy
			mb->set_global_position(pos);
			pos = gui.focus_inv_xform.xform(pos);
			mb->set_position(pos);

			Control *mouse_focus = gui.mouse_focus;

			//disable mouse focus if needed before calling input, this makes popups on mouse press event work better, as the release will never be received otherwise
			if (gui.mouse_focus_mask == 0) {
				gui.mouse_focus = nullptr;
				gui.forced_mouse_focus = false;
			}

			if (mouse_focus && mouse_focus->can_process()) {
				_gui_call_input(mouse_focus, mb);
			}

			/*if (gui.drag_data.get_type()!=Variant::NIL && mb->get_button_index()==BUTTON_LEFT) {
				_propagate_viewport_notification(this,NOTIFICATION_DRAG_END);
				gui.drag_data=Variant(); //always clear
			}*/

			// In case the mouse was released after for example dragging a scrollbar,
			// check whether the current control is different from the stored one. If
			// it is different, rather than wait for it to be updated the next time the
			// mouse is moved, notify the control so that it can e.g. drop the highlight.
			// This code is duplicated from the mm.is_valid()-case further below.
			if (gui.mouse_focus) {
				over = gui.mouse_focus;
			} else {
				over = _gui_find_control(mpos);
			}

			if (gui.mouse_focus_mask == 0 && over != gui.mouse_over) {
				if (gui.mouse_over) {
					_gui_call_notification(gui.mouse_over, Control::NOTIFICATION_MOUSE_EXIT);
				}

				_gui_cancel_tooltip();

				if (over) {
					_gui_call_notification(over, Control::NOTIFICATION_MOUSE_ENTER);
				}
			}

			gui.mouse_over = over;

			set_input_as_handled();
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		gui.key_event_accepted = false;
		Point2 mpos = mm->get_position();

		gui.last_mouse_pos = mpos;

		Control *over = nullptr;

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
							gui.dragging = true;
							gui.drag_data = control->get_drag_data(control->get_global_transform_with_canvas().affine_inverse().xform(mpos) - gui.drag_accum);
							if (gui.drag_data.get_type() != Variant::NIL) {
								gui.mouse_focus = nullptr;
								gui.forced_mouse_focus = false;
								gui.mouse_focus_mask = 0;
								break;
							} else {
								if (gui.drag_preview != nullptr) {
									ERR_PRINT("Don't set a drag preview and return null data. Preview was deleted and drag request ignored.");
									memdelete(gui.drag_preview);
									gui.drag_preview = nullptr;
								}
								gui.dragging = false;
							}

							if (control->data.mouse_filter == Control::MOUSE_FILTER_STOP) {
								break;
							}
						}

						if (ci->is_set_as_top_level()) {
							break;
						}

						ci = ci->get_parent_item();
					}
				}

				gui.drag_attempted = true;
				if (gui.drag_data.get_type() != Variant::NIL) {
					_propagate_viewport_notification(this, NOTIFICATION_DRAG_BEGIN);
				}
			}
		}

		// These sections of code are reused in the mb.is_valid() case further up
		// for the purpose of notifying controls about potential changes in focus
		// when the mousebutton is released.
		if (gui.mouse_focus) {
			over = gui.mouse_focus;
		} else {
			over = _gui_find_control(mpos);
		}

		if (over != gui.mouse_over) {
			if (gui.mouse_over) {
				_gui_call_notification(gui.mouse_over, Control::NOTIFICATION_MOUSE_EXIT);
			}

			_gui_cancel_tooltip();

			if (over) {
				_gui_call_notification(over, Control::NOTIFICATION_MOUSE_ENTER);
			}
		}

		gui.mouse_over = over;

		DisplayServer::CursorShape ds_cursor_shape = (DisplayServer::CursorShape)Input::get_singleton()->get_default_cursor_shape();

		if (over) {
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

				bool is_tooltip_shown = false;

				if (gui.tooltip_popup) {
					if (can_tooltip && gui.tooltip_control) {
						String tooltip = _gui_get_tooltip(over, gui.tooltip_control->get_global_transform().xform_inv(mpos));

						if (tooltip.length() == 0) {
							_gui_cancel_tooltip();
						} else if (gui.tooltip_label) {
							if (tooltip == gui.tooltip_label->get_text()) {
								is_tooltip_shown = true;
							}
						} else {
							Variant t = gui.tooltip_popup->call("get_tooltip_text");

							if (t.get_type() == Variant::STRING) {
								if (tooltip == String(t)) {
									is_tooltip_shown = true;
								}
							} else {
								is_tooltip_shown = true; //well, nothing to compare against, likely using custom control, so if it changes there is nothing we can do
							}
						}
					} else {
						_gui_cancel_tooltip();
					}
				}

				if (can_tooltip && !is_tooltip_shown) {
					gui.tooltip_control = over;
					gui.tooltip_pos = over->get_screen_transform().xform(pos);
					gui.tooltip_timer = gui.tooltip_delay;
				}
			}

			mm->set_position(pos);

			Control::CursorShape cursor_shape = Control::CURSOR_ARROW;
			{
				Control *c = over;
				Vector2 cpos = pos;
				while (c) {
					if (gui.mouse_focus_mask != 0 || c->has_point(cpos)) {
						cursor_shape = c->get_cursor_shape(cpos);
					} else {
						cursor_shape = Control::CURSOR_ARROW;
					}
					cpos = c->get_transform().xform(cpos);
					if (cursor_shape != Control::CURSOR_ARROW) {
						break;
					}
					if (c->data.mouse_filter == Control::MOUSE_FILTER_STOP) {
						break;
					}
					if (c->is_set_as_top_level()) {
						break;
					}
					c = c->get_parent_control();
				}
			}

			ds_cursor_shape = (DisplayServer::CursorShape)cursor_shape;

			if (over && over->can_process()) {
				_gui_call_input(over, mm);
			}

			set_input_as_handled();
		}

		if (gui.drag_data.get_type() != Variant::NIL) {
			//handle dragandrop

			if (gui.drag_preview) {
				gui.drag_preview->set_position(mpos);
			}

			gui.drag_mouse_over = over;
			gui.drag_mouse_over_pos = Vector2();

			//find the window this is above of

			//see if there is an embedder
			Viewport *embedder = nullptr;
			Vector2 viewport_pos;

			if (is_embedding_subwindows()) {
				embedder = this;
				viewport_pos = mpos;
			} else {
				//not an embeder, but may be a subwindow of an embedder
				Window *w = Object::cast_to<Window>(this);
				if (w) {
					if (w->is_embedded()) {
						embedder = w->_get_embedder();

						Transform2D ai = (get_final_transform().affine_inverse() * _get_input_pre_xform()).affine_inverse();

						viewport_pos = ai.xform(mpos) + w->get_position(); //to parent coords
					}
				}
			}

			Viewport *viewport_under = nullptr;

			if (embedder) {
				//use embedder logic

				for (int i = embedder->gui.sub_windows.size() - 1; i >= 0; i--) {
					Window *sw = embedder->gui.sub_windows[i].window;
					Rect2 swrect = Rect2i(sw->get_position(), sw->get_size());
					if (!sw->get_flag(Window::FLAG_BORDERLESS)) {
						int title_height = sw->get_theme_constant("title_height");
						swrect.position.y -= title_height;
						swrect.size.y += title_height;
					}

					if (swrect.has_point(viewport_pos)) {
						viewport_under = sw;
						viewport_pos -= sw->get_position();
					}
				}

				if (!viewport_under) {
					//not in a subwindow, likely in embedder
					viewport_under = embedder;
				}
			} else {
				//use displayserver logic
				Vector2i screen_mouse_pos = DisplayServer::get_singleton()->mouse_get_position();

				DisplayServer::WindowID window_id = DisplayServer::get_singleton()->get_window_at_screen_position(screen_mouse_pos);

				if (window_id != DisplayServer::INVALID_WINDOW_ID) {
					ObjectID object_under = DisplayServer::get_singleton()->window_get_attached_instance_id(window_id);

					if (object_under != ObjectID()) { //fetch window
						Window *w = Object::cast_to<Window>(ObjectDB::get_instance(object_under));
						if (w) {
							viewport_under = w;
							viewport_pos = screen_mouse_pos - w->get_position();
						}
					}
				}
			}

			if (viewport_under) {
				Transform2D ai = (viewport_under->get_final_transform().affine_inverse() * viewport_under->_get_input_pre_xform());
				viewport_pos = ai.xform(viewport_pos);
				//find control under at pos
				gui.drag_mouse_over = viewport_under->_gui_find_control(viewport_pos);
				if (gui.drag_mouse_over) {
					Transform2D localizer = gui.drag_mouse_over->get_global_transform_with_canvas().affine_inverse();
					gui.drag_mouse_over_pos = localizer.xform(viewport_pos);

					if (mm->get_button_mask() & BUTTON_MASK_LEFT) {
						bool can_drop = _gui_drop(gui.drag_mouse_over, gui.drag_mouse_over_pos, true);

						if (!can_drop) {
							ds_cursor_shape = DisplayServer::CURSOR_FORBIDDEN;
						} else {
							ds_cursor_shape = DisplayServer::CURSOR_CAN_DROP;
						}
					}
				}

			} else {
				gui.drag_mouse_over = nullptr;
			}
		}

		DisplayServer::get_singleton()->cursor_set_shape(ds_cursor_shape);
	}

	Ref<InputEventScreenTouch> touch_event = p_event;
	if (touch_event.is_valid()) {
		Size2 pos = touch_event->get_position();
		if (touch_event->is_pressed()) {
			Control *over = _gui_find_control(pos);
			if (over) {
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
				set_input_as_handled();
				return;
			}
		} else if (touch_event->get_index() == 0 && gui.last_mouse_focus) {
			if (gui.last_mouse_focus->can_process()) {
				touch_event = touch_event->xformed_by(Transform2D()); //make a copy
				touch_event->set_position(gui.focus_inv_xform.xform(pos));

				_gui_call_input(gui.last_mouse_focus, touch_event);
			}
			set_input_as_handled();
			return;
		}
	}

	Ref<InputEventGesture> gesture_event = p_event;
	if (gesture_event.is_valid()) {
		gui.key_event_accepted = false;

		_gui_cancel_tooltip();

		Size2 pos = gesture_event->get_position();

		Control *over = _gui_find_control(pos);
		if (over) {
			if (over->can_process()) {
				gesture_event = gesture_event->xformed_by(Transform2D()); //make a copy
				if (over == gui.mouse_focus) {
					pos = gui.focus_inv_xform.xform(pos);
				} else {
					pos = over->get_global_transform_with_canvas().affine_inverse().xform(pos);
				}
				gesture_event->set_position(pos);
				_gui_call_input(over, gesture_event);
			}
			set_input_as_handled();
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

			set_input_as_handled();
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
				gui.key_focus->call(SceneStringNames::get_singleton()->_gui_input, p_event);
				if (gui.key_focus) { //maybe lost it
					gui.key_focus->emit_signal(SceneStringNames::get_singleton()->gui_input, p_event);
				}
			}

			if (gui.key_event_accepted) {
				set_input_as_handled();
				return;
			}
		}

		Control *from = gui.key_focus ? gui.key_focus : nullptr; //hmm

		//keyboard focus
		//if (from && p_event->is_pressed() && !p_event->get_alt() && !p_event->get_metakey() && !p_event->key->get_command()) {

		Ref<InputEventKey> k = p_event;
		//need to check for mods, otherwise any combination of alt/ctrl/shift+<up/down/left/righ/etc> is handled here when it shouldn't be.
		bool mods = k.is_valid() && (k->get_control() || k->get_alt() || k->get_shift() || k->get_metakey());

		if (from && p_event->is_pressed()) {
			Control *next = nullptr;

			Input *input = Input::get_singleton();

			if (p_event->is_action_pressed("ui_focus_next") && input->is_action_just_pressed("ui_focus_next")) {
				next = from->find_next_valid_focus();
			}

			if (p_event->is_action_pressed("ui_focus_prev") && input->is_action_just_pressed("ui_focus_prev")) {
				next = from->find_prev_valid_focus();
			}

			if (!mods && p_event->is_action_pressed("ui_up") && input->is_action_just_pressed("ui_up")) {
				next = from->_get_focus_neighbour(MARGIN_TOP);
			}

			if (!mods && p_event->is_action_pressed("ui_left") && input->is_action_just_pressed("ui_left")) {
				next = from->_get_focus_neighbour(MARGIN_LEFT);
			}

			if (!mods && p_event->is_action_pressed("ui_right") && input->is_action_just_pressed("ui_right")) {
				next = from->_get_focus_neighbour(MARGIN_RIGHT);
			}

			if (!mods && p_event->is_action_pressed("ui_down") && input->is_action_just_pressed("ui_down")) {
				next = from->_get_focus_neighbour(MARGIN_BOTTOM);
			}

			if (next) {
				next->grab_focus();
				set_input_as_handled();
			}
		}
	}
}

List<Control *>::Element *Viewport::_gui_add_root_control(Control *p_control) {
	gui.roots_order_dirty = true;
	return gui.roots.push_back(p_control);
}

void Viewport::_gui_set_root_order_dirty() {
	gui.roots_order_dirty = true;
}

void Viewport::_gui_force_drag(Control *p_base, const Variant &p_data, Control *p_control) {
	ERR_FAIL_COND_MSG(p_data.get_type() == Variant::NIL, "Drag data must be a value.");

	gui.dragging = true;
	gui.drag_data = p_data;
	gui.mouse_focus = nullptr;

	if (p_control) {
		_gui_set_drag_preview(p_base, p_control);
	}
}

void Viewport::_gui_set_drag_preview(Control *p_base, Control *p_control) {
	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND(!Object::cast_to<Control>((Object *)p_control));
	ERR_FAIL_COND(p_control->is_inside_tree());
	ERR_FAIL_COND(p_control->get_parent() != nullptr);

	if (gui.drag_preview) {
		memdelete(gui.drag_preview);
	}
	p_control->set_as_top_level(true);
	p_control->set_position(gui.last_mouse_pos);
	p_base->get_root_parent_control()->add_child(p_control); //add as child of viewport
	p_control->raise();

	gui.drag_preview = p_control;
}

void Viewport::_gui_remove_root_control(List<Control *>::Element *RI) {
	gui.roots.erase(RI);
}

void Viewport::_gui_unfocus_control(Control *p_control) {
	if (gui.key_focus == p_control) {
		gui.key_focus->release_focus();
	}
}

void Viewport::_gui_hide_control(Control *p_control) {
	if (gui.mouse_focus == p_control) {
		_drop_mouse_focus();
	}

	if (gui.key_focus == p_control) {
		_gui_remove_focus();
	}
	if (gui.mouse_over == p_control) {
		gui.mouse_over = nullptr;
	}
	if (gui.drag_mouse_over == p_control) {
		gui.drag_mouse_over = nullptr;
	}
	if (gui.tooltip_control == p_control) {
		_gui_cancel_tooltip();
	}
}

void Viewport::_gui_remove_control(Control *p_control) {
	if (gui.mouse_focus == p_control) {
		gui.mouse_focus = nullptr;
		gui.forced_mouse_focus = false;
		gui.mouse_focus_mask = 0;
	}
	if (gui.last_mouse_focus == p_control) {
		gui.last_mouse_focus = nullptr;
	}
	if (gui.key_focus == p_control) {
		gui.key_focus = nullptr;
	}
	if (gui.mouse_over == p_control) {
		gui.mouse_over = nullptr;
	}
	if (gui.drag_mouse_over == p_control) {
		gui.drag_mouse_over = nullptr;
	}
	if (gui.tooltip_control == p_control) {
		gui.tooltip_control = nullptr;
	}
}

Window *Viewport::get_base_window() const {
	Viewport *v = const_cast<Viewport *>(this);
	Window *w = Object::cast_to<Window>(v);
	while (!w) {
		v = v->get_parent_viewport();
		w = Object::cast_to<Window>(v);
	}

	return w;
}
void Viewport::_gui_remove_focus_for_window(Node *p_window) {
	if (get_base_window() == p_window) {
		_gui_remove_focus();
	}
}

void Viewport::_gui_remove_focus() {
	if (gui.key_focus) {
		Node *f = gui.key_focus;
		gui.key_focus = nullptr;
		f->notification(Control::NOTIFICATION_FOCUS_EXIT, true);
	}
}

bool Viewport::_gui_control_has_focus(const Control *p_control) {
	return gui.key_focus == p_control;
}

void Viewport::_gui_control_grab_focus(Control *p_control) {
	//no need for change
	if (gui.key_focus && gui.key_focus == p_control) {
		return;
	}
	get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, "_viewports", "_gui_remove_focus_for_window", (Node *)get_base_window());
	gui.key_focus = p_control;
	emit_signal("gui_focus_changed", p_control);
	p_control->notification(Control::NOTIFICATION_FOCUS_ENTER);
	p_control->update();
}

void Viewport::_gui_accept_event() {
	gui.key_event_accepted = true;
	if (is_inside_tree()) {
		set_input_as_handled();
	}
}

void Viewport::_drop_mouse_focus() {
	Control *c = gui.mouse_focus;
	int mask = gui.mouse_focus_mask;
	gui.mouse_focus = nullptr;
	gui.forced_mouse_focus = false;
	gui.mouse_focus_mask = 0;

	for (int i = 0; i < 3; i++) {
		if (mask & (1 << i)) {
			Ref<InputEventMouseButton> mb;
			mb.instance();
			mb->set_position(c->get_local_mouse_position());
			mb->set_global_position(c->get_local_mouse_position());
			mb->set_button_index(i + 1);
			mb->set_pressed(false);
			c->call(SceneStringNames::get_singleton()->_gui_input, mb);
		}
	}
}

void Viewport::_drop_physics_mouseover() {
	physics_has_last_mousepos = false;

	while (physics_2d_mouseover.size()) {
		Object *o = ObjectDB::get_instance(physics_2d_mouseover.front()->key());
		if (o) {
			CollisionObject2D *co = Object::cast_to<CollisionObject2D>(o);
			co->_mouse_exit();
		}
		physics_2d_mouseover.erase(physics_2d_mouseover.front());
	}

#ifndef _3D_DISABLED
	if (physics_object_over.is_valid()) {
		CollisionObject3D *co = Object::cast_to<CollisionObject3D>(ObjectDB::get_instance(physics_object_over));
		if (co) {
			co->_mouse_exit();
		}
	}

	physics_object_over = ObjectID();
	physics_object_capture = ObjectID();
#endif
}

Control *Viewport::_gui_get_focus_owner() {
	return gui.key_focus;
}

void Viewport::_gui_grab_click_focus(Control *p_control) {
	gui.mouse_click_grabber = p_control;
	call_deferred("_post_gui_grab_click_focus");
}

void Viewport::_post_gui_grab_click_focus() {
	Control *focus_grabber = gui.mouse_click_grabber;
	if (!focus_grabber) {
		// Redundant grab requests were made
		return;
	}
	gui.mouse_click_grabber = nullptr;

	if (gui.mouse_focus) {
		if (gui.mouse_focus == focus_grabber) {
			return;
		}

		int mask = gui.mouse_focus_mask;
		Point2 click = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);

		for (int i = 0; i < 3; i++) {
			if (mask & (1 << i)) {
				Ref<InputEventMouseButton> mb;
				mb.instance();

				//send unclic

				mb->set_position(click);
				mb->set_button_index(i + 1);
				mb->set_pressed(false);
				gui.mouse_focus->call(SceneStringNames::get_singleton()->_gui_input, mb);
			}
		}

		gui.mouse_focus = focus_grabber;
		gui.focus_inv_xform = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse();
		click = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);

		for (int i = 0; i < 3; i++) {
			if (mask & (1 << i)) {
				Ref<InputEventMouseButton> mb;
				mb.instance();

				//send clic

				mb->set_position(click);
				mb->set_button_index(i + 1);
				mb->set_pressed(true);
				gui.mouse_focus->call_deferred(SceneStringNames::get_singleton()->_gui_input, mb);
			}
		}
	}
}

///////////////////////////////

void Viewport::input_text(const String &p_text) {
	if (gui.subwindow_focused) {
		gui.subwindow_focused->input_text(p_text);
		return;
	}

	if (gui.key_focus) {
		gui.key_focus->call("set_text", p_text);
	}
}

Viewport::SubWindowResize Viewport::_sub_window_get_resize_margin(Window *p_subwindow, const Point2 &p_point) {
	if (p_subwindow->get_flag(Window::FLAG_BORDERLESS)) {
		return SUB_WINDOW_RESIZE_DISABLED;
	}

	Rect2i r = Rect2i(p_subwindow->get_position(), p_subwindow->get_size());

	int title_height = p_subwindow->get_theme_constant("title_height");

	r.position.y -= title_height;
	r.size.y += title_height;

	if (r.has_point(p_point)) {
		return SUB_WINDOW_RESIZE_DISABLED; //it's inside, so no resize
	}

	int dist_x = p_point.x < r.position.x ? (p_point.x - r.position.x) : (p_point.x > (r.position.x + r.size.x) ? (p_point.x - (r.position.x + r.size.x)) : 0);
	int dist_y = p_point.y < r.position.y ? (p_point.y - r.position.y) : (p_point.y > (r.position.y + r.size.y) ? (p_point.y - (r.position.y + r.size.y)) : 0);

	int limit = p_subwindow->get_theme_constant("resize_margin");

	if (ABS(dist_x) > limit) {
		return SUB_WINDOW_RESIZE_DISABLED;
	}

	if (ABS(dist_y) > limit) {
		return SUB_WINDOW_RESIZE_DISABLED;
	}

	if (dist_x < 0 && dist_y < 0) {
		return SUB_WINDOW_RESIZE_TOP_LEFT;
	}

	if (dist_x == 0 && dist_y < 0) {
		return SUB_WINDOW_RESIZE_TOP;
	}

	if (dist_x > 0 && dist_y < 0) {
		return SUB_WINDOW_RESIZE_TOP_RIGHT;
	}

	if (dist_x < 0 && dist_y == 0) {
		return SUB_WINDOW_RESIZE_LEFT;
	}

	if (dist_x > 0 && dist_y == 0) {
		return SUB_WINDOW_RESIZE_RIGHT;
	}

	if (dist_x < 0 && dist_y > 0) {
		return SUB_WINDOW_RESIZE_BOTTOM_LEFT;
	}

	if (dist_x == 0 && dist_y > 0) {
		return SUB_WINDOW_RESIZE_BOTTOM;
	}

	if (dist_x > 0 && dist_y > 0) {
		return SUB_WINDOW_RESIZE_BOTTOM_RIGHT;
	}

	return SUB_WINDOW_RESIZE_DISABLED;
}

bool Viewport::_sub_windows_forward_input(const Ref<InputEvent> &p_event) {
	if (gui.subwindow_drag != SUB_WINDOW_DRAG_DISABLED) {
		ERR_FAIL_COND_V(gui.subwindow_focused == nullptr, false);

		Ref<InputEventMouseButton> mb = p_event;
		if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
			if (gui.subwindow_drag == SUB_WINDOW_DRAG_CLOSE) {
				if (gui.subwindow_drag_close_rect.has_point(mb->get_position())) {
					//close window
					gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_CLOSE_REQUEST);
				}
			}
			gui.subwindow_drag = SUB_WINDOW_DRAG_DISABLED;
			if (gui.subwindow_focused != nullptr) { //may have been erased
				_sub_window_update(gui.subwindow_focused);
			}
		}

		Ref<InputEventMouseMotion> mm = p_event;
		if (mm.is_valid()) {
			if (gui.subwindow_drag == SUB_WINDOW_DRAG_MOVE) {
				Vector2 diff = mm->get_position() - gui.subwindow_drag_from;
				Rect2i new_rect(gui.subwindow_drag_pos + diff, gui.subwindow_focused->get_size());

				if (gui.subwindow_focused->is_clamped_to_embedder()) {
					Size2i limit = get_visible_rect().size;
					if (new_rect.position.x + new_rect.size.x > limit.x) {
						new_rect.position.x = limit.x - new_rect.size.x;
					}
					if (new_rect.position.y + new_rect.size.y > limit.y) {
						new_rect.position.y = limit.y - new_rect.size.y;
					}

					if (new_rect.position.x < 0) {
						new_rect.position.x = 0;
					}

					int title_height = gui.subwindow_focused->get_flag(Window::FLAG_BORDERLESS) ? 0 : gui.subwindow_focused->get_theme_constant("title_height");

					if (new_rect.position.y < title_height) {
						new_rect.position.y = title_height;
					}
				}

				gui.subwindow_focused->_rect_changed_callback(new_rect);
			}
			if (gui.subwindow_drag == SUB_WINDOW_DRAG_CLOSE) {
				gui.subwindow_drag_close_inside = gui.subwindow_drag_close_rect.has_point(mm->get_position());
			}
			if (gui.subwindow_drag == SUB_WINDOW_DRAG_RESIZE) {
				Vector2i diff = mm->get_position() - gui.subwindow_drag_from;
				Size2i min_size = gui.subwindow_focused->get_min_size();
				if (gui.subwindow_focused->is_wrapping_controls()) {
					Size2i cms = gui.subwindow_focused->get_contents_minimum_size();
					min_size.x = MAX(cms.x, min_size.x);
					min_size.y = MAX(cms.y, min_size.y);
				}
				min_size.x = MAX(min_size.x, 1);
				min_size.y = MAX(min_size.y, 1);

				Rect2i r = gui.subwindow_resize_from_rect;

				Size2i limit = r.size - min_size;

				switch (gui.subwindow_resize_mode) {
					case SUB_WINDOW_RESIZE_TOP_LEFT: {
						diff.x = MIN(diff.x, limit.x);
						diff.y = MIN(diff.y, limit.y);
						r.position += diff;
						r.size -= diff;
					} break;
					case SUB_WINDOW_RESIZE_TOP: {
						diff.x = 0;
						diff.y = MIN(diff.y, limit.y);
						r.position += diff;
						r.size -= diff;
					} break;
					case SUB_WINDOW_RESIZE_TOP_RIGHT: {
						diff.x = MAX(diff.x, -limit.x);
						diff.y = MIN(diff.y, limit.y);
						r.position.y += diff.y;
						r.size.y -= diff.y;
						r.size.x += diff.x;
					} break;
					case SUB_WINDOW_RESIZE_LEFT: {
						diff.x = MIN(diff.x, limit.x);
						diff.y = 0;
						r.position += diff;
						r.size -= diff;

					} break;
					case SUB_WINDOW_RESIZE_RIGHT: {
						diff.x = MAX(diff.x, -limit.x);
						r.size.x += diff.x;
					} break;
					case SUB_WINDOW_RESIZE_BOTTOM_LEFT: {
						diff.x = MIN(diff.x, limit.x);
						diff.y = MAX(diff.y, -limit.y);
						r.position.x += diff.x;
						r.size.x -= diff.x;
						r.size.y += diff.y;

					} break;
					case SUB_WINDOW_RESIZE_BOTTOM: {
						diff.y = MAX(diff.y, -limit.y);
						r.size.y += diff.y;
					} break;
					case SUB_WINDOW_RESIZE_BOTTOM_RIGHT: {
						diff.x = MAX(diff.x, -limit.x);
						diff.y = MAX(diff.y, -limit.y);
						r.size += diff;

					} break;
					default: {
					}
				}

				gui.subwindow_focused->_rect_changed_callback(r);
			}

			if (gui.subwindow_focused) { //may have been erased
				_sub_window_update(gui.subwindow_focused);
			}
		}

		return true; //handled
	}
	Ref<InputEventMouseButton> mb = p_event;
	//if the event is a mouse button, we need to check whether another window was clicked

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		bool click_on_window = false;
		for (int i = gui.sub_windows.size() - 1; i >= 0; i--) {
			SubWindow &sw = gui.sub_windows.write[i];

			//clicked inside window?

			Rect2i r = Rect2i(sw.window->get_position(), sw.window->get_size());

			if (!sw.window->get_flag(Window::FLAG_BORDERLESS)) {
				//check top bar
				int title_height = sw.window->get_theme_constant("title_height");
				Rect2i title_bar = r;
				title_bar.position.y -= title_height;
				title_bar.size.y = title_height;

				if (title_bar.has_point(mb->get_position())) {
					click_on_window = true;

					int close_h_ofs = sw.window->get_theme_constant("close_h_ofs");
					int close_v_ofs = sw.window->get_theme_constant("close_v_ofs");
					Ref<Texture2D> close_icon = sw.window->get_theme_icon("close");

					Rect2 close_rect;
					close_rect.position = Vector2(r.position.x + r.size.x - close_v_ofs, r.position.y - close_h_ofs);
					close_rect.size = close_icon->get_size();

					if (gui.subwindow_focused != sw.window) {
						//refocus
						_sub_window_grab_focus(sw.window);
					}

					if (close_rect.has_point(mb->get_position())) {
						gui.subwindow_drag = SUB_WINDOW_DRAG_CLOSE;
						gui.subwindow_drag_close_inside = true; //starts inside
						gui.subwindow_drag_close_rect = close_rect;
					} else {
						gui.subwindow_drag = SUB_WINDOW_DRAG_MOVE;
					}

					gui.subwindow_drag_from = mb->get_position();
					gui.subwindow_drag_pos = sw.window->get_position();

					_sub_window_update(sw.window);
				} else {
					gui.subwindow_resize_mode = _sub_window_get_resize_margin(sw.window, mb->get_position());
					if (gui.subwindow_resize_mode != SUB_WINDOW_RESIZE_DISABLED) {
						gui.subwindow_resize_from_rect = r;
						gui.subwindow_drag_from = mb->get_position();
						gui.subwindow_drag = SUB_WINDOW_DRAG_RESIZE;
						click_on_window = true;
					}
				}
			}
			if (!click_on_window && r.has_point(mb->get_position())) {
				//clicked, see if it needs to fetch focus
				if (gui.subwindow_focused != sw.window) {
					//refocus
					_sub_window_grab_focus(sw.window);
				}

				click_on_window = true;
			}

			if (click_on_window) {
				break;
			}
		}

		if (!click_on_window && gui.subwindow_focused) {
			//no window found and clicked, remove focus
			_sub_window_grab_focus(nullptr);
		}
	}

	if (gui.subwindow_focused) {
		Ref<InputEventMouseMotion> mm = p_event;
		if (mm.is_valid()) {
			SubWindowResize resize = _sub_window_get_resize_margin(gui.subwindow_focused, mm->get_position());
			if (resize != SUB_WINDOW_RESIZE_DISABLED) {
				DisplayServer::CursorShape shapes[SUB_WINDOW_RESIZE_MAX] = {
					DisplayServer::CURSOR_ARROW,
					DisplayServer::CURSOR_FDIAGSIZE,
					DisplayServer::CURSOR_VSIZE,
					DisplayServer::CURSOR_BDIAGSIZE,
					DisplayServer::CURSOR_HSIZE,
					DisplayServer::CURSOR_HSIZE,
					DisplayServer::CURSOR_BDIAGSIZE,
					DisplayServer::CURSOR_VSIZE,
					DisplayServer::CURSOR_FDIAGSIZE
				};

				DisplayServer::get_singleton()->cursor_set_shape(shapes[resize]);

				return true; //reserved for showing the resize cursor
			}
		}
	}

	if (gui.subwindow_drag != SUB_WINDOW_DRAG_DISABLED) {
		return true; // dragging, don't pass the event
	}

	if (!gui.subwindow_focused) {
		return false;
	}

	Transform2D window_ofs;
	window_ofs.set_origin(-gui.subwindow_focused->get_position());

	Ref<InputEvent> ev = p_event->xformed_by(window_ofs);

	gui.subwindow_focused->_window_input(ev);

	return true;
}

void Viewport::input(const Ref<InputEvent> &p_event, bool p_local_coords) {
	ERR_FAIL_COND(!is_inside_tree());

	if (disable_input) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
		return;
	}

	local_input_handled = false;

	Ref<InputEvent> ev;
	if (!p_local_coords) {
		ev = _make_input_local(p_event);
	} else {
		ev = p_event;
	}

	if (is_embedding_subwindows() && _sub_windows_forward_input(p_event)) {
		set_input_as_handled();
		return;
	}

	if (!_can_consume_input_events()) {
		return;
	}

	if (!is_input_handled()) {
		get_tree()->_call_input_pause(input_group, "_input", ev, this); //not a bug, must happen before GUI, order is _input -> gui input -> _unhandled input
	}

	if (!is_input_handled()) {
		_gui_input_event(ev);
	}

	event_count++;
	//get_tree()->call_group(SceneTree::GROUP_CALL_REVERSE|SceneTree::GROUP_CALL_REALTIME|SceneTree::GROUP_CALL_MULIILEVEL,gui_input_group,"_gui_input",ev); //special one for GUI, as controls use their own process check
}

void Viewport::unhandled_input(const Ref<InputEvent> &p_event, bool p_local_coords) {
	ERR_FAIL_COND(!is_inside_tree());

	if (disable_input || !_can_consume_input_events()) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
		return;
	}

	Ref<InputEvent> ev;
	if (!p_local_coords) {
		ev = _make_input_local(p_event);
	} else {
		ev = p_event;
	}

	get_tree()->_call_input_pause(unhandled_input_group, "_unhandled_input", ev, this);
	if (!is_input_handled() && Object::cast_to<InputEventKey>(*ev) != nullptr) {
		get_tree()->_call_input_pause(unhandled_key_input_group, "_unhandled_key_input", ev, this);
	}

	if (physics_object_picking && !is_input_handled()) {
		if (Input::get_singleton()->get_mouse_mode() != Input::MOUSE_MODE_CAPTURED &&
				(Object::cast_to<InputEventMouseButton>(*ev) ||
						Object::cast_to<InputEventMouseMotion>(*ev) ||
						Object::cast_to<InputEventScreenDrag>(*ev) ||
						Object::cast_to<InputEventScreenTouch>(*ev) ||
						Object::cast_to<InputEventKey>(*ev) //to remember state

						)) {
			physics_picking_events.push_back(ev);
		}
	}
}

void Viewport::set_use_own_world_3d(bool p_world_3d) {
	if (p_world_3d == own_world_3d.is_valid()) {
		return;
	}

	if (is_inside_tree()) {
		_propagate_exit_world(this);
	}

	if (!p_world_3d) {
		own_world_3d = Ref<World3D>();
		if (world_3d.is_valid()) {
			world_3d->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &Viewport::_own_world_3d_changed));
		}
	} else {
		if (world_3d.is_valid()) {
			own_world_3d = world_3d->duplicate();
			world_3d->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &Viewport::_own_world_3d_changed));
		} else {
			own_world_3d = Ref<World3D>(memnew(World3D));
		}
	}

	if (is_inside_tree()) {
		_propagate_enter_world(this);
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
	}

	_update_listener();
}

bool Viewport::is_using_own_world_3d() const {
	return own_world_3d.is_valid();
}

void Viewport::set_physics_object_picking(bool p_enable) {
	physics_object_picking = p_enable;
	if (!physics_object_picking) {
		physics_picking_events.clear();
	}
}

bool Viewport::get_physics_object_picking() {
	return physics_object_picking;
}

Vector2 Viewport::get_camera_coords(const Vector2 &p_viewport_coords) const {
	Transform2D xf = get_final_transform();
	return xf.xform(p_viewport_coords);
}

Vector2 Viewport::get_camera_rect_size() const {
	return size;
}

void Viewport::set_disable_input(bool p_disable) {
	disable_input = p_disable;
}

bool Viewport::is_input_disabled() const {
	return disable_input;
}

Variant Viewport::gui_get_drag_data() const {
	return gui.drag_data;
}

String Viewport::get_configuration_warning() const {
	/*if (get_parent() && !Object::cast_to<Control>(get_parent()) && !render_target) {

		return TTR("This viewport is not set as render target. If you intend for it to display its contents directly to the screen, make it a child of a Control so it can obtain a size. Otherwise, make it a RenderTarget and assign its internal texture to some node for display.");
	}*/

	String warning = Node::get_configuration_warning();

	if (size.x == 0 || size.y == 0) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("Viewport size must be greater than 0 to render anything.");
	}
	return warning;
}

void Viewport::gui_reset_canvas_sort_index() {
	gui.canvas_sort_index = 0;
}

int Viewport::gui_get_canvas_sort_index() {
	return gui.canvas_sort_index++;
}

void Viewport::set_msaa(MSAA p_msaa) {
	ERR_FAIL_INDEX(p_msaa, MSAA_MAX);
	if (msaa == p_msaa) {
		return;
	}
	msaa = p_msaa;
	RS::get_singleton()->viewport_set_msaa(viewport, RS::ViewportMSAA(p_msaa));
}

Viewport::MSAA Viewport::get_msaa() const {
	return msaa;
}

void Viewport::set_screen_space_aa(ScreenSpaceAA p_screen_space_aa) {
	ERR_FAIL_INDEX(p_screen_space_aa, SCREEN_SPACE_AA_MAX);
	if (screen_space_aa == p_screen_space_aa) {
		return;
	}
	screen_space_aa = p_screen_space_aa;
	RS::get_singleton()->viewport_set_screen_space_aa(viewport, RS::ViewportScreenSpaceAA(p_screen_space_aa));
}

Viewport::ScreenSpaceAA Viewport::get_screen_space_aa() const {
	return screen_space_aa;
}

void Viewport::set_use_debanding(bool p_use_debanding) {
	if (use_debanding == p_use_debanding)
		return;
	use_debanding = p_use_debanding;
	RS::get_singleton()->viewport_set_use_debanding(viewport, p_use_debanding);
}

bool Viewport::is_using_debanding() const {
	return use_debanding;
}

void Viewport::set_debug_draw(DebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
	RS::get_singleton()->viewport_set_debug_draw(viewport, RS::ViewportDebugDraw(p_debug_draw));
}

Viewport::DebugDraw Viewport::get_debug_draw() const {
	return debug_draw;
}

int Viewport::get_render_info(RenderInfo p_info) {
	return RS::get_singleton()->viewport_get_render_info(viewport, RS::ViewportRenderInfo(p_info));
}

void Viewport::set_snap_controls_to_pixels(bool p_enable) {
	snap_controls_to_pixels = p_enable;
}

bool Viewport::is_snap_controls_to_pixels_enabled() const {
	return snap_controls_to_pixels;
}

void Viewport::set_snap_2d_transforms_to_pixel(bool p_enable) {
	snap_2d_transforms_to_pixel = p_enable;
	RS::get_singleton()->viewport_set_snap_2d_transforms_to_pixel(viewport, snap_2d_transforms_to_pixel);
}

bool Viewport::is_snap_2d_transforms_to_pixel_enabled() const {
	return snap_2d_transforms_to_pixel;
}

void Viewport::set_snap_2d_vertices_to_pixel(bool p_enable) {
	snap_2d_vertices_to_pixel = p_enable;
	RS::get_singleton()->viewport_set_snap_2d_vertices_to_pixel(viewport, snap_2d_vertices_to_pixel);
}

bool Viewport::is_snap_2d_vertices_to_pixel_enabled() const {
	return snap_2d_vertices_to_pixel;
}

bool Viewport::gui_is_dragging() const {
	return gui.dragging;
}

void Viewport::set_input_as_handled() {
	_drop_physics_mouseover();
	if (handle_input_locally) {
		local_input_handled = true;
	} else {
		ERR_FAIL_COND(!is_inside_tree());
		Viewport *vp = this;
		while (true) {
			if (Object::cast_to<Window>(vp)) {
				break;
			}
			if (!vp->get_parent()) {
				break;
			}
			vp = vp->get_parent()->get_viewport();
		}
		vp->set_input_as_handled();
	}
}

bool Viewport::is_input_handled() const {
	if (handle_input_locally) {
		return local_input_handled;
	} else {
		const Viewport *vp = this;
		while (true) {
			if (Object::cast_to<Window>(vp)) {
				break;
			}
			if (!vp->get_parent()) {
				break;
			}
			vp = vp->get_parent()->get_viewport();
		}
		return vp->is_input_handled();
	}
}

void Viewport::set_handle_input_locally(bool p_enable) {
	handle_input_locally = p_enable;
}

bool Viewport::is_handling_input_locally() const {
	return handle_input_locally;
}

void Viewport::_validate_property(PropertyInfo &property) const {
}

void Viewport::set_default_canvas_item_texture_filter(DefaultCanvasItemTextureFilter p_filter) {
	ERR_FAIL_INDEX(p_filter, DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_MAX);

	if (default_canvas_item_texture_filter == p_filter) {
		return;
	}
	default_canvas_item_texture_filter = p_filter;
	switch (default_canvas_item_texture_filter) {
		case DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_filter(viewport, RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
			break;
		case DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_filter(viewport, RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR);
			break;
		case DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_filter(viewport, RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS);
			break;
		case DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_filter(viewport, RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
			break;
		default: {
		}
	}
}

Viewport::DefaultCanvasItemTextureFilter Viewport::get_default_canvas_item_texture_filter() const {
	return default_canvas_item_texture_filter;
}

void Viewport::set_default_canvas_item_texture_repeat(DefaultCanvasItemTextureRepeat p_repeat) {
	ERR_FAIL_INDEX(p_repeat, DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MAX);

	if (default_canvas_item_texture_repeat == p_repeat) {
		return;
	}

	default_canvas_item_texture_repeat = p_repeat;

	switch (default_canvas_item_texture_repeat) {
		case DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_repeat(viewport, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			break;
		case DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_ENABLED:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_repeat(viewport, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			break;
		case DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MIRROR:
			RS::get_singleton()->viewport_set_default_canvas_item_texture_repeat(viewport, RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
			break;
		default: {
		}
	}
}

Viewport::DefaultCanvasItemTextureRepeat Viewport::get_default_canvas_item_texture_repeat() const {
	return default_canvas_item_texture_repeat;
}

DisplayServer::WindowID Viewport::get_window_id() const {
	return DisplayServer::MAIN_WINDOW_ID;
}

Viewport *Viewport::get_parent_viewport() const {
	ERR_FAIL_COND_V(!is_inside_tree(), nullptr);
	if (!get_parent()) {
		return nullptr; //root viewport
	}

	return get_parent()->get_viewport();
}

void Viewport::set_embed_subwindows_hint(bool p_embed) {
	gui.embed_subwindows_hint = p_embed;
}

bool Viewport::get_embed_subwindows_hint() const {
	return gui.embed_subwindows_hint;
}

bool Viewport::is_embedding_subwindows() const {
	return gui.embed_subwindows_hint;
}

void Viewport::pass_mouse_focus_to(Viewport *p_viewport, Control *p_control) {
	ERR_FAIL_NULL(p_viewport);
	ERR_FAIL_NULL(p_control);

	if (gui.mouse_focus) {
		p_viewport->gui.mouse_focus = p_control;
		p_viewport->gui.mouse_focus_mask = gui.mouse_focus_mask;
		p_viewport->gui.key_focus = p_control;
		p_viewport->gui.forced_mouse_focus = true;

		gui.mouse_focus = nullptr;
		gui.forced_mouse_focus = false;
		gui.mouse_focus_mask = 0;
	}
}

void Viewport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world_2d", "world_2d"), &Viewport::set_world_2d);
	ClassDB::bind_method(D_METHOD("get_world_2d"), &Viewport::get_world_2d);
	ClassDB::bind_method(D_METHOD("find_world_2d"), &Viewport::find_world_2d);
	ClassDB::bind_method(D_METHOD("set_world_3d", "world_3d"), &Viewport::set_world_3d);
	ClassDB::bind_method(D_METHOD("get_world_3d"), &Viewport::get_world_3d);
	ClassDB::bind_method(D_METHOD("find_world_3d"), &Viewport::find_world_3d);

	ClassDB::bind_method(D_METHOD("set_canvas_transform", "xform"), &Viewport::set_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_canvas_transform"), &Viewport::get_canvas_transform);

	ClassDB::bind_method(D_METHOD("set_global_canvas_transform", "xform"), &Viewport::set_global_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_global_canvas_transform"), &Viewport::get_global_canvas_transform);
	ClassDB::bind_method(D_METHOD("get_final_transform"), &Viewport::get_final_transform);

	ClassDB::bind_method(D_METHOD("get_visible_rect"), &Viewport::get_visible_rect);
	ClassDB::bind_method(D_METHOD("set_transparent_background", "enable"), &Viewport::set_transparent_background);
	ClassDB::bind_method(D_METHOD("has_transparent_background"), &Viewport::has_transparent_background);

	ClassDB::bind_method(D_METHOD("set_msaa", "msaa"), &Viewport::set_msaa);
	ClassDB::bind_method(D_METHOD("get_msaa"), &Viewport::get_msaa);

	ClassDB::bind_method(D_METHOD("set_screen_space_aa", "screen_space_aa"), &Viewport::set_screen_space_aa);
	ClassDB::bind_method(D_METHOD("get_screen_space_aa"), &Viewport::get_screen_space_aa);

	ClassDB::bind_method(D_METHOD("set_use_debanding", "enable"), &Viewport::set_use_debanding);
	ClassDB::bind_method(D_METHOD("is_using_debanding"), &Viewport::is_using_debanding);

	ClassDB::bind_method(D_METHOD("set_debug_draw", "debug_draw"), &Viewport::set_debug_draw);
	ClassDB::bind_method(D_METHOD("get_debug_draw"), &Viewport::get_debug_draw);

	ClassDB::bind_method(D_METHOD("get_render_info", "info"), &Viewport::get_render_info);

	ClassDB::bind_method(D_METHOD("get_texture"), &Viewport::get_texture);

	ClassDB::bind_method(D_METHOD("set_physics_object_picking", "enable"), &Viewport::set_physics_object_picking);
	ClassDB::bind_method(D_METHOD("get_physics_object_picking"), &Viewport::get_physics_object_picking);

	ClassDB::bind_method(D_METHOD("get_viewport_rid"), &Viewport::get_viewport_rid);
	ClassDB::bind_method(D_METHOD("input_text", "text"), &Viewport::input_text);
	ClassDB::bind_method(D_METHOD("input", "event", "in_local_coords"), &Viewport::input, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("unhandled_input", "event", "in_local_coords"), &Viewport::unhandled_input, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("update_worlds"), &Viewport::update_worlds);

	ClassDB::bind_method(D_METHOD("set_use_own_world_3d", "enable"), &Viewport::set_use_own_world_3d);
	ClassDB::bind_method(D_METHOD("is_using_own_world_3d"), &Viewport::is_using_own_world_3d);

	ClassDB::bind_method(D_METHOD("get_camera"), &Viewport::get_camera);

	ClassDB::bind_method(D_METHOD("set_as_audio_listener", "enable"), &Viewport::set_as_audio_listener);
	ClassDB::bind_method(D_METHOD("is_audio_listener"), &Viewport::is_audio_listener);

	ClassDB::bind_method(D_METHOD("set_as_audio_listener_2d", "enable"), &Viewport::set_as_audio_listener_2d);
	ClassDB::bind_method(D_METHOD("is_audio_listener_2d"), &Viewport::is_audio_listener_2d);

	ClassDB::bind_method(D_METHOD("get_mouse_position"), &Viewport::get_mouse_position);
	ClassDB::bind_method(D_METHOD("warp_mouse", "to_position"), &Viewport::warp_mouse);

	ClassDB::bind_method(D_METHOD("gui_get_drag_data"), &Viewport::gui_get_drag_data);
	ClassDB::bind_method(D_METHOD("gui_is_dragging"), &Viewport::gui_is_dragging);

	ClassDB::bind_method(D_METHOD("set_disable_input", "disable"), &Viewport::set_disable_input);
	ClassDB::bind_method(D_METHOD("is_input_disabled"), &Viewport::is_input_disabled);

	ClassDB::bind_method(D_METHOD("_gui_remove_focus_for_window"), &Viewport::_gui_remove_focus_for_window);
	ClassDB::bind_method(D_METHOD("_post_gui_grab_click_focus"), &Viewport::_post_gui_grab_click_focus);

	ClassDB::bind_method(D_METHOD("set_shadow_atlas_size", "size"), &Viewport::set_shadow_atlas_size);
	ClassDB::bind_method(D_METHOD("get_shadow_atlas_size"), &Viewport::get_shadow_atlas_size);

	ClassDB::bind_method(D_METHOD("set_snap_controls_to_pixels", "enabled"), &Viewport::set_snap_controls_to_pixels);
	ClassDB::bind_method(D_METHOD("is_snap_controls_to_pixels_enabled"), &Viewport::is_snap_controls_to_pixels_enabled);

	ClassDB::bind_method(D_METHOD("set_snap_2d_transforms_to_pixel", "enabled"), &Viewport::set_snap_2d_transforms_to_pixel);
	ClassDB::bind_method(D_METHOD("is_snap_2d_transforms_to_pixel_enabled"), &Viewport::is_snap_2d_transforms_to_pixel_enabled);

	ClassDB::bind_method(D_METHOD("set_snap_2d_vertices_to_pixel", "enabled"), &Viewport::set_snap_2d_vertices_to_pixel);
	ClassDB::bind_method(D_METHOD("is_snap_2d_vertices_to_pixel_enabled"), &Viewport::is_snap_2d_vertices_to_pixel_enabled);

	ClassDB::bind_method(D_METHOD("set_shadow_atlas_quadrant_subdiv", "quadrant", "subdiv"), &Viewport::set_shadow_atlas_quadrant_subdiv);
	ClassDB::bind_method(D_METHOD("get_shadow_atlas_quadrant_subdiv", "quadrant"), &Viewport::get_shadow_atlas_quadrant_subdiv);

	ClassDB::bind_method(D_METHOD("set_input_as_handled"), &Viewport::set_input_as_handled);
	ClassDB::bind_method(D_METHOD("is_input_handled"), &Viewport::is_input_handled);

	ClassDB::bind_method(D_METHOD("set_handle_input_locally", "enable"), &Viewport::set_handle_input_locally);
	ClassDB::bind_method(D_METHOD("is_handling_input_locally"), &Viewport::is_handling_input_locally);

	ClassDB::bind_method(D_METHOD("set_default_canvas_item_texture_filter", "mode"), &Viewport::set_default_canvas_item_texture_filter);
	ClassDB::bind_method(D_METHOD("get_default_canvas_item_texture_filter"), &Viewport::get_default_canvas_item_texture_filter);

	ClassDB::bind_method(D_METHOD("set_embed_subwindows_hint", "enable"), &Viewport::set_embed_subwindows_hint);
	ClassDB::bind_method(D_METHOD("get_embed_subwindows_hint"), &Viewport::get_embed_subwindows_hint);
	ClassDB::bind_method(D_METHOD("is_embedding_subwindows"), &Viewport::is_embedding_subwindows);

	ClassDB::bind_method(D_METHOD("set_default_canvas_item_texture_repeat", "mode"), &Viewport::set_default_canvas_item_texture_repeat);
	ClassDB::bind_method(D_METHOD("get_default_canvas_item_texture_repeat"), &Viewport::get_default_canvas_item_texture_repeat);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "own_world_3d"), "set_use_own_world_3d", "is_using_own_world_3d");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "world_3d", PROPERTY_HINT_RESOURCE_TYPE, "World3D"), "set_world_3d", "get_world_3d");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "world_2d", PROPERTY_HINT_RESOURCE_TYPE, "World2D", 0), "set_world_2d", "get_world_2d");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transparent_bg"), "set_transparent_background", "has_transparent_background");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "handle_input_locally"), "set_handle_input_locally", "is_handling_input_locally");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "snap_2d_transforms_to_pixel"), "set_snap_2d_transforms_to_pixel", "is_snap_2d_transforms_to_pixel_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "snap_2d_vertices_to_pixel"), "set_snap_2d_vertices_to_pixel", "is_snap_2d_vertices_to_pixel_enabled");
	ADD_GROUP("Rendering", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "msaa", PROPERTY_HINT_ENUM, "Disabled,2x,4x,8x,16x,AndroidVR 2x,AndroidVR 4x"), "set_msaa", "get_msaa");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "screen_space_aa", PROPERTY_HINT_ENUM, "Disabled,FXAA"), "set_screen_space_aa", "get_screen_space_aa");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_debanding"), "set_use_debanding", "is_using_debanding");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_draw", PROPERTY_HINT_ENUM, "Disabled,Unshaded,Overdraw,Wireframe"), "set_debug_draw", "get_debug_draw");
	ADD_GROUP("Canvas Items", "canvas_item_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "canvas_item_default_texture_filter", PROPERTY_HINT_ENUM, "Nearest,Linear,MipmapLinear,MipmapNearest"), "set_default_canvas_item_texture_filter", "get_default_canvas_item_texture_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "canvas_item_default_texture_repeat", PROPERTY_HINT_ENUM, "Disabled,Enabled,Mirror"), "set_default_canvas_item_texture_repeat", "get_default_canvas_item_texture_repeat");
	ADD_GROUP("Audio Listener", "audio_listener_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_listener_enable_2d"), "set_as_audio_listener_2d", "is_audio_listener_2d");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_listener_enable_3d"), "set_as_audio_listener", "is_audio_listener");
	ADD_GROUP("Physics", "physics_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "physics_object_picking"), "set_physics_object_picking", "get_physics_object_picking");
	ADD_GROUP("GUI", "gui_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_disable_input"), "set_disable_input", "is_input_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_snap_controls_to_pixels"), "set_snap_controls_to_pixels", "is_snap_controls_to_pixels_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_embed_subwindows"), "set_embed_subwindows_hint", "get_embed_subwindows_hint");
	ADD_GROUP("Shadow Atlas", "shadow_atlas_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_atlas_size"), "set_shadow_atlas_size", "get_shadow_atlas_size");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_0", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 0);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_1", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_2", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 2);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_3", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 3);
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "canvas_transform", PROPERTY_HINT_NONE, "", 0), "set_canvas_transform", "get_canvas_transform");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "global_canvas_transform", PROPERTY_HINT_NONE, "", 0), "set_global_canvas_transform", "get_global_canvas_transform");

	ADD_SIGNAL(MethodInfo("size_changed"));
	ADD_SIGNAL(MethodInfo("gui_focus_changed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Control")));

	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_DISABLED);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_1);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_4);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_16);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_64);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_256);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_1024);
	BIND_ENUM_CONSTANT(SHADOW_ATLAS_QUADRANT_SUBDIV_MAX);

	BIND_ENUM_CONSTANT(MSAA_DISABLED);
	BIND_ENUM_CONSTANT(MSAA_2X);
	BIND_ENUM_CONSTANT(MSAA_4X);
	BIND_ENUM_CONSTANT(MSAA_8X);
	BIND_ENUM_CONSTANT(MSAA_16X);
	BIND_ENUM_CONSTANT(MSAA_MAX);

	BIND_ENUM_CONSTANT(SCREEN_SPACE_AA_DISABLED);
	BIND_ENUM_CONSTANT(SCREEN_SPACE_AA_FXAA);
	BIND_ENUM_CONSTANT(SCREEN_SPACE_AA_MAX);

	BIND_ENUM_CONSTANT(RENDER_INFO_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_VERTICES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_MATERIAL_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_SHADER_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_SURFACE_CHANGES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_MAX);

	BIND_ENUM_CONSTANT(DEBUG_DRAW_DISABLED);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_UNSHADED);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_LIGHTING);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_OVERDRAW);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_WIREFRAME);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_NORMAL_BUFFER);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_GI_PROBE_ALBEDO);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_GI_PROBE_LIGHTING);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_GI_PROBE_EMISSION);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SHADOW_ATLAS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SCENE_LUMINANCE);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SSAO);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_PSSM_SPLITS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_DECAL_ATLAS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SDFGI);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SDFGI_PROBES);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_GI_BUFFER);

	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_MAX);

	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MAX);
}

Viewport::Viewport() {
	world_2d = Ref<World2D>(memnew(World2D));

	viewport = RenderingServer::get_singleton()->viewport_create();
	texture_rid = RenderingServer::get_singleton()->viewport_get_texture(viewport);

	default_texture.instance();
	default_texture->vp = const_cast<Viewport *>(this);
	viewport_textures.insert(default_texture.ptr());
	default_texture->proxy = RS::get_singleton()->texture_proxy_create(texture_rid);

	audio_listener = false;
	//internal_listener_2d = SpatialSound2DServer::get_singleton()->listener_create();
	audio_listener_2d = false;
	transparent_bg = false;
	parent = nullptr;
	listener = nullptr;
	camera = nullptr;
	override_canvas_transform = false;
	canvas_layers.insert(nullptr); // This eases picking code (interpreted as the canvas of the Viewport)

	gen_mipmaps = false;

	//clear=true;

	physics_object_picking = false;
	physics_has_last_mousepos = false;
	physics_last_mousepos = Vector2(Math_INF, Math_INF);

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

	// Window tooltip.
	gui.tooltip_timer = -1;

	gui.tooltip_delay = GLOBAL_DEF("gui/timers/tooltip_delay_sec", 0.5);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/timers/tooltip_delay_sec", PropertyInfo(Variant::FLOAT, "gui/timers/tooltip_delay_sec", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater")); // No negative numbers

	gui.tooltip_control = nullptr;
	gui.tooltip_label = nullptr;
	gui.drag_preview = nullptr;
	gui.drag_attempted = false;
	gui.canvas_sort_index = 0;
	gui.roots_order_dirty = false;
	gui.mouse_focus = nullptr;
	gui.forced_mouse_focus = false;
	gui.last_mouse_focus = nullptr;
	gui.subwindow_focused = nullptr;
	gui.subwindow_drag = SUB_WINDOW_DRAG_DISABLED;

	msaa = MSAA_DISABLED;
	screen_space_aa = SCREEN_SPACE_AA_DISABLED;
	debug_draw = DEBUG_DRAW_DISABLED;

	snap_controls_to_pixels = true;
	snap_2d_transforms_to_pixel = false;
	snap_2d_vertices_to_pixel = false;

	physics_last_mouse_state.alt = false;
	physics_last_mouse_state.control = false;
	physics_last_mouse_state.shift = false;
	physics_last_mouse_state.meta = false;
	physics_last_mouse_state.mouse_mask = 0;
	local_input_handled = false;
	handle_input_locally = true;

	size_allocated = false;

	default_canvas_item_texture_filter = DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR;
	default_canvas_item_texture_repeat = DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED;
}

Viewport::~Viewport() {
	//erase itself from viewport textures
	for (Set<ViewportTexture *>::Element *E = viewport_textures.front(); E; E = E->next()) {
		E->get()->vp = nullptr;
	}
	RenderingServer::get_singleton()->free(viewport);
}

/////////////////////////////////

void SubViewport::set_use_xr(bool p_use_xr) {
	xr = p_use_xr;

	RS::get_singleton()->viewport_set_use_xr(get_viewport_rid(), xr);
}

bool SubViewport::is_using_xr() {
	return xr;
}

void SubViewport::set_size(const Size2i &p_size) {
	_set_size(p_size, _get_size_2d_override(), Rect2i(), _stretch_transform(), true);
}

Size2i SubViewport::get_size() const {
	return _get_size();
}

void SubViewport::set_size_2d_override(const Size2i &p_size) {
	_set_size(_get_size(), p_size, Rect2i(), _stretch_transform(), true);
}

Size2i SubViewport::get_size_2d_override() const {
	return _get_size_2d_override();
}

void SubViewport::set_size_2d_override_stretch(bool p_enable) {
	if (p_enable == size_2d_override_stretch) {
		return;
	}

	size_2d_override_stretch = p_enable;
	_set_size(_get_size(), _get_size_2d_override(), Rect2i(), _stretch_transform(), true);
}

bool SubViewport::is_size_2d_override_stretch_enabled() const {
	return size_2d_override_stretch;
}

void SubViewport::set_update_mode(UpdateMode p_mode) {
	update_mode = p_mode;
	RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::ViewportUpdateMode(p_mode));
}

SubViewport::UpdateMode SubViewport::get_update_mode() const {
	return update_mode;
}

void SubViewport::set_clear_mode(ClearMode p_mode) {
	clear_mode = p_mode;
	RS::get_singleton()->viewport_set_clear_mode(get_viewport_rid(), RS::ViewportClearMode(p_mode));
}

SubViewport::ClearMode SubViewport::get_clear_mode() const {
	return clear_mode;
}

DisplayServer::WindowID SubViewport::get_window_id() const {
	return DisplayServer::INVALID_WINDOW_ID;
}

Transform2D SubViewport::_stretch_transform() {
	Transform2D transform = Transform2D();
	Size2i view_size_2d_override = _get_size_2d_override();
	if (size_2d_override_stretch && view_size_2d_override.width > 0 && view_size_2d_override.height > 0) {
		Size2 scale = _get_size() / view_size_2d_override;
		transform.scale(scale);
	}

	return transform;
}

void SubViewport::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		RS::get_singleton()->viewport_set_active(get_viewport_rid(), true);
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {
		RS::get_singleton()->viewport_set_active(get_viewport_rid(), false);
	}
}

void SubViewport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_use_xr", "use"), &SubViewport::set_use_xr);
	ClassDB::bind_method(D_METHOD("is_using_xr"), &SubViewport::is_using_xr);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &SubViewport::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &SubViewport::get_size);

	ClassDB::bind_method(D_METHOD("set_size_2d_override", "size"), &SubViewport::set_size_2d_override);
	ClassDB::bind_method(D_METHOD("get_size_2d_override"), &SubViewport::get_size_2d_override);

	ClassDB::bind_method(D_METHOD("set_size_2d_override_stretch", "enable"), &SubViewport::set_size_2d_override_stretch);
	ClassDB::bind_method(D_METHOD("is_size_2d_override_stretch_enabled"), &SubViewport::is_size_2d_override_stretch_enabled);

	ClassDB::bind_method(D_METHOD("set_update_mode", "mode"), &SubViewport::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &SubViewport::get_update_mode);

	ClassDB::bind_method(D_METHOD("set_clear_mode", "mode"), &SubViewport::set_clear_mode);
	ClassDB::bind_method(D_METHOD("get_clear_mode"), &SubViewport::get_clear_mode);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "xr"), "set_use_xr", "is_using_xr");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size_2d_override"), "set_size_2d_override", "get_size_2d_override");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "size_2d_override_stretch"), "set_size_2d_override_stretch", "is_size_2d_override_stretch_enabled");
	ADD_GROUP("Render Target", "render_target_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_target_clear_mode", PROPERTY_HINT_ENUM, "Always,Never,Next Frame"), "set_clear_mode", "get_clear_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_target_update_mode", PROPERTY_HINT_ENUM, "Disabled,Once,When Visible,Always"), "set_update_mode", "get_update_mode");

	BIND_ENUM_CONSTANT(CLEAR_MODE_ALWAYS);
	BIND_ENUM_CONSTANT(CLEAR_MODE_NEVER);
	BIND_ENUM_CONSTANT(CLEAR_MODE_ONLY_NEXT_FRAME);

	BIND_ENUM_CONSTANT(UPDATE_DISABLED);
	BIND_ENUM_CONSTANT(UPDATE_ONCE);
	BIND_ENUM_CONSTANT(UPDATE_WHEN_VISIBLE);
	BIND_ENUM_CONSTANT(UPDATE_WHEN_PARENT_VISIBLE);
	BIND_ENUM_CONSTANT(UPDATE_ALWAYS);
}

SubViewport::SubViewport() {
	xr = false;
	size_2d_override_stretch = false;
	update_mode = UPDATE_WHEN_VISIBLE;
	clear_mode = CLEAR_MODE_ALWAYS;
}

SubViewport::~SubViewport() {
}
