/*************************************************************************/
/*  viewport.cpp                                                         */
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

#include "viewport.h"

#include "core/core_string_names.h"
#include "core/debugger/engine_debugger.h"
#include "core/object/message_queue.h"
#include "core/string/translation.h"
#include "core/templates/pair.h"
#include "scene/2d/audio_listener_2d.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/collision_object_2d.h"
#ifndef _3D_DISABLED
#include "scene/3d/audio_listener_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/collision_object_3d.h"
#include "scene/3d/world_environment.h"
#endif // _3D_DISABLED
#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/popup.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/subviewport_container.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/window.h"
#include "scene/resources/mesh.h"
#include "scene/resources/text_line.h"
#include "scene/resources/world_2d.h"
#include "scene/scene_string_names.h"
#include "servers/audio_server.h"
#include "servers/rendering/rendering_server_globals.h"

void ViewportTexture::setup_local_to_scene() {
	Node *local_scene = get_local_scene();
	if (!local_scene) {
		return;
	}

	if (vp) {
		vp->viewport_textures.erase(this);
	}

	vp = nullptr;

	Node *vpn = local_scene->get_node(path);
	ERR_FAIL_COND_MSG(!vpn, "ViewportTexture: Path to node is invalid.");

	vp = Object::cast_to<Viewport>(vpn);

	ERR_FAIL_COND_MSG(!vp, "ViewportTexture: Path to node does not point to a viewport.");

	vp->viewport_textures.insert(this);

	if (proxy_ph.is_valid()) {
		RS::get_singleton()->texture_proxy_update(proxy, vp->texture_rid);
		RS::get_singleton()->free(proxy_ph);
	} else {
		ERR_FAIL_COND(proxy.is_valid()); // Should be invalid.
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
	if (proxy.is_null()) {
		proxy_ph = RS::get_singleton()->texture_2d_placeholder_create();
		proxy = RS::get_singleton()->texture_proxy_create(proxy_ph);
	}
	return proxy;
}

bool ViewportTexture::has_alpha() const {
	return false;
}

Ref<Image> ViewportTexture::get_image() const {
	ERR_FAIL_COND_V_MSG(!vp, Ref<Image>(), "Viewport Texture must be set to use it.");
	return RS::get_singleton()->texture_2d_get(vp->texture_rid);
}

void ViewportTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_viewport_path_in_scene", "path"), &ViewportTexture::set_viewport_path_in_scene);
	ClassDB::bind_method(D_METHOD("get_viewport_path_in_scene"), &ViewportTexture::get_viewport_path_in_scene);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "viewport_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "SubViewport", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT), "set_viewport_path_in_scene", "get_viewport_path_in_scene");
}

ViewportTexture::ViewportTexture() {
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
		Ref<StyleBox> panel = p_window->get_theme_stylebox(SNAME("embedded_border"));
		panel->draw(sw.canvas_item, r);

		// Draw the title bar text.
		Ref<Font> title_font = p_window->get_theme_font(SNAME("title_font"));
		int font_size = p_window->get_theme_font_size(SNAME("title_font_size"));
		Color title_color = p_window->get_theme_color(SNAME("title_color"));
		int title_height = p_window->get_theme_constant(SNAME("title_height"));
		int close_h_ofs = p_window->get_theme_constant(SNAME("close_h_ofs"));
		int close_v_ofs = p_window->get_theme_constant(SNAME("close_v_ofs"));

		TextLine title_text = TextLine(p_window->atr(p_window->get_title()), title_font, font_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		title_text.set_width(r.size.width - panel->get_minimum_size().x - close_h_ofs);
		title_text.set_direction(p_window->is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		int x = (r.size.width - title_text.get_size().x) / 2;
		int y = (-title_height - title_text.get_size().y) / 2;

		Color font_outline_color = p_window->get_theme_color(SNAME("title_outline_modulate"));
		int outline_size = p_window->get_theme_constant(SNAME("title_outline_size"));
		if (outline_size > 0 && font_outline_color.a > 0) {
			title_text.draw_outline(sw.canvas_item, r.position + Point2(x, y), outline_size, font_outline_color);
		}
		title_text.draw(sw.canvas_item, r.position + Point2(x, y), title_color);

		bool pressed = gui.subwindow_focused == sw.window && gui.subwindow_drag == SUB_WINDOW_DRAG_CLOSE && gui.subwindow_drag_close_inside;
		Ref<Texture2D> close_icon = p_window->get_theme_icon(pressed ? "close_pressed" : "close");
		close_icon->draw(sw.canvas_item, r.position + Vector2(r.size.width - close_h_ofs, -close_v_ofs));
	}

	RS::get_singleton()->canvas_item_add_texture_rect(sw.canvas_item, r, sw.window->get_texture()->get_rid());
}

void Viewport::_sub_window_grab_focus(Window *p_window) {
	if (p_window == nullptr) {
		// Release current focus.
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
		// Can only move to foreground, but no focus granted.
		SubWindow sw = gui.sub_windows[index];
		gui.sub_windows.remove_at(index);
		gui.sub_windows.push_back(sw);
		index = gui.sub_windows.size() - 1;
		_sub_window_update_order();
		return;
	}

	if (gui.subwindow_focused) {
		if (gui.subwindow_focused == p_window) {
			return; // Nothing to do.
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

	{ // Move to foreground.
		SubWindow sw = gui.sub_windows[index];
		gui.sub_windows.remove_at(index);
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
			gui.sub_windows.remove_at(i);
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

void Viewport::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (get_parent()) {
				parent = get_parent()->get_viewport();
				RenderingServer::get_singleton()->viewport_set_parent_viewport(viewport, parent->get_viewport_rid());
			} else {
				parent = nullptr;
			}

			current_canvas = find_world_2d()->get_canvas();
			RenderingServer::get_singleton()->viewport_attach_canvas(viewport, current_canvas);
			_update_audio_listener_2d();
#ifndef _3D_DISABLED
			RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
			_update_audio_listener_3d();
#endif // _3D_DISABLED

			add_to_group("_viewports");
			if (get_tree()->is_debugging_collisions_hint()) {
				PhysicsServer2D::get_singleton()->space_set_debug_contacts(find_world_2d()->get_space(), get_tree()->get_collision_debug_contact_count());
				contact_2d_debug = RenderingServer::get_singleton()->canvas_item_create();
				RenderingServer::get_singleton()->canvas_item_set_parent(contact_2d_debug, find_world_2d()->get_canvas());
#ifndef _3D_DISABLED
				PhysicsServer3D::get_singleton()->space_set_debug_contacts(find_world_3d()->get_space(), get_tree()->get_collision_debug_contact_count());
				contact_3d_debug_multimesh = RenderingServer::get_singleton()->multimesh_create();
				RenderingServer::get_singleton()->multimesh_allocate_data(contact_3d_debug_multimesh, get_tree()->get_collision_debug_contact_count(), RS::MULTIMESH_TRANSFORM_3D, false);
				RenderingServer::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh, 0);
				RenderingServer::get_singleton()->multimesh_set_mesh(contact_3d_debug_multimesh, get_tree()->get_debug_contact_mesh()->get_rid());
				contact_3d_debug_instance = RenderingServer::get_singleton()->instance_create();
				RenderingServer::get_singleton()->instance_set_base(contact_3d_debug_instance, contact_3d_debug_multimesh);
				RenderingServer::get_singleton()->instance_set_scenario(contact_3d_debug_instance, find_world_3d()->get_scenario());
				RenderingServer::get_singleton()->instance_geometry_set_flag(contact_3d_debug_instance, RS::INSTANCE_FLAG_DRAW_NEXT_FRAME_IF_VISIBLE, true);
#endif // _3D_DISABLED
				set_physics_process_internal(true);
			}

		} break;
		case NOTIFICATION_READY: {
#ifndef _3D_DISABLED
			if (audio_listener_3d_set.size() && !audio_listener_3d) {
				AudioListener3D *first = nullptr;
				for (Set<AudioListener3D *>::Element *E = audio_listener_3d_set.front(); E; E = E->next()) {
					if (first == nullptr || first->is_greater_than(E->get())) {
						first = E->get();
					}
				}

				if (first) {
					first->make_current();
				}
			}

			if (camera_3d_set.size() && !camera_3d) {
				// There are cameras but no current camera, pick first in tree and make it current.
				Camera3D *first = nullptr;
				for (Set<Camera3D *>::Element *E = camera_3d_set.front(); E; E = E->next()) {
					if (first == nullptr || first->is_greater_than(E->get())) {
						first = E->get();
					}
				}

				if (first) {
					first->make_current();
				}
			}
#endif // _3D_DISABLED
		} break;
		case NOTIFICATION_EXIT_TREE: {
			_gui_cancel_tooltip();

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
			set_physics_process_internal(false);

			RS::get_singleton()->viewport_set_active(viewport, false);
			RenderingServer::get_singleton()->viewport_set_parent_viewport(viewport, RID());
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!get_tree()) {
				return;
			}

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
#ifndef _3D_DISABLED
			if (get_tree()->is_debugging_collisions_hint() && contact_3d_debug_multimesh.is_valid()) {
				Vector<Vector3> points = PhysicsServer3D::get_singleton()->space_get_contacts(find_world_3d()->get_space());
				int point_count = PhysicsServer3D::get_singleton()->space_get_contact_count(find_world_3d()->get_space());

				RS::get_singleton()->multimesh_set_visible_instances(contact_3d_debug_multimesh, point_count);

				for (int i = 0; i < point_count; i++) {
					Transform3D point_transform;
					point_transform.origin = points[i];
					RS::get_singleton()->multimesh_instance_set_transform(contact_3d_debug_multimesh, i, point_transform);
				}
			}
#endif // _3D_DISABLED
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

void Viewport::_process_picking() {
	if (!is_inside_tree()) {
		return;
	}
	if (!physics_object_picking) {
		return;
	}
	if (to_screen_rect != Rect2i() && Input::get_singleton()->get_mouse_mode() == Input::MOUSE_MODE_CAPTURED) {
		return;
	}

	_drop_physics_mouseover(true);

#ifndef _3D_DISABLED
	Vector2 last_pos(1e20, 1e20);
	CollisionObject3D *last_object = nullptr;
	ObjectID last_id;
	PhysicsDirectSpaceState3D::RayResult result;
#endif // _3D_DISABLED

	PhysicsDirectSpaceState2D *ss2d = PhysicsServer2D::get_singleton()->space_get_direct_state(find_world_2d()->get_space());

	if (physics_has_last_mousepos) {
		// If no mouse event exists, create a motion one. This is necessary because objects or camera may have moved.
		// While this extra event is sent, it is checked if both camera and last object and last ID did not move.
		// If nothing changed, the event is discarded to avoid flooding with unnecessary motion events every frame.
		bool has_mouse_event = false;
		for (const Ref<InputEvent> &m : physics_picking_events) {
			if (m.is_valid()) {
				has_mouse_event = true;
				break;
			}
		}

		if (!has_mouse_event) {
			Ref<InputEventMouseMotion> mm;
			mm.instantiate();

			mm->set_device(InputEvent::DEVICE_ID_INTERNAL);
			mm->set_global_position(physics_last_mousepos);
			mm->set_position(physics_last_mousepos);
			mm->set_alt_pressed(physics_last_mouse_state.alt);
			mm->set_shift_pressed(physics_last_mouse_state.shift);
			mm->set_ctrl_pressed(physics_last_mouse_state.control);
			mm->set_meta_pressed(physics_last_mouse_state.meta);
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
			physics_last_mouse_state.alt = mm->is_alt_pressed();
			physics_last_mouse_state.shift = mm->is_shift_pressed();
			physics_last_mouse_state.control = mm->is_ctrl_pressed();
			physics_last_mouse_state.meta = mm->is_meta_pressed();
			physics_last_mouse_state.mouse_mask = mm->get_button_mask();
		}

		Ref<InputEventMouseButton> mb = ev;

		if (mb.is_valid()) {
			pos = mb->get_position();
			is_mouse = true;

			physics_has_last_mousepos = true;
			physics_last_mousepos = pos;
			physics_last_mouse_state.alt = mb->is_alt_pressed();
			physics_last_mouse_state.shift = mb->is_shift_pressed();
			physics_last_mouse_state.control = mb->is_ctrl_pressed();
			physics_last_mouse_state.meta = mb->is_meta_pressed();

			if (mb->is_pressed()) {
				physics_last_mouse_state.mouse_mask |= mouse_button_to_mask(mb->get_button_index());
			} else {
				physics_last_mouse_state.mouse_mask &= ~mouse_button_to_mask(mb->get_button_index());

				// If touch mouse raised, assume we don't know last mouse pos until new events come
				if (mb->get_device() == InputEvent::DEVICE_ID_TOUCH_MOUSE) {
					physics_has_last_mousepos = false;
				}
			}
		}

		Ref<InputEventKey> k = ev;
		if (k.is_valid()) {
			// Only for mask.
			physics_last_mouse_state.alt = k->is_alt_pressed();
			physics_last_mouse_state.shift = k->is_shift_pressed();
			physics_last_mouse_state.control = k->is_ctrl_pressed();
			physics_last_mouse_state.meta = k->is_meta_pressed();
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
			// Send to 2D.

			uint64_t frame = get_tree()->get_frame();

			PhysicsDirectSpaceState2D::ShapeResult res[64];
			for (Set<CanvasLayer *>::Element *E = canvas_layers.front(); E; E = E->next()) {
				Transform2D canvas_transform;
				ObjectID canvas_layer_id;
				if (E->get()) {
					// A descendant CanvasLayer.
					canvas_transform = E->get()->get_transform();
					canvas_layer_id = E->get()->get_instance_id();
				} else {
					// This Viewport's builtin canvas.
					canvas_transform = get_canvas_transform();
					canvas_layer_id = ObjectID();
				}

				Vector2 point = canvas_transform.affine_inverse().xform(pos);

				PhysicsDirectSpaceState2D::PointParameters point_params;
				point_params.position = point;
				point_params.canvas_instance_id = canvas_layer_id;
				point_params.collide_with_areas = true;
				point_params.pick_point = true;

				int rc = ss2d->intersect_point(point_params, res, 64);
				for (int i = 0; i < rc; i++) {
					if (res[i].collider_id.is_valid() && res[i].collider) {
						CollisionObject2D *co = Object::cast_to<CollisionObject2D>(res[i].collider);
						if (co && co->can_process()) {
							bool send_event = true;
							if (is_mouse) {
								Map<ObjectID, uint64_t>::Element *F = physics_2d_mouseover.find(res[i].collider_id);
								if (!F) {
									physics_2d_mouseover.insert(res[i].collider_id, frame);
									co->_mouse_enter();
								} else {
									F->get() = frame;
									// It was already hovered, so don't send the event if it's faked.
									if (mm.is_valid() && mm->get_device() == InputEvent::DEVICE_ID_INTERNAL) {
										send_event = false;
									}
								}
								Map<Pair<ObjectID, int>, uint64_t, PairSort<ObjectID, int>>::Element *SF = physics_2d_shape_mouseover.find(Pair(res[i].collider_id, res[i].shape));
								if (!SF) {
									physics_2d_shape_mouseover.insert(Pair(res[i].collider_id, res[i].shape), frame);
									co->_mouse_shape_enter(res[i].shape);
								} else {
									SF->get() = frame;
								}
							}

							if (send_event) {
								co->_input_event_call(this, ev, res[i].shape);
							}
						}
					}
				}
			}

			if (is_mouse) {
				_cleanup_mouseover_colliders(false, false, frame);
			}
		}

#ifndef _3D_DISABLED
		bool captured = false;

		if (physics_object_capture.is_valid()) {
			CollisionObject3D *co = Object::cast_to<CollisionObject3D>(ObjectDB::get_instance(physics_object_capture));
			if (co && camera_3d) {
				_collision_object_3d_input_event(co, camera_3d, ev, Vector3(), Vector3(), 0);
				captured = true;
				if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
					physics_object_capture = ObjectID();
				}

			} else {
				physics_object_capture = ObjectID();
			}
		}

		if (captured) {
			// None.
		} else if (pos == last_pos) {
			if (last_id.is_valid()) {
				if (ObjectDB::get_instance(last_id) && last_object) {
					// Good, exists.
					_collision_object_3d_input_event(last_object, camera_3d, ev, result.position, result.normal, result.shape);
					if (last_object->get_capture_input_on_drag() && mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
						physics_object_capture = last_id;
					}
				}
			}
		} else {
			if (camera_3d) {
				Vector3 from = camera_3d->project_ray_origin(pos);
				Vector3 dir = camera_3d->project_ray_normal(pos);
				real_t far = camera_3d->far;

				PhysicsDirectSpaceState3D *space = PhysicsServer3D::get_singleton()->space_get_direct_state(find_world_3d()->get_space());
				if (space) {
					PhysicsDirectSpaceState3D::RayParameters ray_params;
					ray_params.from = from;
					ray_params.to = from + dir * far;
					ray_params.collide_with_areas = true;
					ray_params.pick_ray = true;

					bool col = space->intersect_ray(ray_params, result);
					ObjectID new_collider;
					if (col) {
						CollisionObject3D *co = Object::cast_to<CollisionObject3D>(result.collider);
						if (co && co->can_process()) {
							_collision_object_3d_input_event(co, camera_3d, ev, result.position, result.normal, result.shape);
							last_object = co;
							last_id = result.collider_id;
							new_collider = last_id;
							if (co->get_capture_input_on_drag() && mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
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
#endif // _3D_DISABLED
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
	if (size == p_size && size_allocated == p_allocated && stretch_transform == p_stretch_transform && p_size_2d_override == size_2d_override && to_screen_rect == p_to_screen_rect) {
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

	emit_signal(SNAME("size_changed"));
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

void Viewport::_update_audio_listener_2d() {
	if (AudioServer::get_singleton()) {
		AudioServer::get_singleton()->notify_listener_changed();
	}
}

void Viewport::set_as_audio_listener_2d(bool p_enable) {
	if (p_enable == is_audio_listener_2d_enabled) {
		return;
	}

	is_audio_listener_2d_enabled = p_enable;
	_update_audio_listener_2d();
}

bool Viewport::is_audio_listener_2d() const {
	return is_audio_listener_2d_enabled;
}

AudioListener2D *Viewport::get_audio_listener_2d() const {
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

void Viewport::_camera_2d_set(Camera2D *p_camera_2d) {
	camera_2d = p_camera_2d;
}

void Viewport::_audio_listener_2d_set(AudioListener2D *p_listener) {
	if (audio_listener_2d == p_listener) {
		return;
	} else if (audio_listener_2d) {
		audio_listener_2d->clear_current();
	}
	audio_listener_2d = p_listener;
}

void Viewport::_audio_listener_2d_remove(AudioListener2D *p_listener) {
	if (audio_listener_2d == p_listener) {
		audio_listener_2d = nullptr;
	}
}

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
		WARN_PRINT("Unable to use parent world_2d as world_2d");
		return;
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_remove_canvas(viewport, current_canvas);
	}

	if (p_world_2d.is_valid()) {
		world_2d = p_world_2d;
	} else {
		WARN_PRINT("Invalid world_2d");
		world_2d = Ref<World2D>(memnew(World2D));
	}

	_update_audio_listener_2d();

	if (is_inside_tree()) {
		current_canvas = find_world_2d()->get_canvas();
		RenderingServer::get_singleton()->viewport_attach_canvas(viewport, current_canvas);
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

Ref<World2D> Viewport::get_world_2d() const {
	return world_2d;
}

Camera2D *Viewport::get_camera_2d() const {
	return camera_2d;
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
	shadow_atlas_size = p_size;
	RS::get_singleton()->viewport_set_shadow_atlas_size(viewport, p_size, shadow_atlas_16_bits);
}

int Viewport::get_shadow_atlas_size() const {
	return shadow_atlas_size;
}

void Viewport::set_shadow_atlas_16_bits(bool p_16_bits) {
	if (shadow_atlas_16_bits == p_16_bits) {
		return;
	}

	shadow_atlas_16_bits = p_16_bits;
	RS::get_singleton()->viewport_set_shadow_atlas_size(viewport, shadow_atlas_size, shadow_atlas_16_bits);
}

bool Viewport::get_shadow_atlas_16_bits() const {
	return shadow_atlas_16_bits;
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
		pre_xf.scale(Vector2(size) / to_screen_rect.size);
	}

	return pre_xf;
}

Ref<InputEvent> Viewport::_make_input_local(const Ref<InputEvent> &ev) {
	if (ev.is_null()) {
		return ev; // No transformation defined for null event
	}

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
	if (gui.tooltip_timer.is_valid()) {
		gui.tooltip_timer->release_connections();
		gui.tooltip_timer = Ref<SceneTreeTimer>();
	}
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

		// Temporary solution for PopupMenus.
		PopupMenu *menu = Object::cast_to<PopupMenu>(this);
		if (menu) {
			tooltip = menu->get_tooltip(pos);
		}

		if (r_tooltip_owner) {
			*r_tooltip_owner = p_control;
		}

		// If we found a tooltip, we stop here.
		if (!tooltip.is_empty()) {
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
	tooltip_text = tooltip_text.strip_edges();
	if (tooltip_text.is_empty()) {
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

	// Popup window which houses the tooltip content.
	TooltipPanel *panel = memnew(TooltipPanel);

	// Controls can implement `make_custom_tooltip` to provide their own tooltip.
	// This should be a Control node which will be added as child to a TooltipPanel.
	Control *base_tooltip = tooltip_owner->make_custom_tooltip(tooltip_text);

	// If no custom tooltip is given, use a default implementation.
	if (!base_tooltip) {
		gui.tooltip_label = memnew(TooltipLabel);
		gui.tooltip_label->set_auto_translate(gui.tooltip_control->is_auto_translating());
		gui.tooltip_label->set_text(tooltip_text);
		base_tooltip = gui.tooltip_label;
		panel->connect("mouse_entered", callable_mp(this, &Viewport::_gui_cancel_tooltip));
	}

	base_tooltip->set_anchors_and_offsets_preset(Control::PRESET_WIDE);

	panel->set_transient(false);
	panel->set_flag(Window::FLAG_NO_FOCUS, true);
	panel->set_wrap_controls(true);
	panel->add_child(base_tooltip);

	gui.tooltip_popup = panel;

	tooltip_owner->add_child(gui.tooltip_popup);

	Point2 tooltip_offset = ProjectSettings::get_singleton()->get("display/mouse_cursor/tooltip_position_offset");
	Rect2 r(gui.tooltip_pos + tooltip_offset, gui.tooltip_popup->get_contents_minimum_size());

	Window *window = gui.tooltip_popup->get_parent_visible_window();
	Rect2i vr = window->get_usable_parent_rect();

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

	gui.tooltip_popup->set_current_screen(window->get_current_screen());
	gui.tooltip_popup->set_position(r.position);
	gui.tooltip_popup->set_size(r.size);

	gui.tooltip_popup->show();
	gui.tooltip_popup->child_controls_changed();
}

void Viewport::_gui_call_input(Control *p_control, const Ref<InputEvent> &p_input) {
	Ref<InputEvent> ev = p_input;

	// Mouse wheel events can't be stopped.
	Ref<InputEventMouseButton> mb = p_input;

	bool cant_stop_me_now = (mb.is_valid() &&
			(mb->get_button_index() == MouseButton::WHEEL_DOWN ||
					mb->get_button_index() == MouseButton::WHEEL_UP ||
					mb->get_button_index() == MouseButton::WHEEL_LEFT ||
					mb->get_button_index() == MouseButton::WHEEL_RIGHT));
	Ref<InputEventPanGesture> pn = p_input;
	cant_stop_me_now = pn.is_valid() || cant_stop_me_now;

	bool ismouse = ev.is_valid() || Object::cast_to<InputEventMouseMotion>(*p_input) != nullptr;

	CanvasItem *ci = p_control;
	while (ci) {
		Control *control = Object::cast_to<Control>(ci);
		if (control) {
			if (control->data.mouse_filter != Control::MOUSE_FILTER_IGNORE) {
				control->_call_gui_input(ev);
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

		ev = ev->xformed_by(ci->get_transform()); // Transform event upwards.
		ci = ci->get_parent_item();
	}
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
}

Control *Viewport::gui_find_control(const Point2 &p_global) {
	// Handle subwindows.
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
		return nullptr; // Canvas item hidden, discard.
	}

	Transform2D matrix = p_xform * p_node->get_transform();
	// matrix.basis_determinant() == 0.0f implies that node does not exist on scene
	if (matrix.basis_determinant() == 0.0f) {
		return nullptr;
	}

	Control *c = Object::cast_to<Control>(p_node);

	if (!c || !c->is_clipping_contents() || c->has_point(matrix.affine_inverse().xform(p_global))) {
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

	if (!c || c->data.mouse_filter == Control::MOUSE_FILTER_IGNORE) {
		return nullptr;
	}

	matrix.affine_invert();
	if (!c->has_point(matrix.xform(p_global))) {
		return nullptr;
	}

	Control *drag_preview = _gui_get_drag_preview();
	if (!drag_preview || (c != drag_preview && !drag_preview->is_ancestor_of(c))) {
		r_inv_xform = matrix;
		return c;
	}

	return nullptr;
}

bool Viewport::_gui_drop(Control *p_at_control, Point2 p_at_pos, bool p_just_check) {
	// Attempt grab, try parent controls too.
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

	return false;
}

void Viewport::_gui_input_event(Ref<InputEvent> p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		gui.key_event_accepted = false;

		Control *over = nullptr;

		Point2 mpos = mb->get_position();
		gui.last_mouse_pos = mpos;
		if (mb->is_pressed()) {
			Size2 pos = mpos;
			if (gui.mouse_focus_mask != MouseButton::NONE) {
				// Do not steal mouse focus and stuff while a focus mask exists.
				gui.mouse_focus_mask |= mouse_button_to_mask(mb->get_button_index());
			} else {
				gui.mouse_focus = gui_find_control(pos);
				gui.last_mouse_focus = gui.mouse_focus;

				if (!gui.mouse_focus) {
					gui.mouse_focus_mask = MouseButton::NONE;
					return;
				}

				gui.mouse_focus_mask = mouse_button_to_mask(mb->get_button_index());

				if (mb->get_button_index() == MouseButton::LEFT) {
					gui.drag_accum = Vector2();
					gui.drag_attempted = false;
				}
			}

			mb = mb->xformed_by(Transform2D()); // Make a copy of the event.

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

			if (mb->get_button_index() == MouseButton::LEFT) { // Assign focus.
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

			if (gui.drag_data.get_type() != Variant::NIL && mb->get_button_index() == MouseButton::LEFT) {
				// Alternate drop use (when using force_drag(), as proposed by #5342).
				gui.drag_successful = false;
				if (gui.mouse_focus) {
					gui.drag_successful = _gui_drop(gui.mouse_focus, pos, false);
				}

				gui.drag_data = Variant();
				gui.dragging = false;

				Control *drag_preview = _gui_get_drag_preview();
				if (drag_preview) {
					memdelete(drag_preview);
					gui.drag_preview_id = ObjectID();
				}
				_propagate_viewport_notification(this, NOTIFICATION_DRAG_END);
				// Change mouse accordingly.
			}

			_gui_cancel_tooltip();
		} else {
			if (gui.drag_data.get_type() != Variant::NIL && mb->get_button_index() == MouseButton::LEFT) {
				gui.drag_successful = false;
				if (gui.drag_mouse_over) {
					gui.drag_successful = _gui_drop(gui.drag_mouse_over, gui.drag_mouse_over_pos, false);
				}

				Control *drag_preview = _gui_get_drag_preview();
				if (drag_preview) {
					memdelete(drag_preview);
					gui.drag_preview_id = ObjectID();
				}

				gui.drag_data = Variant();
				gui.dragging = false;
				gui.drag_mouse_over = nullptr;
				_propagate_viewport_notification(this, NOTIFICATION_DRAG_END);
				// Change mouse accordingly.
			}

			gui.mouse_focus_mask &= ~mouse_button_to_mask(mb->get_button_index()); // Remove from mask.

			if (!gui.mouse_focus) {
				// Release event is only sent if a mouse focus (previously pressed button) exists.
				return;
			}

			Size2 pos = mpos;

			mb = mb->xformed_by(Transform2D()); // Make a copy.
			mb->set_global_position(pos);
			pos = gui.focus_inv_xform.xform(pos);
			mb->set_position(pos);

			Control *mouse_focus = gui.mouse_focus;

			// Disable mouse focus if needed before calling input,
			// this makes popups on mouse press event work better,
			// as the release will never be received otherwise.
			if (gui.mouse_focus_mask == MouseButton::NONE) {
				gui.mouse_focus = nullptr;
				gui.forced_mouse_focus = false;
			}

			if (mouse_focus && mouse_focus->can_process()) {
				_gui_call_input(mouse_focus, mb);
			}

			// In case the mouse was released after for example dragging a scrollbar,
			// check whether the current control is different from the stored one. If
			// it is different, rather than wait for it to be updated the next time the
			// mouse is moved, notify the control so that it can e.g. drop the highlight.
			// This code is duplicated from the mm.is_valid()-case further below.
			if (gui.mouse_focus) {
				over = gui.mouse_focus;
			} else {
				over = gui_find_control(mpos);
			}

			if (gui.mouse_focus_mask == MouseButton::NONE && over != gui.mouse_over) {
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

		// Drag & drop.
		if (!gui.drag_attempted && gui.mouse_focus && (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
			gui.drag_accum += mm->get_relative();
			float len = gui.drag_accum.length();
			if (len > 10) {
				{ // Attempt grab, try parent controls too.
					CanvasItem *ci = gui.mouse_focus;
					while (ci) {
						Control *control = Object::cast_to<Control>(ci);
						if (control) {
							gui.dragging = true;
							gui.drag_data = control->get_drag_data(control->get_global_transform_with_canvas().affine_inverse().xform(mpos) - gui.drag_accum);
							if (gui.drag_data.get_type() != Variant::NIL) {
								gui.mouse_focus = nullptr;
								gui.forced_mouse_focus = false;
								gui.mouse_focus_mask = MouseButton::NONE;
								break;
							} else {
								Control *drag_preview = _gui_get_drag_preview();
								if (drag_preview) {
									ERR_PRINT("Don't set a drag preview and return null data. Preview was deleted and drag request ignored.");
									memdelete(drag_preview);
									gui.drag_preview_id = ObjectID();
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
			over = gui_find_control(mpos);
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
			Vector2 velocity = localizer.basis_xform(mm->get_velocity());
			Vector2 rel = localizer.basis_xform(mm->get_relative());

			mm = mm->xformed_by(Transform2D()); // Make a copy.

			mm->set_global_position(mpos);
			mm->set_velocity(velocity);
			mm->set_relative(rel);

			if (mm->get_button_mask() == MouseButton::NONE) {
				// Nothing pressed.

				bool is_tooltip_shown = false;

				if (gui.tooltip_popup) {
					if (gui.tooltip_control) {
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
								is_tooltip_shown = true; // Nothing to compare against, likely using custom control, so if it changes there is nothing we can do.
							}
						}
					} else {
						_gui_cancel_tooltip();
					}
				}

				if (!is_tooltip_shown && over->can_process()) {
					if (gui.tooltip_timer.is_valid()) {
						gui.tooltip_timer->release_connections();
						gui.tooltip_timer = Ref<SceneTreeTimer>();
					}
					gui.tooltip_control = over;
					gui.tooltip_pos = over->get_screen_transform().xform(pos);
					gui.tooltip_timer = get_tree()->create_timer(gui.tooltip_delay);
					gui.tooltip_timer->set_ignore_time_scale(true);
					gui.tooltip_timer->connect("timeout", callable_mp(this, &Viewport::_gui_show_tooltip));
				}
			}

			mm->set_position(pos);

			Control::CursorShape cursor_shape = Control::CURSOR_ARROW;
			{
				Control *c = over;
				Vector2 cpos = pos;
				while (c) {
					if (gui.mouse_focus_mask != MouseButton::NONE || c->has_point(cpos)) {
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
			// Handle drag & drop.

			Control *drag_preview = _gui_get_drag_preview();
			if (drag_preview) {
				drag_preview->set_position(mpos);
			}

			gui.drag_mouse_over = over;
			gui.drag_mouse_over_pos = Vector2();

			// Find the window this is above of.
			// See if there is an embedder.
			Viewport *embedder = nullptr;
			Vector2 viewport_pos;

			if (is_embedding_subwindows()) {
				embedder = this;
				viewport_pos = mpos;
			} else {
				// Not an embedder, but may be a subwindow of an embedder.
				Window *w = Object::cast_to<Window>(this);
				if (w) {
					if (w->is_embedded()) {
						embedder = w->_get_embedder();

						Transform2D ai = (get_final_transform().affine_inverse() * _get_input_pre_xform()).affine_inverse();

						viewport_pos = ai.xform(mpos) + w->get_position(); // To parent coords.
					}
				}
			}

			Viewport *viewport_under = nullptr;

			if (embedder) {
				// Use embedder logic.

				for (int i = embedder->gui.sub_windows.size() - 1; i >= 0; i--) {
					Window *sw = embedder->gui.sub_windows[i].window;
					Rect2 swrect = Rect2i(sw->get_position(), sw->get_size());
					if (!sw->get_flag(Window::FLAG_BORDERLESS)) {
						int title_height = sw->get_theme_constant(SNAME("title_height"));
						swrect.position.y -= title_height;
						swrect.size.y += title_height;
					}

					if (swrect.has_point(viewport_pos)) {
						viewport_under = sw;
						viewport_pos -= sw->get_position();
					}
				}

				if (!viewport_under) {
					// Not in a subwindow, likely in embedder.
					viewport_under = embedder;
				}
			} else {
				// Use DisplayServer logic.
				Vector2i screen_mouse_pos = DisplayServer::get_singleton()->mouse_get_position();

				DisplayServer::WindowID window_id = DisplayServer::get_singleton()->get_window_at_screen_position(screen_mouse_pos);

				if (window_id != DisplayServer::INVALID_WINDOW_ID) {
					ObjectID object_under = DisplayServer::get_singleton()->window_get_attached_instance_id(window_id);

					if (object_under != ObjectID()) { // Fetch window.
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
				// Find control under at position.
				gui.drag_mouse_over = viewport_under->gui_find_control(viewport_pos);
				if (gui.drag_mouse_over) {
					Transform2D localizer = gui.drag_mouse_over->get_global_transform_with_canvas().affine_inverse();
					gui.drag_mouse_over_pos = localizer.xform(viewport_pos);

					bool can_drop = _gui_drop(gui.drag_mouse_over, gui.drag_mouse_over_pos, true);

					if (!can_drop) {
						ds_cursor_shape = DisplayServer::CURSOR_FORBIDDEN;
					} else {
						ds_cursor_shape = DisplayServer::CURSOR_CAN_DROP;
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
			Control *over = gui_find_control(pos);
			if (over) {
				if (over->can_process()) {
					touch_event = touch_event->xformed_by(Transform2D()); // Make a copy.
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
				touch_event = touch_event->xformed_by(Transform2D()); // Make a copy.
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

		Control *over = gui_find_control(pos);
		if (over) {
			if (over->can_process()) {
				gesture_event = gesture_event->xformed_by(Transform2D()); // Make a copy.
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
			over = gui_find_control(drag_event->get_position());
		}
		if (over) {
			if (over->can_process()) {
				Transform2D localizer = over->get_global_transform_with_canvas().affine_inverse();
				Size2 pos = localizer.xform(drag_event->get_position());
				Vector2 velocity = localizer.basis_xform(drag_event->get_velocity());
				Vector2 rel = localizer.basis_xform(drag_event->get_relative());

				drag_event = drag_event->xformed_by(Transform2D()); // Make a copy.

				drag_event->set_velocity(velocity);
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
				gui.key_focus->_call_gui_input(p_event);
			}

			if (gui.key_event_accepted) {
				set_input_as_handled();
				return;
			}
		}

		Control *from = gui.key_focus ? gui.key_focus : nullptr;

		if (from && p_event->is_pressed()) {
			Control *next = nullptr;

			if (p_event->is_action_pressed("ui_focus_next", true, true)) {
				next = from->find_next_valid_focus();
			}

			if (p_event->is_action_pressed("ui_focus_prev", true, true)) {
				next = from->find_prev_valid_focus();
			}

			if (p_event->is_action_pressed("ui_up", true, true)) {
				next = from->_get_focus_neighbor(SIDE_TOP);
			}

			if (p_event->is_action_pressed("ui_left", true, true)) {
				next = from->_get_focus_neighbor(SIDE_LEFT);
			}

			if (p_event->is_action_pressed("ui_right", true, true)) {
				next = from->_get_focus_neighbor(SIDE_RIGHT);
			}

			if (p_event->is_action_pressed("ui_down", true, true)) {
				next = from->_get_focus_neighbor(SIDE_BOTTOM);
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
	_propagate_viewport_notification(this, NOTIFICATION_DRAG_BEGIN);
}

void Viewport::_gui_set_drag_preview(Control *p_base, Control *p_control) {
	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND(!Object::cast_to<Control>((Object *)p_control));
	ERR_FAIL_COND(p_control->is_inside_tree());
	ERR_FAIL_COND(p_control->get_parent() != nullptr);

	Control *drag_preview = _gui_get_drag_preview();
	if (drag_preview) {
		memdelete(drag_preview);
	}
	p_control->set_as_top_level(true);
	p_control->set_position(gui.last_mouse_pos);
	p_base->get_root_parent_control()->add_child(p_control); // Add as child of viewport.
	p_control->raise();

	gui.drag_preview_id = p_control->get_instance_id();
}

Control *Viewport::_gui_get_drag_preview() {
	if (gui.drag_preview_id.is_null()) {
		return nullptr;
	} else {
		Control *drag_preview = Object::cast_to<Control>(ObjectDB::get_instance(gui.drag_preview_id));
		if (!drag_preview) {
			ERR_PRINT("Don't free the control set as drag preview.");
			gui.drag_preview_id = ObjectID();
		}
		return drag_preview;
	}
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
		gui.mouse_focus_mask = MouseButton::NONE;
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
	ERR_FAIL_COND_V(!is_inside_tree(), nullptr);

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
	if (gui.key_focus && gui.key_focus == p_control) {
		// No need for change.
		return;
	}
	get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, "_viewports", "_gui_remove_focus_for_window", (Node *)get_base_window());
	gui.key_focus = p_control;
	emit_signal(SNAME("gui_focus_changed"), p_control);
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
	MouseButton mask = gui.mouse_focus_mask;
	gui.mouse_focus = nullptr;
	gui.forced_mouse_focus = false;
	gui.mouse_focus_mask = MouseButton::NONE;

	for (int i = 0; i < 3; i++) {
		if ((int)mask & (1 << i)) {
			Ref<InputEventMouseButton> mb;
			mb.instantiate();
			mb->set_position(c->get_local_mouse_position());
			mb->set_global_position(c->get_local_mouse_position());
			mb->set_button_index(MouseButton(i + 1));
			mb->set_pressed(false);
			c->_call_gui_input(mb);
		}
	}
}

void Viewport::_drop_physics_mouseover(bool p_paused_only) {
	physics_has_last_mousepos = false;

	_cleanup_mouseover_colliders(true, p_paused_only);

#ifndef _3D_DISABLED
	if (physics_object_over.is_valid()) {
		CollisionObject3D *co = Object::cast_to<CollisionObject3D>(ObjectDB::get_instance(physics_object_over));
		if (co) {
			if (!co->is_inside_tree()) {
				physics_object_over = ObjectID();
				physics_object_capture = ObjectID();
			} else if (!(p_paused_only && co->can_process())) {
				co->_mouse_exit();
				physics_object_over = ObjectID();
				physics_object_capture = ObjectID();
			}
		}
	}
#endif // _3D_DISABLED
}

void Viewport::_cleanup_mouseover_colliders(bool p_clean_all_frames, bool p_paused_only, uint64_t p_frame_reference) {
	List<Map<ObjectID, uint64_t>::Element *> to_erase;

	for (Map<ObjectID, uint64_t>::Element *E = physics_2d_mouseover.front(); E; E = E->next()) {
		if (!p_clean_all_frames && E->get() == p_frame_reference) {
			continue;
		}

		Object *o = ObjectDB::get_instance(E->key());
		if (o) {
			CollisionObject2D *co = Object::cast_to<CollisionObject2D>(o);
			if (co && co->is_inside_tree()) {
				if (p_clean_all_frames && p_paused_only && co->can_process()) {
					continue;
				}
				co->_mouse_exit();
			}
		}
		to_erase.push_back(E);
	}

	while (to_erase.size()) {
		physics_2d_mouseover.erase(to_erase.front()->get());
		to_erase.pop_front();
	}

	// Per-shape.
	List<Map<Pair<ObjectID, int>, uint64_t, PairSort<ObjectID, int>>::Element *> shapes_to_erase;

	for (Map<Pair<ObjectID, int>, uint64_t, PairSort<ObjectID, int>>::Element *E = physics_2d_shape_mouseover.front(); E; E = E->next()) {
		if (!p_clean_all_frames && E->get() == p_frame_reference) {
			continue;
		}

		Object *o = ObjectDB::get_instance(E->key().first);
		if (o) {
			CollisionObject2D *co = Object::cast_to<CollisionObject2D>(o);
			if (co && co->is_inside_tree()) {
				if (p_clean_all_frames && p_paused_only && co->can_process()) {
					continue;
				}
				co->_mouse_shape_exit(E->key().second);
			}
		}
		shapes_to_erase.push_back(E);
	}

	while (shapes_to_erase.size()) {
		physics_2d_shape_mouseover.erase(shapes_to_erase.front()->get());
		shapes_to_erase.pop_front();
	}
}

Control *Viewport::_gui_get_focus_owner() {
	return gui.key_focus;
}

void Viewport::_gui_grab_click_focus(Control *p_control) {
	gui.mouse_click_grabber = p_control;
	call_deferred(SNAME("_post_gui_grab_click_focus"));
}

void Viewport::_post_gui_grab_click_focus() {
	Control *focus_grabber = gui.mouse_click_grabber;
	if (!focus_grabber) {
		// Redundant grab requests were made.
		return;
	}
	gui.mouse_click_grabber = nullptr;

	if (gui.mouse_focus) {
		if (gui.mouse_focus == focus_grabber) {
			return;
		}

		MouseButton mask = gui.mouse_focus_mask;
		Point2 click = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);

		for (int i = 0; i < 3; i++) {
			if ((int)mask & (1 << i)) {
				Ref<InputEventMouseButton> mb;
				mb.instantiate();

				// Send unclick.

				mb->set_position(click);
				mb->set_button_index(MouseButton(i + 1));
				mb->set_pressed(false);
				gui.mouse_focus->_call_gui_input(mb);
			}
		}

		gui.mouse_focus = focus_grabber;
		gui.focus_inv_xform = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse();
		click = gui.mouse_focus->get_global_transform_with_canvas().affine_inverse().xform(gui.last_mouse_pos);

		for (int i = 0; i < 3; i++) {
			if ((int)mask & (1 << i)) {
				Ref<InputEventMouseButton> mb;
				mb.instantiate();

				// Send click.

				mb->set_position(click);
				mb->set_button_index(MouseButton(i + 1));
				mb->set_pressed(true);
				MessageQueue::get_singleton()->push_callable(callable_mp(gui.mouse_focus, &Control::_call_gui_input), mb);
			}
		}
	}
}

///////////////////////////////

void Viewport::push_text_input(const String &p_text) {
	if (gui.subwindow_focused) {
		gui.subwindow_focused->push_text_input(p_text);
		return;
	}

	if (gui.key_focus) {
		gui.key_focus->call("set_text", p_text);
	}
}

Viewport::SubWindowResize Viewport::_sub_window_get_resize_margin(Window *p_subwindow, const Point2 &p_point) {
	if (p_subwindow->get_flag(Window::FLAG_BORDERLESS) || p_subwindow->get_flag(Window::FLAG_RESIZE_DISABLED)) {
		return SUB_WINDOW_RESIZE_DISABLED;
	}

	Rect2i r = Rect2i(p_subwindow->get_position(), p_subwindow->get_size());

	int title_height = p_subwindow->get_theme_constant(SNAME("title_height"));

	r.position.y -= title_height;
	r.size.y += title_height;

	if (r.has_point(p_point)) {
		return SUB_WINDOW_RESIZE_DISABLED; // It's inside, so no resize.
	}

	int dist_x = p_point.x < r.position.x ? (p_point.x - r.position.x) : (p_point.x > (r.position.x + r.size.x) ? (p_point.x - (r.position.x + r.size.x)) : 0);
	int dist_y = p_point.y < r.position.y ? (p_point.y - r.position.y) : (p_point.y > (r.position.y + r.size.y) ? (p_point.y - (r.position.y + r.size.y)) : 0);

	int limit = p_subwindow->get_theme_constant(SNAME("resize_margin"));

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
		if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			if (gui.subwindow_drag == SUB_WINDOW_DRAG_CLOSE) {
				if (gui.subwindow_drag_close_rect.has_point(mb->get_position())) {
					// Close window.
					gui.subwindow_focused->_event_callback(DisplayServer::WINDOW_EVENT_CLOSE_REQUEST);
				}
			}
			gui.subwindow_drag = SUB_WINDOW_DRAG_DISABLED;
			if (gui.subwindow_focused != nullptr) { // May have been erased.
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

					int title_height = gui.subwindow_focused->get_flag(Window::FLAG_BORDERLESS) ? 0 : gui.subwindow_focused->get_theme_constant(SNAME("title_height"));

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

			if (gui.subwindow_focused) { // May have been erased.
				_sub_window_update(gui.subwindow_focused);
			}
		}

		return true; // Handled.
	}
	Ref<InputEventMouseButton> mb = p_event;
	// If the event is a mouse button, we need to check whether another window was clicked.

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		bool click_on_window = false;
		for (int i = gui.sub_windows.size() - 1; i >= 0; i--) {
			SubWindow sw = gui.sub_windows.write[i];

			// Clicked inside window?

			Rect2i r = Rect2i(sw.window->get_position(), sw.window->get_size());

			if (!sw.window->get_flag(Window::FLAG_BORDERLESS)) {
				// Check top bar.
				int title_height = sw.window->get_theme_constant(SNAME("title_height"));
				Rect2i title_bar = r;
				title_bar.position.y -= title_height;
				title_bar.size.y = title_height;

				if (title_bar.has_point(mb->get_position())) {
					click_on_window = true;

					int close_h_ofs = sw.window->get_theme_constant(SNAME("close_h_ofs"));
					int close_v_ofs = sw.window->get_theme_constant(SNAME("close_v_ofs"));
					Ref<Texture2D> close_icon = sw.window->get_theme_icon(SNAME("close"));

					Rect2 close_rect;
					close_rect.position = Vector2(r.position.x + r.size.x - close_v_ofs, r.position.y - close_h_ofs);
					close_rect.size = close_icon->get_size();

					if (gui.subwindow_focused != sw.window) {
						// Refocus.
						_sub_window_grab_focus(sw.window);
					}

					if (close_rect.has_point(mb->get_position())) {
						gui.subwindow_drag = SUB_WINDOW_DRAG_CLOSE;
						gui.subwindow_drag_close_inside = true; // Starts inside.
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
				// Clicked, see if it needs to fetch focus.
				if (gui.subwindow_focused != sw.window) {
					// Refocus.
					_sub_window_grab_focus(sw.window);
				}

				click_on_window = true;
			}

			if (click_on_window) {
				break;
			}
		}

		if (!click_on_window && gui.subwindow_focused) {
			// No window found and clicked, remove focus.
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

				return true; // Reserved for showing the resize cursor.
			}
		}
	}

	if (gui.subwindow_drag != SUB_WINDOW_DRAG_DISABLED) {
		return true; // Dragging, don't pass the event.
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

void Viewport::push_input(const Ref<InputEvent> &p_event, bool p_local_coords) {
	ERR_FAIL_COND(!is_inside_tree());

	if (disable_input) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_ancestor_of(this)) {
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
		get_tree()->_call_input_pause(input_group, SceneTree::CALL_INPUT_TYPE_INPUT, ev, this); //not a bug, must happen before GUI, order is _input -> gui input -> _unhandled input
	}

	if (!is_input_handled()) {
		_gui_input_event(ev);
	}

	event_count++;
}

void Viewport::push_unhandled_input(const Ref<InputEvent> &p_event, bool p_local_coords) {
	ERR_FAIL_COND(p_event.is_null());
	ERR_FAIL_COND(!is_inside_tree());
	local_input_handled = false;

	if (disable_input || !_can_consume_input_events()) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_ancestor_of(this)) {
		return;
	}

	Ref<InputEvent> ev;
	if (!p_local_coords) {
		ev = _make_input_local(p_event);
	} else {
		ev = p_event;
	}

	// Unhandled Input.
	get_tree()->_call_input_pause(unhandled_input_group, SceneTree::CALL_INPUT_TYPE_UNHANDLED_INPUT, ev, this);

	// Unhandled key Input - used for performance reasons - This is called a lot less than _unhandled_input since it ignores MouseMotion, etc.
	if (!is_input_handled() && (Object::cast_to<InputEventKey>(*ev) != nullptr || Object::cast_to<InputEventShortcut>(*ev) != nullptr)) {
		get_tree()->_call_input_pause(unhandled_key_input_group, SceneTree::CALL_INPUT_TYPE_UNHANDLED_KEY_INPUT, ev, this);
	}

	if (physics_object_picking && !is_input_handled()) {
		if (Input::get_singleton()->get_mouse_mode() != Input::MOUSE_MODE_CAPTURED &&
				(Object::cast_to<InputEventMouseButton>(*ev) ||
						Object::cast_to<InputEventMouseMotion>(*ev) ||
						Object::cast_to<InputEventScreenDrag>(*ev) ||
						Object::cast_to<InputEventScreenTouch>(*ev) ||
						Object::cast_to<InputEventKey>(*ev) // To remember state.

						)) {
			physics_picking_events.push_back(ev);
		}
	}
}

void Viewport::set_physics_object_picking(bool p_enable) {
	physics_object_picking = p_enable;
	if (physics_object_picking) {
		add_to_group("_picking_viewports");
	} else {
		physics_picking_events.clear();
		if (is_in_group("_picking_viewports")) {
			remove_from_group("_picking_viewports");
		}
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

TypedArray<String> Viewport::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (size.x == 0 || size.y == 0) {
		warnings.push_back(TTR("Viewport size must be greater than 0 to render anything."));
	}
	return warnings;
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
	if (use_debanding == p_use_debanding) {
		return;
	}
	use_debanding = p_use_debanding;
	RS::get_singleton()->viewport_set_use_debanding(viewport, p_use_debanding);
}

bool Viewport::is_using_debanding() const {
	return use_debanding;
}

void Viewport::set_mesh_lod_threshold(float p_pixels) {
	mesh_lod_threshold = p_pixels;
	RS::get_singleton()->viewport_set_mesh_lod_threshold(viewport, mesh_lod_threshold);
}

float Viewport::get_mesh_lod_threshold() const {
	return mesh_lod_threshold;
}

void Viewport::set_use_occlusion_culling(bool p_use_occlusion_culling) {
	if (use_occlusion_culling == p_use_occlusion_culling) {
		return;
	}

	use_occlusion_culling = p_use_occlusion_culling;
	RS::get_singleton()->viewport_set_use_occlusion_culling(viewport, p_use_occlusion_culling);

	notify_property_list_changed();
}

bool Viewport::is_using_occlusion_culling() const {
	return use_occlusion_culling;
}

void Viewport::set_debug_draw(DebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
	RS::get_singleton()->viewport_set_debug_draw(viewport, RS::ViewportDebugDraw(p_debug_draw));
}

Viewport::DebugDraw Viewport::get_debug_draw() const {
	return debug_draw;
}

int Viewport::get_render_info(RenderInfoType p_type, RenderInfo p_info) {
	return RS::get_singleton()->viewport_get_render_info(viewport, RS::ViewportRenderInfoType(p_type), RS::ViewportRenderInfo(p_info));
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

bool Viewport::gui_is_drag_successful() const {
	return gui.drag_successful;
}

void Viewport::set_input_as_handled() {
	_drop_physics_mouseover();

	if (!handle_input_locally) {
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
		if (vp != this) {
			vp->set_input_as_handled();
			return;
		}
	}

	local_input_handled = true;
}

bool Viewport::is_input_handled() const {
	if (!handle_input_locally) {
		ERR_FAIL_COND_V(!is_inside_tree(), false);
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
		if (vp != this) {
			return vp->is_input_handled();
		}
	}
	return local_input_handled;
}

void Viewport::set_handle_input_locally(bool p_enable) {
	handle_input_locally = p_enable;
}

bool Viewport::is_handling_input_locally() const {
	return handle_input_locally;
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
		gui.mouse_focus_mask = MouseButton::NONE;
	}
}

void Viewport::set_sdf_oversize(SDFOversize p_sdf_oversize) {
	ERR_FAIL_INDEX(p_sdf_oversize, SDF_OVERSIZE_MAX);
	sdf_oversize = p_sdf_oversize;
	RS::get_singleton()->viewport_set_sdf_oversize_and_scale(viewport, RS::ViewportSDFOversize(sdf_oversize), RS::ViewportSDFScale(sdf_scale));
}

Viewport::SDFOversize Viewport::get_sdf_oversize() const {
	return sdf_oversize;
}

void Viewport::set_sdf_scale(SDFScale p_sdf_scale) {
	ERR_FAIL_INDEX(p_sdf_scale, SDF_SCALE_MAX);
	sdf_scale = p_sdf_scale;
	RS::get_singleton()->viewport_set_sdf_oversize_and_scale(viewport, RS::ViewportSDFOversize(sdf_oversize), RS::ViewportSDFScale(sdf_scale));
}

Viewport::SDFScale Viewport::get_sdf_scale() const {
	return sdf_scale;
}

#ifndef _3D_DISABLED
AudioListener3D *Viewport::get_audio_listener_3d() const {
	return audio_listener_3d;
}

void Viewport::set_as_audio_listener_3d(bool p_enable) {
	if (p_enable == is_audio_listener_3d_enabled) {
		return;
	}

	is_audio_listener_3d_enabled = p_enable;
	_update_audio_listener_3d();
}

bool Viewport::is_audio_listener_3d() const {
	return is_audio_listener_3d_enabled;
}

void Viewport::_update_audio_listener_3d() {
	if (AudioServer::get_singleton()) {
		AudioServer::get_singleton()->notify_listener_changed();
	}
}

void Viewport::_listener_transform_3d_changed_notify() {
}

void Viewport::_audio_listener_3d_set(AudioListener3D *p_listener) {
	if (audio_listener_3d == p_listener) {
		return;
	}

	audio_listener_3d = p_listener;

	_update_audio_listener_3d();
	_listener_transform_3d_changed_notify();
}

bool Viewport::_audio_listener_3d_add(AudioListener3D *p_listener) {
	audio_listener_3d_set.insert(p_listener);
	return audio_listener_3d_set.size() == 1;
}

void Viewport::_audio_listener_3d_remove(AudioListener3D *p_listener) {
	audio_listener_3d_set.erase(p_listener);
	if (audio_listener_3d == p_listener) {
		audio_listener_3d = nullptr;
	}
}

void Viewport::_audio_listener_3d_make_next_current(AudioListener3D *p_exclude) {
	if (audio_listener_3d_set.size() > 0) {
		for (Set<AudioListener3D *>::Element *E = audio_listener_3d_set.front(); E; E = E->next()) {
			if (p_exclude == E->get()) {
				continue;
			}
			if (!E->get()->is_inside_tree()) {
				continue;
			}
			if (audio_listener_3d != nullptr) {
				return;
			}

			E->get()->make_current();
		}
	} else {
		// Attempt to reset listener to the camera position.
		if (camera_3d != nullptr) {
			_update_audio_listener_3d();
			_camera_3d_transform_changed_notify();
		}
	}
}

void Viewport::_collision_object_3d_input_event(CollisionObject3D *p_object, Camera3D *p_camera, const Ref<InputEvent> &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape) {
	Transform3D object_transform = p_object->get_global_transform();
	Transform3D camera_transform = p_camera->get_global_transform();
	ObjectID id = p_object->get_instance_id();

	// Avoid sending the fake event unnecessarily if nothing really changed in the context.
	if (object_transform == physics_last_object_transform && camera_transform == physics_last_camera_transform && physics_last_id == id) {
		Ref<InputEventMouseMotion> mm = p_input_event;
		if (mm.is_valid() && mm->get_device() == InputEvent::DEVICE_ID_INTERNAL) {
			return; // Discarded.
		}
	}
	p_object->_input_event_call(camera_3d, p_input_event, p_pos, p_normal, p_shape);
	physics_last_object_transform = object_transform;
	physics_last_camera_transform = camera_transform;
	physics_last_id = id;
}

Camera3D *Viewport::get_camera_3d() const {
	return camera_3d;
}

void Viewport::_camera_3d_transform_changed_notify() {
}

void Viewport::_camera_3d_set(Camera3D *p_camera) {
	if (camera_3d == p_camera) {
		return;
	}

	if (camera_3d) {
		camera_3d->notification(Camera3D::NOTIFICATION_LOST_CURRENT);
	}

	camera_3d = p_camera;

	if (!camera_3d_override) {
		if (camera_3d) {
			RenderingServer::get_singleton()->viewport_attach_camera(viewport, camera_3d->get_camera());
		} else {
			RenderingServer::get_singleton()->viewport_attach_camera(viewport, RID());
		}
	}

	if (camera_3d) {
		camera_3d->notification(Camera3D::NOTIFICATION_BECAME_CURRENT);
	}

	_update_audio_listener_3d();
	_camera_3d_transform_changed_notify();
}

bool Viewport::_camera_3d_add(Camera3D *p_camera) {
	camera_3d_set.insert(p_camera);
	return camera_3d_set.size() == 1;
}

void Viewport::_camera_3d_remove(Camera3D *p_camera) {
	camera_3d_set.erase(p_camera);
	if (camera_3d == p_camera) {
		camera_3d->notification(Camera3D::NOTIFICATION_LOST_CURRENT);
		camera_3d = nullptr;
	}
}

void Viewport::_camera_3d_make_next_current(Camera3D *p_exclude) {
	for (Set<Camera3D *>::Element *E = camera_3d_set.front(); E; E = E->next()) {
		if (p_exclude == E->get()) {
			continue;
		}
		if (!E->get()->is_inside_tree()) {
			continue;
		}
		if (camera_3d != nullptr) {
			return;
		}

		E->get()->make_current();
	}
}

void Viewport::enable_camera_3d_override(bool p_enable) {
	if (p_enable == camera_3d_override) {
		return;
	}

	if (p_enable) {
		camera_3d_override.rid = RenderingServer::get_singleton()->camera_create();
	} else {
		RenderingServer::get_singleton()->free(camera_3d_override.rid);
		camera_3d_override.rid = RID();
	}

	if (p_enable) {
		RenderingServer::get_singleton()->viewport_attach_camera(viewport, camera_3d_override.rid);
	} else if (camera_3d) {
		RenderingServer::get_singleton()->viewport_attach_camera(viewport, camera_3d->get_camera());
	} else {
		RenderingServer::get_singleton()->viewport_attach_camera(viewport, RID());
	}
}

void Viewport::set_camera_3d_override_perspective(real_t p_fovy_degrees, real_t p_z_near, real_t p_z_far) {
	if (camera_3d_override) {
		if (camera_3d_override.fov == p_fovy_degrees && camera_3d_override.z_near == p_z_near &&
				camera_3d_override.z_far == p_z_far && camera_3d_override.projection == Camera3DOverrideData::PROJECTION_PERSPECTIVE) {
			return;
		}

		camera_3d_override.fov = p_fovy_degrees;
		camera_3d_override.z_near = p_z_near;
		camera_3d_override.z_far = p_z_far;
		camera_3d_override.projection = Camera3DOverrideData::PROJECTION_PERSPECTIVE;

		RenderingServer::get_singleton()->camera_set_perspective(camera_3d_override.rid, camera_3d_override.fov, camera_3d_override.z_near, camera_3d_override.z_far);
	}
}

void Viewport::set_camera_3d_override_orthogonal(real_t p_size, real_t p_z_near, real_t p_z_far) {
	if (camera_3d_override) {
		if (camera_3d_override.size == p_size && camera_3d_override.z_near == p_z_near &&
				camera_3d_override.z_far == p_z_far && camera_3d_override.projection == Camera3DOverrideData::PROJECTION_ORTHOGONAL) {
			return;
		}

		camera_3d_override.size = p_size;
		camera_3d_override.z_near = p_z_near;
		camera_3d_override.z_far = p_z_far;
		camera_3d_override.projection = Camera3DOverrideData::PROJECTION_ORTHOGONAL;

		RenderingServer::get_singleton()->camera_set_orthogonal(camera_3d_override.rid, camera_3d_override.size, camera_3d_override.z_near, camera_3d_override.z_far);
	}
}

void Viewport::set_disable_3d(bool p_disable) {
	disable_3d = p_disable;
	RenderingServer::get_singleton()->viewport_set_disable_3d(viewport, disable_3d);
}

bool Viewport::is_3d_disabled() const {
	return disable_3d;
}

bool Viewport::is_camera_3d_override_enabled() const {
	return camera_3d_override;
}

void Viewport::set_camera_3d_override_transform(const Transform3D &p_transform) {
	if (camera_3d_override) {
		camera_3d_override.transform = p_transform;
		RenderingServer::get_singleton()->camera_set_transform(camera_3d_override.rid, p_transform);
	}
}

Transform3D Viewport::get_camera_3d_override_transform() const {
	if (camera_3d_override) {
		return camera_3d_override.transform;
	}

	return Transform3D();
}

Ref<World3D> Viewport::get_world_3d() const {
	return world_3d;
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

void Viewport::set_world_3d(const Ref<World3D> &p_world_3d) {
	if (world_3d == p_world_3d) {
		return;
	}

	if (is_inside_tree()) {
		_propagate_exit_world_3d(this);
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
		_propagate_enter_world_3d(this);
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
	}

	_update_audio_listener_3d();
}

void Viewport::_own_world_3d_changed() {
	ERR_FAIL_COND(world_3d.is_null());
	ERR_FAIL_COND(own_world_3d.is_null());

	if (is_inside_tree()) {
		_propagate_exit_world_3d(this);
	}

	own_world_3d = world_3d->duplicate();

	if (is_inside_tree()) {
		_propagate_enter_world_3d(this);
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
	}

	_update_audio_listener_3d();
}

void Viewport::set_use_own_world_3d(bool p_world_3d) {
	if (p_world_3d == own_world_3d.is_valid()) {
		return;
	}

	if (is_inside_tree()) {
		_propagate_exit_world_3d(this);
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
		_propagate_enter_world_3d(this);
	}

	if (is_inside_tree()) {
		RenderingServer::get_singleton()->viewport_set_scenario(viewport, find_world_3d()->get_scenario());
	}

	_update_audio_listener_3d();
}

bool Viewport::is_using_own_world_3d() const {
	return own_world_3d.is_valid();
}

void Viewport::_propagate_enter_world_3d(Node *p_node) {
	if (p_node != this) {
		if (!p_node->is_inside_tree()) { //may not have entered scene yet
			return;
		}

		if (Object::cast_to<Node3D>(p_node) || Object::cast_to<WorldEnvironment>(p_node)) {
			p_node->notification(Node3D::NOTIFICATION_ENTER_WORLD);
		} else {
			Viewport *v = Object::cast_to<Viewport>(p_node);
			if (v) {
				if (v->world_3d.is_valid() || v->own_world_3d.is_valid()) {
					return;
				}
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_propagate_enter_world_3d(p_node->get_child(i));
	}
}

void Viewport::_propagate_exit_world_3d(Node *p_node) {
	if (p_node != this) {
		if (!p_node->is_inside_tree()) { //may have exited scene already
			return;
		}

		if (Object::cast_to<Node3D>(p_node) || Object::cast_to<WorldEnvironment>(p_node)) {
			p_node->notification(Node3D::NOTIFICATION_EXIT_WORLD);
		} else {
			Viewport *v = Object::cast_to<Viewport>(p_node);
			if (v) {
				if (v->world_3d.is_valid() || v->own_world_3d.is_valid()) {
					return;
				}
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_propagate_exit_world_3d(p_node->get_child(i));
	}
}

void Viewport::set_use_xr(bool p_use_xr) {
	use_xr = p_use_xr;

	RS::get_singleton()->viewport_set_use_xr(viewport, use_xr);
}

bool Viewport::is_using_xr() {
	return use_xr;
}

void Viewport::set_scaling_3d_mode(Scaling3DMode p_scaling_3d_mode) {
	if (scaling_3d_mode == p_scaling_3d_mode) {
		return;
	}

	scaling_3d_mode = p_scaling_3d_mode;
	RS::get_singleton()->viewport_set_scaling_3d_mode(viewport, (RS::ViewportScaling3DMode)(int)p_scaling_3d_mode);
}

Viewport::Scaling3DMode Viewport::get_scaling_3d_mode() const {
	return scaling_3d_mode;
}

void Viewport::set_scaling_3d_scale(float p_scaling_3d_scale) {
	// Clamp to reasonable values that are actually useful.
	// Values above 2.0 don't serve a practical purpose since the viewport
	// isn't displayed with mipmaps.
	scaling_3d_scale = CLAMP(p_scaling_3d_scale, 0.1, 2.0);

	RS::get_singleton()->viewport_set_scaling_3d_scale(viewport, scaling_3d_scale);
}

float Viewport::get_scaling_3d_scale() const {
	return scaling_3d_scale;
}

void Viewport::set_fsr_sharpness(float p_fsr_sharpness) {
	if (fsr_sharpness == p_fsr_sharpness) {
		return;
	}

	if (p_fsr_sharpness < 0.0f) {
		p_fsr_sharpness = 0.0f;
	}

	fsr_sharpness = p_fsr_sharpness;
	RS::get_singleton()->viewport_set_fsr_sharpness(viewport, p_fsr_sharpness);
}

float Viewport::get_fsr_sharpness() const {
	return fsr_sharpness;
}

void Viewport::set_fsr_mipmap_bias(float p_fsr_mipmap_bias) {
	if (fsr_mipmap_bias == p_fsr_mipmap_bias) {
		return;
	}

	fsr_mipmap_bias = p_fsr_mipmap_bias;
	RS::get_singleton()->viewport_set_fsr_mipmap_bias(viewport, p_fsr_mipmap_bias);
}

float Viewport::get_fsr_mipmap_bias() const {
	return fsr_mipmap_bias;
}

#endif // _3D_DISABLED

void Viewport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world_2d", "world_2d"), &Viewport::set_world_2d);
	ClassDB::bind_method(D_METHOD("get_world_2d"), &Viewport::get_world_2d);
	ClassDB::bind_method(D_METHOD("find_world_2d"), &Viewport::find_world_2d);

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

	ClassDB::bind_method(D_METHOD("set_use_occlusion_culling", "enable"), &Viewport::set_use_occlusion_culling);
	ClassDB::bind_method(D_METHOD("is_using_occlusion_culling"), &Viewport::is_using_occlusion_culling);

	ClassDB::bind_method(D_METHOD("set_debug_draw", "debug_draw"), &Viewport::set_debug_draw);
	ClassDB::bind_method(D_METHOD("get_debug_draw"), &Viewport::get_debug_draw);

	ClassDB::bind_method(D_METHOD("get_render_info", "type", "info"), &Viewport::get_render_info);

	ClassDB::bind_method(D_METHOD("get_texture"), &Viewport::get_texture);

	ClassDB::bind_method(D_METHOD("set_physics_object_picking", "enable"), &Viewport::set_physics_object_picking);
	ClassDB::bind_method(D_METHOD("get_physics_object_picking"), &Viewport::get_physics_object_picking);

	ClassDB::bind_method(D_METHOD("get_viewport_rid"), &Viewport::get_viewport_rid);
	ClassDB::bind_method(D_METHOD("push_text_input", "text"), &Viewport::push_text_input);
	ClassDB::bind_method(D_METHOD("push_input", "event", "in_local_coords"), &Viewport::push_input, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("push_unhandled_input", "event", "in_local_coords"), &Viewport::push_unhandled_input, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_camera_2d"), &Viewport::get_camera_2d);
	ClassDB::bind_method(D_METHOD("set_as_audio_listener_2d", "enable"), &Viewport::set_as_audio_listener_2d);
	ClassDB::bind_method(D_METHOD("is_audio_listener_2d"), &Viewport::is_audio_listener_2d);

	ClassDB::bind_method(D_METHOD("get_mouse_position"), &Viewport::get_mouse_position);
	ClassDB::bind_method(D_METHOD("warp_mouse", "to_position"), &Viewport::warp_mouse);

	ClassDB::bind_method(D_METHOD("gui_get_drag_data"), &Viewport::gui_get_drag_data);
	ClassDB::bind_method(D_METHOD("gui_is_dragging"), &Viewport::gui_is_dragging);
	ClassDB::bind_method(D_METHOD("gui_is_drag_successful"), &Viewport::gui_is_drag_successful);

	ClassDB::bind_method(D_METHOD("set_disable_input", "disable"), &Viewport::set_disable_input);
	ClassDB::bind_method(D_METHOD("is_input_disabled"), &Viewport::is_input_disabled);

	ClassDB::bind_method(D_METHOD("_gui_remove_focus_for_window"), &Viewport::_gui_remove_focus_for_window);
	ClassDB::bind_method(D_METHOD("_post_gui_grab_click_focus"), &Viewport::_post_gui_grab_click_focus);

	ClassDB::bind_method(D_METHOD("set_shadow_atlas_size", "size"), &Viewport::set_shadow_atlas_size);
	ClassDB::bind_method(D_METHOD("get_shadow_atlas_size"), &Viewport::get_shadow_atlas_size);

	ClassDB::bind_method(D_METHOD("set_shadow_atlas_16_bits", "enable"), &Viewport::set_shadow_atlas_16_bits);
	ClassDB::bind_method(D_METHOD("get_shadow_atlas_16_bits"), &Viewport::get_shadow_atlas_16_bits);

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

	ClassDB::bind_method(D_METHOD("set_sdf_oversize", "oversize"), &Viewport::set_sdf_oversize);
	ClassDB::bind_method(D_METHOD("get_sdf_oversize"), &Viewport::get_sdf_oversize);

	ClassDB::bind_method(D_METHOD("set_sdf_scale", "scale"), &Viewport::set_sdf_scale);
	ClassDB::bind_method(D_METHOD("get_sdf_scale"), &Viewport::get_sdf_scale);

	ClassDB::bind_method(D_METHOD("set_mesh_lod_threshold", "pixels"), &Viewport::set_mesh_lod_threshold);
	ClassDB::bind_method(D_METHOD("get_mesh_lod_threshold"), &Viewport::get_mesh_lod_threshold);

	ClassDB::bind_method(D_METHOD("_process_picking"), &Viewport::_process_picking);

#ifndef _3D_DISABLED
	ClassDB::bind_method(D_METHOD("set_world_3d", "world_3d"), &Viewport::set_world_3d);
	ClassDB::bind_method(D_METHOD("get_world_3d"), &Viewport::get_world_3d);
	ClassDB::bind_method(D_METHOD("find_world_3d"), &Viewport::find_world_3d);

	ClassDB::bind_method(D_METHOD("set_use_own_world_3d", "enable"), &Viewport::set_use_own_world_3d);
	ClassDB::bind_method(D_METHOD("is_using_own_world_3d"), &Viewport::is_using_own_world_3d);

	ClassDB::bind_method(D_METHOD("get_camera_3d"), &Viewport::get_camera_3d);
	ClassDB::bind_method(D_METHOD("set_as_audio_listener_3d", "enable"), &Viewport::set_as_audio_listener_3d);
	ClassDB::bind_method(D_METHOD("is_audio_listener_3d"), &Viewport::is_audio_listener_3d);

	ClassDB::bind_method(D_METHOD("set_disable_3d", "disable"), &Viewport::set_disable_3d);
	ClassDB::bind_method(D_METHOD("is_3d_disabled"), &Viewport::is_3d_disabled);

	ClassDB::bind_method(D_METHOD("set_use_xr", "use"), &Viewport::set_use_xr);
	ClassDB::bind_method(D_METHOD("is_using_xr"), &Viewport::is_using_xr);

	ClassDB::bind_method(D_METHOD("set_scaling_3d_mode", "scaling_3d_mode"), &Viewport::set_scaling_3d_mode);
	ClassDB::bind_method(D_METHOD("get_scaling_3d_mode"), &Viewport::get_scaling_3d_mode);

	ClassDB::bind_method(D_METHOD("set_scaling_3d_scale", "scale"), &Viewport::set_scaling_3d_scale);
	ClassDB::bind_method(D_METHOD("get_scaling_3d_scale"), &Viewport::get_scaling_3d_scale);

	ClassDB::bind_method(D_METHOD("set_fsr_sharpness", "fsr_sharpness"), &Viewport::set_fsr_sharpness);
	ClassDB::bind_method(D_METHOD("get_fsr_sharpness"), &Viewport::get_fsr_sharpness);

	ClassDB::bind_method(D_METHOD("set_fsr_mipmap_bias", "fsr_mipmap_bias"), &Viewport::set_fsr_mipmap_bias);
	ClassDB::bind_method(D_METHOD("get_fsr_mipmap_bias"), &Viewport::get_fsr_mipmap_bias);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_3d"), "set_disable_3d", "is_3d_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_xr"), "set_use_xr", "is_using_xr");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_listener_enable_3d"), "set_as_audio_listener_3d", "is_audio_listener_3d");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "own_world_3d"), "set_use_own_world_3d", "is_using_own_world_3d");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "world_3d", PROPERTY_HINT_RESOURCE_TYPE, "World3D"), "set_world_3d", "get_world_3d");
#endif // _3D_DISABLED
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "world_2d", PROPERTY_HINT_RESOURCE_TYPE, "World2D", PROPERTY_USAGE_NONE), "set_world_2d", "get_world_2d");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transparent_bg"), "set_transparent_background", "has_transparent_background");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "handle_input_locally"), "set_handle_input_locally", "is_handling_input_locally");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "snap_2d_transforms_to_pixel"), "set_snap_2d_transforms_to_pixel", "is_snap_2d_transforms_to_pixel_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "snap_2d_vertices_to_pixel"), "set_snap_2d_vertices_to_pixel", "is_snap_2d_vertices_to_pixel_enabled");
	ADD_GROUP("Rendering", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "msaa", PROPERTY_HINT_ENUM, String::utf8("Disabled (Fastest),2× (Average),4× (Slow),8× (Slowest)")), "set_msaa", "get_msaa");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "screen_space_aa", PROPERTY_HINT_ENUM, "Disabled (Fastest),FXAA (Fast)"), "set_screen_space_aa", "get_screen_space_aa");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_debanding"), "set_use_debanding", "is_using_debanding");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_occlusion_culling"), "set_use_occlusion_culling", "is_using_occlusion_culling");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mesh_lod_threshold", PROPERTY_HINT_RANGE, "0,1024,0.1"), "set_mesh_lod_threshold", "get_mesh_lod_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_draw", PROPERTY_HINT_ENUM, "Disabled,Unshaded,Overdraw,Wireframe"), "set_debug_draw", "get_debug_draw");
#ifndef _3D_DISABLED
	ADD_GROUP("Scaling 3D", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scaling_3d_mode", PROPERTY_HINT_ENUM, "Disabled (Slowest),Bilinear (Fastest),FSR (Fast)"), "set_scaling_3d_mode", "get_scaling_3d_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scaling_3d_scale", PROPERTY_HINT_RANGE, "0.25,2.0,0.01"), "set_scaling_3d_scale", "get_scaling_3d_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fsr_mipmap_bias", PROPERTY_HINT_RANGE, "-2,2,0.1"), "set_fsr_mipmap_bias", "get_fsr_mipmap_bias");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fsr_sharpness", PROPERTY_HINT_RANGE, "0,2,0.1"), "set_fsr_sharpness", "get_fsr_sharpness");
#endif
	ADD_GROUP("Canvas Items", "canvas_item_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "canvas_item_default_texture_filter", PROPERTY_HINT_ENUM, "Nearest,Linear,Linear Mipmap,Nearest Mipmap"), "set_default_canvas_item_texture_filter", "get_default_canvas_item_texture_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "canvas_item_default_texture_repeat", PROPERTY_HINT_ENUM, "Disabled,Enabled,Mirror"), "set_default_canvas_item_texture_repeat", "get_default_canvas_item_texture_repeat");
	ADD_GROUP("Audio Listener", "audio_listener_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_listener_enable_2d"), "set_as_audio_listener_2d", "is_audio_listener_2d");
	ADD_GROUP("Physics", "physics_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "physics_object_picking"), "set_physics_object_picking", "get_physics_object_picking");
	ADD_GROUP("GUI", "gui_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_disable_input"), "set_disable_input", "is_input_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_snap_controls_to_pixels"), "set_snap_controls_to_pixels", "is_snap_controls_to_pixels_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gui_embed_subwindows"), "set_embed_subwindows_hint", "get_embed_subwindows_hint");
	ADD_GROUP("SDF", "sdf_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sdf_oversize", PROPERTY_HINT_ENUM, "100%,120%,150%,200%"), "set_sdf_oversize", "get_sdf_oversize");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sdf_scale", PROPERTY_HINT_ENUM, "100%,50%,25%"), "set_sdf_scale", "get_sdf_scale");
	ADD_GROUP("Shadow Atlas", "shadow_atlas_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_atlas_size"), "set_shadow_atlas_size", "get_shadow_atlas_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow_atlas_16_bits"), "set_shadow_atlas_16_bits", "get_shadow_atlas_16_bits");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_0", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 0);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_1", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 1);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_2", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 2);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "shadow_atlas_quad_3", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"), "set_shadow_atlas_quadrant_subdiv", "get_shadow_atlas_quadrant_subdiv", 3);
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "canvas_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_canvas_transform", "get_canvas_transform");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "global_canvas_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_global_canvas_transform", "get_global_canvas_transform");

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

	BIND_ENUM_CONSTANT(SCALING_3D_MODE_BILINEAR);
	BIND_ENUM_CONSTANT(SCALING_3D_MODE_FSR);
	BIND_ENUM_CONSTANT(SCALING_3D_MODE_MAX);

	BIND_ENUM_CONSTANT(MSAA_DISABLED);
	BIND_ENUM_CONSTANT(MSAA_2X);
	BIND_ENUM_CONSTANT(MSAA_4X);
	BIND_ENUM_CONSTANT(MSAA_8X);
	BIND_ENUM_CONSTANT(MSAA_MAX);

	BIND_ENUM_CONSTANT(SCREEN_SPACE_AA_DISABLED);
	BIND_ENUM_CONSTANT(SCREEN_SPACE_AA_FXAA);
	BIND_ENUM_CONSTANT(SCREEN_SPACE_AA_MAX);

	BIND_ENUM_CONSTANT(RENDER_INFO_OBJECTS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_PRIMITIVES_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_DRAW_CALLS_IN_FRAME);
	BIND_ENUM_CONSTANT(RENDER_INFO_MAX);

	BIND_ENUM_CONSTANT(RENDER_INFO_TYPE_VISIBLE);
	BIND_ENUM_CONSTANT(RENDER_INFO_TYPE_SHADOW);
	BIND_ENUM_CONSTANT(RENDER_INFO_TYPE_MAX);

	BIND_ENUM_CONSTANT(DEBUG_DRAW_DISABLED);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_UNSHADED);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_LIGHTING);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_OVERDRAW);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_WIREFRAME);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_NORMAL_BUFFER);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_VOXEL_GI_ALBEDO);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_VOXEL_GI_LIGHTING);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_VOXEL_GI_EMISSION);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SHADOW_ATLAS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SCENE_LUMINANCE);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SSAO);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SSIL);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_PSSM_SPLITS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_DECAL_ATLAS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SDFGI);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_SDFGI_PROBES);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_GI_BUFFER);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_DISABLE_LOD);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_CLUSTER_OMNI_LIGHTS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_CLUSTER_SPOT_LIGHTS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_CLUSTER_DECALS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_CLUSTER_REFLECTION_PROBES);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_OCCLUDERS)

	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_FILTER_MAX);

	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MIRROR);
	BIND_ENUM_CONSTANT(DEFAULT_CANVAS_ITEM_TEXTURE_REPEAT_MAX);

	BIND_ENUM_CONSTANT(SDF_OVERSIZE_100_PERCENT);
	BIND_ENUM_CONSTANT(SDF_OVERSIZE_120_PERCENT);
	BIND_ENUM_CONSTANT(SDF_OVERSIZE_150_PERCENT);
	BIND_ENUM_CONSTANT(SDF_OVERSIZE_200_PERCENT);
	BIND_ENUM_CONSTANT(SDF_OVERSIZE_MAX);

	BIND_ENUM_CONSTANT(SDF_SCALE_100_PERCENT);
	BIND_ENUM_CONSTANT(SDF_SCALE_50_PERCENT);
	BIND_ENUM_CONSTANT(SDF_SCALE_25_PERCENT);
	BIND_ENUM_CONSTANT(SDF_SCALE_MAX);
}

Viewport::Viewport() {
	world_2d = Ref<World2D>(memnew(World2D));

	viewport = RenderingServer::get_singleton()->viewport_create();
	texture_rid = RenderingServer::get_singleton()->viewport_get_texture(viewport);

	default_texture.instantiate();
	default_texture->vp = const_cast<Viewport *>(this);
	viewport_textures.insert(default_texture.ptr());
	default_texture->proxy = RS::get_singleton()->texture_proxy_create(texture_rid);

	canvas_layers.insert(nullptr); // This eases picking code (interpreted as the canvas of the Viewport).

	set_shadow_atlas_size(shadow_atlas_size);

	for (int i = 0; i < 4; i++) {
		shadow_atlas_quadrant_subdiv[i] = SHADOW_ATLAS_QUADRANT_SUBDIV_MAX;
	}
	set_shadow_atlas_quadrant_subdiv(0, SHADOW_ATLAS_QUADRANT_SUBDIV_4);
	set_shadow_atlas_quadrant_subdiv(1, SHADOW_ATLAS_QUADRANT_SUBDIV_4);
	set_shadow_atlas_quadrant_subdiv(2, SHADOW_ATLAS_QUADRANT_SUBDIV_16);
	set_shadow_atlas_quadrant_subdiv(3, SHADOW_ATLAS_QUADRANT_SUBDIV_64);

	set_mesh_lod_threshold(mesh_lod_threshold);

	String id = itos(get_instance_id());
	input_group = "_vp_input" + id;
	gui_input_group = "_vp_gui_input" + id;
	unhandled_input_group = "_vp_unhandled_input" + id;
	unhandled_key_input_group = "_vp_unhandled_key_input" + id;

	// Window tooltip.
	gui.tooltip_delay = GLOBAL_DEF("gui/timers/tooltip_delay_sec", 0.5);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/timers/tooltip_delay_sec", PropertyInfo(Variant::FLOAT, "gui/timers/tooltip_delay_sec", PROPERTY_HINT_RANGE, "0,5,0.01,or_greater")); // No negative numbers

#ifndef _3D_DISABLED
	Viewport::Scaling3DMode scaling_3d_mode = (Viewport::Scaling3DMode)(int)GLOBAL_GET("rendering/scaling_3d/mode");
	set_scaling_3d_mode(scaling_3d_mode);

	set_scaling_3d_scale(GLOBAL_GET("rendering/scaling_3d/scale"));

	float fsr_sharpness = GLOBAL_GET("rendering/scaling_3d/fsr_sharpness");
	set_fsr_sharpness(fsr_sharpness);

	float fsr_mipmap_bias = GLOBAL_GET("rendering/scaling_3d/fsr_mipmap_bias");
	set_fsr_mipmap_bias(fsr_mipmap_bias);
#endif // _3D_DISABLED

	set_sdf_oversize(sdf_oversize); // Set to server.
}

Viewport::~Viewport() {
	// Erase itself from viewport textures.
	for (Set<ViewportTexture *>::Element *E = viewport_textures.front(); E; E = E->next()) {
		E->get()->vp = nullptr;
	}
	RenderingServer::get_singleton()->free(viewport);
}

/////////////////////////////////

void SubViewport::set_size(const Size2i &p_size) {
	_set_size(p_size, _get_size_2d_override(), Rect2i(), _stretch_transform(), true);

	SubViewportContainer *c = Object::cast_to<SubViewportContainer>(get_parent());
	if (c) {
		c->update_minimum_size();
	}
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

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "size_2d_override"), "set_size_2d_override", "get_size_2d_override");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "size_2d_override_stretch"), "set_size_2d_override_stretch", "is_size_2d_override_stretch_enabled");
	ADD_GROUP("Render Target", "render_target_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_target_clear_mode", PROPERTY_HINT_ENUM, "Always,Never,Next Frame"), "set_clear_mode", "get_clear_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_target_update_mode", PROPERTY_HINT_ENUM, "Disabled,Once,When Visible,When Parent Visible,Always"), "set_update_mode", "get_update_mode");

	BIND_ENUM_CONSTANT(CLEAR_MODE_ALWAYS);
	BIND_ENUM_CONSTANT(CLEAR_MODE_NEVER);
	BIND_ENUM_CONSTANT(CLEAR_MODE_ONCE);

	BIND_ENUM_CONSTANT(UPDATE_DISABLED);
	BIND_ENUM_CONSTANT(UPDATE_ONCE);
	BIND_ENUM_CONSTANT(UPDATE_WHEN_VISIBLE);
	BIND_ENUM_CONSTANT(UPDATE_WHEN_PARENT_VISIBLE);
	BIND_ENUM_CONSTANT(UPDATE_ALWAYS);
}

SubViewport::SubViewport() {}

SubViewport::~SubViewport() {}
