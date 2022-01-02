/*************************************************************************/
/*  graph_edit.cpp                                                       */
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

#include "graph_edit.h"

#include "core/input/input.h"
#include "core/math/math_funcs.h"
#include "core/os/keyboard.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"

constexpr int MINIMAP_OFFSET = 12;
constexpr int MINIMAP_PADDING = 5;

bool GraphEditFilter::has_point(const Point2 &p_point) const {
	return ge->_filter_input(p_point);
}

GraphEditFilter::GraphEditFilter(GraphEdit *p_edit) {
	ge = p_edit;
}

GraphEditMinimap::GraphEditMinimap(GraphEdit *p_edit) {
	ge = p_edit;

	graph_proportions = Vector2(1, 1);
	graph_padding = Vector2(0, 0);
	camera_position = Vector2(100, 50);
	camera_size = Vector2(200, 200);
	minimap_padding = Vector2(MINIMAP_PADDING, MINIMAP_PADDING);
	minimap_offset = minimap_padding + _convert_from_graph_position(graph_padding);

	is_pressing = false;
	is_resizing = false;
}

void GraphEditMinimap::update_minimap() {
	Vector2 graph_offset = _get_graph_offset();
	Vector2 graph_size = _get_graph_size();

	camera_position = ge->get_scroll_ofs() - graph_offset;
	camera_size = ge->get_size();

	Vector2 render_size = _get_render_size();
	float target_ratio = render_size.x / render_size.y;
	float graph_ratio = graph_size.x / graph_size.y;

	graph_proportions = graph_size;
	graph_padding = Vector2(0, 0);
	if (graph_ratio > target_ratio) {
		graph_proportions.x = graph_size.x;
		graph_proportions.y = graph_size.x / target_ratio;
		graph_padding.y = Math::abs(graph_size.y - graph_proportions.y) / 2;
	} else {
		graph_proportions.x = graph_size.y * target_ratio;
		graph_proportions.y = graph_size.y;
		graph_padding.x = Math::abs(graph_size.x - graph_proportions.x) / 2;
	}

	// This centers minimap inside the minimap rectangle.
	minimap_offset = minimap_padding + _convert_from_graph_position(graph_padding);
}

Rect2 GraphEditMinimap::get_camera_rect() {
	Vector2 camera_center = _convert_from_graph_position(camera_position + camera_size / 2) + minimap_offset;
	Vector2 camera_viewport = _convert_from_graph_position(camera_size);
	Vector2 camera_position = (camera_center - camera_viewport / 2);
	return Rect2(camera_position, camera_viewport);
}

Vector2 GraphEditMinimap::_get_render_size() {
	if (!is_inside_tree()) {
		return Vector2(0, 0);
	}

	return get_size() - 2 * minimap_padding;
}

Vector2 GraphEditMinimap::_get_graph_offset() {
	return Vector2(ge->h_scroll->get_min(), ge->v_scroll->get_min());
}

Vector2 GraphEditMinimap::_get_graph_size() {
	Vector2 graph_size = Vector2(ge->h_scroll->get_max(), ge->v_scroll->get_max()) - Vector2(ge->h_scroll->get_min(), ge->v_scroll->get_min());

	if (graph_size.x == 0) {
		graph_size.x = 1;
	}
	if (graph_size.y == 0) {
		graph_size.y = 1;
	}

	return graph_size;
}

Vector2 GraphEditMinimap::_convert_from_graph_position(const Vector2 &p_position) {
	Vector2 map_position = Vector2(0, 0);
	Vector2 render_size = _get_render_size();

	map_position.x = p_position.x * render_size.x / graph_proportions.x;
	map_position.y = p_position.y * render_size.y / graph_proportions.y;

	return map_position;
}

Vector2 GraphEditMinimap::_convert_to_graph_position(const Vector2 &p_position) {
	Vector2 graph_position = Vector2(0, 0);
	Vector2 render_size = _get_render_size();

	graph_position.x = p_position.x * graph_proportions.x / render_size.x;
	graph_position.y = p_position.y * graph_proportions.y / render_size.y;

	return graph_position;
}

void GraphEditMinimap::gui_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	if (!ge->is_minimap_enabled()) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_ev;
	Ref<InputEventMouseMotion> mm = p_ev;

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			is_pressing = true;

			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));
			Rect2 resizer_hitbox = Rect2(Point2(), resizer->get_size());
			if (resizer_hitbox.has_point(mb->get_position())) {
				is_resizing = true;
			} else {
				Vector2 click_position = _convert_to_graph_position(mb->get_position() - minimap_padding) - graph_padding;
				_adjust_graph_scroll(click_position);
			}
		} else {
			is_pressing = false;
			is_resizing = false;
		}
		accept_event();
	} else if (mm.is_valid() && is_pressing) {
		if (is_resizing) {
			// Prevent setting minimap wider than GraphEdit
			Vector2 new_minimap_size;
			new_minimap_size.x = MIN(get_size().x - mm->get_relative().x, ge->get_size().x - 2.0 * minimap_padding.x);
			new_minimap_size.y = MIN(get_size().y - mm->get_relative().y, ge->get_size().y - 2.0 * minimap_padding.y);
			ge->set_minimap_size(new_minimap_size);

			update();
		} else {
			Vector2 click_position = _convert_to_graph_position(mm->get_position() - minimap_padding) - graph_padding;
			_adjust_graph_scroll(click_position);
		}
		accept_event();
	}
}

void GraphEditMinimap::_adjust_graph_scroll(const Vector2 &p_offset) {
	Vector2 graph_offset = _get_graph_offset();
	ge->set_scroll_ofs(p_offset + graph_offset - camera_size / 2);
}

Error GraphEdit::connect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	if (is_node_connected(p_from, p_from_port, p_to, p_to_port)) {
		return OK;
	}
	Connection c;
	c.from = p_from;
	c.from_port = p_from_port;
	c.to = p_to;
	c.to_port = p_to_port;
	c.activity = 0;
	connections.push_back(c);
	top_layer->update();
	minimap->update();
	update();
	connections_layer->update();

	return OK;
}

bool GraphEdit::is_node_connected(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	for (const Connection &E : connections) {
		if (E.from == p_from && E.from_port == p_from_port && E.to == p_to && E.to_port == p_to_port) {
			return true;
		}
	}

	return false;
}

void GraphEdit::disconnect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	for (const List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		if (E->get().from == p_from && E->get().from_port == p_from_port && E->get().to == p_to && E->get().to_port == p_to_port) {
			connections.erase(E);
			top_layer->update();
			minimap->update();
			update();
			connections_layer->update();
			return;
		}
	}
}

void GraphEdit::get_connection_list(List<Connection> *r_connections) const {
	*r_connections = connections;
}

void GraphEdit::set_scroll_ofs(const Vector2 &p_ofs) {
	setting_scroll_ofs = true;
	h_scroll->set_value(p_ofs.x);
	v_scroll->set_value(p_ofs.y);
	_update_scroll();
	setting_scroll_ofs = false;
}

Vector2 GraphEdit::get_scroll_ofs() const {
	return Vector2(h_scroll->get_value(), v_scroll->get_value());
}

void GraphEdit::_scroll_moved(double) {
	if (!awaiting_scroll_offset_update) {
		call_deferred(SNAME("_update_scroll_offset"));
		awaiting_scroll_offset_update = true;
	}
	top_layer->update();
	minimap->update();
	update();

	if (!setting_scroll_ofs) { //in godot, signals on change value are avoided as a convention
		emit_signal(SNAME("scroll_offset_changed"), get_scroll_ofs());
	}
}

void GraphEdit::_update_scroll_offset() {
	set_block_minimum_size_adjust(true);

	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		Point2 pos = gn->get_position_offset() * zoom;
		pos -= Point2(h_scroll->get_value(), v_scroll->get_value());
		gn->set_position(pos);
		if (gn->get_scale() != Vector2(zoom, zoom)) {
			gn->set_scale(Vector2(zoom, zoom));
		}
	}

	connections_layer->set_position(-Point2(h_scroll->get_value(), v_scroll->get_value()));
	set_block_minimum_size_adjust(false);
	awaiting_scroll_offset_update = false;
}

void GraphEdit::_update_scroll() {
	if (updating) {
		return;
	}

	updating = true;

	set_block_minimum_size_adjust(true);

	Rect2 screen;
	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		Rect2 r;
		r.position = gn->get_position_offset() * zoom;
		r.size = gn->get_size() * zoom;
		screen = screen.merge(r);
	}

	screen.position -= get_size();
	screen.size += get_size() * 2.0;

	h_scroll->set_min(screen.position.x);
	h_scroll->set_max(screen.position.x + screen.size.x);
	h_scroll->set_page(get_size().x);
	if (h_scroll->get_max() - h_scroll->get_min() <= h_scroll->get_page()) {
		h_scroll->hide();
	} else {
		h_scroll->show();
	}

	v_scroll->set_min(screen.position.y);
	v_scroll->set_max(screen.position.y + screen.size.y);
	v_scroll->set_page(get_size().y);

	if (v_scroll->get_max() - v_scroll->get_min() <= v_scroll->get_page()) {
		v_scroll->hide();
	} else {
		v_scroll->show();
	}

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	// Avoid scrollbar overlapping.
	h_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, v_scroll->is_visible() ? -vmin.width : 0);
	v_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, h_scroll->is_visible() ? -hmin.height : 0);

	set_block_minimum_size_adjust(false);

	if (!awaiting_scroll_offset_update) {
		call_deferred(SNAME("_update_scroll_offset"));
		awaiting_scroll_offset_update = true;
	}

	updating = false;
}

void GraphEdit::_graph_node_raised(Node *p_gn) {
	GraphNode *gn = Object::cast_to<GraphNode>(p_gn);
	ERR_FAIL_COND(!gn);
	if (gn->is_comment()) {
		move_child(gn, 0);
	} else {
		gn->raise();
	}
	emit_signal(SNAME("node_selected"), p_gn);
}

void GraphEdit::_graph_node_moved(Node *p_gn) {
	GraphNode *gn = Object::cast_to<GraphNode>(p_gn);
	ERR_FAIL_COND(!gn);
	top_layer->update();
	minimap->update();
	update();
	connections_layer->update();
}

void GraphEdit::_graph_node_slot_updated(int p_index, Node *p_gn) {
	GraphNode *gn = Object::cast_to<GraphNode>(p_gn);
	ERR_FAIL_COND(!gn);
	top_layer->update();
	minimap->update();
	update();
	connections_layer->update();
}

void GraphEdit::add_child_notify(Node *p_child) {
	Control::add_child_notify(p_child);

	top_layer->call_deferred(SNAME("raise")); // Top layer always on top!

	GraphNode *gn = Object::cast_to<GraphNode>(p_child);
	if (gn) {
		gn->set_scale(Vector2(zoom, zoom));
		gn->connect("position_offset_changed", callable_mp(this, &GraphEdit::_graph_node_moved), varray(gn));
		gn->connect("slot_updated", callable_mp(this, &GraphEdit::_graph_node_slot_updated), varray(gn));
		gn->connect("raise_request", callable_mp(this, &GraphEdit::_graph_node_raised), varray(gn));
		gn->connect("item_rect_changed", callable_mp((CanvasItem *)connections_layer, &CanvasItem::update));
		gn->connect("item_rect_changed", callable_mp((CanvasItem *)minimap, &GraphEditMinimap::update));
		_graph_node_moved(gn);
		gn->set_mouse_filter(MOUSE_FILTER_PASS);
	}
}

void GraphEdit::remove_child_notify(Node *p_child) {
	Control::remove_child_notify(p_child);

	if (p_child == top_layer) {
		top_layer = nullptr;
		minimap = nullptr;
	} else if (p_child == connections_layer) {
		connections_layer = nullptr;
	}

	if (top_layer != nullptr && is_inside_tree()) {
		top_layer->call_deferred(SNAME("raise")); // Top layer always on top!
	}

	GraphNode *gn = Object::cast_to<GraphNode>(p_child);
	if (gn) {
		gn->disconnect("position_offset_changed", callable_mp(this, &GraphEdit::_graph_node_moved));
		gn->disconnect("slot_updated", callable_mp(this, &GraphEdit::_graph_node_slot_updated));
		gn->disconnect("raise_request", callable_mp(this, &GraphEdit::_graph_node_raised));

		// In case of the whole GraphEdit being destroyed these references can already be freed.
		if (connections_layer != nullptr && connections_layer->is_inside_tree()) {
			gn->disconnect("item_rect_changed", callable_mp((CanvasItem *)connections_layer, &CanvasItem::update));
		}
		if (minimap != nullptr && minimap->is_inside_tree()) {
			gn->disconnect("item_rect_changed", callable_mp((CanvasItem *)minimap, &GraphEditMinimap::update));
		}
	}
}

void GraphEdit::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		port_grab_distance_horizontal = get_theme_constant(SNAME("port_grab_distance_horizontal"));
		port_grab_distance_vertical = get_theme_constant(SNAME("port_grab_distance_vertical"));

		zoom_minus->set_icon(get_theme_icon(SNAME("minus")));
		zoom_reset->set_icon(get_theme_icon(SNAME("reset")));
		zoom_plus->set_icon(get_theme_icon(SNAME("more")));
		snap_button->set_icon(get_theme_icon(SNAME("snap")));
		minimap_button->set_icon(get_theme_icon(SNAME("minimap")));
		layout_button->set_icon(get_theme_icon(SNAME("layout")));

		zoom_label->set_custom_minimum_size(Size2(48, 0) * get_theme_default_base_scale());
	}
	if (p_what == NOTIFICATION_READY) {
		Size2 hmin = h_scroll->get_combined_minimum_size();
		Size2 vmin = v_scroll->get_combined_minimum_size();

		h_scroll->set_anchor_and_offset(SIDE_LEFT, ANCHOR_BEGIN, 0);
		h_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
		h_scroll->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -hmin.height);
		h_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);

		v_scroll->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -vmin.width);
		v_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
		v_scroll->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, 0);
		v_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);
	}
	if (p_what == NOTIFICATION_DRAW) {
		draw_style_box(get_theme_stylebox(SNAME("bg")), Rect2(Point2(), get_size()));

		if (is_using_snap()) {
			//draw grid

			int snap = get_snap();

			Vector2 offset = get_scroll_ofs() / zoom;
			Size2 size = get_size() / zoom;

			Point2i from = (offset / float(snap)).floor();
			Point2i len = (size / float(snap)).floor() + Vector2(1, 1);

			Color grid_minor = get_theme_color(SNAME("grid_minor"));
			Color grid_major = get_theme_color(SNAME("grid_major"));

			for (int i = from.x; i < from.x + len.x; i++) {
				Color color;

				if (ABS(i) % 10 == 0) {
					color = grid_major;
				} else {
					color = grid_minor;
				}

				float base_ofs = i * snap * zoom - offset.x * zoom;
				draw_line(Vector2(base_ofs, 0), Vector2(base_ofs, get_size().height), color);
			}

			for (int i = from.y; i < from.y + len.y; i++) {
				Color color;

				if (ABS(i) % 10 == 0) {
					color = grid_major;
				} else {
					color = grid_minor;
				}

				float base_ofs = i * snap * zoom - offset.y * zoom;
				draw_line(Vector2(0, base_ofs), Vector2(get_size().width, base_ofs), color);
			}
		}
	}

	if (p_what == NOTIFICATION_RESIZED) {
		_update_scroll();
		top_layer->update();
		minimap->update();
	}
}

void GraphEdit::_update_comment_enclosed_nodes_list(GraphNode *p_node, HashMap<StringName, Vector<GraphNode *>> &p_comment_enclosed_nodes) {
	Rect2 comment_node_rect = p_node->get_rect();
	Vector<GraphNode *> enclosed_nodes;

	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn || gn->is_selected()) {
			continue;
		}

		Rect2 node_rect = gn->get_rect();
		bool included = comment_node_rect.encloses(node_rect);
		if (included) {
			enclosed_nodes.push_back(gn);
		}
	}

	p_comment_enclosed_nodes.set(p_node->get_name(), enclosed_nodes);
}

void GraphEdit::_set_drag_comment_enclosed_nodes(GraphNode *p_node, HashMap<StringName, Vector<GraphNode *>> &p_comment_enclosed_nodes, bool p_drag) {
	for (int i = 0; i < p_comment_enclosed_nodes[p_node->get_name()].size(); i++) {
		p_comment_enclosed_nodes[p_node->get_name()][i]->set_drag(p_drag);
	}
}

void GraphEdit::_set_position_of_comment_enclosed_nodes(GraphNode *p_node, HashMap<StringName, Vector<GraphNode *>> &p_comment_enclosed_nodes, Vector2 p_drag_accum) {
	for (int i = 0; i < p_comment_enclosed_nodes[p_node->get_name()].size(); i++) {
		Vector2 pos = (p_comment_enclosed_nodes[p_node->get_name()][i]->get_drag_from() * zoom + drag_accum) / zoom;
		if (is_using_snap() ^ Input::get_singleton()->is_key_pressed(Key::CTRL)) {
			const int snap = get_snap();
			pos = pos.snapped(Vector2(snap, snap));
		}
		p_comment_enclosed_nodes[p_node->get_name()][i]->set_position_offset(pos);
	}
}

bool GraphEdit::_filter_input(const Point2 &p_point) {
	Ref<Texture2D> port = get_theme_icon(SNAME("port"), SNAME("GraphNode"));
	Vector2i port_size = Vector2i(port->get_width(), port->get_height());

	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		for (int j = 0; j < gn->get_connection_output_count(); j++) {
			if (is_in_output_hotzone(gn, j, p_point / zoom, port_size)) {
				return true;
			}
		}

		for (int j = 0; j < gn->get_connection_input_count(); j++) {
			if (is_in_input_hotzone(gn, j, p_point / zoom, port_size)) {
				return true;
			}
		}
	}

	return false;
}

void GraphEdit::_top_layer_input(const Ref<InputEvent> &p_ev) {
	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
		Ref<Texture2D> port = get_theme_icon(SNAME("port"), SNAME("GraphNode"));
		Vector2i port_size = Vector2i(port->get_width(), port->get_height());

		connecting_valid = false;
		click_pos = mb->get_position() / zoom;
		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
			if (!gn) {
				continue;
			}

			for (int j = 0; j < gn->get_connection_output_count(); j++) {
				Vector2 pos = gn->get_connection_output_position(j) + gn->get_position();
				if (is_in_output_hotzone(gn, j, click_pos, port_size)) {
					if (valid_left_disconnect_types.has(gn->get_connection_output_type(j))) {
						//check disconnect
						for (const Connection &E : connections) {
							if (E.from == gn->get_name() && E.from_port == j) {
								Node *to = get_node(String(E.to));
								if (Object::cast_to<GraphNode>(to)) {
									connecting_from = E.to;
									connecting_index = E.to_port;
									connecting_out = false;
									connecting_type = Object::cast_to<GraphNode>(to)->get_connection_input_type(E.to_port);
									connecting_color = Object::cast_to<GraphNode>(to)->get_connection_input_color(E.to_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconnected = true;

									emit_signal(SNAME("disconnection_request"), E.from, E.from_port, E.to, E.to_port);
									to = get_node(String(connecting_from)); //maybe it was erased
									if (Object::cast_to<GraphNode>(to)) {
										connecting = true;
									}
									return;
								}
							}
						}
					}

					connecting = true;
					connecting_from = gn->get_name();
					connecting_index = j;
					connecting_out = true;
					connecting_type = gn->get_connection_output_type(j);
					connecting_color = gn->get_connection_output_color(j);
					connecting_target = false;
					connecting_to = pos;
					just_disconnected = false;
					return;
				}
			}

			for (int j = 0; j < gn->get_connection_input_count(); j++) {
				Vector2 pos = gn->get_connection_input_position(j) + gn->get_position();
				if (is_in_input_hotzone(gn, j, click_pos, port_size)) {
					if (right_disconnects || valid_right_disconnect_types.has(gn->get_connection_input_type(j))) {
						//check disconnect
						for (const Connection &E : connections) {
							if (E.to == gn->get_name() && E.to_port == j) {
								Node *fr = get_node(String(E.from));
								if (Object::cast_to<GraphNode>(fr)) {
									connecting_from = E.from;
									connecting_index = E.from_port;
									connecting_out = true;
									connecting_type = Object::cast_to<GraphNode>(fr)->get_connection_output_type(E.from_port);
									connecting_color = Object::cast_to<GraphNode>(fr)->get_connection_output_color(E.from_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconnected = true;

									emit_signal(SNAME("disconnection_request"), E.from, E.from_port, E.to, E.to_port);
									fr = get_node(String(connecting_from)); //maybe it was erased
									if (Object::cast_to<GraphNode>(fr)) {
										connecting = true;
									}
									return;
								}
							}
						}
					}

					connecting = true;
					connecting_from = gn->get_name();
					connecting_index = j;
					connecting_out = false;
					connecting_type = gn->get_connection_input_type(j);
					connecting_color = gn->get_connection_input_color(j);
					connecting_target = false;
					connecting_to = pos;
					just_disconnected = false;

					return;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_ev;
	if (mm.is_valid() && connecting) {
		connecting_to = mm->get_position();
		connecting_target = false;
		top_layer->update();
		minimap->update();
		connecting_valid = just_disconnected || click_pos.distance_to(connecting_to / zoom) > 20.0;

		if (connecting_valid) {
			Ref<Texture2D> port = get_theme_icon(SNAME("port"), SNAME("GraphNode"));
			Vector2i port_size = Vector2i(port->get_width(), port->get_height());

			Vector2 mpos = mm->get_position() / zoom;
			for (int i = get_child_count() - 1; i >= 0; i--) {
				GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
				if (!gn) {
					continue;
				}

				if (!connecting_out) {
					for (int j = 0; j < gn->get_connection_output_count(); j++) {
						Vector2 pos = gn->get_connection_output_position(j) + gn->get_position();
						int type = gn->get_connection_output_type(j);
						if ((type == connecting_type || valid_connection_types.has(ConnType(type, connecting_type))) && is_in_output_hotzone(gn, j, mpos, port_size)) {
							connecting_target = true;
							connecting_to = pos;
							connecting_target_to = gn->get_name();
							connecting_target_index = j;
							return;
						}
					}
				} else {
					for (int j = 0; j < gn->get_connection_input_count(); j++) {
						Vector2 pos = gn->get_connection_input_position(j) + gn->get_position();
						int type = gn->get_connection_input_type(j);
						if ((type == connecting_type || valid_connection_types.has(ConnType(type, connecting_type))) && is_in_input_hotzone(gn, j, mpos, port_size)) {
							connecting_target = true;
							connecting_to = pos;
							connecting_target_to = gn->get_name();
							connecting_target_index = j;
							return;
						}
					}
				}
			}
		}
	}

	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed()) {
		if (connecting_valid) {
			if (connecting && connecting_target) {
				String from = connecting_from;
				int from_slot = connecting_index;
				String to = connecting_target_to;
				int to_slot = connecting_target_index;

				if (!connecting_out) {
					SWAP(from, to);
					SWAP(from_slot, to_slot);
				}
				emit_signal(SNAME("connection_request"), from, from_slot, to, to_slot);

			} else if (!just_disconnected) {
				String from = connecting_from;
				int from_slot = connecting_index;
				Vector2 ofs = mb->get_position();

				if (!connecting_out) {
					emit_signal(SNAME("connection_from_empty"), from, from_slot, ofs);
				} else {
					emit_signal(SNAME("connection_to_empty"), from, from_slot, ofs);
				}
			}
		}

		connecting = false;
		top_layer->update();
		minimap->update();
		update();
		connections_layer->update();
	}
}

bool GraphEdit::_check_clickable_control(Control *p_control, const Vector2 &pos) {
	if (p_control->is_set_as_top_level() || !p_control->is_visible()) {
		return false;
	}

	if (!p_control->has_point(pos) || p_control->get_mouse_filter() == MOUSE_FILTER_IGNORE) {
		//test children
		for (int i = 0; i < p_control->get_child_count(); i++) {
			Control *subchild = Object::cast_to<Control>(p_control->get_child(i));
			if (!subchild) {
				continue;
			}
			if (_check_clickable_control(subchild, pos - subchild->get_position())) {
				return true;
			}
		}

		return false;
	} else {
		return true;
	}
}

bool GraphEdit::is_in_input_hotzone(GraphNode *p_graph_node, int p_slot_index, const Vector2 &p_mouse_pos, const Vector2i &p_port_size) {
	if (get_script_instance() && get_script_instance()->has_method("_is_in_input_hotzone")) {
		return get_script_instance()->call("_is_in_input_hotzone", p_graph_node, p_slot_index, p_mouse_pos);
	} else {
		Vector2 pos = p_graph_node->get_connection_input_position(p_slot_index) + p_graph_node->get_position();
		return is_in_port_hotzone(pos / zoom, p_mouse_pos, p_port_size, true);
	}
}

bool GraphEdit::is_in_output_hotzone(GraphNode *p_graph_node, int p_slot_index, const Vector2 &p_mouse_pos, const Vector2i &p_port_size) {
	if (get_script_instance() && get_script_instance()->has_method("_is_in_output_hotzone")) {
		return get_script_instance()->call("_is_in_output_hotzone", p_graph_node, p_slot_index, p_mouse_pos);
	} else {
		Vector2 pos = p_graph_node->get_connection_output_position(p_slot_index) + p_graph_node->get_position();
		return is_in_port_hotzone(pos / zoom, p_mouse_pos, p_port_size, false);
	}
}

bool GraphEdit::is_in_port_hotzone(const Vector2 &pos, const Vector2 &p_mouse_pos, const Vector2i &p_port_size, bool p_left) {
	if (!Rect2(pos.x - port_grab_distance_horizontal, pos.y - port_grab_distance_vertical, port_grab_distance_horizontal * 2, port_grab_distance_vertical * 2).has_point(p_mouse_pos)) {
		return false;
	}

	for (int i = 0; i < get_child_count(); i++) {
		Control *child = Object::cast_to<Control>(get_child(i));
		if (!child) {
			continue;
		}
		Rect2 rect = child->get_rect();

		// To prevent intersections with other nodes.
		rect.position *= zoom;
		rect.size *= zoom;

		if (rect.has_point(p_mouse_pos)) {
			//check sub-controls
			Vector2 subpos = p_mouse_pos - rect.position;

			for (int j = 0; j < child->get_child_count(); j++) {
				Control *subchild = Object::cast_to<Control>(child->get_child(j));
				if (!subchild) {
					continue;
				}

				if (_check_clickable_control(subchild, subpos - subchild->get_position())) {
					return false;
				}
			}
		}
	}

	return true;
}

PackedVector2Array GraphEdit::get_connection_line(const Vector2 &p_from, const Vector2 &p_to) {
	Vector<Vector2> ret;
	if (GDVIRTUAL_CALL(_get_connection_line, p_from, p_to, ret)) {
		return ret;
	}

	Curve2D curve;
	Vector<Color> colors;
	curve.add_point(p_from);
	curve.set_point_out(0, Vector2(60, 0));
	curve.add_point(p_to);
	curve.set_point_in(1, Vector2(-60, 0));
	return curve.tessellate();
}

void GraphEdit::_draw_connection_line(CanvasItem *p_where, const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, const Color &p_to_color, float p_width, float p_zoom) {
	Vector<Vector2> points = get_connection_line(p_from / p_zoom, p_to / p_zoom);
	Vector<Vector2> scaled_points;
	Vector<Color> colors;
	float length = (p_from / p_zoom).distance_to(p_to / p_zoom);
	for (int i = 0; i < points.size(); i++) {
		float d = (p_from / p_zoom).distance_to(points[i]) / length;
		colors.push_back(p_color.lerp(p_to_color, d));
		scaled_points.push_back(points[i] * p_zoom);
	}

	p_where->draw_polyline_colors(scaled_points, colors, Math::floor(p_width * get_theme_default_base_scale()), lines_antialiased);
}

void GraphEdit::_connections_layer_draw() {
	Color activity_color = get_theme_color(SNAME("activity"));
	//draw connections
	List<List<Connection>::Element *> to_erase;
	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		NodePath fromnp(E->get().from);

		Node *from = get_node(fromnp);
		if (!from) {
			to_erase.push_back(E);
			continue;
		}

		GraphNode *gfrom = Object::cast_to<GraphNode>(from);

		if (!gfrom) {
			to_erase.push_back(E);
			continue;
		}

		NodePath tonp(E->get().to);
		Node *to = get_node(tonp);
		if (!to) {
			to_erase.push_back(E);
			continue;
		}

		GraphNode *gto = Object::cast_to<GraphNode>(to);

		if (!gto) {
			to_erase.push_back(E);
			continue;
		}

		Vector2 frompos = gfrom->get_connection_output_position(E->get().from_port) + gfrom->get_position_offset() * zoom;
		Color color = gfrom->get_connection_output_color(E->get().from_port);
		Vector2 topos = gto->get_connection_input_position(E->get().to_port) + gto->get_position_offset() * zoom;
		Color tocolor = gto->get_connection_input_color(E->get().to_port);

		if (E->get().activity > 0) {
			color = color.lerp(activity_color, E->get().activity);
			tocolor = tocolor.lerp(activity_color, E->get().activity);
		}
		_draw_connection_line(connections_layer, frompos, topos, color, tocolor, lines_thickness, zoom);
	}

	while (to_erase.size()) {
		connections.erase(to_erase.front()->get());
		to_erase.pop_front();
	}
}

void GraphEdit::_top_layer_draw() {
	_update_scroll();

	if (connecting) {
		Node *fromn = get_node(connecting_from);
		ERR_FAIL_COND(!fromn);
		GraphNode *from = Object::cast_to<GraphNode>(fromn);
		ERR_FAIL_COND(!from);
		Vector2 pos;
		if (connecting_out) {
			pos = from->get_connection_output_position(connecting_index);
		} else {
			pos = from->get_connection_input_position(connecting_index);
		}
		pos += from->get_position();

		Vector2 topos;
		topos = connecting_to;

		Color col = connecting_color;

		if (connecting_target) {
			col.r += 0.4;
			col.g += 0.4;
			col.b += 0.4;
		}

		if (!connecting_out) {
			SWAP(pos, topos);
		}
		_draw_connection_line(top_layer, pos, topos, col, col, lines_thickness, zoom);
	}

	if (box_selecting) {
		top_layer->draw_rect(box_selecting_rect, get_theme_color(SNAME("selection_fill")));
		top_layer->draw_rect(box_selecting_rect, get_theme_color(SNAME("selection_stroke")), false);
	}
}

void GraphEdit::_minimap_draw() {
	if (!is_minimap_enabled()) {
		return;
	}

	minimap->update_minimap();

	// Draw the minimap background.
	Rect2 minimap_rect = Rect2(Point2(), minimap->get_size());
	minimap->draw_style_box(minimap->get_theme_stylebox(SNAME("bg")), minimap_rect);

	Vector2 graph_offset = minimap->_get_graph_offset();
	Vector2 minimap_offset = minimap->minimap_offset;

	// Draw comment graph nodes.
	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn || !gn->is_comment()) {
			continue;
		}

		Vector2 node_position = minimap->_convert_from_graph_position(gn->get_position_offset() * zoom - graph_offset) + minimap_offset;
		Vector2 node_size = minimap->_convert_from_graph_position(gn->get_size() * zoom);
		Rect2 node_rect = Rect2(node_position, node_size);

		Ref<StyleBoxFlat> sb_minimap = minimap->get_theme_stylebox(SNAME("node"))->duplicate();

		// Override default values with colors provided by the GraphNode's stylebox, if possible.
		Ref<StyleBoxFlat> sbf = gn->get_theme_stylebox(gn->is_selected() ? "commentfocus" : "comment");
		if (sbf.is_valid()) {
			Color node_color = sbf->get_bg_color();
			sb_minimap->set_bg_color(node_color);
		}

		minimap->draw_style_box(sb_minimap, node_rect);
	}

	// Draw regular graph nodes.
	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn || gn->is_comment()) {
			continue;
		}

		Vector2 node_position = minimap->_convert_from_graph_position(gn->get_position_offset() * zoom - graph_offset) + minimap_offset;
		Vector2 node_size = minimap->_convert_from_graph_position(gn->get_size() * zoom);
		Rect2 node_rect = Rect2(node_position, node_size);

		Ref<StyleBoxFlat> sb_minimap = minimap->get_theme_stylebox(SNAME("node"))->duplicate();

		// Override default values with colors provided by the GraphNode's stylebox, if possible.
		Ref<StyleBoxFlat> sbf = gn->get_theme_stylebox(gn->is_selected() ? "selectedframe" : "frame");
		if (sbf.is_valid()) {
			Color node_color = sbf->get_border_color();
			sb_minimap->set_bg_color(node_color);
		}

		minimap->draw_style_box(sb_minimap, node_rect);
	}

	// Draw node connections.
	Color activity_color = get_theme_color(SNAME("activity"));
	for (const Connection &E : connections) {
		NodePath fromnp(E.from);

		Node *from = get_node(fromnp);
		if (!from) {
			continue;
		}
		GraphNode *gfrom = Object::cast_to<GraphNode>(from);
		if (!gfrom) {
			continue;
		}

		NodePath tonp(E.to);
		Node *to = get_node(tonp);
		if (!to) {
			continue;
		}
		GraphNode *gto = Object::cast_to<GraphNode>(to);
		if (!gto) {
			continue;
		}

		Vector2 from_slot_position = gfrom->get_position_offset() * zoom + gfrom->get_connection_output_position(E.from_port);
		Vector2 from_position = minimap->_convert_from_graph_position(from_slot_position - graph_offset) + minimap_offset;
		Color from_color = gfrom->get_connection_output_color(E.from_port);
		Vector2 to_slot_position = gto->get_position_offset() * zoom + gto->get_connection_input_position(E.to_port);
		Vector2 to_position = minimap->_convert_from_graph_position(to_slot_position - graph_offset) + minimap_offset;
		Color to_color = gto->get_connection_input_color(E.to_port);

		if (E.activity > 0) {
			from_color = from_color.lerp(activity_color, E.activity);
			to_color = to_color.lerp(activity_color, E.activity);
		}
		_draw_connection_line(minimap, from_position, to_position, from_color, to_color, 0.1, minimap->_convert_from_graph_position(Vector2(zoom, zoom)).length());
	}

	// Draw the "camera" viewport.
	Rect2 camera_rect = minimap->get_camera_rect();
	minimap->draw_style_box(minimap->get_theme_stylebox(SNAME("camera")), camera_rect);

	// Draw the resizer control.
	Ref<Texture2D> resizer = minimap->get_theme_icon(SNAME("resizer"));
	Color resizer_color = minimap->get_theme_color(SNAME("resizer_color"));
	minimap->draw_texture(resizer, Point2(), resizer_color);
}

void GraphEdit::set_selected(Node *p_child) {
	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		gn->set_selected(gn == p_child);
	}
}

void GraphEdit::gui_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventMouseMotion> mm = p_ev;
	if (mm.is_valid() && ((mm->get_button_mask() & MouseButton::MASK_MIDDLE) != MouseButton::NONE || ((mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE && Input::get_singleton()->is_key_pressed(Key::SPACE)))) {
		Vector2i relative = Input::get_singleton()->warp_mouse_motion(mm, get_global_rect());
		h_scroll->set_value(h_scroll->get_value() - relative.x);
		v_scroll->set_value(v_scroll->get_value() - relative.y);
	}

	if (mm.is_valid() && dragging) {
		if (!moving_selection) {
			emit_signal(SNAME("begin_node_move"));
			moving_selection = true;
		}

		just_selected = true;
		drag_accum += mm->get_relative();
		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
			if (gn && gn->is_selected()) {
				Vector2 pos = (gn->get_drag_from() * zoom + drag_accum) / zoom;

				// Snapping can be toggled temporarily by holding down Ctrl.
				// This is done here as to not toggle the grid when holding down Ctrl.
				if (is_using_snap() ^ Input::get_singleton()->is_key_pressed(Key::CTRL)) {
					const int snap = get_snap();
					pos = pos.snapped(Vector2(snap, snap));
				}

				gn->set_position_offset(pos);
				if (gn->is_comment()) {
					_set_position_of_comment_enclosed_nodes(gn, comment_enclosed_nodes, drag_accum);
				}
			}
		}
	}

	if (mm.is_valid() && box_selecting) {
		box_selecting_to = mm->get_position();

		box_selecting_rect = Rect2(box_selecting_from.min(box_selecting_to), (box_selecting_from - box_selecting_to).abs());

		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
			if (!gn) {
				continue;
			}

			Rect2 r = gn->get_rect();
			r.size *= zoom;
			bool in_box = r.intersects(box_selecting_rect);

			if (in_box) {
				if (!gn->is_selected() && box_selection_mode_additive) {
					emit_signal(SNAME("node_selected"), gn);
				} else if (gn->is_selected() && !box_selection_mode_additive) {
					emit_signal(SNAME("node_deselected"), gn);
				}
				gn->set_selected(box_selection_mode_additive);
			} else {
				bool select = (previous_selected.find(gn) != nullptr);
				if (gn->is_selected() && !select) {
					emit_signal(SNAME("node_deselected"), gn);
				} else if (!gn->is_selected() && select) {
					emit_signal(SNAME("node_selected"), gn);
				}
				gn->set_selected(select);
			}
		}

		top_layer->update();
		minimap->update();
	}

	Ref<InputEventMouseButton> b = p_ev;
	if (b.is_valid()) {
		if (b->get_button_index() == MouseButton::RIGHT && b->is_pressed()) {
			if (box_selecting) {
				box_selecting = false;
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
					if (!gn) {
						continue;
					}

					bool select = (previous_selected.find(gn) != nullptr);
					if (gn->is_selected() && !select) {
						emit_signal(SNAME("node_deselected"), gn);
					} else if (!gn->is_selected() && select) {
						emit_signal(SNAME("node_selected"), gn);
					}
					gn->set_selected(select);
				}
				top_layer->update();
				minimap->update();
			} else {
				if (connecting) {
					connecting = false;
					top_layer->update();
					minimap->update();
				} else {
					emit_signal(SNAME("popup_request"), get_screen_position() + b->get_position());
				}
			}
		}

		if (b->get_button_index() == MouseButton::LEFT && !b->is_pressed() && dragging) {
			if (!just_selected && drag_accum == Vector2() && Input::get_singleton()->is_key_pressed(Key::CTRL)) {
				//deselect current node
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));

					if (gn) {
						Rect2 r = gn->get_rect();
						r.size *= zoom;
						if (r.has_point(b->get_position())) {
							emit_signal(SNAME("node_deselected"), gn);
							gn->set_selected(false);
						}
					}
				}
			}

			if (drag_accum != Vector2()) {
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
					if (gn && gn->is_selected()) {
						gn->set_drag(false);
						if (gn->is_comment()) {
							_set_drag_comment_enclosed_nodes(gn, comment_enclosed_nodes, false);
						}
					}
				}
			}

			if (moving_selection) {
				emit_signal(SNAME("end_node_move"));
				moving_selection = false;
			}

			dragging = false;

			top_layer->update();
			minimap->update();
			update();
			connections_layer->update();
		}

		if (b->get_button_index() == MouseButton::LEFT && b->is_pressed()) {
			GraphNode *gn = nullptr;

			for (int i = get_child_count() - 1; i >= 0; i--) {
				GraphNode *gn_selected = Object::cast_to<GraphNode>(get_child(i));

				if (gn_selected) {
					if (gn_selected->is_resizing()) {
						continue;
					}

					if (gn_selected->has_point((b->get_position() - gn_selected->get_position()) / zoom)) {
						gn = gn_selected;
						break;
					}
				}
			}

			if (gn) {
				if (_filter_input(b->get_position())) {
					return;
				}

				dragging = true;
				drag_accum = Vector2();
				just_selected = !gn->is_selected();
				if (!gn->is_selected() && !Input::get_singleton()->is_key_pressed(Key::CTRL)) {
					for (int i = 0; i < get_child_count(); i++) {
						GraphNode *o_gn = Object::cast_to<GraphNode>(get_child(i));
						if (o_gn) {
							if (o_gn == gn) {
								o_gn->set_selected(true);
							} else {
								if (o_gn->is_selected()) {
									emit_signal(SNAME("node_deselected"), o_gn);
								}
								o_gn->set_selected(false);
							}
						}
					}
				}

				gn->set_selected(true);
				for (int i = 0; i < get_child_count(); i++) {
					GraphNode *o_gn = Object::cast_to<GraphNode>(get_child(i));
					if (!o_gn) {
						continue;
					}
					if (o_gn->is_selected()) {
						o_gn->set_drag(true);
						if (o_gn->is_comment()) {
							_update_comment_enclosed_nodes_list(o_gn, comment_enclosed_nodes);
							_set_drag_comment_enclosed_nodes(o_gn, comment_enclosed_nodes, true);
						}
					}
				}

			} else {
				if (_filter_input(b->get_position())) {
					return;
				}
				if (Input::get_singleton()->is_key_pressed(Key::SPACE)) {
					return;
				}

				box_selecting = true;
				box_selecting_from = b->get_position();
				if (b->is_ctrl_pressed()) {
					box_selection_mode_additive = true;
					previous_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
						if (!gn2 || !gn2->is_selected()) {
							continue;
						}

						previous_selected.push_back(gn2);
					}
				} else if (b->is_shift_pressed()) {
					box_selection_mode_additive = false;
					previous_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
						if (!gn2 || !gn2->is_selected()) {
							continue;
						}

						previous_selected.push_back(gn2);
					}
				} else {
					box_selection_mode_additive = true;
					previous_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
						if (!gn2) {
							continue;
						}
						if (gn2->is_selected()) {
							emit_signal(SNAME("node_deselected"), gn2);
						}
						gn2->set_selected(false);
					}
				}
			}
		}

		if (b->get_button_index() == MouseButton::LEFT && !b->is_pressed() && box_selecting) {
			box_selecting = false;
			box_selecting_rect = Rect2();
			previous_selected.clear();
			top_layer->update();
			minimap->update();
		}

		int scroll_direction = (b->get_button_index() == MouseButton::WHEEL_DOWN) - (b->get_button_index() == MouseButton::WHEEL_UP);
		if (scroll_direction != 0) {
			if (b->is_ctrl_pressed()) {
				if (b->is_shift_pressed()) {
					// Horizontal scrolling.
					h_scroll->set_value(h_scroll->get_value() + (h_scroll->get_page() * b->get_factor() / 8) * scroll_direction);
				} else {
					// Vertical scrolling.
					v_scroll->set_value(v_scroll->get_value() + (v_scroll->get_page() * b->get_factor() / 8) * scroll_direction);
				}
			} else {
				// Zooming.
				set_zoom_custom(scroll_direction < 0 ? zoom * zoom_step : zoom / zoom_step, b->get_position());
			}
		}
	}

	if (p_ev->is_pressed()) {
		if (p_ev->is_action("ui_graph_duplicate")) {
			emit_signal(SNAME("duplicate_nodes_request"));
			accept_event();
		} else if (p_ev->is_action("ui_copy")) {
			emit_signal(SNAME("copy_nodes_request"));
			accept_event();
		} else if (p_ev->is_action("ui_paste")) {
			emit_signal(SNAME("paste_nodes_request"));
			accept_event();
		} else if (p_ev->is_action("ui_graph_delete")) {
			emit_signal(SNAME("delete_nodes_request"));
			accept_event();
		}
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_ev;
	if (magnify_gesture.is_valid()) {
		set_zoom_custom(zoom * magnify_gesture->get_factor(), magnify_gesture->get_position());
	}

	Ref<InputEventPanGesture> pan_gesture = p_ev;
	if (pan_gesture.is_valid()) {
		h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * pan_gesture->get_delta().x / 8);
		v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * pan_gesture->get_delta().y / 8);
	}
}

void GraphEdit::set_connection_activity(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port, float p_activity) {
	for (Connection &E : connections) {
		if (E.from == p_from && E.from_port == p_from_port && E.to == p_to && E.to_port == p_to_port) {
			if (Math::is_equal_approx(E.activity, p_activity)) {
				//update only if changed
				top_layer->update();
				minimap->update();
				connections_layer->update();
			}
			E.activity = p_activity;
			return;
		}
	}
}

void GraphEdit::clear_connections() {
	connections.clear();
	minimap->update();
	update();
	connections_layer->update();
}

void GraphEdit::set_zoom(float p_zoom) {
	set_zoom_custom(p_zoom, get_size() / 2);
}

void GraphEdit::set_zoom_custom(float p_zoom, const Vector2 &p_center) {
	p_zoom = CLAMP(p_zoom, zoom_min, zoom_max);
	if (zoom == p_zoom) {
		return;
	}

	Vector2 sbofs = (Vector2(h_scroll->get_value(), v_scroll->get_value()) + p_center) / zoom;

	zoom = p_zoom;
	top_layer->update();

	zoom_minus->set_disabled(zoom == zoom_min);
	zoom_plus->set_disabled(zoom == zoom_max);

	_update_scroll();
	minimap->update();
	connections_layer->update();

	if (is_visible_in_tree()) {
		Vector2 ofs = sbofs * zoom - p_center;
		h_scroll->set_value(ofs.x);
		v_scroll->set_value(ofs.y);
	}

	_update_zoom_label();
	update();
}

float GraphEdit::get_zoom() const {
	return zoom;
}

void GraphEdit::set_zoom_step(float p_zoom_step) {
	p_zoom_step = abs(p_zoom_step);
	if (zoom_step == p_zoom_step) {
		return;
	}

	zoom_step = p_zoom_step;
}

float GraphEdit::get_zoom_step() const {
	return zoom_step;
}

void GraphEdit::set_zoom_min(float p_zoom_min) {
	ERR_FAIL_COND_MSG(p_zoom_min > zoom_max, "Cannot set min zoom level greater than max zoom level.");

	if (zoom_min == p_zoom_min) {
		return;
	}

	zoom_min = p_zoom_min;
	set_zoom(zoom);
}

float GraphEdit::get_zoom_min() const {
	return zoom_min;
}

void GraphEdit::set_zoom_max(float p_zoom_max) {
	ERR_FAIL_COND_MSG(p_zoom_max < zoom_min, "Cannot set max zoom level lesser than min zoom level.");

	if (zoom_max == p_zoom_max) {
		return;
	}

	zoom_max = p_zoom_max;
	set_zoom(zoom);
}

float GraphEdit::get_zoom_max() const {
	return zoom_max;
}

void GraphEdit::set_show_zoom_label(bool p_enable) {
	if (zoom_label->is_visible() == p_enable) {
		return;
	}

	zoom_label->set_visible(p_enable);
}

bool GraphEdit::is_showing_zoom_label() const {
	return zoom_label->is_visible();
}

void GraphEdit::set_right_disconnects(bool p_enable) {
	right_disconnects = p_enable;
}

bool GraphEdit::is_right_disconnects_enabled() const {
	return right_disconnects;
}

void GraphEdit::add_valid_right_disconnect_type(int p_type) {
	valid_right_disconnect_types.insert(p_type);
}

void GraphEdit::remove_valid_right_disconnect_type(int p_type) {
	valid_right_disconnect_types.erase(p_type);
}

void GraphEdit::add_valid_left_disconnect_type(int p_type) {
	valid_left_disconnect_types.insert(p_type);
}

void GraphEdit::remove_valid_left_disconnect_type(int p_type) {
	valid_left_disconnect_types.erase(p_type);
}

Array GraphEdit::_get_connection_list() const {
	List<Connection> conns;
	get_connection_list(&conns);
	Array arr;
	for (const Connection &E : conns) {
		Dictionary d;
		d["from"] = E.from;
		d["from_port"] = E.from_port;
		d["to"] = E.to;
		d["to_port"] = E.to_port;
		arr.push_back(d);
	}
	return arr;
}

void GraphEdit::_zoom_minus() {
	set_zoom(zoom / zoom_step);
}

void GraphEdit::_zoom_reset() {
	set_zoom(1);
}

void GraphEdit::_zoom_plus() {
	set_zoom(zoom * zoom_step);
}

void GraphEdit::_update_zoom_label() {
	int zoom_percent = static_cast<int>(Math::round(zoom * 100));
	String zoom_text = itos(zoom_percent) + "%";
	zoom_label->set_text(zoom_text);
}

void GraphEdit::add_valid_connection_type(int p_type, int p_with_type) {
	ConnType ct;
	ct.type_a = p_type;
	ct.type_b = p_with_type;

	valid_connection_types.insert(ct);
}

void GraphEdit::remove_valid_connection_type(int p_type, int p_with_type) {
	ConnType ct;
	ct.type_a = p_type;
	ct.type_b = p_with_type;

	valid_connection_types.erase(ct);
}

bool GraphEdit::is_valid_connection_type(int p_type, int p_with_type) const {
	ConnType ct;
	ct.type_a = p_type;
	ct.type_b = p_with_type;

	return valid_connection_types.has(ct);
}

void GraphEdit::set_use_snap(bool p_enable) {
	snap_button->set_pressed(p_enable);
	update();
}

bool GraphEdit::is_using_snap() const {
	return snap_button->is_pressed();
}

int GraphEdit::get_snap() const {
	return snap_amount->get_value();
}

void GraphEdit::set_snap(int p_snap) {
	ERR_FAIL_COND(p_snap < 5);
	snap_amount->set_value(p_snap);
	update();
}

void GraphEdit::_snap_toggled() {
	update();
}

void GraphEdit::_snap_value_changed(double) {
	update();
}

void GraphEdit::set_minimap_size(Vector2 p_size) {
	minimap->set_size(p_size);
	Vector2 minimap_size = minimap->get_size(); // The size might've been adjusted by the minimum size.

	minimap->set_anchors_preset(Control::PRESET_BOTTOM_RIGHT);
	minimap->set_offset(Side::SIDE_LEFT, -minimap_size.x - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_TOP, -minimap_size.y - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_RIGHT, -MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_BOTTOM, -MINIMAP_OFFSET);
	minimap->update();
}

Vector2 GraphEdit::get_minimap_size() const {
	return minimap->get_size();
}

void GraphEdit::set_minimap_opacity(float p_opacity) {
	minimap->set_modulate(Color(1, 1, 1, p_opacity));
	minimap->update();
}

float GraphEdit::get_minimap_opacity() const {
	Color minimap_modulate = minimap->get_modulate();
	return minimap_modulate.a;
}

void GraphEdit::set_minimap_enabled(bool p_enable) {
	minimap_button->set_pressed(p_enable);
	minimap->update();
}

bool GraphEdit::is_minimap_enabled() const {
	return minimap_button->is_pressed();
}

void GraphEdit::_minimap_toggled() {
	if (is_minimap_enabled()) {
		minimap->set_visible(true);
		minimap->update();
	} else {
		minimap->set_visible(false);
	}
}

void GraphEdit::set_connection_lines_thickness(float p_thickness) {
	lines_thickness = p_thickness;
	update();
}

float GraphEdit::get_connection_lines_thickness() const {
	return lines_thickness;
}

void GraphEdit::set_connection_lines_antialiased(bool p_antialiased) {
	lines_antialiased = p_antialiased;
	update();
}

bool GraphEdit::is_connection_lines_antialiased() const {
	return lines_antialiased;
}

HBoxContainer *GraphEdit::get_zoom_hbox() {
	return zoom_hb;
}

int GraphEdit::_set_operations(SET_OPERATIONS p_operation, Set<StringName> &r_u, const Set<StringName> &r_v) {
	switch (p_operation) {
		case GraphEdit::IS_EQUAL: {
			for (Set<StringName>::Element *E = r_u.front(); E; E = E->next()) {
				if (!r_v.has(E->get()))
					return 0;
			}
			return r_u.size() == r_v.size();
		} break;
		case GraphEdit::IS_SUBSET: {
			if (r_u.size() == r_v.size() && !r_u.size()) {
				return 1;
			}
			for (Set<StringName>::Element *E = r_u.front(); E; E = E->next()) {
				if (!r_v.has(E->get()))
					return 0;
			}
			return 1;
		} break;
		case GraphEdit::DIFFERENCE: {
			for (Set<StringName>::Element *E = r_u.front(); E; E = E->next()) {
				if (r_v.has(E->get())) {
					r_u.erase(E->get());
				}
			}
			return r_u.size();
		} break;
		case GraphEdit::UNION: {
			for (Set<StringName>::Element *E = r_v.front(); E; E = E->next()) {
				if (!r_u.has(E->get())) {
					r_u.insert(E->get());
				}
			}
			return r_v.size();
		} break;
		default:
			break;
	}
	return -1;
}

HashMap<int, Vector<StringName>> GraphEdit::_layering(const Set<StringName> &r_selected_nodes, const HashMap<StringName, Set<StringName>> &r_upper_neighbours) {
	HashMap<int, Vector<StringName>> l;

	Set<StringName> p = r_selected_nodes, q = r_selected_nodes, u, z;
	int current_layer = 0;
	bool selected = false;

	while (!_set_operations(GraphEdit::IS_EQUAL, q, u)) {
		_set_operations(GraphEdit::DIFFERENCE, p, u);
		for (const Set<StringName>::Element *E = p.front(); E; E = E->next()) {
			Set<StringName> n = r_upper_neighbours[E->get()];
			if (_set_operations(GraphEdit::IS_SUBSET, n, z)) {
				Vector<StringName> t;
				t.push_back(E->get());
				if (!l.has(current_layer)) {
					l.set(current_layer, Vector<StringName>{});
				}
				selected = true;
				t.append_array(l[current_layer]);
				l.set(current_layer, t);
				Set<StringName> V;
				V.insert(E->get());
				_set_operations(GraphEdit::UNION, u, V);
			}
		}
		if (!selected) {
			current_layer++;
			_set_operations(GraphEdit::UNION, z, u);
		}
		selected = false;
	}

	return l;
}

Vector<StringName> GraphEdit::_split(const Vector<StringName> &r_layer, const HashMap<StringName, Dictionary> &r_crossings) {
	if (!r_layer.size()) {
		return Vector<StringName>();
	}

	StringName p = r_layer[Math::random(0, r_layer.size() - 1)];
	Vector<StringName> left;
	Vector<StringName> right;

	for (int i = 0; i < r_layer.size(); i++) {
		if (p != r_layer[i]) {
			StringName q = r_layer[i];
			int cross_pq = r_crossings[p][q];
			int cross_qp = r_crossings[q][p];
			if (cross_pq > cross_qp) {
				left.push_back(q);
			} else {
				right.push_back(q);
			}
		}
	}

	left.push_back(p);
	left.append_array(right);
	return left;
}

void GraphEdit::_horizontal_alignment(Dictionary &r_root, Dictionary &r_align, const HashMap<int, Vector<StringName>> &r_layers, const HashMap<StringName, Set<StringName>> &r_upper_neighbours, const Set<StringName> &r_selected_nodes) {
	for (const Set<StringName>::Element *E = r_selected_nodes.front(); E; E = E->next()) {
		r_root[E->get()] = E->get();
		r_align[E->get()] = E->get();
	}

	if (r_layers.size() == 1) {
		return;
	}

	for (unsigned int i = 1; i < r_layers.size(); i++) {
		Vector<StringName> lower_layer = r_layers[i];
		Vector<StringName> upper_layer = r_layers[i - 1];
		int r = -1;

		for (int j = 0; j < lower_layer.size(); j++) {
			Vector<Pair<int, StringName>> up;
			StringName current_node = lower_layer[j];
			for (int k = 0; k < upper_layer.size(); k++) {
				StringName adjacent_neighbour = upper_layer[k];
				if (r_upper_neighbours[current_node].has(adjacent_neighbour)) {
					up.push_back(Pair<int, StringName>(k, adjacent_neighbour));
				}
			}

			int start = up.size() / 2;
			int end = up.size() % 2 ? start : start + 1;
			for (int p = start; p <= end; p++) {
				StringName Align = r_align[current_node];
				if (Align == current_node && r < up[p].first) {
					r_align[up[p].second] = lower_layer[j];
					r_root[current_node] = r_root[up[p].second];
					r_align[current_node] = r_root[up[p].second];
					r = up[p].first;
				}
			}
		}
	}
}

void GraphEdit::_crossing_minimisation(HashMap<int, Vector<StringName>> &r_layers, const HashMap<StringName, Set<StringName>> &r_upper_neighbours) {
	if (r_layers.size() == 1) {
		return;
	}

	for (unsigned int i = 1; i < r_layers.size(); i++) {
		Vector<StringName> upper_layer = r_layers[i - 1];
		Vector<StringName> lower_layer = r_layers[i];
		HashMap<StringName, Dictionary> c;

		for (int j = 0; j < lower_layer.size(); j++) {
			StringName p = lower_layer[j];
			Dictionary d;

			for (int k = 0; k < lower_layer.size(); k++) {
				unsigned int crossings = 0;
				StringName q = lower_layer[k];

				if (j != k) {
					for (int h = 1; h < upper_layer.size(); h++) {
						if (r_upper_neighbours[p].has(upper_layer[h])) {
							for (int g = 0; g < h; g++) {
								if (r_upper_neighbours[q].has(upper_layer[g])) {
									crossings++;
								}
							}
						}
					}
				}
				d[q] = crossings;
			}
			c.set(p, d);
		}

		r_layers.set(i, _split(lower_layer, c));
	}
}

void GraphEdit::_calculate_inner_shifts(Dictionary &r_inner_shifts, const Dictionary &r_root, const Dictionary &r_node_names, const Dictionary &r_align, const Set<StringName> &r_block_heads, const HashMap<StringName, Pair<int, int>> &r_port_info) {
	for (const Set<StringName>::Element *E = r_block_heads.front(); E; E = E->next()) {
		real_t left = 0;
		StringName u = E->get();
		StringName v = r_align[u];
		while (u != v && (StringName)r_root[u] != v) {
			String _connection = String(u) + " " + String(v);
			GraphNode *gfrom = Object::cast_to<GraphNode>(r_node_names[u]);
			GraphNode *gto = Object::cast_to<GraphNode>(r_node_names[v]);

			Pair<int, int> ports = r_port_info[_connection];
			int pfrom = ports.first;
			int pto = ports.second;
			Vector2 frompos = gfrom->get_connection_output_position(pfrom);
			Vector2 topos = gto->get_connection_input_position(pto);

			real_t s = (real_t)r_inner_shifts[u] + (frompos.y - topos.y) / zoom;
			r_inner_shifts[v] = s;
			left = MIN(left, s);

			u = v;
			v = (StringName)r_align[v];
		}

		u = E->get();
		do {
			r_inner_shifts[u] = (real_t)r_inner_shifts[u] - left;
			u = (StringName)r_align[u];
		} while (u != E->get());
	}
}

float GraphEdit::_calculate_threshold(StringName p_v, StringName p_w, const Dictionary &r_node_names, const HashMap<int, Vector<StringName>> &r_layers, const Dictionary &r_root, const Dictionary &r_align, const Dictionary &r_inner_shift, real_t p_current_threshold, const HashMap<StringName, Vector2> &r_node_positions) {
#define MAX_ORDER 2147483647
#define ORDER(node, layers)                            \
	for (unsigned int i = 0; i < layers.size(); i++) { \
		int index = layers[i].find(node);              \
		if (index > 0) {                               \
			order = index;                             \
			break;                                     \
		}                                              \
		order = MAX_ORDER;                             \
	}

	int order = MAX_ORDER;
	float threshold = p_current_threshold;
	if (p_v == p_w) {
		int min_order = MAX_ORDER;
		Connection incoming;
		for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
			if (E->get().to == p_w) {
				ORDER(E->get().from, r_layers);
				if (min_order > order) {
					min_order = order;
					incoming = E->get();
				}
			}
		}

		if (incoming.from != StringName()) {
			GraphNode *gfrom = Object::cast_to<GraphNode>(r_node_names[incoming.from]);
			GraphNode *gto = Object::cast_to<GraphNode>(r_node_names[p_w]);
			Vector2 frompos = gfrom->get_connection_output_position(incoming.from_port);
			Vector2 topos = gto->get_connection_input_position(incoming.to_port);

			//If connected block node is selected, calculate thershold or add current block to list
			if (gfrom->is_selected()) {
				Vector2 connected_block_pos = r_node_positions[r_root[incoming.from]];
				if (connected_block_pos.y != FLT_MAX) {
					//Connected block is placed. Calculate threshold
					threshold = connected_block_pos.y + (real_t)r_inner_shift[incoming.from] - (real_t)r_inner_shift[p_w] + frompos.y - topos.y;
				}
			}
		}
	}
	if (threshold == FLT_MIN && (StringName)r_align[p_w] == p_v) {
		//This time, pick an outgoing edge and repeat as above!
		int min_order = MAX_ORDER;
		Connection outgoing;
		for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
			if (E->get().from == p_w) {
				ORDER(E->get().to, r_layers);
				if (min_order > order) {
					min_order = order;
					outgoing = E->get();
				}
			}
		}

		if (outgoing.to != StringName()) {
			GraphNode *gfrom = Object::cast_to<GraphNode>(r_node_names[p_w]);
			GraphNode *gto = Object::cast_to<GraphNode>(r_node_names[outgoing.to]);
			Vector2 frompos = gfrom->get_connection_output_position(outgoing.from_port);
			Vector2 topos = gto->get_connection_input_position(outgoing.to_port);

			//If connected block node is selected, calculate thershold or add current block to list
			if (gto->is_selected()) {
				Vector2 connected_block_pos = r_node_positions[r_root[outgoing.to]];
				if (connected_block_pos.y != FLT_MAX) {
					//Connected block is placed. Calculate threshold
					threshold = connected_block_pos.y + (real_t)r_inner_shift[outgoing.to] - (real_t)r_inner_shift[p_w] + frompos.y - topos.y;
				}
			}
		}
	}
#undef MAX_ORDER
#undef ORDER
	return threshold;
}

void GraphEdit::_place_block(StringName p_v, float p_delta, const HashMap<int, Vector<StringName>> &r_layers, const Dictionary &r_root, const Dictionary &r_align, const Dictionary &r_node_name, const Dictionary &r_inner_shift, Dictionary &r_sink, Dictionary &r_shift, HashMap<StringName, Vector2> &r_node_positions) {
#define PRED(node, layers)                             \
	for (unsigned int i = 0; i < layers.size(); i++) { \
		int index = layers[i].find(node);              \
		if (index > 0) {                               \
			predecessor = layers[i][index - 1];        \
			break;                                     \
		}                                              \
		predecessor = StringName();                    \
	}

	StringName predecessor;
	StringName successor;
	Vector2 pos = r_node_positions[p_v];

	if (pos.y == FLT_MAX) {
		pos.y = 0;
		bool initial = false;
		StringName w = p_v;
		real_t threshold = FLT_MIN;
		do {
			PRED(w, r_layers);
			if (predecessor != StringName()) {
				StringName u = r_root[predecessor];
				_place_block(u, p_delta, r_layers, r_root, r_align, r_node_name, r_inner_shift, r_sink, r_shift, r_node_positions);
				threshold = _calculate_threshold(p_v, w, r_node_name, r_layers, r_root, r_align, r_inner_shift, threshold, r_node_positions);
				if ((StringName)r_sink[p_v] == p_v) {
					r_sink[p_v] = r_sink[u];
				}

				Vector2 predecessor_root_pos = r_node_positions[u];
				Vector2 predecessor_node_size = Object::cast_to<GraphNode>(r_node_name[predecessor])->get_size();
				if (r_sink[p_v] != r_sink[u]) {
					real_t sc = pos.y + (real_t)r_inner_shift[w] - predecessor_root_pos.y - (real_t)r_inner_shift[predecessor] - predecessor_node_size.y - p_delta;
					r_shift[r_sink[u]] = MIN(sc, (real_t)r_shift[r_sink[u]]);
				} else {
					real_t sb = predecessor_root_pos.y + (real_t)r_inner_shift[predecessor] + predecessor_node_size.y - (real_t)r_inner_shift[w] + p_delta;
					sb = MAX(sb, threshold);
					if (initial) {
						pos.y = sb;
					} else {
						pos.y = MAX(pos.y, sb);
					}
					initial = false;
				}
			}
			threshold = _calculate_threshold(p_v, w, r_node_name, r_layers, r_root, r_align, r_inner_shift, threshold, r_node_positions);
			w = r_align[w];
		} while (w != p_v);
		r_node_positions.set(p_v, pos);
	}

#undef PRED
}

void GraphEdit::arrange_nodes() {
	if (!arranging_graph) {
		arranging_graph = true;
	} else {
		return;
	}

	Dictionary node_names;
	Set<StringName> selected_nodes;

	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		node_names[gn->get_name()] = gn;
	}

	HashMap<StringName, Set<StringName>> upper_neighbours;
	HashMap<StringName, Pair<int, int>> port_info;
	Vector2 origin(FLT_MAX, FLT_MAX);

	float gap_v = 100.0f;
	float gap_h = 100.0f;

	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		if (gn->is_selected()) {
			selected_nodes.insert(gn->get_name());
			Set<StringName> s;
			for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
				GraphNode *p_from = Object::cast_to<GraphNode>(node_names[E->get().from]);
				if (E->get().to == gn->get_name() && p_from->is_selected()) {
					if (!s.has(p_from->get_name())) {
						s.insert(p_from->get_name());
					}
					String s_connection = String(p_from->get_name()) + " " + String(E->get().to);
					StringName _connection(s_connection);
					Pair<int, int> ports(E->get().from_port, E->get().to_port);
					if (port_info.has(_connection)) {
						Pair<int, int> p_ports = port_info[_connection];
						if (p_ports.first < ports.first) {
							ports = p_ports;
						}
					}
					port_info.set(_connection, ports);
				}
			}
			upper_neighbours.set(gn->get_name(), s);
		}
	}

	if (!selected_nodes.size()) {
		arranging_graph = false;
		return;
	}

	HashMap<int, Vector<StringName>> layers = _layering(selected_nodes, upper_neighbours);
	_crossing_minimisation(layers, upper_neighbours);

	Dictionary root, align, sink, shift;
	_horizontal_alignment(root, align, layers, upper_neighbours, selected_nodes);

	HashMap<StringName, Vector2> new_positions;
	Vector2 default_position(FLT_MAX, FLT_MAX);
	Dictionary inner_shift;
	Set<StringName> block_heads;

	for (const Set<StringName>::Element *E = selected_nodes.front(); E; E = E->next()) {
		inner_shift[E->get()] = 0.0f;
		sink[E->get()] = E->get();
		shift[E->get()] = FLT_MAX;
		new_positions.set(E->get(), default_position);
		if ((StringName)root[E->get()] == E->get()) {
			block_heads.insert(E->get());
		}
	}

	_calculate_inner_shifts(inner_shift, root, node_names, align, block_heads, port_info);

	for (const Set<StringName>::Element *E = block_heads.front(); E; E = E->next()) {
		_place_block(E->get(), gap_v, layers, root, align, node_names, inner_shift, sink, shift, new_positions);
	}
	origin.y = Object::cast_to<GraphNode>(node_names[layers[0][0]])->get_position_offset().y - (new_positions[layers[0][0]].y + (float)inner_shift[layers[0][0]]);
	origin.x = Object::cast_to<GraphNode>(node_names[layers[0][0]])->get_position_offset().x;

	for (const Set<StringName>::Element *E = block_heads.front(); E; E = E->next()) {
		StringName u = E->get();
		float start_from = origin.y + new_positions[E->get()].y;
		do {
			Vector2 cal_pos;
			cal_pos.y = start_from + (real_t)inner_shift[u];
			new_positions.set(u, cal_pos);
			u = align[u];
		} while (u != E->get());
	}

	// Compute horizontal coordinates individually for layers to get uniform gap.
	float start_from = origin.x;
	float largest_node_size = 0.0f;

	for (unsigned int i = 0; i < layers.size(); i++) {
		Vector<StringName> layer = layers[i];
		for (int j = 0; j < layer.size(); j++) {
			float current_node_size = Object::cast_to<GraphNode>(node_names[layer[j]])->get_size().x;
			largest_node_size = MAX(largest_node_size, current_node_size);
		}

		for (int j = 0; j < layer.size(); j++) {
			float current_node_size = Object::cast_to<GraphNode>(node_names[layer[j]])->get_size().x;
			Vector2 cal_pos = new_positions[layer[j]];

			if (current_node_size == largest_node_size) {
				cal_pos.x = start_from;
			} else {
				float current_node_start_pos = start_from;
				if (current_node_size < largest_node_size / 2) {
					if (!(i || j)) {
						start_from -= (largest_node_size - current_node_size);
					}
					current_node_start_pos = start_from + largest_node_size - current_node_size;
				}
				cal_pos.x = current_node_start_pos;
			}
			new_positions.set(layer[j], cal_pos);
		}

		start_from += largest_node_size + gap_h;
		largest_node_size = 0.0f;
	}

	emit_signal(SNAME("begin_node_move"));
	for (const Set<StringName>::Element *E = selected_nodes.front(); E; E = E->next()) {
		GraphNode *gn = Object::cast_to<GraphNode>(node_names[E->get()]);
		gn->set_drag(true);
		Vector2 pos = (new_positions[E->get()]);

		if (is_using_snap()) {
			const int snap = get_snap();
			pos = pos.snapped(Vector2(snap, snap));
		}
		gn->set_position_offset(pos);
		gn->set_drag(false);
	}
	emit_signal(SNAME("end_node_move"));
	arranging_graph = false;
}

void GraphEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("connect_node", "from", "from_port", "to", "to_port"), &GraphEdit::connect_node);
	ClassDB::bind_method(D_METHOD("is_node_connected", "from", "from_port", "to", "to_port"), &GraphEdit::is_node_connected);
	ClassDB::bind_method(D_METHOD("disconnect_node", "from", "from_port", "to", "to_port"), &GraphEdit::disconnect_node);
	ClassDB::bind_method(D_METHOD("set_connection_activity", "from", "from_port", "to", "to_port", "amount"), &GraphEdit::set_connection_activity);
	ClassDB::bind_method(D_METHOD("get_connection_list"), &GraphEdit::_get_connection_list);
	ClassDB::bind_method(D_METHOD("clear_connections"), &GraphEdit::clear_connections);
	ClassDB::bind_method(D_METHOD("get_scroll_ofs"), &GraphEdit::get_scroll_ofs);
	ClassDB::bind_method(D_METHOD("set_scroll_ofs", "ofs"), &GraphEdit::set_scroll_ofs);

	ClassDB::bind_method(D_METHOD("add_valid_right_disconnect_type", "type"), &GraphEdit::add_valid_right_disconnect_type);
	ClassDB::bind_method(D_METHOD("remove_valid_right_disconnect_type", "type"), &GraphEdit::remove_valid_right_disconnect_type);
	ClassDB::bind_method(D_METHOD("add_valid_left_disconnect_type", "type"), &GraphEdit::add_valid_left_disconnect_type);
	ClassDB::bind_method(D_METHOD("remove_valid_left_disconnect_type", "type"), &GraphEdit::remove_valid_left_disconnect_type);
	ClassDB::bind_method(D_METHOD("add_valid_connection_type", "from_type", "to_type"), &GraphEdit::add_valid_connection_type);
	ClassDB::bind_method(D_METHOD("remove_valid_connection_type", "from_type", "to_type"), &GraphEdit::remove_valid_connection_type);
	ClassDB::bind_method(D_METHOD("is_valid_connection_type", "from_type", "to_type"), &GraphEdit::is_valid_connection_type);
	ClassDB::bind_method(D_METHOD("get_connection_line", "from", "to"), &GraphEdit::get_connection_line);

	ClassDB::bind_method(D_METHOD("set_zoom", "zoom"), &GraphEdit::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &GraphEdit::get_zoom);

	ClassDB::bind_method(D_METHOD("set_zoom_min", "zoom_min"), &GraphEdit::set_zoom_min);
	ClassDB::bind_method(D_METHOD("get_zoom_min"), &GraphEdit::get_zoom_min);

	ClassDB::bind_method(D_METHOD("set_zoom_max", "zoom_max"), &GraphEdit::set_zoom_max);
	ClassDB::bind_method(D_METHOD("get_zoom_max"), &GraphEdit::get_zoom_max);

	ClassDB::bind_method(D_METHOD("set_zoom_step", "zoom_step"), &GraphEdit::set_zoom_step);
	ClassDB::bind_method(D_METHOD("get_zoom_step"), &GraphEdit::get_zoom_step);

	ClassDB::bind_method(D_METHOD("set_show_zoom_label", "enable"), &GraphEdit::set_show_zoom_label);
	ClassDB::bind_method(D_METHOD("is_showing_zoom_label"), &GraphEdit::is_showing_zoom_label);

	ClassDB::bind_method(D_METHOD("set_snap", "pixels"), &GraphEdit::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &GraphEdit::get_snap);

	ClassDB::bind_method(D_METHOD("set_use_snap", "enable"), &GraphEdit::set_use_snap);
	ClassDB::bind_method(D_METHOD("is_using_snap"), &GraphEdit::is_using_snap);

	ClassDB::bind_method(D_METHOD("set_connection_lines_thickness", "pixels"), &GraphEdit::set_connection_lines_thickness);
	ClassDB::bind_method(D_METHOD("get_connection_lines_thickness"), &GraphEdit::get_connection_lines_thickness);

	ClassDB::bind_method(D_METHOD("set_connection_lines_antialiased", "pixels"), &GraphEdit::set_connection_lines_antialiased);
	ClassDB::bind_method(D_METHOD("is_connection_lines_antialiased"), &GraphEdit::is_connection_lines_antialiased);

	ClassDB::bind_method(D_METHOD("set_minimap_size", "size"), &GraphEdit::set_minimap_size);
	ClassDB::bind_method(D_METHOD("get_minimap_size"), &GraphEdit::get_minimap_size);
	ClassDB::bind_method(D_METHOD("set_minimap_opacity", "opacity"), &GraphEdit::set_minimap_opacity);
	ClassDB::bind_method(D_METHOD("get_minimap_opacity"), &GraphEdit::get_minimap_opacity);

	ClassDB::bind_method(D_METHOD("set_minimap_enabled", "enable"), &GraphEdit::set_minimap_enabled);
	ClassDB::bind_method(D_METHOD("is_minimap_enabled"), &GraphEdit::is_minimap_enabled);

	ClassDB::bind_method(D_METHOD("set_right_disconnects", "enable"), &GraphEdit::set_right_disconnects);
	ClassDB::bind_method(D_METHOD("is_right_disconnects_enabled"), &GraphEdit::is_right_disconnects_enabled);

	ClassDB::bind_method(D_METHOD("_update_scroll_offset"), &GraphEdit::_update_scroll_offset);
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_is_in_input_hotzone", PropertyInfo(Variant::OBJECT, "graph_node"), PropertyInfo(Variant::INT, "slot_index"), PropertyInfo(Variant::VECTOR2, "mouse_position")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_is_in_output_hotzone", PropertyInfo(Variant::OBJECT, "graph_node"), PropertyInfo(Variant::INT, "slot_index"), PropertyInfo(Variant::VECTOR2, "mouse_position")));

	ClassDB::bind_method(D_METHOD("get_zoom_hbox"), &GraphEdit::get_zoom_hbox);

	ClassDB::bind_method(D_METHOD("arrange_nodes"), &GraphEdit::arrange_nodes);

	ClassDB::bind_method(D_METHOD("set_selected", "node"), &GraphEdit::set_selected);

	GDVIRTUAL_BIND(_get_connection_line, "from", "to")

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "right_disconnects"), "set_right_disconnects", "is_right_disconnects_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scroll_offset"), "set_scroll_ofs", "get_scroll_ofs");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "snap_distance"), "set_snap", "get_snap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_snap"), "set_use_snap", "is_using_snap");

	ADD_GROUP("Connection Lines", "connection_lines");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "connection_lines_thickness"), "set_connection_lines_thickness", "get_connection_lines_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "connection_lines_antialiased"), "set_connection_lines_antialiased", "is_connection_lines_antialiased");

	ADD_GROUP("Zoom", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom"), "set_zoom", "get_zoom");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom_min"), "set_zoom_min", "get_zoom_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom_max"), "set_zoom_max", "get_zoom_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom_step"), "set_zoom_step", "get_zoom_step");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_zoom_label"), "set_show_zoom_label", "is_showing_zoom_label");

	ADD_GROUP("Minimap", "minimap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "minimap_enabled"), "set_minimap_enabled", "is_minimap_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "minimap_size"), "set_minimap_size", "get_minimap_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "minimap_opacity"), "set_minimap_opacity", "get_minimap_opacity");

	ADD_SIGNAL(MethodInfo("connection_request", PropertyInfo(Variant::STRING_NAME, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::STRING_NAME, "to"), PropertyInfo(Variant::INT, "to_slot")));
	ADD_SIGNAL(MethodInfo("disconnection_request", PropertyInfo(Variant::STRING_NAME, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::STRING_NAME, "to"), PropertyInfo(Variant::INT, "to_slot")));
	ADD_SIGNAL(MethodInfo("popup_request", PropertyInfo(Variant::VECTOR2, "position")));
	ADD_SIGNAL(MethodInfo("duplicate_nodes_request"));
	ADD_SIGNAL(MethodInfo("copy_nodes_request"));
	ADD_SIGNAL(MethodInfo("paste_nodes_request"));
	ADD_SIGNAL(MethodInfo("node_selected", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_deselected", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("connection_to_empty", PropertyInfo(Variant::STRING_NAME, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::VECTOR2, "release_position")));
	ADD_SIGNAL(MethodInfo("connection_from_empty", PropertyInfo(Variant::STRING_NAME, "to"), PropertyInfo(Variant::INT, "to_slot"), PropertyInfo(Variant::VECTOR2, "release_position")));
	ADD_SIGNAL(MethodInfo("delete_nodes_request"));
	ADD_SIGNAL(MethodInfo("begin_node_move"));
	ADD_SIGNAL(MethodInfo("end_node_move"));
	ADD_SIGNAL(MethodInfo("scroll_offset_changed", PropertyInfo(Variant::VECTOR2, "ofs")));
}

GraphEdit::GraphEdit() {
	set_focus_mode(FOCUS_ALL);

	// Allow dezooming 8 times from the default zoom level.
	// At low zoom levels, text is unreadable due to its small size and poor filtering,
	// but this is still useful for previewing and navigation.
	zoom_min = (1 / Math::pow(zoom_step, 8));
	// Allow zooming 4 times from the default zoom level.
	zoom_max = (1 * Math::pow(zoom_step, 4));

	top_layer = memnew(GraphEditFilter(this));
	add_child(top_layer, false, INTERNAL_MODE_BACK);
	top_layer->set_mouse_filter(MOUSE_FILTER_PASS);
	top_layer->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	top_layer->connect("draw", callable_mp(this, &GraphEdit::_top_layer_draw));
	top_layer->connect("gui_input", callable_mp(this, &GraphEdit::_top_layer_input));

	connections_layer = memnew(Control);
	add_child(connections_layer, false, INTERNAL_MODE_FRONT);
	connections_layer->connect("draw", callable_mp(this, &GraphEdit::_connections_layer_draw));
	connections_layer->set_name("CLAYER");
	connections_layer->set_disable_visibility_clip(true); // so it can draw freely and be offset
	connections_layer->set_mouse_filter(MOUSE_FILTER_IGNORE);

	h_scroll = memnew(HScrollBar);
	h_scroll->set_name("_h_scroll");
	top_layer->add_child(h_scroll);

	v_scroll = memnew(VScrollBar);
	v_scroll->set_name("_v_scroll");
	top_layer->add_child(v_scroll);

	//set large minmax so it can scroll even if not resized yet
	h_scroll->set_min(-10000);
	h_scroll->set_max(10000);

	v_scroll->set_min(-10000);
	v_scroll->set_max(10000);

	h_scroll->connect("value_changed", callable_mp(this, &GraphEdit::_scroll_moved));
	v_scroll->connect("value_changed", callable_mp(this, &GraphEdit::_scroll_moved));

	zoom_hb = memnew(HBoxContainer);
	top_layer->add_child(zoom_hb);
	zoom_hb->set_position(Vector2(10, 10));

	zoom_label = memnew(Label);
	zoom_hb->add_child(zoom_label);
	zoom_label->set_visible(false);
	zoom_label->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	zoom_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	zoom_label->set_custom_minimum_size(Size2(48, 0));
	_update_zoom_label();

	zoom_minus = memnew(Button);
	zoom_minus->set_flat(true);
	zoom_hb->add_child(zoom_minus);
	zoom_minus->set_tooltip(RTR("Zoom Out"));
	zoom_minus->connect("pressed", callable_mp(this, &GraphEdit::_zoom_minus));
	zoom_minus->set_focus_mode(FOCUS_NONE);

	zoom_reset = memnew(Button);
	zoom_reset->set_flat(true);
	zoom_hb->add_child(zoom_reset);
	zoom_reset->set_tooltip(RTR("Zoom Reset"));
	zoom_reset->connect("pressed", callable_mp(this, &GraphEdit::_zoom_reset));
	zoom_reset->set_focus_mode(FOCUS_NONE);

	zoom_plus = memnew(Button);
	zoom_plus->set_flat(true);
	zoom_hb->add_child(zoom_plus);
	zoom_plus->set_tooltip(RTR("Zoom In"));
	zoom_plus->connect("pressed", callable_mp(this, &GraphEdit::_zoom_plus));
	zoom_plus->set_focus_mode(FOCUS_NONE);

	snap_button = memnew(Button);
	snap_button->set_flat(true);
	snap_button->set_toggle_mode(true);
	snap_button->set_tooltip(RTR("Enable snap and show grid."));
	snap_button->connect("pressed", callable_mp(this, &GraphEdit::_snap_toggled));
	snap_button->set_pressed(true);
	snap_button->set_focus_mode(FOCUS_NONE);
	zoom_hb->add_child(snap_button);

	snap_amount = memnew(SpinBox);
	snap_amount->set_min(5);
	snap_amount->set_max(100);
	snap_amount->set_step(1);
	snap_amount->set_value(20);
	snap_amount->connect("value_changed", callable_mp(this, &GraphEdit::_snap_value_changed));
	zoom_hb->add_child(snap_amount);

	minimap_button = memnew(Button);
	minimap_button->set_flat(true);
	minimap_button->set_toggle_mode(true);
	minimap_button->set_tooltip(RTR("Enable grid minimap."));
	minimap_button->connect("pressed", callable_mp(this, &GraphEdit::_minimap_toggled));
	minimap_button->set_pressed(true);
	minimap_button->set_focus_mode(FOCUS_NONE);
	zoom_hb->add_child(minimap_button);

	layout_button = memnew(Button);
	layout_button->set_flat(true);
	zoom_hb->add_child(layout_button);
	layout_button->set_tooltip(RTR("Arrange nodes."));
	layout_button->connect("pressed", callable_mp(this, &GraphEdit::arrange_nodes));
	layout_button->set_focus_mode(FOCUS_NONE);

	Vector2 minimap_size = Vector2(240, 160);
	float minimap_opacity = 0.65;

	minimap = memnew(GraphEditMinimap(this));
	top_layer->add_child(minimap);
	minimap->set_name("_minimap");
	minimap->set_modulate(Color(1, 1, 1, minimap_opacity));
	minimap->set_mouse_filter(MOUSE_FILTER_PASS);
	minimap->set_custom_minimum_size(Vector2(50, 50));
	minimap->set_size(minimap_size);
	minimap->set_anchors_preset(Control::PRESET_BOTTOM_RIGHT);
	minimap->set_offset(Side::SIDE_LEFT, -minimap_size.x - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_TOP, -minimap_size.y - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_RIGHT, -MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_BOTTOM, -MINIMAP_OFFSET);
	minimap->connect("draw", callable_mp(this, &GraphEdit::_minimap_draw));

	set_clip_contents(true);
}
