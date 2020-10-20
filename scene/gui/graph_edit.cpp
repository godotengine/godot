/*************************************************************************/
/*  graph_edit.cpp                                                       */
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

#include "graph_edit.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

#define ZOOM_SCALE 1.2

#define MIN_ZOOM (((1 / ZOOM_SCALE) / ZOOM_SCALE) / ZOOM_SCALE)
#define MAX_ZOOM (1 * ZOOM_SCALE * ZOOM_SCALE * ZOOM_SCALE)

bool GraphEditFilter::has_point(const Point2 &p_point) const {
	return ge->_filter_input(p_point);
}

GraphEditFilter::GraphEditFilter(GraphEdit *p_edit) {
	ge = p_edit;
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
	update();
	connections_layer->update();

	return OK;
}

bool GraphEdit::is_node_connected(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		if (E->get().from == p_from && E->get().from_port == p_from_port && E->get().to == p_to && E->get().to_port == p_to_port) {
			return true;
		}
	}

	return false;
}

void GraphEdit::disconnect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		if (E->get().from == p_from && E->get().from_port == p_from_port && E->get().to == p_to && E->get().to_port == p_to_port) {
			connections.erase(E);
			top_layer->update();
			update();
			connections_layer->update();
			return;
		}
	}
}

bool GraphEdit::clips_input() const {
	return true;
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
		call_deferred("_update_scroll_offset");
		awaiting_scroll_offset_update = true;
	}
	top_layer->update();
	update();

	if (!setting_scroll_ofs) { //in godot, signals on change value are avoided as a convention
		emit_signal("scroll_offset_changed", get_scroll_ofs());
	}
}

void GraphEdit::_update_scroll_offset() {
	set_block_minimum_size_adjust(true);

	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		Point2 pos = gn->get_offset() * zoom;
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
		r.position = gn->get_offset() * zoom;
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
	h_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, v_scroll->is_visible() ? -vmin.width : 0);
	v_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, h_scroll->is_visible() ? -hmin.height : 0);

	set_block_minimum_size_adjust(false);

	if (!awaiting_scroll_offset_update) {
		call_deferred("_update_scroll_offset");
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
	int first_not_comment = 0;
	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
		if (gn2 && !gn2->is_comment()) {
			first_not_comment = i;
			break;
		}
	}

	move_child(connections_layer, first_not_comment);
	top_layer->raise();
	emit_signal("node_selected", p_gn);
}

void GraphEdit::_graph_node_moved(Node *p_gn) {
	GraphNode *gn = Object::cast_to<GraphNode>(p_gn);
	ERR_FAIL_COND(!gn);
	top_layer->update();
	update();
	connections_layer->update();
}

void GraphEdit::add_child_notify(Node *p_child) {
	Control::add_child_notify(p_child);

	top_layer->call_deferred("raise"); //top layer always on top!
	GraphNode *gn = Object::cast_to<GraphNode>(p_child);
	if (gn) {
		gn->set_scale(Vector2(zoom, zoom));
		gn->connect("offset_changed", callable_mp(this, &GraphEdit::_graph_node_moved), varray(gn));
		gn->connect("raise_request", callable_mp(this, &GraphEdit::_graph_node_raised), varray(gn));
		gn->connect("item_rect_changed", callable_mp((CanvasItem *)connections_layer, &CanvasItem::update));
		_graph_node_moved(gn);
		gn->set_mouse_filter(MOUSE_FILTER_PASS);
	}
}

void GraphEdit::remove_child_notify(Node *p_child) {
	Control::remove_child_notify(p_child);
	if (is_inside_tree()) {
		top_layer->call_deferred("raise"); //top layer always on top!
	}
	GraphNode *gn = Object::cast_to<GraphNode>(p_child);
	if (gn) {
		gn->disconnect("offset_changed", callable_mp(this, &GraphEdit::_graph_node_moved));
		gn->disconnect("raise_request", callable_mp(this, &GraphEdit::_graph_node_raised));
		gn->disconnect("item_rect_changed", callable_mp((CanvasItem *)connections_layer, &CanvasItem::update));
	}
}

void GraphEdit::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		port_grab_distance_horizontal = get_theme_constant("port_grab_distance_horizontal");
		port_grab_distance_vertical = get_theme_constant("port_grab_distance_vertical");

		zoom_minus->set_icon(get_theme_icon("minus"));
		zoom_reset->set_icon(get_theme_icon("reset"));
		zoom_plus->set_icon(get_theme_icon("more"));
		snap_button->set_icon(get_theme_icon("snap"));
	}
	if (p_what == NOTIFICATION_READY) {
		Size2 hmin = h_scroll->get_combined_minimum_size();
		Size2 vmin = v_scroll->get_combined_minimum_size();

		h_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 0);
		h_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
		h_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_END, -hmin.height);
		h_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

		v_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, -vmin.width);
		v_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
		v_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 0);
		v_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);
	}
	if (p_what == NOTIFICATION_DRAW) {
		draw_style_box(get_theme_stylebox("bg"), Rect2(Point2(), get_size()));

		if (is_using_snap()) {
			//draw grid

			int snap = get_snap();

			Vector2 offset = get_scroll_ofs() / zoom;
			Size2 size = get_size() / zoom;

			Point2i from = (offset / float(snap)).floor();
			Point2i len = (size / float(snap)).floor() + Vector2(1, 1);

			Color grid_minor = get_theme_color("grid_minor");
			Color grid_major = get_theme_color("grid_major");

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
	}
}

bool GraphEdit::_filter_input(const Point2 &p_point) {
	Ref<Texture2D> port = get_theme_icon("port", "GraphNode");

	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
		if (!gn) {
			continue;
		}

		for (int j = 0; j < gn->get_connection_output_count(); j++) {
			Vector2 pos = gn->get_connection_output_position(j) + gn->get_position();
			if (is_in_hot_zone(pos, p_point)) {
				return true;
			}
		}

		for (int j = 0; j < gn->get_connection_input_count(); j++) {
			Vector2 pos = gn->get_connection_input_position(j) + gn->get_position();
			if (is_in_hot_zone(pos, p_point)) {
				return true;
			}
		}
	}

	return false;
}

void GraphEdit::_top_layer_input(const Ref<InputEvent> &p_ev) {
	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT && mb->is_pressed()) {
		connecting_valid = false;
		Ref<Texture2D> port = get_theme_icon("port", "GraphNode");
		click_pos = mb->get_position();
		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
			if (!gn) {
				continue;
			}

			for (int j = 0; j < gn->get_connection_output_count(); j++) {
				Vector2 pos = gn->get_connection_output_position(j) + gn->get_position();
				if (is_in_hot_zone(pos, click_pos)) {
					if (valid_left_disconnect_types.has(gn->get_connection_output_type(j))) {
						//check disconnect
						for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
							if (E->get().from == gn->get_name() && E->get().from_port == j) {
								Node *to = get_node(String(E->get().to));
								if (Object::cast_to<GraphNode>(to)) {
									connecting_from = E->get().to;
									connecting_index = E->get().to_port;
									connecting_out = false;
									connecting_type = Object::cast_to<GraphNode>(to)->get_connection_input_type(E->get().to_port);
									connecting_color = Object::cast_to<GraphNode>(to)->get_connection_input_color(E->get().to_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconnected = true;

									emit_signal("disconnection_request", E->get().from, E->get().from_port, E->get().to, E->get().to_port);
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
				if (is_in_hot_zone(pos, click_pos)) {
					if (right_disconnects || valid_right_disconnect_types.has(gn->get_connection_input_type(j))) {
						//check disconnect
						for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
							if (E->get().to == gn->get_name() && E->get().to_port == j) {
								Node *fr = get_node(String(E->get().from));
								if (Object::cast_to<GraphNode>(fr)) {
									connecting_from = E->get().from;
									connecting_index = E->get().from_port;
									connecting_out = true;
									connecting_type = Object::cast_to<GraphNode>(fr)->get_connection_output_type(E->get().from_port);
									connecting_color = Object::cast_to<GraphNode>(fr)->get_connection_output_color(E->get().from_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconnected = true;

									emit_signal("disconnection_request", E->get().from, E->get().from_port, E->get().to, E->get().to_port);
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
		connecting_valid = just_disconnected || click_pos.distance_to(connecting_to) > 20.0 * zoom;

		if (connecting_valid) {
			Ref<Texture2D> port = get_theme_icon("port", "GraphNode");
			Vector2 mpos = mm->get_position();
			for (int i = get_child_count() - 1; i >= 0; i--) {
				GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
				if (!gn) {
					continue;
				}

				if (!connecting_out) {
					for (int j = 0; j < gn->get_connection_output_count(); j++) {
						Vector2 pos = gn->get_connection_output_position(j) + gn->get_position();
						int type = gn->get_connection_output_type(j);
						if ((type == connecting_type || valid_connection_types.has(ConnType(type, connecting_type))) && is_in_hot_zone(pos, mpos)) {
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
						if ((type == connecting_type || valid_connection_types.has(ConnType(type, connecting_type))) && is_in_hot_zone(pos, mpos)) {
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

	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT && !mb->is_pressed()) {
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
				emit_signal("connection_request", from, from_slot, to, to_slot);

			} else if (!just_disconnected) {
				String from = connecting_from;
				int from_slot = connecting_index;
				Vector2 ofs = Vector2(mb->get_position().x, mb->get_position().y);

				if (!connecting_out) {
					emit_signal("connection_from_empty", from, from_slot, ofs);
				} else {
					emit_signal("connection_to_empty", from, from_slot, ofs);
				}
			}
		}

		connecting = false;
		top_layer->update();
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

bool GraphEdit::is_in_hot_zone(const Vector2 &pos, const Vector2 &p_mouse_pos) {
	if (!Rect2(pos.x - port_grab_distance_horizontal, pos.y - port_grab_distance_vertical, port_grab_distance_horizontal * 2, port_grab_distance_vertical * 2).has_point(p_mouse_pos)) {
		return false;
	}

	for (int i = 0; i < get_child_count(); i++) {
		Control *child = Object::cast_to<Control>(get_child(i));
		if (!child) {
			continue;
		}
		Rect2 rect = child->get_rect();
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

template <class Vector2>
static _FORCE_INLINE_ Vector2 _bezier_interp(real_t t, Vector2 start, Vector2 control_1, Vector2 control_2, Vector2 end) {
	/* Formula from Wikipedia article on Bezier curves. */
	real_t omt = (1.0 - t);
	real_t omt2 = omt * omt;
	real_t omt3 = omt2 * omt;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
}

void GraphEdit::_bake_segment2d(Vector<Vector2> &points, Vector<Color> &colors, float p_begin, float p_end, const Vector2 &p_a, const Vector2 &p_out, const Vector2 &p_b, const Vector2 &p_in, int p_depth, int p_min_depth, int p_max_depth, float p_tol, const Color &p_color, const Color &p_to_color, int &lines) const {
	float mp = p_begin + (p_end - p_begin) * 0.5;
	Vector2 beg = _bezier_interp(p_begin, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 mid = _bezier_interp(mp, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 end = _bezier_interp(p_end, p_a, p_a + p_out, p_b + p_in, p_b);

	Vector2 na = (mid - beg).normalized();
	Vector2 nb = (end - mid).normalized();
	float dp = Math::rad2deg(Math::acos(na.dot(nb)));

	if (p_depth >= p_min_depth && (dp < p_tol || p_depth >= p_max_depth)) {
		points.push_back((beg + end) * 0.5);
		colors.push_back(p_color.lerp(p_to_color, mp));
		lines++;
	} else {
		_bake_segment2d(points, colors, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_min_depth, p_max_depth, p_tol, p_color, p_to_color, lines);
		_bake_segment2d(points, colors, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_min_depth, p_max_depth, p_tol, p_color, p_to_color, lines);
	}
}

void GraphEdit::_draw_cos_line(CanvasItem *p_where, const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, const Color &p_to_color) {
	//cubic bezier code
	float diff = p_to.x - p_from.x;
	float cp_offset;
	int cp_len = get_theme_constant("bezier_len_pos");
	int cp_neg_len = get_theme_constant("bezier_len_neg");

	if (diff > 0) {
		cp_offset = MIN(cp_len, diff * 0.5);
	} else {
		cp_offset = MAX(MIN(cp_len - diff, cp_neg_len), -diff * 0.5);
	}

	Vector2 c1 = Vector2(cp_offset * zoom, 0);
	Vector2 c2 = Vector2(-cp_offset * zoom, 0);

	int lines = 0;

	Vector<Point2> points;
	Vector<Color> colors;
	points.push_back(p_from);
	colors.push_back(p_color);
	_bake_segment2d(points, colors, 0, 1, p_from, c1, p_to, c2, 0, 3, 9, 3, p_color, p_to_color, lines);
	points.push_back(p_to);
	colors.push_back(p_to_color);

#ifdef TOOLS_ENABLED
	p_where->draw_polyline_colors(points, colors, Math::floor(2 * EDSCALE));
#else
	p_where->draw_polyline_colors(points, colors, 2);
#endif
}

void GraphEdit::_connections_layer_draw() {
	Color activity_color = get_theme_color("activity");
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

		Vector2 frompos = gfrom->get_connection_output_position(E->get().from_port) + gfrom->get_offset() * zoom;
		Color color = gfrom->get_connection_output_color(E->get().from_port);
		Vector2 topos = gto->get_connection_input_position(E->get().to_port) + gto->get_offset() * zoom;
		Color tocolor = gto->get_connection_input_color(E->get().to_port);

		if (E->get().activity > 0) {
			color = color.lerp(activity_color, E->get().activity);
			tocolor = tocolor.lerp(activity_color, E->get().activity);
		}
		_draw_cos_line(connections_layer, frompos, topos, color, tocolor);
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
		_draw_cos_line(top_layer, pos, topos, col, col);
	}

	if (box_selecting) {
		top_layer->draw_rect(box_selecting_rect, get_theme_color("selection_fill"));
		top_layer->draw_rect(box_selecting_rect, get_theme_color("selection_stroke"), false);
	}
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

void GraphEdit::_gui_input(const Ref<InputEvent> &p_ev) {
	Ref<InputEventMouseMotion> mm = p_ev;
	if (mm.is_valid() && (mm->get_button_mask() & BUTTON_MASK_MIDDLE || (mm->get_button_mask() & BUTTON_MASK_LEFT && Input::get_singleton()->is_key_pressed(KEY_SPACE)))) {
		h_scroll->set_value(h_scroll->get_value() - mm->get_relative().x);
		v_scroll->set_value(v_scroll->get_value() - mm->get_relative().y);
	}

	if (mm.is_valid() && dragging) {
		if (!moving_selection) {
			emit_signal("begin_node_move");
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
				if (is_using_snap() ^ Input::get_singleton()->is_key_pressed(KEY_CONTROL)) {
					const int snap = get_snap();
					pos = pos.snapped(Vector2(snap, snap));
				}

				gn->set_offset(pos);
			}
		}
	}

	if (mm.is_valid() && box_selecting) {
		box_selecting_to = mm->get_position();

		box_selecting_rect = Rect2(MIN(box_selecting_from.x, box_selecting_to.x),
				MIN(box_selecting_from.y, box_selecting_to.y),
				ABS(box_selecting_from.x - box_selecting_to.x),
				ABS(box_selecting_from.y - box_selecting_to.y));

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
					emit_signal("node_selected", gn);
				} else if (gn->is_selected() && !box_selection_mode_additive) {
					emit_signal("node_unselected", gn);
				}
				gn->set_selected(box_selection_mode_additive);
			} else {
				bool select = (previus_selected.find(gn) != nullptr);
				if (gn->is_selected() && !select) {
					emit_signal("node_unselected", gn);
				} else if (!gn->is_selected() && select) {
					emit_signal("node_selected", gn);
				}
				gn->set_selected(select);
			}
		}

		top_layer->update();
	}

	Ref<InputEventMouseButton> b = p_ev;
	if (b.is_valid()) {
		if (b->get_button_index() == BUTTON_RIGHT && b->is_pressed()) {
			if (box_selecting) {
				box_selecting = false;
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
					if (!gn) {
						continue;
					}

					bool select = (previus_selected.find(gn) != nullptr);
					if (gn->is_selected() && !select) {
						emit_signal("node_unselected", gn);
					} else if (!gn->is_selected() && select) {
						emit_signal("node_selected", gn);
					}
					gn->set_selected(select);
				}
				top_layer->update();
			} else {
				if (connecting) {
					connecting = false;
					top_layer->update();
				} else {
					emit_signal("popup_request", b->get_global_position());
				}
			}
		}

		if (b->get_button_index() == BUTTON_LEFT && !b->is_pressed() && dragging) {
			if (!just_selected && drag_accum == Vector2() && Input::get_singleton()->is_key_pressed(KEY_CONTROL)) {
				//deselect current node
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));

					if (gn) {
						Rect2 r = gn->get_rect();
						r.size *= zoom;
						if (r.has_point(b->get_position())) {
							emit_signal("node_unselected", gn);
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
					}
				}
			}

			if (moving_selection) {
				emit_signal("end_node_move");
				moving_selection = false;
			}

			dragging = false;

			top_layer->update();
			update();
			connections_layer->update();
		}

		if (b->get_button_index() == BUTTON_LEFT && b->is_pressed()) {
			GraphNode *gn = nullptr;

			for (int i = get_child_count() - 1; i >= 0; i--) {
				GraphNode *gn_selected = Object::cast_to<GraphNode>(get_child(i));

				if (gn_selected) {
					if (gn_selected->is_resizing()) {
						continue;
					}

					if (gn_selected->has_point(b->get_position() - gn_selected->get_position())) {
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
				if (!gn->is_selected() && !Input::get_singleton()->is_key_pressed(KEY_CONTROL)) {
					for (int i = 0; i < get_child_count(); i++) {
						GraphNode *o_gn = Object::cast_to<GraphNode>(get_child(i));
						if (o_gn) {
							if (o_gn == gn) {
								o_gn->set_selected(true);
							} else {
								if (o_gn->is_selected()) {
									emit_signal("node_unselected", o_gn);
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
					}
				}

			} else {
				if (_filter_input(b->get_position())) {
					return;
				}
				if (Input::get_singleton()->is_key_pressed(KEY_SPACE)) {
					return;
				}

				box_selecting = true;
				box_selecting_from = b->get_position();
				if (b->get_control()) {
					box_selection_mode_additive = true;
					previus_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
						if (!gn2 || !gn2->is_selected()) {
							continue;
						}

						previus_selected.push_back(gn2);
					}
				} else if (b->get_shift()) {
					box_selection_mode_additive = false;
					previus_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
						if (!gn2 || !gn2->is_selected()) {
							continue;
						}

						previus_selected.push_back(gn2);
					}
				} else {
					box_selection_mode_additive = true;
					previus_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphNode *gn2 = Object::cast_to<GraphNode>(get_child(i));
						if (!gn2) {
							continue;
						}
						if (gn2->is_selected()) {
							emit_signal("node_unselected", gn2);
						}
						gn2->set_selected(false);
					}
				}
			}
		}

		if (b->get_button_index() == BUTTON_LEFT && !b->is_pressed() && box_selecting) {
			box_selecting = false;
			previus_selected.clear();
			top_layer->update();
		}

		if (b->get_button_index() == BUTTON_WHEEL_UP && b->is_pressed()) {
			//too difficult to get right
			//set_zoom(zoom*ZOOM_SCALE);
		}

		if (b->get_button_index() == BUTTON_WHEEL_DOWN && b->is_pressed()) {
			//too difficult to get right
			//set_zoom(zoom/ZOOM_SCALE);
		}
		if (b->get_button_index() == BUTTON_WHEEL_UP && !Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
			v_scroll->set_value(v_scroll->get_value() - v_scroll->get_page() * b->get_factor() / 8);
		}
		if (b->get_button_index() == BUTTON_WHEEL_DOWN && !Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
			v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * b->get_factor() / 8);
		}
		if (b->get_button_index() == BUTTON_WHEEL_RIGHT || (b->get_button_index() == BUTTON_WHEEL_DOWN && Input::get_singleton()->is_key_pressed(KEY_SHIFT))) {
			h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * b->get_factor() / 8);
		}
		if (b->get_button_index() == BUTTON_WHEEL_LEFT || (b->get_button_index() == BUTTON_WHEEL_UP && Input::get_singleton()->is_key_pressed(KEY_SHIFT))) {
			h_scroll->set_value(h_scroll->get_value() - h_scroll->get_page() * b->get_factor() / 8);
		}
	}

	Ref<InputEventKey> k = p_ev;

	if (k.is_valid()) {
		if (k->get_keycode() == KEY_D && k->is_pressed() && k->get_command()) {
			emit_signal("duplicate_nodes_request");
			accept_event();
		}

		if (k->get_keycode() == KEY_C && k->is_pressed() && k->get_command()) {
			emit_signal("copy_nodes_request");
			accept_event();
		}

		if (k->get_keycode() == KEY_V && k->is_pressed() && k->get_command()) {
			emit_signal("paste_nodes_request");
			accept_event();
		}

		if (k->get_keycode() == KEY_DELETE && k->is_pressed()) {
			emit_signal("delete_nodes_request");
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
	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		if (E->get().from == p_from && E->get().from_port == p_from_port && E->get().to == p_to && E->get().to_port == p_to_port) {
			if (Math::is_equal_approx(E->get().activity, p_activity)) {
				//update only if changed
				top_layer->update();
				connections_layer->update();
			}
			E->get().activity = p_activity;
			return;
		}
	}
}

void GraphEdit::clear_connections() {
	connections.clear();
	update();
	connections_layer->update();
}

void GraphEdit::set_zoom(float p_zoom) {
	set_zoom_custom(p_zoom, get_size() / 2);
}

void GraphEdit::set_zoom_custom(float p_zoom, const Vector2 &p_center) {
	p_zoom = CLAMP(p_zoom, MIN_ZOOM, MAX_ZOOM);
	if (zoom == p_zoom) {
		return;
	}

	zoom_minus->set_disabled(zoom == MIN_ZOOM);
	zoom_plus->set_disabled(zoom == MAX_ZOOM);

	Vector2 sbofs = (Vector2(h_scroll->get_value(), v_scroll->get_value()) + p_center) / zoom;

	zoom = p_zoom;
	top_layer->update();

	_update_scroll();
	connections_layer->update();

	if (is_visible_in_tree()) {
		Vector2 ofs = sbofs * zoom - p_center;
		h_scroll->set_value(ofs.x);
		v_scroll->set_value(ofs.y);
	}

	update();
}

float GraphEdit::get_zoom() const {
	return zoom;
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
	for (List<Connection>::Element *E = conns.front(); E; E = E->next()) {
		Dictionary d;
		d["from"] = E->get().from;
		d["from_port"] = E->get().from_port;
		d["to"] = E->get().to;
		d["to_port"] = E->get().to_port;
		arr.push_back(d);
	}
	return arr;
}

void GraphEdit::_zoom_minus() {
	set_zoom(zoom / ZOOM_SCALE);
}

void GraphEdit::_zoom_reset() {
	set_zoom(1);
}

void GraphEdit::_zoom_plus() {
	set_zoom(zoom * ZOOM_SCALE);
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

HBoxContainer *GraphEdit::get_zoom_hbox() {
	return zoom_hb;
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

	ClassDB::bind_method(D_METHOD("set_zoom", "p_zoom"), &GraphEdit::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &GraphEdit::get_zoom);

	ClassDB::bind_method(D_METHOD("set_snap", "pixels"), &GraphEdit::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &GraphEdit::get_snap);

	ClassDB::bind_method(D_METHOD("set_use_snap", "enable"), &GraphEdit::set_use_snap);
	ClassDB::bind_method(D_METHOD("is_using_snap"), &GraphEdit::is_using_snap);

	ClassDB::bind_method(D_METHOD("set_right_disconnects", "enable"), &GraphEdit::set_right_disconnects);
	ClassDB::bind_method(D_METHOD("is_right_disconnects_enabled"), &GraphEdit::is_right_disconnects_enabled);

	ClassDB::bind_method(D_METHOD("_gui_input"), &GraphEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("_update_scroll_offset"), &GraphEdit::_update_scroll_offset);

	ClassDB::bind_method(D_METHOD("get_zoom_hbox"), &GraphEdit::get_zoom_hbox);

	ClassDB::bind_method(D_METHOD("set_selected", "node"), &GraphEdit::set_selected);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "right_disconnects"), "set_right_disconnects", "is_right_disconnects_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scroll_offset"), "set_scroll_ofs", "get_scroll_ofs");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "snap_distance"), "set_snap", "get_snap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_snap"), "set_use_snap", "is_using_snap");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom"), "set_zoom", "get_zoom");

	ADD_SIGNAL(MethodInfo("connection_request", PropertyInfo(Variant::STRING_NAME, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::STRING_NAME, "to"), PropertyInfo(Variant::INT, "to_slot")));
	ADD_SIGNAL(MethodInfo("disconnection_request", PropertyInfo(Variant::STRING_NAME, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::STRING_NAME, "to"), PropertyInfo(Variant::INT, "to_slot")));
	ADD_SIGNAL(MethodInfo("popup_request", PropertyInfo(Variant::VECTOR2, "position")));
	ADD_SIGNAL(MethodInfo("duplicate_nodes_request"));
	ADD_SIGNAL(MethodInfo("copy_nodes_request"));
	ADD_SIGNAL(MethodInfo("paste_nodes_request"));
	ADD_SIGNAL(MethodInfo("node_selected", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_unselected", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("connection_to_empty", PropertyInfo(Variant::STRING_NAME, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::VECTOR2, "release_position")));
	ADD_SIGNAL(MethodInfo("connection_from_empty", PropertyInfo(Variant::STRING_NAME, "to"), PropertyInfo(Variant::INT, "to_slot"), PropertyInfo(Variant::VECTOR2, "release_position")));
	ADD_SIGNAL(MethodInfo("delete_nodes_request"));
	ADD_SIGNAL(MethodInfo("begin_node_move"));
	ADD_SIGNAL(MethodInfo("end_node_move"));
	ADD_SIGNAL(MethodInfo("scroll_offset_changed", PropertyInfo(Variant::VECTOR2, "ofs")));
}

GraphEdit::GraphEdit() {
	set_focus_mode(FOCUS_ALL);

	awaiting_scroll_offset_update = false;
	top_layer = nullptr;
	top_layer = memnew(GraphEditFilter(this));
	add_child(top_layer);
	top_layer->set_mouse_filter(MOUSE_FILTER_PASS);
	top_layer->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	top_layer->connect("draw", callable_mp(this, &GraphEdit::_top_layer_draw));
	top_layer->connect("gui_input", callable_mp(this, &GraphEdit::_top_layer_input));

	connections_layer = memnew(Control);
	add_child(connections_layer);
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

	updating = false;
	connecting = false;
	right_disconnects = false;

	box_selecting = false;
	dragging = false;

	//set large minmax so it can scroll even if not resized yet
	h_scroll->set_min(-10000);
	h_scroll->set_max(10000);

	v_scroll->set_min(-10000);
	v_scroll->set_max(10000);

	h_scroll->connect("value_changed", callable_mp(this, &GraphEdit::_scroll_moved));
	v_scroll->connect("value_changed", callable_mp(this, &GraphEdit::_scroll_moved));

	zoom = 1;

	zoom_hb = memnew(HBoxContainer);
	top_layer->add_child(zoom_hb);
	zoom_hb->set_position(Vector2(10, 10));

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

	setting_scroll_ofs = false;
	just_disconnected = false;
	set_clip_contents(true);
}
