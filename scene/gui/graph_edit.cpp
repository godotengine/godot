/*************************************************************************/
/*  graph_edit.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "graph_edit.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "scene/gui/box_container.h"

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

	if (is_node_connected(p_from, p_from_port, p_to, p_to_port))
		return OK;
	Connection c;
	c.from = p_from;
	c.from_port = p_from_port;
	c.to = p_to;
	c.to_port = p_to_port;
	connections.push_back(c);
	top_layer->update();
	update();
	connections_layer->update();

	return OK;
}

bool GraphEdit::is_node_connected(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {

	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {

		if (E->get().from == p_from && E->get().from_port == p_from_port && E->get().to == p_to && E->get().to_port == p_to_port)
			return true;
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

		GraphNode *gn = get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		Point2 pos = gn->get_offset() * zoom;
		pos -= Point2(h_scroll->get_value(), v_scroll->get_value());
		gn->set_pos(pos);
		if (gn->get_scale() != Vector2(zoom, zoom)) {
			gn->set_scale(Vector2(zoom, zoom));
		}
	}

	connections_layer->set_pos(-Point2(h_scroll->get_value(), v_scroll->get_value()));
	set_block_minimum_size_adjust(false);
	awaiting_scroll_offset_update = false;
}

void GraphEdit::_update_scroll() {

	if (updating)
		return;

	updating = true;

	set_block_minimum_size_adjust(true);

	Rect2 screen;
	for (int i = 0; i < get_child_count(); i++) {

		GraphNode *gn = get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		Rect2 r;
		r.pos = gn->get_offset() * zoom;
		r.size = gn->get_size() * zoom;
		screen = screen.merge(r);
	}

	screen.pos -= get_size();
	screen.size += get_size() * 2.0;

	h_scroll->set_min(screen.pos.x);
	h_scroll->set_max(screen.pos.x + screen.size.x);
	h_scroll->set_page(get_size().x);
	if (h_scroll->get_max() - h_scroll->get_min() <= h_scroll->get_page())
		h_scroll->hide();
	else
		h_scroll->show();

	v_scroll->set_min(screen.pos.y);
	v_scroll->set_max(screen.pos.y + screen.size.y);
	v_scroll->set_page(get_size().y);

	if (v_scroll->get_max() - v_scroll->get_min() <= v_scroll->get_page())
		v_scroll->hide();
	else
		v_scroll->show();

	set_block_minimum_size_adjust(false);

	if (!awaiting_scroll_offset_update) {
		call_deferred("_update_scroll_offset");
		awaiting_scroll_offset_update = true;
	}

	updating = false;
}

void GraphEdit::_graph_node_raised(Node *p_gn) {

	GraphNode *gn = p_gn->cast_to<GraphNode>();
	ERR_FAIL_COND(!gn);
	gn->raise();
	if (gn->is_comment()) {
		move_child(gn, 0);
	}
	int first_not_comment = 0;
	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *gn = get_child(i)->cast_to<GraphNode>();
		if (gn && !gn->is_comment()) {
			first_not_comment = i;
			break;
		}
	}

	move_child(connections_layer, first_not_comment);
	top_layer->raise();
	emit_signal("node_selected", p_gn);
}

void GraphEdit::_graph_node_moved(Node *p_gn) {

	GraphNode *gn = p_gn->cast_to<GraphNode>();
	ERR_FAIL_COND(!gn);
	top_layer->update();
	update();
	connections_layer->update();
}

void GraphEdit::add_child_notify(Node *p_child) {

	Control::add_child_notify(p_child);

	top_layer->call_deferred("raise"); //top layer always on top!
	GraphNode *gn = p_child->cast_to<GraphNode>();
	if (gn) {
		gn->set_scale(Vector2(zoom, zoom));
		gn->connect("offset_changed", this, "_graph_node_moved", varray(gn));
		gn->connect("raise_request", this, "_graph_node_raised", varray(gn));
		gn->connect("item_rect_changed", connections_layer, "update");
		_graph_node_moved(gn);
		gn->set_mouse_filter(MOUSE_FILTER_PASS);
	}
}

void GraphEdit::remove_child_notify(Node *p_child) {

	Control::remove_child_notify(p_child);

	top_layer->call_deferred("raise"); //top layer always on top!
	GraphNode *gn = p_child->cast_to<GraphNode>();
	if (gn) {
		gn->disconnect("offset_changed", this, "_graph_node_moved");
		gn->disconnect("raise_request", this, "_graph_node_raised");
	}
}

void GraphEdit::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {
		Size2 hmin = h_scroll->get_combined_minimum_size();
		Size2 vmin = v_scroll->get_combined_minimum_size();

		v_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, vmin.width);
		v_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
		v_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 0);
		v_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

		h_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 0);
		h_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
		h_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_END, hmin.height);
		h_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

		zoom_minus->set_icon(get_icon("minus"));
		zoom_reset->set_icon(get_icon("reset"));
		zoom_plus->set_icon(get_icon("more"));
		snap_button->set_icon(get_icon("snap"));
		//zoom_icon->set_texture( get_icon("Zoom", "EditorIcons"));
	}
	if (p_what == NOTIFICATION_DRAW) {

		draw_style_box(get_stylebox("bg"), Rect2(Point2(), get_size()));

		if (is_using_snap()) {
			//draw grid

			int snap = get_snap();

			Vector2 offset = get_scroll_ofs() / zoom;
			Size2 size = get_size() / zoom;

			Point2i from = (offset / float(snap)).floor();
			Point2i len = (size / float(snap)).floor() + Vector2(1, 1);

			Color grid_minor = get_color("grid_minor");
			Color grid_major = get_color("grid_major");

			for (int i = from.x; i < from.x + len.x; i++) {

				Color color;

				if (ABS(i) % 10 == 0)
					color = grid_major;
				else
					color = grid_minor;

				float base_ofs = i * snap * zoom - offset.x * zoom;
				draw_line(Vector2(base_ofs, 0), Vector2(base_ofs, get_size().height), color);
			}

			for (int i = from.y; i < from.y + len.y; i++) {

				Color color;

				if (ABS(i) % 10 == 0)
					color = grid_major;
				else
					color = grid_minor;

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

	Ref<Texture> port = get_icon("port", "GraphNode");

	float grab_r_extend = 2.0;
	float grab_r = port->get_width() * 0.5 * grab_r_extend;
	for (int i = get_child_count() - 1; i >= 0; i--) {

		GraphNode *gn = get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		for (int j = 0; j < gn->get_connection_output_count(); j++) {

			Vector2 pos = gn->get_connection_output_pos(j) + gn->get_pos();
			if (pos.distance_to(p_point) < grab_r)
				return true;
		}

		for (int j = 0; j < gn->get_connection_input_count(); j++) {

			Vector2 pos = gn->get_connection_input_pos(j) + gn->get_pos();
			if (pos.distance_to(p_point) < grab_r) {
				return true;
			}
		}
	}

	return false;
}

void GraphEdit::_top_layer_input(const InputEvent &p_ev) {

	float grab_r_extend = 2.0;
	if (p_ev.type == InputEvent::MOUSE_BUTTON && p_ev.mouse_button.button_index == BUTTON_LEFT && p_ev.mouse_button.pressed) {

		Ref<Texture> port = get_icon("port", "GraphNode");
		Vector2 mpos(p_ev.mouse_button.x, p_ev.mouse_button.y);
		float grab_r = port->get_width() * 0.5 * grab_r_extend;
		for (int i = get_child_count() - 1; i >= 0; i--) {

			GraphNode *gn = get_child(i)->cast_to<GraphNode>();
			if (!gn)
				continue;

			for (int j = 0; j < gn->get_connection_output_count(); j++) {

				Vector2 pos = gn->get_connection_output_pos(j) + gn->get_pos();
				if (pos.distance_to(mpos) < grab_r) {

					if (valid_left_disconnect_types.has(gn->get_connection_output_type(j))) {
						//check disconnect
						for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {

							if (E->get().from == gn->get_name() && E->get().from_port == j) {

								Node *to = get_node(String(E->get().to));
								if (to && to->cast_to<GraphNode>()) {

									connecting_from = E->get().to;
									connecting_index = E->get().to_port;
									connecting_out = false;
									connecting_type = to->cast_to<GraphNode>()->get_connection_input_type(E->get().to_port);
									connecting_color = to->cast_to<GraphNode>()->get_connection_input_color(E->get().to_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconected = true;

									emit_signal("disconnection_request", E->get().from, E->get().from_port, E->get().to, E->get().to_port);
									to = get_node(String(connecting_from)); //maybe it was erased
									if (to && to->cast_to<GraphNode>()) {
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
					just_disconected = false;
					return;
				}
			}

			for (int j = 0; j < gn->get_connection_input_count(); j++) {

				Vector2 pos = gn->get_connection_input_pos(j) + gn->get_pos();

				if (pos.distance_to(mpos) < grab_r) {

					if (right_disconnects || valid_right_disconnect_types.has(gn->get_connection_input_type(j))) {
						//check disconnect
						for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {

							if (E->get().to == gn->get_name() && E->get().to_port == j) {

								Node *fr = get_node(String(E->get().from));
								if (fr && fr->cast_to<GraphNode>()) {

									connecting_from = E->get().from;
									connecting_index = E->get().from_port;
									connecting_out = true;
									connecting_type = fr->cast_to<GraphNode>()->get_connection_output_type(E->get().from_port);
									connecting_color = fr->cast_to<GraphNode>()->get_connection_output_color(E->get().from_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconected = true;

									emit_signal("disconnection_request", E->get().from, E->get().from_port, E->get().to, E->get().to_port);
									fr = get_node(String(connecting_from)); //maybe it was erased
									if (fr && fr->cast_to<GraphNode>()) {
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
					just_disconected = true;

					return;
				}
			}
		}
	}

	if (p_ev.type == InputEvent::MOUSE_MOTION && connecting) {

		connecting_to = Vector2(p_ev.mouse_motion.x, p_ev.mouse_motion.y);
		connecting_target = false;
		top_layer->update();

		Ref<Texture> port = get_icon("port", "GraphNode");
		Vector2 mpos(p_ev.mouse_button.x, p_ev.mouse_button.y);
		float grab_r = port->get_width() * 0.5 * grab_r_extend;
		for (int i = get_child_count() - 1; i >= 0; i--) {

			GraphNode *gn = get_child(i)->cast_to<GraphNode>();
			if (!gn)
				continue;

			if (!connecting_out) {
				for (int j = 0; j < gn->get_connection_output_count(); j++) {

					Vector2 pos = gn->get_connection_output_pos(j) + gn->get_pos();
					int type = gn->get_connection_output_type(j);
					if ((type == connecting_type || valid_connection_types.has(ConnType(type, connecting_type))) && pos.distance_to(mpos) < grab_r) {

						connecting_target = true;
						connecting_to = pos;
						connecting_target_to = gn->get_name();
						connecting_target_index = j;
						return;
					}
				}
			} else {

				for (int j = 0; j < gn->get_connection_input_count(); j++) {

					Vector2 pos = gn->get_connection_input_pos(j) + gn->get_pos();
					int type = gn->get_connection_input_type(j);
					if ((type == connecting_type || valid_connection_types.has(ConnType(type, connecting_type))) && pos.distance_to(mpos) < grab_r) {
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

	if (p_ev.type == InputEvent::MOUSE_BUTTON && p_ev.mouse_button.button_index == BUTTON_LEFT && !p_ev.mouse_button.pressed) {

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

		} else if (!just_disconected) {
			String from = connecting_from;
			int from_slot = connecting_index;
			Vector2 ofs = Vector2(p_ev.mouse_button.x, p_ev.mouse_button.y);
			emit_signal("connection_to_empty", from, from_slot, ofs);
		}
		connecting = false;
		top_layer->update();
		update();
		connections_layer->update();
	}
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

void GraphEdit::_bake_segment2d(CanvasItem *p_where, float p_begin, float p_end, const Vector2 &p_a, const Vector2 &p_out, const Vector2 &p_b, const Vector2 &p_in, int p_depth, int p_min_depth, int p_max_depth, float p_tol, const Color &p_color, const Color &p_to_color, int &lines) const {

	float mp = p_begin + (p_end - p_begin) * 0.5;
	Vector2 beg = _bezier_interp(p_begin, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 mid = _bezier_interp(mp, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 end = _bezier_interp(p_end, p_a, p_a + p_out, p_b + p_in, p_b);

	Vector2 na = (mid - beg).normalized();
	Vector2 nb = (end - mid).normalized();
	float dp = Math::rad2deg(Math::acos(na.dot(nb)));

	if (p_depth >= p_min_depth && (dp < p_tol || p_depth >= p_max_depth)) {

		p_where->draw_line(beg, end, p_color.linear_interpolate(p_to_color, mp), 2, true);
		lines++;
	} else {
		_bake_segment2d(p_where, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_min_depth, p_max_depth, p_tol, p_color, p_to_color, lines);
		_bake_segment2d(p_where, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_min_depth, p_max_depth, p_tol, p_color, p_to_color, lines);
	}
}

void GraphEdit::_draw_cos_line(CanvasItem *p_where, const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, const Color &p_to_color) {

#if 1

	//cubic bezier code
	float diff = p_to.x - p_from.x;
	float cp_offset;
	int cp_len = get_constant("bezier_len_pos");
	int cp_neg_len = get_constant("bezier_len_neg");

	if (diff > 0) {
		cp_offset = MAX(cp_len, diff * 0.5);
	} else {
		cp_offset = MAX(MIN(cp_len - diff, cp_neg_len), -diff * 0.5);
	}

	Vector2 c1 = Vector2(cp_offset * zoom, 0);
	Vector2 c2 = Vector2(-cp_offset * zoom, 0);

	int lines = 0;
	_bake_segment2d(p_where, 0, 1, p_from, c1, p_to, c2, 0, 3, 9, 8, p_color, p_to_color, lines);

#else

	static const int steps = 20;

	//old cosine code
	Rect2 r;
	r.pos = p_from;
	r.expand_to(p_to);
	Vector2 sign = Vector2((p_from.x < p_to.x) ? 1 : -1, (p_from.y < p_to.y) ? 1 : -1);
	bool flip = sign.x * sign.y < 0;

	Vector2 prev;
	for (int i = 0; i <= steps; i++) {

		float d = i / float(steps);
		float c = -Math::cos(d * Math_PI) * 0.5 + 0.5;
		if (flip)
			c = 1.0 - c;
		Vector2 p = r.pos + Vector2(d * r.size.width, c * r.size.height);

		if (i > 0) {

			p_where->draw_line(prev, p, p_color.linear_interpolate(p_to_color, d), 2);
		}

		prev = p;
	}
#endif
}

void GraphEdit::_connections_layer_draw() {

	{
		//draw connections
		List<List<Connection>::Element *> to_erase;
		for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {

			NodePath fromnp(E->get().from);

			Node *from = get_node(fromnp);
			if (!from) {
				to_erase.push_back(E);
				continue;
			}

			GraphNode *gfrom = from->cast_to<GraphNode>();

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

			GraphNode *gto = to->cast_to<GraphNode>();

			if (!gto) {
				to_erase.push_back(E);
				continue;
			}

			Vector2 frompos = gfrom->get_connection_output_pos(E->get().from_port) + gfrom->get_offset() * zoom;
			Color color = gfrom->get_connection_output_color(E->get().from_port);
			Vector2 topos = gto->get_connection_input_pos(E->get().to_port) + gto->get_offset() * zoom;
			Color tocolor = gto->get_connection_input_color(E->get().to_port);
			_draw_cos_line(connections_layer, frompos, topos, color, tocolor);
		}

		while (to_erase.size()) {
			connections.erase(to_erase.front()->get());
			to_erase.pop_front();
		}
	}
}

void GraphEdit::_top_layer_draw() {

	_update_scroll();

	if (connecting) {

		Node *fromn = get_node(connecting_from);
		ERR_FAIL_COND(!fromn);
		GraphNode *from = fromn->cast_to<GraphNode>();
		ERR_FAIL_COND(!from);
		Vector2 pos;
		if (connecting_out)
			pos = from->get_connection_output_pos(connecting_index);
		else
			pos = from->get_connection_input_pos(connecting_index);
		pos += from->get_pos();

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

	if (box_selecting)
		top_layer->draw_rect(box_selecting_rect, Color(0.7, 0.7, 1.0, 0.3));
}

void GraphEdit::set_selected(Node *p_child) {

	for (int i = get_child_count() - 1; i >= 0; i--) {

		GraphNode *gn = get_child(i)->cast_to<GraphNode>();
		if (!gn)
			continue;

		gn->set_selected(gn == p_child);
	}
}

void GraphEdit::_gui_input(const InputEvent &p_ev) {

	if (p_ev.type == InputEvent::MOUSE_MOTION && (p_ev.mouse_motion.button_mask & BUTTON_MASK_MIDDLE || (p_ev.mouse_motion.button_mask & BUTTON_MASK_LEFT && Input::get_singleton()->is_key_pressed(KEY_SPACE)))) {
		h_scroll->set_value(h_scroll->get_value() - p_ev.mouse_motion.relative_x);
		v_scroll->set_value(v_scroll->get_value() - p_ev.mouse_motion.relative_y);
	}

	if (p_ev.type == InputEvent::MOUSE_MOTION && dragging) {

		just_selected = true;
		// TODO: Remove local mouse pos hack if/when InputEventMouseMotion is fixed to support floats
		//drag_accum+=Vector2(p_ev.mouse_motion.relative_x,p_ev.mouse_motion.relative_y);
		drag_accum = get_local_mouse_pos() - drag_origin;
		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphNode *gn = get_child(i)->cast_to<GraphNode>();
			if (gn && gn->is_selected()) {

				Vector2 pos = (gn->get_drag_from() * zoom + drag_accum) / zoom;
				if (is_using_snap()) {
					int snap = get_snap();
					pos = pos.snapped(Vector2(snap, snap));
				}

				gn->set_offset(pos);
			}
		}
	}

	if (p_ev.type == InputEvent::MOUSE_MOTION && box_selecting) {
		box_selecting_to = get_local_mouse_pos();

		box_selecting_rect = Rect2(MIN(box_selecting_from.x, box_selecting_to.x),
				MIN(box_selecting_from.y, box_selecting_to.y),
				ABS(box_selecting_from.x - box_selecting_to.x),
				ABS(box_selecting_from.y - box_selecting_to.y));

		for (int i = get_child_count() - 1; i >= 0; i--) {

			GraphNode *gn = get_child(i)->cast_to<GraphNode>();
			if (!gn)
				continue;

			Rect2 r = gn->get_rect();
			r.size *= zoom;
			bool in_box = r.intersects(box_selecting_rect);

			if (in_box)
				gn->set_selected(box_selection_mode_aditive);
			else
				gn->set_selected(previus_selected.find(gn) != NULL);
		}

		top_layer->update();
	}

	if (p_ev.type == InputEvent::MOUSE_BUTTON) {

		const InputEventMouseButton &b = p_ev.mouse_button;

		if (b.button_index == BUTTON_RIGHT && b.pressed) {
			if (box_selecting) {
				box_selecting = false;
				for (int i = get_child_count() - 1; i >= 0; i--) {

					GraphNode *gn = get_child(i)->cast_to<GraphNode>();
					if (!gn)
						continue;

					gn->set_selected(previus_selected.find(gn) != NULL);
				}
				top_layer->update();
			} else {
				if (connecting) {
					connecting = false;
					top_layer->update();
				} else {
					emit_signal("popup_request", Vector2(b.global_x, b.global_y));
				}
			}
		}

		if (b.button_index == BUTTON_LEFT && !b.pressed && dragging) {
			if (!just_selected && drag_accum == Vector2() && Input::get_singleton()->is_key_pressed(KEY_CONTROL)) {
				//deselect current node
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = get_child(i)->cast_to<GraphNode>();

					if (gn) {
						Rect2 r = gn->get_rect();
						r.size *= zoom;
						if (r.has_point(get_local_mouse_pos()))
							gn->set_selected(false);
					}
				}
			}

			if (drag_accum != Vector2()) {

				emit_signal("_begin_node_move");

				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphNode *gn = get_child(i)->cast_to<GraphNode>();
					if (gn && gn->is_selected())
						gn->set_drag(false);
				}

				emit_signal("_end_node_move");
			}

			dragging = false;

			top_layer->update();
			update();
			connections_layer->update();
		}

		if (b.button_index == BUTTON_LEFT && b.pressed) {

			GraphNode *gn = NULL;
			GraphNode *gn_selected = NULL;
			for (int i = get_child_count() - 1; i >= 0; i--) {

				gn_selected = get_child(i)->cast_to<GraphNode>();

				if (gn_selected) {

					if (gn_selected->is_resizing())
						continue;

					Rect2 r = gn_selected->get_rect();
					r.size *= zoom;
					if (r.has_point(get_local_mouse_pos()))
						gn = gn_selected;
					break;
				}
			}

			if (gn) {

				if (_filter_input(Vector2(b.x, b.y)))
					return;

				dragging = true;
				drag_accum = Vector2();
				drag_origin = get_local_mouse_pos();
				just_selected = !gn->is_selected();
				if (!gn->is_selected() && !Input::get_singleton()->is_key_pressed(KEY_CONTROL)) {
					for (int i = 0; i < get_child_count(); i++) {
						GraphNode *o_gn = get_child(i)->cast_to<GraphNode>();
						if (o_gn)
							o_gn->set_selected(o_gn == gn);
					}
				}

				gn->set_selected(true);
				for (int i = 0; i < get_child_count(); i++) {
					GraphNode *o_gn = get_child(i)->cast_to<GraphNode>();
					if (!o_gn)
						continue;
					if (o_gn->is_selected())
						o_gn->set_drag(true);
				}

			} else {
				if (_filter_input(Vector2(b.x, b.y)))
					return;
				if (Input::get_singleton()->is_key_pressed(KEY_SPACE))
					return;

				box_selecting = true;
				box_selecting_from = get_local_mouse_pos();
				if (b.mod.control) {
					box_selection_mode_aditive = true;
					previus_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {

						GraphNode *gn = get_child(i)->cast_to<GraphNode>();
						if (!gn || !gn->is_selected())
							continue;

						previus_selected.push_back(gn);
					}
				} else if (b.mod.shift) {
					box_selection_mode_aditive = false;
					previus_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {

						GraphNode *gn = get_child(i)->cast_to<GraphNode>();
						if (!gn || !gn->is_selected())
							continue;

						previus_selected.push_back(gn);
					}
				} else {
					box_selection_mode_aditive = true;
					previus_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {

						GraphNode *gn = get_child(i)->cast_to<GraphNode>();
						if (!gn)
							continue;

						gn->set_selected(false);
					}
				}
			}
		}

		if (b.button_index == BUTTON_LEFT && !b.pressed && box_selecting) {
			box_selecting = false;
			previus_selected.clear();
			top_layer->update();
		}

		if (b.button_index == BUTTON_WHEEL_UP && b.pressed) {
			//too difficult to get right
			//set_zoom(zoom*ZOOM_SCALE);
		}

		if (b.button_index == BUTTON_WHEEL_DOWN && b.pressed) {
			//too difficult to get right
			//set_zoom(zoom/ZOOM_SCALE);
		}
	}

	if (p_ev.type == InputEvent::KEY && p_ev.key.scancode == KEY_D && p_ev.key.pressed && p_ev.key.mod.command) {
		emit_signal("duplicate_nodes_request");
		accept_event();
	}

	if (p_ev.type == InputEvent::KEY && p_ev.key.scancode == KEY_DELETE && p_ev.key.pressed) {
		emit_signal("delete_nodes_request");
		accept_event();
	}
}

void GraphEdit::clear_connections() {

	connections.clear();
	update();
	connections_layer->update();
}

void GraphEdit::set_zoom(float p_zoom) {

	p_zoom = CLAMP(p_zoom, MIN_ZOOM, MAX_ZOOM);
	if (zoom == p_zoom)
		return;

	zoom_minus->set_disabled(zoom == MIN_ZOOM);
	zoom_plus->set_disabled(zoom == MAX_ZOOM);

	Vector2 sbofs = (Vector2(h_scroll->get_value(), v_scroll->get_value()) + get_size() / 2) / zoom;

	zoom = p_zoom;
	top_layer->update();

	_update_scroll();
	connections_layer->update();

	if (is_visible_in_tree()) {

		Vector2 ofs = sbofs * zoom - get_size() / 2;
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

void GraphEdit::_bind_methods() {

	ClassDB::bind_method(D_METHOD("connect_node:Error", "from", "from_port", "to", "to_port"), &GraphEdit::connect_node);
	ClassDB::bind_method(D_METHOD("is_node_connected", "from", "from_port", "to", "to_port"), &GraphEdit::is_node_connected);
	ClassDB::bind_method(D_METHOD("disconnect_node", "from", "from_port", "to", "to_port"), &GraphEdit::disconnect_node);
	ClassDB::bind_method(D_METHOD("get_connection_list"), &GraphEdit::_get_connection_list);
	ClassDB::bind_method(D_METHOD("get_scroll_ofs"), &GraphEdit::get_scroll_ofs);
	ClassDB::bind_method(D_METHOD("set_scroll_ofs", "ofs"), &GraphEdit::set_scroll_ofs);

	ClassDB::bind_method(D_METHOD("set_zoom", "p_zoom"), &GraphEdit::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &GraphEdit::get_zoom);

	ClassDB::bind_method(D_METHOD("set_snap", "pixels"), &GraphEdit::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &GraphEdit::get_snap);

	ClassDB::bind_method(D_METHOD("set_use_snap", "enable"), &GraphEdit::set_use_snap);
	ClassDB::bind_method(D_METHOD("is_using_snap"), &GraphEdit::is_using_snap);

	ClassDB::bind_method(D_METHOD("set_right_disconnects", "enable"), &GraphEdit::set_right_disconnects);
	ClassDB::bind_method(D_METHOD("is_right_disconnects_enabled"), &GraphEdit::is_right_disconnects_enabled);

	ClassDB::bind_method(D_METHOD("_graph_node_moved"), &GraphEdit::_graph_node_moved);
	ClassDB::bind_method(D_METHOD("_graph_node_raised"), &GraphEdit::_graph_node_raised);

	ClassDB::bind_method(D_METHOD("_top_layer_input"), &GraphEdit::_top_layer_input);
	ClassDB::bind_method(D_METHOD("_top_layer_draw"), &GraphEdit::_top_layer_draw);
	ClassDB::bind_method(D_METHOD("_scroll_moved"), &GraphEdit::_scroll_moved);
	ClassDB::bind_method(D_METHOD("_zoom_minus"), &GraphEdit::_zoom_minus);
	ClassDB::bind_method(D_METHOD("_zoom_reset"), &GraphEdit::_zoom_reset);
	ClassDB::bind_method(D_METHOD("_zoom_plus"), &GraphEdit::_zoom_plus);
	ClassDB::bind_method(D_METHOD("_snap_toggled"), &GraphEdit::_snap_toggled);
	ClassDB::bind_method(D_METHOD("_snap_value_changed"), &GraphEdit::_snap_value_changed);

	ClassDB::bind_method(D_METHOD("_gui_input"), &GraphEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("_update_scroll_offset"), &GraphEdit::_update_scroll_offset);
	ClassDB::bind_method(D_METHOD("_connections_layer_draw"), &GraphEdit::_connections_layer_draw);

	ClassDB::bind_method(D_METHOD("set_selected", "node"), &GraphEdit::set_selected);

	ADD_SIGNAL(MethodInfo("connection_request", PropertyInfo(Variant::STRING, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::STRING, "to"), PropertyInfo(Variant::INT, "to_slot")));
	ADD_SIGNAL(MethodInfo("disconnection_request", PropertyInfo(Variant::STRING, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::STRING, "to"), PropertyInfo(Variant::INT, "to_slot")));
	ADD_SIGNAL(MethodInfo("popup_request", PropertyInfo(Variant::VECTOR2, "p_position")));
	ADD_SIGNAL(MethodInfo("duplicate_nodes_request"));
	ADD_SIGNAL(MethodInfo("node_selected", PropertyInfo(Variant::OBJECT, "node")));
	ADD_SIGNAL(MethodInfo("connection_to_empty", PropertyInfo(Variant::STRING, "from"), PropertyInfo(Variant::INT, "from_slot"), PropertyInfo(Variant::VECTOR2, "release_pos")));
	ADD_SIGNAL(MethodInfo("delete_nodes_request"));
	ADD_SIGNAL(MethodInfo("_begin_node_move"));
	ADD_SIGNAL(MethodInfo("_end_node_move"));
	ADD_SIGNAL(MethodInfo("scroll_offset_changed", PropertyInfo(Variant::VECTOR2, "ofs")));
}

GraphEdit::GraphEdit() {
	set_focus_mode(FOCUS_ALL);

	awaiting_scroll_offset_update = false;
	top_layer = NULL;
	top_layer = memnew(GraphEditFilter(this));
	add_child(top_layer);
	top_layer->set_mouse_filter(MOUSE_FILTER_PASS);
	top_layer->set_area_as_parent_rect();
	top_layer->connect("draw", this, "_top_layer_draw");
	top_layer->set_mouse_filter(MOUSE_FILTER_PASS);
	top_layer->connect("gui_input", this, "_top_layer_input");

	connections_layer = memnew(Control);
	add_child(connections_layer);
	connections_layer->connect("draw", this, "_connections_layer_draw");
	connections_layer->set_name("CLAYER");
	connections_layer->set_disable_visibility_clip(true); // so it can draw freely and be offseted

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

	h_scroll->connect("value_changed", this, "_scroll_moved");
	v_scroll->connect("value_changed", this, "_scroll_moved");

	zoom = 1;

	HBoxContainer *zoom_hb = memnew(HBoxContainer);
	top_layer->add_child(zoom_hb);
	zoom_hb->set_pos(Vector2(10, 10));

	zoom_minus = memnew(ToolButton);
	zoom_hb->add_child(zoom_minus);
	zoom_minus->connect("pressed", this, "_zoom_minus");
	zoom_minus->set_focus_mode(FOCUS_NONE);

	zoom_reset = memnew(ToolButton);
	zoom_hb->add_child(zoom_reset);
	zoom_reset->connect("pressed", this, "_zoom_reset");
	zoom_reset->set_focus_mode(FOCUS_NONE);

	zoom_plus = memnew(ToolButton);
	zoom_hb->add_child(zoom_plus);
	zoom_plus->connect("pressed", this, "_zoom_plus");
	zoom_plus->set_focus_mode(FOCUS_NONE);

	snap_button = memnew(ToolButton);
	snap_button->set_toggle_mode(true);
	snap_button->connect("pressed", this, "_snap_toggled");
	snap_button->set_pressed(true);
	snap_button->set_focus_mode(FOCUS_NONE);
	zoom_hb->add_child(snap_button);

	snap_amount = memnew(SpinBox);
	snap_amount->set_min(5);
	snap_amount->set_max(100);
	snap_amount->set_step(1);
	snap_amount->set_value(20);
	snap_amount->connect("value_changed", this, "_snap_value_changed");
	zoom_hb->add_child(snap_amount);

	setting_scroll_ofs = false;
	just_disconected = false;
	set_clip_contents(true);
}
