/**************************************************************************/
/*  graph_edit.cpp                                                        */
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

#include "graph_edit.h"
#include "graph_edit.compat.inc"

#include "core/input/input.h"
#include "core/math/math_funcs.h"
#include "core/os/keyboard.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit_arranger.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/view_panner.h"
#include "scene/resources/style_box_flat.h"
#include "scene/theme/theme_db.h"

constexpr int MINIMAP_OFFSET = 12;
constexpr int MINIMAP_PADDING = 5;
constexpr int MIN_DRAG_DISTANCE_FOR_VALID_CONNECTION = 20;
constexpr int MAX_CONNECTION_LINE_CURVE_TESSELATION_STAGES = 5;
constexpr int GRID_MINOR_STEPS_PER_MAJOR_LINE = 10;
constexpr int GRID_MIN_SNAPPING_DISTANCE = 2;
constexpr int GRID_MAX_SNAPPING_DISTANCE = 100;
constexpr float CONNECTING_TARGET_LINE_COLOR_BRIGHTENING = 0.4;

bool GraphEditFilter::has_point(const Point2 &p_point) const {
	return ge->_filter_input(p_point);
}

GraphEditFilter::GraphEditFilter(GraphEdit *p_edit) {
	ge = p_edit;
}

Control::CursorShape GraphEditMinimap::get_cursor_shape(const Point2 &p_pos) const {
	if (is_resizing || (p_pos.x < theme_cache.resizer->get_width() && p_pos.y < theme_cache.resizer->get_height())) {
		return CURSOR_FDIAGSIZE;
	}

	return Control::get_cursor_shape(p_pos);
}

void GraphEditMinimap::update_minimap() {
	Vector2 graph_offset = _get_graph_offset();
	Vector2 graph_size = _get_graph_size();

	camera_position = ge->get_scroll_offset() - graph_offset;
	camera_size = ge->get_size();

	Vector2 render_size = _get_render_size();
	float target_ratio = render_size.width / render_size.height;
	float graph_ratio = graph_size.width / graph_size.height;

	graph_proportions = graph_size;
	graph_padding = Vector2(0, 0);
	if (graph_ratio > target_ratio) {
		graph_proportions.width = graph_size.width;
		graph_proportions.height = graph_size.width / target_ratio;
		graph_padding.y = Math::abs(graph_size.height - graph_proportions.y) / 2;
	} else {
		graph_proportions.width = graph_size.height * target_ratio;
		graph_proportions.height = graph_size.height;
		graph_padding.x = Math::abs(graph_size.width - graph_proportions.x) / 2;
	}

	// This centers minimap inside the minimap rectangle.
	minimap_offset = minimap_padding + _convert_from_graph_position(graph_padding);
}

Rect2 GraphEditMinimap::get_camera_rect() {
	Vector2 camera_center = _convert_from_graph_position(camera_position + camera_size / 2) + minimap_offset;
	Vector2 camera_viewport = _convert_from_graph_position(camera_size);
	Vector2 camera_pos = (camera_center - camera_viewport / 2);
	return Rect2(camera_pos, camera_viewport);
}

Vector2 GraphEditMinimap::_get_render_size() {
	if (!is_inside_tree()) {
		return Vector2(0, 0);
	}

	return get_size() - 2 * minimap_padding;
}

Vector2 GraphEditMinimap::_get_graph_offset() {
	return Vector2(ge->h_scrollbar->get_min(), ge->v_scrollbar->get_min());
}

Vector2 GraphEditMinimap::_get_graph_size() {
	Vector2 graph_size = Vector2(ge->h_scrollbar->get_max(), ge->v_scrollbar->get_max()) - Vector2(ge->h_scrollbar->get_min(), ge->v_scrollbar->get_min());

	if (graph_size.width == 0) {
		graph_size.width = 1;
	}
	if (graph_size.height == 0) {
		graph_size.height = 1;
	}

	return graph_size;
}

Vector2 GraphEditMinimap::_convert_from_graph_position(const Vector2 &p_position) {
	Vector2 map_position = Vector2(0, 0);
	Vector2 render_size = _get_render_size();

	map_position.x = p_position.x * render_size.width / graph_proportions.x;
	map_position.y = p_position.y * render_size.height / graph_proportions.y;

	return map_position;
}

Vector2 GraphEditMinimap::_convert_to_graph_position(const Vector2 &p_position) {
	Vector2 graph_position = Vector2(0, 0);
	Vector2 render_size = _get_render_size();

	graph_position.x = p_position.x * graph_proportions.x / render_size.width;
	graph_position.y = p_position.y * graph_proportions.y / render_size.height;

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

			Rect2 resizer_hitbox = Rect2(Point2(), theme_cache.resizer->get_size());
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
			// Prevent setting minimap wider than GraphEdit.
			Vector2 new_minimap_size;
			new_minimap_size.width = MIN(get_size().width - mm->get_relative().x, ge->get_size().width - 2.0 * minimap_padding.x);
			new_minimap_size.height = MIN(get_size().height - mm->get_relative().y, ge->get_size().height - 2.0 * minimap_padding.y);
			ge->set_minimap_size(new_minimap_size);

			queue_redraw();
		} else {
			Vector2 click_position = _convert_to_graph_position(mm->get_position() - minimap_padding) - graph_padding;
			_adjust_graph_scroll(click_position);
		}
		accept_event();
	}
}

void GraphEditMinimap::_adjust_graph_scroll(const Vector2 &p_offset) {
	Vector2 graph_offset = _get_graph_offset();
	ge->set_scroll_offset(p_offset + graph_offset - camera_size / 2);
}

void GraphEditMinimap::_bind_methods() {
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphEditMinimap, panel);
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, GraphEditMinimap, node_style, "node");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, GraphEditMinimap, camera_style, "camera");
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEditMinimap, resizer);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphEditMinimap, resizer_color);
}

GraphEditMinimap::GraphEditMinimap(GraphEdit *p_edit) {
	ge = p_edit;

	minimap_padding = Vector2(MINIMAP_PADDING, MINIMAP_PADDING);
	minimap_offset = minimap_padding + _convert_from_graph_position(graph_padding);
}

Control::CursorShape GraphEdit::get_cursor_shape(const Point2 &p_pos) const {
	if (moving_selection) {
		return CURSOR_MOVE;
	}

	return Control::get_cursor_shape(p_pos);
}

PackedStringArray GraphEdit::get_configuration_warnings() const {
	PackedStringArray warnings = Control::get_configuration_warnings();

	warnings.push_back(RTR("Please be aware that GraphEdit and GraphNode will undergo extensive refactoring in a future 4.x version involving compatibility-breaking API changes."));

	return warnings;
}

Error GraphEdit::connect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	if (is_node_connected(p_from, p_from_port, p_to, p_to_port)) {
		return OK;
	}
	Connection c;
	c.from_node = p_from;
	c.from_port = p_from_port;
	c.to_node = p_to;
	c.to_port = p_to_port;
	c.activity = 0;
	connections.push_back(c);
	top_layer->queue_redraw();
	minimap->queue_redraw();
	queue_redraw();
	connections_layer->queue_redraw();

	return OK;
}

bool GraphEdit::is_node_connected(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	for (const Connection &E : connections) {
		if (E.from_node == p_from && E.from_port == p_from_port && E.to_node == p_to && E.to_port == p_to_port) {
			return true;
		}
	}

	return false;
}

void GraphEdit::disconnect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	for (const List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		if (E->get().from_node == p_from && E->get().from_port == p_from_port && E->get().to_node == p_to && E->get().to_port == p_to_port) {
			connections.erase(E);
			top_layer->queue_redraw();
			minimap->queue_redraw();
			queue_redraw();
			connections_layer->queue_redraw();
			return;
		}
	}
}

void GraphEdit::get_connection_list(List<Connection> *r_connections) const {
	*r_connections = connections;
}

void GraphEdit::set_scroll_offset(const Vector2 &p_offset) {
	setting_scroll_offset = true;
	h_scrollbar->set_value(p_offset.x);
	v_scrollbar->set_value(p_offset.y);
	_update_scroll();
	setting_scroll_offset = false;
}

Vector2 GraphEdit::get_scroll_offset() const {
	return Vector2(h_scrollbar->get_value(), v_scrollbar->get_value());
}

void GraphEdit::_scroll_moved(double) {
	if (!awaiting_scroll_offset_update) {
		callable_mp(this, &GraphEdit::_update_scroll_offset).call_deferred();
		awaiting_scroll_offset_update = true;
	}
	top_layer->queue_redraw();
	minimap->queue_redraw();
	queue_redraw();
}

void GraphEdit::_update_scroll_offset() {
	set_block_minimum_size_adjust(true);

	for (int i = 0; i < get_child_count(); i++) {
		GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));
		if (!graph_element) {
			continue;
		}

		Point2 pos = graph_element->get_position_offset() * zoom;
		pos -= Point2(h_scrollbar->get_value(), v_scrollbar->get_value());
		graph_element->set_position(pos);
		if (graph_element->get_scale() != Vector2(zoom, zoom)) {
			graph_element->set_scale(Vector2(zoom, zoom));
		}
	}

	connections_layer->set_position(-Point2(h_scrollbar->get_value(), v_scrollbar->get_value()));
	set_block_minimum_size_adjust(false);
	awaiting_scroll_offset_update = false;

	// In Godot, signals on value change are avoided by convention.
	if (!setting_scroll_offset) {
		emit_signal(SNAME("scroll_offset_changed"), get_scroll_offset());
	}
}

void GraphEdit::_update_scroll() {
	if (updating) {
		return;
	}
	updating = true;

	set_block_minimum_size_adjust(true);

	Rect2 screen_rect;
	for (int i = 0; i < get_child_count(); i++) {
		GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));
		if (!graph_element) {
			continue;
		}

		Rect2 node_rect;
		node_rect.position = graph_element->get_position_offset() * zoom;
		node_rect.size = graph_element->get_size() * zoom;
		screen_rect = screen_rect.merge(node_rect);
	}

	screen_rect.position -= get_size();
	screen_rect.size += get_size() * 2.0;

	h_scrollbar->set_min(screen_rect.position.x);
	h_scrollbar->set_max(screen_rect.position.x + screen_rect.size.width);
	h_scrollbar->set_page(get_size().x);
	if (h_scrollbar->get_max() - h_scrollbar->get_min() <= h_scrollbar->get_page()) {
		h_scrollbar->hide();
	} else {
		h_scrollbar->show();
	}

	v_scrollbar->set_min(screen_rect.position.y);
	v_scrollbar->set_max(screen_rect.position.y + screen_rect.size.height);
	v_scrollbar->set_page(get_size().height);

	if (v_scrollbar->get_max() - v_scrollbar->get_min() <= v_scrollbar->get_page()) {
		v_scrollbar->hide();
	} else {
		v_scrollbar->show();
	}

	Size2 hmin = h_scrollbar->get_combined_minimum_size();
	Size2 vmin = v_scrollbar->get_combined_minimum_size();

	// Avoid scrollbar overlapping.
	h_scrollbar->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, v_scrollbar->is_visible() ? -vmin.width : 0);
	v_scrollbar->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, h_scrollbar->is_visible() ? -hmin.height : 0);

	set_block_minimum_size_adjust(false);

	if (!awaiting_scroll_offset_update) {
		callable_mp(this, &GraphEdit::_update_scroll_offset).call_deferred();
		awaiting_scroll_offset_update = true;
	}

	updating = false;
}

void GraphEdit::_graph_element_moved_to_front(Node *p_node) {
	GraphElement *graph_element = Object::cast_to<GraphElement>(p_node);
	ERR_FAIL_NULL(graph_element);

	graph_element->move_to_front();
}

void GraphEdit::_graph_element_selected(Node *p_node) {
	GraphElement *graph_element = Object::cast_to<GraphElement>(p_node);
	ERR_FAIL_NULL(graph_element);

	emit_signal(SNAME("node_selected"), graph_element);
}

void GraphEdit::_graph_element_deselected(Node *p_node) {
	GraphElement *graph_element = Object::cast_to<GraphElement>(p_node);
	ERR_FAIL_NULL(graph_element);

	emit_signal(SNAME("node_deselected"), graph_element);
}

void GraphEdit::_graph_element_resized(Vector2 p_new_minsize, Node *p_node) {
	GraphElement *graph_element = Object::cast_to<GraphElement>(p_node);
	ERR_FAIL_NULL(graph_element);

	graph_element->set_size(p_new_minsize);
}

void GraphEdit::_graph_element_moved(Node *p_node) {
	GraphElement *graph_element = Object::cast_to<GraphElement>(p_node);
	ERR_FAIL_NULL(graph_element);

	top_layer->queue_redraw();
	minimap->queue_redraw();
	queue_redraw();
	connections_layer->queue_redraw();
}

void GraphEdit::_graph_node_slot_updated(int p_index, Node *p_node) {
	GraphNode *graph_node = Object::cast_to<GraphNode>(p_node);
	ERR_FAIL_NULL(graph_node);

	top_layer->queue_redraw();
	minimap->queue_redraw();
	queue_redraw();
	connections_layer->queue_redraw();
}

void GraphEdit::add_child_notify(Node *p_child) {
	Control::add_child_notify(p_child);

	// Keep the top layer always on top!
	callable_mp((CanvasItem *)top_layer, &CanvasItem::move_to_front).call_deferred();

	GraphElement *graph_element = Object::cast_to<GraphElement>(p_child);
	if (graph_element) {
		graph_element->connect("position_offset_changed", callable_mp(this, &GraphEdit::_graph_element_moved).bind(graph_element));
		graph_element->connect("node_selected", callable_mp(this, &GraphEdit::_graph_element_selected).bind(graph_element));
		graph_element->connect("node_deselected", callable_mp(this, &GraphEdit::_graph_element_deselected).bind(graph_element));

		GraphNode *graph_node = Object::cast_to<GraphNode>(graph_element);
		if (graph_node) {
			graph_element->connect("slot_updated", callable_mp(this, &GraphEdit::_graph_node_slot_updated).bind(graph_element));
		}

		graph_element->connect("raise_request", callable_mp(this, &GraphEdit::_graph_element_moved_to_front).bind(graph_element));
		graph_element->connect("resize_request", callable_mp(this, &GraphEdit::_graph_element_resized).bind(graph_element));
		graph_element->connect("item_rect_changed", callable_mp((CanvasItem *)connections_layer, &CanvasItem::queue_redraw));
		graph_element->connect("item_rect_changed", callable_mp((CanvasItem *)minimap, &GraphEditMinimap::queue_redraw));

		graph_element->set_scale(Vector2(zoom, zoom));
		_graph_element_moved(graph_element);
		graph_element->set_mouse_filter(MOUSE_FILTER_PASS);
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
		// Keep the top layer always on top!
		callable_mp((CanvasItem *)top_layer, &CanvasItem::move_to_front).call_deferred();
	}

	GraphElement *graph_element = Object::cast_to<GraphElement>(p_child);
	if (graph_element) {
		graph_element->disconnect("position_offset_changed", callable_mp(this, &GraphEdit::_graph_element_moved));
		graph_element->disconnect("node_selected", callable_mp(this, &GraphEdit::_graph_element_selected));
		graph_element->disconnect("node_deselected", callable_mp(this, &GraphEdit::_graph_element_deselected));

		GraphNode *graph_node = Object::cast_to<GraphNode>(graph_element);
		if (graph_node) {
			graph_element->disconnect("slot_updated", callable_mp(this, &GraphEdit::_graph_node_slot_updated));
		}

		graph_element->disconnect("raise_request", callable_mp(this, &GraphEdit::_graph_element_moved_to_front));
		graph_element->disconnect("resize_request", callable_mp(this, &GraphEdit::_graph_element_resized));

		// In case of the whole GraphEdit being destroyed these references can already be freed.
		if (connections_layer != nullptr && connections_layer->is_inside_tree()) {
			graph_element->disconnect("item_rect_changed", callable_mp((CanvasItem *)connections_layer, &CanvasItem::queue_redraw));
		}
		if (minimap != nullptr && minimap->is_inside_tree()) {
			graph_element->disconnect("item_rect_changed", callable_mp((CanvasItem *)minimap, &GraphEditMinimap::queue_redraw));
		}
	}
}

void GraphEdit::_update_theme_item_cache() {
	Control::_update_theme_item_cache();

	theme_cache.base_scale = get_theme_default_base_scale();
}

void GraphEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			zoom_minus_button->set_icon(theme_cache.zoom_out);
			zoom_reset_button->set_icon(theme_cache.zoom_reset);
			zoom_plus_button->set_icon(theme_cache.zoom_in);

			toggle_snapping_button->set_icon(theme_cache.snapping_toggle);
			toggle_grid_button->set_icon(theme_cache.grid_toggle);
			minimap_button->set_icon(theme_cache.minimap_toggle);
			arrange_button->set_icon(theme_cache.layout);

			zoom_label->set_custom_minimum_size(Size2(48, 0) * theme_cache.base_scale);

			menu_panel->add_theme_style_override("panel", theme_cache.menu_panel);
		} break;

		case NOTIFICATION_READY: {
			Size2 hmin = h_scrollbar->get_combined_minimum_size();
			Size2 vmin = v_scrollbar->get_combined_minimum_size();

			h_scrollbar->set_anchor_and_offset(SIDE_LEFT, ANCHOR_BEGIN, 0);
			h_scrollbar->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
			h_scrollbar->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -hmin.height);
			h_scrollbar->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);

			v_scrollbar->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -vmin.width);
			v_scrollbar->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
			v_scrollbar->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, 0);
			v_scrollbar->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);
		} break;

		case NOTIFICATION_DRAW: {
			// Draw background fill.
			draw_style_box(theme_cache.panel, Rect2(Point2(), get_size()));

			// Draw background grid.
			if (show_grid) {
				Vector2 offset = get_scroll_offset() / zoom;
				Size2 size = get_size() / zoom;

				Point2i from_pos = (offset / float(snapping_distance)).floor();
				Point2i len = (size / float(snapping_distance)).floor() + Vector2(1, 1);

				for (int i = from_pos.x; i < from_pos.x + len.x; i++) {
					Color color;

					if (ABS(i) % GRID_MINOR_STEPS_PER_MAJOR_LINE == 0) {
						color = theme_cache.grid_major;
					} else {
						color = theme_cache.grid_minor;
					}

					float base_offset = i * snapping_distance * zoom - offset.x * zoom;
					draw_line(Vector2(base_offset, 0), Vector2(base_offset, get_size().height), color);
				}

				for (int i = from_pos.y; i < from_pos.y + len.y; i++) {
					Color color;

					if (ABS(i) % GRID_MINOR_STEPS_PER_MAJOR_LINE == 0) {
						color = theme_cache.grid_major;
					} else {
						color = theme_cache.grid_minor;
					}

					float base_offset = i * snapping_distance * zoom - offset.y * zoom;
					draw_line(Vector2(0, base_offset), Vector2(get_size().width, base_offset), color);
				}
			}
		} break;

		case NOTIFICATION_RESIZED: {
			_update_scroll();
			top_layer->queue_redraw();
			minimap->queue_redraw();
		} break;
	}
}

bool GraphEdit::_filter_input(const Point2 &p_point) {
	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *graph_node = Object::cast_to<GraphNode>(get_child(i));
		if (!graph_node || !graph_node->is_visible_in_tree()) {
			continue;
		}

		Ref<Texture2D> port_icon = graph_node->theme_cache.port;

		for (int j = 0; j < graph_node->get_input_port_count(); j++) {
			Vector2i port_size = Vector2i(port_icon->get_width(), port_icon->get_height());

			// Determine slot height.
			int slot_index = graph_node->get_input_port_slot(j);
			Control *child = Object::cast_to<Control>(graph_node->get_child(slot_index, false));

			port_size.height = MAX(port_size.height, child ? child->get_size().y : 0);

			if (is_in_input_hotzone(graph_node, j, p_point / zoom, port_size)) {
				return true;
			}
		}

		for (int j = 0; j < graph_node->get_output_port_count(); j++) {
			Vector2i port_size = Vector2i(port_icon->get_width(), port_icon->get_height());

			// Determine slot height.
			int slot_index = graph_node->get_output_port_slot(j);
			Control *child = Object::cast_to<Control>(graph_node->get_child(slot_index, false));
			port_size.height = MAX(port_size.height, child ? child->get_size().y : 0);

			if (is_in_output_hotzone(graph_node, j, p_point / zoom, port_size)) {
				return true;
			}
		}
	}

	return false;
}

void GraphEdit::_top_layer_input(const Ref<InputEvent> &p_ev) {
	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
		connecting_valid = false;
		click_pos = mb->get_position() / zoom;
		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphNode *graph_node = Object::cast_to<GraphNode>(get_child(i));
			if (!graph_node || !graph_node->is_visible_in_tree()) {
				continue;
			}

			Ref<Texture2D> port_icon = graph_node->theme_cache.port;

			for (int j = 0; j < graph_node->get_output_port_count(); j++) {
				Vector2 pos = graph_node->get_output_port_position(j) * zoom + graph_node->get_position();
				Vector2i port_size = Vector2i(port_icon->get_width(), port_icon->get_height());

				// Determine slot height.
				int slot_index = graph_node->get_output_port_slot(j);
				Control *child = Object::cast_to<Control>(graph_node->get_child(slot_index, false));
				port_size.height = MAX(port_size.height, child ? child->get_size().y : 0);

				if (is_in_output_hotzone(graph_node, j, click_pos, port_size)) {
					if (valid_left_disconnect_types.has(graph_node->get_output_port_type(j))) {
						// Check disconnect.
						for (const Connection &E : connections) {
							if (E.from_node == graph_node->get_name() && E.from_port == j) {
								Node *to = get_node(NodePath(E.to_node));
								if (Object::cast_to<GraphNode>(to)) {
									connecting_from = E.to_node;
									connecting_index = E.to_port;
									connecting_out = false;
									connecting_type = Object::cast_to<GraphNode>(to)->get_input_port_type(E.to_port);
									connecting_color = Object::cast_to<GraphNode>(to)->get_input_port_color(E.to_port);
									connecting_target = false;
									connecting_to = pos;

									if (connecting_type >= 0) {
										just_disconnected = true;

										emit_signal(SNAME("disconnection_request"), E.from_node, E.from_port, E.to_node, E.to_port);
										to = get_node(NodePath(connecting_from)); // Maybe it was erased.
										if (Object::cast_to<GraphNode>(to)) {
											connecting = true;
											emit_signal(SNAME("connection_drag_started"), connecting_from, connecting_index, false);
										}
									}
									return;
								}
							}
						}
					}

					connecting_from = graph_node->get_name();
					connecting_index = j;
					connecting_out = true;
					connecting_type = graph_node->get_output_port_type(j);
					connecting_color = graph_node->get_output_port_color(j);
					connecting_target = false;
					connecting_to = pos;
					if (connecting_type >= 0) {
						connecting = true;
						just_disconnected = false;
						emit_signal(SNAME("connection_drag_started"), connecting_from, connecting_index, true);
					}
					return;
				}
			}

			for (int j = 0; j < graph_node->get_input_port_count(); j++) {
				Vector2 pos = graph_node->get_input_port_position(j) * zoom + graph_node->get_position();

				Vector2i port_size = Vector2i(port_icon->get_width(), port_icon->get_height());

				// Determine slot height.
				int slot_index = graph_node->get_input_port_slot(j);
				Control *child = Object::cast_to<Control>(graph_node->get_child(slot_index, false));
				port_size.height = MAX(port_size.height, child ? child->get_size().y : 0);

				if (is_in_input_hotzone(graph_node, j, click_pos, port_size)) {
					if (right_disconnects || valid_right_disconnect_types.has(graph_node->get_input_port_type(j))) {
						// Check disconnect.
						for (const Connection &E : connections) {
							if (E.to_node == graph_node->get_name() && E.to_port == j) {
								Node *fr = get_node(NodePath(E.from_node));
								if (Object::cast_to<GraphNode>(fr)) {
									connecting_from = E.from_node;
									connecting_index = E.from_port;
									connecting_out = true;
									connecting_type = Object::cast_to<GraphNode>(fr)->get_output_port_type(E.from_port);
									connecting_color = Object::cast_to<GraphNode>(fr)->get_output_port_color(E.from_port);
									connecting_target = false;
									connecting_to = pos;
									just_disconnected = true;

									if (connecting_type >= 0) {
										emit_signal(SNAME("disconnection_request"), E.from_node, E.from_port, E.to_node, E.to_port);
										fr = get_node(NodePath(connecting_from));
										if (Object::cast_to<GraphNode>(fr)) {
											connecting = true;
											emit_signal(SNAME("connection_drag_started"), connecting_from, connecting_index, true);
										}
									}
									return;
								}
							}
						}
					}

					connecting_from = graph_node->get_name();
					connecting_index = j;
					connecting_out = false;
					connecting_type = graph_node->get_input_port_type(j);
					connecting_color = graph_node->get_input_port_color(j);
					connecting_target = false;
					connecting_to = pos;
					if (connecting_type >= 0) {
						connecting = true;
						just_disconnected = false;
						emit_signal(SNAME("connection_drag_started"), connecting_from, connecting_index, false);
					}
					return;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_ev;
	if (mm.is_valid() && connecting) {
		connecting_to = mm->get_position();
		connecting_target = false;
		top_layer->queue_redraw();
		minimap->queue_redraw();

		connecting_valid = just_disconnected || click_pos.distance_to(connecting_to / zoom) > MIN_DRAG_DISTANCE_FOR_VALID_CONNECTION;

		if (connecting_valid) {
			Vector2 mpos = mm->get_position() / zoom;
			for (int i = get_child_count() - 1; i >= 0; i--) {
				GraphNode *graph_node = Object::cast_to<GraphNode>(get_child(i));
				if (!graph_node || !graph_node->is_visible_in_tree()) {
					continue;
				}

				Ref<Texture2D> port_icon = graph_node->theme_cache.port;

				if (!connecting_out) {
					for (int j = 0; j < graph_node->get_output_port_count(); j++) {
						Vector2 pos = graph_node->get_output_port_position(j) * zoom + graph_node->get_position();
						Vector2i port_size = Vector2i(port_icon->get_width(), port_icon->get_height());

						// Determine slot height.
						int slot_index = graph_node->get_output_port_slot(j);
						Control *child = Object::cast_to<Control>(graph_node->get_child(slot_index, false));
						port_size.height = MAX(port_size.height, child ? child->get_size().y : 0);

						int type = graph_node->get_output_port_type(j);
						if ((type == connecting_type ||
									valid_connection_types.has(ConnectionType(type, connecting_type))) &&
								is_in_output_hotzone(graph_node, j, mpos, port_size)) {
							if (!is_node_hover_valid(graph_node->get_name(), j, connecting_from, connecting_index)) {
								continue;
							}
							connecting_target = true;
							connecting_to = pos;
							connecting_target_to = graph_node->get_name();
							connecting_target_index = j;
							return;
						}
					}
				} else {
					for (int j = 0; j < graph_node->get_input_port_count(); j++) {
						Vector2 pos = graph_node->get_input_port_position(j) * zoom + graph_node->get_position();
						Vector2i port_size = Vector2i(port_icon->get_width(), port_icon->get_height());

						// Determine slot height.
						int slot_index = graph_node->get_input_port_slot(j);
						Control *child = Object::cast_to<Control>(graph_node->get_child(slot_index, false));
						port_size.height = MAX(port_size.height, child ? child->get_size().y : 0);

						int type = graph_node->get_input_port_type(j);
						if ((type == connecting_type || valid_connection_types.has(ConnectionType(connecting_type, type))) &&
								is_in_input_hotzone(graph_node, j, mpos, port_size)) {
							if (!is_node_hover_valid(connecting_from, connecting_index, graph_node->get_name(), j)) {
								continue;
							}
							connecting_target = true;
							connecting_to = pos;
							connecting_target_to = graph_node->get_name();
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
				if (connecting_out) {
					emit_signal(SNAME("connection_request"), connecting_from, connecting_index, connecting_target_to, connecting_target_index);
				} else {
					emit_signal(SNAME("connection_request"), connecting_target_to, connecting_target_index, connecting_from, connecting_index);
				}
			} else if (!just_disconnected) {
				if (connecting_out) {
					emit_signal(SNAME("connection_to_empty"), connecting_from, connecting_index, mb->get_position());
				} else {
					emit_signal(SNAME("connection_from_empty"), connecting_from, connecting_index, mb->get_position());
				}
			}
		}

		if (connecting) {
			force_connection_drag_end();
		}
	}
}

bool GraphEdit::_check_clickable_control(Control *p_control, const Vector2 &mpos, const Vector2 &p_offset) {
	if (p_control->is_set_as_top_level() || !p_control->is_visible() || !p_control->is_inside_tree()) {
		return false;
	}

	Rect2 control_rect = p_control->get_rect();
	control_rect.position *= zoom;
	control_rect.size *= zoom;
	control_rect.position += p_offset;

	if (!control_rect.has_point(mpos) || p_control->get_mouse_filter() == MOUSE_FILTER_IGNORE) {
		// Test children.
		for (int i = 0; i < p_control->get_child_count(); i++) {
			Control *child_rect = Object::cast_to<Control>(p_control->get_child(i));
			if (!child_rect) {
				continue;
			}
			if (_check_clickable_control(child_rect, mpos, control_rect.position)) {
				return true;
			}
		}

		return false;
	} else {
		return true;
	}
}

bool GraphEdit::is_in_input_hotzone(GraphNode *p_graph_node, int p_port_idx, const Vector2 &p_mouse_pos, const Vector2i &p_port_size) {
	bool success;
	if (GDVIRTUAL_CALL(_is_in_input_hotzone, p_graph_node, p_port_idx, p_mouse_pos, success)) {
		return success;
	} else {
		Vector2 pos = p_graph_node->get_input_port_position(p_port_idx) * zoom + p_graph_node->get_position();
		return is_in_port_hotzone(pos / zoom, p_mouse_pos, p_port_size, true);
	}
}

bool GraphEdit::is_in_output_hotzone(GraphNode *p_graph_node, int p_port_idx, const Vector2 &p_mouse_pos, const Vector2i &p_port_size) {
	if (p_graph_node->is_resizable()) {
		Ref<Texture2D> resizer = p_graph_node->theme_cache.resizer;
		Rect2 resizer_rect = Rect2(p_graph_node->get_position() / zoom + p_graph_node->get_size() - resizer->get_size(), resizer->get_size());
		if (resizer_rect.has_point(p_mouse_pos)) {
			return false;
		}
	}

	bool success;
	if (GDVIRTUAL_CALL(_is_in_output_hotzone, p_graph_node, p_port_idx, p_mouse_pos, success)) {
		return success;
	} else {
		Vector2 pos = p_graph_node->get_output_port_position(p_port_idx) * zoom + p_graph_node->get_position();
		return is_in_port_hotzone(pos / zoom, p_mouse_pos, p_port_size, false);
	}
}

bool GraphEdit::is_in_port_hotzone(const Vector2 &p_pos, const Vector2 &p_mouse_pos, const Vector2i &p_port_size, bool p_left) {
	Rect2 hotzone = Rect2(
			p_pos.x - (p_left ? theme_cache.port_hotzone_outer_extent : theme_cache.port_hotzone_inner_extent),
			p_pos.y - p_port_size.height / 2.0,
			theme_cache.port_hotzone_inner_extent + theme_cache.port_hotzone_outer_extent,
			p_port_size.height);

	if (!hotzone.has_point(p_mouse_pos)) {
		return false;
	}

	for (int i = 0; i < get_child_count(); i++) {
		GraphNode *child = Object::cast_to<GraphNode>(get_child(i));
		if (!child) {
			continue;
		}

		Rect2 child_rect = child->get_rect();
		if (child_rect.has_point(p_mouse_pos * zoom)) {
			for (int j = 0; j < child->get_child_count(); j++) {
				Control *subchild = Object::cast_to<Control>(child->get_child(j));
				if (!subchild) {
					continue;
				}

				if (_check_clickable_control(subchild, p_mouse_pos * zoom, child_rect.position)) {
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

	float x_diff = (p_to.x - p_from.x);
	float cp_offset = x_diff * lines_curvature;
	if (x_diff < 0) {
		cp_offset *= -1;
	}

	Curve2D curve;
	curve.add_point(p_from);
	curve.set_point_out(0, Vector2(cp_offset, 0));
	curve.add_point(p_to);
	curve.set_point_in(1, Vector2(-cp_offset, 0));

	if (lines_curvature > 0) {
		return curve.tessellate(MAX_CONNECTION_LINE_CURVE_TESSELATION_STAGES, 2.0);
	} else {
		return curve.tessellate(1);
	}
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

	// Thickness below 0.5 doesn't look good on the graph or its minimap.
	p_where->draw_polyline_colors(scaled_points, colors, MAX(0.5, Math::floor(p_width * theme_cache.base_scale)), lines_antialiased);
}

void GraphEdit::_connections_layer_draw() {
	// Draw connections.
	List<List<Connection>::Element *> to_erase;
	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		const Connection &c = E->get();

		Node *from = get_node(NodePath(c.from_node));
		GraphNode *gnode_from = Object::cast_to<GraphNode>(from);

		if (!gnode_from) {
			to_erase.push_back(E);
			continue;
		}

		Node *to = get_node(NodePath(c.to_node));
		GraphNode *gnode_to = Object::cast_to<GraphNode>(to);

		if (!gnode_to) {
			to_erase.push_back(E);
			continue;
		}

		Vector2 frompos = gnode_from->get_output_port_position(c.from_port) * zoom + gnode_from->get_position_offset() * zoom;
		Color color = gnode_from->get_output_port_color(c.from_port);
		Vector2 topos = gnode_to->get_input_port_position(c.to_port) * zoom + gnode_to->get_position_offset() * zoom;
		Color tocolor = gnode_to->get_input_port_color(c.to_port);

		if (c.activity > 0) {
			color = color.lerp(theme_cache.activity_color, c.activity);
			tocolor = tocolor.lerp(theme_cache.activity_color, c.activity);
		}
		_draw_connection_line(connections_layer, frompos, topos, color, tocolor, lines_thickness, zoom);
	}

	for (List<Connection>::Element *&E : to_erase) {
		connections.erase(E);
	}
}

void GraphEdit::_top_layer_draw() {
	_update_scroll();

	if (connecting) {
		Node *node_from = get_node_or_null(NodePath(connecting_from));
		ERR_FAIL_NULL(node_from);
		GraphNode *graph_node_from = Object::cast_to<GraphNode>(node_from);
		ERR_FAIL_NULL(graph_node_from);
		Vector2 pos;
		if (connecting_out) {
			pos = graph_node_from->get_output_port_position(connecting_index) * zoom;
		} else {
			pos = graph_node_from->get_input_port_position(connecting_index) * zoom;
		}
		pos += graph_node_from->get_position();

		Vector2 to_pos = connecting_to;
		Color line_color = connecting_color;

		// Draw the line to the mouse cursor brighter when it's over a valid target port.
		if (connecting_target) {
			line_color.r += CONNECTING_TARGET_LINE_COLOR_BRIGHTENING;
			line_color.g += CONNECTING_TARGET_LINE_COLOR_BRIGHTENING;
			line_color.b += CONNECTING_TARGET_LINE_COLOR_BRIGHTENING;
		}

		if (!connecting_out) {
			SWAP(pos, to_pos);
		}
		_draw_connection_line(top_layer, pos, to_pos, line_color, line_color, lines_thickness, zoom);
	}

	if (box_selecting) {
		top_layer->draw_rect(box_selecting_rect, theme_cache.selection_fill);
		top_layer->draw_rect(box_selecting_rect, theme_cache.selection_stroke, false);
	}
}

void GraphEdit::_minimap_draw() {
	if (!is_minimap_enabled()) {
		return;
	}

	minimap->update_minimap();

	// Draw the minimap background.
	Rect2 minimap_rect = Rect2(Point2(), minimap->get_size());
	minimap->draw_style_box(minimap->theme_cache.panel, minimap_rect);

	Vector2 graph_offset = minimap->_get_graph_offset();
	Vector2 minimap_offset = minimap->minimap_offset;

	// Draw graph nodes.
	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *graph_node = Object::cast_to<GraphNode>(get_child(i));
		if (!graph_node || !graph_node->is_visible()) {
			continue;
		}

		Vector2 node_position = minimap->_convert_from_graph_position(graph_node->get_position_offset() * zoom - graph_offset) + minimap_offset;
		Vector2 node_size = minimap->_convert_from_graph_position(graph_node->get_size() * zoom);
		Rect2 node_rect = Rect2(node_position, node_size);

		Ref<StyleBoxFlat> sb_minimap = minimap->theme_cache.node_style->duplicate();

		// Override default values with colors provided by the GraphNode's stylebox, if possible.
		Ref<StyleBoxFlat> sb_frame = graph_node->is_selected() ? graph_node->theme_cache.panel_selected : graph_node->theme_cache.panel;
		if (sb_frame.is_valid()) {
			Color node_color = sb_frame->get_bg_color();
			sb_minimap->set_bg_color(node_color);
		}

		minimap->draw_style_box(sb_minimap, node_rect);
	}

	// Draw node connections.
	for (const Connection &E : connections) {
		Node *from = get_node(NodePath(E.from_node));
		GraphNode *graph_node_from = Object::cast_to<GraphNode>(from);
		if (!graph_node_from) {
			continue;
		}

		Node *node_to = get_node(NodePath(E.to_node));
		GraphNode *graph_node_to = Object::cast_to<GraphNode>(node_to);
		if (!graph_node_to) {
			continue;
		}

		Vector2 from_port_position = graph_node_from->get_position_offset() * zoom + graph_node_from->get_output_port_position(E.from_port) * zoom;
		Vector2 from_position = minimap->_convert_from_graph_position(from_port_position - graph_offset) + minimap_offset;
		Color from_color = graph_node_from->get_output_port_color(E.from_port);
		Vector2 to_port_position = graph_node_to->get_position_offset() * zoom + graph_node_to->get_input_port_position(E.to_port) * zoom;
		Vector2 to_position = minimap->_convert_from_graph_position(to_port_position - graph_offset) + minimap_offset;
		Color to_color = graph_node_to->get_input_port_color(E.to_port);

		if (E.activity > 0) {
			from_color = from_color.lerp(theme_cache.activity_color, E.activity);
			to_color = to_color.lerp(theme_cache.activity_color, E.activity);
		}
		_draw_connection_line(minimap, from_position, to_position, from_color, to_color, 0.5, minimap->_convert_from_graph_position(Vector2(zoom, zoom)).length());
	}

	// Draw the "camera" viewport.
	Rect2 camera_rect = minimap->get_camera_rect();
	minimap->draw_style_box(minimap->theme_cache.camera_style, camera_rect);

	// Draw the resizer control.
	Ref<Texture2D> resizer = minimap->theme_cache.resizer;
	Color resizer_color = minimap->theme_cache.resizer_color;
	minimap->draw_texture(resizer, Point2(), resizer_color);
}

void GraphEdit::set_selected(Node *p_child) {
	for (int i = get_child_count() - 1; i >= 0; i--) {
		GraphNode *graph_node = Object::cast_to<GraphNode>(get_child(i));
		if (!graph_node) {
			continue;
		}

		graph_node->set_selected(graph_node == p_child);
	}
}

void GraphEdit::gui_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());
	if (panner->gui_input(p_ev, warped_panning ? get_global_rect() : Rect2())) {
		return;
	}

	Ref<InputEventMouseMotion> mm = p_ev;

	if (mm.is_valid() && dragging) {
		if (!moving_selection) {
			emit_signal(SNAME("begin_node_move"));
			moving_selection = true;
		}

		just_selected = true;
		drag_accum += mm->get_relative();
		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));
			if (graph_element && graph_element->is_selected() && graph_element->is_draggable()) {
				Vector2 pos = (graph_element->get_drag_from() * zoom + drag_accum) / zoom;

				// Snapping can be toggled temporarily by holding down Ctrl.
				// This is done here as to not toggle the grid when holding down Ctrl.
				if (snapping_enabled ^ Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
					pos = pos.snapped(Vector2(snapping_distance, snapping_distance));
				}

				graph_element->set_position_offset(pos);
			}
		}
	}

	if (mm.is_valid() && box_selecting) {
		box_selecting_to = mm->get_position();

		box_selecting_rect = Rect2(box_selecting_from.min(box_selecting_to), (box_selecting_from - box_selecting_to).abs());

		for (int i = get_child_count() - 1; i >= 0; i--) {
			GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));
			if (!graph_element) {
				continue;
			}

			Rect2 r = graph_element->get_rect();
			bool in_box = r.intersects(box_selecting_rect);

			if (in_box) {
				graph_element->set_selected(box_selection_mode_additive);
			} else {
				graph_element->set_selected(prev_selected.find(graph_element) != nullptr);
			}
		}

		top_layer->queue_redraw();
		minimap->queue_redraw();
	}

	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
			if (box_selecting) {
				box_selecting = false;
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));
					if (!graph_element) {
						continue;
					}

					graph_element->set_selected(prev_selected.find(graph_element) != nullptr);
				}
				top_layer->queue_redraw();
				minimap->queue_redraw();
			} else {
				if (connecting) {
					force_connection_drag_end();
				} else {
					emit_signal(SNAME("popup_request"), mb->get_position());
				}
			}
		}

		if (mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed() && dragging) {
			if (!just_selected && drag_accum == Vector2() && Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
				// Deselect current node.
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));

					if (graph_element) {
						Rect2 r = graph_element->get_rect();
						if (r.has_point(mb->get_position())) {
							graph_element->set_selected(false);
						}
					}
				}
			}

			if (drag_accum != Vector2()) {
				for (int i = get_child_count() - 1; i >= 0; i--) {
					GraphElement *graph_element = Object::cast_to<GraphElement>(get_child(i));
					if (graph_element && graph_element->is_selected()) {
						graph_element->set_drag(false);
					}
				}
			}

			if (moving_selection) {
				emit_signal(SNAME("end_node_move"));
				moving_selection = false;
			}

			dragging = false;

			top_layer->queue_redraw();
			minimap->queue_redraw();
			queue_redraw();
			connections_layer->queue_redraw();
		}

		// Node selection logic.
		if (mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
			GraphElement *graph_element = nullptr;

			// Find node which was clicked on.
			for (int i = get_child_count() - 1; i >= 0; i--) {
				GraphElement *selected_element = Object::cast_to<GraphElement>(get_child(i));

				if (!selected_element) {
					continue;
				}

				if (selected_element->is_resizing()) {
					continue;
				}

				if (selected_element->has_point((mb->get_position() - selected_element->get_position()) / zoom)) {
					graph_element = selected_element;
					break;
				}
			}

			if (graph_element) {
				if (_filter_input(mb->get_position())) {
					return;
				}

				// Left-clicked on a node, select it.
				dragging = true;
				drag_accum = Vector2();
				just_selected = !graph_element->is_selected();
				if (!graph_element->is_selected() && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
					for (int i = 0; i < get_child_count(); i++) {
						GraphElement *child_element = Object::cast_to<GraphElement>(get_child(i));
						if (!child_element) {
							continue;
						}

						child_element->set_selected(child_element == graph_element);
					}
				}

				graph_element->set_selected(true);
				for (int i = 0; i < get_child_count(); i++) {
					GraphElement *child_element = Object::cast_to<GraphElement>(get_child(i));
					if (!child_element) {
						continue;
					}
					if (child_element->is_selected()) {
						child_element->set_drag(true);
					}
				}

			} else {
				if (_filter_input(mb->get_position())) {
					return;
				}
				if (panner->is_panning()) {
					return;
				}

				// Left-clicked on empty space, start box select.
				box_selecting = true;
				box_selecting_from = mb->get_position();
				if (mb->is_command_or_control_pressed()) {
					box_selection_mode_additive = true;
					prev_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphElement *child_element = Object::cast_to<GraphElement>(get_child(i));
						if (!child_element || !child_element->is_selected()) {
							continue;
						}

						prev_selected.push_back(child_element);
					}
				} else if (mb->is_shift_pressed()) {
					box_selection_mode_additive = false;
					prev_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphElement *child_element = Object::cast_to<GraphElement>(get_child(i));
						if (!child_element || !child_element->is_selected()) {
							continue;
						}

						prev_selected.push_back(child_element);
					}
				} else {
					box_selection_mode_additive = true;
					prev_selected.clear();
					for (int i = get_child_count() - 1; i >= 0; i--) {
						GraphElement *child_element = Object::cast_to<GraphElement>(get_child(i));
						if (!child_element) {
							continue;
						}

						child_element->set_selected(false);
					}
				}
			}
		}

		if (mb->get_button_index() == MouseButton::LEFT && !mb->is_pressed() && box_selecting) {
			// Box selection ended. Nodes were selected during mouse movement.
			box_selecting = false;
			box_selecting_rect = Rect2();
			prev_selected.clear();
			top_layer->queue_redraw();
			minimap->queue_redraw();
		}
	}

	if (p_ev->is_pressed()) {
		if (p_ev->is_action("ui_graph_duplicate", true)) {
			emit_signal(SNAME("duplicate_nodes_request"));
			accept_event();
		} else if (p_ev->is_action("ui_copy", true)) {
			emit_signal(SNAME("copy_nodes_request"));
			accept_event();
		} else if (p_ev->is_action("ui_paste", true)) {
			emit_signal(SNAME("paste_nodes_request"));
			accept_event();
		} else if (p_ev->is_action("ui_graph_delete", true)) {
			TypedArray<StringName> nodes;

			for (int i = 0; i < get_child_count(); i++) {
				GraphNode *gn = Object::cast_to<GraphNode>(get_child(i));
				if (!gn) {
					continue;
				}
				if (gn->is_selected()) {
					nodes.push_back(gn->get_name());
				}
			}

			emit_signal(SNAME("delete_nodes_request"), nodes);
			accept_event();
		}
	}
}

void GraphEdit::_pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event) {
	h_scrollbar->set_value(h_scrollbar->get_value() - p_scroll_vec.x);
	v_scrollbar->set_value(v_scrollbar->get_value() - p_scroll_vec.y);
}

void GraphEdit::_zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event) {
	set_zoom_custom(zoom * p_zoom_factor, p_origin);
}

void GraphEdit::set_connection_activity(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port, float p_activity) {
	for (Connection &E : connections) {
		if (E.from_node == p_from && E.from_port == p_from_port && E.to_node == p_to && E.to_port == p_to_port) {
			if (!Math::is_equal_approx(E.activity, p_activity)) {
				// Update only if changed.
				top_layer->queue_redraw();
				minimap->queue_redraw();
				connections_layer->queue_redraw();
			}
			E.activity = p_activity;
			return;
		}
	}
}

void GraphEdit::clear_connections() {
	connections.clear();
	minimap->queue_redraw();
	queue_redraw();
	connections_layer->queue_redraw();
}

void GraphEdit::force_connection_drag_end() {
	ERR_FAIL_COND_MSG(!connecting, "Drag end requested without active drag!");
	connecting = false;
	connecting_valid = false;
	top_layer->queue_redraw();
	minimap->queue_redraw();
	queue_redraw();
	connections_layer->queue_redraw();
	emit_signal(SNAME("connection_drag_ended"));
}

bool GraphEdit::is_node_hover_valid(const StringName &p_from, const int p_from_port, const StringName &p_to, const int p_to_port) {
	bool valid = true;
	GDVIRTUAL_CALL(_is_node_hover_valid, p_from, p_from_port, p_to, p_to_port, valid);
	return valid;
}

void GraphEdit::set_panning_scheme(PanningScheme p_scheme) {
	panning_scheme = p_scheme;
	panner->set_control_scheme((ViewPanner::ControlScheme)p_scheme);
}

GraphEdit::PanningScheme GraphEdit::get_panning_scheme() const {
	return panning_scheme;
}

void GraphEdit::set_zoom(float p_zoom) {
	set_zoom_custom(p_zoom, get_size() / 2);
}

void GraphEdit::set_zoom_custom(float p_zoom, const Vector2 &p_center) {
	p_zoom = CLAMP(p_zoom, zoom_min, zoom_max);
	if (zoom == p_zoom) {
		return;
	}

	Vector2 scrollbar_offset = (Vector2(h_scrollbar->get_value(), v_scrollbar->get_value()) + p_center) / zoom;

	zoom = p_zoom;
	top_layer->queue_redraw();

	zoom_minus_button->set_disabled(zoom == zoom_min);
	zoom_plus_button->set_disabled(zoom == zoom_max);

	_update_scroll();
	minimap->queue_redraw();
	connections_layer->queue_redraw();

	if (is_visible_in_tree()) {
		Vector2 offset = scrollbar_offset * zoom - p_center;
		h_scrollbar->set_value(offset.x);
		v_scrollbar->set_value(offset.y);
	}

	_update_zoom_label();
	queue_redraw();
}

float GraphEdit::get_zoom() const {
	return zoom;
}

void GraphEdit::set_zoom_step(float p_zoom_step) {
	p_zoom_step = abs(p_zoom_step);
	ERR_FAIL_COND(!isfinite(p_zoom_step));
	if (zoom_step == p_zoom_step) {
		return;
	}

	zoom_step = p_zoom_step;
	panner->set_scroll_zoom_factor(zoom_step);
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

TypedArray<Dictionary> GraphEdit::_get_connection_list() const {
	List<Connection> conns;
	get_connection_list(&conns);
	TypedArray<Dictionary> arr;
	for (const Connection &E : conns) {
		Dictionary d;
		d["from_node"] = E.from_node;
		d["from_port"] = E.from_port;
		d["to_node"] = E.to_node;
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
	ConnectionType ct(p_type, p_with_type);
	valid_connection_types.insert(ct);
}

void GraphEdit::remove_valid_connection_type(int p_type, int p_with_type) {
	ConnectionType ct(p_type, p_with_type);
	valid_connection_types.erase(ct);
}

bool GraphEdit::is_valid_connection_type(int p_type, int p_with_type) const {
	ConnectionType ct(p_type, p_with_type);
	return valid_connection_types.has(ct);
}

void GraphEdit::set_snapping_enabled(bool p_enable) {
	if (snapping_enabled == p_enable) {
		return;
	}

	snapping_enabled = p_enable;
	toggle_snapping_button->set_pressed(p_enable);
	queue_redraw();
}

bool GraphEdit::is_snapping_enabled() const {
	return snapping_enabled;
}

void GraphEdit::set_snapping_distance(int p_snapping_distance) {
	ERR_FAIL_COND_MSG(p_snapping_distance < GRID_MIN_SNAPPING_DISTANCE || p_snapping_distance > GRID_MAX_SNAPPING_DISTANCE,
			vformat("GraphEdit's snapping distance must be between %d and %d (inclusive)", GRID_MIN_SNAPPING_DISTANCE, GRID_MAX_SNAPPING_DISTANCE));
	snapping_distance = p_snapping_distance;
	snapping_distance_spinbox->set_value(p_snapping_distance);
	queue_redraw();
}

int GraphEdit::get_snapping_distance() const {
	return snapping_distance;
}

void GraphEdit::set_show_grid(bool p_show) {
	if (show_grid == p_show) {
		return;
	}

	show_grid = p_show;
	toggle_grid_button->set_pressed(p_show);
	queue_redraw();
}

bool GraphEdit::is_showing_grid() const {
	return show_grid;
}

void GraphEdit::_snapping_toggled() {
	snapping_enabled = toggle_snapping_button->is_pressed();
}

void GraphEdit::_snapping_distance_changed(double) {
	snapping_distance = snapping_distance_spinbox->get_value();
	queue_redraw();
}

void GraphEdit::_show_grid_toggled() {
	show_grid = toggle_grid_button->is_pressed();
	queue_redraw();
}

void GraphEdit::set_minimap_size(Vector2 p_size) {
	minimap->set_size(p_size);
	Vector2 minimap_size = minimap->get_size(); // The size might've been adjusted by the minimum size.

	minimap->set_anchors_preset(Control::PRESET_BOTTOM_RIGHT);
	minimap->set_offset(Side::SIDE_LEFT, -minimap_size.width - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_TOP, -minimap_size.height - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_RIGHT, -MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_BOTTOM, -MINIMAP_OFFSET);
	minimap->queue_redraw();
}

Vector2 GraphEdit::get_minimap_size() const {
	return minimap->get_size();
}

void GraphEdit::set_minimap_opacity(float p_opacity) {
	if (minimap->get_modulate().a == p_opacity) {
		return;
	}
	minimap->set_modulate(Color(1, 1, 1, p_opacity));
	minimap->queue_redraw();
}

float GraphEdit::get_minimap_opacity() const {
	Color minimap_modulate = minimap->get_modulate();
	return minimap_modulate.a;
}

void GraphEdit::set_minimap_enabled(bool p_enable) {
	if (minimap_button->is_pressed() == p_enable) {
		return;
	}
	minimap_button->set_pressed(p_enable);
	_minimap_toggled();
	minimap->queue_redraw();
}

bool GraphEdit::is_minimap_enabled() const {
	return minimap_button->is_pressed();
}

void GraphEdit::set_show_menu(bool p_hidden) {
	show_menu = p_hidden;
	menu_panel->set_visible(show_menu);
}

bool GraphEdit::is_showing_menu() const {
	return show_menu;
}

void GraphEdit::set_show_zoom_label(bool p_hidden) {
	show_zoom_label = p_hidden;
	zoom_label->set_visible(show_zoom_label);
}

bool GraphEdit::is_showing_zoom_label() const {
	return show_zoom_label;
}

void GraphEdit::set_show_zoom_buttons(bool p_hidden) {
	show_zoom_buttons = p_hidden;

	zoom_minus_button->set_visible(show_zoom_buttons);
	zoom_reset_button->set_visible(show_zoom_buttons);
	zoom_plus_button->set_visible(show_zoom_buttons);
}

bool GraphEdit::is_showing_zoom_buttons() const {
	return show_zoom_buttons;
}

void GraphEdit::set_show_grid_buttons(bool p_hidden) {
	show_grid_buttons = p_hidden;

	toggle_grid_button->set_visible(show_grid_buttons);
	toggle_snapping_button->set_visible(show_grid_buttons);
	snapping_distance_spinbox->set_visible(show_grid_buttons);
}

bool GraphEdit::is_showing_grid_buttons() const {
	return show_grid_buttons;
}

void GraphEdit::set_show_minimap_button(bool p_hidden) {
	show_minimap_button = p_hidden;
	minimap_button->set_visible(show_minimap_button);
}

bool GraphEdit::is_showing_minimap_button() const {
	return show_minimap_button;
}

void GraphEdit::set_show_arrange_button(bool p_hidden) {
	show_arrange_button = p_hidden;
	arrange_button->set_visible(show_arrange_button);
}

bool GraphEdit::is_showing_arrange_button() const {
	return show_arrange_button;
}

void GraphEdit::_minimap_toggled() {
	if (is_minimap_enabled()) {
		minimap->set_visible(true);
		minimap->queue_redraw();
	} else {
		minimap->set_visible(false);
	}
}

void GraphEdit::set_connection_lines_curvature(float p_curvature) {
	lines_curvature = p_curvature;
	queue_redraw();
}

float GraphEdit::get_connection_lines_curvature() const {
	return lines_curvature;
}

void GraphEdit::set_connection_lines_thickness(float p_thickness) {
	if (lines_thickness == p_thickness) {
		return;
	}
	lines_thickness = p_thickness;
	queue_redraw();
}

float GraphEdit::get_connection_lines_thickness() const {
	return lines_thickness;
}

void GraphEdit::set_connection_lines_antialiased(bool p_antialiased) {
	if (lines_antialiased == p_antialiased) {
		return;
	}
	lines_antialiased = p_antialiased;
	queue_redraw();
}

bool GraphEdit::is_connection_lines_antialiased() const {
	return lines_antialiased;
}

HBoxContainer *GraphEdit::get_menu_hbox() {
	return menu_hbox;
}

Ref<ViewPanner> GraphEdit::get_panner() {
	return panner;
}

void GraphEdit::set_warped_panning(bool p_warped) {
	warped_panning = p_warped;
}

void GraphEdit::arrange_nodes() {
	arranger->arrange_nodes();
}

void GraphEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("connect_node", "from_node", "from_port", "to_node", "to_port"), &GraphEdit::connect_node);
	ClassDB::bind_method(D_METHOD("is_node_connected", "from_node", "from_port", "to_node", "to_port"), &GraphEdit::is_node_connected);
	ClassDB::bind_method(D_METHOD("disconnect_node", "from_node", "from_port", "to_node", "to_port"), &GraphEdit::disconnect_node);
	ClassDB::bind_method(D_METHOD("set_connection_activity", "from_node", "from_port", "to_node", "to_port", "amount"), &GraphEdit::set_connection_activity);
	ClassDB::bind_method(D_METHOD("get_connection_list"), &GraphEdit::_get_connection_list);
	ClassDB::bind_method(D_METHOD("clear_connections"), &GraphEdit::clear_connections);
	ClassDB::bind_method(D_METHOD("force_connection_drag_end"), &GraphEdit::force_connection_drag_end);
	ClassDB::bind_method(D_METHOD("get_scroll_offset"), &GraphEdit::get_scroll_offset);
	ClassDB::bind_method(D_METHOD("set_scroll_offset", "offset"), &GraphEdit::set_scroll_offset);

	ClassDB::bind_method(D_METHOD("add_valid_right_disconnect_type", "type"), &GraphEdit::add_valid_right_disconnect_type);
	ClassDB::bind_method(D_METHOD("remove_valid_right_disconnect_type", "type"), &GraphEdit::remove_valid_right_disconnect_type);
	ClassDB::bind_method(D_METHOD("add_valid_left_disconnect_type", "type"), &GraphEdit::add_valid_left_disconnect_type);
	ClassDB::bind_method(D_METHOD("remove_valid_left_disconnect_type", "type"), &GraphEdit::remove_valid_left_disconnect_type);
	ClassDB::bind_method(D_METHOD("add_valid_connection_type", "from_type", "to_type"), &GraphEdit::add_valid_connection_type);
	ClassDB::bind_method(D_METHOD("remove_valid_connection_type", "from_type", "to_type"), &GraphEdit::remove_valid_connection_type);
	ClassDB::bind_method(D_METHOD("is_valid_connection_type", "from_type", "to_type"), &GraphEdit::is_valid_connection_type);
	ClassDB::bind_method(D_METHOD("get_connection_line", "from_node", "to_node"), &GraphEdit::get_connection_line);

	ClassDB::bind_method(D_METHOD("set_panning_scheme", "scheme"), &GraphEdit::set_panning_scheme);
	ClassDB::bind_method(D_METHOD("get_panning_scheme"), &GraphEdit::get_panning_scheme);

	ClassDB::bind_method(D_METHOD("set_zoom", "zoom"), &GraphEdit::set_zoom);
	ClassDB::bind_method(D_METHOD("get_zoom"), &GraphEdit::get_zoom);

	ClassDB::bind_method(D_METHOD("set_zoom_min", "zoom_min"), &GraphEdit::set_zoom_min);
	ClassDB::bind_method(D_METHOD("get_zoom_min"), &GraphEdit::get_zoom_min);

	ClassDB::bind_method(D_METHOD("set_zoom_max", "zoom_max"), &GraphEdit::set_zoom_max);
	ClassDB::bind_method(D_METHOD("get_zoom_max"), &GraphEdit::get_zoom_max);

	ClassDB::bind_method(D_METHOD("set_zoom_step", "zoom_step"), &GraphEdit::set_zoom_step);
	ClassDB::bind_method(D_METHOD("get_zoom_step"), &GraphEdit::get_zoom_step);

	ClassDB::bind_method(D_METHOD("set_show_grid", "enable"), &GraphEdit::set_show_grid);
	ClassDB::bind_method(D_METHOD("is_showing_grid"), &GraphEdit::is_showing_grid);

	ClassDB::bind_method(D_METHOD("set_snapping_enabled", "enable"), &GraphEdit::set_snapping_enabled);
	ClassDB::bind_method(D_METHOD("is_snapping_enabled"), &GraphEdit::is_snapping_enabled);

	ClassDB::bind_method(D_METHOD("set_snapping_distance", "pixels"), &GraphEdit::set_snapping_distance);
	ClassDB::bind_method(D_METHOD("get_snapping_distance"), &GraphEdit::get_snapping_distance);

	ClassDB::bind_method(D_METHOD("set_connection_lines_curvature", "curvature"), &GraphEdit::set_connection_lines_curvature);
	ClassDB::bind_method(D_METHOD("get_connection_lines_curvature"), &GraphEdit::get_connection_lines_curvature);

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

	ClassDB::bind_method(D_METHOD("set_show_menu", "hidden"), &GraphEdit::set_show_menu);
	ClassDB::bind_method(D_METHOD("is_showing_menu"), &GraphEdit::is_showing_menu);

	ClassDB::bind_method(D_METHOD("set_show_zoom_label", "enable"), &GraphEdit::set_show_zoom_label);
	ClassDB::bind_method(D_METHOD("is_showing_zoom_label"), &GraphEdit::is_showing_zoom_label);

	ClassDB::bind_method(D_METHOD("set_show_grid_buttons", "hidden"), &GraphEdit::set_show_grid_buttons);
	ClassDB::bind_method(D_METHOD("is_showing_grid_buttons"), &GraphEdit::is_showing_grid_buttons);

	ClassDB::bind_method(D_METHOD("set_show_zoom_buttons", "hidden"), &GraphEdit::set_show_zoom_buttons);
	ClassDB::bind_method(D_METHOD("is_showing_zoom_buttons"), &GraphEdit::is_showing_zoom_buttons);

	ClassDB::bind_method(D_METHOD("set_show_minimap_button", "hidden"), &GraphEdit::set_show_minimap_button);
	ClassDB::bind_method(D_METHOD("is_showing_minimap_button"), &GraphEdit::is_showing_minimap_button);

	ClassDB::bind_method(D_METHOD("set_show_arrange_button", "hidden"), &GraphEdit::set_show_arrange_button);
	ClassDB::bind_method(D_METHOD("is_showing_arrange_button"), &GraphEdit::is_showing_arrange_button);

	ClassDB::bind_method(D_METHOD("set_right_disconnects", "enable"), &GraphEdit::set_right_disconnects);
	ClassDB::bind_method(D_METHOD("is_right_disconnects_enabled"), &GraphEdit::is_right_disconnects_enabled);

	GDVIRTUAL_BIND(_is_in_input_hotzone, "in_node", "in_port", "mouse_position");
	GDVIRTUAL_BIND(_is_in_output_hotzone, "in_node", "in_port", "mouse_position");

	ClassDB::bind_method(D_METHOD("get_menu_hbox"), &GraphEdit::get_menu_hbox);

	ClassDB::bind_method(D_METHOD("arrange_nodes"), &GraphEdit::arrange_nodes);

	ClassDB::bind_method(D_METHOD("set_selected", "node"), &GraphEdit::set_selected);

	GDVIRTUAL_BIND(_get_connection_line, "from_position", "to_position")
	GDVIRTUAL_BIND(_is_node_hover_valid, "from_node", "from_port", "to_node", "to_port");

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scroll_offset", PROPERTY_HINT_NONE, "suffix:px"), "set_scroll_offset", "get_scroll_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_grid"), "set_show_grid", "is_showing_grid");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "snapping_enabled"), "set_snapping_enabled", "is_snapping_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "snapping_distance", PROPERTY_HINT_NONE, "suffix:px"), "set_snapping_distance", "get_snapping_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "panning_scheme", PROPERTY_HINT_ENUM, "Scroll Zooms,Scroll Pans"), "set_panning_scheme", "get_panning_scheme");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "right_disconnects"), "set_right_disconnects", "is_right_disconnects_enabled");

	ADD_GROUP("Connection Lines", "connection_lines");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "connection_lines_curvature"), "set_connection_lines_curvature", "get_connection_lines_curvature");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "connection_lines_thickness", PROPERTY_HINT_NONE, "suffix:px"), "set_connection_lines_thickness", "get_connection_lines_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "connection_lines_antialiased"), "set_connection_lines_antialiased", "is_connection_lines_antialiased");

	ADD_GROUP("Zoom", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom"), "set_zoom", "get_zoom");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom_min"), "set_zoom_min", "get_zoom_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom_max"), "set_zoom_max", "get_zoom_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "zoom_step"), "set_zoom_step", "get_zoom_step");

	ADD_GROUP("Minimap", "minimap_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "minimap_enabled"), "set_minimap_enabled", "is_minimap_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "minimap_size", PROPERTY_HINT_NONE, "suffix:px"), "set_minimap_size", "get_minimap_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "minimap_opacity"), "set_minimap_opacity", "get_minimap_opacity");

	ADD_GROUP("Toolbar Menu", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_menu"), "set_show_menu", "is_showing_menu");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_zoom_label"), "set_show_zoom_label", "is_showing_zoom_label");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_zoom_buttons"), "set_show_zoom_buttons", "is_showing_zoom_buttons");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_grid_buttons"), "set_show_grid_buttons", "is_showing_grid_buttons");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_minimap_button"), "set_show_minimap_button", "is_showing_minimap_button");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_arrange_button"), "set_show_arrange_button", "is_showing_arrange_button");

	ADD_SIGNAL(MethodInfo("connection_request", PropertyInfo(Variant::STRING_NAME, "from_node"), PropertyInfo(Variant::INT, "from_port"), PropertyInfo(Variant::STRING_NAME, "to_node"), PropertyInfo(Variant::INT, "to_port")));
	ADD_SIGNAL(MethodInfo("disconnection_request", PropertyInfo(Variant::STRING_NAME, "from_node"), PropertyInfo(Variant::INT, "from_port"), PropertyInfo(Variant::STRING_NAME, "to_node"), PropertyInfo(Variant::INT, "to_port")));
	ADD_SIGNAL(MethodInfo("connection_to_empty", PropertyInfo(Variant::STRING_NAME, "from_node"), PropertyInfo(Variant::INT, "from_port"), PropertyInfo(Variant::VECTOR2, "release_position")));
	ADD_SIGNAL(MethodInfo("connection_from_empty", PropertyInfo(Variant::STRING_NAME, "to_node"), PropertyInfo(Variant::INT, "to_port"), PropertyInfo(Variant::VECTOR2, "release_position")));
	ADD_SIGNAL(MethodInfo("connection_drag_started", PropertyInfo(Variant::STRING_NAME, "from_node"), PropertyInfo(Variant::INT, "from_port"), PropertyInfo(Variant::BOOL, "is_output")));
	ADD_SIGNAL(MethodInfo("connection_drag_ended"));

	ADD_SIGNAL(MethodInfo("copy_nodes_request"));
	ADD_SIGNAL(MethodInfo("paste_nodes_request"));
	ADD_SIGNAL(MethodInfo("duplicate_nodes_request"));
	ADD_SIGNAL(MethodInfo("delete_nodes_request", PropertyInfo(Variant::ARRAY, "nodes", PROPERTY_HINT_ARRAY_TYPE, "StringName")));

	ADD_SIGNAL(MethodInfo("node_selected", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("node_deselected", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));

	ADD_SIGNAL(MethodInfo("popup_request", PropertyInfo(Variant::VECTOR2, "position")));

	ADD_SIGNAL(MethodInfo("begin_node_move"));
	ADD_SIGNAL(MethodInfo("end_node_move"));
	ADD_SIGNAL(MethodInfo("scroll_offset_changed", PropertyInfo(Variant::VECTOR2, "offset")));

	BIND_ENUM_CONSTANT(SCROLL_ZOOMS);
	BIND_ENUM_CONSTANT(SCROLL_PANS);

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphEdit, panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphEdit, grid_major);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphEdit, grid_minor);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, GraphEdit, activity_color, "activity");
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphEdit, selection_fill);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphEdit, selection_stroke);

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphEdit, menu_panel);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, zoom_in);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, zoom_out);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, zoom_reset);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, snapping_toggle);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, grid_toggle);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, minimap_toggle);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphEdit, layout);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphEdit, port_hotzone_inner_extent);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphEdit, port_hotzone_outer_extent);
}

GraphEdit::GraphEdit() {
	set_focus_mode(FOCUS_ALL);

	// Allow dezooming 8 times from the default zoom level.
	// At low zoom levels, text is unreadable due to its small size and poor filtering,
	// but this is still useful for previewing and navigation.
	zoom_min = (1 / Math::pow(zoom_step, 8));
	// Allow zooming 4 times from the default zoom level.
	zoom_max = (1 * Math::pow(zoom_step, 4));

	panner.instantiate();
	panner->set_callbacks(callable_mp(this, &GraphEdit::_pan_callback), callable_mp(this, &GraphEdit::_zoom_callback));

	top_layer = memnew(GraphEditFilter(this));
	add_child(top_layer, false, INTERNAL_MODE_BACK);
	top_layer->set_mouse_filter(MOUSE_FILTER_PASS);
	top_layer->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	top_layer->connect("draw", callable_mp(this, &GraphEdit::_top_layer_draw));
	top_layer->connect("gui_input", callable_mp(this, &GraphEdit::_top_layer_input));
	top_layer->connect("focus_exited", callable_mp(panner.ptr(), &ViewPanner::release_pan_key));

	connections_layer = memnew(Control);
	add_child(connections_layer, false, INTERNAL_MODE_FRONT);
	connections_layer->connect("draw", callable_mp(this, &GraphEdit::_connections_layer_draw));
	connections_layer->set_name("_connection_layer");
	connections_layer->set_disable_visibility_clip(true); // Necessary, so it can draw freely and be offset.
	connections_layer->set_mouse_filter(MOUSE_FILTER_IGNORE);

	h_scrollbar = memnew(HScrollBar);
	h_scrollbar->set_name("_h_scroll");
	top_layer->add_child(h_scrollbar);

	v_scrollbar = memnew(VScrollBar);
	v_scrollbar->set_name("_v_scroll");
	top_layer->add_child(v_scrollbar);

	// Set large minmax so it can scroll even if not resized yet.
	h_scrollbar->set_min(-10000);
	h_scrollbar->set_max(10000);

	v_scrollbar->set_min(-10000);
	v_scrollbar->set_max(10000);

	h_scrollbar->connect("value_changed", callable_mp(this, &GraphEdit::_scroll_moved));
	v_scrollbar->connect("value_changed", callable_mp(this, &GraphEdit::_scroll_moved));

	// Toolbar menu.

	menu_panel = memnew(PanelContainer);
	menu_panel->set_visible(show_menu);
	top_layer->add_child(menu_panel);
	menu_panel->set_position(Vector2(10, 10));

	menu_hbox = memnew(HBoxContainer);
	menu_panel->add_child(menu_hbox);

	// Zoom label and controls.

	zoom_label = memnew(Label);
	zoom_label->set_visible(show_zoom_label);
	zoom_label->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	zoom_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	zoom_label->set_custom_minimum_size(Size2(48, 0));
	menu_hbox->add_child(zoom_label);
	_update_zoom_label();

	zoom_minus_button = memnew(Button);
	zoom_minus_button->set_theme_type_variation("FlatButton");
	zoom_minus_button->set_visible(show_zoom_buttons);
	zoom_minus_button->set_tooltip_text(RTR("Zoom Out"));
	zoom_minus_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(zoom_minus_button);
	zoom_minus_button->connect("pressed", callable_mp(this, &GraphEdit::_zoom_minus));

	zoom_reset_button = memnew(Button);
	zoom_reset_button->set_theme_type_variation("FlatButton");
	zoom_reset_button->set_visible(show_zoom_buttons);
	zoom_reset_button->set_tooltip_text(RTR("Zoom Reset"));
	zoom_reset_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(zoom_reset_button);
	zoom_reset_button->connect("pressed", callable_mp(this, &GraphEdit::_zoom_reset));

	zoom_plus_button = memnew(Button);
	zoom_plus_button->set_theme_type_variation("FlatButton");
	zoom_plus_button->set_visible(show_zoom_buttons);
	zoom_plus_button->set_tooltip_text(RTR("Zoom In"));
	zoom_plus_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(zoom_plus_button);
	zoom_plus_button->connect("pressed", callable_mp(this, &GraphEdit::_zoom_plus));

	// Grid controls.

	toggle_grid_button = memnew(Button);
	toggle_grid_button->set_theme_type_variation("FlatButton");
	toggle_grid_button->set_visible(show_grid_buttons);
	toggle_grid_button->set_toggle_mode(true);
	toggle_grid_button->set_pressed(true);
	toggle_grid_button->set_tooltip_text(RTR("Toggle the visual grid."));
	toggle_grid_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(toggle_grid_button);
	toggle_grid_button->connect("pressed", callable_mp(this, &GraphEdit::_show_grid_toggled));

	toggle_snapping_button = memnew(Button);
	toggle_snapping_button->set_theme_type_variation("FlatButton");
	toggle_snapping_button->set_visible(show_grid_buttons);
	toggle_snapping_button->set_toggle_mode(true);
	toggle_snapping_button->set_tooltip_text(RTR("Toggle snapping to the grid."));
	toggle_snapping_button->set_pressed(snapping_enabled);
	toggle_snapping_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(toggle_snapping_button);
	toggle_snapping_button->connect("pressed", callable_mp(this, &GraphEdit::_snapping_toggled));

	snapping_distance_spinbox = memnew(SpinBox);
	snapping_distance_spinbox->set_visible(show_grid_buttons);
	snapping_distance_spinbox->set_min(GRID_MIN_SNAPPING_DISTANCE);
	snapping_distance_spinbox->set_max(GRID_MAX_SNAPPING_DISTANCE);
	snapping_distance_spinbox->set_step(1);
	snapping_distance_spinbox->set_value(snapping_distance);
	snapping_distance_spinbox->set_tooltip_text(RTR("Change the snapping distance."));
	menu_hbox->add_child(snapping_distance_spinbox);
	snapping_distance_spinbox->connect("value_changed", callable_mp(this, &GraphEdit::_snapping_distance_changed));

	// Extra controls.

	minimap_button = memnew(Button);
	minimap_button->set_theme_type_variation("FlatButton");
	minimap_button->set_visible(show_minimap_button);
	minimap_button->set_toggle_mode(true);
	minimap_button->set_tooltip_text(RTR("Toggle the graph minimap."));
	minimap_button->set_pressed(show_grid);
	minimap_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(minimap_button);
	minimap_button->connect("pressed", callable_mp(this, &GraphEdit::_minimap_toggled));

	arrange_button = memnew(Button);
	arrange_button->set_theme_type_variation("FlatButton");
	arrange_button->set_visible(show_arrange_button);
	arrange_button->connect("pressed", callable_mp(this, &GraphEdit::arrange_nodes));
	arrange_button->set_focus_mode(FOCUS_NONE);
	menu_hbox->add_child(arrange_button);
	arrange_button->set_tooltip_text(RTR("Automatically arrange selected nodes."));

	// Minimap.

	const Vector2 minimap_size = Vector2(240, 160);
	const float minimap_opacity = 0.65;

	minimap = memnew(GraphEditMinimap(this));
	top_layer->add_child(minimap);
	minimap->set_name("_minimap");
	minimap->set_modulate(Color(1, 1, 1, minimap_opacity));
	minimap->set_mouse_filter(MOUSE_FILTER_PASS);
	minimap->set_custom_minimum_size(Vector2(50, 50));
	minimap->set_size(minimap_size);
	minimap->set_anchors_preset(Control::PRESET_BOTTOM_RIGHT);
	minimap->set_offset(Side::SIDE_LEFT, -minimap_size.width - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_TOP, -minimap_size.height - MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_RIGHT, -MINIMAP_OFFSET);
	minimap->set_offset(Side::SIDE_BOTTOM, -MINIMAP_OFFSET);
	minimap->connect("draw", callable_mp(this, &GraphEdit::_minimap_draw));

	set_clip_contents(true);

	arranger = Ref<GraphEditArranger>(memnew(GraphEditArranger(this)));
}
