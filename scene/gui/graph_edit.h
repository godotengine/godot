/**************************************************************************/
/*  graph_edit.h                                                          */
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

#pragma once

#include "core/templates/vector.h"
#include "core/variant/typed_dictionary.h"
#include "scene/gui/box_container.h"
#include "scene/gui/graph_connection.h"
#include "scene/gui/graph_frame.h"
#include "scene/gui/graph_node_indexed.h"
#include "scene/gui/graph_port.h"

class Button;
class GraphEdit;
class GraphEditArranger;
class HScrollBar;
class Label;
class Line2D;
class PanelContainer;
class SpinBox;
class ViewPanner;
class VScrollBar;

class GraphEditFilter : public Control {
	GDCLASS(GraphEditFilter, Control);

	friend class GraphEdit;
	friend class GraphEditMinimap;

	GraphEdit *ge = nullptr;

	virtual bool has_point(const Point2 &p_point) const override;

public:
	GraphEditFilter(GraphEdit *p_edit);
};

class GraphEditMinimap : public Control {
	GDCLASS(GraphEditMinimap, Control);

	friend class GraphEdit;
	friend class GraphEditFilter;

	GraphEdit *ge = nullptr;

	Vector2 minimap_padding;
	Vector2 minimap_offset;
	Vector2 graph_proportions = Vector2(1, 1);
	Vector2 graph_padding = Vector2(0, 0);
	Vector2 camera_position = Vector2(100, 50);
	Vector2 camera_size = Vector2(200, 200);

	bool is_pressing = false;
	bool is_resizing = false;

	struct ThemeCache {
		Ref<StyleBox> panel;
		Ref<StyleBox> node_style;
		Ref<StyleBox> camera_style;

		Ref<Texture2D> resizer;
		Color resizer_color;
	} theme_cache;

	Vector2 _get_render_size();
	Vector2 _get_graph_offset();
	Vector2 _get_graph_size();

	Vector2 _convert_from_graph_position(const Vector2 &p_position);
	Vector2 _convert_to_graph_position(const Vector2 &p_position);

	virtual void gui_input(const Ref<InputEvent> &p_ev) override;

	void _adjust_graph_scroll(const Vector2 &p_offset);

protected:
	static void _bind_methods();

public:
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	void update_minimap();
	Rect2 get_camera_rect();

	GraphEditMinimap(GraphEdit *p_edit);
};

class GraphEdit : public Control {
	GDCLASS(GraphEdit, Control);

public:
	// Should be in sync with ControlScheme in ViewPanner.
	enum PanningScheme {
		SCROLL_ZOOMS,
		SCROLL_PANS,
	};

	enum GridPattern {
		GRID_PATTERN_LINES,
		GRID_PATTERN_DOTS
	};

private:
	Label *zoom_label = nullptr;
	Button *zoom_minus_button = nullptr;
	Button *zoom_reset_button = nullptr;
	Button *zoom_plus_button = nullptr;

	Button *toggle_snapping_button = nullptr;
	SpinBox *snapping_distance_spinbox = nullptr;
	SpinBox *snapping_distance_spinbox_vertical = nullptr;
	Button *toggle_grid_button = nullptr;
	Button *minimap_button = nullptr;
	Button *arrange_button = nullptr;

	HScrollBar *h_scrollbar = nullptr;
	VScrollBar *v_scrollbar = nullptr;

	Ref<ViewPanner> panner;
	bool warped_panning = true;

	bool allow_self_connection = false;

	bool show_menu = true;
	bool show_zoom_label = false;
	bool show_grid_buttons = true;
	bool show_zoom_buttons = true;
	bool show_minimap_button = true;
	bool show_arrange_button = true;

	bool snapping_enabled = true;
	Vector2i snapping_distance = Vector2i(20, 20);
	bool separate_snapping_distances = false;
	bool show_grid = true;
	GridPattern grid_pattern = GRID_PATTERN_LINES;

	bool keyboard_connecting = false;
	bool connecting = false;
	bool connecting_valid = false;
	bool connecting_target_valid = false;
	const GraphPort *connecting_from_port = nullptr;
	Vector2 connecting_to_point; // In local screen space.
	const GraphPort *connecting_to_port = nullptr;

	bool just_disconnected = false;

	Vector2 click_pos;

	PanningScheme panning_scheme = SCROLL_ZOOMS;
	bool dragging = false;
	bool just_selected = false;
	bool moving_selection = false;
	Vector2 drag_accum;

	float zoom = 1.0;
	float zoom_step = 1.2;
	// Proper values set in constructor.
	float zoom_min = 0.0;
	float zoom_max = 0.0;

	Vector2 min_scroll_offset;
	Vector2 max_scroll_offset;
	Vector2 scroll_offset;

	bool box_selecting = false;
	bool box_selection_mode_additive = false;
	Point2 box_selecting_from;
	Point2 box_selecting_to;
	Rect2 box_selecting_rect;
	List<GraphElement *> prev_selected;

	bool setting_scroll_offset = false;
	bool input_disconnects = true;
	bool updating = false;
	bool awaiting_scroll_offset_update = false;

	TypedArray<Ref<GraphConnection>> graph_connections;
	HashMap<const GraphPort *, TypedArray<Ref<GraphConnection>>> connection_map;
	Ref<GraphConnection> hovered_connection;

	float lines_thickness = 4.0f;
	float lines_curvature = 0.5f;
	bool lines_antialiased = true;

	PanelContainer *menu_panel = nullptr;
	HBoxContainer *menu_hbox = nullptr;
	Control *connections_layer = nullptr;

	GraphEditFilter *top_connection_layer = nullptr; // Draws a dragged connection. Necessary since the connection line shader can't be applied to the whole top layer.
	Line2D *dragged_connection_line = nullptr;
	Control *top_layer = nullptr; // Used for drawing the box selection rect. Contains the minimap, menu panel and the scrollbars.

	GraphEditMinimap *minimap = nullptr;

	static Ref<Shader> default_connections_shader;
	Ref<Shader> connections_shader;

	Ref<GraphEditArranger> arranger;

	HashSet<GraphConnection::ConnectionType, GraphConnection::ConnectionType> valid_connection_types;
	HashSet<int> valid_input_disconnect_types;
	HashSet<int> valid_output_disconnect_types;
	HashSet<int> valid_undirected_disconnect_types;

	struct ThemeCache {
		float base_scale = 1.0;

		Ref<StyleBox> panel;
		Ref<StyleBox> panel_focus;
		Color grid_major;
		Color grid_minor;

		Color activity_color;
		Color connection_hover_tint_color;
		int connection_hover_thickness;
		Color connection_valid_target_tint_color;
		Color connection_rim_color;

		Color selection_fill;
		Color selection_stroke;

		Ref<StyleBox> menu_panel;

		Ref<Texture2D> zoom_in;
		Ref<Texture2D> zoom_out;
		Ref<Texture2D> zoom_reset;

		Ref<Texture2D> snapping_toggle;
		Ref<Texture2D> grid_toggle;
		Ref<Texture2D> minimap_toggle;
		Ref<Texture2D> layout;
	} theme_cache;

	// This separates the children in two layers to ensure the order
	// of both background nodes (e.g frame nodes) and foreground nodes (connectable nodes).
	int background_nodes_separator_idx = 0;

	HashMap<StringName, HashSet<StringName>> frame_attached_nodes;
	HashMap<StringName, StringName> linked_parent_map;

	Dictionary type_names;
	TypedArray<Color> type_colors;

	void _pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event);
	void _zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event);

	void _zoom_minus();
	void _zoom_reset();
	void _zoom_plus();
	void _update_zoom_label();

	void _graph_element_selected(Node *p_node);
	void _graph_element_deselected(Node *p_node);
	void _graph_element_resize_request(const Vector2 &p_new_minsize, Node *p_node);
	void _graph_frame_autoshrink_changed(const Vector2 &p_new_minsize, GraphFrame *p_frame);
	void _graph_element_moved(Node *p_node);
	void _graph_node_ports_updated(GraphNode *p_node);
	void _graph_node_rect_changed(GraphNode *p_node);

	void _ensure_node_order_from_root(const StringName &p_node);
	void _ensure_node_order_from(Node *p_node);

	void _update_scrollbars();
	void _update_scroll_offset();
	void _scrollbar_moved(double);
	virtual void gui_input(const Ref<InputEvent> &p_ev) override;
	void _top_connection_layer_input(const Ref<InputEvent> &p_ev);

	float _get_shader_line_width() const;
	void _draw_minimap_connection_line(const Vector2 &p_from_graph_position, const Vector2 &p_to_graph_position, const Color &p_from_color, const Color &p_to_color);
	void _invalidate_connection_line_cache();
	void _invalidate_graph_node_connections(GraphNode *p_node);
	void _update_top_connection_layer();
	void _update_connections();

	void _top_layer_draw();
	void _minimap_draw();
	void _draw_grid();
	void _validate_property(PropertyInfo &p_property) const;

	const TypedArray<Ref<GraphConnection>> _get_connections() const;
	Ref<GraphConnection> _get_closest_connection_at_point(const Vector2 &p_point, float p_max_distance = 4.0) const;
	TypedArray<Ref<GraphConnection>> _get_connections_intersecting_with_rect(const Rect2 &p_rect) const;
	TypedArray<Ref<GraphConnection>> _get_connections_by_node(const GraphNode *p_node) const;
	TypedArray<Ref<GraphConnection>> _get_connections_by_port(const GraphPort *p_port) const;
	Error _add_connection(Ref<GraphConnection> p_connection);
	bool _is_connection_valid(const GraphPort *p_port) const;
	void _mark_connections_dirty_by_port(const GraphPort *p_port);

	Rect2 _compute_shrinked_frame_rect(const GraphFrame *p_frame);
	void _set_drag_frame_attached_nodes(GraphFrame *p_frame, bool p_drag);
	void _set_position_of_frame_attached_nodes(GraphFrame *p_frame, const Vector2 &p_pos);

	friend class GraphEditFilter;
	bool _filter_input(const Point2 &p_point);
	void _snapping_toggled();
	void _snapping_distance_changed(double);
	void _snapping_distance_vertical_changed(double);
	void _show_grid_toggled();

	friend class GraphEditMinimap;
	void _minimap_toggled();

	bool _check_clickable_control(const Control *p_control, const Vector2 &r_mouse_pos, const Vector2 &p_offset) const;

#ifndef DISABLE_DEPRECATED
	bool _is_arrange_nodes_button_hidden_bind_compat_81582() const;
	void _set_arrange_nodes_button_hidden_bind_compat_81582(bool p_enable);
	PackedVector2Array _get_connection_line_bind_compat_86158(const Vector2 &p_from, const Vector2 &p_to);
	Ref<GraphConnection> _connect_node_bind_compat_97449(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);
	bool _is_in_input_hotzone_bind_compat_108099(Object *in_node, int in_port, Vector2 mouse_position);
	bool _is_in_output_hotzone_bind_compat_108099(Object *in_node, int in_port, Vector2 mouse_position);
	void _add_valid_left_disconnect_type_bind_compat_108099(int type);
	void _add_valid_right_disconnect_type_bind_compat_108099(int type);
	Error _connect_node_bind_compat_108099(StringName from_node, int from_port, StringName to_node, int to_port, bool keep_alive = false);
	void _disconnect_node_bind_compat_108099(StringName from_node, int from_port, StringName to_node, int to_port);
	Array _get_connection_list_bind_compat_108099();
	bool _is_right_disconnects_enabled_bind_compat_108099();
	void _remove_valid_left_disconnect_type_bind_compat_108099(int type);
	void _remove_valid_right_disconnect_type_bind_compat_108099(int type);
	void _set_right_disconnects_bind_compat_108099(bool value);
	// properties: right_disconnects
	// signals: connection_from_empty
#endif

protected:
	virtual void _update_theme_item_cache() override;

	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

	void _notification(int p_what);
	static void _bind_methods();
#ifndef DISABLE_DEPRECATED
	static void _bind_compatibility_methods();
#endif

	bool is_in_port_hotzone(const GraphPort *p_port, const Vector2 &p_local_pos) const;

	GDVIRTUAL2RC(Vector<Vector2>, _get_connection_line, Vector2, Vector2)
	GDVIRTUAL2R(bool, _is_in_port_hotzone, const GraphPort *, Vector2)
	GDVIRTUAL2R(bool, _is_node_hover_valid, const GraphPort *, const GraphPort *);

public:
	static void init_shaders();
	static void finish_shaders();

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	PackedStringArray get_configuration_warnings() const override;

	void key_input(const Ref<InputEvent> &p_ev);

	// This method has to be public (for undo redo).
	// TODO: Find a better way to do this.
	void _update_graph_frame(GraphFrame *p_frame);

	// Connection related methods.
	Ref<GraphConnection> connect_nodes(GraphPort *p_first_port, GraphPort *p_second_port, bool p_clear_if_invalid = true);
	Ref<GraphConnection> connect_nodes_indexed(String p_first_node, int p_first_port, String p_second_node, int p_second_port, bool p_clear_if_invalid = true);
	Ref<GraphConnection> connect_nodes_indexed_legacy(String p_from_node, int p_from_port, String p_to_node, int p_to_port, bool p_keep_alive = false);
	Error add_connection(Ref<GraphConnection> p_connection);
	void disconnect_nodes_indexed(String p_first_node, int p_first_port, String p_second_node, int p_second_port);
	void disconnect_nodes_indexed_legacy(String p_from_node, int p_from_port, String p_to_node, int p_to_port);
	void remove_connection(Ref<GraphConnection> p_connection);
	void disconnect_nodes(GraphPort *p_first_port, GraphPort *p_second_port);
	bool are_nodes_connected(const GraphNode *p_first_node, const GraphNode *p_second_node) const;
	bool are_ports_connected(const GraphPort *p_first_port, const GraphPort *p_second_port) const;
	bool is_node_connected(const GraphNode *p_node) const;
	bool is_port_connected(const GraphPort *p_port) const;
	TypedArray<GraphPort> get_connected_ports(const GraphPort *p_port) const;
	TypedArray<GraphNode> get_connected_nodes(const GraphNode *p_node) const;
	int get_connection_count_by_port(const GraphPort *p_port) const;
	int get_connection_count_by_node(const GraphNode *p_node) const;
	GraphNode *get_connection_target(const GraphPort *p_port) const;
	String get_connections_description(const GraphPort *p_port) const;

	void set_connections(const TypedArray<Ref<GraphConnection>> p_connections);
	void set_port_connections(const GraphPort *p_port, const TypedArray<Ref<GraphConnection>> p_connections);
	void set_node_connections(const GraphNode *p_node, const TypedArray<Ref<GraphConnection>> p_connections);
	void add_connections(const TypedArray<Ref<GraphConnection>> p_connections);
	TypedArray<Ref<GraphConnection>> get_connections() const;
	TypedArray<Ref<GraphConnection>> get_connections_by_port(const GraphPort *p_port) const;
	TypedArray<Ref<GraphConnection>> get_connections_by_node(const GraphNode *p_node) const;
	Ref<GraphConnection> get_first_connection_by_port(const GraphPort *p_port) const;
	GraphPort *get_first_connected_port(const GraphPort *p_port) const;
	GraphNode *get_first_connected_node(const GraphPort *p_port) const;
	TypedArray<Ref<GraphConnection>> get_filtered_connections_by_node(const GraphNode *p_node, GraphPort::PortDirection p_filter_direction) const;
	void move_connections(GraphPort *p_from_port, GraphPort *p_to_port);
	void clear_connections();
	void clear_port_connections(const GraphPort *p_port);
	void clear_node_connections(const GraphNode *p_node);
	Ref<GraphConnection> get_connection(const GraphPort *p_first_port, const GraphPort *p_second_port) const;
	virtual PackedVector2Array get_connection_line(const Vector2 &p_from, const Vector2 &p_to) const;
	Ref<GraphConnection> get_closest_connection_at_point(const Vector2 &p_point, float p_max_distance = 4.0) const;
	TypedArray<Ref<GraphConnection>> get_connections_intersecting_with_rect(const Rect2 &p_rect) const;
	void force_connection_drag_end();

	bool is_keyboard_connecting() const { return keyboard_connecting; }
	void start_connecting(const GraphPort *p_port, bool is_keyboard);
	void end_connecting(const GraphPort *p_port, bool is_keyboard);

	Dictionary get_type_names() const;
	void set_type_names(const Dictionary &p_names);

	const TypedArray<Color> &get_type_colors();
	void set_type_colors(const TypedArray<Color> &p_type_colors);

	virtual bool is_node_hover_valid(const GraphPort *p_first_port, const GraphPort *p_second_port);

	void set_connection_activity(Ref<GraphConnection> p_conn, float p_activity);
	void set_connection_activity_indexed_legacy(String p_first_node, int p_first_port, String p_second_node, int p_second_port, float p_activity);
	void reset_all_connection_activity();
	void add_valid_connection_type(int p_type, int p_with_type);
	void remove_valid_connection_type(int p_type, int p_with_type);
	bool is_valid_connection_type(int p_type, int p_with_type) const;

	// GraphFrame related methods.
	void attach_graph_element_to_frame(const StringName &p_graph_element, const StringName &p_parent_frame);
	void detach_graph_element_from_frame(const StringName &p_graph_element);
	GraphFrame *get_element_frame(const StringName &p_attached_graph_element) const;
	TypedArray<StringName> get_attached_nodes_of_frame(const StringName &p_graph_frame) const;

	void set_panning_scheme(PanningScheme p_scheme);
	PanningScheme get_panning_scheme() const;

	void set_zoom(float p_zoom);
	void set_zoom_custom(float p_zoom, const Vector2 &p_center);
	float get_zoom() const;

	void set_zoom_min(float p_zoom_min);
	float get_zoom_min() const;

	void set_zoom_max(float p_zoom_max);
	float get_zoom_max() const;

	void set_zoom_step(float p_zoom_step);
	float get_zoom_step() const;

	void set_minimap_size(Vector2 p_size);
	Vector2 get_minimap_size() const;
	void set_minimap_opacity(float p_opacity);
	float get_minimap_opacity() const;

	void set_minimap_enabled(bool p_enable);
	bool is_minimap_enabled() const;

	void set_show_menu(bool p_hidden);
	bool is_showing_menu() const;
	void set_show_zoom_label(bool p_hidden);
	bool is_showing_zoom_label() const;
	void set_show_grid_buttons(bool p_hidden);
	bool is_showing_grid_buttons() const;
	void set_show_zoom_buttons(bool p_hidden);
	bool is_showing_zoom_buttons() const;
	void set_show_minimap_button(bool p_hidden);
	bool is_showing_minimap_button() const;
	void set_show_arrange_button(bool p_hidden);
	bool is_showing_arrange_button() const;

	Control *get_top_layer() const { return top_layer; }
	GraphEditMinimap *get_minimap() const { return minimap; }

	void override_connections_shader(const Ref<Shader> &p_shader);

	void set_input_disconnects(bool p_enable);
	bool is_input_disconnects_enabled() const;

	void set_allow_self_connection(bool p_allowed);
	bool is_self_connection_allowed() const;

	void add_valid_input_disconnect_type(int p_type);
	void remove_valid_input_disconnect_type(int p_type);

	void add_valid_output_disconnect_type(int p_type);
	void remove_valid_output_disconnect_type(int p_type);

	void add_valid_undirected_disconnect_type(int p_type);
	void remove_valid_undirected_disconnect_type(int p_type);

	void set_scroll_offset(const Vector2 &p_ofs);
	Vector2 get_scroll_offset() const;

	void set_selected(Node *p_child);

	void set_snapping_enabled(bool p_enable);
	bool is_snapping_enabled() const;

	void set_snapping_distance(int p_snapping_distance);
	void set_snapping_distances(Vector2i p_snapping_distances);
	int get_snapping_distance() const;
	Vector2i get_snapping_distances() const;

	void set_separate_snapping_distances(bool p_enable);
	bool is_separate_snapping_distances() const;

	void set_show_grid(bool p_enable);
	bool is_showing_grid() const;

	void set_grid_pattern(GridPattern p_pattern);
	GridPattern get_grid_pattern() const;

	void set_connection_lines_curvature(float p_curvature);
	float get_connection_lines_curvature() const;

	void set_connection_lines_thickness(float p_thickness);
	float get_connection_lines_thickness() const;

	void set_connection_lines_antialiased(bool p_antialiased);
	bool is_connection_lines_antialiased() const;

	HBoxContainer *get_menu_hbox() const;
	Ref<ViewPanner> get_panner() const;
	void set_warped_panning(bool p_warped);
	void update_warped_panning();

	void arrange_nodes();

	GraphEdit();
};

VARIANT_ENUM_CAST(GraphEdit::PanningScheme);
VARIANT_ENUM_CAST(GraphEdit::GridPattern);
