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

#ifndef GRAPH_EDIT_H
#define GRAPH_EDIT_H

#include "scene/gui/box_container.h"
#include "scene/gui/graph_node.h"

class Button;
class GraphEdit;
class GraphEditArranger;
class HScrollBar;
class Label;
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
	struct Connection {
		StringName from_node;
		StringName to_node;
		int from_port = 0;
		int to_port = 0;
		float activity = 0.0;
	};

	// Should be in sync with ControlScheme in ViewPanner.
	enum PanningScheme {
		SCROLL_ZOOMS,
		SCROLL_PANS,
	};

private:
	struct ConnectionType {
		union {
			struct {
				uint32_t type_a;
				uint32_t type_b;
			};
			uint64_t key = 0;
		};

		static uint32_t hash(const ConnectionType &p_conn) {
			return hash_one_uint64(p_conn.key);
		}
		bool operator==(const ConnectionType &p_type) const {
			return key == p_type.key;
		}

		ConnectionType(uint32_t a = 0, uint32_t b = 0) {
			type_a = a;
			type_b = b;
		}
	};

	Label *zoom_label = nullptr;
	Button *zoom_minus_button = nullptr;
	Button *zoom_reset_button = nullptr;
	Button *zoom_plus_button = nullptr;

	Button *toggle_snapping_button = nullptr;
	SpinBox *snapping_distance_spinbox = nullptr;
	Button *toggle_grid_button = nullptr;
	Button *minimap_button = nullptr;
	Button *arrange_button = nullptr;

	HScrollBar *h_scrollbar = nullptr;
	VScrollBar *v_scrollbar = nullptr;

	Ref<ViewPanner> panner;
	bool warped_panning = true;

	bool show_menu = true;
	bool show_zoom_label = false;
	bool show_grid_buttons = true;
	bool show_zoom_buttons = true;
	bool show_minimap_button = true;
	bool show_arrange_button = true;

	bool snapping_enabled = true;
	int snapping_distance = 20;
	bool show_grid = true;

	bool connecting = false;
	String connecting_from;
	bool connecting_out = false;
	int connecting_index = 0;
	int connecting_type = 0;
	Color connecting_color;
	bool connecting_target = false;
	Vector2 connecting_to;
	StringName connecting_target_to;
	int connecting_target_index = 0;

	bool just_disconnected = false;
	bool connecting_valid = false;

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

	bool box_selecting = false;
	bool box_selection_mode_additive = false;
	Point2 box_selecting_from;
	Point2 box_selecting_to;
	Rect2 box_selecting_rect;
	List<GraphElement *> prev_selected;

	bool setting_scroll_offset = false;
	bool right_disconnects = false;
	bool updating = false;
	bool awaiting_scroll_offset_update = false;
	List<Connection> connections;

	float lines_thickness = 2.0f;
	float lines_curvature = 0.5f;
	bool lines_antialiased = true;

	PanelContainer *menu_panel = nullptr;
	HBoxContainer *menu_hbox = nullptr;
	Control *connections_layer = nullptr;
	GraphEditFilter *top_layer = nullptr;
	GraphEditMinimap *minimap = nullptr;

	Ref<GraphEditArranger> arranger;

	HashSet<ConnectionType, ConnectionType> valid_connection_types;
	HashSet<int> valid_left_disconnect_types;
	HashSet<int> valid_right_disconnect_types;

	struct ThemeCache {
		float base_scale = 1.0;

		Ref<StyleBox> panel;
		Color grid_major;
		Color grid_minor;

		Color activity_color;
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

		float port_hotzone_inner_extent = 0.0;
		float port_hotzone_outer_extent = 0.0;
	} theme_cache;

	void _pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event);
	void _zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event);

	void _zoom_minus();
	void _zoom_reset();
	void _zoom_plus();
	void _update_zoom_label();

	void _draw_connection_line(CanvasItem *p_where, const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, const Color &p_to_color, float p_width, float p_zoom);

	void _graph_element_selected(Node *p_node);
	void _graph_element_deselected(Node *p_node);
	void _graph_element_moved_to_front(Node *p_node);
	void _graph_element_resized(Vector2 p_new_minsize, Node *p_node);
	void _graph_element_moved(Node *p_node);
	void _graph_node_slot_updated(int p_index, Node *p_node);

	void _update_scroll();
	void _update_scroll_offset();
	void _scroll_moved(double);
	virtual void gui_input(const Ref<InputEvent> &p_ev) override;
	void _top_layer_input(const Ref<InputEvent> &p_ev);

	bool is_in_port_hotzone(const Vector2 &p_pos, const Vector2 &p_mouse_pos, const Vector2i &p_port_size, bool p_left);

	void _top_layer_draw();
	void _connections_layer_draw();
	void _minimap_draw();

	TypedArray<Dictionary> _get_connection_list() const;

	friend class GraphEditFilter;
	bool _filter_input(const Point2 &p_point);
	void _snapping_toggled();
	void _snapping_distance_changed(double);
	void _show_grid_toggled();

	friend class GraphEditMinimap;
	void _minimap_toggled();

	bool _check_clickable_control(Control *p_control, const Vector2 &r_mouse_pos, const Vector2 &p_offset);

#ifndef DISABLE_DEPRECATED
	bool _is_arrange_nodes_button_hidden_bind_compat_81582() const;
	void _set_arrange_nodes_button_hidden_bind_compat_81582(bool p_enable);
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

	virtual bool is_in_input_hotzone(GraphNode *p_graph_node, int p_port_idx, const Vector2 &p_mouse_pos, const Vector2i &p_port_size);
	virtual bool is_in_output_hotzone(GraphNode *p_graph_node, int p_port_idx, const Vector2 &p_mouse_pos, const Vector2i &p_port_size);

	GDVIRTUAL2RC(Vector<Vector2>, _get_connection_line, Vector2, Vector2)
	GDVIRTUAL3R(bool, _is_in_input_hotzone, Object *, int, Vector2)
	GDVIRTUAL3R(bool, _is_in_output_hotzone, Object *, int, Vector2)
	GDVIRTUAL4R(bool, _is_node_hover_valid, StringName, int, StringName, int);

public:
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	PackedStringArray get_configuration_warnings() const override;

	Error connect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);
	bool is_node_connected(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);
	void disconnect_node(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);
	void clear_connections();
	void force_connection_drag_end();

	virtual PackedVector2Array get_connection_line(const Vector2 &p_from, const Vector2 &p_to);
	virtual bool is_node_hover_valid(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);

	void set_connection_activity(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port, float p_activity);

	void add_valid_connection_type(int p_type, int p_with_type);
	void remove_valid_connection_type(int p_type, int p_with_type);
	bool is_valid_connection_type(int p_type, int p_with_type) const;

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

	GraphEditFilter *get_top_layer() const { return top_layer; }
	GraphEditMinimap *get_minimap() const { return minimap; }

	void get_connection_list(List<Connection> *r_connections) const;

	void set_right_disconnects(bool p_enable);
	bool is_right_disconnects_enabled() const;

	void add_valid_right_disconnect_type(int p_type);
	void remove_valid_right_disconnect_type(int p_type);

	void add_valid_left_disconnect_type(int p_type);
	void remove_valid_left_disconnect_type(int p_type);

	void set_scroll_offset(const Vector2 &p_ofs);
	Vector2 get_scroll_offset() const;

	void set_selected(Node *p_child);

	void set_snapping_enabled(bool p_enable);
	bool is_snapping_enabled() const;

	void set_snapping_distance(int p_snapping_distance);
	int get_snapping_distance() const;

	void set_show_grid(bool p_enable);
	bool is_showing_grid() const;

	void set_connection_lines_curvature(float p_curvature);
	float get_connection_lines_curvature() const;

	void set_connection_lines_thickness(float p_thickness);
	float get_connection_lines_thickness() const;

	void set_connection_lines_antialiased(bool p_antialiased);
	bool is_connection_lines_antialiased() const;

	HBoxContainer *get_menu_hbox();
	Ref<ViewPanner> get_panner();
	void set_warped_panning(bool p_warped);

	void arrange_nodes();

	GraphEdit();
};

VARIANT_ENUM_CAST(GraphEdit::PanningScheme);

#endif // GRAPH_EDIT_H
