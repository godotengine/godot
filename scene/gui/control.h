/*************************************************************************/
/*  control.h                                                            */
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
#ifndef CONTROL_H
#define CONTROL_H

#include "math_2d.h"
#include "rid.h"
#include "scene/2d/canvas_item.h"
#include "scene/gui/shortcut.h"
#include "scene/main/node.h"
#include "scene/main/timer.h"
#include "scene/resources/theme.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Viewport;
class Label;
class Panel;

class Control : public CanvasItem {

	GDCLASS(Control, CanvasItem);
	OBJ_CATEGORY("GUI Nodes");

public:
	enum Anchor {

		ANCHOR_BEGIN = 0,
		ANCHOR_END = 1
	};

	enum GrowDirection {
		GROW_DIRECTION_BEGIN,
		GROW_DIRECTION_END
	};

	enum FocusMode {
		FOCUS_NONE,
		FOCUS_CLICK,
		FOCUS_ALL
	};

	enum SizeFlags {

		SIZE_FILL = 1,
		SIZE_EXPAND = 2,
		SIZE_EXPAND_FILL = SIZE_EXPAND | SIZE_FILL,
		SIZE_SHRINK_CENTER = 4, //ignored by expand or fill
		SIZE_SHRINK_END = 8, //ignored by expand or fil

	};

	enum MouseFilter {
		MOUSE_FILTER_STOP,
		MOUSE_FILTER_PASS,
		MOUSE_FILTER_IGNORE
	};

	enum CursorShape {
		CURSOR_ARROW,
		CURSOR_IBEAM,
		CURSOR_POINTING_HAND,
		CURSOR_CROSS,
		CURSOR_WAIT,
		CURSOR_BUSY,
		CURSOR_DRAG,
		CURSOR_CAN_DROP,
		CURSOR_FORBIDDEN,
		CURSOR_VSIZE,
		CURSOR_HSIZE,
		CURSOR_BDIAGSIZE,
		CURSOR_FDIAGSIZE,
		CURSOR_MOVE,
		CURSOR_VSPLIT,
		CURSOR_HSPLIT,
		CURSOR_HELP,
		CURSOR_MAX
	};

	enum LayoutPreset {
		PRESET_TOP_LEFT,
		PRESET_TOP_RIGHT,
		PRESET_BOTTOM_LEFT,
		PRESET_BOTTOM_RIGHT,
		PRESET_CENTER_LEFT,
		PRESET_CENTER_TOP,
		PRESET_CENTER_RIGHT,
		PRESET_CENTER_BOTTOM,
		PRESET_CENTER,
		PRESET_LEFT_WIDE,
		PRESET_TOP_WIDE,
		PRESET_RIGHT_WIDE,
		PRESET_BOTTOM_WIDE,
		PRESET_VCENTER_WIDE,
		PRESET_HCENTER_WIDE,
		PRESET_WIDE
	};

	enum LayoutPresetMode {
		PRESET_MODE_MINSIZE,
		PRESET_MODE_KEEP_WIDTH,
		PRESET_MODE_KEEP_HEIGHT,
		PRESET_MODE_KEEP_SIZE
	};

private:
	struct CComparator {

		bool operator()(const Control *p_a, const Control *p_b) const {
			if (p_a->get_canvas_layer() == p_b->get_canvas_layer())
				return p_b->is_greater_than(p_a);
			else
				return p_a->get_canvas_layer() < p_b->get_canvas_layer();
		}
	};

	struct Data {

		Point2 pos_cache;
		Size2 size_cache;

		float margin[4];
		float anchor[4];
		FocusMode focus_mode;
		GrowDirection h_grow;
		GrowDirection v_grow;

		float rotation;
		Vector2 scale;
		Vector2 pivot_offset;

		bool pending_resize;

		int h_size_flags;
		int v_size_flags;
		float expand;
		bool pending_min_size_update;
		Point2 custom_minimum_size;

		bool pass_on_modal_close_click;

		MouseFilter mouse_filter;

		bool clip_contents;

		bool block_minimum_size_adjust;
		bool disable_visibility_clip;

		Control *parent;
		ObjectID drag_owner;
		bool modal;
		bool modal_exclusive;
		uint64_t modal_frame; //frame used to put something as modal
		Ref<Theme> theme;
		Control *theme_owner;
		String tooltip;
		CursorShape default_cursor;

		List<Control *>::Element *MI; //modal item
		List<Control *>::Element *SI;
		List<Control *>::Element *RI;

		CanvasItem *parent_canvas_item;

		ObjectID modal_prev_focus_owner;

		NodePath focus_neighbour[4];
		NodePath focus_next;
		NodePath focus_prev;

		HashMap<StringName, Ref<Texture>, StringNameHasher> icon_override;
		HashMap<StringName, Ref<Shader>, StringNameHasher> shader_override;
		HashMap<StringName, Ref<StyleBox>, StringNameHasher> style_override;
		HashMap<StringName, Ref<Font>, StringNameHasher> font_override;
		HashMap<StringName, Color, StringNameHasher> color_override;
		HashMap<StringName, int, StringNameHasher> constant_override;
		Map<Ref<Font>, int> font_refcount;

	} data;

	// used internally
	Control *_find_control_at_pos(CanvasItem *p_node, const Point2 &p_pos, const Transform2D &p_xform, Transform2D &r_inv_xform);

	void _window_find_focus_neighbour(const Vector2 &p_dir, Node *p_at, const Point2 *p_points, float p_min, float &r_closest_dist, Control **r_closest);
	Control *_get_focus_neighbour(Margin p_margin, int p_count = 0);

	void _set_anchor(Margin p_margin, float p_anchor);

	float _get_parent_range(int p_idx) const;
	float _get_range(int p_idx) const;
	float _s2a(float p_val, float p_anchor, float p_range) const;
	float _a2s(float p_val, float p_anchor, float p_range) const;
	void _propagate_theme_changed(CanvasItem *p_at, Control *p_owner, bool p_assign = true);
	void _theme_changed();

	void _change_notify_margins();
	void _update_minimum_size();

	void _update_scroll();
	void _resize(const Size2 &p_size);

	void _size_changed();
	String _get_tooltip() const;

	void _ref_font(Ref<Font> p_sc);
	void _unref_font(Ref<Font> p_sc);
	void _font_changed();

	void _update_canvas_item_transform();

	Transform2D _get_internal_transform() const;

	friend class Viewport;
	void _modal_stack_remove();
	void _modal_set_prev_focus_owner(ObjectID p_prev);

protected:
	virtual void add_child_notify(Node *p_child);
	virtual void remove_child_notify(Node *p_child);

	//virtual void _window_gui_input(InputEvent p_event);

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_notification);

	static void _bind_methods();

	//bind helpers

public:
	enum {

		/*		NOTIFICATION_DRAW=30,
		NOTIFICATION_VISIBILITY_CHANGED=38*/
		NOTIFICATION_RESIZED = 40,
		NOTIFICATION_MOUSE_ENTER = 41,
		NOTIFICATION_MOUSE_EXIT = 42,
		NOTIFICATION_FOCUS_ENTER = 43,
		NOTIFICATION_FOCUS_EXIT = 44,
		NOTIFICATION_THEME_CHANGED = 45,
		NOTIFICATION_MODAL_CLOSE = 46,

	};

	virtual Dictionary _edit_get_state() const;
	virtual void _edit_set_state(const Dictionary &p_state);

	virtual void _edit_set_position(const Point2 &p_position);
	virtual Point2 _edit_get_position() const;

	virtual void _edit_set_rect(const Rect2 &p_edit_rect);
	virtual Rect2 _edit_get_rect() const;
	virtual bool _edit_use_rect() const;

	virtual void _edit_set_rotation(float p_rotation);
	virtual float _edit_get_rotation() const;
	virtual bool _edit_use_rotation() const;

	virtual void _edit_set_pivot(const Point2 &p_pivot);
	virtual Point2 _edit_get_pivot() const;
	virtual bool _edit_use_pivot() const;

	virtual Size2 _edit_get_minimum_size() const;

	void accept_event();

	virtual Size2 get_minimum_size() const;
	virtual Size2 get_combined_minimum_size() const;
	virtual bool has_point(const Point2 &p_point) const;
	virtual bool clips_input() const;
	virtual void set_drag_forwarding(Control *p_target);
	virtual Variant get_drag_data(const Point2 &p_point);
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);
	void set_drag_preview(Control *p_control);
	void force_drag(const Variant &p_data, Control *p_control);

	void set_custom_minimum_size(const Size2 &p_custom);
	Size2 get_custom_minimum_size() const;

	bool is_window_modal_on_top() const;
	uint64_t get_modal_frame() const; //frame in which this was made modal

	Control *get_parent_control() const;

	/* POSITIONING */

	void set_anchors_preset(LayoutPreset p_preset, bool p_keep_margin = false);
	void set_margins_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode = PRESET_MODE_MINSIZE, int p_margin = 0);
	void set_anchors_and_margins_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode = PRESET_MODE_MINSIZE, int p_margin = 0);

	void set_anchor(Margin p_margin, float p_anchor, bool p_keep_margin = false, bool p_push_opposite_anchor = true);
	float get_anchor(Margin p_margin) const;

	void set_margin(Margin p_margin, float p_value);
	float get_margin(Margin p_margin) const;

	void set_anchor_and_margin(Margin p_margin, float p_anchor, float p_pos, bool p_push_opposite_anchor = true);

	void set_begin(const Point2 &p_point); // helper
	void set_end(const Point2 &p_point); // helper

	Point2 get_begin() const;
	Point2 get_end() const;

	void set_position(const Point2 &p_point);
	void set_global_position(const Point2 &p_point);
	Point2 get_position() const;
	Point2 get_global_position() const;

	void set_size(const Size2 &p_size);
	Size2 get_size() const;

	Rect2 get_rect() const;
	Rect2 get_global_rect() const;
	Rect2 get_window_rect() const; ///< use with care, as it blocks waiting for the visual server

	void set_rotation(float p_radians);
	void set_rotation_degrees(float p_degrees);
	float get_rotation() const;
	float get_rotation_degrees() const;

	void set_h_grow_direction(GrowDirection p_direction);
	GrowDirection get_h_grow_direction() const;

	void set_v_grow_direction(GrowDirection p_direction);
	GrowDirection get_v_grow_direction() const;

	void set_pivot_offset(const Vector2 &p_pivot);
	Vector2 get_pivot_offset() const;

	void set_scale(const Vector2 &p_scale);
	Vector2 get_scale() const;

	void show_modal(bool p_exclusive = false);

	void set_theme(const Ref<Theme> &p_theme);
	Ref<Theme> get_theme() const;

	void set_h_size_flags(int p_flags);
	int get_h_size_flags() const;

	void set_v_size_flags(int p_flags);
	int get_v_size_flags() const;

	void set_stretch_ratio(float p_ratio);
	float get_stretch_ratio() const;

	void minimum_size_changed();

	/* FOCUS */

	void set_focus_mode(FocusMode p_focus_mode);
	FocusMode get_focus_mode() const;
	bool has_focus() const;
	void grab_focus();
	void release_focus();

	Control *find_next_valid_focus() const;
	Control *find_prev_valid_focus() const;

	void set_focus_neighbour(Margin p_margin, const NodePath &p_neighbour);
	NodePath get_focus_neighbour(Margin p_margin) const;

	void set_focus_next(const NodePath &p_next);
	NodePath get_focus_next() const;
	void set_focus_previous(const NodePath &p_prev);
	NodePath get_focus_previous() const;

	Control *get_focus_owner() const;

	void set_mouse_filter(MouseFilter p_filter);
	MouseFilter get_mouse_filter() const;

	void set_pass_on_modal_close_click(bool p_pass_on);
	bool pass_on_modal_close_click() const;

	/* SKINNING */

	void add_icon_override(const StringName &p_name, const Ref<Texture> &p_icon);
	void add_shader_override(const StringName &p_name, const Ref<Shader> &p_shader);
	void add_style_override(const StringName &p_name, const Ref<StyleBox> &p_style);
	void add_font_override(const StringName &p_name, const Ref<Font> &p_font);
	void add_color_override(const StringName &p_name, const Color &p_color);
	void add_constant_override(const StringName &p_name, int p_constant);

	Ref<Texture> get_icon(const StringName &p_name, const StringName &p_type = StringName()) const;
	Ref<Shader> get_shader(const StringName &p_name, const StringName &p_type = StringName()) const;
	Ref<StyleBox> get_stylebox(const StringName &p_name, const StringName &p_type = StringName()) const;
	Ref<Font> get_font(const StringName &p_name, const StringName &p_type = StringName()) const;
	Color get_color(const StringName &p_name, const StringName &p_type = StringName()) const;
	int get_constant(const StringName &p_name, const StringName &p_type = StringName()) const;

	bool has_icon_override(const StringName &p_name) const;
	bool has_shader_override(const StringName &p_name) const;
	bool has_stylebox_override(const StringName &p_name) const;
	bool has_font_override(const StringName &p_name) const;
	bool has_color_override(const StringName &p_name) const;
	bool has_constant_override(const StringName &p_name) const;

	bool has_icon(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_shader(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_stylebox(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_font(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_color(const StringName &p_name, const StringName &p_type = StringName()) const;
	bool has_constant(const StringName &p_name, const StringName &p_type = StringName()) const;

	/* TOOLTIP */

	void set_tooltip(const String &p_tooltip);
	virtual String get_tooltip(const Point2 &p_pos) const;

	/* CURSOR */

	void set_default_cursor_shape(CursorShape p_shape);
	CursorShape get_default_cursor_shape() const;
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const;

	virtual Transform2D get_transform() const;

	bool is_toplevel_control() const;

	Size2 get_parent_area_size() const;

	void grab_click_focus();

	void warp_mouse(const Point2 &p_to_pos);

	virtual bool is_text_field() const;

	Control *get_root_parent_control() const;

	void set_clip_contents(bool p_clip);
	bool is_clipping_contents();

	void set_block_minimum_size_adjust(bool p_block);
	bool is_minimum_size_adjust_blocked() const;

	void set_disable_visibility_clip(bool p_ignore);
	bool is_visibility_clip_disabled() const;

	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;

	Control();
	~Control();
};

VARIANT_ENUM_CAST(Control::FocusMode);
VARIANT_ENUM_CAST(Control::SizeFlags);
VARIANT_ENUM_CAST(Control::CursorShape);
VARIANT_ENUM_CAST(Control::LayoutPreset);
VARIANT_ENUM_CAST(Control::LayoutPresetMode);
VARIANT_ENUM_CAST(Control::MouseFilter);
VARIANT_ENUM_CAST(Control::GrowDirection);
VARIANT_ENUM_CAST(Control::Anchor);

#endif
