/*************************************************************************/
/*  control.h                                                            */
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

#ifndef CONTROL_H
#define CONTROL_H

#include "core/input/shortcut.h"
#include "core/math/transform_2d.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/templates/rid.h"
#include "scene/main/canvas_item.h"
#include "scene/main/node.h"
#include "scene/main/timer.h"
#include "scene/resources/theme.h"

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
		GROW_DIRECTION_END,
		GROW_DIRECTION_BOTH
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
		SIZE_SHRINK_END = 8, //ignored by expand or fill

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

	enum LayoutDirection {
		LAYOUT_DIRECTION_INHERITED,
		LAYOUT_DIRECTION_LOCALE,
		LAYOUT_DIRECTION_LTR,
		LAYOUT_DIRECTION_RTL
	};

	enum TextDirection {
		TEXT_DIRECTION_AUTO = TextServer::DIRECTION_AUTO,
		TEXT_DIRECTION_LTR = TextServer::DIRECTION_LTR,
		TEXT_DIRECTION_RTL = TextServer::DIRECTION_RTL,
		TEXT_DIRECTION_INHERITED,
	};

	enum StructuredTextParser {
		STRUCTURED_TEXT_DEFAULT,
		STRUCTURED_TEXT_URI,
		STRUCTURED_TEXT_FILE,
		STRUCTURED_TEXT_EMAIL,
		STRUCTURED_TEXT_LIST,
		STRUCTURED_TEXT_NONE,
		STRUCTURED_TEXT_CUSTOM
	};

private:
	struct CComparator {
		bool operator()(const Control *p_a, const Control *p_b) const {
			if (p_a->get_canvas_layer() == p_b->get_canvas_layer()) {
				return p_b->is_greater_than(p_a);
			}

			return p_a->get_canvas_layer() < p_b->get_canvas_layer();
		}
	};

	struct Data {
		Point2 pos_cache;
		Size2 size_cache;
		Size2 minimum_size_cache;
		bool minimum_size_valid = false;

		Size2 last_minimum_size;
		bool updating_last_minimum_size = false;

		real_t offset[4] = { 0.0, 0.0, 0.0, 0.0 };
		real_t anchor[4] = { ANCHOR_BEGIN, ANCHOR_BEGIN, ANCHOR_BEGIN, ANCHOR_BEGIN };
		FocusMode focus_mode = FOCUS_NONE;
		GrowDirection h_grow = GROW_DIRECTION_END;
		GrowDirection v_grow = GROW_DIRECTION_END;

		LayoutDirection layout_dir = LAYOUT_DIRECTION_INHERITED;
		bool is_rtl_dirty = true;
		bool is_rtl = false;

		bool auto_translate = true;

		real_t rotation = 0.0;
		Vector2 scale = Vector2(1, 1);
		Vector2 pivot_offset;
		bool size_warning = true;

		int h_size_flags = SIZE_FILL;
		int v_size_flags = SIZE_FILL;
		real_t expand = 1.0;
		Point2 custom_minimum_size;

		MouseFilter mouse_filter = MOUSE_FILTER_STOP;

		bool clip_contents = false;

		bool block_minimum_size_adjust = false;
		bool disable_visibility_clip = false;

		Control *parent = nullptr;
		ObjectID drag_owner;
		Ref<Theme> theme;
		Control *theme_owner = nullptr;
		Window *theme_owner_window = nullptr;
		Window *parent_window = nullptr;
		StringName theme_type_variation;

		String tooltip;
		CursorShape default_cursor = CURSOR_ARROW;

		List<Control *>::Element *RI = nullptr;

		CanvasItem *parent_canvas_item = nullptr;

		NodePath focus_neighbor[4];
		NodePath focus_next;
		NodePath focus_prev;

		bool bulk_theme_override = false;
		HashMap<StringName, Ref<Texture2D>> icon_override;
		HashMap<StringName, Ref<StyleBox>> style_override;
		HashMap<StringName, Ref<Font>> font_override;
		HashMap<StringName, int> font_size_override;
		HashMap<StringName, Color> color_override;
		HashMap<StringName, int> constant_override;

	} data;

	static constexpr unsigned properties_managed_by_container_count = 11;
	static String properties_managed_by_container[properties_managed_by_container_count];

	void _window_find_focus_neighbor(const Vector2 &p_dir, Node *p_at, const Point2 *p_points, real_t p_min, real_t &r_closest_dist, Control **r_closest);
	Control *_get_focus_neighbor(Side p_side, int p_count = 0);

	void _set_anchor(Side p_side, real_t p_anchor);
	void _set_position(const Point2 &p_point);
	void _set_global_position(const Point2 &p_point);
	void _set_size(const Size2 &p_size);

	void _theme_changed();
	void _notify_theme_changed();

	void _update_minimum_size();

	void _clear_size_warning();

	void _compute_offsets(Rect2 p_rect, const real_t p_anchors[4], real_t (&r_offsets)[4]);
	void _compute_anchors(Rect2 p_rect, const real_t p_offsets[4], real_t (&r_anchors)[4]);

	void _size_changed();
	String _get_tooltip() const;

	void _override_changed();

	void _update_canvas_item_transform();

	Transform2D _get_internal_transform() const;

	friend class Viewport;

	void _call_gui_input(const Ref<InputEvent> &p_event);

	void _update_minimum_size_cache();
	friend class Window;
	static void _propagate_theme_changed(Node *p_at, Control *p_owner, Window *p_owner_window, bool p_assign = true);

	template <class T>
	static T get_theme_item_in_types(Control *p_theme_owner, Window *p_theme_owner_window, Theme::DataType p_data_type, const StringName &p_name, List<StringName> p_theme_types);
	static bool has_theme_item_in_types(Control *p_theme_owner, Window *p_theme_owner_window, Theme::DataType p_data_type, const StringName &p_name, List<StringName> p_theme_types);
	_FORCE_INLINE_ void _get_theme_type_dependencies(const StringName &p_theme_type, List<StringName> *p_list) const;

protected:
	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

	//virtual void _window_gui_input(InputEvent p_event);

	virtual Array structured_text_parser(StructuredTextParser p_theme_type, const Array &p_args, const String p_text) const;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_notification);
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;

	//bind helpers

	GDVIRTUAL1RC(bool, _has_point, Vector2)
	GDVIRTUAL2RC(Array, _structured_text_parser, Array, String)
	GDVIRTUAL0RC(Vector2, _get_minimum_size)

	GDVIRTUAL1RC(Variant, _get_drag_data, Vector2)
	GDVIRTUAL2RC(bool, _can_drop_data, Vector2, Variant)
	GDVIRTUAL2(_drop_data, Vector2, Variant)
	GDVIRTUAL1RC(Object *, _make_custom_tooltip, String)

	GDVIRTUAL1(_gui_input, Ref<InputEvent>)

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
		NOTIFICATION_SCROLL_BEGIN = 47,
		NOTIFICATION_SCROLL_END = 48,
		NOTIFICATION_LAYOUT_DIRECTION_CHANGED = 49,

	};

	/* EDITOR */
#ifdef TOOLS_ENABLED
	virtual Dictionary _edit_get_state() const override;
	virtual void _edit_set_state(const Dictionary &p_state) override;

	virtual void _edit_set_position(const Point2 &p_position) override;
	virtual Point2 _edit_get_position() const override;

	virtual void _edit_set_scale(const Size2 &p_scale) override;
	virtual Size2 _edit_get_scale() const override;

	virtual void _edit_set_rect(const Rect2 &p_edit_rect) override;
	virtual Rect2 _edit_get_rect() const override;
	virtual bool _edit_use_rect() const override;

	virtual void _edit_set_rotation(real_t p_rotation) override;
	virtual real_t _edit_get_rotation() const override;
	virtual bool _edit_use_rotation() const override;

	virtual void _edit_set_pivot(const Point2 &p_pivot) override;
	virtual Point2 _edit_get_pivot() const override;
	virtual bool _edit_use_pivot() const override;

	virtual Size2 _edit_get_minimum_size() const override;
#endif

	virtual void gui_input(const Ref<InputEvent> &p_event);

	void accept_event();

	virtual Size2 get_minimum_size() const;
	virtual Size2 get_combined_minimum_size() const;
	virtual bool has_point(const Point2 &p_point) const;
	virtual void set_drag_forwarding(Object *p_target);
	virtual Variant get_drag_data(const Point2 &p_point);
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);
	void set_drag_preview(Control *p_control);
	void force_drag(const Variant &p_data, Control *p_control);
	bool is_drag_successful() const;

	void set_custom_minimum_size(const Size2 &p_custom);
	Size2 get_custom_minimum_size() const;

	Control *get_parent_control() const;
	Window *get_parent_window() const;

	void set_layout_direction(LayoutDirection p_direction);
	LayoutDirection get_layout_direction() const;
	virtual bool is_layout_rtl() const;

	void set_auto_translate(bool p_enable);
	bool is_auto_translating() const;
	_FORCE_INLINE_ String atr(const String p_string) const { return is_auto_translating() ? tr(p_string) : p_string; };

	/* POSITIONING */

	void set_anchors_preset(LayoutPreset p_preset, bool p_keep_offsets = true);
	void set_offsets_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode = PRESET_MODE_MINSIZE, int p_margin = 0);
	void set_anchors_and_offsets_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode = PRESET_MODE_MINSIZE, int p_margin = 0);

	void set_anchor(Side p_side, real_t p_anchor, bool p_keep_offset = true, bool p_push_opposite_anchor = true);
	real_t get_anchor(Side p_side) const;

	void set_offset(Side p_side, real_t p_value);
	real_t get_offset(Side p_side) const;

	void set_anchor_and_offset(Side p_side, real_t p_anchor, real_t p_pos, bool p_push_opposite_anchor = true);

	void set_begin(const Point2 &p_point); // helper
	void set_end(const Point2 &p_point); // helper

	Point2 get_begin() const;
	Point2 get_end() const;

	void set_position(const Point2 &p_point, bool p_keep_offsets = false);
	void set_global_position(const Point2 &p_point, bool p_keep_offsets = false);
	Point2 get_position() const;
	Point2 get_global_position() const;
	Point2 get_screen_position() const;

	void set_size(const Size2 &p_size, bool p_keep_offsets = false);
	Size2 get_size() const;
	void reset_size();

	Rect2 get_rect() const;
	Rect2 get_global_rect() const;
	Rect2 get_screen_rect() const;
	Rect2 get_window_rect() const; ///< use with care, as it blocks waiting for the rendering server
	Rect2 get_anchorable_rect() const override;

	void set_rect(const Rect2 &p_rect); // Reset anchors to begin and set rect, for faster container children sorting.

	void set_rotation(real_t p_radians);
	real_t get_rotation() const;

	void set_h_grow_direction(GrowDirection p_direction);
	GrowDirection get_h_grow_direction() const;

	void set_v_grow_direction(GrowDirection p_direction);
	GrowDirection get_v_grow_direction() const;

	void set_pivot_offset(const Vector2 &p_pivot);
	Vector2 get_pivot_offset() const;

	void set_scale(const Vector2 &p_scale);
	Vector2 get_scale() const;

	void set_theme(const Ref<Theme> &p_theme);
	Ref<Theme> get_theme() const;

	void set_theme_type_variation(const StringName &p_theme_type);
	StringName get_theme_type_variation() const;

	void set_h_size_flags(int p_flags);
	int get_h_size_flags() const;

	void set_v_size_flags(int p_flags);
	int get_v_size_flags() const;

	void set_stretch_ratio(real_t p_ratio);
	real_t get_stretch_ratio() const;

	void minimum_size_changed();

	/* FOCUS */

	void set_focus_mode(FocusMode p_focus_mode);
	FocusMode get_focus_mode() const;
	bool has_focus() const;
	void grab_focus();
	void release_focus();

	Control *find_next_valid_focus() const;
	Control *find_prev_valid_focus() const;

	void set_focus_neighbor(Side p_side, const NodePath &p_neighbor);
	NodePath get_focus_neighbor(Side p_side) const;

	void set_focus_next(const NodePath &p_next);
	NodePath get_focus_next() const;
	void set_focus_previous(const NodePath &p_prev);
	NodePath get_focus_previous() const;

	Control *get_focus_owner() const;

	void set_mouse_filter(MouseFilter p_filter);
	MouseFilter get_mouse_filter() const;

	/* SKINNING */

	void begin_bulk_theme_override();
	void end_bulk_theme_override();

	void add_theme_icon_override(const StringName &p_name, const Ref<Texture2D> &p_icon);
	void add_theme_style_override(const StringName &p_name, const Ref<StyleBox> &p_style);
	void add_theme_font_override(const StringName &p_name, const Ref<Font> &p_font);
	void add_theme_font_size_override(const StringName &p_name, int p_font_size);
	void add_theme_color_override(const StringName &p_name, const Color &p_color);
	void add_theme_constant_override(const StringName &p_name, int p_constant);

	void remove_theme_icon_override(const StringName &p_name);
	void remove_theme_style_override(const StringName &p_name);
	void remove_theme_font_override(const StringName &p_name);
	void remove_theme_font_size_override(const StringName &p_name);
	void remove_theme_color_override(const StringName &p_name);
	void remove_theme_constant_override(const StringName &p_name);

	Ref<Texture2D> get_theme_icon(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	Ref<StyleBox> get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	Ref<Font> get_theme_font(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	int get_theme_font_size(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	Color get_theme_color(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	int get_theme_constant(const StringName &p_name, const StringName &p_theme_type = StringName()) const;

	bool has_theme_icon_override(const StringName &p_name) const;
	bool has_theme_stylebox_override(const StringName &p_name) const;
	bool has_theme_font_override(const StringName &p_name) const;
	bool has_theme_font_size_override(const StringName &p_name) const;
	bool has_theme_color_override(const StringName &p_name) const;
	bool has_theme_constant_override(const StringName &p_name) const;

	bool has_theme_icon(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	bool has_theme_stylebox(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	bool has_theme_font(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	bool has_theme_font_size(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	bool has_theme_color(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	bool has_theme_constant(const StringName &p_name, const StringName &p_theme_type = StringName()) const;

	static float fetch_theme_default_base_scale(Control *p_theme_owner, Window *p_theme_owner_window);
	static Ref<Font> fetch_theme_default_font(Control *p_theme_owner, Window *p_theme_owner_window);
	static int fetch_theme_default_font_size(Control *p_theme_owner, Window *p_theme_owner_window);

	float get_theme_default_base_scale() const;
	Ref<Font> get_theme_default_font() const;
	int get_theme_default_font_size() const;

	/* TOOLTIP */

	void set_tooltip(const String &p_tooltip);
	virtual String get_tooltip(const Point2 &p_pos) const;
	virtual Control *make_custom_tooltip(const String &p_text) const;

	/* CURSOR */

	void set_default_cursor_shape(CursorShape p_shape);
	CursorShape get_default_cursor_shape() const;
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const;

	virtual Transform2D get_transform() const override;

	bool is_top_level_control() const;

	Size2 get_parent_area_size() const;
	Rect2 get_parent_anchorable_rect() const;

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

	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;
	TypedArray<String> get_configuration_warnings() const override;

	Control() {}
};

VARIANT_ENUM_CAST(Control::FocusMode);
VARIANT_ENUM_CAST(Control::SizeFlags);
VARIANT_ENUM_CAST(Control::CursorShape);
VARIANT_ENUM_CAST(Control::LayoutPreset);
VARIANT_ENUM_CAST(Control::LayoutPresetMode);
VARIANT_ENUM_CAST(Control::MouseFilter);
VARIANT_ENUM_CAST(Control::GrowDirection);
VARIANT_ENUM_CAST(Control::Anchor);
VARIANT_ENUM_CAST(Control::LayoutDirection);
VARIANT_ENUM_CAST(Control::TextDirection);
VARIANT_ENUM_CAST(Control::StructuredTextParser);

#endif
