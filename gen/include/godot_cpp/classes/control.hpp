/**************************************************************************/
/*  control.hpp                                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/canvas_item.hpp>
#include <godot_cpp/classes/display_server.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Array;
class Callable;
class Font;
class InputEvent;
class Object;
class StyleBox;
class Texture2D;
class Theme;

class Control : public CanvasItem {
	GDEXTENSION_CLASS(Control, CanvasItem)

public:
	enum FocusMode {
		FOCUS_NONE = 0,
		FOCUS_CLICK = 1,
		FOCUS_ALL = 2,
		FOCUS_ACCESSIBILITY = 3,
	};

	enum FocusBehaviorRecursive {
		FOCUS_BEHAVIOR_INHERITED = 0,
		FOCUS_BEHAVIOR_DISABLED = 1,
		FOCUS_BEHAVIOR_ENABLED = 2,
	};

	enum MouseBehaviorRecursive {
		MOUSE_BEHAVIOR_INHERITED = 0,
		MOUSE_BEHAVIOR_DISABLED = 1,
		MOUSE_BEHAVIOR_ENABLED = 2,
	};

	enum CursorShape {
		CURSOR_ARROW = 0,
		CURSOR_IBEAM = 1,
		CURSOR_POINTING_HAND = 2,
		CURSOR_CROSS = 3,
		CURSOR_WAIT = 4,
		CURSOR_BUSY = 5,
		CURSOR_DRAG = 6,
		CURSOR_CAN_DROP = 7,
		CURSOR_FORBIDDEN = 8,
		CURSOR_VSIZE = 9,
		CURSOR_HSIZE = 10,
		CURSOR_BDIAGSIZE = 11,
		CURSOR_FDIAGSIZE = 12,
		CURSOR_MOVE = 13,
		CURSOR_VSPLIT = 14,
		CURSOR_HSPLIT = 15,
		CURSOR_HELP = 16,
	};

	enum LayoutPreset {
		PRESET_TOP_LEFT = 0,
		PRESET_TOP_RIGHT = 1,
		PRESET_BOTTOM_LEFT = 2,
		PRESET_BOTTOM_RIGHT = 3,
		PRESET_CENTER_LEFT = 4,
		PRESET_CENTER_TOP = 5,
		PRESET_CENTER_RIGHT = 6,
		PRESET_CENTER_BOTTOM = 7,
		PRESET_CENTER = 8,
		PRESET_LEFT_WIDE = 9,
		PRESET_TOP_WIDE = 10,
		PRESET_RIGHT_WIDE = 11,
		PRESET_BOTTOM_WIDE = 12,
		PRESET_VCENTER_WIDE = 13,
		PRESET_HCENTER_WIDE = 14,
		PRESET_FULL_RECT = 15,
	};

	enum LayoutPresetMode {
		PRESET_MODE_MINSIZE = 0,
		PRESET_MODE_KEEP_WIDTH = 1,
		PRESET_MODE_KEEP_HEIGHT = 2,
		PRESET_MODE_KEEP_SIZE = 3,
	};

	enum SizeFlags : uint64_t {
		SIZE_SHRINK_BEGIN = 0,
		SIZE_FILL = 1,
		SIZE_EXPAND = 2,
		SIZE_EXPAND_FILL = 3,
		SIZE_SHRINK_CENTER = 4,
		SIZE_SHRINK_END = 8,
	};

	enum MouseFilter {
		MOUSE_FILTER_STOP = 0,
		MOUSE_FILTER_PASS = 1,
		MOUSE_FILTER_IGNORE = 2,
	};

	enum GrowDirection {
		GROW_DIRECTION_BEGIN = 0,
		GROW_DIRECTION_END = 1,
		GROW_DIRECTION_BOTH = 2,
	};

	enum Anchor {
		ANCHOR_BEGIN = 0,
		ANCHOR_END = 1,
	};

	enum LayoutDirection {
		LAYOUT_DIRECTION_INHERITED = 0,
		LAYOUT_DIRECTION_APPLICATION_LOCALE = 1,
		LAYOUT_DIRECTION_LTR = 2,
		LAYOUT_DIRECTION_RTL = 3,
		LAYOUT_DIRECTION_SYSTEM_LOCALE = 4,
		LAYOUT_DIRECTION_MAX = 5,
		LAYOUT_DIRECTION_LOCALE = 1,
	};

	enum TextDirection {
		TEXT_DIRECTION_INHERITED = 3,
		TEXT_DIRECTION_AUTO = 0,
		TEXT_DIRECTION_LTR = 1,
		TEXT_DIRECTION_RTL = 2,
	};

	static const int NOTIFICATION_RESIZED = 40;
	static const int NOTIFICATION_MOUSE_ENTER = 41;
	static const int NOTIFICATION_MOUSE_EXIT = 42;
	static const int NOTIFICATION_MOUSE_ENTER_SELF = 60;
	static const int NOTIFICATION_MOUSE_EXIT_SELF = 61;
	static const int NOTIFICATION_FOCUS_ENTER = 43;
	static const int NOTIFICATION_FOCUS_EXIT = 44;
	static const int NOTIFICATION_THEME_CHANGED = 45;
	static const int NOTIFICATION_SCROLL_BEGIN = 47;
	static const int NOTIFICATION_SCROLL_END = 48;
	static const int NOTIFICATION_LAYOUT_DIRECTION_CHANGED = 49;

	void accept_event();
	Vector2 get_minimum_size() const;
	Vector2 get_combined_minimum_size() const;
	void set_anchors_preset(Control::LayoutPreset p_preset, bool p_keep_offsets = false);
	void set_offsets_preset(Control::LayoutPreset p_preset, Control::LayoutPresetMode p_resize_mode = (Control::LayoutPresetMode)0, int32_t p_margin = 0);
	void set_anchors_and_offsets_preset(Control::LayoutPreset p_preset, Control::LayoutPresetMode p_resize_mode = (Control::LayoutPresetMode)0, int32_t p_margin = 0);
	void set_anchor(Side p_side, float p_anchor, bool p_keep_offset = false, bool p_push_opposite_anchor = true);
	float get_anchor(Side p_side) const;
	void set_offset(Side p_side, float p_offset);
	float get_offset(Side p_offset) const;
	void set_anchor_and_offset(Side p_side, float p_anchor, float p_offset, bool p_push_opposite_anchor = false);
	void set_begin(const Vector2 &p_position);
	void set_end(const Vector2 &p_position);
	void set_position(const Vector2 &p_position, bool p_keep_offsets = false);
	void set_size(const Vector2 &p_size, bool p_keep_offsets = false);
	void reset_size();
	void set_custom_minimum_size(const Vector2 &p_size);
	void set_global_position(const Vector2 &p_position, bool p_keep_offsets = false);
	void set_rotation(float p_radians);
	void set_rotation_degrees(float p_degrees);
	void set_scale(const Vector2 &p_scale);
	void set_pivot_offset(const Vector2 &p_pivot_offset);
	void set_pivot_offset_ratio(const Vector2 &p_ratio);
	Vector2 get_begin() const;
	Vector2 get_end() const;
	Vector2 get_position() const;
	Vector2 get_size() const;
	float get_rotation() const;
	float get_rotation_degrees() const;
	Vector2 get_scale() const;
	Vector2 get_pivot_offset() const;
	Vector2 get_pivot_offset_ratio() const;
	Vector2 get_combined_pivot_offset() const;
	Vector2 get_custom_minimum_size() const;
	Vector2 get_parent_area_size() const;
	Vector2 get_global_position() const;
	Vector2 get_screen_position() const;
	Rect2 get_rect() const;
	Rect2 get_global_rect() const;
	void set_focus_mode(Control::FocusMode p_mode);
	Control::FocusMode get_focus_mode() const;
	Control::FocusMode get_focus_mode_with_override() const;
	void set_focus_behavior_recursive(Control::FocusBehaviorRecursive p_focus_behavior_recursive);
	Control::FocusBehaviorRecursive get_focus_behavior_recursive() const;
	bool has_focus(bool p_ignore_hidden_focus = false) const;
	void grab_focus(bool p_hide_focus = false);
	void release_focus();
	Control *find_prev_valid_focus() const;
	Control *find_next_valid_focus() const;
	Control *find_valid_focus_neighbor(Side p_side) const;
	void set_h_size_flags(BitField<Control::SizeFlags> p_flags);
	BitField<Control::SizeFlags> get_h_size_flags() const;
	void set_stretch_ratio(float p_ratio);
	float get_stretch_ratio() const;
	void set_v_size_flags(BitField<Control::SizeFlags> p_flags);
	BitField<Control::SizeFlags> get_v_size_flags() const;
	void set_theme(const Ref<Theme> &p_theme);
	Ref<Theme> get_theme() const;
	void set_theme_type_variation(const StringName &p_theme_type);
	StringName get_theme_type_variation() const;
	void begin_bulk_theme_override();
	void end_bulk_theme_override();
	void add_theme_icon_override(const StringName &p_name, const Ref<Texture2D> &p_texture);
	void add_theme_stylebox_override(const StringName &p_name, const Ref<StyleBox> &p_stylebox);
	void add_theme_font_override(const StringName &p_name, const Ref<Font> &p_font);
	void add_theme_font_size_override(const StringName &p_name, int32_t p_font_size);
	void add_theme_color_override(const StringName &p_name, const Color &p_color);
	void add_theme_constant_override(const StringName &p_name, int32_t p_constant);
	void remove_theme_icon_override(const StringName &p_name);
	void remove_theme_stylebox_override(const StringName &p_name);
	void remove_theme_font_override(const StringName &p_name);
	void remove_theme_font_size_override(const StringName &p_name);
	void remove_theme_color_override(const StringName &p_name);
	void remove_theme_constant_override(const StringName &p_name);
	Ref<Texture2D> get_theme_icon(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	Ref<StyleBox> get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	Ref<Font> get_theme_font(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	int32_t get_theme_font_size(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	Color get_theme_color(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
	int32_t get_theme_constant(const StringName &p_name, const StringName &p_theme_type = StringName()) const;
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
	float get_theme_default_base_scale() const;
	Ref<Font> get_theme_default_font() const;
	int32_t get_theme_default_font_size() const;
	Control *get_parent_control() const;
	void set_h_grow_direction(Control::GrowDirection p_direction);
	Control::GrowDirection get_h_grow_direction() const;
	void set_v_grow_direction(Control::GrowDirection p_direction);
	Control::GrowDirection get_v_grow_direction() const;
	void set_tooltip_auto_translate_mode(Node::AutoTranslateMode p_mode);
	Node::AutoTranslateMode get_tooltip_auto_translate_mode() const;
	void set_tooltip_text(const String &p_hint);
	String get_tooltip_text() const;
	String get_tooltip(const Vector2 &p_at_position = Vector2(0, 0)) const;
	void set_default_cursor_shape(Control::CursorShape p_shape);
	Control::CursorShape get_default_cursor_shape() const;
	Control::CursorShape get_cursor_shape(const Vector2 &p_position = Vector2(0, 0)) const;
	void set_focus_neighbor(Side p_side, const NodePath &p_neighbor);
	NodePath get_focus_neighbor(Side p_side) const;
	void set_focus_next(const NodePath &p_next);
	NodePath get_focus_next() const;
	void set_focus_previous(const NodePath &p_previous);
	NodePath get_focus_previous() const;
	void force_drag(const Variant &p_data, Control *p_preview);
	void accessibility_drag();
	void accessibility_drop();
	void set_accessibility_name(const String &p_name);
	String get_accessibility_name() const;
	void set_accessibility_description(const String &p_description);
	String get_accessibility_description() const;
	void set_accessibility_live(DisplayServer::AccessibilityLiveMode p_mode);
	DisplayServer::AccessibilityLiveMode get_accessibility_live() const;
	void set_accessibility_controls_nodes(const TypedArray<NodePath> &p_node_path);
	TypedArray<NodePath> get_accessibility_controls_nodes() const;
	void set_accessibility_described_by_nodes(const TypedArray<NodePath> &p_node_path);
	TypedArray<NodePath> get_accessibility_described_by_nodes() const;
	void set_accessibility_labeled_by_nodes(const TypedArray<NodePath> &p_node_path);
	TypedArray<NodePath> get_accessibility_labeled_by_nodes() const;
	void set_accessibility_flow_to_nodes(const TypedArray<NodePath> &p_node_path);
	TypedArray<NodePath> get_accessibility_flow_to_nodes() const;
	void set_mouse_filter(Control::MouseFilter p_filter);
	Control::MouseFilter get_mouse_filter() const;
	Control::MouseFilter get_mouse_filter_with_override() const;
	void set_mouse_behavior_recursive(Control::MouseBehaviorRecursive p_mouse_behavior_recursive);
	Control::MouseBehaviorRecursive get_mouse_behavior_recursive() const;
	void set_force_pass_scroll_events(bool p_force_pass_scroll_events);
	bool is_force_pass_scroll_events() const;
	void set_clip_contents(bool p_enable);
	bool is_clipping_contents();
	void grab_click_focus();
	void set_drag_forwarding(const Callable &p_drag_func, const Callable &p_can_drop_func, const Callable &p_drop_func);
	void set_drag_preview(Control *p_control);
	bool is_drag_successful() const;
	void warp_mouse(const Vector2 &p_position);
	void set_shortcut_context(Node *p_node);
	Node *get_shortcut_context() const;
	void update_minimum_size();
	void set_layout_direction(Control::LayoutDirection p_direction);
	Control::LayoutDirection get_layout_direction() const;
	bool is_layout_rtl() const;
	void set_auto_translate(bool p_enable);
	bool is_auto_translating() const;
	void set_localize_numeral_system(bool p_enable);
	bool is_localizing_numeral_system() const;
	virtual bool _has_point(const Vector2 &p_point) const;
	virtual TypedArray<Vector3i> _structured_text_parser(const Array &p_args, const String &p_text) const;
	virtual Vector2 _get_minimum_size() const;
	virtual String _get_tooltip(const Vector2 &p_at_position) const;
	virtual Variant _get_drag_data(const Vector2 &p_at_position);
	virtual bool _can_drop_data(const Vector2 &p_at_position, const Variant &p_data) const;
	virtual void _drop_data(const Vector2 &p_at_position, const Variant &p_data);
	virtual Object *_make_custom_tooltip(const String &p_for_text) const;
	virtual String _accessibility_get_contextual_info() const;
	virtual String _get_accessibility_container_name(Node *p_node) const;
	virtual void _gui_input(const Ref<InputEvent> &p_event);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		CanvasItem::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_has_point), decltype(&T::_has_point)>) {
			BIND_VIRTUAL_METHOD(T, _has_point, 556197845);
		}
		if constexpr (!std::is_same_v<decltype(&B::_structured_text_parser), decltype(&T::_structured_text_parser)>) {
			BIND_VIRTUAL_METHOD(T, _structured_text_parser, 1292548940);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_minimum_size), decltype(&T::_get_minimum_size)>) {
			BIND_VIRTUAL_METHOD(T, _get_minimum_size, 3341600327);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_tooltip), decltype(&T::_get_tooltip)>) {
			BIND_VIRTUAL_METHOD(T, _get_tooltip, 3674420000);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_drag_data), decltype(&T::_get_drag_data)>) {
			BIND_VIRTUAL_METHOD(T, _get_drag_data, 2233896889);
		}
		if constexpr (!std::is_same_v<decltype(&B::_can_drop_data), decltype(&T::_can_drop_data)>) {
			BIND_VIRTUAL_METHOD(T, _can_drop_data, 2603004011);
		}
		if constexpr (!std::is_same_v<decltype(&B::_drop_data), decltype(&T::_drop_data)>) {
			BIND_VIRTUAL_METHOD(T, _drop_data, 3699746064);
		}
		if constexpr (!std::is_same_v<decltype(&B::_make_custom_tooltip), decltype(&T::_make_custom_tooltip)>) {
			BIND_VIRTUAL_METHOD(T, _make_custom_tooltip, 1976279298);
		}
		if constexpr (!std::is_same_v<decltype(&B::_accessibility_get_contextual_info), decltype(&T::_accessibility_get_contextual_info)>) {
			BIND_VIRTUAL_METHOD(T, _accessibility_get_contextual_info, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_accessibility_container_name), decltype(&T::_get_accessibility_container_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_accessibility_container_name, 2174079723);
		}
		if constexpr (!std::is_same_v<decltype(&B::_gui_input), decltype(&T::_gui_input)>) {
			BIND_VIRTUAL_METHOD(T, _gui_input, 3754044979);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Control::FocusMode);
VARIANT_ENUM_CAST(Control::FocusBehaviorRecursive);
VARIANT_ENUM_CAST(Control::MouseBehaviorRecursive);
VARIANT_ENUM_CAST(Control::CursorShape);
VARIANT_ENUM_CAST(Control::LayoutPreset);
VARIANT_ENUM_CAST(Control::LayoutPresetMode);
VARIANT_BITFIELD_CAST(Control::SizeFlags);
VARIANT_ENUM_CAST(Control::MouseFilter);
VARIANT_ENUM_CAST(Control::GrowDirection);
VARIANT_ENUM_CAST(Control::Anchor);
VARIANT_ENUM_CAST(Control::LayoutDirection);
VARIANT_ENUM_CAST(Control::TextDirection);

