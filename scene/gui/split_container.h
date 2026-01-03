/**************************************************************************/
/*  split_container.h                                                     */
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

#include "scene/gui/container.h"

class TextureRect;

class SplitContainerDragger : public Control {
	GDCLASS(SplitContainerDragger, Control);
	friend class SplitContainer;

	Rect2 split_bar_rect;
	TextureRect *touch_dragger = nullptr;

	void _touch_dragger_mouse_exited();
	void _touch_dragger_gui_input(const Ref<InputEvent> &p_event);

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

	void _accessibility_action_inc(const Variant &p_data);
	void _accessibility_action_dec(const Variant &p_data);
	void _accessibility_action_set_value(const Variant &p_data);

private:
	bool dragging = false;
	int drag_from = 0;
	int start_drag_split_offset = 0;
	bool mouse_inside = false;

public:
	int dragger_index = -1;

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	void set_touch_dragger_enabled(bool p_enabled);
	void update_touch_dragger();
	bool is_touch_dragger_enabled() const;

	SplitContainerDragger();
};

class SplitContainer : public Container {
	GDCLASS(SplitContainer, Container);
	friend class SplitContainerDragger;
	friend class ContainerEditorPlugin;

public:
	enum DraggerVisibility {
		DRAGGER_VISIBLE,
		DRAGGER_HIDDEN,
		DRAGGER_HIDDEN_COLLAPSED
	};

private:
	bool show_drag_area = false;
	int drag_area_margin_begin = 0;
	int drag_area_margin_end = 0;
	int drag_area_offset = 0;

	PackedInt32Array split_offsets;
	LocalVector<int> default_dragger_positions;
	LocalVector<int> dragger_positions;
	LocalVector<Control *> valid_children;
	LocalVector<SplitContainerDragger *> dragging_area_controls;

	bool vertical = false;
	bool collapsed = false;
	DraggerVisibility dragger_visibility = DRAGGER_VISIBLE;
	bool dragging_enabled = true;
	bool split_offset_pending = false;
	bool can_use_desired_sizes = false;
	bool initialized = false;

	bool touch_dragger_enabled = false;

	struct ThemeCache {
		Color touch_dragger_color;
		Color touch_dragger_pressed_color;
		Color touch_dragger_hover_color;
		int separation = 0;
		int minimum_grab_thickness = 0;
		bool autohide = false;
		Ref<Texture2D> touch_dragger_icon;
		Ref<Texture2D> touch_dragger_icon_h;
		Ref<Texture2D> touch_dragger_icon_v;
		Ref<Texture2D> grabber_icon;
		Ref<Texture2D> grabber_icon_h;
		Ref<Texture2D> grabber_icon_v;
		float base_scale = 1.0;
		Ref<StyleBox> split_bar_background;
	} theme_cache;

	Ref<Texture2D> _get_grabber_icon() const;
	Ref<Texture2D> _get_touch_dragger_icon() const;
	Point2i _get_valid_range(int p_dragger_index) const;

	PackedInt32Array _get_desired_sizes() const;
	void _set_desired_sizes(const PackedInt32Array &p_desired_sizes, int p_priority_index = -1);

	void _update_default_dragger_positions();
	void _update_dragger_positions(int p_clamp_index = -1);
	int _get_separation() const;
	void _resort();
	void _update_draggers();
	void _on_child_visibility_changed(Control *p_control);
	void _add_valid_child(Control *p_control);
	void _remove_valid_child(Control *p_control);

protected:
	bool is_fixed = false;

	void _notification(int p_what);
	void _validate_property(PropertyInfo &p_property) const;

	virtual void add_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;
	virtual void move_child_notify(Node *p_child) override;
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	void _clamp_split_offset_compat_90411();
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	void set_split_offset(int p_offset, int p_index = 0);
	int get_split_offset(int p_index = 0) const;

	void set_split_offsets(const PackedInt32Array &p_offsets);
	PackedInt32Array get_split_offsets() const;

	void clamp_split_offset(int p_priority_index = 0);

	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;

	void set_dragger_visibility(DraggerVisibility p_visibility);
	DraggerVisibility get_dragger_visibility() const;

	void set_vertical(bool p_vertical);
	bool is_vertical() const;

	void set_dragging_enabled(bool p_enabled);
	bool is_dragging_enabled() const;

	virtual Size2 get_minimum_size() const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	void set_drag_area_margin_begin(int p_margin);
	int get_drag_area_margin_begin() const;

	void set_drag_area_margin_end(int p_margin);
	int get_drag_area_margin_end() const;

	void set_drag_area_offset(int p_offset);
	int get_drag_area_offset() const;

	void set_show_drag_area_enabled(bool p_enabled);
	bool is_show_drag_area_enabled() const;

	TypedArray<Control> get_drag_area_controls();

	void set_touch_dragger_enabled(bool p_enabled);
	bool is_touch_dragger_enabled() const;

#ifndef DISABLE_DEPRECATED
	Control *get_drag_area_control() { return dragging_area_controls[0]; }
	void _set_split_offset_first(int p_offset) { set_split_offset(p_offset); }
	int _get_split_offset_first() const { return get_split_offset(); }
#endif

	SplitContainer(bool p_vertical = false);
};

VARIANT_ENUM_CAST(SplitContainer::DraggerVisibility);

class HSplitContainer : public SplitContainer {
	GDCLASS(HSplitContainer, SplitContainer);

public:
	HSplitContainer() :
			SplitContainer(false) { is_fixed = true; }
};

class VSplitContainer : public SplitContainer {
	GDCLASS(VSplitContainer, SplitContainer);

public:
	VSplitContainer() :
			SplitContainer(true) { is_fixed = true; }
};
