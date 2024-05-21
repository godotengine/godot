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

#ifndef SPLIT_CONTAINER_H
#define SPLIT_CONTAINER_H

#include "scene/gui/container.h"

class SplitContainerDragger : public Control {
	GDCLASS(SplitContainerDragger, Control);

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

private:
	bool dragging = false;
	int drag_from = 0;
	int drag_ofs = 0;
	bool mouse_inside = false;

public:
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;
};

class SplitContainer : public Container {
	GDCLASS(SplitContainer, Container);
	friend class SplitContainerDragger;

public:
	enum DraggerVisibility {
		DRAGGER_VISIBLE,
		DRAGGER_HIDDEN,
		DRAGGER_HIDDEN_COLLAPSED
	};

private:
	int split_offset = 0;
	int middle_sep = 0;
	bool vertical = false;
	bool collapsed = false;
	DraggerVisibility dragger_visibility = DRAGGER_VISIBLE;

	SplitContainerDragger *dragging_area_control = nullptr;

	struct ThemeCache {
		int separation = 0;
		int minimum_grab_thickness = 0;
		bool autohide = false;
		Ref<Texture2D> grabber_icon;
		Ref<Texture2D> grabber_icon_h;
		Ref<Texture2D> grabber_icon_v;
	} theme_cache;

	Ref<Texture2D> _get_grabber_icon() const;
	void _compute_middle_sep(bool p_clamp);
	void _resort();
	Control *_get_sortable_child(int p_idx) const;

protected:
	bool is_fixed = false;

	void _notification(int p_what);
	void _validate_property(PropertyInfo &p_property) const;
	static void _bind_methods();

public:
	void set_split_offset(int p_offset);
	int get_split_offset() const;
	void clamp_split_offset();

	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;

	void set_dragger_visibility(DraggerVisibility p_visibility);
	DraggerVisibility get_dragger_visibility() const;

	void set_vertical(bool p_vertical);
	bool is_vertical() const;

	virtual Size2 get_minimum_size() const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

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

#endif // SPLIT_CONTAINER_H
