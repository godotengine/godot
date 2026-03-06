/**************************************************************************/
/*  sticky_container.h                                                    */
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

class ScrollContainer;

class StickyContainer : public Container {
	GDCLASS(StickyContainer, Container);

public:
	enum StickyStatus {
		STATUS_NORMAL,
		STATUS_STICKING,
		STATUS_HIDDEN,
	};

private:
	ScrollContainer *scroll_container = nullptr;
	Control *bounding_container = nullptr;
	NodePath bounding_path;
	Vector<ObjectID> managed_children;
	StickyStatus sticky_status = STATUS_NORMAL;

	BitField<Side> sticky_sides = (1 << SIDE_LEFT) + (1 << SIDE_TOP);
	bool stacking = false;

	void _reparent_children();

protected:
	Size2 get_minimum_size() const override;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_side_sticky(Side p_side, bool p_sticky);
	bool is_side_sticky(Side p_side);

	void set_stacking(bool p_enabled);
	bool is_stacking();

	void set_sticky_status(StickyStatus p_status);
	StickyStatus get_sticky_status();

	void set_bounding_container_path(const NodePath &p_path);
	NodePath get_bounding_container_path();
	Control *get_bounding_container();

	int get_sticky_child_count();
	Control *get_sticky_child(int p_i);

	StickyContainer();
};

VARIANT_ENUM_CAST(StickyContainer::StickyStatus);
