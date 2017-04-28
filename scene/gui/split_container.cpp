/*************************************************************************/
/*  split_container.cpp                                                  */
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
#include "split_container.h"

#include "label.h"
#include "margin_container.h"

struct _MinSizeCache {

	int min_size;
	bool will_stretch;
	int final_size;
};

Control *SplitContainer::_getch(int p_idx) const {

	int idx = 0;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = get_child(i)->cast_to<Control>();
		if (!c || !c->is_visible_in_tree())
			continue;
		if (c->is_set_as_toplevel())
			continue;

		if (idx == p_idx)
			return c;

		idx++;
	}

	return NULL;
}

void SplitContainer::_resort() {

	/** First pass, determine minimum size AND amount of stretchable elements */

	int axis = vertical ? 1 : 0;

	bool has_first = _getch(0);
	bool has_second = _getch(1);

	if (!has_first && !has_second) {
		return;
	} else if (!(has_first && has_second)) {
		if (has_first)
			fit_child_in_rect(_getch(0), Rect2(Point2(), get_size()));
		else
			fit_child_in_rect(_getch(1), Rect2(Point2(), get_size()));

		return;
	}

	Control *first = _getch(0);
	Control *second = _getch(1);

	bool ratiomode = false;
	bool expand_first_mode = false;

	if (vertical) {

		ratiomode = first->get_v_size_flags() & SIZE_EXPAND && second->get_v_size_flags() & SIZE_EXPAND;
		expand_first_mode = first->get_v_size_flags() & SIZE_EXPAND && !(second->get_v_size_flags() & SIZE_EXPAND);
	} else {

		ratiomode = first->get_h_size_flags() & SIZE_EXPAND && second->get_h_size_flags() & SIZE_EXPAND;
		expand_first_mode = first->get_h_size_flags() & SIZE_EXPAND && !(second->get_h_size_flags() & SIZE_EXPAND);
	}

	int sep = get_constant("separation");
	Ref<Texture> g = get_icon("grabber");

	if (dragger_visibility == DRAGGER_HIDDEN_COLLAPSED) {
		sep = 0;
	} else {
		sep = MAX(sep, vertical ? g->get_height() : g->get_width());
	}

	int total = vertical ? get_size().height : get_size().width;

	total -= sep;

	int minimum = 0;

	Size2 ms_first = first->get_combined_minimum_size();
	Size2 ms_second = second->get_combined_minimum_size();

	if (vertical) {

		minimum = ms_first.height + ms_second.height;
	} else {

		minimum = ms_first.width + ms_second.width;
	}

	int available = total - minimum;
	if (available < 0)
		available = 0;

	middle_sep = 0;

	if (collapsed) {

		if (ratiomode) {

			middle_sep = ms_first[axis] + available / 2;

		} else if (expand_first_mode) {

			middle_sep = get_size()[axis] - ms_second[axis] - sep;

		} else {

			middle_sep = ms_first[axis];
		}

	} else if (ratiomode) {

		if (expand_ofs < -(available / 2))
			expand_ofs = -(available / 2);
		else if (expand_ofs > (available / 2))
			expand_ofs = (available / 2);

		middle_sep = ms_first[axis] + available / 2 + expand_ofs;

	} else if (expand_first_mode) {

		if (expand_ofs > 0)
			expand_ofs = 0;

		if (expand_ofs < -available)
			expand_ofs = -available;

		middle_sep = get_size()[axis] - ms_second[axis] - sep + expand_ofs;

	} else {

		if (expand_ofs < 0)
			expand_ofs = 0;

		if (expand_ofs > available)
			expand_ofs = available;

		middle_sep = ms_first[axis] + expand_ofs;
	}

	if (vertical) {

		fit_child_in_rect(first, Rect2(Point2(0, 0), Size2(get_size().width, middle_sep)));
		int sofs = middle_sep + sep;
		fit_child_in_rect(second, Rect2(Point2(0, sofs), Size2(get_size().width, get_size().height - sofs)));

	} else {

		fit_child_in_rect(first, Rect2(Point2(0, 0), Size2(middle_sep, get_size().height)));
		int sofs = middle_sep + sep;
		fit_child_in_rect(second, Rect2(Point2(sofs, 0), Size2(get_size().width - sofs, get_size().height)));
	}

	update();
	_change_notify("split_offset");
}

Size2 SplitContainer::get_minimum_size() const {

	/* Calculate MINIMUM SIZE */

	Size2i minimum;
	int sep = get_constant("separation");
	Ref<Texture> g = get_icon("grabber");
	sep = (dragger_visibility != DRAGGER_HIDDEN_COLLAPSED) ? MAX(sep, vertical ? g->get_height() : g->get_width()) : 0;

	for (int i = 0; i < 2; i++) {

		if (!_getch(i))
			break;

		if (i == 1) {

			if (vertical)
				minimum.height += sep;
			else
				minimum.width += sep;
		}

		Size2 ms = _getch(i)->get_combined_minimum_size();

		if (vertical) {

			minimum.height += ms.height;
			minimum.width = MAX(minimum.width, ms.width);
		} else {

			minimum.width += ms.width;
			minimum.height = MAX(minimum.height, ms.height);
		}
	}

	return minimum;
}

void SplitContainer::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_SORT_CHILDREN: {

			_resort();
		} break;
		case NOTIFICATION_MOUSE_ENTER: {
			mouse_inside = true;
			update();
		} break;
		case NOTIFICATION_MOUSE_EXIT: {
			mouse_inside = false;
			update();
		} break;
		case NOTIFICATION_DRAW: {

			if (!_getch(0) || !_getch(1))
				return;

			if (collapsed || (!mouse_inside && get_constant("autohide")))
				return;
			int sep = dragger_visibility != DRAGGER_HIDDEN_COLLAPSED ? get_constant("separation") : 0;
			Ref<Texture> tex = get_icon("grabber");
			Size2 size = get_size();
			if (vertical) {

				//draw_style_box( get_stylebox("bg"), Rect2(0,middle_sep,get_size().width,sep));
				if (dragger_visibility == DRAGGER_VISIBLE)
					draw_texture(tex, Point2i((size.x - tex->get_width()) / 2, middle_sep + (sep - tex->get_height()) / 2));

			} else {

				//draw_style_box( get_stylebox("bg"), Rect2(middle_sep,0,sep,get_size().height));
				if (dragger_visibility == DRAGGER_VISIBLE)
					draw_texture(tex, Point2i(middle_sep + (sep - tex->get_width()) / 2, (size.y - tex->get_height()) / 2));
			}

		} break;
	}
}

void SplitContainer::_gui_input(const InputEvent &p_event) {

	if (collapsed || !_getch(0) || !_getch(1) || dragger_visibility != DRAGGER_VISIBLE)
		return;

	if (p_event.type == InputEvent::MOUSE_BUTTON) {

		const InputEventMouseButton &mb = p_event.mouse_button;

		if (mb.button_index == BUTTON_LEFT) {

			if (mb.pressed) {
				int sep = get_constant("separation");

				if (vertical) {

					if (mb.y > middle_sep && mb.y < middle_sep + sep) {
						dragging = true;
						drag_from = mb.y;
						drag_ofs = expand_ofs;
					}
				} else {

					if (mb.x > middle_sep && mb.x < middle_sep + sep) {
						dragging = true;
						drag_from = mb.x;
						drag_ofs = expand_ofs;
					}
				}
			} else {

				dragging = false;
			}
		}
	}

	if (p_event.type == InputEvent::MOUSE_MOTION) {

		const InputEventMouseMotion &mm = p_event.mouse_motion;

		if (dragging) {

			expand_ofs = drag_ofs + ((vertical ? mm.y : mm.x) - drag_from);
			queue_sort();
			emit_signal("dragged", get_split_offset());
		}
	}
}

Control::CursorShape SplitContainer::get_cursor_shape(const Point2 &p_pos) const {

	if (collapsed)
		return Control::get_cursor_shape(p_pos);

	if (dragging)
		return (vertical ? CURSOR_VSIZE : CURSOR_HSIZE);

	int sep = get_constant("separation");

	if (vertical) {

		if (p_pos.y > middle_sep && p_pos.y < middle_sep + sep) {
			return CURSOR_VSIZE;
		}
	} else {

		if (p_pos.x > middle_sep && p_pos.x < middle_sep + sep) {
			return CURSOR_HSIZE;
		}
	}

	return Control::get_cursor_shape(p_pos);
}

void SplitContainer::set_split_offset(int p_offset) {

	if (expand_ofs == p_offset)
		return;
	expand_ofs = p_offset;
	queue_sort();
}

int SplitContainer::get_split_offset() const {

	return expand_ofs;
}

void SplitContainer::set_collapsed(bool p_collapsed) {

	if (collapsed == p_collapsed)
		return;
	collapsed = p_collapsed;
	queue_sort();
}

void SplitContainer::set_dragger_visibility(DraggerVisibility p_visibility) {

	dragger_visibility = p_visibility;
	queue_sort();
	update();
}

SplitContainer::DraggerVisibility SplitContainer::get_dragger_visibility() const {

	return dragger_visibility;
}

bool SplitContainer::is_collapsed() const {

	return collapsed;
}

void SplitContainer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &SplitContainer::_gui_input);
	ClassDB::bind_method(D_METHOD("set_split_offset", "offset"), &SplitContainer::set_split_offset);
	ClassDB::bind_method(D_METHOD("get_split_offset"), &SplitContainer::get_split_offset);

	ClassDB::bind_method(D_METHOD("set_collapsed", "collapsed"), &SplitContainer::set_collapsed);
	ClassDB::bind_method(D_METHOD("is_collapsed"), &SplitContainer::is_collapsed);

	ClassDB::bind_method(D_METHOD("set_dragger_visibility", "mode"), &SplitContainer::set_dragger_visibility);
	ClassDB::bind_method(D_METHOD("get_dragger_visibility"), &SplitContainer::get_dragger_visibility);

	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::INT, "offset")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "split_offset"), "set_split_offset", "get_split_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collapsed"), "set_collapsed", "is_collapsed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dragger_visibility", PROPERTY_HINT_ENUM, "Visible,Hidden,Hidden & Collapsed"), "set_dragger_visibility", "get_dragger_visibility");

	BIND_CONSTANT(DRAGGER_VISIBLE);
	BIND_CONSTANT(DRAGGER_HIDDEN);
	BIND_CONSTANT(DRAGGER_HIDDEN_COLLAPSED);
}

SplitContainer::SplitContainer(bool p_vertical) {

	mouse_inside = false;
	expand_ofs = 0;
	middle_sep = 0;
	vertical = p_vertical;
	dragging = false;
	collapsed = false;
	dragger_visibility = DRAGGER_VISIBLE;
}
