/**************************************************************************/
/*  fitting_container.cpp                                                 */
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

#include "fitting_container.h"

#include "core/object/object.h"
#include "scene/gui/container.h"

void FittingContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			_update_active_control();
		} break;
	}
}

Size2 FittingContainer::get_minimum_size() const {
	return Size2();
}

Control *FittingContainer::get_active_control() const {
	return _active_control;
}

void FittingContainer::_update_active_control() {
	Size2 size = get_size();

	_active_control = nullptr;
	Control *last_control = nullptr;
	for (int i = 0; i < get_child_count(); i++) {
		Control *current_control = as_sortable_control(get_child(i), SortableVisibilityMode::IGNORE);
		if (!current_control) {
			continue;
		}
		last_control = current_control;

		// If axis is set to NONE, we actually activate the first control.
		if (_axis == FITTING_AXIS_NONE) {
			_active_control = last_control;
			break;
		}
		if ((_axis == FITTING_AXIS_HORIZONTAL || _axis == FITTING_AXIS_BOTH) && current_control->get_combined_minimum_size().x > size.x) {
			continue;
		}
		if ((_axis == FITTING_AXIS_VERTICAL || _axis == FITTING_AXIS_BOTH) && current_control->get_combined_minimum_size().y > size.y) {
			continue;
		}

		_active_control = current_control;
		break;
	}

	if (!_active_control && last_control) {
		_active_control = last_control;
	}
	if (!_active_control) {
		return;
	}

	for (int i = 0; i < get_child_count(); i++) {
		Control *current_control = static_cast<Control *>(get_child(i));
		if (!current_control) {
			continue;
		}

		current_control->set_visible(current_control == _active_control);
		current_control->set_position(Point2());
	}
}

void FittingContainer::set_axis(FittingAxis p_axis) {
	_axis = p_axis;
}

FittingContainer::FittingAxis FittingContainer::get_axis() const {
	return _axis;
}

void FittingContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_axis", "axis"), &FittingContainer::set_axis);
	ClassDB::bind_method(D_METHOD("get_axis"), &FittingContainer::get_axis);
	ClassDB::bind_method(D_METHOD("get_active_control"), &FittingContainer::get_active_control);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "axis", PROPERTY_HINT_ENUM, "None,Horizontal,Vertical,Both"), "set_axis", "get_axis");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "active_control", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_active_control");
}
