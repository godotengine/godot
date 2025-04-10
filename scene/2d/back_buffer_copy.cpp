/**************************************************************************/
/*  back_buffer_copy.cpp                                                  */
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

#include "back_buffer_copy.h"

void BackBufferCopy::_update_copy_mode() {
	switch (copy_mode) {
		case COPY_MODE_DISABLED: {
			RS::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(), false, Rect2());
		} break;
		case COPY_MODE_RECT: {
			RS::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(), true, rect);
		} break;
		case COPY_MODE_VIEWPORT: {
			RS::get_singleton()->canvas_item_set_copy_to_backbuffer(get_canvas_item(), true, Rect2());

		} break;
	}
}

#ifdef DEBUG_ENABLED
Rect2 BackBufferCopy::_edit_get_rect() const {
	return rect;
}

bool BackBufferCopy::_edit_use_rect() const {
	return true;
}
#endif // DEBUG_ENABLED

Rect2 BackBufferCopy::get_anchorable_rect() const {
	return rect;
}

void BackBufferCopy::set_rect(const Rect2 &p_rect) {
	rect = p_rect;
	_update_copy_mode();
	item_rect_changed();
}

Rect2 BackBufferCopy::get_rect() const {
	return rect;
}

void BackBufferCopy::set_copy_mode(CopyMode p_mode) {
	copy_mode = p_mode;
	_update_copy_mode();
	notify_property_list_changed();
}

BackBufferCopy::CopyMode BackBufferCopy::get_copy_mode() const {
	return copy_mode;
}

void BackBufferCopy::_validate_property(PropertyInfo &p_property) const {
	if (copy_mode != COPY_MODE_RECT && p_property.name == "rect") {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void BackBufferCopy::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_rect", "rect"), &BackBufferCopy::set_rect);
	ClassDB::bind_method(D_METHOD("get_rect"), &BackBufferCopy::get_rect);

	ClassDB::bind_method(D_METHOD("set_copy_mode", "copy_mode"), &BackBufferCopy::set_copy_mode);
	ClassDB::bind_method(D_METHOD("get_copy_mode"), &BackBufferCopy::get_copy_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "copy_mode", PROPERTY_HINT_ENUM, "Disabled,Rect,Viewport"), "set_copy_mode", "get_copy_mode");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "rect", PROPERTY_HINT_NONE, "suffix:px"), "set_rect", "get_rect");

	BIND_ENUM_CONSTANT(COPY_MODE_DISABLED);
	BIND_ENUM_CONSTANT(COPY_MODE_RECT);
	BIND_ENUM_CONSTANT(COPY_MODE_VIEWPORT);
}

BackBufferCopy::BackBufferCopy() {
	_update_copy_mode();
	set_hide_clip_children(true);
}

BackBufferCopy::~BackBufferCopy() {
}
