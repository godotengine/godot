/**************************************************************************/
/*  canvas_item.cpp                                                       */
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

#ifdef TOOLS_ENABLED

bool CanvasItem::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	if (_edit_use_rect()) {
		return _edit_get_rect().has_point(p_point);
	} else {
		return p_point.length() < p_tolerance;
	}
}

Transform2D CanvasItem::_edit_get_transform() const {
	return Transform2D(_edit_get_rotation(), _edit_get_position() + _edit_get_pivot());
}

void CanvasItem::_bind_editor_methods() {
	ClassDB::bind_method(D_METHOD("_edit_set_state", "state"), &CanvasItem::_edit_set_state);
	ClassDB::bind_method(D_METHOD("_edit_get_state"), &CanvasItem::_edit_get_state);
	ClassDB::bind_method(D_METHOD("_edit_set_position", "position"), &CanvasItem::_edit_set_position);
	ClassDB::bind_method(D_METHOD("_edit_get_position"), &CanvasItem::_edit_get_position);
	ClassDB::bind_method(D_METHOD("_edit_set_scale", "scale"), &CanvasItem::_edit_set_scale);
	ClassDB::bind_method(D_METHOD("_edit_get_scale"), &CanvasItem::_edit_get_scale);
	ClassDB::bind_method(D_METHOD("_edit_set_rect", "rect"), &CanvasItem::_edit_set_rect);
	ClassDB::bind_method(D_METHOD("_edit_get_rect"), &CanvasItem::_edit_get_rect);
	ClassDB::bind_method(D_METHOD("_edit_use_rect"), &CanvasItem::_edit_use_rect);
	ClassDB::bind_method(D_METHOD("_edit_set_rotation", "degrees"), &CanvasItem::_edit_set_rotation);
	ClassDB::bind_method(D_METHOD("_edit_get_rotation"), &CanvasItem::_edit_get_rotation);
	ClassDB::bind_method(D_METHOD("_edit_use_rotation"), &CanvasItem::_edit_use_rotation);
	ClassDB::bind_method(D_METHOD("_edit_set_pivot", "pivot"), &CanvasItem::_edit_set_pivot);
	ClassDB::bind_method(D_METHOD("_edit_get_pivot"), &CanvasItem::_edit_get_pivot);
	ClassDB::bind_method(D_METHOD("_edit_use_pivot"), &CanvasItem::_edit_use_pivot);
	ClassDB::bind_method(D_METHOD("_edit_get_transform"), &CanvasItem::_edit_get_transform);
}

#endif
