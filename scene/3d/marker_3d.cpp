/**************************************************************************/
/*  marker_3d.cpp                                                         */
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

#include "marker_3d.h"

void Marker3D::set_gizmo_extents(real_t p_extents) {
	if (Math::is_equal_approx(gizmo_extents, p_extents)) {
		return;
	}
	gizmo_extents = p_extents;
	update_gizmos();
}

real_t Marker3D::get_gizmo_extents() const {
	return gizmo_extents;
}

void Marker3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gizmo_extents", "extents"), &Marker3D::set_gizmo_extents);
	ClassDB::bind_method(D_METHOD("get_gizmo_extents"), &Marker3D::get_gizmo_extents);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gizmo_extents", PROPERTY_HINT_RANGE, "0,10,0.01,or_greater,suffix:m"), "set_gizmo_extents", "get_gizmo_extents");
}

Marker3D::Marker3D() {
}
