/**************************************************************************/
/*  plane.cpp                                                             */
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

#include "plane.h"

#include "core/variant/variant.h"

Variant Plane::intersect_3_bind(const Plane &p_plane1, const Plane &p_plane2) const {
	Vector3 inters;
	if (intersect_3(p_plane1, p_plane2, &inters)) {
		return inters;
	} else {
		return Variant();
	}
}

Variant Plane::intersects_ray_bind(const Vector3 &p_from, const Vector3 &p_dir) const {
	Vector3 inters;
	if (intersects_ray(p_from, p_dir, &inters)) {
		return inters;
	} else {
		return Variant();
	}
}

Variant Plane::intersects_segment_bind(const Vector3 &p_begin, const Vector3 &p_end) const {
	Vector3 inters;
	if (intersects_segment(p_begin, p_end, &inters)) {
		return inters;
	} else {
		return Variant();
	}
}

Plane::operator String() const {
	return "[N: " + normal.operator String() + ", D: " + String::num_real(d, false) + "]";
}
