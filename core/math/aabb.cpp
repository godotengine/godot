/**************************************************************************/
/*  aabb.cpp                                                              */
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

#include "aabb.h"

#include "core/string/ustring.h"
#include "core/variant/variant.h"

Variant AABB::intersects_segment_bind(const Vector3 &p_from, const Vector3 &p_to) const {
	Vector3 inters;
	if (intersects_segment(p_from, p_to, &inters)) {
		return inters;
	}
	return Variant();
}

Variant AABB::intersects_ray_bind(const Vector3 &p_from, const Vector3 &p_dir) const {
	Vector3 inters;
	bool inside = false;

	if (find_intersects_ray(p_from, p_dir, inside, &inters)) {
		// When inside the intersection point may be BEHIND the ray,
		// so for general use we return the ray origin.
		if (inside) {
			return p_from;
		}

		return inters;
	}
	return Variant();
}

AABB::operator String() const {
	return "[P: " + position.operator String() + ", S: " + size + "]";
}
