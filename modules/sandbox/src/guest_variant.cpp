/**************************************************************************/
/*  guest_variant.cpp                                                     */
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

#include "guest_datatypes.h"

#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include <libriscv/util/crc32.hpp>

// Note: GuestVariant method implementations have been moved to sandbox.cpp
// to avoid circular include dependencies.

static const char *variant_type_names[] = {
	"Nil",

	"Bool", // 1
	"Int",
	"Float",
	"String",

	"Vector2", // 5
	"Vector2i",
	"Rect2",
	"Rect2i",
	"Vector3",
	"Vector3i",
	"Transform2D",
	"Vector4",
	"Vector4i",
	"Plane",
	"Quaternion",
	"AABB",
	"Basis",
	"Transform3D",
	"Projection",

	"Color", // 20
	"StringName",
	"NodePath",
	"RID",
	"Object",
	"Callable",
	"Signal",
	"Dictionary",
	"Array",

	"PackedByteArray", // 29
	"PackedInt32Array",
	"PackedInt64Array",
	"PackedFloat32Array",
	"PackedFloat64Array",
	"PackedStringArray",
	"PackedVector2Array",
	"PackedVector3Array",
	"PackedColorArray",
	"PackedVector4Array",

	"Max",
};

const char *GuestVariant::type_name(int type) {
	if (type < 0 || type >= Variant::VARIANT_MAX) {
		return "Unknown";
	}
	return variant_type_names[type];
}
