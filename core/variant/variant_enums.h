/**************************************************************************/
/*  variant_enums.h                                                       */
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

// HACK: By wrapping these enums in a namespace, we're able to hoist the enumeration itself into
//  the global namespace while keeping the enumeration constants constrained. This has a similar
//  effect to the behavior of scoped enums, where the constants will require the type preceding it,
//  but does so without changing the existing type/conversion properties. This workaround can be
//  removed if these enums ever become scoped (ie: enum class).
// NOTE: The secondary layer of wrappers, the "Scoped<EnumName>" namespaces, are required in order
//  to support enums with shared names. As of now, this is relevant for exactly one case: `MAX`.

namespace Internal {

namespace ScopedVariantType {
// WARNING: If this changes, the table in variant_op must be updated.
enum VariantType {
	NIL,

	// Atomic types.
	BOOL,
	INT,
	FLOAT,
	STRING,

	// Math types.
	VECTOR2,
	VECTOR2I,
	RECT2,
	RECT2I,
	VECTOR3,
	VECTOR3I,
	TRANSFORM2D,
	VECTOR4,
	VECTOR4I,
	PLANE,
	QUATERNION,
	AABB,
	BASIS,
	TRANSFORM3D,
	PROJECTION,

	// Misc types.
	COLOR,
	STRING_NAME,
	NODE_PATH,
	RID,
	OBJECT,
	CALLABLE,
	SIGNAL,
	DICTIONARY,
	ARRAY,

	// Packed arrays.
	PACKED_BYTE_ARRAY,
	PACKED_INT32_ARRAY,
	PACKED_INT64_ARRAY,
	PACKED_FLOAT32_ARRAY,
	PACKED_FLOAT64_ARRAY,
	PACKED_STRING_ARRAY,
	PACKED_VECTOR2_ARRAY,
	PACKED_VECTOR3_ARRAY,
	PACKED_COLOR_ARRAY,
	PACKED_VECTOR4_ARRAY,

	// Enum size.
	MAX,
};
} // namespace ScopedVariantType

namespace ScopedVariantOperator {
// WARNING: If this changes, the table in variant_op must be updated.
enum VariantOperator {
	// Comparison operators.
	EQUAL,
	NOT_EQUAL,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL,

	// Mathematic operators.
	ADD,
	SUBTRACT,
	MULTIPLY,
	DIVIDE,
	NEGATE,
	POSITIVE,
	MODULE,
	POWER,

	// Bitwise operators.
	SHIFT_LEFT,
	SHIFT_RIGHT,
	BIT_AND,
	BIT_OR,
	BIT_XOR,
	BIT_NEGATE,

	// Logical operators.
	AND,
	OR,
	XOR,
	NOT,

	// Containment operators.
	IN,

	// Enum size.
	MAX,
};

} // namespace ScopedVariantOperator
} // namespace Internal

using VariantType = Internal::ScopedVariantType::VariantType;
using VariantOperator = Internal::ScopedVariantOperator::VariantOperator;
