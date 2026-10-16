/**************************************************************************/
/*  interop_types.h                                                       */
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

#include "core/math/math_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Keep in sync with modules/mono/glue/GodotSharp/GodotSharp/Core/NativeInterop/InteropStructs.cs,
// and C# math structs.

typedef struct {
	struct godot_variant_obj_data {
		uint64_t _dont_touch_that_0;
		void *_dont_touch_that_1;
	};
	struct godot_variant_vector4 {
		real_t _dont_touch_that_0;
		real_t _dont_touch_that_1;
		real_t _dont_touch_that_2;
		real_t _dont_touch_that_3;
	};
	int32_t _dont_touch_that_0;
	union {
		void *_dont_touch_that_0;
		struct godot_variant_obj_data _dont_touch_that_1;
		struct godot_variant_vector4 _dont_touch_that_2;
	} _dont_touch_that_1 alignas(8);
} godot_variant;

typedef struct {
	union {
		uint8_t _dont_touch_that_0;
		void *_dont_touch_that_1;
	};
} godot_array;

typedef struct {
	union {
		uint8_t _dont_touch_that_0;
		void *_dont_touch_that_1;
	};
} godot_dictionary;

typedef struct {
	void *_dont_touch_that;
} godot_string;

typedef struct {
	void *_dont_touch_that;
} godot_string_name;

typedef struct {
	void *_dont_touch_that_0;
	void *_dont_touch_that_1;
} godot_packed_array;

typedef struct {
	real_t _dont_touch_that_0;
	real_t _dont_touch_that_1;
} godot_vector2;

typedef struct {
	int32_t _dont_touch_that_0;
	int32_t _dont_touch_that_1;
} godot_vector2i;

typedef struct {
	godot_vector2 _dont_touch_that_0;
	godot_vector2 _dont_touch_that_1;
} godot_rect2;

typedef struct {
	godot_vector2i _dont_touch_that_0;
	godot_vector2i _dont_touch_that_1;
} godot_rect2i;

typedef struct {
	real_t _dont_touch_that_0;
	real_t _dont_touch_that_1;
	real_t _dont_touch_that_2;
} godot_vector3;

typedef struct {
	int32_t _dont_touch_that_0;
	int32_t _dont_touch_that_1;
	int32_t _dont_touch_that_2;
} godot_vector3i;

typedef struct {
	godot_vector2 _dont_touch_that_0;
	godot_vector2 _dont_touch_that_1;
	godot_vector2 _dont_touch_that_2;
} godot_transform2d;

typedef struct {
	real_t _dont_touch_that_0;
	real_t _dont_touch_that_1;
	real_t _dont_touch_that_2;
	real_t _dont_touch_that_3;
} godot_vector4;

typedef struct {
	int32_t _dont_touch_that_0;
	int32_t _dont_touch_that_1;
	int32_t _dont_touch_that_2;
	int32_t _dont_touch_that_3;
} godot_vector4i;

typedef struct {
	godot_vector3 _dont_touch_that_0;
	real_t _dont_touch_that_1;
} godot_plane;

typedef struct {
	real_t _dont_touch_that_0;
	real_t _dont_touch_that_1;
	real_t _dont_touch_that_2;
	real_t _dont_touch_that_3;
} godot_quaternion;

typedef struct {
	godot_vector3 _dont_touch_that_0;
	godot_vector3 _dont_touch_that_1;
} godot_aabb;

typedef struct {
	godot_vector3 _dont_touch_that_0;
	godot_vector3 _dont_touch_that_1;
	godot_vector3 _dont_touch_that_2;
} godot_basis;

typedef struct {
	godot_basis _dont_touch_that_0;
	godot_vector3 _dont_touch_that_1;
} godot_transform3d;

typedef struct {
	godot_vector4 _dont_touch_that_0;
	godot_vector4 _dont_touch_that_1;
	godot_vector4 _dont_touch_that_2;
	godot_vector4 _dont_touch_that_3;
} godot_projection;

// Colors should always use 32-bit floats, so don't use real_t here.
typedef struct {
	float _dont_touch_that_0;
	float _dont_touch_that_1;
	float _dont_touch_that_2;
	float _dont_touch_that_3;
} godot_color;

typedef struct {
	void *_dont_touch_that;
} godot_node_path;

typedef struct {
	uint64_t _dont_touch_that;
} godot_rid;

typedef struct {
	union {
		uint8_t _dont_touch_that_0;
		godot_string_name _dont_touch_that_1;
	} _dont_touch_that_0;
	union {
		uint64_t _dont_touch_that_0;
		void *_dont_touch_that_1;
	} _dont_touch_that_1 alignas(8);
} godot_callable;

typedef struct {
	union {
		uint8_t _dont_touch_that_0;
		godot_string_name _dont_touch_that_1;
	} _dont_touch_that_0;
	uint64_t _dont_touch_that_1 alignas(8);
} godot_signal;

#ifdef __cplusplus
}
#endif
