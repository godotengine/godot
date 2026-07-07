/**************************************************************************/
/*  openxr_uuid.h                                                         */
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

// Godot helper functions for OpenXR XrUuidExt data type
#include "core/templates/hashfuncs.h"

#include <openxr/openxr.h>

struct HashMapHasherXrUuidEXT {
	static _FORCE_INLINE_ uint32_t hash(const XrUuidEXT &p_uuid) { return hash_murmur3_buffer(p_uuid.data, XR_UUID_SIZE_EXT); }
};

template <>
struct HashMapComparatorDefault<XrUuidEXT> {
	static bool compare(const XrUuidEXT &p_lhs, const XrUuidEXT &p_rhs) {
		for (int i = 0; i < XR_UUID_SIZE_EXT; i++) {
			if (p_lhs.data[i] != p_rhs.data[i]) {
				return false;
			}
		}
		return true;
	}
};
