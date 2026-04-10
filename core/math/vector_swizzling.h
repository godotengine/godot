/**************************************************************************/
/*  vector_swizzling.h                                                    */
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

#define VECTOR_SWIZZLING_SETGET() \
	template <typename V> \
	void swizzled_set(int comp1, int comp2, const V &other) { \
		coord[comp1] = other[0]; \
		coord[comp2] = other[1]; \
	} \
	template <typename V> \
	V swizzled_get(int comp1, int comp2) const { \
		return V( \
				coord[comp1], \
				coord[comp2]); \
	} \
	template <typename V> \
	void swizzled_set(int comp1, int comp2, int comp3, const V &other) { \
		coord[comp1] = other[0]; \
		coord[comp2] = other[1]; \
		coord[comp3] = other[2]; \
	} \
	template <typename V> \
	V swizzled_get(int comp1, int comp2, int comp3) const { \
		return V( \
				coord[comp1], \
				coord[comp2], \
				coord[comp3]); \
	} \
	template <typename V> \
	void swizzled_set(int comp1, int comp2, int comp3, int comp4, const V &other) { \
		coord[comp1] = other[0]; \
		coord[comp2] = other[1]; \
		coord[comp3] = other[2]; \
		coord[comp4] = other[3]; \
	} \
	template <typename V> \
	V swizzled_get(int comp1, int comp2, int comp3, int comp4) const { \
		return V( \
				coord[comp1], \
				coord[comp2], \
				coord[comp3], \
				coord[comp4]); \
	}

#define COLOR_SWIZZLING_SETGET() \
	template <typename V> \
	void swizzled_set(int comp1, int comp2, const V &other) { \
		components[comp1] = other[0]; \
		components[comp2] = other[1]; \
	} \
	template <typename V> \
	V swizzled_get(int comp1, int comp2) const { \
		return V( \
				components[comp1], \
				components[comp2]); \
	} \
	template <typename V> \
	void swizzled_set(int comp1, int comp2, int comp3, const V &other) { \
		components[comp1] = other[0]; \
		components[comp2] = other[1]; \
		components[comp3] = other[2]; \
	} \
	template <typename V> \
	V swizzled_get(int comp1, int comp2, int comp3) const { \
		return V( \
				components[comp1], \
				components[comp2], \
				components[comp3]); \
	} \
	template <typename V> \
	void swizzled_set(int comp1, int comp2, int comp3, int comp4, const V &other) { \
		components[comp1] = other[0]; \
		components[comp2] = other[1]; \
		components[comp3] = other[2]; \
		components[comp4] = other[3]; \
	} \
	template <typename V> \
	V swizzled_get(int comp1, int comp2, int comp3, int comp4) const { \
		return V( \
				components[comp1], \
				components[comp2], \
				components[comp3], \
				components[comp4]); \
	}
