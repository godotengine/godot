/**************************************************************************/
/*  texture_pivot_utils.cpp                                               */
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

#include "texture_pivot_utils.h"

Point2 TexturePivotUtils::get_pivot(const Ref<Texture2D> &p_texture, const Size2 &p_size, const Point2 &p_offset, Texture2D::Pivot p_mode) {
	Point2 pivot;
	switch (p_mode) {
		case Texture2D::PIVOT_ANCHOR: {
			pivot = p_texture->get_anchor();
		} break;
		// PIVOT_FREE and PIVOT_FREE_RELATIVE do not need to be inverted
		// so we can immediately return the pivot.
		case Texture2D::PIVOT_FREE:
			return p_offset;
		case Texture2D::PIVOT_FREE_RELATIVE:
			return p_offset * p_size;
		// Top left pivot is just 0,0 so no need to do anything here.
		case Texture2D::PIVOT_TOP_LEFT:
			break;
		case Texture2D::PIVOT_CENTER: {
			pivot = p_size / 2;
		} break;
		case Texture2D::PIVOT_TOP_CENTER: {
			pivot = Point2(p_size.width / 2, 0);
		} break;
		case Texture2D::PIVOT_TOP_RIGHT: {
			pivot = Point2(p_size.width, 0);
		} break;
		case Texture2D::PIVOT_CENTER_RIGHT: {
			pivot = Point2(p_size.width, p_size.height / 2);
		} break;
		case Texture2D::PIVOT_BOTTOM_RIGHT: {
			pivot = p_size;
		} break;
		case Texture2D::PIVOT_BOTTOM_CENTER: {
			pivot = Point2(p_size.width / 2, p_size.height);
		} break;
		case Texture2D::PIVOT_BOTTOM_LEFT: {
			pivot = Point2(0, p_size.height);
		} break;
		case Texture2D::PIVOT_CENTER_LEFT: {
			pivot = Point2(0, p_size.height / 2);
		} break;
#ifndef DISABLE_DEPRECATED
		// Legacy mode is basically centered + offset
		case Texture2D::PIVOT_LEGACY_CENTER:
			return p_offset - p_size / 2;
#endif
	}
	// Pivot must be subtracted from position so we invert it here.
	return pivot * -1;
}
