/**************************************************************************/
/*  transform_interpolator.cpp                                            */
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

#include "transform_interpolator.h"

#include "core/math/transform_2d.h"

void TransformInterpolator::interpolate_transform_2d(const Transform2D &p_prev, const Transform2D &p_curr, Transform2D &r_result, real_t p_fraction) {
	// Special case for physics interpolation, if flipping, don't interpolate basis.
	// If the determinant polarity changes, the handedness of the coordinate system changes.
	if (_sign(p_prev.determinant()) != _sign(p_curr.determinant())) {
		r_result.columns[0] = p_curr.columns[0];
		r_result.columns[1] = p_curr.columns[1];
		r_result.set_origin(p_prev.get_origin().lerp(p_curr.get_origin(), p_fraction));
		return;
	}

	r_result = p_prev.interpolate_with(p_curr, p_fraction);
}
