/**************************************************************************/
/*  interpolated_property.h                                               */
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

#ifndef INTERPOLATED_PROPERTY_H
#define INTERPOLATED_PROPERTY_H

#include "core/math/transform_interpolator.h"
#include <type_traits>

// This class is intended to reduce the boiler plate involved to
// support custom properties to be physics interpolated.

template <class T>
class InterpolatedProperty {
	// Only needs interpolating / updating the servers when
	// curr and prev are different.
	bool _needs_interpolating = false;
	T _interp;
	T curr;
	T prev;

public:
	void pump() {
		prev = curr;
		_needs_interpolating = false;
	}
	void reset() { pump(); }

	void set_interpolated_value(const T &p_val) {
		_interp = p_val;
	}
	const T &interp() const { return _interp; }
	bool needs_interpolating() const { return _needs_interpolating; }

	bool interpolate(float p_interpolation_fraction) {
		if (_needs_interpolating) {
			_interp = TransformInterpolator::interpolated_property_lerp(prev, curr, p_interpolation_fraction);
			return true;
		}
		return false;
	}

	operator T() const {
		return curr;
	}

	bool operator==(const T &p_o) const {
		return p_o == *this;
	}

	bool operator!=(const T &p_o) const {
		return p_o != *this;
	}

	InterpolatedProperty &operator=(T p_val) {
		curr = p_val;
		_interp = p_val;
		_needs_interpolating = true;
		return *this;
	}
	InterpolatedProperty(T p_val) {
		curr = p_val;
		_interp = p_val;
		pump();
	}
	InterpolatedProperty() {
		// Ensure either the constructor is run,
		// or the memory is zeroed if using a fundamental type.
		_interp = T{};
		curr = T{};
		prev = T{};
	}
};

#endif // INTERPOLATED_PROPERTY_H
