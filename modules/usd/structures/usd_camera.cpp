/**************************************************************************/
/*  usd_camera.cpp                                                        */
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

#include "usd_camera.h"

#include "core/math/math_funcs.h"

void USDCamera::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_projection"), &USDCamera::get_projection);
	ClassDB::bind_method(D_METHOD("set_projection", "projection"), &USDCamera::set_projection);
	ClassDB::bind_method(D_METHOD("get_focal_length"), &USDCamera::get_focal_length);
	ClassDB::bind_method(D_METHOD("set_focal_length", "focal_length"), &USDCamera::set_focal_length);
	ClassDB::bind_method(D_METHOD("get_horizontal_aperture"), &USDCamera::get_horizontal_aperture);
	ClassDB::bind_method(D_METHOD("set_horizontal_aperture", "horizontal_aperture"), &USDCamera::set_horizontal_aperture);
	ClassDB::bind_method(D_METHOD("get_vertical_aperture"), &USDCamera::get_vertical_aperture);
	ClassDB::bind_method(D_METHOD("set_vertical_aperture", "vertical_aperture"), &USDCamera::set_vertical_aperture);
	ClassDB::bind_method(D_METHOD("get_near_clip"), &USDCamera::get_near_clip);
	ClassDB::bind_method(D_METHOD("set_near_clip", "near_clip"), &USDCamera::set_near_clip);
	ClassDB::bind_method(D_METHOD("get_far_clip"), &USDCamera::get_far_clip);
	ClassDB::bind_method(D_METHOD("set_far_clip", "far_clip"), &USDCamera::set_far_clip);
	ClassDB::bind_method(D_METHOD("get_fov"), &USDCamera::get_fov);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "projection", PROPERTY_HINT_ENUM, "Perspective,Orthographic"), "set_projection", "get_projection");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "focal_length"), "set_focal_length", "get_focal_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "horizontal_aperture"), "set_horizontal_aperture", "get_horizontal_aperture");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vertical_aperture"), "set_vertical_aperture", "get_vertical_aperture");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "near_clip"), "set_near_clip", "get_near_clip");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "far_clip"), "set_far_clip", "get_far_clip");

	BIND_ENUM_CONSTANT(PERSPECTIVE);
	BIND_ENUM_CONSTANT(ORTHOGRAPHIC);
}

USDCamera::ProjectionType USDCamera::get_projection() const {
	return projection;
}

void USDCamera::set_projection(ProjectionType p_projection) {
	projection = p_projection;
}

float USDCamera::get_focal_length() const {
	return focal_length;
}

void USDCamera::set_focal_length(float p_focal_length) {
	focal_length = p_focal_length;
}

float USDCamera::get_horizontal_aperture() const {
	return horizontal_aperture;
}

void USDCamera::set_horizontal_aperture(float p_horizontal_aperture) {
	horizontal_aperture = p_horizontal_aperture;
}

float USDCamera::get_vertical_aperture() const {
	return vertical_aperture;
}

void USDCamera::set_vertical_aperture(float p_vertical_aperture) {
	vertical_aperture = p_vertical_aperture;
}

float USDCamera::get_near_clip() const {
	return near_clip;
}

void USDCamera::set_near_clip(float p_near_clip) {
	near_clip = p_near_clip;
}

float USDCamera::get_far_clip() const {
	return far_clip;
}

void USDCamera::set_far_clip(float p_far_clip) {
	far_clip = p_far_clip;
}

float USDCamera::get_fov() const {
	// USD stores camera parameters as focal length (mm) and aperture (mm).
	// Vertical FOV = 2 * atan(vertical_aperture / (2 * focal_length)).
	// Return the result in degrees for Godot compatibility.
	if (focal_length <= 0.0) {
		return 75.0; // Godot default.
	}
	return Math::rad_to_deg(2.0 * Math::atan(vertical_aperture / (2.0 * focal_length)));
}
