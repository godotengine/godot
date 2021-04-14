/*************************************************************************/
/*  android_pre_rotation.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "android_pre_rotation.h"

AndroidPreRotation::AndroidPreRotation() {
	const Vector3 axis(0, 0, -1);

	rotation90.rotate(axis, Math_PI / 2.0f);
	rotation180.rotate(axis, Math_PI);
	rotation270.rotate(axis, Math_PI + Math_PI / 2.0f);
}

AndroidPreRotation &AndroidPreRotation::get_instance() {
	static AndroidPreRotation instance;

	return instance;
}

void AndroidPreRotation::set_current_transform(VkSurfaceTransformFlagBitsKHR currentTransform) {
	isPreRotationRequired = true;
	isSizeSwapRequired = false;

	switch (currentTransform) {
		case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
			currentRotation = rotation90;
			isSizeSwapRequired = true;
			break;

		case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
			currentRotation = rotation180;
			break;

		case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
			currentRotation = rotation270;
			isSizeSwapRequired = true;
			break;

		default:
			isPreRotationRequired = false;
	}
}

bool AndroidPreRotation::is_pre_rotation_required() const {
	return isPreRotationRequired;
}

bool AndroidPreRotation::is_size_swap_required() const {
	return isSizeSwapRequired;
}

const Transform3D &AndroidPreRotation::get_pre_rotation_transform() const {
	return currentRotation;
}
