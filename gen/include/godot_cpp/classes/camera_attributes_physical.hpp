/**************************************************************************/
/*  camera_attributes_physical.hpp                                        */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CameraAttributesPhysical : public CameraAttributes {
	GDEXTENSION_CLASS(CameraAttributesPhysical, CameraAttributes)

public:
	void set_aperture(float p_aperture);
	float get_aperture() const;
	void set_shutter_speed(float p_shutter_speed);
	float get_shutter_speed() const;
	void set_focal_length(float p_focal_length);
	float get_focal_length() const;
	void set_focus_distance(float p_focus_distance);
	float get_focus_distance() const;
	void set_near(float p_near);
	float get_near() const;
	void set_far(float p_far);
	float get_far() const;
	float get_fov() const;
	void set_auto_exposure_max_exposure_value(float p_exposure_value_max);
	float get_auto_exposure_max_exposure_value() const;
	void set_auto_exposure_min_exposure_value(float p_exposure_value_min);
	float get_auto_exposure_min_exposure_value() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		CameraAttributes::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

