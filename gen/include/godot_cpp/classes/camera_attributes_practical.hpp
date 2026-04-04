/**************************************************************************/
/*  camera_attributes_practical.hpp                                       */
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

class CameraAttributesPractical : public CameraAttributes {
	GDEXTENSION_CLASS(CameraAttributesPractical, CameraAttributes)

public:
	void set_dof_blur_far_enabled(bool p_enabled);
	bool is_dof_blur_far_enabled() const;
	void set_dof_blur_far_distance(float p_distance);
	float get_dof_blur_far_distance() const;
	void set_dof_blur_far_transition(float p_distance);
	float get_dof_blur_far_transition() const;
	void set_dof_blur_near_enabled(bool p_enabled);
	bool is_dof_blur_near_enabled() const;
	void set_dof_blur_near_distance(float p_distance);
	float get_dof_blur_near_distance() const;
	void set_dof_blur_near_transition(float p_distance);
	float get_dof_blur_near_transition() const;
	void set_dof_blur_amount(float p_amount);
	float get_dof_blur_amount() const;
	void set_auto_exposure_max_sensitivity(float p_max_sensitivity);
	float get_auto_exposure_max_sensitivity() const;
	void set_auto_exposure_min_sensitivity(float p_min_sensitivity);
	float get_auto_exposure_min_sensitivity() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		CameraAttributes::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

