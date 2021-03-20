/*************************************************************************/
/*  camera_effects.h                                                     */
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

#ifndef CAMERA_EFFECTS_H
#define CAMERA_EFFECTS_H

#include "core/io/resource.h"
#include "core/templates/rid.h"

class CameraEffects : public Resource {
	GDCLASS(CameraEffects, Resource);

private:
	RID camera_effects;

	// DOF blur
	bool dof_blur_far_enabled = false;
	float dof_blur_far_distance = 10.0;
	float dof_blur_far_transition = 5.0;

	bool dof_blur_near_enabled = false;
	float dof_blur_near_distance = 2.0;
	float dof_blur_near_transition = 1.0;

	float dof_blur_amount = 0.1;
	void _update_dof_blur();

	// Override exposure
	bool override_exposure_enabled = false;
	float override_exposure = 1.0;
	void _update_override_exposure();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const override;

public:
	virtual RID get_rid() const override;

	// DOF blur
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

	// Override exposure
	void set_override_exposure_enabled(bool p_enabled);
	bool is_override_exposure_enabled() const;
	void set_override_exposure(float p_exposure);
	float get_override_exposure() const;

	CameraEffects();
	~CameraEffects();
};

#endif // CAMERA_EFFECTS_H
