/**************************************************************************/
/*  look_at_modifier_2d.h                                                 */
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

#include "scene/2d/skeleton_modifier_2d.h"

class LookAtModifier2D : public SkeletonModifier2D {
	GDCLASS(LookAtModifier2D, SkeletonModifier2D);

	int bone = -1;
	NodePath target_node;
	real_t additional_rotation = 0.0;
	bool enable_constraint = false;
	bool constraint_in_localspace = true;
	bool constraint_angle_invert = false;
	real_t constraint_angle_min = -Math::PI;
	real_t constraint_angle_max = Math::PI;

	real_t _clamp_angle(real_t p_angle) const;

protected:
	static void _bind_methods();
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual void _process_modification(double p_delta) override;

public:
	void set_bone(int p_bone);
	int get_bone() const;
	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_additional_rotation(real_t p_rotation);
	real_t get_additional_rotation() const;
	void set_enable_constraint(bool p_enabled);
	bool is_constraint_enabled() const;
	void set_constraint_in_localspace(bool p_enabled);
	bool is_constraint_in_localspace() const;
	void set_constraint_angle_min(real_t p_angle);
	real_t get_constraint_angle_min() const;
	void set_constraint_angle_max(real_t p_angle);
	real_t get_constraint_angle_max() const;
	void set_constraint_angle_invert(bool p_invert);
	bool is_constraint_angle_inverted() const;
};
