/**************************************************************************/
/*  jiggle_modifier_2d.h                                                  */
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

#include "core/templates/local_vector.h"
#include "scene/2d/skeleton_modifier_2d.h"

class JiggleModifier2D : public SkeletonModifier2D {
	GDCLASS(JiggleModifier2D, SkeletonModifier2D);

	struct JiggleState {
		int bone = -1;
		Vector2 position;
		Vector2 velocity;
		bool initialized = false;
	};

	int root_bone = -1;
	int tip_bone = -1;
	real_t stiffness = 8.0;
	real_t damping = 1.0;
	real_t mass = 1.0;
	bool use_gravity = true;
	Vector2 gravity = Vector2(0, 98);
	LocalVector<JiggleState> states;

	JiggleState *_get_state(int p_bone);
	void _reset_jiggle_state();

protected:
	static void _bind_methods();
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual void _skeleton_changed(Skeleton2D *p_old, Skeleton2D *p_new) override;
	virtual void _process_modification(double p_delta) override;

public:
	void set_root_bone(int p_bone);
	int get_root_bone() const;
	void set_tip_bone(int p_bone);
	int get_tip_bone() const;
	void set_stiffness(real_t p_stiffness);
	real_t get_stiffness() const;
	void set_damping(real_t p_damping);
	real_t get_damping() const;
	void set_mass(real_t p_mass);
	real_t get_mass() const;
	void set_use_gravity(bool p_use_gravity);
	bool is_using_gravity() const;
	void set_gravity(const Vector2 &p_gravity);
	Vector2 get_gravity() const;
	void reset();
};
