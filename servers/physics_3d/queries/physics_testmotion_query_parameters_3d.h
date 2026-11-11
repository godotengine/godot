/**************************************************************************/
/*  physics_testmotion_query_parameters_3d.h                              */
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

#include "core/object/ref_counted.h"
#include "servers/physics_3d/physics_server_3d_types.h"

class PhysicsTestMotionParameters3D : public RefCounted {
	GDCLASS(PhysicsTestMotionParameters3D, RefCounted);

	PS3DT::MotionParameters parameters;

protected:
	static void _bind_methods();

public:
	const PS3DT::MotionParameters &get_parameters() const { return parameters; }

	const Transform3D &get_from() const { return parameters.from; }
	void set_from(const Transform3D &p_from) { parameters.from = p_from; }

	const Vector3 &get_motion() const { return parameters.motion; }
	void set_motion(const Vector3 &p_motion) { parameters.motion = p_motion; }

	real_t get_margin() const { return parameters.margin; }
	void set_margin(real_t p_margin) { parameters.margin = p_margin; }

	int get_max_collisions() const { return parameters.max_collisions; }
	void set_max_collisions(int p_max_collisions) { parameters.max_collisions = p_max_collisions; }

	bool is_collide_separation_ray_enabled() const { return parameters.collide_separation_ray; }
	void set_collide_separation_ray_enabled(bool p_enabled) { parameters.collide_separation_ray = p_enabled; }

	TypedArray<RID> get_exclude_bodies() const;
	void set_exclude_bodies(const TypedArray<RID> &p_exclude);

	TypedArray<uint64_t> get_exclude_objects() const;
	void set_exclude_objects(const TypedArray<uint64_t> &p_exclude);

	bool is_recovery_as_collision_enabled() const { return parameters.recovery_as_collision; }
	void set_recovery_as_collision_enabled(bool p_enabled) { parameters.recovery_as_collision = p_enabled; }
};
