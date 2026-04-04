/**************************************************************************/
/*  physical_bone2d.hpp                                                   */
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

#include <godot_cpp/classes/rigid_body2d.hpp>
#include <godot_cpp/variant/node_path.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Joint2D;

class PhysicalBone2D : public RigidBody2D {
	GDEXTENSION_CLASS(PhysicalBone2D, RigidBody2D)

public:
	Joint2D *get_joint() const;
	bool get_auto_configure_joint() const;
	void set_auto_configure_joint(bool p_auto_configure_joint);
	void set_simulate_physics(bool p_simulate_physics);
	bool get_simulate_physics() const;
	bool is_simulating_physics() const;
	void set_bone2d_nodepath(const NodePath &p_nodepath);
	NodePath get_bone2d_nodepath() const;
	void set_bone2d_index(int32_t p_bone_index);
	int32_t get_bone2d_index() const;
	void set_follow_bone_when_simulating(bool p_follow_bone);
	bool get_follow_bone_when_simulating() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RigidBody2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

