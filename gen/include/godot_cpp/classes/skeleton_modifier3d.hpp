/**************************************************************************/
/*  skeleton_modifier3d.hpp                                               */
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

#include <godot_cpp/classes/node3d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Skeleton3D;

class SkeletonModifier3D : public Node3D {
	GDEXTENSION_CLASS(SkeletonModifier3D, Node3D)

public:
	enum BoneAxis {
		BONE_AXIS_PLUS_X = 0,
		BONE_AXIS_MINUS_X = 1,
		BONE_AXIS_PLUS_Y = 2,
		BONE_AXIS_MINUS_Y = 3,
		BONE_AXIS_PLUS_Z = 4,
		BONE_AXIS_MINUS_Z = 5,
	};

	enum BoneDirection {
		BONE_DIRECTION_PLUS_X = 0,
		BONE_DIRECTION_MINUS_X = 1,
		BONE_DIRECTION_PLUS_Y = 2,
		BONE_DIRECTION_MINUS_Y = 3,
		BONE_DIRECTION_PLUS_Z = 4,
		BONE_DIRECTION_MINUS_Z = 5,
		BONE_DIRECTION_FROM_PARENT = 6,
	};

	enum SecondaryDirection {
		SECONDARY_DIRECTION_NONE = 0,
		SECONDARY_DIRECTION_PLUS_X = 1,
		SECONDARY_DIRECTION_MINUS_X = 2,
		SECONDARY_DIRECTION_PLUS_Y = 3,
		SECONDARY_DIRECTION_MINUS_Y = 4,
		SECONDARY_DIRECTION_PLUS_Z = 5,
		SECONDARY_DIRECTION_MINUS_Z = 6,
		SECONDARY_DIRECTION_CUSTOM = 7,
	};

	enum RotationAxis {
		ROTATION_AXIS_X = 0,
		ROTATION_AXIS_Y = 1,
		ROTATION_AXIS_Z = 2,
		ROTATION_AXIS_ALL = 3,
		ROTATION_AXIS_CUSTOM = 4,
	};

	Skeleton3D *get_skeleton() const;
	void set_active(bool p_active);
	bool is_active() const;
	void set_influence(float p_influence);
	float get_influence() const;
	virtual void _process_modification_with_delta(double p_delta);
	virtual void _process_modification();
	virtual void _skeleton_changed(Skeleton3D *p_old_skeleton, Skeleton3D *p_new_skeleton);
	virtual void _validate_bone_names();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_process_modification_with_delta), decltype(&T::_process_modification_with_delta)>) {
			BIND_VIRTUAL_METHOD(T, _process_modification_with_delta, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_process_modification), decltype(&T::_process_modification)>) {
			BIND_VIRTUAL_METHOD(T, _process_modification, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_skeleton_changed), decltype(&T::_skeleton_changed)>) {
			BIND_VIRTUAL_METHOD(T, _skeleton_changed, 2926744397);
		}
		if constexpr (!std::is_same_v<decltype(&B::_validate_bone_names), decltype(&T::_validate_bone_names)>) {
			BIND_VIRTUAL_METHOD(T, _validate_bone_names, 3218959716);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SkeletonModifier3D::BoneAxis);
VARIANT_ENUM_CAST(SkeletonModifier3D::BoneDirection);
VARIANT_ENUM_CAST(SkeletonModifier3D::SecondaryDirection);
VARIANT_ENUM_CAST(SkeletonModifier3D::RotationAxis);

