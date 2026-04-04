/**************************************************************************/
/*  open_xr_hand.hpp                                                      */
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
#include <godot_cpp/variant/node_path.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRHand : public Node3D {
	GDEXTENSION_CLASS(OpenXRHand, Node3D)

public:
	enum Hands {
		HAND_LEFT = 0,
		HAND_RIGHT = 1,
		HAND_MAX = 2,
	};

	enum MotionRange {
		MOTION_RANGE_UNOBSTRUCTED = 0,
		MOTION_RANGE_CONFORM_TO_CONTROLLER = 1,
		MOTION_RANGE_MAX = 2,
	};

	enum SkeletonRig {
		SKELETON_RIG_OPENXR = 0,
		SKELETON_RIG_HUMANOID = 1,
		SKELETON_RIG_MAX = 2,
	};

	enum BoneUpdate {
		BONE_UPDATE_FULL = 0,
		BONE_UPDATE_ROTATION_ONLY = 1,
		BONE_UPDATE_MAX = 2,
	};

	void set_hand(OpenXRHand::Hands p_hand);
	OpenXRHand::Hands get_hand() const;
	void set_hand_skeleton(const NodePath &p_hand_skeleton);
	NodePath get_hand_skeleton() const;
	void set_motion_range(OpenXRHand::MotionRange p_motion_range);
	OpenXRHand::MotionRange get_motion_range() const;
	void set_skeleton_rig(OpenXRHand::SkeletonRig p_skeleton_rig);
	OpenXRHand::SkeletonRig get_skeleton_rig() const;
	void set_bone_update(OpenXRHand::BoneUpdate p_bone_update);
	OpenXRHand::BoneUpdate get_bone_update() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRHand::Hands);
VARIANT_ENUM_CAST(OpenXRHand::MotionRange);
VARIANT_ENUM_CAST(OpenXRHand::SkeletonRig);
VARIANT_ENUM_CAST(OpenXRHand::BoneUpdate);

