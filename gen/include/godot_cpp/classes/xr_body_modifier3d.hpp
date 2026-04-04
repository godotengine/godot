/**************************************************************************/
/*  xr_body_modifier3d.hpp                                                */
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

#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class XRBodyModifier3D : public SkeletonModifier3D {
	GDEXTENSION_CLASS(XRBodyModifier3D, SkeletonModifier3D)

public:
	enum BodyUpdate : uint64_t {
		BODY_UPDATE_UPPER_BODY = 1,
		BODY_UPDATE_LOWER_BODY = 2,
		BODY_UPDATE_HANDS = 4,
	};

	enum BoneUpdate {
		BONE_UPDATE_FULL = 0,
		BONE_UPDATE_ROTATION_ONLY = 1,
		BONE_UPDATE_MAX = 2,
	};

	void set_body_tracker(const StringName &p_tracker_name);
	StringName get_body_tracker() const;
	void set_body_update(BitField<XRBodyModifier3D::BodyUpdate> p_body_update);
	BitField<XRBodyModifier3D::BodyUpdate> get_body_update() const;
	void set_bone_update(XRBodyModifier3D::BoneUpdate p_bone_update);
	XRBodyModifier3D::BoneUpdate get_bone_update() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_BITFIELD_CAST(XRBodyModifier3D::BodyUpdate);
VARIANT_ENUM_CAST(XRBodyModifier3D::BoneUpdate);

