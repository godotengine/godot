/**************************************************************************/
/*  gpu_particles_collision_height_field3d.hpp                            */
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

#include <godot_cpp/classes/gpu_particles_collision3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class GPUParticlesCollisionHeightField3D : public GPUParticlesCollision3D {
	GDEXTENSION_CLASS(GPUParticlesCollisionHeightField3D, GPUParticlesCollision3D)

public:
	enum Resolution {
		RESOLUTION_256 = 0,
		RESOLUTION_512 = 1,
		RESOLUTION_1024 = 2,
		RESOLUTION_2048 = 3,
		RESOLUTION_4096 = 4,
		RESOLUTION_8192 = 5,
		RESOLUTION_MAX = 6,
	};

	enum UpdateMode {
		UPDATE_MODE_WHEN_MOVED = 0,
		UPDATE_MODE_ALWAYS = 1,
	};

	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;
	void set_resolution(GPUParticlesCollisionHeightField3D::Resolution p_resolution);
	GPUParticlesCollisionHeightField3D::Resolution get_resolution() const;
	void set_update_mode(GPUParticlesCollisionHeightField3D::UpdateMode p_update_mode);
	GPUParticlesCollisionHeightField3D::UpdateMode get_update_mode() const;
	void set_heightfield_mask(uint32_t p_heightfield_mask);
	uint32_t get_heightfield_mask() const;
	void set_heightfield_mask_value(int32_t p_layer_number, bool p_value);
	bool get_heightfield_mask_value(int32_t p_layer_number) const;
	void set_follow_camera_enabled(bool p_enabled);
	bool is_follow_camera_enabled() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GPUParticlesCollision3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(GPUParticlesCollisionHeightField3D::Resolution);
VARIANT_ENUM_CAST(GPUParticlesCollisionHeightField3D::UpdateMode);

