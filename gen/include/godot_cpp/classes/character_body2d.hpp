/**************************************************************************/
/*  character_body2d.hpp                                                  */
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

#include <godot_cpp/classes/physics_body2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class KinematicCollision2D;

class CharacterBody2D : public PhysicsBody2D {
	GDEXTENSION_CLASS(CharacterBody2D, PhysicsBody2D)

public:
	enum MotionMode {
		MOTION_MODE_GROUNDED = 0,
		MOTION_MODE_FLOATING = 1,
	};

	enum PlatformOnLeave {
		PLATFORM_ON_LEAVE_ADD_VELOCITY = 0,
		PLATFORM_ON_LEAVE_ADD_UPWARD_VELOCITY = 1,
		PLATFORM_ON_LEAVE_DO_NOTHING = 2,
	};

	bool move_and_slide();
	void apply_floor_snap();
	void set_velocity(const Vector2 &p_velocity);
	Vector2 get_velocity() const;
	void set_safe_margin(float p_margin);
	float get_safe_margin() const;
	bool is_floor_stop_on_slope_enabled() const;
	void set_floor_stop_on_slope_enabled(bool p_enabled);
	void set_floor_constant_speed_enabled(bool p_enabled);
	bool is_floor_constant_speed_enabled() const;
	void set_floor_block_on_wall_enabled(bool p_enabled);
	bool is_floor_block_on_wall_enabled() const;
	void set_slide_on_ceiling_enabled(bool p_enabled);
	bool is_slide_on_ceiling_enabled() const;
	void set_platform_floor_layers(uint32_t p_exclude_layer);
	uint32_t get_platform_floor_layers() const;
	void set_platform_wall_layers(uint32_t p_exclude_layer);
	uint32_t get_platform_wall_layers() const;
	int32_t get_max_slides() const;
	void set_max_slides(int32_t p_max_slides);
	float get_floor_max_angle() const;
	void set_floor_max_angle(float p_radians);
	float get_floor_snap_length();
	void set_floor_snap_length(float p_floor_snap_length);
	float get_wall_min_slide_angle() const;
	void set_wall_min_slide_angle(float p_radians);
	Vector2 get_up_direction() const;
	void set_up_direction(const Vector2 &p_up_direction);
	void set_motion_mode(CharacterBody2D::MotionMode p_mode);
	CharacterBody2D::MotionMode get_motion_mode() const;
	void set_platform_on_leave(CharacterBody2D::PlatformOnLeave p_on_leave_apply_velocity);
	CharacterBody2D::PlatformOnLeave get_platform_on_leave() const;
	bool is_on_floor() const;
	bool is_on_floor_only() const;
	bool is_on_ceiling() const;
	bool is_on_ceiling_only() const;
	bool is_on_wall() const;
	bool is_on_wall_only() const;
	Vector2 get_floor_normal() const;
	Vector2 get_wall_normal() const;
	Vector2 get_last_motion() const;
	Vector2 get_position_delta() const;
	Vector2 get_real_velocity() const;
	float get_floor_angle(const Vector2 &p_up_direction = Vector2(0, -1)) const;
	Vector2 get_platform_velocity() const;
	int32_t get_slide_collision_count() const;
	Ref<KinematicCollision2D> get_slide_collision(int32_t p_slide_idx);
	Ref<KinematicCollision2D> get_last_slide_collision();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PhysicsBody2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CharacterBody2D::MotionMode);
VARIANT_ENUM_CAST(CharacterBody2D::PlatformOnLeave);

