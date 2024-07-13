/**************************************************************************/
/*  renik_placement.cpp                                                   */
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

#include "renik_placement.h"
#include "core/string/print_string.h"
#include "renik_helper.h"

#ifndef _3D_DISABLED

void RenIKPlacement::save_previous_transforms() {
	prev_hip = target_hip;
	prev_left_foot = target_left_foot;
	prev_right_foot = target_right_foot;
}

void RenIKPlacement::interpolate_transforms(float p_fraction, bool p_update_hips,
		bool p_update_feet) {
	if (p_update_hips) {
		interpolated_hip = prev_hip.interpolate_with(target_hip, p_fraction);
	}
	if (p_update_feet) {
		interpolated_left_foot =
				prev_left_foot.interpolate_with(target_left_foot, p_fraction);
		interpolated_right_foot =
				prev_right_foot.interpolate_with(target_right_foot, p_fraction);
	}
}

void RenIKPlacement::hip_place(float p_delta, Transform3D p_head,
		Transform3D p_left_foot, Transform3D p_right_foot,
		float p_twist, bool p_instant) {
	Vector3 left_middle =
			(p_left_foot.translated_local(Vector3(0, 0, left_foot_length / 2))).origin;
	Vector3 right_middle =
			(p_right_foot.translated_local(Vector3(0, 0, right_foot_length / 2))).origin;
	float left_distance = left_middle.distance_squared_to(p_head.origin);
	float right_distance = right_middle.distance_squared_to(p_head.origin);
	Vector3 foot_median = left_middle.lerp(right_middle, 0.5);
	Vector3 foot = left_distance > right_distance ? left_middle : right_middle;
	Vector3 foot_direction =
			(foot - p_head.origin).project(foot_median - p_head.origin);

	target_hip.basis =
			RenIKHelper::align_vectors(Vector3(0, -1, 0), foot_direction);
	Vector3 head_forward = p_head.basis.inverse()[2];
	Vector3 feet_forward = (p_left_foot.interpolate_with(p_right_foot, 0.5)).basis[2];
	Vector3 hip_forward = feet_forward.lerp(head_forward, 0.5);

	Vector3 hip_y = -foot_direction.normalized();
	Vector3 hip_z = RenIKHelper::vector_rejection(hip_forward.normalized(), hip_y)
							.normalized();
	Vector3 hip_x = hip_y.cross(hip_z).normalized();
	target_hip.basis = Basis(hip_x, hip_y, hip_z).orthonormalized();

	float crouch_distance = p_head.origin.distance_to(foot) * crouch_ratio;
	float extra_hip_distance = hip_offset.length() - crouch_distance;
	Vector3 follow_hip_direction =
			target_hip.basis.xform_inv(p_head.basis.xform(hip_offset));

	Vector3 effective_hip_direction =
			hip_offset.lerp(follow_hip_direction, hip_follow_head_influence)
					.normalized();

	target_hip.origin = p_head.origin;
	target_hip.translate_local(crouch_distance * effective_hip_direction.normalized());
	if (extra_hip_distance > 0) {
		target_hip.translate_local(
				Vector3(0, 0, -extra_hip_distance * (1 / hunch_ratio)));
	}

	if (p_instant) {
		prev_hip = target_hip;
	}
}

/*foot_place requires raycasting unless a raycast result is provided.
		Raycasting needs to happen inside of a physics update
*/
void RenIKPlacement::foot_place(float p_delta, Transform3D p_head, Ref<World3D> p_world_3d,
		bool p_instant) {
	ERR_FAIL_COND(p_world_3d.is_null());

	PhysicsDirectSpaceState3D *dss =
			PhysicsServer3D::get_singleton()->space_get_direct_state(
					p_world_3d->get_space());
	ERR_FAIL_COND(!dss);

	PhysicsDirectSpaceState3D::RayResult left_raycast;
	PhysicsDirectSpaceState3D::RayResult right_raycast;
	PhysicsDirectSpaceState3D::RayResult laying_raycast;

	float startOffset = ((spine_length) * -center_of_balance_position) / sqrt(2);
	Vector3 leftStart =
			p_head.translated_local(Vector3(0, startOffset, startOffset) + left_hip_offset)
					.origin;
	Vector3 rightStart =
			p_head.translated_local(Vector3(0, startOffset, startOffset) + right_hip_offset)
					.origin;
	Vector3 leftStop = p_head.origin +
			Vector3(0,
					(-spine_length - left_leg_length - floor_offset) *
									(1 + raycast_allowance) +
							left_hip_offset[1],
					0) +
			p_head.basis.xform(left_hip_offset);
	Vector3 rightStop =
			p_head.origin +
			Vector3(0,
					(-spine_length - right_leg_length - floor_offset) *
									(1 + raycast_allowance) +
							right_hip_offset[1],
					0) +
			p_head.basis.xform(right_hip_offset);

	PhysicsDirectSpaceState3D::RayParameters ray_query_parameters;
	ray_query_parameters.from = leftStart;
	ray_query_parameters.to = leftStop;
	ray_query_parameters.collision_mask = collision_mask;
	ray_query_parameters.collide_with_areas = collide_with_areas;
	ray_query_parameters.collide_with_bodies = collide_with_bodies;
	bool left_collided = dss->intersect_ray(ray_query_parameters, left_raycast);
	ray_query_parameters.from = rightStart;
	ray_query_parameters.to = rightStop;
	bool right_collided = dss->intersect_ray(ray_query_parameters, right_raycast);
	ray_query_parameters.from = p_head.origin;
	ray_query_parameters.to = p_head.origin - Vector3(0, spine_length + floor_offset, 0);
	bool laying_down = dss->intersect_ray(ray_query_parameters, laying_raycast);
	if (!left_collided) {
		left_raycast.collider = nullptr;
	}
	if (!right_collided) {
		right_raycast.collider = nullptr;
	}
	if (!laying_down) {
		laying_raycast.collider = nullptr;
	}
	Vector3 left_offset =
			(leftStart - leftStop).normalized() * floor_offset * left_leg_length;
	Vector3 right_offset =
			(rightStart - rightStop).normalized() * floor_offset * right_leg_length;
	Vector3 laying_offset =
			Vector3(0, floor_offset * (left_leg_length + right_leg_length) / 2, 0);
	left_raycast.position += left_offset;
	right_raycast.position += right_offset;
	laying_raycast.position += laying_offset;
	foot_place(p_delta, p_head, left_raycast, right_raycast, laying_raycast, p_instant);
}

Transform3D RenIKPlacement::dangle_foot(Transform3D p_head, float p_distance,
		float p_leg_length, Vector3 p_hip_offset) {
	Transform3D foot;
	Basis upright_head =
			RenIKHelper::align_vectors(Vector3(0, 1, 0), p_head.basis[1])
					.slerp(Quaternion(), 1 - dangle_follow_head);
	Vector3 dangle_vector = Vector3(0, spine_length + p_leg_length, 0) - p_hip_offset;
	Basis dangle_basis = p_head.basis * upright_head;
	foot.basis = dangle_basis * Basis(Vector3(1, 0, 0), dangle_angle);
	foot.origin = p_head.origin + dangle_basis.xform(-dangle_vector);
	return foot;
}

void RenIKPlacement::initialize_loop(Vector3 p_velocity, Vector3 p_left_ground,
		Vector3 p_right_ground, bool p_left_grounded,
		bool p_right_grounded) {
	if (p_left_grounded && p_right_grounded) {
		Vector3 foot_diff = target_left_foot.origin - target_right_foot.origin;
		float dot = foot_diff.dot(p_velocity);
		if (dot == 0) {
			float left_dist =
					target_left_foot.origin.distance_squared_to(p_left_ground);
			float right_dist =
					target_right_foot.origin.distance_squared_to(p_right_ground);
			if (left_dist < right_dist) {
				// left foot more off balance
				step_progress = 0;
			} else {
				// right foot more off balance
				step_progress = 0.5;
			}
		} else if (dot > 0) {
			// left foot in front
			step_progress = 0;
		} else {
			// right foot in front
			step_progress = 0.5;
		}
	} else if (p_left_grounded) {
		step_progress = 0;
	} else {
		step_progress = 0.5;
	}
}

int RenIKPlacement::get_loop_state(float p_loop_state_scaling,
		float p_loop_progress,
		float &r_loop_state_progress, Gait p_gait) {
	int state = -1;
	float ground_time = p_gait.ground_time - p_gait.ground_time * p_loop_state_scaling;
	float lift_time =
			p_gait.lift_time_base + p_gait.lift_time_scalar * p_loop_state_scaling;
	float apex_in_time =
			p_gait.apex_in_time_base + p_gait.apex_in_time_scalar * p_loop_state_scaling;
	float apex_out_time =
			p_gait.apex_out_time_base + p_gait.apex_out_time_scalar * p_loop_state_scaling;
	float drop_time =
			p_gait.drop_time_base + p_gait.drop_time_scalar * p_loop_state_scaling;
	float total_time = ground_time + lift_time + apex_in_time + apex_out_time +
			drop_time + ground_time;

	float progress_time = p_loop_progress * total_time;

	if (progress_time < ground_time) {
		state = LOOP_GROUND_IN;
		r_loop_state_progress = (progress_time) / ground_time;
	} else if (progress_time < ground_time + lift_time) {
		state = LOOP_LIFT;
		r_loop_state_progress = (progress_time - ground_time) / lift_time;
	} else if (progress_time < ground_time + lift_time + apex_in_time) {
		state = LOOP_APEX_IN;
		r_loop_state_progress =
				(progress_time - ground_time - lift_time) / apex_in_time;
	} else if (progress_time <
			ground_time + lift_time + apex_in_time + apex_out_time) {
		state = LOOP_APEX_OUT;
		r_loop_state_progress =
				(progress_time - ground_time - lift_time - apex_in_time) /
				apex_out_time;
	} else if (progress_time < ground_time + lift_time + apex_in_time +
					apex_out_time + drop_time) {
		state = LOOP_DROP;
		r_loop_state_progress = (progress_time - ground_time - lift_time -
										apex_in_time - apex_out_time) /
				drop_time;
	} else {
		state = LOOP_GROUND_OUT;
		r_loop_state_progress = (progress_time - ground_time - lift_time -
										apex_in_time - apex_out_time - drop_time) /
				ground_time;
	}

	return state;
}

void RenIKPlacement::loop_foot(Transform3D &r_step, Transform3D &r_stand,
		Transform3D &r_stand_local, Node3D *p_ground,
		Node3D **p_prev_ground, int &r_loop_state,
		Vector3 &r_grounded_stop, Transform3D p_head,
		float p_leg_length, float p_foot_length,
		Vector3 p_velocity, float p_loop_scaling,
		float p_step_progress, Vector3 p_ground_position,
		Vector3 p_ground_normal, Gait p_gait) {
	Quaternion upright_foot = RenIKHelper::align_vectors(
			Vector3(0, 1, 0), p_head.basis.xform_inv(p_ground_normal));
	if (p_ground_normal.dot(p_head.basis[1]) < cos(rotation_threshold) &&
			p_ground_normal.dot(Vector3(0, 1, 0)) < cos(rotation_threshold)) {
		upright_foot = Quaternion();
	}
	Vector3 ground_velocity =
			RenIKHelper::vector_rejection(p_velocity, p_ground_normal);
	if (ground_velocity.length() > max_threshold * step_pace) {
		ground_velocity = ground_velocity.normalized() * max_threshold * step_pace;
	}
	float loop_state_progress = 0;
	r_loop_state =
			get_loop_state(p_loop_scaling, p_step_progress, loop_state_progress, p_gait);
	float head_distance = p_head.origin.distance_to(p_ground_position);
	float ease_scaling = p_loop_scaling * p_loop_scaling * p_loop_scaling *
			p_loop_scaling; // ease the growth a little
	float vertical_scaling = head_distance * ease_scaling;
	float horizontal_scaling = p_leg_length * ease_scaling;
	Transform3D grounded_foot =
			Transform3D(p_head.basis * upright_foot, p_ground_position);
	Transform3D lifted_foot = Transform3D(
			grounded_foot.basis.rotated_local(Vector3(1, 0, 0),
					ease_scaling * p_gait.lift_angle),
			p_ground_position +
					p_ground_normal * vertical_scaling * p_gait.lift_vertical_scalar +
					p_ground_normal * head_distance * p_gait.lift_vertical -
					ground_velocity.normalized() * horizontal_scaling *
							p_gait.lift_horizontal_scalar);
	Transform3D apex_foot = Transform3D(
			grounded_foot.basis.rotated_local(Vector3(1, 0, 0),
					ease_scaling * p_gait.apex_angle),
			p_ground_position +
					p_ground_normal * vertical_scaling * p_gait.apex_vertical_scalar +
					p_ground_normal * head_distance * p_gait.apex_vertical);
	Transform3D drop_foot = Transform3D(
			grounded_foot.basis.rotated_local(Vector3(1, 0, 0),
					ease_scaling * p_gait.drop_angle),
			p_ground_position +
					p_ground_normal * vertical_scaling * p_gait.drop_vertical_scalar +
					p_ground_normal * head_distance * p_gait.drop_vertical_scalar +
					ground_velocity.normalized() * horizontal_scaling *
							p_gait.drop_horizontal_scalar);

	switch (r_loop_state) {
		case LOOP_GROUND_IN:
		case LOOP_GROUND_OUT: {
			// stick to where it landed
			if (p_ground != nullptr && p_ground == *p_prev_ground) {
				stand_foot(grounded_foot, r_stand, r_stand_local, p_ground);
			} else if (p_ground != nullptr) {
				*p_prev_ground = p_ground;
				r_stand = grounded_foot;
				Transform3D ground_global = p_ground->get_global_transform();
				ground_global.basis.orthonormalize();
				r_stand_local = ground_global.affine_inverse() * r_stand;
			} else {
				r_stand = grounded_foot;
			}
			r_step = r_stand;

			float step_distance = r_step.origin.distance_to(p_ground_position) / p_leg_length;
			Transform3D lean_offset;
			float tip_toe_angle = step_distance * p_gait.tip_toe_distance_scalar +
					horizontal_scaling * p_gait.tip_toe_speed_scalar;
			tip_toe_angle = tip_toe_angle > p_gait.tip_toe_angle_max
					? p_gait.tip_toe_angle_max
					: tip_toe_angle;
			lean_offset.origin = Vector3(0, p_foot_length * sin(tip_toe_angle), 0);
			lean_offset.rotate_basis(Vector3(1, 0, 0), tip_toe_angle);
			r_step *= lean_offset;

			r_grounded_stop = r_step.origin;

			break;
		}
		case LOOP_LIFT: {
			float step_distance = r_step.origin.distance_to(p_ground_position) / p_leg_length;
			Transform3D lean_offset;
			float tip_toe_angle = step_distance * p_gait.tip_toe_distance_scalar +
					horizontal_scaling * p_gait.tip_toe_speed_scalar;
			tip_toe_angle = tip_toe_angle > p_gait.tip_toe_angle_max
					? p_gait.tip_toe_angle_max
					: tip_toe_angle;

			r_step.basis =
					grounded_foot.basis.rotated_local(Vector3(1, 0, 0), tip_toe_angle)
							.slerp(lifted_foot.basis, loop_state_progress);
			r_step.origin = r_grounded_stop.cubic_interpolate(
					lifted_foot.origin,
					r_grounded_stop - ground_velocity * horizontal_scaling,
					lifted_foot.origin + p_ground_normal * vertical_scaling,
					loop_state_progress);
			break;
		}
		case LOOP_APEX_IN:
			r_step.basis = lifted_foot.basis.slerp(apex_foot.basis, loop_state_progress);
			r_step.origin = lifted_foot.origin.cubic_interpolate(
					apex_foot.origin, lifted_foot.origin - p_ground_normal * vertical_scaling,
					apex_foot.origin + ground_velocity * p_leg_length, loop_state_progress);
			break;
		case LOOP_APEX_OUT:
			r_step.basis = apex_foot.basis.slerp(drop_foot.basis, loop_state_progress);
			r_step.origin = apex_foot.origin.cubic_interpolate(
					drop_foot.origin,
					apex_foot.origin - ground_velocity * horizontal_scaling,
					drop_foot.origin - p_ground_normal * vertical_scaling,
					loop_state_progress);
			break;
		case LOOP_DROP:
			r_step.basis =
					drop_foot.basis.slerp(grounded_foot.basis, loop_state_progress);
			r_step.origin = drop_foot.origin.cubic_interpolate(
					grounded_foot.origin,
					drop_foot.origin + p_ground_normal * vertical_scaling,
					grounded_foot.origin - ground_velocity * horizontal_scaling,
					loop_state_progress);
			break;
	}

	if (r_loop_state != LOOP_GROUND_IN && r_loop_state != LOOP_GROUND_OUT) {
		// update standing positions to ensure a smooth transition to standing
		r_stand.origin = p_ground_position;
		r_stand.basis = grounded_foot.basis;
		if (p_ground != nullptr) {
			Transform3D ground_global = p_ground->get_global_transform();
			ground_global.basis.orthonormalize();
			r_stand_local = ground_global.affine_inverse() * r_stand;
		}
		if (walk_state != LOOP_LIFT) {
			r_grounded_stop = r_step.origin;
		} else {
			float contact_easing = p_gait.contact_point_ease +
					p_gait.contact_point_ease_scalar * p_loop_scaling;
			contact_easing = contact_easing > 1 ? 1 : contact_easing;
			r_grounded_stop = r_grounded_stop.lerp(p_ground_position, contact_easing);
		}
	}
}

void RenIKPlacement::loop(Transform3D p_head, Vector3 p_velocity,
		Vector3 p_left_ground_position, Vector3 p_left_normal,
		Vector3 p_right_ground_pos, Vector3 p_right_normal,
		bool p_left_grounded, bool p_right_grounded, Gait p_gait) {
	float stride_speed = step_pace * p_velocity.length() /
			((left_leg_length + right_leg_length) / 2);
	stride_speed = log(1 + stride_speed);
	stride_speed = stride_speed > max_threshold ? max_threshold : stride_speed;
	stride_speed = stride_speed < min_threshold ? min_threshold : stride_speed;
	float new_loop_scaling =
			max_threshold > min_threshold
			? (stride_speed - min_threshold) / (max_threshold - min_threshold)
			: 0;
	loop_scaling = (loop_scaling * p_gait.scaling_ease +
			new_loop_scaling * (1 - p_gait.scaling_ease));
	step_progress =
			Math::fmod((step_progress +
							   stride_speed * (p_gait.speed_scalar_min * (1 - loop_scaling) + p_gait.speed_scalar_max * loop_scaling)),
					1.0f);

	if (p_left_grounded) {
		loop_foot(left_step, left_stand, left_stand_local, left_ground,
				&prev_left_ground, left_loop_state, left_grounded_stop, p_head,
				left_leg_length, left_foot_length, p_velocity, loop_scaling,
				step_progress, p_left_ground_position, p_left_normal, p_gait);
	} else {
		Transform3D left_dangle =
				dangle_foot(p_head, (spine_length + left_leg_length) * dangle_ratio,
						left_leg_length, left_hip_offset);
		left_step.basis = left_step.basis.slerp(left_dangle.basis,
				1.0f - (1.0f / dangle_stiffness));
		left_step.origin = RenIKHelper::log_clamp(
				left_step.origin, left_dangle.origin, 1.0 / dangle_stiffness);
	}

	if (p_right_grounded) {
		loop_foot(right_step, right_stand, right_stand_local, right_ground,
				&prev_right_ground, right_loop_state, right_grounded_stop, p_head,
				right_leg_length, right_foot_length, p_velocity, loop_scaling,
				Math::fmod((step_progress + 0.5f), 1.0f), p_right_ground_pos,
				p_right_normal, p_gait);
	} else {
		Transform3D right_dangle =
				dangle_foot(p_head, (spine_length + right_leg_length) * dangle_ratio,
						right_leg_length, right_hip_offset);
		right_step.basis = right_step.basis.slerp(right_dangle.basis,
				1.0f - (1.0f / dangle_stiffness));
		right_step.origin = RenIKHelper::log_clamp(
				right_step.origin, right_dangle.origin, 1.0 / dangle_stiffness);
	}
}

void RenIKPlacement::step_direction(Vector3 p_forward, Vector3 p_side,
		Vector3 p_velocity, Vector3 p_left_ground,
		Vector3 p_right_ground, bool p_left_grounded,
		bool p_right_grounded) {
	Vector3 normalized_velocity = p_velocity.normalized();
	Vector3 normalized_forward = p_forward.normalized();
	Vector3 normalized_side = p_side.normalized();
	if (Math::abs(normalized_velocity.dot(normalized_side)) >
			strafe_angle_limit) {
		if (walk_state != STRAFING && walk_state != STRAFING_TRANSITION) {
			walk_state = STRAFING_TRANSITION;
			walk_transition_progress =
					stepping_transition_duration; // In units of loop progression
			initialize_loop(normalized_velocity, p_left_ground, p_right_ground,
					p_left_grounded, p_right_grounded);
		}
	} else if (normalized_velocity.dot(normalized_forward) < 0) {
		if (walk_state != BACKSTEPPING && walk_state != BACKSTEPPING_TRANSITION) {
			walk_state = BACKSTEPPING_TRANSITION;
			walk_transition_progress =
					stepping_transition_duration; // In units of loop progression
			initialize_loop(normalized_velocity, p_left_ground, p_right_ground,
					p_left_grounded, p_right_grounded);
		}
	} else {
		if (walk_state != STEPPING && walk_state != STEPPING_TRANSITION) {
			walk_state = STEPPING_TRANSITION;
			walk_transition_progress =
					stepping_transition_duration; // In units of loop progression
			initialize_loop(normalized_velocity, p_left_ground, p_right_ground,
					p_left_grounded, p_right_grounded);
		}
	}
}

void RenIKPlacement::stand_foot(Transform3D p_foot, Transform3D &r_stand,
		Transform3D &r_stand_local, Node3D *p_ground) {
	Transform3D ground_global = p_ground->get_global_transform();
	ground_global.basis.orthonormalize();
	r_stand = ground_global * r_stand_local;
	r_stand.basis.orthonormalize();
}

/*
Step 1: Figure out what state we're in.
If we're far from the ground, we're FALLING
If we're too close to the ground, we're LAYING
If we're moving too fast forward or off-balance, we're STEPPING
If we're moving too fast backward, we're BACKSTEPPING
Else we're just STANDING

There are transition states between all these base states

Step 2: Based on the state we place the feet.
FALLING: Dangle the feet down.
LAYING: Align feet with the rejection of our head's -z axis on the ground
normal. STEPPING: DO THE LOOP STANDING: If any foot is in the air we lerp it to
where the raycast hit the ground. If any foot was already on the ground, we
leave it there. Transitions to the STANDING state is only possible from the
stepping state, so we'll know if a foot is already on the ground based on where
it was in the stepping loop.

THE LOOP: Made up of 6 parts
1. The push - From when foot is on the ground directly below the center of
gravity until it lifts off the ground.
2. The kick - Foot kicks up to the furthest point backward of the loop.
3. Enter saddle - Foot swings down to point directly below center of gravity.
It's still above the ground.
4. Exit saddle - Foot continues swing up to the furthest point forward of the
loop.
5. The buildup - Foot gains speed as it comes in contact with the ground.
6. The landing - Foot touches down and sticks the ground until it's under the
center of gravity

Parts 1 and 6 are made by keeping the foot in place in world space.
Parts 2-5 are animated with bezier curves with continuous tangents between
parts. Parts 5 and 2 have vertical tangents, 2 and 3 have horizontal tangents, 3
and 4 have vertical tangents

At high speeds the durations of parts 1 and 6 will be 0 which makes the loop an
uninterrupted loop of bezier curves At low speeds the durations of 2 and 5 will
be almost 0 (though I don't plan to go all the way to 0)

Progress through loop will be represented with a float that goes from 0.0 to 1.0
where 0.0 is the beginning of part 1 and 1.0 is the end of part 6. The progress
from 0.0 to 1.0 happens smoothly and linearly with movement speed. What range of
numbers represents each part of the loop changes dynamically with movement
speed.
*/
void RenIKPlacement::foot_place(
		float p_delta, Transform3D p_head,
		PhysicsDirectSpaceState3D::RayResult p_left_raycast,
		PhysicsDirectSpaceState3D::RayResult p_right_raycast,
		PhysicsDirectSpaceState3D::RayResult p_laying_raycast, bool p_instant) {
	// Step 1: Find the proper state
	// Note we always enter transition states when possible

	left_ground = (Node3D *)p_left_raycast.collider;
	right_ground = (Node3D *)p_right_raycast.collider;
	Vector3 velocity = (p_head.origin - prevHead) / p_delta;
	Vector3 left_velocity;
	Vector3 right_velocity;
	if (p_left_raycast.collider != nullptr) {
		left_velocity =
				RenIKHelper::vector_rejection(velocity, p_left_raycast.normal);
	} else {
		left_velocity = RenIKHelper::vector_rejection(velocity, Vector3(0, 1, 0));
	}
	if (p_right_raycast.collider != nullptr) {
		right_velocity =
				RenIKHelper::vector_rejection(velocity, p_right_raycast.normal);
	} else {
		right_velocity = RenIKHelper::vector_rejection(velocity, Vector3(0, 1, 0));
	}

	float effective_min_threshold =
			min_threshold * ((left_leg_length + right_leg_length) / 2) / step_pace;
	if ((!p_left_raycast.collider && !p_right_raycast.collider &&
				!p_laying_raycast.collider) ||
			fall_override) {
		// If none of the raycasts hit anything then there isn't any ground to stand
		// on
		walk_state = FALLING;
		walk_transition_progress = 0;
	} else if (p_laying_raycast.collider || prone_override) {
		// If we're close enough for the laying raycast to trigger and we aren't
		// already laying down transition to laying down
		if (walk_state != LAYING && walk_state != LAYING_TRANSITION) {
			walk_state = LAYING_TRANSITION;
			walk_transition_progress =
					laying_transition_duration; // In units of loop progression
		}
	} else {
		Vector3 left_forward =
				RenIKHelper::vector_rejection(left_stand.basis[2], p_left_raycast.normal)
						.normalized();
		Vector3 right_forward = RenIKHelper::vector_rejection(right_stand.basis[2],
				p_right_raycast.normal)
										.normalized();
		Vector3 forward = (left_forward + right_forward).normalized();
		Vector3 upward = p_head.basis[1];
		Vector3 left_upward = left_stand.basis[1];
		Vector3 right_upward = right_stand.basis[1];
		Vector3 feet_sideways =
				(left_stand.basis[0] + right_stand.basis[0]).normalized();
		forward[0] = -forward[0]; // Flip the x for some reason
		feet_sideways[0] = -feet_sideways[0]; // Flip the x for some reason
		switch (walk_state) {
			case STANDING: { // brackets to stop declarations from breaking case labels
				// test that the feet aren't twisted in weird ways
				Vector3 left_head_forward =
						(p_head.basis *
								RenIKHelper::align_vectors(
										Vector3(0, 1, 0), p_head.basis.xform_inv(p_left_raycast.normal)))[2];
				Vector3 right_head_forward =
						(p_head.basis * RenIKHelper::align_vectors(Vector3(0, 1, 0), p_head.basis.xform_inv(p_right_raycast.normal)))[2];
				// left_head_forward = RenIKHelper::vector_rejection(left_head_forward,
				// ground_normal).normalized(); Vector3 left_forward =
				// RenIKHelper::vector_rejection(left_stand.basis[2],
				// left_raycast.normal).normalized(); Vector3 right_forward =
				// RenIKHelper::vector_rejection(right_stand.basis[2],
				// right_raycast.normal).normalized(); Vector3 forward =
				// left_forward.lerp(right_forward, 0.5).normalized(); Vector3 upward =
				// head.basis[1]; Vector3 left_upward = left_stand.basis[1]; Vector3
				// right_upward = right_stand.basis[1]; Vector3 feet_sideways =
				// left_stand.basis[0].lerp(right_stand.basis[0], 0.5).normalized();

				if (left_velocity.length() > effective_min_threshold ||
						right_velocity.length() > effective_min_threshold ||
						(p_left_raycast.collider != nullptr &&
								p_right_raycast.collider != nullptr &&
								!is_balanced(target_left_foot, target_right_foot)) ||
						(p_left_raycast.collider != nullptr &&
								left_stand.origin.distance_squared_to(p_left_raycast.position) >
										balance_threshold * (left_leg_length + right_leg_length) / 2) ||
						(p_right_raycast.collider != nullptr &&
								right_stand.origin.distance_squared_to(p_right_raycast.position) >
										balance_threshold * (left_leg_length + right_leg_length) / 2) ||
						(p_left_raycast.collider != nullptr &&
								(Node3D *)p_left_raycast.collider != left_ground) ||
						(p_right_raycast.collider != nullptr &&
								(Node3D *)p_right_raycast.collider != right_ground) ||
						left_head_forward.dot(left_forward) < cos(rotation_threshold) ||
						right_head_forward.dot(right_forward) < cos(rotation_threshold) ||
						(p_left_raycast.collider != nullptr &&
								p_left_raycast.normal.dot(left_upward) < cos(rotation_threshold) &&
								upward.dot(left_upward) < cos(rotation_threshold)) ||
						(p_right_raycast.collider != nullptr &&
								p_right_raycast.normal.dot(right_upward) < cos(rotation_threshold) &&
								upward.dot(right_upward) < cos(rotation_threshold))) {
					step_direction(forward, feet_sideways, velocity, p_left_raycast.position,
							p_right_raycast.position, p_left_raycast.collider != nullptr,
							p_right_raycast.collider != nullptr);
				}
				break;
			}
			case STANDING_TRANSITION:
				if (left_velocity.length() > effective_min_threshold ||
						right_velocity.length() > effective_min_threshold ||
						(p_left_raycast.collider != nullptr &&
								(Node3D *)p_left_raycast.collider != left_ground) ||
						(p_right_raycast.collider != nullptr &&
								(Node3D *)p_right_raycast.collider != right_ground)) {
					step_direction(forward, feet_sideways, velocity, p_left_raycast.position,
							p_right_raycast.position, p_left_raycast.collider != nullptr,
							p_right_raycast.collider != nullptr);
				}
				break;
			case STEPPING:
			case STEPPING_TRANSITION:
			case BACKSTEPPING:
			case BACKSTEPPING_TRANSITION:
			case STRAFING:
			case STRAFING_TRANSITION:
				if (left_velocity.length() < effective_min_threshold &&
						right_velocity.length() < effective_min_threshold &&
						walk_transition_progress == 0 &&
						(p_left_raycast.collider == nullptr ||
								left_stand.origin.distance_squared_to(p_left_raycast.position) <
										balance_threshold * (left_leg_length + right_leg_length) / 2) &&
						(p_right_raycast.collider == nullptr ||
								right_stand.origin.distance_squared_to(p_right_raycast.position) <
										balance_threshold * (left_leg_length + right_leg_length) / 2)) {
					walk_state = STANDING_TRANSITION;
					walk_transition_progress =
							standing_transition_duration; // In units of loop progression
				} else {
					step_direction(forward, feet_sideways, velocity, p_left_raycast.position,
							p_right_raycast.position, p_left_raycast.collider != nullptr,
							p_right_raycast.collider != nullptr);
				}
				break;
			default:
				step_direction(forward, feet_sideways, velocity, p_left_raycast.position,
						p_right_raycast.position, p_left_raycast.collider != nullptr,
						p_right_raycast.collider != nullptr);
				break;
		}
	}

	float stride_speed = step_pace * velocity.length() /
			((left_leg_length + right_leg_length) / 2);
	walk_transition_progress -=
			stride_speed < min_transition_speed ? min_transition_speed : stride_speed;
	walk_transition_progress =
			walk_transition_progress > 0 ? walk_transition_progress : 0;
	if (walk_transition_progress == 0 && walk_state < 0) {
		walk_state *= -1;
	}
	// Step 2: Place foot based on state
	switch (walk_state) {
		case FALLING: {
			Transform3D left_dangle =
					dangle_foot(p_head, (spine_length + left_leg_length) * dangle_ratio,
							left_leg_length, left_hip_offset);
			Transform3D right_dangle =
					dangle_foot(p_head, (spine_length + right_leg_length) * dangle_ratio,
							right_leg_length, right_hip_offset);

			target_left_foot.basis =
					target_left_foot.basis.slerp(left_dangle.basis * foot_basis_offset,
							1.0f - (1.0f / dangle_stiffness));
			target_left_foot.origin = RenIKHelper::log_clamp(
					target_left_foot.origin, left_dangle.origin, 1.0 / dangle_stiffness);

			target_right_foot.basis =
					target_right_foot.basis.slerp(right_dangle.basis * foot_basis_offset,
							1.0f - (1.0f / dangle_stiffness));
			target_right_foot.origin = RenIKHelper::log_clamp(
					target_right_foot.origin, right_dangle.origin, 1.0 / dangle_stiffness);

			// for easy transitions
			left_stand = target_left_foot;
			right_stand = target_right_foot;
			left_step = target_left_foot;
			right_step = target_right_foot;
			left_grounded_stop = target_left_foot.origin;
			right_grounded_stop = target_right_foot.origin;
			left_ground = nullptr;
			right_ground = nullptr;
			prev_left_ground = nullptr;
			prev_right_ground = nullptr;
			break;
		}
		case STANDING_TRANSITION:
		case STANDING: {
			float effective_transition_progress =
					walk_transition_progress / standing_transition_duration;
			effective_transition_progress = effective_transition_progress <= 1
					? effective_transition_progress
					: 1.0;
			if (left_ground != nullptr) {
				stand_foot(target_left_foot, left_stand, left_stand_local, left_ground);
				target_left_foot =
						Transform3D(left_stand.basis * foot_basis_offset, left_stand.origin)
								.interpolate_with(target_left_foot,
										effective_transition_progress);
				left_grounded_stop = left_stand.origin;
			} else {
				Transform3D left_dangle =
						dangle_foot(p_head, (spine_length + left_leg_length) * dangle_ratio,
								left_leg_length, left_hip_offset);
				target_left_foot.basis =
						target_left_foot.basis.slerp(left_dangle.basis * foot_basis_offset,
								1.0f - (1.0f / dangle_stiffness));
				target_left_foot.origin = RenIKHelper::log_clamp(
						target_left_foot.origin, left_dangle.origin, 1.0 / dangle_stiffness);
			}

			if (right_ground != nullptr) {
				stand_foot(target_right_foot, right_stand, right_stand_local,
						right_ground);
				target_right_foot =
						Transform3D(right_stand.basis * foot_basis_offset, right_stand.origin)
								.interpolate_with(target_right_foot,
										effective_transition_progress);
				right_grounded_stop = right_stand.origin;
			} else {
				Transform3D right_dangle =
						dangle_foot(p_head, (spine_length + right_leg_length) * dangle_ratio,
								right_leg_length, right_hip_offset);
				target_right_foot.basis =
						target_right_foot.basis.slerp(right_dangle.basis * foot_basis_offset,
								1.0f - (1.0f / dangle_stiffness));
				target_right_foot.origin =
						RenIKHelper::log_clamp(target_right_foot.origin, right_dangle.origin,
								1.0 / dangle_stiffness);
			}
			break;
		}
		case STEPPING_TRANSITION:
		case STEPPING: {
			float effective_transition_progress =
					walk_transition_progress / stepping_transition_duration;
			effective_transition_progress = effective_transition_progress <= 1
					? effective_transition_progress
					: 1.0;
			loop(p_head, velocity, p_left_raycast.position, p_left_raycast.normal,
					p_right_raycast.position, p_right_raycast.normal,
					p_left_raycast.collider != nullptr, p_right_raycast.collider != nullptr,
					forward_gait);
			target_left_foot =
					Transform3D(left_step.basis * foot_basis_offset, left_step.origin)
							.interpolate_with(target_left_foot, effective_transition_progress);
			target_right_foot =
					Transform3D(right_step.basis * foot_basis_offset, right_step.origin)
							.interpolate_with(target_right_foot, effective_transition_progress);
			break;
		}
		case BACKSTEPPING_TRANSITION:
		case BACKSTEPPING: {
			float effective_transition_progress =
					walk_transition_progress / stepping_transition_duration;
			effective_transition_progress = effective_transition_progress <= 1
					? effective_transition_progress
					: 1.0;
			loop(p_head, velocity, p_left_raycast.position, p_left_raycast.normal,
					p_right_raycast.position, p_right_raycast.normal,
					p_left_raycast.collider != nullptr, p_right_raycast.collider != nullptr,
					backward_gait);
			target_left_foot =
					Transform3D(left_step.basis * foot_basis_offset, left_step.origin)
							.interpolate_with(target_left_foot, effective_transition_progress);
			target_right_foot =
					Transform3D(right_step.basis * foot_basis_offset, right_step.origin)
							.interpolate_with(target_right_foot, effective_transition_progress);
			break;
		}
		case STRAFING_TRANSITION:
		case STRAFING: {
			float effective_transition_progress =
					walk_transition_progress / stepping_transition_duration;
			effective_transition_progress = effective_transition_progress <= 1
					? effective_transition_progress
					: 1.0;
			loop(p_head, velocity, p_left_raycast.position, p_left_raycast.normal,
					p_right_raycast.position, p_right_raycast.normal,
					p_left_raycast.collider != nullptr, p_right_raycast.collider != nullptr,
					sideways_gait);
			target_left_foot =
					Transform3D(left_step.basis * foot_basis_offset, left_step.origin)
							.interpolate_with(target_left_foot, effective_transition_progress);
			target_right_foot =
					Transform3D(right_step.basis * foot_basis_offset, right_step.origin)
							.interpolate_with(target_right_foot, effective_transition_progress);
			break;
		}
		case LAYING_TRANSITION:
			break;
		case LAYING:
			break;
		case OTHER_TRANSITION:
			break;
		case OTHER:
			break;
	}

	if (p_instant) {
		prev_left_foot = target_left_foot;
		prev_right_foot = target_right_foot;
	}

	prevHead = p_head.origin;
}

bool RenIKPlacement::is_balanced(Transform3D p_left, Transform3D p_right) {
	Vector3 relative_right = p_left.xform_inv(p_right.origin);
	Vector3 relative_left = p_right.xform_inv(p_left.origin);

	return relative_right.dot(Vector3(1, 0, 0)) > 0 ||
			relative_left.dot(Vector3(1, 0, 0)) <
			0; // when these point in the same direction, then both the left
			   // and the right are on the same side of the center
}

void RenIKPlacement::set_falling(bool p_falling) { fall_override = p_falling; }

void RenIKPlacement::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t RenIKPlacement::get_collision_mask() const { return collision_mask; }

void RenIKPlacement::set_collision_mask_bit(int p_bit, bool p_value) {
	uint32_t mask = get_collision_mask();
	if (p_value)
		mask |= 1 << p_bit;
	else
		mask &= ~(1 << p_bit);
	set_collision_mask(mask);
}

bool RenIKPlacement::get_collision_mask_bit(int p_bit) const {
	return get_collision_mask() & (1 << p_bit);
}

void RenIKPlacement::set_collide_with_areas(bool p_clip) {
	collide_with_areas = p_clip;
}

bool RenIKPlacement::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void RenIKPlacement::set_collide_with_bodies(bool p_clip) {
	collide_with_bodies = p_clip;
}

bool RenIKPlacement::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

#endif // _3D_DISABLED
