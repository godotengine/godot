#include "spatial_velocity_tracker.h"
#include "engine.h"

void SpatialVelocityTracker::set_track_physics_step(bool p_track_physics_step) {

	physics_step = p_track_physics_step;
}

bool SpatialVelocityTracker::is_tracking_physics_step() const {

	return physics_step;
}
void SpatialVelocityTracker::update_position(const Vector3 &p_position) {

	PositionHistory ph;
	ph.position = p_position;
	if (physics_step) {
		ph.frame = Engine::get_singleton()->get_physics_frames();
	} else {
		ph.frame = Engine::get_singleton()->get_idle_frame_ticks();
	}

	if (position_history_len == 0 || position_history[0].frame != ph.frame) { //in same frame, use latest
		position_history_len = MIN(position_history.size(), position_history_len + 1);
		for (int i = position_history_len - 1; i > 0; i--) {
			position_history[i] = position_history[i - 1];
		}
	}

	position_history[0] = ph;
}
Vector3 SpatialVelocityTracker::get_tracked_linear_velocity() const {

	Vector3 linear_velocity;

	float max_time = 1 / 5.0; //maximum time to interpolate a velocity

	Vector3 distance_accum;
	float time_accum = 0.0;
	float base_time = 0.0;

	if (position_history_len) {
		if (physics_step) {
			uint64_t base = Engine::get_singleton()->get_physics_frames();
			base_time = float(base - position_history[0].frame) / Engine::get_singleton()->get_iterations_per_second();
		} else {
			uint64_t base = Engine::get_singleton()->get_idle_frame_ticks();
			base_time = double(base - position_history[0].frame) / 1000000.0;
		}
	}

	for (int i = 0; i < position_history_len - 1; i++) {
		float delta = 0.0;
		uint64_t diff = position_history[i].frame - position_history[i + 1].frame;
		Vector3 distance = position_history[i].position - position_history[i + 1].position;

		if (physics_step) {
			delta = float(diff) / Engine::get_singleton()->get_iterations_per_second();
		} else {
			delta = double(diff) / 1000000.0;
		}

		if (base_time + time_accum + delta > max_time)
			break;

		distance_accum += distance;
		time_accum += delta;
	}

	if (time_accum) {
		linear_velocity = distance_accum / time_accum;
	}

	return linear_velocity;
}

void SpatialVelocityTracker::reset(const Vector3 &p_new_pos) {

	PositionHistory ph;
	ph.position = p_new_pos;
	if (physics_step) {
		ph.frame = Engine::get_singleton()->get_physics_frames();
	} else {
		ph.frame = Engine::get_singleton()->get_idle_frame_ticks();
	}

	position_history[0] = ph;
	position_history_len = 1;
}

void SpatialVelocityTracker::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_track_physics_step", "enable"), &SpatialVelocityTracker::set_track_physics_step);
	ClassDB::bind_method(D_METHOD("is_tracking_physics_step"), &SpatialVelocityTracker::is_tracking_physics_step);
	ClassDB::bind_method(D_METHOD("update_position", "position"), &SpatialVelocityTracker::update_position);
	ClassDB::bind_method(D_METHOD("get_tracked_linear_velocity"), &SpatialVelocityTracker::get_tracked_linear_velocity);
	ClassDB::bind_method(D_METHOD("reset", "position"), &SpatialVelocityTracker::reset);
}

SpatialVelocityTracker::SpatialVelocityTracker() {
	position_history.resize(4); // should be configurable
	position_history_len = 0;
	physics_step = false;
}
