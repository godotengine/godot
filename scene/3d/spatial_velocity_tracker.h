#ifndef SPATIAL_VELOCITY_TRACKER_H
#define SPATIAL_VELOCITY_TRACKER_H

#include "scene/3d/spatial.h"

class SpatialVelocityTracker : public Reference {
	GDCLASS(SpatialVelocityTracker, Reference)

	struct PositionHistory {
		uint64_t frame;
		Vector3 position;
	};

	bool fixed_step;
	Vector<PositionHistory> position_history;
	int position_history_len;

protected:
	static void _bind_methods();

public:
	void reset(const Vector3 &p_new_pos);
	void set_track_fixed_step(bool p_track_fixed_step);
	bool is_tracking_fixed_step() const;
	void update_position(const Vector3 &p_position);
	Vector3 get_tracked_linear_velocity() const;

	SpatialVelocityTracker();
};

#endif // SPATIAL_VELOCITY_TRACKER_H
