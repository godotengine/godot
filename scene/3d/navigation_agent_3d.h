/**************************************************************************/
/*  navigation_agent_3d.h                                                 */
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

#ifndef NAVIGATION_AGENT_3D_H
#define NAVIGATION_AGENT_3D_H

#include "scene/main/node.h"
#include "servers/navigation/navigation_path_query_parameters_3d.h"
#include "servers/navigation/navigation_path_query_result_3d.h"

class Node3D;

class NavigationAgent3D : public Node {
	GDCLASS(NavigationAgent3D, Node);

	Node3D *agent_parent = nullptr;

	RID agent;
	RID map_before_pause;
	RID map_override;

	bool avoidance_enabled = false;
	uint32_t navigation_layers = 1;
	BitField<NavigationPathQueryParameters3D::PathMetadataFlags> path_metadata_flags = NavigationPathQueryParameters3D::PathMetadataFlags::PATH_METADATA_INCLUDE_ALL;

	real_t path_desired_distance = 1.0;
	real_t target_desired_distance = 1.0;
	real_t radius = 0.0;
	real_t navigation_height_offset = 0.0;
	bool ignore_y = false;
	real_t neighbor_distance = 0.0;
	int max_neighbors = 0;
	real_t time_horizon = 0.0;
	real_t max_speed = 0.0;

	real_t path_max_distance = 3.0;

	Vector3 target_position;
	bool target_position_submitted = false;
	Ref<NavigationPathQueryParameters3D> navigation_query;
	Ref<NavigationPathQueryResult3D> navigation_result;
	int navigation_path_index = 0;
	bool velocity_submitted = false;
	Vector3 prev_safe_velocity;
	/// The submitted target velocity
	Vector3 target_velocity;
	bool target_reached = false;
	bool navigation_finished = true;
	// No initialized on purpose
	uint32_t update_frame_id = 0;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	NavigationAgent3D();
	virtual ~NavigationAgent3D();

	RID get_rid() const {
		return agent;
	}

	void set_avoidance_enabled(bool p_enabled);
	bool get_avoidance_enabled() const;

	void set_agent_parent(Node *p_agent_parent);

	void set_navigation_layers(uint32_t p_navigation_layers);
	uint32_t get_navigation_layers() const;

	void set_navigation_layer_value(int p_layer_number, bool p_value);
	bool get_navigation_layer_value(int p_layer_number) const;

	void set_path_metadata_flags(BitField<NavigationPathQueryParameters3D::PathMetadataFlags> p_flags);
	BitField<NavigationPathQueryParameters3D::PathMetadataFlags> get_path_metadata_flags() const {
		return path_metadata_flags;
	}

	void set_navigation_map(RID p_navigation_map);
	RID get_navigation_map() const;

	void set_path_desired_distance(real_t p_dd);
	real_t get_path_desired_distance() const {
		return path_desired_distance;
	}

	void set_target_desired_distance(real_t p_dd);
	real_t get_target_desired_distance() const {
		return target_desired_distance;
	}

	void set_radius(real_t p_radius);
	real_t get_radius() const {
		return radius;
	}

	void set_agent_height_offset(real_t p_hh);
	real_t get_agent_height_offset() const {
		return navigation_height_offset;
	}

	void set_ignore_y(bool p_ignore_y);
	bool get_ignore_y() const {
		return ignore_y;
	}

	void set_neighbor_distance(real_t p_distance);
	real_t get_neighbor_distance() const {
		return neighbor_distance;
	}

	void set_max_neighbors(int p_count);
	int get_max_neighbors() const {
		return max_neighbors;
	}

	void set_time_horizon(real_t p_time);
	real_t get_time_horizon() const {
		return time_horizon;
	}

	void set_max_speed(real_t p_max_speed);
	real_t get_max_speed() const {
		return max_speed;
	}

	void set_path_max_distance(real_t p_pmd);
	real_t get_path_max_distance();

	void set_target_position(Vector3 p_position);
	Vector3 get_target_position() const;

	Vector3 get_next_path_position();

	Ref<NavigationPathQueryResult3D> get_current_navigation_result() const {
		return navigation_result;
	}
	const Vector<Vector3> &get_current_navigation_path() const {
		return navigation_result->get_path();
	}
	int get_current_navigation_path_index() const {
		return navigation_path_index;
	}

	real_t distance_to_target() const;
	bool is_target_reached() const;
	bool is_target_reachable();
	bool is_navigation_finished();
	Vector3 get_final_position();

	void set_velocity(Vector3 p_velocity);
	void _avoidance_done(Vector3 p_new_velocity);

	PackedStringArray get_configuration_warnings() const override;

private:
	void update_navigation();
	void _request_repath();
	void _check_distance_to_target();
};

#endif // NAVIGATION_AGENT_3D_H
