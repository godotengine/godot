#pragma once
#include "../common.h"
#include "misc/error_macros.hpp"

class JoltProjectSettings {
public:
	static void register_settings();

	static bool is_sleep_enabled();

	static float get_sleep_velocity_threshold();

	static float get_sleep_time_threshold();

	static bool use_shape_margins();

	static bool areas_detect_static_bodies();

	static bool report_all_kinematic_contacts();

	static bool use_enhanced_edge_removal();

	static float get_soft_body_point_margin();

	static bool use_joint_world_node_a();

	static float get_ccd_movement_threshold();

	static float get_ccd_max_penetration();

	static int32_t get_kinematic_recovery_iterations();

	static float get_kinematic_recovery_amount();

	static int32_t get_velocity_iterations();

	static int32_t get_position_iterations();

	static float get_position_correction();

	static float get_active_edge_threshold();

	static float get_bounce_velocity_threshold();

	static float get_contact_distance();

	static float get_contact_penetration();

	static bool is_pair_cache_enabled();

	static float get_pair_cache_distance();

	static float get_pair_cache_angle();

	static float get_max_linear_velocity();

	static float get_max_angular_velocity();

	static int32_t get_max_bodies();

	static int32_t get_max_pairs();

	static int32_t get_max_contact_constraints();

	static int32_t get_max_temp_memory_mib();

	static int64_t get_max_temp_memory_b();

	static bool should_run_on_separate_thread();

	static int32_t get_max_threads();
};
