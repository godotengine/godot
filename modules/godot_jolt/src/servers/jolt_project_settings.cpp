#include "jolt_project_settings.hpp"
#include "misc/math.hpp"

namespace {

enum JointWorldNode : int32_t {
	JOINT_WORLD_NODE_A,
	JOINT_WORLD_NODE_B
};

// clang-format off

constexpr char SLEEP_ENABLED[] = "physics/jolt_3d/sleep/enabled";
constexpr char SLEEP_VELOCITY_THRESHOLD[] = "physics/jolt_3d/sleep/velocity_threshold";
constexpr char SLEEP_TIME_THRESHOLD[] = "physics/jolt_3d/sleep/time_threshold";

constexpr char SHAPE_MARGINS[] = "physics/jolt_3d/collisions/use_shape_margins";
constexpr char EDGE_REMOVAL[] = "physics/jolt_3d/collisions/use_enhanced_internal_edge_removal";
constexpr char AREAS_DETECT_STATIC[] = "physics/jolt_3d/collisions/areas_detect_static_bodies";
constexpr char KINEMATIC_CONTACTS[] = "physics/jolt_3d/collisions/report_all_kinematic_contacts";
constexpr char SOFT_BODY_POINT_MARGIN[] = "physics/jolt_3d/collisions/soft_body_point_margin";
constexpr char PAIR_CACHE_ENABLED[] = "physics/jolt_3d/collisions/body_pair_cache_enabled";
constexpr char PAIR_CACHE_DISTANCE[] = "physics/jolt_3d/collisions/body_pair_cache_distance_threshold";
constexpr char PAIR_CACHE_ANGLE[] = "physics/jolt_3d/collisions/body_pair_cache_angle_threshold";

constexpr char JOINT_WORLD_NODE[] = "physics/jolt_3d/joints/world_node";

constexpr char CCD_MOVEMENT_THRESHOLD[] = "physics/jolt_3d/continuous_cd/movement_threshold";
constexpr char CCD_MAX_PENETRATION[] = "physics/jolt_3d/continuous_cd/max_penetration";

constexpr char RECOVERY_ITERATIONS[] = "physics/jolt_3d/kinematics/recovery_iterations";
constexpr char RECOVERY_AMOUNT[] = "physics/jolt_3d/kinematics/recovery_amount";

constexpr char POSITION_ITERATIONS[] = "physics/jolt_3d/solver/position_iterations";
constexpr char VELOCITY_ITERATIONS[] = "physics/jolt_3d/solver/velocity_iterations";
constexpr char POSITION_CORRECTION[] = "physics/jolt_3d/solver/position_correction";
constexpr char ACTIVE_EDGE_THRESHOLD[] = "physics/jolt_3d/solver/active_edge_threshold";
constexpr char BOUNCE_VELOCITY_THRESHOLD[] = "physics/jolt_3d/solver/bounce_velocity_threshold";
constexpr char CONTACT_DISTANCE[] = "physics/jolt_3d/solver/contact_speculative_distance";
constexpr char CONTACT_PENETRATION[] = "physics/jolt_3d/solver/contact_allowed_penetration";

constexpr char MAX_LINEAR_VELOCITY[] = "physics/jolt_3d/limits/max_linear_velocity";
constexpr char MAX_ANGULAR_VELOCITY[] = "physics/jolt_3d/limits/max_angular_velocity";
constexpr char MAX_BODIES[] = "physics/jolt_3d/limits/max_bodies";
constexpr char MAX_PAIRS[] = "physics/jolt_3d/limits/max_body_pairs";
constexpr char MAX_CONTACTS[] = "physics/jolt_3d/limits/max_contact_constraints";
constexpr char MAX_TEMP_MEMORY[] = "physics/jolt_3d/limits/max_temporary_memory";

constexpr char RUN_ON_SEPARATE_THREAD[] = "physics/3d/run_on_separate_thread";
constexpr char MAX_THREADS[] = "threading/worker_pool/max_threads";

// clang-format on

void register_setting(
	const String& p_name,
	const Variant& p_value,
	bool p_needs_restart,
	PropertyHint p_hint,
	const String& p_hint_string
) {
	ProjectSettings* project_settings = ProjectSettings::get_singleton();

	if (!project_settings->has_setting(p_name)) {
		project_settings->set(p_name, p_value);
	}

	Dictionary property_info;
	property_info["name"] = p_name;
	property_info["type"] = p_value.get_type();
	property_info["hint"] = p_hint;
	property_info["hint_string"] = p_hint_string;

	project_settings->_add_property_info_bind(property_info);
	project_settings->set_initial_value(p_name, p_value);
	project_settings->set_restart_if_changed(p_name, p_needs_restart);

	// HACK(mihe): We want our settings to appear in the order we register them in, but if we start
	// the order at 0 we end up moving the entire `physics/` group to the top of the tree view, so
	// instead we give it a hefty starting order and increment from there, which seems to give us
	// the desired effect.
	static int32_t order = 1000000;

	project_settings->set_order(p_name, order++);
}

void register_setting_plain(
	const String& p_name,
	const Variant& p_value,
	bool p_needs_restart = false
) {
	register_setting(p_name, p_value, p_needs_restart, PROPERTY_HINT_NONE, {});
}

void register_setting_hinted(
	const String& p_name,
	const Variant& p_value,
	const String& p_hint_string,
	bool p_needs_restart = false
) {
	register_setting(p_name, p_value, p_needs_restart, PROPERTY_HINT_NONE, p_hint_string);
}

void register_setting_ranged(
	const String& p_name,
	const Variant& p_value,
	const String& p_hint_string,
	bool p_needs_restart = false
) {
	register_setting(p_name, p_value, p_needs_restart, PROPERTY_HINT_RANGE, p_hint_string);
}

void register_setting_enum(
	const String& p_name,
	const Variant& p_value,
	const String& p_hint_string,
	bool p_needs_restart = false
) {
	register_setting(p_name, p_value, p_needs_restart, PROPERTY_HINT_ENUM, p_hint_string);
}

template<typename TType>
TType get_setting(const char* p_setting) {
	const ProjectSettings* project_settings = ProjectSettings::get_singleton();
	const Variant setting_value = project_settings->get_setting_with_override(p_setting);
	const Variant::Type setting_type = setting_value.get_type();
	const Variant::Type expected_type = Variant(TType()).get_type();

	ERR_FAIL_COND_D_MSG(
		setting_type != expected_type,
		vformat(
			"Unexpected type for setting '%s'. Expected type '%s' but found '%s'.",
			p_setting,
			Variant::get_type_name(expected_type),
			Variant::get_type_name(setting_type)
		)
	);

	return setting_value;
}

} // namespace

void JoltProjectSettings::register_settings() {
	register_setting_plain(SLEEP_ENABLED, true);
	register_setting_hinted(SLEEP_VELOCITY_THRESHOLD, 0.03f, U"suffix:m/s");
	register_setting_ranged(SLEEP_TIME_THRESHOLD, 0.5f, U"0,5,0.01,or_greater,suffix:s");

	register_setting_plain(SHAPE_MARGINS, true);
	register_setting_plain(EDGE_REMOVAL, true);
	register_setting_plain(AREAS_DETECT_STATIC, false);
	register_setting_plain(KINEMATIC_CONTACTS, false);

	register_setting_ranged(SOFT_BODY_POINT_MARGIN, 0.01f, U"0,1,0.001,or_greater,suffix:m");

	register_setting_enum(JOINT_WORLD_NODE, JOINT_WORLD_NODE_A, "Node A,Node B");

	register_setting_ranged(CCD_MOVEMENT_THRESHOLD, 75.0f, U"0,100,0.1,suffix:%");
	register_setting_ranged(CCD_MAX_PENETRATION, 25.0f, U"0,100,0.1,suffix:%");

	register_setting_ranged(RECOVERY_ITERATIONS, 4, U"1,8,or_greater");
	register_setting_ranged(RECOVERY_AMOUNT, 40.0f, U"0,100,0.1,suffix:%");

	register_setting_ranged(VELOCITY_ITERATIONS, 10, U"2,16,or_greater");
	register_setting_ranged(POSITION_ITERATIONS, 2, U"1,16,or_greater");
	register_setting_ranged(POSITION_CORRECTION, 20.0f, U"0,100,0.1,suffix:%");
	register_setting_ranged(ACTIVE_EDGE_THRESHOLD, Math::deg_to_rad(50.0f), U"0,90,0.01,radians");
	register_setting_hinted(BOUNCE_VELOCITY_THRESHOLD, 1.0f, U"suffix:m/s");
	register_setting_ranged(CONTACT_DISTANCE, 0.02f, U"0,1,0.00001,or_greater,suffix:m");
	register_setting_ranged(CONTACT_PENETRATION, 0.02f, U"0,1,0.00001,or_greater,suffix:m");
	register_setting_plain(PAIR_CACHE_ENABLED, true);
	register_setting_ranged(PAIR_CACHE_DISTANCE, 0.001f, U"0,0.01,0.00001,or_greater,suffix:m");
	register_setting_ranged(PAIR_CACHE_ANGLE, Math::deg_to_rad(2.0f), U"0,180,0.01,radians");

	register_setting_ranged(MAX_LINEAR_VELOCITY, 500.0f, U"0,500,0.01,or_greater,suffix:m/s");
	register_setting_ranged(MAX_ANGULAR_VELOCITY, 2700.0f, U"0,2700,0.01,or_greater,suffix:°/s");
	register_setting_ranged(MAX_BODIES, 10240, U"1,10240,or_greater", true);
	register_setting_ranged(MAX_PAIRS, 65536, U"8,65536,or_greater");
	register_setting_ranged(MAX_CONTACTS, 20480, U"8,20480,or_greater");
	register_setting_ranged(MAX_TEMP_MEMORY, 32, U"1,32,or_greater,suffix:MiB");
}

bool JoltProjectSettings::is_sleep_enabled() {
	static const auto value = get_setting<bool>(SLEEP_ENABLED);
	return value;
}

float JoltProjectSettings::get_sleep_velocity_threshold() {
	static const auto value = get_setting<float>(SLEEP_VELOCITY_THRESHOLD);
	return value;
}

float JoltProjectSettings::get_sleep_time_threshold() {
	static const auto value = get_setting<float>(SLEEP_TIME_THRESHOLD);
	return value;
}

bool JoltProjectSettings::use_shape_margins() {
	static const auto value = get_setting<bool>(SHAPE_MARGINS);
	return value;
}

bool JoltProjectSettings::areas_detect_static_bodies() {
	static const auto value = get_setting<bool>(AREAS_DETECT_STATIC);
	return value;
}

bool JoltProjectSettings::report_all_kinematic_contacts() {
	static const auto value = get_setting<bool>(KINEMATIC_CONTACTS);
	return value;
}

bool JoltProjectSettings::use_enhanced_edge_removal() {
	static const auto value = get_setting<bool>(EDGE_REMOVAL);
	return value;
}

float JoltProjectSettings::get_soft_body_point_margin() {
	static const auto value = get_setting<float>(SOFT_BODY_POINT_MARGIN);
	return value;
}

bool JoltProjectSettings::use_joint_world_node_a() {
	static const auto value = get_setting<int32_t>(JOINT_WORLD_NODE) == JOINT_WORLD_NODE_A;
	return value;
}

float JoltProjectSettings::get_ccd_movement_threshold() {
	static const auto value = get_setting<float>(CCD_MOVEMENT_THRESHOLD) / 100.0f;
	return value;
}

float JoltProjectSettings::get_ccd_max_penetration() {
	static const auto value = get_setting<float>(CCD_MAX_PENETRATION) / 100.0f;
	return value;
}

int32_t JoltProjectSettings::get_kinematic_recovery_iterations() {
	static const auto value = get_setting<int32_t>(RECOVERY_ITERATIONS);
	return value;
}

float JoltProjectSettings::get_kinematic_recovery_amount() {
	static const auto value = get_setting<float>(RECOVERY_AMOUNT) / 100.0f;
	return value;
}

int32_t JoltProjectSettings::get_velocity_iterations() {
	static const auto value = get_setting<int32_t>(VELOCITY_ITERATIONS);
	return value;
}

int32_t JoltProjectSettings::get_position_iterations() {
	static const auto value = get_setting<int32_t>(POSITION_ITERATIONS);
	return value;
}

float JoltProjectSettings::get_position_correction() {
	static const auto value = get_setting<float>(POSITION_CORRECTION) / 100.0f;
	return value;
}

float JoltProjectSettings::get_active_edge_threshold() {
	static const auto value = use_enhanced_edge_removal()
		? 0.996195f // Math::cos(5 degrees)
		: Math::cos(get_setting<float>(ACTIVE_EDGE_THRESHOLD));

	return value;
}

float JoltProjectSettings::get_bounce_velocity_threshold() {
	static const auto value = get_setting<float>(BOUNCE_VELOCITY_THRESHOLD);
	return value;
}

float JoltProjectSettings::get_contact_distance() {
	static const auto value = get_setting<float>(CONTACT_DISTANCE);
	return value;
}

float JoltProjectSettings::get_contact_penetration() {
	static const auto value = get_setting<float>(CONTACT_PENETRATION);
	return value;
}

bool JoltProjectSettings::is_pair_cache_enabled() {
	static const auto value = get_setting<bool>(PAIR_CACHE_ENABLED);
	return value;
}

float JoltProjectSettings::get_pair_cache_distance() {
	static const auto value = square(get_setting<float>(PAIR_CACHE_DISTANCE));
	return value;
}

float JoltProjectSettings::get_pair_cache_angle() {
	static const auto value = Math::cos(get_setting<float>(PAIR_CACHE_ANGLE) / 2.0f);
	return value;
}

float JoltProjectSettings::get_max_linear_velocity() {
	static const auto value = get_setting<float>(MAX_LINEAR_VELOCITY);
	return value;
}

float JoltProjectSettings::get_max_angular_velocity() {
	static const auto value = Math::deg_to_rad(get_setting<float>(MAX_ANGULAR_VELOCITY));
	return value;
}

int32_t JoltProjectSettings::get_max_bodies() {
	static const auto value = get_setting<int32_t>(MAX_BODIES);
	return value;
}

int32_t JoltProjectSettings::get_max_pairs() {
	static const auto value = get_setting<int32_t>(MAX_PAIRS);
	return value;
}

int32_t JoltProjectSettings::get_max_contact_constraints() {
	static const auto value = get_setting<int32_t>(MAX_CONTACTS);
	return value;
}

int32_t JoltProjectSettings::get_max_temp_memory_mib() {
	static const auto value = get_setting<int32_t>(MAX_TEMP_MEMORY);
	return value;
}

int64_t JoltProjectSettings::get_max_temp_memory_b() {
	static const auto value = get_max_temp_memory_mib() * 1024 * 1024;
	return value;
}

bool JoltProjectSettings::should_run_on_separate_thread() {
	static const auto value = get_setting<bool>(RUN_ON_SEPARATE_THREAD);
	return value;
}

int32_t JoltProjectSettings::get_max_threads() {
	static const auto value = get_setting<int32_t>(MAX_THREADS);
	return value;
}
