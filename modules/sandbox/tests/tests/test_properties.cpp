#include "api.hpp"

// If we need per-node properties in shared sandboxes, we will
// need to use the PER_OBJECT macro to distinguish between them.
struct PlayerState {
	float jump_velocity = -300.0f;
	float player_speed = 150.0f;
	std::string player_name = "Slide Knight";
};
PER_OBJECT(PlayerState);
static PlayerState &get_player_state() {
	return GetPlayerState(get_node());
}

// clang-format off
SANDBOXED_PROPERTIES(3, {
	.name = "player_speed",
	.type = Variant::FLOAT,
	.getter = []() -> Variant { return get_player_state().player_speed; },
	.setter = [](Variant value) -> Variant { return get_player_state().player_speed = value; },
	.default_value = Variant{get_player_state().player_speed},
}, {
	.name = "player_jump_vel",
	.type = Variant::FLOAT,
	.getter = []() -> Variant { return get_player_state().jump_velocity; },
	.setter = [](Variant value) -> Variant { return get_player_state().jump_velocity = value; },
	.default_value = Variant{get_player_state().jump_velocity},
}, {
	.name = "player_name",
	.type = Variant::STRING,
	.getter = []() -> Variant { return get_player_state().player_name; },
	.setter = [](Variant value) -> Variant { return get_player_state().player_name = value.as_std_string(); },
	.default_value = Variant{"Slide Knight"},
});
// clang-format on
