/**************************************************************************/
/*  test_properties.cpp                                                   */
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
