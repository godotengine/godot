/**
 * bt_comment.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_comment.h"

#include "bt_task.h"

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/engine.hpp>
using namespace godot;
#endif

Ref<BTTask> BTComment::clone() const {
	if (Engine::get_singleton()->is_editor_hint()) {
		return BTTask::clone();
	}
	return nullptr;
}

PackedStringArray BTComment::get_configuration_warnings() {
	PackedStringArray warnings = BTTask::get_configuration_warnings();
	if (get_child_count_excluding_comments() > 0) {
		warnings.append("Can only have other comment tasks as children.");
	}
	if (get_parent() == nullptr) {
		warnings.append("Can't be the root task.");
	}
	return warnings;
}
