/**
 * mode_switch_button.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "mode_switch_button.h"


#include "core/error/error_macros.h"
#include "core/object/object.h"
#include "core/variant/variant.h"

void ModeSwitchButton::add_mode(int p_id, const Ref<Texture2D> &p_icon, const String &p_tooltip) {
	bool unique_id = true;
	for (int i = 0; i < modes.size(); i++) {
		if (modes[i].id == p_id) {
			unique_id = false;
			break;
		}
	}
	ERR_FAIL_COND_MSG(unique_id == false, "ID is already in use by another button state: " + itos(p_id));

	Mode mode;
	mode.id = p_id;
	mode.icon = p_icon;
	mode.tooltip = p_tooltip;
	modes.append(mode);

	if (current_mode_index == -1) {
		_set_mode_by_index(0);
	}
}

void ModeSwitchButton::set_mode(int p_id, bool p_no_signal) {
	ERR_FAIL_COND_MSG(modes.is_empty(), "Cannot set button state with zero states.");

	int idx = -1;
	for (int i = 0; i < modes.size(); i++) {
		if (modes[i].id == p_id) {
			idx = i;
			break;
		}
	}
	ERR_FAIL_COND_MSG(idx == -1, "Button state not found with such id: " + itos(p_id));

	_set_mode_by_index(idx);
	if (!p_no_signal) {
		emit_signal(SNAME("mode_changed"));
	}
}

void ModeSwitchButton::clear() {
	current_mode_index = -1;
	modes.clear();
}

void ModeSwitchButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect(SNAME("pressed"), callable_mp(this, &ModeSwitchButton::next_mode));
		} break;
	}
}

void ModeSwitchButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_mode", "id", "icon", "tooltip"), &ModeSwitchButton::add_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &ModeSwitchButton::get_mode);
	ClassDB::bind_method(D_METHOD("set_mode", "id"), &ModeSwitchButton::set_mode);
	ClassDB::bind_method(D_METHOD("next_mode"), &ModeSwitchButton::next_mode);

	ADD_SIGNAL(MethodInfo("mode_changed"));
}

ModeSwitchButton::ModeSwitchButton() {
}

#endif // ! TOOLS_ENABLED
