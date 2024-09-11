/**
 * mode_switch_button.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef MODE_SWITCH_BUTTON
#define MODE_SWITCH_BUTTON


#include "core/typedefs.h"
#include "scene/gui/button.h"
#include "scene/resources/texture.h"

class ModeSwitchButton : public Button {
	GDCLASS(ModeSwitchButton, Button);

private:
	struct Mode {
		int id = 0;
		Ref<Texture2D> icon = nullptr;
		String tooltip = "";
	};
	int current_mode_index = -1;

	Vector<Mode> modes;

	_FORCE_INLINE_ void _set_mode_by_index(int p_index) {
		current_mode_index = p_index;
		this->set_icon(modes[current_mode_index].icon);
		if (!modes[current_mode_index].tooltip.is_empty()) {
			set_tooltip_text(modes[current_mode_index].tooltip);
		}
	}

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	void add_mode(int p_id, const Ref<Texture2D> &p_icon, const String &p_tooltip = "");
	int get_mode() const { return modes.size() > 0 ? modes[current_mode_index].id : -1; }
	void set_mode(int p_id, bool p_no_signal = false);
	_FORCE_INLINE_ void next_mode() { set_mode((current_mode_index + 1) % modes.size()); };
	void clear();

	ModeSwitchButton();
};

#endif // MODE_SWITCH_BUTTON_H

#endif // ! TOOLS_ENABLED
