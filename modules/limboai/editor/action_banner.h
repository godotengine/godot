/**
 * action_banner.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* action_banner.h */

#ifdef TOOLS_ENABLED

#ifndef ACTION_BANNER_H
#define ACTION_BANNER_H

#ifdef LIMBOAI_MODULE
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/texture_rect.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/margin_container.hpp>
#include <godot_cpp/classes/texture_rect.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class ActionBanner : public MarginContainer {
	GDCLASS(ActionBanner, MarginContainer);

private:
	TextureRect *icon;
	Label *message;
	HBoxContainer *hbox;

	void _execute_action(const Callable &p_action, bool p_auto_close);

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	void set_text(const String &p_text);
	String get_text() const;

	void add_action(const String &p_name, const Callable &p_action, bool p_auto_close = false);
	void add_spacer();

	void close();

	ActionBanner();
};

#endif // ACTION_BANNER_H

#endif // TOOLS_ENABLED
