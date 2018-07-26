/*************************************************************************/
/*  input_map.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef INPUT_MAP_H
#define INPUT_MAP_H

#include "core/object.h"
#include "core/os/input_event.h"

class InputMap : public Object {

	GDCLASS(InputMap, Object);

public:
	enum ActionPlayer {
		PLAYER_ALL = 0,
		PLAYER_1, // 1
		PLAYER_2,
		PLAYER_3,
		PLAYER_4,
		PLAYER_5,
		PLAYER_6,
		PLAYER_7,
		PLAYER_8,
		PLAYER_9,
		PLAYER_10,
		PLAYER_11,
		PLAYER_12,
		PLAYER_13,
		PLAYER_14,
		PLAYER_15,
		PLAYER_16
	};

	/**
	* A special value used to signify that a given Action can be triggered by any device
	*/
	static int ALL_DEVICES;

	struct ActionInput {
		ActionPlayer player;
		Ref<InputEvent> event;
	};

	struct Action {
		int id;
		float deadzone;
		List<ActionInput> inputs;
	};

private:
	static InputMap *singleton;

	mutable Map<StringName, Action> input_map;

	List<ActionInput>::Element *_find_event(Action &p_action, const Ref<InputEvent> &p_event, ActionPlayer p_player = PLAYER_ALL, bool p_exact_player = false, bool *p_pressed = NULL, float *p_strength = NULL) const;

	Array _get_action_list(const StringName &p_action, ActionPlayer player = PLAYER_ALL);
	Array _get_actions();

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ InputMap *get_singleton() { return singleton; }

	bool has_action(const StringName &p_action) const;
	List<StringName> get_actions() const;
	void add_action(const StringName &p_action, float p_deadzone = 0.5);
	void erase_action(const StringName &p_action);

	void action_set_deadzone(const StringName &p_action, float p_deadzone);
	void action_add_event(const StringName &p_action, const Ref<InputEvent> &p_event, ActionPlayer player = PLAYER_ALL);
	bool action_has_event(const StringName &p_action, const Ref<InputEvent> &p_event, ActionPlayer player = PLAYER_ALL);
	void action_erase_event(const StringName &p_action, const Ref<InputEvent> &p_event, ActionPlayer player = PLAYER_ALL);
	void action_erase_events(const StringName &p_action);

	const List<ActionInput> *get_action_list(const StringName &p_action, ActionPlayer player = PLAYER_ALL);
	bool event_is_action(const Ref<InputEvent> &p_event, const StringName &p_action, ActionPlayer player = PLAYER_ALL) const;
	bool event_get_action_status(const Ref<InputEvent> &p_event, const StringName &p_action, bool *p_pressed = NULL, float *p_strength = NULL, ActionPlayer player = PLAYER_ALL) const;

	const Map<StringName, Action> &get_action_map() const;
	void load_from_globals();
	void load_default();

	InputMap();
};

VARIANT_ENUM_CAST(InputMap::ActionPlayer);

#endif // INPUT_MAP_H
