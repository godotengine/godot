/*************************************************************************/
/*  input_map.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "object.h"

class InputMap : public Object {

	GDCLASS(InputMap, Object);

public:
	struct Action {
		int id;
		List<InputEvent> inputs;
	};

private:
	static InputMap *singleton;

	mutable Map<StringName, Action> input_map;
	mutable Map<int, StringName> input_id_map;

	List<InputEvent>::Element *_find_event(List<InputEvent> &p_list, const InputEvent &p_event, bool p_action_test = false) const;

	Array _get_action_list(const StringName &p_action);
	Array _get_actions();

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ InputMap *get_singleton() { return singleton; }

	bool has_action(const StringName &p_action) const;
	int get_action_id(const StringName &p_action) const;
	StringName get_action_from_id(int p_id) const;
	List<StringName> get_actions() const;
	void add_action(const StringName &p_action);
	void erase_action(const StringName &p_action);

	void action_add_event(const StringName &p_action, const InputEvent &p_event);
	bool action_has_event(const StringName &p_action, const InputEvent &p_event);
	void action_erase_event(const StringName &p_action, const InputEvent &p_event);

	const List<InputEvent> *get_action_list(const StringName &p_action);
	bool event_is_action(const InputEvent &p_event, const StringName &p_action) const;

	const Map<StringName, Action> &get_action_map() const;
	void load_from_globals();
	void load_default();

	InputMap();
};

#endif // INPUT_MAP_H
