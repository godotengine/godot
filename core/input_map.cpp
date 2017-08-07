/*************************************************************************/
/*  input_map.cpp                                                        */
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
#include "input_map.h"

#include "os/keyboard.h"
#include "project_settings.h"

InputMap *InputMap::singleton = NULL;

void InputMap::_bind_methods() {

	ClassDB::bind_method(D_METHOD("has_action", "action"), &InputMap::has_action);
	ClassDB::bind_method(D_METHOD("get_actions"), &InputMap::_get_actions);
	ClassDB::bind_method(D_METHOD("add_action", "action"), &InputMap::add_action);
	ClassDB::bind_method(D_METHOD("erase_action", "action"), &InputMap::erase_action);

	ClassDB::bind_method(D_METHOD("action_add_event", "action", "event:InputEvent"), &InputMap::action_add_event);
	ClassDB::bind_method(D_METHOD("action_has_event", "action", "event:InputEvent"), &InputMap::action_has_event);
	ClassDB::bind_method(D_METHOD("action_erase_event", "action", "event:InputEvent"), &InputMap::action_erase_event);
	ClassDB::bind_method(D_METHOD("get_action_list", "action"), &InputMap::_get_action_list);
	ClassDB::bind_method(D_METHOD("event_is_action", "event:InputEvent", "action"), &InputMap::event_is_action);
	ClassDB::bind_method(D_METHOD("load_from_globals"), &InputMap::load_from_globals);
}

void InputMap::add_action(const StringName &p_action) {

	ERR_FAIL_COND(input_map.has(p_action));
	input_map[p_action] = Action();
	static int last_id = 1;
	input_map[p_action].id = last_id;
	last_id++;
}

void InputMap::erase_action(const StringName &p_action) {

	ERR_FAIL_COND(!input_map.has(p_action));
	input_map.erase(p_action);
}

Array InputMap::_get_actions() {

	Array ret;
	List<StringName> actions = get_actions();
	if (actions.empty())
		return ret;

	for (const List<StringName>::Element *E = actions.front(); E; E = E->next()) {

		ret.push_back(E->get());
	}

	return ret;
}

List<StringName> InputMap::get_actions() const {

	List<StringName> actions = List<StringName>();
	if (input_map.empty()) {
		return actions;
	}

	for (Map<StringName, Action>::Element *E = input_map.front(); E; E = E->next()) {
		actions.push_back(E->key());
	}

	return actions;
}

List<Ref<InputEvent> >::Element *InputMap::_find_event(List<Ref<InputEvent> > &p_list, const Ref<InputEvent> &p_event, bool p_action_test) const {

	for (List<Ref<InputEvent> >::Element *E = p_list.front(); E; E = E->next()) {

		const Ref<InputEvent> e = E->get();

		//if (e.type != Ref<InputEvent>::KEY && e.device != p_event.device) -- unsure about the KEY comparison, why is this here?
		//	continue;

		if (e->get_device() != p_event->get_device())
			continue;
		if (e->action_match(p_event))
			return E;
	}

	return NULL;
}

bool InputMap::has_action(const StringName &p_action) const {

	return input_map.has(p_action);
}

void InputMap::action_add_event(const StringName &p_action, const Ref<InputEvent> &p_event) {

	ERR_FAIL_COND(p_event.is_null());
	ERR_FAIL_COND(!input_map.has(p_action));
	if (_find_event(input_map[p_action].inputs, p_event))
		return; //already gots

	input_map[p_action].inputs.push_back(p_event);
}

bool InputMap::action_has_event(const StringName &p_action, const Ref<InputEvent> &p_event) {

	ERR_FAIL_COND_V(!input_map.has(p_action), false);
	return (_find_event(input_map[p_action].inputs, p_event) != NULL);
}

void InputMap::action_erase_event(const StringName &p_action, const Ref<InputEvent> &p_event) {

	ERR_FAIL_COND(!input_map.has(p_action));

	List<Ref<InputEvent> >::Element *E = _find_event(input_map[p_action].inputs, p_event);
	if (E)
		input_map[p_action].inputs.erase(E);
}

Array InputMap::_get_action_list(const StringName &p_action) {

	Array ret;
	const List<Ref<InputEvent> > *al = get_action_list(p_action);
	if (al) {
		for (const List<Ref<InputEvent> >::Element *E = al->front(); E; E = E->next()) {

			ret.push_back(E->get());
		}
	}

	return ret;
}

const List<Ref<InputEvent> > *InputMap::get_action_list(const StringName &p_action) {

	const Map<StringName, Action>::Element *E = input_map.find(p_action);
	if (!E)
		return NULL;

	return &E->get().inputs;
}

bool InputMap::event_is_action(const Ref<InputEvent> &p_event, const StringName &p_action) const {

	Map<StringName, Action>::Element *E = input_map.find(p_action);
	if (!E) {
		ERR_EXPLAIN("Request for nonexistent InputMap action: " + String(p_action));
		ERR_FAIL_COND_V(!E, false);
	}

	Ref<InputEventAction> iea = p_event;
	if (iea.is_valid()) {
		return iea->get_action() == p_action;
	}

	return _find_event(E->get().inputs, p_event, true) != NULL;
}

const Map<StringName, InputMap::Action> &InputMap::get_action_map() const {
	return input_map;
}

void InputMap::load_from_globals() {

	input_map.clear();

	List<PropertyInfo> pinfo;
	ProjectSettings::get_singleton()->get_property_list(&pinfo);

	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		const PropertyInfo &pi = E->get();

		if (!pi.name.begins_with("input/"))
			continue;

		String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());

		add_action(name);

		Array va = ProjectSettings::get_singleton()->get(pi.name);

		for (int i = 0; i < va.size(); i++) {

			Ref<InputEvent> ie = va[i];
			if (ie.is_null())
				continue;
			action_add_event(name, ie);
		}
	}
}

void InputMap::load_default() {

	Ref<InputEventKey> key;

	add_action("ui_accept");
	key.instance();
	key->set_scancode(KEY_ENTER);
	action_add_event("ui_accept", key);

	key.instance();
	key->set_scancode(KEY_KP_ENTER);
	action_add_event("ui_accept", key);

	key.instance();
	key->set_scancode(KEY_SPACE);
	action_add_event("ui_accept", key);

	add_action("ui_select");
	key.instance();
	key->set_scancode(KEY_SPACE);
	action_add_event("ui_select", key);

	add_action("ui_cancel");
	key.instance();
	key->set_scancode(KEY_ESCAPE);
	action_add_event("ui_cancel", key);

	add_action("ui_focus_next");
	key.instance();
	key->set_scancode(KEY_TAB);
	action_add_event("ui_focus_next", key);

	add_action("ui_focus_prev");
	key.instance();
	key->set_scancode(KEY_TAB);
	key->set_shift(true);
	action_add_event("ui_focus_prev", key);

	add_action("ui_left");
	key.instance();
	key->set_scancode(KEY_LEFT);
	action_add_event("ui_left", key);

	add_action("ui_right");
	key.instance();
	key->set_scancode(KEY_RIGHT);
	action_add_event("ui_right", key);

	add_action("ui_up");
	key.instance();
	key->set_scancode(KEY_UP);
	action_add_event("ui_up", key);

	add_action("ui_down");
	key.instance();
	key->set_scancode(KEY_DOWN);
	action_add_event("ui_down", key);

	add_action("ui_page_up");
	key.instance();
	key->set_scancode(KEY_PAGEUP);
	action_add_event("ui_page_up", key);

	add_action("ui_page_down");
	key.instance();
	key->set_scancode(KEY_PAGEDOWN);
	action_add_event("ui_page_down", key);

	//set("display/window/handheld/orientation", "landscape");
}

InputMap::InputMap() {

	ERR_FAIL_COND(singleton);
	singleton = this;
}
