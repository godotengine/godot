/**************************************************************************/
/*  actor.h                                                               */
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

#pragma once

#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "modules/components/component.h"

class Actor : public Object {
	GDCLASS(Actor, Object);

protected:
	HashMap<StringName, Ref<Component>> _component_resources;
	HashSet<Ref<Component>> _process_group;
	HashSet<Ref<Component>> _physics_process_group;
	HashSet<Ref<Component>> _input_group;
	HashSet<Ref<Component>> _shortcut_input_group;
	HashSet<Ref<Component>> _unhandled_input_group;
	HashSet<Ref<Component>> _unhandled_key_input_group;

public:
	Actor() = default;
	virtual ~Actor() = default;

	bool has_component(StringName component_class) const;
	Ref<Component> get_component(StringName component_class) const;
	virtual void set_component(Ref<Component> value);
	virtual void remove_component(StringName component_class);
	void get_component_list(List<Ref<Component>> *out) const;
	void get_component_class_list(List<StringName> *out) const;

	void call_components_enter_tree();
	void call_components_exit_tree();
	void call_components_ready();
	void call_components_process(double delta);
	void call_components_physics_process(double delta);

	bool call_components_input(const Ref<InputEvent> &p_event);
	bool call_components_shortcut_input(const Ref<InputEvent> &p_key_event);
	bool call_components_unhandled_input(const Ref<InputEvent> &p_event);
	bool call_components_unhandled_key_input(const Ref<InputEvent> &p_key_event);

protected:
	static void _bind_methods();
	void _get_property_list(List<PropertyInfo> *out) const;
	bool _get(const StringName &p_property, Variant &r_value) const;
	bool _set(const StringName &p_property, const Variant &p_value);

	bool _remove_component(StringName component_class);
};
