/**************************************************************************/
/*  component.h                                                           */
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

#include "core/input/input_event.h"
#include "core/io/resource.h"
#include "core/object/object.h"
#include "core/string/string_name.h"

class Component : public Resource {
	GDCLASS(Component, Resource);

public:
	Object *owner = nullptr;

public:
	Component() = default;

	StringName get_component_class();

	void enter_tree();
	void exit_tree();
	void ready();
	void process(double delta);
	void physics_process(double delta);

	bool input(const Ref<InputEvent> &p_event);
	bool shortcut_input(const Ref<InputEvent> &p_key_event);
	bool unhandled_input(const Ref<InputEvent> &p_event);
	bool unhandled_key_input(const Ref<InputEvent> &p_key_event);

	bool is_process_overridden() const;
	bool is_physics_process_overridden() const;

	bool is_input_overridden() const;
	bool is_shortcut_input_overridden() const;
	bool is_unhandled_input_overridden() const;
	bool is_unhandled_key_input_overridden() const;

protected:
	static void _bind_methods();

	GDVIRTUAL0(_enter_tree)
	GDVIRTUAL0(_exit_tree)
	GDVIRTUAL0(_ready)
	GDVIRTUAL1(_process, double)
	GDVIRTUAL1(_physics_process, double)

	GDVIRTUAL1R(bool, _input, Ref<InputEvent>)
	GDVIRTUAL1R(bool, _shortcut_input, Ref<InputEvent>)
	GDVIRTUAL1R(bool, _unhandled_input, Ref<InputEvent>)
	GDVIRTUAL1R(bool, _unhandled_key_input, Ref<InputEvent>)
};
