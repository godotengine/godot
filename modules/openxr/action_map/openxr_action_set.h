/*************************************************************************/
/*  openxr_action_set.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef OPENXR_ACTION_SET_H
#define OPENXR_ACTION_SET_H

#include "core/io/resource.h"

#include "openxr_action.h"

class OpenXRActionSet : public Resource {
	GDCLASS(OpenXRActionSet, Resource);

private:
	String localized_name;
	int priority = 0;

	Array actions;

protected:
	static void _bind_methods();

public:
	static Ref<OpenXRActionSet> new_action_set(const char *p_name, const char *p_localized_name, const int p_priority = 0);

	void set_localized_name(const String p_localized_name);
	String get_localized_name() const;

	void set_priority(const int p_priority);
	int get_priority() const;

	void set_actions(Array p_actions);
	Array get_actions() const;

	void add_action(Ref<OpenXRAction> p_action);
	void remove_action(Ref<OpenXRAction> p_action);

	Ref<OpenXRAction> add_new_action(const char *p_name, const char *p_localized_name, const OpenXRAction::ActionType p_action_type, const char *p_toplevel_paths);

	~OpenXRActionSet();
};

#endif // !OPENXR_ACTION_SET_H
