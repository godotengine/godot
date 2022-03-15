/*************************************************************************/
/*  openxr_action.h                                                      */
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

#ifndef OPENXR_ACTION_H
#define OPENXR_ACTION_H

#include "core/io/resource.h"

class OpenXRAction : public Resource {
	GDCLASS(OpenXRAction, Resource);

public:
	enum ActionType {
		OPENXR_ACTION_BOOL,
		OPENXR_ACTION_FLOAT,
		OPENXR_ACTION_VECTOR2,
		OPENXR_ACTION_POSE,
		OPENXR_ACTION_HAPTIC,
	};

private:
	String localized_name;
	ActionType action_type = OPENXR_ACTION_FLOAT;

	PackedStringArray toplevel_paths;

protected:
	static void _bind_methods();

public:
	static Ref<OpenXRAction> new_action(const char *p_name, const char *p_localized_name, const ActionType p_action_type, const char *p_toplevel_paths);

	void set_localized_name(const String p_localized_name);
	String get_localized_name() const;

	void set_action_type(const ActionType p_action_type);
	ActionType get_action_type() const;

	void set_toplevel_paths(const PackedStringArray p_toplevel_paths);
	PackedStringArray get_toplevel_paths() const;

	void parse_toplevel_paths(const String p_toplevel_paths);
};

VARIANT_ENUM_CAST(OpenXRAction::ActionType);

#endif // !OPENXR_ACTION_H
