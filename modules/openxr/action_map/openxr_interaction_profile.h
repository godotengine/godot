/*************************************************************************/
/*  openxr_interaction_profile.h                                         */
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

#ifndef OPENXR_INTERACTION_PROFILE_H
#define OPENXR_INTERACTION_PROFILE_H

#include "core/io/resource.h"

#include "openxr_action.h"

class OpenXRIPBinding : public Resource {
	GDCLASS(OpenXRIPBinding, Resource);

private:
	Ref<OpenXRAction> action;
	PackedStringArray paths;

protected:
	static void _bind_methods();

public:
	static Ref<OpenXRIPBinding> new_binding(const Ref<OpenXRAction> p_action, const char *p_paths);

	void set_action(const Ref<OpenXRAction> p_action);
	Ref<OpenXRAction> get_action() const;

	void set_paths(const PackedStringArray p_paths);
	PackedStringArray get_paths() const;

	void parse_paths(const String p_paths);

	~OpenXRIPBinding();
};

class OpenXRInteractionProfile : public Resource {
	GDCLASS(OpenXRInteractionProfile, Resource);

private:
	String interaction_profile_path;
	Array bindings;

protected:
	static void _bind_methods();

public:
	static Ref<OpenXRInteractionProfile> new_profile(const char *p_input_profile_path);

	void set_interaction_profile_path(const String p_input_profile_path);
	String get_interaction_profile_path() const;

	void set_bindings(Array p_bindings);
	Array get_bindings() const;

	void add_binding(Ref<OpenXRIPBinding> p_binding);
	void remove_binding(Ref<OpenXRIPBinding> p_binding);

	void add_new_binding(const Ref<OpenXRAction> p_action, const char *p_paths);

	~OpenXRInteractionProfile();
};

#endif // !OPENXR_INTERACTION_PROFILE_H
