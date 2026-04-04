/**************************************************************************/
/*  open_xr_action_map.hpp                                                */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRActionSet;
class OpenXRInteractionProfile;
class String;

class OpenXRActionMap : public Resource {
	GDEXTENSION_CLASS(OpenXRActionMap, Resource)

public:
	void set_action_sets(const Array &p_action_sets);
	Array get_action_sets() const;
	int32_t get_action_set_count() const;
	Ref<OpenXRActionSet> find_action_set(const String &p_name) const;
	Ref<OpenXRActionSet> get_action_set(int32_t p_idx) const;
	void add_action_set(const Ref<OpenXRActionSet> &p_action_set);
	void remove_action_set(const Ref<OpenXRActionSet> &p_action_set);
	void set_interaction_profiles(const Array &p_interaction_profiles);
	Array get_interaction_profiles() const;
	int32_t get_interaction_profile_count() const;
	Ref<OpenXRInteractionProfile> find_interaction_profile(const String &p_name) const;
	Ref<OpenXRInteractionProfile> get_interaction_profile(int32_t p_idx) const;
	void add_interaction_profile(const Ref<OpenXRInteractionProfile> &p_interaction_profile);
	void remove_interaction_profile(const Ref<OpenXRInteractionProfile> &p_interaction_profile);
	void create_default_action_sets();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

