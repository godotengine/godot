/*************************************************************************/
/*  openxr_action_map.h                                                  */
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

#ifndef OPENXR_ACTION_SETS_H
#define OPENXR_ACTION_SETS_H

#include "core/io/resource.h"

#include "openxr_action_set.h"
#include "openxr_interaction_profile.h"

class OpenXRActionMap : public Resource {
	GDCLASS(OpenXRActionMap, Resource);

private:
	Array action_sets;
	Array interaction_profiles;

protected:
	static void _bind_methods();

public:
	void set_action_sets(Array p_action_sets);
	Array get_action_sets() const;

	void add_action_set(Ref<OpenXRActionSet> p_action_set);
	void remove_action_set(Ref<OpenXRActionSet> p_action_set);

	void set_interaction_profiles(Array p_interaction_profiles);
	Array get_interaction_profiles() const;

	void add_interaction_profile(Ref<OpenXRInteractionProfile> p_interaction_profile);
	void remove_interaction_profile(Ref<OpenXRInteractionProfile> p_interaction_profile);

	void create_default_action_sets();
	void create_editor_action_sets();

	~OpenXRActionMap();
};

#endif // !OPENXR_ACTION_SETS_H
