/**
 * blackboard.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BLACKBOARD_H
#define BLACKBOARD_H

#include "bb_variable.h"

#ifdef LIMBOAI_MODULE
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class Blackboard : public RefCounted {
	GDCLASS(Blackboard, RefCounted);

private:
	HashMap<StringName, BBVariable> data;
	Ref<Blackboard> parent;
	Callable changed_value_callback;

protected:
	static void _bind_methods();

public:
	void set_changed_value_callback(const Callable &p_callback) { changed_value_callback = p_callback; }
	
	void set_parent(const Ref<Blackboard> &p_blackboard) { parent = p_blackboard; }
	Ref<Blackboard> get_parent() const { return parent; }

	Ref<Blackboard> top() const;

	Variant get_var(const StringName &p_name, const Variant &p_default = Variant(), bool p_complain = true) const;
	void set_var(const StringName &p_name, const Variant &p_value);
	bool has_var(const StringName &p_name) const;
	void erase_var(const StringName &p_name);
	void clear() { data.clear(); }
	TypedArray<StringName> list_vars() const;

	Dictionary get_vars_as_dict() const;
	void populate_from_dict(const Dictionary &p_dictionary);

	void bind_var_to_property(const StringName &p_name, Object *p_object, const StringName &p_property, bool p_create = false);
	void unbind_var(const StringName &p_name);

	void assign_var(const StringName &p_name, const BBVariable &p_var);

	void link_var(const StringName &p_name, const Ref<Blackboard> &p_target_blackboard, const StringName &p_target_var, bool p_create = false);
};

#endif // BLACKBOARD_H
