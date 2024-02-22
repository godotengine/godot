/**
 * blackboard.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
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
	HashMap<String, BBVariable> data;
	Ref<Blackboard> parent;

protected:
	static void _bind_methods();

public:
	void set_parent(const Ref<Blackboard> &p_blackboard) { parent = p_blackboard; }
	Ref<Blackboard> get_parent() const { return parent; }

	Ref<Blackboard> top() const;

	Variant get_var(const String &p_name, const Variant &p_default, bool p_complain = true) const;
	void set_var(const String &p_name, const Variant &p_value);
	bool has_var(const String &p_name) const;
	void erase_var(const String &p_name);

	void bind_var_to_property(const String &p_name, Object *p_object, const StringName &p_property);
	void unbind_var(const String &p_name);

	void add_var(const String &p_name, const BBVariable &p_var);

	void prefetch_nodepath_vars(Node *p_node);

	// TODO: Add serialization API.
};

#endif // BLACKBOARD_H
