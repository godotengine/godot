/**
 * blackboard.cpp
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "blackboard.h"

#ifdef LIMBOAI_MODULE
#include "core/variant/variant.h"
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/object.hpp>
using namespace godot;
#endif

Ref<Blackboard> Blackboard::top() const {
	Ref<Blackboard> bb(this);
	while (bb->get_parent().is_valid()) {
		bb = bb->get_parent();
	}
	return bb;
}

Variant Blackboard::get_var(const String &p_name, const Variant &p_default, bool p_complain) const {
	if (data.has(p_name)) {
		return data.get(p_name).get_value();
	} else if (parent.is_valid()) {
		return parent->get_var(p_name, p_default);
	} else {
		if (p_complain) {
			ERR_PRINT(vformat("Blackboard: Variable \"%s\" not found.", p_name));
		}
		return p_default;
	}
}

void Blackboard::set_var(const String &p_name, const Variant &p_value) {
	if (data.has(p_name)) {
		// Not checking type - allowing duck-typing.
		data[p_name].set_value(p_value);
	} else {
		BBVariable var(p_value.get_type());
		var.set_value(p_value);
		data.insert(p_name, var);
	}
}

bool Blackboard::has_var(const String &p_name) const {
	return data.has(p_name) || (parent.is_valid() && parent->has_var(p_name));
}

void Blackboard::erase_var(const String &p_name) {
	data.erase(p_name);
}

void Blackboard::bind_var_to_property(const String &p_name, Object *p_object, const StringName &p_property) {
	ERR_FAIL_COND_MSG(!data.has(p_name), "Blackboard: Binding failed - can't bind variable that doesn't exist.");
	data[p_name].bind(p_object, p_property);
}

void Blackboard::unbind_var(const String &p_name) {
	ERR_FAIL_COND_MSG(data.has(p_name), "Blackboard: Can't unbind variable that doesn't exist.");
	data[p_name].unbind();
}

void Blackboard::add_var(const String &p_name, const BBVariable &p_var) {
	ERR_FAIL_COND(data.has(p_name));
	data.insert(p_name, p_var);
}

void Blackboard::prefetch_nodepath_vars(Node *p_node) {
	ERR_FAIL_COND(p_node == nullptr);
	for (const KeyValue<String, BBVariable> &kv : data) {
		BBVariable var = kv.value;
		if (var.get_value().get_type() == Variant::NODE_PATH) {
			Node *fetched_node = p_node->get_node_or_null(var.get_value());
			if (fetched_node != nullptr) {
				var.set_value(fetched_node);
			}
		}
	}
}

void Blackboard::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_var", "p_name", "p_default", "p_complain"), &Blackboard::get_var, Variant(), true);
	ClassDB::bind_method(D_METHOD("set_var", "p_name", "p_value"), &Blackboard::set_var);
	ClassDB::bind_method(D_METHOD("has_var", "p_name"), &Blackboard::has_var);
	ClassDB::bind_method(D_METHOD("set_parent", "p_blackboard"), &Blackboard::set_parent);
	ClassDB::bind_method(D_METHOD("get_parent"), &Blackboard::get_parent);
	ClassDB::bind_method(D_METHOD("erase_var", "p_name"), &Blackboard::erase_var);
	ClassDB::bind_method(D_METHOD("prefetch_nodepath_vars", "p_node"), &Blackboard::prefetch_nodepath_vars);
	ClassDB::bind_method(D_METHOD("top"), &Blackboard::top);
	ClassDB::bind_method(D_METHOD("bind_var_to_property", "p_name", "p_object", "p_property"), &Blackboard::bind_var_to_property);
	ClassDB::bind_method(D_METHOD("unbind_var", "p_name"), &Blackboard::unbind_var);
}
