/**
 * bb_variable.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_VARIABLE_H
#define BB_VARIABLE_H

#ifdef LIMBOAI_MODULE
#include "core/object/object.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include "godot_cpp/core/object.hpp"
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class BBVariable {
private:
	struct Data {
		// Is used to decide if the value needs to be synced in a derived plan.
		bool value_changed = false;

		SafeRefCount refcount;
		Variant value;
		Variant::Type type = Variant::NIL;
		PropertyHint hint = PropertyHint::PROPERTY_HINT_NONE;
		String hint_string;

		NodePath binding_path;
		uint64_t bound_object = 0;
		StringName bound_property;
	};

	Data *data = nullptr;
	void unref();

public:
	void set_value(const Variant &p_value);
	Variant get_value() const;

	void set_type(Variant::Type p_type);
	Variant::Type get_type() const;

	void set_hint(PropertyHint p_hint);
	PropertyHint get_hint() const;

	void set_hint_string(const String &p_hint_string);
	String get_hint_string() const;

	BBVariable duplicate() const;

	_FORCE_INLINE_ bool is_value_changed() const { return data->value_changed; }
	_FORCE_INLINE_ void reset_value_changed() { data->value_changed = false; }

	bool is_same_prop_info(const BBVariable &p_other) const;
	void copy_prop_info(const BBVariable &p_other);

	// * Editor binding methods
	NodePath get_binding_path() const { return data->binding_path; }
	void set_binding_path(const NodePath &p_binding_path) { data->binding_path = p_binding_path; }
	bool has_binding() { return data->binding_path.is_empty(); }

	// * Runtime binding methods
	_FORCE_INLINE_ bool is_bound() const { return data->bound_object != 0; }
	void bind(Object *p_object, const StringName &p_property);
	void unbind();

	bool operator==(const BBVariable &p_var) const;
	bool operator!=(const BBVariable &p_var) const;
	void operator=(const BBVariable &p_var);

	BBVariable(const BBVariable &p_var);
	BBVariable(Variant::Type p_type = Variant::Type::NIL,const Variant &p_value = Variant(), PropertyHint p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "");
	~BBVariable();
};

#endif // BB_VARIABLE_H
