/**************************************************************************/
/*  script_instance.h                                                     */
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

#include "../sandbox.h"
#include "core/core_constants.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/object/script_instance.h"
#include "core/object/script_language.h"
#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/templates/pair.h"
#include "core/templates/self_list.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

class ELFScript;

class ELFScriptInstance : public ScriptInstance {
	Object *owner;
	Ref<ELFScript> script;
	Sandbox *current_sandbox = nullptr;
	mutable List<MethodInfo> methods_info;
	mutable bool has_updated_methods = false;
	bool auto_created_sandbox = false;
	bool recursive_trap = false;

	void update_methods() const;

	// Retrieve the sandbox and whether it was created automatically or not
	std::tuple<Sandbox *, bool> get_sandbox() const;
	Sandbox *create_sandbox(const Ref<ELFScript> &p_script);
	friend class ELFScript;
	friend class CPPScriptInstance;

	static inline std::vector<StringName> godot_functions;
	static inline std::unordered_set<std::string> sandbox_functions;

public:
	bool set(const StringName &p_name, const Variant &p_value) override;
	bool get(const StringName &p_name, Variant &r_ret) const override;
	void get_property_list(List<PropertyInfo> *p_properties) const override;
	Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid = nullptr) const override;
	void validate_property(PropertyInfo &p_property) const override;
	bool property_can_revert(const StringName &p_name) const override;
	bool property_get_revert(const StringName &p_name, Variant &r_ret) const override;
	Object *get_owner() override;
	void get_method_list(List<MethodInfo> *p_list) const override;
	bool has_method(const StringName &p_method) const override;
	int get_method_argument_count(const StringName &p_method, bool *r_is_valid = nullptr) const override;
	Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) override;
	void notification(int p_notification, bool p_reversed = false) override;
	String to_string(bool *r_valid) override;
	void refcount_incremented() override;
	bool refcount_decremented() override;
	Ref<Script> get_script() const override;
	bool is_placeholder() const override;
	void property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid = nullptr) override;
	Variant property_get_fallback(const StringName &p_name, bool *r_valid = nullptr) override;
	ScriptLanguage *get_language() override;

	ELFScript *get_elf_script() const {
		return script.ptr();
	}

	ELFScriptInstance(Object *p_owner, const Ref<ELFScript> p_script);
	~ELFScriptInstance();
};
