/**************************************************************************/
/*  script_cpp_instance.cpp                                               */
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

#include "script_cpp_instance.h"

#include "../elf/script_elf.h"
#include "../elf/script_instance.h"
#include "../elf/script_instance_helper.h" // register_types.h
#include "../scoped_tree_base.h"
#include "core/object/object.h"
#include "core/templates/local_vector.h"
#include "script_cpp.h"
#include "script_language_cpp.h"
static constexpr bool VERBOSE_LOGGING = false;

void CPPScriptInstance::set_script_instance(ELFScriptInstance *p_instance) {
	this->elf_script_instance = p_instance;
	if (p_instance) {
		// XXX: If elf_script is already set, and is different, that is a problem.
		if (p_instance->script.is_null()) {
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::set_script_instance: p_instance->script is null");
			}
			return;
		}
		this->script->elf_script = p_instance->script;
	}
}
void CPPScriptInstance::unset_script_instance() {
	if (this->elf_script_instance) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT(String("CPPScriptInstance::unset_script_instance: ") +
					String(Object::cast_to<Node>(this->elf_script_instance->get_owner())->get_path()));
		}
		this->elf_script_instance = nullptr;
	}
	if (this->managed_esi != nullptr) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScriptInstance::unset_script_instance: managed_esi is not null, deleting it");
		}
		memdelete(this->managed_esi);
		this->managed_esi = nullptr;
	}
}
void CPPScriptInstance::manage_script_instance(ELFScript *p_script) {
	if (this->managed_esi != nullptr) {
		// If we already have a managed ESI, we need to free it.
		memdelete(this->managed_esi);
	}
	this->managed_esi = memnew(ELFScriptInstance(get_owner(), p_script));
	if (this->managed_esi == nullptr) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScriptInstance::manage_script_instance: managed_esi is null");
		}
		return;
	}
	this->set_script_instance(this->managed_esi);
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::manage_script_instance: managed_esi set to " +
				String(Object::cast_to<Node>(this->managed_esi->get_owner())->get_path()));
	}
}

bool CPPScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	if (p_name == StringName("associated_script")) {
		// This is a property setter to set the associated script
		Object *object = p_value.operator Object *();
		if (object == nullptr) {
			this->unset_script_instance();
			return true;
		}
		ELFScript *new_elf_script = Object::cast_to<ELFScript>(object->get_script());
		if (new_elf_script == nullptr) {
			// XXX: TODO: It may be possible to create an artificial ELFScriptInstance based
			// on p_value being an ELFScript, but that is not implemented yet. We could then
			// set the script instance to that.
			if (ELFScript *elf_script = Object::cast_to<ELFScript>(p_value.operator Object *()); elf_script) {
				// This is an ELFScript, but we need an ELFScriptInstance in order to proxy
				// the calls to the underlying Sandbox instance. Create a new instance?
				if constexpr (VERBOSE_LOGGING) {
					ERR_PRINT("CPPScriptInstance::set: associated_script argument is an ELFScript");
				}
				this->manage_script_instance(elf_script);
				return true;
			}
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::set: associated_script argument is not an ELFScript");
			}
			return false;
		}
		this->unset_script_instance();
		this->set_script_instance(new_elf_script->get_script_instance(object));
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScriptInstance::set: associated_script to " +
					new_elf_script->get_path());
		}
		return true;
	}

	if (elf_script_instance) {
		return elf_script_instance->set(p_name, p_value);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::set " + p_name);
	}
	return false;
}

bool CPPScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	static const StringName s_script("script");
	static const StringName s_associated_script("associated_script");
	if (p_name == s_associated_script) {
		// This is a property getter to get the associated script
		if (this->managed_esi != nullptr) {
			// If we have a managed script instance, we can return it.
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::get: associated_script is managed");
			}
			r_ret = this->managed_esi->script;
			return true;
		} else if (elf_script_instance) {
			r_ret = elf_script_instance->get_owner();
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::get: associated_script is " +
						String(Object::cast_to<Node>(elf_script_instance->get_owner())->get_path()));
			}
			return true;
		}
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScriptInstance::get: associated_script is not set");
		}
		return false;
	} else if (p_name == s_script) {
		r_ret = script;
		return true;
	}

	if (elf_script_instance) {
		return elf_script_instance->get(p_name, r_ret);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::get " + p_name);
	}
	return false;
}

::String CPPScriptInstance::to_string(bool *r_is_valid) {
	return "<CPPScript>";
}

void CPPScriptInstance::notification(int32_t p_what, bool p_reversed) {
}

Variant CPPScriptInstance::callp(
		const StringName &p_method,
		const Variant **p_args, const int p_argument_count,
		Callable::CallError &r_error) {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::callp " + p_method);
	}
	if (p_method == StringName("set_associated_script")) {
		if (p_argument_count != 1) {
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::callp: set_associated_script requires exactly one argument");
			}
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			return Variant();
		}
		Object *object = p_args[0]->operator Object *();
		if (object == nullptr) {
			this->unset_script_instance();
			r_error.error = Callable::CallError::CALL_OK;
			return Variant();
		}
		ELFScript *new_elf_script = Object::cast_to<ELFScript>(object->get_script().operator Object *());
		if (new_elf_script == nullptr) {
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::callp: set_associated_script argument is not an ELFScript");
			}
			if (ELFScript *elf_script = Object::cast_to<ELFScript>(object); elf_script) {
				// This is an ELFScript, but we need an ELFScriptInstance in order to proxy
				// the calls to the underlying Sandbox instance. Create a new instance?
				if constexpr (VERBOSE_LOGGING) {
					ERR_PRINT("CPPScriptInstance::callp: set_associated_script argument is an ELFScript");
				}
				this->manage_script_instance(elf_script);
				r_error.error = Callable::CallError::CALL_OK;
				return Variant();
			}
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			return Variant();
		}
		this->unset_script_instance();
		this->set_script_instance(new_elf_script->get_script_instance(object));
		r_error.error = Callable::CallError::CALL_OK;
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScriptInstance::callp: set_associated_script to " +
					String(Object::cast_to<Node>(elf_script_instance->get_owner())->get_path()));
		}
		return Variant();
	} else if (p_method == StringName("get_associated_script")) {
		// This is a property getter to get the associated script
		if (this->managed_esi != nullptr) {
			// If we have a managed script instance, we can return it.
			r_error.error = Callable::CallError::CALL_OK;
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::callp: get_associated_script is managed");
			}
			return this->managed_esi->script;
		} else if (elf_script_instance) {
			r_error.error = Callable::CallError::CALL_OK;
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::callp: get_associated_script is set to " +
						String(Object::cast_to<Node>(elf_script_instance->get_owner())->get_path()));
			}
			return elf_script_instance->get_owner();
		}
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScriptInstance::callp: get_associated_script is not set");
		}
		r_error.error = Callable::CallError::CALL_OK;
		return Variant();
	} else if (elf_script_instance) {
		Ref<ELFScript> &elf_script = elf_script_instance->script;
		if (!elf_script.is_valid()) {
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::callp: script is null");
			}
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return Variant();
		}

		// Try to call the method on the elf_script_instance, but use
		// this instance owner as the base for the Sandbox node-tree.
		if (elf_script->function_names.has(p_method)) {
			auto [sandbox, auto_created] = elf_script_instance->get_sandbox();
			if (sandbox && sandbox->has_program_loaded()) {
				// Set the Sandbox instance tree base to the owner node
				ScopedTreeBase stb(sandbox, ::Object::cast_to<Node>(this->owner));
				// Need to convert Callable::CallError to compatible type for vmcall_fn
				Callable::CallError temp_error;
				Variant result = sandbox->vmcall_fn(p_method, p_args, p_argument_count, temp_error);
				r_error = temp_error;
				return result;
			}
		}
		if (p_method == StringName("_get_editor_name")) {
			r_error.error = Callable::CallError::CALL_OK;
			return Variant("CPPScriptInstance");
		}
		// Fallback: callp on the elf_script_instance directly
		return elf_script_instance->callp(p_method, p_args, p_argument_count, r_error);
	}
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

void CPPScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	if (elf_script_instance) {
		elf_script_instance->get_method_list(p_list);
	}

	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::get_method_list");
	}
}

void CPPScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	if (elf_script_instance) {
		elf_script_instance->get_property_list(p_properties);
	}

	// Add a property for 'associated_script'
	PropertyInfo prop(Variant::OBJECT, "associated_script", PROPERTY_HINT_RESOURCE_TYPE, "ELFScript",
			PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT);
	p_properties->push_back(prop);

	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::get_property_list: added associated_script property");
	}
}

Variant::Type CPPScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	if (p_name == StringName("associated_script")) {
		// This is a property getter to get the associated script
		if (r_is_valid) {
			*r_is_valid = true;
		}
		return Variant::OBJECT; // The type of the associated script is an Object
	}
	if (elf_script_instance) {
		return elf_script_instance->get_property_type(p_name, r_is_valid);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::get_property_type " + p_name);
	}
	*r_is_valid = false;
	return Variant::NIL;
}

void CPPScriptInstance::validate_property(PropertyInfo &p_property) const {
	if (p_property.name == StringName("associated_script")) {
		return;
	}
	if (elf_script_instance) {
		elf_script_instance->validate_property(p_property);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::validate_property");
	}
}

int CPPScriptInstance::get_method_argument_count(const StringName &p_method, bool *r_is_valid) const {
	if (r_is_valid) {
		*r_is_valid = false;
	}
	return 0;
}

bool CPPScriptInstance::has_method(const StringName &p_name) const {
	if (p_name == StringName("set_associated_script")) {
		return true; // This method is always available
	} else if (p_name == StringName("get_associated_script")) {
		return true; // This method is always available
	}
	if (elf_script_instance) {
		return elf_script_instance->has_method(p_name);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::has_method " + p_name);
	}
	return false;
}

bool CPPScriptInstance::property_can_revert(const StringName &p_name) const {
	if (p_name == StringName("associated_script")) {
		return true; // The associated_script can always be reverted
	}
	if (elf_script_instance) {
		return elf_script_instance->property_can_revert(p_name);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::property_can_revert " + p_name);
	}
	return false;
}

bool CPPScriptInstance::property_get_revert(const StringName &p_name, Variant &r_ret) const {
	if (p_name == StringName("associated_script")) {
		r_ret = Variant();
		return true;
	}
	if (elf_script_instance) {
		return elf_script_instance->property_get_revert(p_name, r_ret);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScriptInstance::property_get_revert " + p_name);
	}
	r_ret = Variant();
	return false;
}

void CPPScriptInstance::refcount_incremented() {
}

bool CPPScriptInstance::refcount_decremented() {
	return false;
}

Object *CPPScriptInstance::get_owner() {
	return owner;
}

Ref<Script> CPPScriptInstance::get_script() const {
	return script;
}

bool CPPScriptInstance::is_placeholder() const {
	return false;
}

void CPPScriptInstance::property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) {
	*r_valid = false;
}

Variant CPPScriptInstance::property_get_fallback(const StringName &p_name, bool *r_valid) {
	*r_valid = false;
	return Variant::NIL;
}

ScriptLanguage *CPPScriptInstance::get_language() {
	return get_elf_language_singleton();
}

CPPScriptInstance::CPPScriptInstance(Object *p_owner, const Ref<CPPScript> p_script) :
		owner(p_owner), script(p_script) {
	if (script->elf_script.is_null()) {
		script->detect_script_instance();
	}
	if (script->elf_script.is_valid()) {
		// If the script has an associated ELFScript, we can create an ELFScriptInstance
		this->managed_esi = memnew(ELFScriptInstance(p_owner, script->elf_script));
		if (this->managed_esi == nullptr) {
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScriptInstance::CPPScriptInstance: managed_esi is null");
			}
			return;
		}
		if constexpr (VERBOSE_LOGGING) {
			bool r_valid;
			ERR_PRINT("CPPScriptInstance: managed_esi set to " +
					this->managed_esi->to_string(&r_valid));
		}
		this->set_script_instance(this->managed_esi);
	} else {
		this->managed_esi = nullptr;
	}
}

CPPScriptInstance::~CPPScriptInstance() {
	if (this->script.is_valid()) {
		script->remove_instance(this);
	}
	if (this->managed_esi != nullptr) {
		memdelete(this->managed_esi);
		this->managed_esi = nullptr;
	}
}
