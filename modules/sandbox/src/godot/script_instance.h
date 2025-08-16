/*************************************************************************/
/* Copyright (c) 2023 David Snopek                                       */
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

#pragma once

#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_language.hpp>
#include <godot_cpp/templates/list.hpp>
#include <godot_cpp/templates/pair.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

using namespace godot;

namespace godot {

class ScriptInstanceExtension {
	static GDExtensionScriptInstanceInfo3 script_instance_info;

public:
	static GDExtensionScriptInstancePtr create_native_instance(ScriptInstanceExtension *p_instance) {
		return internal::gdextension_interface_script_instance_create3(&script_instance_info, p_instance);
	}

	virtual bool set(const StringName &p_name, const Variant &p_value) = 0;
	virtual bool get(const StringName &p_name, Variant &r_ret) const = 0;
	virtual const GDExtensionPropertyInfo *get_property_list(uint32_t *r_count) const = 0;
	virtual void free_property_list(const GDExtensionPropertyInfo *p_list, uint32_t p_count) const = 0;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const = 0;
	virtual bool validate_property(GDExtensionPropertyInfo &p_property) const = 0;
	virtual bool get_class_category(GDExtensionPropertyInfo &r_class_category) const;
	virtual bool property_can_revert(const StringName &p_name) const = 0;
	virtual bool property_get_revert(const StringName &p_name, Variant &r_ret) const = 0;
	virtual Object *get_owner() = 0;
	virtual void get_property_state(GDExtensionScriptInstancePropertyStateAdd p_add_func, void *p_userdata) = 0;
	virtual const GDExtensionMethodInfo *get_method_list(uint32_t *r_count) const = 0;
	virtual void free_method_list(const GDExtensionMethodInfo *p_list, uint32_t p_count) const = 0;
	virtual bool has_method(const StringName &p_method) const = 0;
	virtual GDExtensionInt get_method_argument_count(const StringName &p_method, bool &r_valid) const = 0;
	// @todo Should godot-cpp have a Callable::CallError?
	virtual Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, GDExtensionCallError &r_error) = 0;
	virtual void notification(int p_notification, bool p_reversed) = 0;
	virtual String to_string(bool *r_valid) = 0;
	virtual void refcount_incremented() = 0;
	virtual bool refcount_decremented() = 0;
	virtual Ref<Script> get_script() const = 0;
	virtual bool is_placeholder() const = 0;
	virtual void property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) = 0;
	virtual Variant property_get_fallback(const StringName &p_name, bool *r_valid) = 0;
	virtual ScriptLanguage *_get_language() = 0;

	virtual ~ScriptInstanceExtension() {};
};

}; // namespace godot
