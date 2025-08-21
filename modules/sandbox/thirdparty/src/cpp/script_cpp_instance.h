#pragma once

#include <gdextension_interface.h>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/script_extension.hpp>
#include <godot_cpp/classes/script_language_extension.hpp>
#include <godot_cpp/core/type_info.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include "../godot/script_instance.h"
using namespace godot;

class CPPScript;
class ELFScript;
class ELFScriptInstance;

class CPPScriptInstance : public ScriptInstanceExtension {
	Object *owner;
	Ref<CPPScript> script;
	ELFScriptInstance *managed_esi = nullptr;
	ELFScriptInstance *elf_script_instance = nullptr;

	friend class CPPScript;

public:
	bool set(const StringName &p_name, const Variant &p_value) override;
	bool get(const StringName &p_name, Variant &r_ret) const override;
	const GDExtensionPropertyInfo *get_property_list(uint32_t *r_count) const override;
	void free_property_list(const GDExtensionPropertyInfo *p_list, uint32_t p_count) const override;
	Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const override;
	bool validate_property(GDExtensionPropertyInfo &p_property) const override;
	bool property_can_revert(const StringName &p_name) const override;
	bool property_get_revert(const StringName &p_name, Variant &r_ret) const override;
	Object *get_owner() override;
	void get_property_state(GDExtensionScriptInstancePropertyStateAdd p_add_func, void *p_userdata) override;
	const GDExtensionMethodInfo *get_method_list(uint32_t *r_count) const override;
	void free_method_list(const GDExtensionMethodInfo *p_list, uint32_t p_count) const override;
	bool has_method(const StringName &p_method) const override;
	GDExtensionInt get_method_argument_count(const StringName &p_method, bool &r_valid) const override;
	Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, GDExtensionCallError &r_error) override;
	void notification(int p_notification, bool p_reversed) override;
	String to_string(bool *r_valid) override;
	void refcount_incremented() override;
	bool refcount_decremented() override;
	Ref<Script> get_script() const override;
	bool is_placeholder() const override;
	void property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) override;
	Variant property_get_fallback(const StringName &p_name, bool *r_valid) override;
	ScriptLanguage *_get_language() override;

	ELFScriptInstance *get_script_instance() const { return elf_script_instance; }
	void set_script_instance(ELFScriptInstance *p_instance);
	void unset_script_instance();
	void manage_script_instance(ELFScript *p_script);

	CPPScriptInstance(Object *p_owner, const Ref<CPPScript> p_script);
	~CPPScriptInstance();
};
