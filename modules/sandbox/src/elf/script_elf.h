/**************************************************************************/
/*  script_elf.h                                                          */
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

#include "core/error/error_macros.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/object/script_language.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

class ELFScriptInstance;
class Sandbox;
class ScriptInstance;

class ELFScript : public Script {
	GDCLASS(ELFScript, Script);

protected:
	static void _bind_methods();
	PackedByteArray source_code;
	String global_name;
	String path;
	std::string std_path;
	int source_version = 0;
	int elf_api_version;
	String elf_programming_language;

	mutable HashSet<ELFScriptInstance *> instances;
	friend class ELFScriptInstance;

	static inline HashMap<String, HashSet<Sandbox *>> sandbox_map;

public:
	Array functions;
	PackedStringArray function_names;

	void set_public_api_functions(Array &&p_functions);
	void update_public_api_functions();
	String get_elf_programming_language() const;
	int get_elf_api_version() const noexcept { return elf_api_version; }
	int get_source_version() const noexcept { return source_version; }
	String get_dockerized_program_path() const;
	const String &get_path() const noexcept { return path; }
	const std::string &get_std_path() const noexcept { return std_path; }

	/// @brief Retrieve a Sandbox instance based on a given owner object.
	/// @param p_for_object The owner object.
	/// @return The Sandbox instance, or nullptr if not found.
	Sandbox *get_sandbox_for(Object *p_for_object) const;

	/// @brief Retrieve all Objects that share a Sandbox instance that uses this ELF resource.
	/// @return An array of Objects that share a Sandbox instance.
	Array get_sandbox_objects() const;

	/// @brief Retrieve the content of the ELF resource as a byte array.
	/// @return An ELF program as a byte array.
	const PackedByteArray &get_content();

	/// @brief Get an ELFScript instance using a Node as the owner.
	/// @param p_for_object The owner Node.
	/// @return A reference to the ELFScript instance.
	ELFScriptInstance *get_script_instance(Object *p_for_object) const;

	void register_instance(Sandbox *p_sandbox) { sandbox_map[path].insert(p_sandbox); }
	void unregister_instance(Sandbox *p_sandbox) { sandbox_map[path].erase(p_sandbox); }

	// Internal Script API methods (no underscore prefix)
	virtual bool can_instantiate() const override;
	virtual Ref<Script> get_base_script() const override;
	virtual StringName get_global_name() const override;
	virtual bool inherits_script(const Ref<Script> &p_script) const override;
	virtual StringName get_instance_base_type() const override;
	virtual ScriptInstance *instance_create(Object *p_for_object) override;
	virtual PlaceHolderScriptInstance *placeholder_instance_create(Object *p_for_object) override;
	virtual bool instance_has(const Object *p_object) const override;
	virtual bool has_source_code() const override;
	virtual String get_source_code() const override;
	virtual void set_source_code(const String &p_code) override;
	virtual Error reload(bool p_keep_state = false) override;
	virtual bool has_method(const StringName &p_method) const override;
	virtual bool has_static_method(const StringName &p_method) const override;
	virtual MethodInfo get_method_info(const StringName &p_method) const override;
	virtual bool is_tool() const override;
	virtual bool is_valid() const override;
	virtual bool is_abstract() const override;
	virtual ScriptLanguage *get_language() const override;
	virtual bool has_script_signal(const StringName &p_signal) const override;
	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const override;
	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const override;
	virtual void update_exports() override;
	virtual void get_script_method_list(List<MethodInfo> *p_list) const override;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const override;
	virtual int get_member_line(const StringName &p_member) const override;
	virtual void get_constants(HashMap<StringName, Variant> *p_constants) override;
	virtual void get_members(HashSet<StringName> *p_members) override;
	virtual const Variant get_rpc_config() const override;

#ifdef TOOLS_ENABLED
	virtual StringName get_doc_class_name() const override;
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder) override;
	virtual String get_class_icon_path() const override;
	virtual Vector<DocData::ClassDoc> get_documentation() const override;
	virtual bool is_placeholder_fallback_enabled() const override;
#endif

	void set_file(const String &path);

	ELFScript() {}
	~ELFScript() {}
};

// Function to get the ELF script language singleton
ScriptLanguage *get_elf_language_singleton();
