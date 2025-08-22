/**************************************************************************/
/*  script_cpp.h                                                          */
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

#include "core/doc_data.h"
#include "core/error/error_list.h"
#include "core/io/file_access.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/script_language.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

class CPPScriptInstance;
class ELFScriptInstance;

// Forward declare ELFScript properly
class ELFScript;

class CPPScript : public Script {
	GDCLASS(CPPScript, Script);

protected:
	static void _bind_methods() {}
	String source_code;

public:
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

	/// @brief Detects if the project is a CMake or SCons project.
	/// @return true if the project is a CMake or SCons project.
	static bool DetectCMakeOrSConsProject();

	void set_file(const String &p_path);
	CPPScriptInstance *get_cpp_script_instance() const;
	const String &get_path() const { return path; }
	bool detect_script_instance();
	void remove_instance(CPPScriptInstance *p_instance);

	static String PathToGlobalName(const String &p_path) {
		return "CPPScript_" + p_path.get_basename().replace("res://", "").replace("/", "_").replace("-", "_").capitalize().replace(" ", "");
	}

	CPPScript();
	~CPPScript();

private:
	String path;
	mutable HashSet<CPPScriptInstance *> instances;
	Ref<ELFScript> elf_script;
	friend class CPPScriptInstance;
};
