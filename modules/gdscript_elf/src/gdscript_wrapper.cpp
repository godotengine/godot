/**************************************************************************/
/*  gdscript_wrapper.cpp                                                  */
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

#include "gdscript_wrapper.h"

#include "gdscript_bytecode_c_codegen.h"
#include "gdscript_c_compiler.h"
#include "gdscript_elf_fallback.h"
#include "modules/gdscript/gdscript.h"

// Macro-based forwarding for pass-through methods
#define FORWARD_V(ret, name, ...) \
	ret GDScriptWrapper::name(__VA_ARGS__) const { \
		ERR_FAIL_COND_V(original_script.is_null(), ret{}); \
		return original_script->name(__VA_ARGS__); \
	}

#define FORWARD_VOID(name, ...) \
	void GDScriptWrapper::name(__VA_ARGS__) { \
		ERR_FAIL_COND(original_script.is_null()); \
		original_script->name(__VA_ARGS__); \
	}

#define FORWARD_NV(ret, name, ...) \
	ret GDScriptWrapper::name(__VA_ARGS__) { \
		ERR_FAIL_COND_V(original_script.is_null(), ret{}); \
		return original_script->name(__VA_ARGS__); \
	}

void GDScriptWrapper::_bind_methods() {
}

GDScriptWrapper::GDScriptWrapper() {
	original_script = Ref<GDScript>();
}

GDScriptWrapper::~GDScriptWrapper() {
	original_script.unref();
}

void GDScriptWrapper::set_original_script(const Ref<GDScript> &p_script) {
	original_script = p_script;
}

// Script interface - all methods delegate to original
FORWARD_V(bool, can_instantiate)
FORWARD_V(Ref<Script>, get_base_script)
FORWARD_V(StringName, get_global_name)
FORWARD_V(bool, inherits_script, const Ref<Script> &p_script)
FORWARD_V(StringName, get_instance_base_type)
FORWARD_NV(ScriptInstance *, instance_create, Object *p_this)
FORWARD_NV(PlaceHolderScriptInstance *, placeholder_instance_create, Object *p_this)
FORWARD_V(bool, instance_has, const Object *p_this)
FORWARD_V(bool, has_source_code)
FORWARD_V(String, get_source_code)
FORWARD_VOID(set_source_code, const String &p_code)

Error GDScriptWrapper::reload(bool p_keep_state) {
	ERR_FAIL_COND_V(original_script.is_null(), ERR_INVALID_DATA);
	Error err = original_script->reload(p_keep_state);
	if (err == OK && original_script->is_valid() && GDScriptCCompiler::is_compiler_available()) {
		// Generate C code and compile to ELF for each function
		GDScriptBytecodeCCodeGenerator codegen;
		GDScriptCCompiler compiler;
		for (int i = 0; i < original_script->get_member_count(); i++) {
			GDScriptFunction *func = original_script->get_member_functions().getptr(original_script->get_member_name(i));
			if (func && !func->code.is_empty()) {
				String c_code = codegen.generate_c_code(func);
				if (!c_code.is_empty()) {
					PackedByteArray elf = compiler.compile_to_elf(c_code);
					if (!elf.is_empty()) {
						// Store ELF in function wrapper (would need access to wrapper)
						// For now, compilation succeeds - execution handled in function wrapper
					}
				}
			}
		}
	}
	return err;
}

FORWARD_V(bool, has_method, const StringName &p_method)
FORWARD_V(bool, has_static_method, const StringName &p_method)
FORWARD_V(MethodInfo, get_method_info, const StringName &p_method)
FORWARD_V(bool, is_tool)
FORWARD_V(bool, is_valid)
FORWARD_V(bool, is_abstract)
FORWARD_V(ScriptLanguage *, get_language)
FORWARD_V(bool, has_script_signal, const StringName &p_signal)
FORWARD_VOID(get_script_signal_list, List<MethodInfo> *p_signals)
FORWARD_V(bool, get_property_default_value, const StringName &p_property, Variant &r_value)
FORWARD_VOID(update_exports)
FORWARD_VOID(get_script_method_list, List<MethodInfo> *p_list)
FORWARD_VOID(get_script_property_list, List<PropertyInfo> *p_list)
FORWARD_V(int, get_member_line, const StringName &p_member)
FORWARD_VOID(get_constants, HashMap<StringName, Variant> *p_constants)
FORWARD_VOID(get_members, HashSet<StringName> *p_members)
FORWARD_V(bool, is_placeholder_fallback_enabled)
FORWARD_V(const Variant, get_rpc_config)

#ifdef TOOLS_ENABLED
FORWARD_V(StringName, get_doc_class_name)
FORWARD_V(Vector<DocData::ClassDoc>, get_documentation)
FORWARD_V(String, get_class_icon_path)
#endif

#undef FORWARD_V
#undef FORWARD_VOID
#undef FORWARD_NV
