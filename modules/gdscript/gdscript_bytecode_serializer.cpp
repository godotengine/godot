/**************************************************************************/
/*  gdscript_bytecode_serializer.cpp                                      */
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

#include "gdscript_bytecode_serializer.h"

#include "gdscript_cache.h"
#include "gdscript_utility_functions.h"

#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "core/variant/variant.h"

namespace {

class GDScriptBytecodeScriptReference : public RefCounted {
	GDCLASS(GDScriptBytecodeScriptReference, RefCounted);

	String path;
	String fqcn;

protected:
	static void _bind_methods() {}

public:
	GDScriptBytecodeScriptReference() = default;
	GDScriptBytecodeScriptReference(const String &p_path, const String &p_fqcn) {
		path = p_path;
		fqcn = p_fqcn;
	}

	const String &get_path() const { return path; }
	const String &get_fqcn() const { return fqcn; }
};

static bool find_global_object_name(const Object *p_object, StringName &r_name) {
	if (p_object == nullptr) {
		return false;
	}

	GDScriptLanguage *language = GDScriptLanguage::get_singleton();
	if (language == nullptr || language->get_global_array() == nullptr) {
		return false;
	}

	for (const KeyValue<StringName, int> &E : language->get_global_map()) {
		const Variant &global = language->get_global_array()[E.value];
		if (global.get_type() != Variant::OBJECT) {
			continue;
		}

		Object *global_object = global;
		if (global_object == p_object && Object::cast_to<GDScriptNativeClass>(global_object) == nullptr) {
			r_name = E.key;
			return true;
		}
	}

	return false;
}

} // namespace

Ref<Script> GDScriptBytecodeSerializer::resolve_script_reference(GDScript *p_script, const String &p_path, const String &p_fqcn) {
	if (p_path.is_empty()) {
		return Ref<Script>();
	}

	GDScript *root_script = p_script->get_root_script();
	if (GDScript::is_canonically_equal_paths(root_script->get_path(), p_path)) {
		if (p_fqcn.is_empty() || root_script->get_fully_qualified_name() == p_fqcn) {
			return Ref<GDScript>(root_script);
		}

		if (GDScript *resolved = root_script->find_class(p_fqcn)) {
			return Ref<GDScript>(resolved);
		}
	}

	Ref<GDScript> cached_script = GDScriptCache::get_cached_script(p_path);
	if (cached_script.is_valid() && cached_script->is_valid()) {
		if (p_fqcn.is_empty()) {
			return cached_script;
		}

		if (GDScript *resolved = cached_script->find_class(p_fqcn)) {
			return Ref<GDScript>(resolved);
		}
	}

	Error err = OK;
	Ref<Resource> loaded_resource = ResourceLoader::load(p_path, "GDScript", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
	ERR_FAIL_COND_V_MSG(err != OK || loaded_resource.is_null(), Ref<Script>(), vformat("Failed to resolve bytecode script reference '%s'.", p_path));

	Ref<Script> loaded_script = loaded_resource;
	ERR_FAIL_COND_V_MSG(loaded_script.is_null(), Ref<Script>(), vformat("Resolved bytecode script reference '%s' is not a Script resource.", p_path));

	if (p_fqcn.is_empty()) {
		return loaded_script;
	}

	Ref<GDScript> loaded_gdscript = loaded_script;
	ERR_FAIL_COND_V_MSG(loaded_gdscript.is_null(), Ref<Script>(), vformat("Cannot resolve inner class '%s' in non-GDScript resource '%s'.", p_fqcn, p_path));

	GDScript *resolved = loaded_gdscript->find_class(p_fqcn);
	ERR_FAIL_NULL_V_MSG(resolved, Ref<Script>(), vformat("Failed to resolve bytecode script reference '%s' in '%s'.", p_fqcn, p_path));
	return Ref<GDScript>(resolved);
}

void GDScriptBytecodeSerializer::resolve_variant_script_references(GDScript *p_script, Variant &r_variant) {
	switch (r_variant.get_type()) {
		case Variant::ARRAY: {
			Array array = r_variant;
			for (int i = 0; i < array.size(); i++) {
				Variant value = array[i];
				resolve_variant_script_references(p_script, value);
				array[i] = value;
			}
			r_variant = array;
		} break;
		case Variant::DICTIONARY: {
			Dictionary dictionary = r_variant;
			Array keys = dictionary.keys();
			for (int i = 0; i < keys.size(); i++) {
				Variant value = dictionary[keys[i]];
				resolve_variant_script_references(p_script, value);
				dictionary[keys[i]] = value;
			}
			r_variant = dictionary;
		} break;
		case Variant::OBJECT: {
			Object *object = r_variant;
			GDScriptBytecodeScriptReference *reference = Object::cast_to<GDScriptBytecodeScriptReference>(object);
			if (reference != nullptr) {
				r_variant = resolve_script_reference(p_script, reference->get_path(), reference->get_fqcn());
			}
		} break;
		default:
			break;
	}
}

void GDScriptBytecodeSerializer::resolve_data_type_script_references(GDScript *p_script, GDScriptDataType &r_data_type) {
	for (int i = 0; i < r_data_type.container_element_types.size(); i++) {
		resolve_data_type_script_references(p_script, r_data_type.container_element_types.write[i]);
	}

	if ((r_data_type.kind == GDScriptDataType::SCRIPT || r_data_type.kind == GDScriptDataType::GDSCRIPT) && !r_data_type.serialized_script_path.is_empty()) {
		Ref<Script> script = resolve_script_reference(p_script, r_data_type.serialized_script_path, r_data_type.serialized_script_fqcn);
		r_data_type.script_type_ref = script;
		r_data_type.script_type = script.ptr();
	}

	r_data_type.serialized_script_path = String();
	r_data_type.serialized_script_fqcn = String();
}

void GDScriptBytecodeSerializer::resolve_method_info_script_references(GDScript *p_script, MethodInfo &r_method_info) {
	for (int i = 0; i < r_method_info.default_arguments.size(); i++) {
		resolve_variant_script_references(p_script, r_method_info.default_arguments.write[i]);
	}
}

void GDScriptBytecodeSerializer::resolve_function_script_references(GDScript *p_script, GDScriptFunction *p_function) {
	for (int i = 0; i < p_function->argument_types.size(); i++) {
		resolve_data_type_script_references(p_script, p_function->argument_types.write[i]);
	}
	resolve_data_type_script_references(p_script, p_function->return_type);

	resolve_method_info_script_references(p_script, p_function->method_info);
	resolve_variant_script_references(p_script, p_function->rpc_config);

	for (int i = 0; i < p_function->constants.size(); i++) {
		resolve_variant_script_references(p_script, p_function->constants.write[i]);
	}

	for (int i = 0; i < p_function->lambdas.size(); i++) {
		resolve_function_script_references(p_script, p_function->lambdas[i]);
	}
}

void GDScriptBytecodeSerializer::resolve_script_references(GDScript *p_script) {
	for (KeyValue<StringName, GDScript::MemberInfo> &E : p_script->member_indices) {
		resolve_data_type_script_references(p_script, E.value.data_type);
	}

	for (KeyValue<StringName, GDScript::MemberInfo> &E : p_script->static_variables_indices) {
		resolve_data_type_script_references(p_script, E.value.data_type);
	}

	for (KeyValue<StringName, Variant> &E : p_script->constants) {
		resolve_variant_script_references(p_script, E.value);
	}

	for (KeyValue<StringName, MethodInfo> &E : p_script->_signals) {
		resolve_method_info_script_references(p_script, E.value);
	}

	Variant rpc_config = p_script->rpc_config;
	resolve_variant_script_references(p_script, rpc_config);
	p_script->rpc_config = rpc_config;

	for (const KeyValue<StringName, GDScriptFunction *> &E : p_script->member_functions) {
		resolve_function_script_references(p_script, E.value);
	}

	if (p_script->implicit_initializer != nullptr) {
		resolve_function_script_references(p_script, p_script->implicit_initializer);
	}
	if (p_script->implicit_ready != nullptr) {
		resolve_function_script_references(p_script, p_script->implicit_ready);
	}
	if (p_script->static_initializer != nullptr) {
		resolve_function_script_references(p_script, p_script->static_initializer);
	}

	for (const KeyValue<StringName, Ref<GDScript>> &E : p_script->subclasses) {
		resolve_script_references(E.value.ptr());
	}
}

// =============================================================================
// Binary encoding helpers
// =============================================================================

void GDScriptBytecodeSerializer::write_uint8(Vector<uint8_t> &buf, uint8_t val) {
	buf.push_back(val);
}

void GDScriptBytecodeSerializer::write_uint32(Vector<uint8_t> &buf, uint32_t val) {
	int ofs = buf.size();
	buf.resize(ofs + 4);
	encode_uint32(val, &buf.write[ofs]);
}

void GDScriptBytecodeSerializer::write_int32(Vector<uint8_t> &buf, int32_t val) {
	write_uint32(buf, (uint32_t)val);
}

void GDScriptBytecodeSerializer::write_string(Vector<uint8_t> &buf, const String &s) {
	CharString cs = s.utf8();
	write_uint32(buf, cs.length());
	if (cs.length() > 0) {
		int ofs = buf.size();
		buf.resize(ofs + cs.length());
		memcpy(&buf.write[ofs], cs.get_data(), cs.length());
	}
}

void GDScriptBytecodeSerializer::write_string_name(Vector<uint8_t> &buf, const StringName &s) {
	write_string(buf, String(s));
}

// Variant tag for special object types that can't go through encode_variant.
enum VariantObjectTag : uint8_t {
	VAR_TAG_REGULAR = 0, // Normal variant, use encode_variant/decode_variant.
	VAR_TAG_NATIVE_CLASS = 1, // GDScriptNativeClass ? stored as class name.
	VAR_TAG_GDSCRIPT = 2, // GDScript reference ? stored as script path.
	VAR_TAG_NULL_OBJECT = 3, // Object that can't be serialized ? stored as null.
};

static constexpr uint8_t VAR_TAG_RESOURCE = 4;
static constexpr uint8_t VAR_TAG_GLOBAL_OBJECT = 5;

void GDScriptBytecodeSerializer::write_variant(Vector<uint8_t> &buf, const Variant &v) {
	// Objects need special handling since encode_variant can't serialize GDScript types.
	if (v.get_type() == Variant::OBJECT) {
		Object *obj = v;
		if (!obj) {
			write_uint8(buf, VAR_TAG_NULL_OBJECT);
			return;
		}

		GDScriptNativeClass *nc = Object::cast_to<GDScriptNativeClass>(obj);
		if (nc) {
			write_uint8(buf, VAR_TAG_NATIVE_CLASS);
			write_string_name(buf, nc->get_name());
			return;
		}

		GDScript *gs = Object::cast_to<GDScript>(obj);
		if (gs) {
			write_uint8(buf, VAR_TAG_GDSCRIPT);
			write_string(buf, gs->get_path());
			write_string(buf, gs->get_fully_qualified_name());
			return;
		}

		StringName global_name;
		if (find_global_object_name(obj, global_name)) {
			write_uint8(buf, VAR_TAG_GLOBAL_OBJECT);
			write_string_name(buf, global_name);
			return;
		}
		Resource *resource = Object::cast_to<Resource>(obj);
		if (resource != nullptr && !resource->get_path().is_empty()) {
			write_uint8(buf, VAR_TAG_RESOURCE);
			write_string(buf, resource->get_path());
			return;
		}

		// Unknown object type ? serialize as null.
		write_uint8(buf, VAR_TAG_NULL_OBJECT);
		return;
	}

	write_uint8(buf, VAR_TAG_REGULAR);
	int len = 0;
	encode_variant(v, nullptr, len, false);
	int ofs = buf.size();
	write_uint32(buf, len);
	buf.resize(ofs + 4 + len);
	encode_variant(v, &buf.write[ofs + 4], len, false);
}

void GDScriptBytecodeSerializer::write_data_type(Vector<uint8_t> &buf, const GDScriptDataType &dt) {
	write_uint8(buf, (uint8_t)dt.kind);
	write_uint32(buf, (uint32_t)dt.builtin_type);
	write_string_name(buf, dt.native_type);

	String script_path;
	String script_fqcn;
	if (dt.kind == GDScriptDataType::SCRIPT || dt.kind == GDScriptDataType::GDSCRIPT) {
		const Script *script = dt.script_type_ref.is_valid() ? dt.script_type_ref.ptr() : dt.script_type;
		if (script != nullptr) {
			script_path = script->get_path();
		}
		if (const GDScript *gdscript = Object::cast_to<GDScript>(const_cast<Script *>(script))) {
			script_fqcn = gdscript->get_fully_qualified_name();
		}
	}
	write_string(buf, script_path);
	write_string(buf, script_fqcn);

	// Container element types.
	write_uint32(buf, dt.container_element_types.size());
	for (int i = 0; i < dt.container_element_types.size(); i++) {
		write_data_type(buf, dt.container_element_types[i]);
	}
}

void GDScriptBytecodeSerializer::write_property_info(Vector<uint8_t> &buf, const PropertyInfo &pi) {
	write_string_name(buf, pi.name);
	write_uint32(buf, (uint32_t)pi.type);
	write_string_name(buf, pi.class_name);
	write_uint32(buf, (uint32_t)pi.hint);
	write_string(buf, pi.hint_string);
	write_uint32(buf, pi.usage);
}

void GDScriptBytecodeSerializer::write_method_info(Vector<uint8_t> &buf, const MethodInfo &mi) {
	write_string_name(buf, mi.name);
	write_property_info(buf, mi.return_val);
	write_uint32(buf, mi.flags);
	write_int32(buf, mi.id);
	write_int32(buf, mi.return_val_metadata);

	write_uint32(buf, mi.arguments.size());
	for (const PropertyInfo &arg : mi.arguments) {
		write_property_info(buf, arg);
	}

	write_uint32(buf, mi.default_arguments.size());
	for (const Variant &def : mi.default_arguments) {
		write_variant(buf, def);
	}

	write_uint32(buf, mi.arguments_metadata.size());
	for (int i = 0; i < mi.arguments_metadata.size(); i++) {
		write_int32(buf, mi.arguments_metadata[i]);
	}
}

void GDScriptBytecodeSerializer::write_function(Vector<uint8_t> &buf, const GDScriptFunction *fn) {
	// Metadata.
	write_string_name(buf, fn->name);
	write_string_name(buf, fn->source);
	write_uint8(buf, fn->_static ? 1 : 0);
	write_int32(buf, fn->_argument_count);
	write_int32(buf, fn->_vararg_index);
	write_int32(buf, fn->_stack_size);
	write_int32(buf, fn->_instruction_args_size);
	write_int32(buf, fn->_initial_line);

	// Argument types.
	write_uint32(buf, fn->argument_types.size());
	for (int i = 0; i < fn->argument_types.size(); i++) {
		write_data_type(buf, fn->argument_types[i]);
	}

	// Return type.
	write_data_type(buf, fn->return_type);

	// Method info.
	write_method_info(buf, fn->method_info);

	// RPC config.
	write_variant(buf, fn->rpc_config);

	// Default arguments.
	write_uint32(buf, fn->default_arguments.size());
	for (int i = 0; i < fn->default_arguments.size(); i++) {
		write_int32(buf, fn->default_arguments[i]);
	}

	// Code section (raw opcodes).
	write_uint32(buf, fn->code.size());
	for (int i = 0; i < fn->code.size(); i++) {
		write_int32(buf, fn->code[i]);
	}

	// Constants section.
	write_uint32(buf, fn->constants.size());
	for (int i = 0; i < fn->constants.size(); i++) {
		write_variant(buf, fn->constants[i]);
	}

	// Global names section.
	write_uint32(buf, fn->global_names.size());
	for (int i = 0; i < fn->global_names.size(); i++) {
		write_string_name(buf, fn->global_names[i]);
	}

	// Temporary slots.
	write_uint32(buf, fn->temporary_slots.size());
	for (const Pair<int, Variant::Type> &E : fn->temporary_slots) {
		write_int32(buf, E.first);
		write_uint32(buf, (uint32_t)E.second);
	}

	// Import table.
	const GDScriptFunction::ImportTable &it = fn->import_table;

	// Methods (MethodBind refs).
	write_uint32(buf, it.methods.size());
	for (int i = 0; i < it.methods.size(); i++) {
		write_string_name(buf, it.methods[i].class_name);
		write_string_name(buf, it.methods[i].method_name);
	}

	// Operators.
	write_uint32(buf, it.operators.size());
	for (int i = 0; i < it.operators.size(); i++) {
		write_uint32(buf, (uint32_t)it.operators[i].op);
		write_uint32(buf, (uint32_t)it.operators[i].left_type);
		write_uint32(buf, (uint32_t)it.operators[i].right_type);
	}

	// Setters.
	write_uint32(buf, it.setters.size());
	for (int i = 0; i < it.setters.size(); i++) {
		write_uint32(buf, (uint32_t)it.setters[i].type);
		write_string_name(buf, it.setters[i].member);
	}

	// Getters.
	write_uint32(buf, it.getters.size());
	for (int i = 0; i < it.getters.size(); i++) {
		write_uint32(buf, (uint32_t)it.getters[i].type);
		write_string_name(buf, it.getters[i].member);
	}

	// Keyed setters.
	write_uint32(buf, it.keyed_setters.size());
	for (int i = 0; i < it.keyed_setters.size(); i++) {
		write_uint32(buf, (uint32_t)it.keyed_setters[i].type);
	}

	// Keyed getters.
	write_uint32(buf, it.keyed_getters.size());
	for (int i = 0; i < it.keyed_getters.size(); i++) {
		write_uint32(buf, (uint32_t)it.keyed_getters[i].type);
	}

	// Indexed setters.
	write_uint32(buf, it.indexed_setters.size());
	for (int i = 0; i < it.indexed_setters.size(); i++) {
		write_uint32(buf, (uint32_t)it.indexed_setters[i].type);
	}

	// Indexed getters.
	write_uint32(buf, it.indexed_getters.size());
	for (int i = 0; i < it.indexed_getters.size(); i++) {
		write_uint32(buf, (uint32_t)it.indexed_getters[i].type);
	}

	// Builtin methods.
	write_uint32(buf, it.builtin_methods.size());
	for (int i = 0; i < it.builtin_methods.size(); i++) {
		write_uint32(buf, (uint32_t)it.builtin_methods[i].type);
		write_string_name(buf, it.builtin_methods[i].method);
	}

	// Constructors.
	write_uint32(buf, it.constructors.size());
	for (int i = 0; i < it.constructors.size(); i++) {
		write_uint32(buf, (uint32_t)it.constructors[i].type);
		write_int32(buf, it.constructors[i].constructor_idx);
	}

	// Utilities.
	write_uint32(buf, it.utilities.size());
	for (int i = 0; i < it.utilities.size(); i++) {
		write_string_name(buf, it.utilities[i].name);
	}

	// GDS utilities.
	write_uint32(buf, it.gds_utilities.size());
	for (int i = 0; i < it.gds_utilities.size(); i++) {
		write_string_name(buf, it.gds_utilities[i].name);
	}

	// Lambda section.
	write_uint32(buf, fn->lambdas.size());
	for (int i = 0; i < fn->lambdas.size(); i++) {
		write_function(buf, fn->lambdas[i]);
	}
}

void GDScriptBytecodeSerializer::write_script_data(Vector<uint8_t> &buf, const GDScript *p_script) {
	// Script metadata.
	write_uint8(buf, p_script->tool ? 1 : 0);
	write_uint8(buf, p_script->_is_abstract ? 1 : 0);
	write_string(buf, p_script->fully_qualified_name);
	write_string_name(buf, p_script->local_name);
	write_string_name(buf, p_script->global_name);
	write_string(buf, p_script->simplified_icon_path);

	// Native base class.
	StringName native_name;
	if (p_script->native.is_valid()) {
		native_name = p_script->native->get_name();
	}
	write_string_name(buf, native_name);

	// Base GDScript path.
	String base_path;
	if (p_script->base.is_valid()) {
		base_path = p_script->base->get_path();
	}
	write_string(buf, base_path);

	// Members (the HashSet<StringName>).
	write_uint32(buf, p_script->members.size());
	for (const StringName &member_name : p_script->members) {
		write_string_name(buf, member_name);
	}

	// Member indices.
	write_uint32(buf, p_script->member_indices.size());
	for (const KeyValue<StringName, GDScript::MemberInfo> &E : p_script->member_indices) {
		write_string_name(buf, E.key);
		write_int32(buf, E.value.index);
		write_string_name(buf, E.value.setter);
		write_string_name(buf, E.value.getter);
		write_data_type(buf, E.value.data_type);
		write_property_info(buf, E.value.property_info);
	}

	// Static variable indices.
	write_uint32(buf, p_script->static_variables_indices.size());
	for (const KeyValue<StringName, GDScript::MemberInfo> &E : p_script->static_variables_indices) {
		write_string_name(buf, E.key);
		write_int32(buf, E.value.index);
		write_string_name(buf, E.value.setter);
		write_string_name(buf, E.value.getter);
		write_data_type(buf, E.value.data_type);
		write_property_info(buf, E.value.property_info);
	}

	// Static variables count (for resizing the vector; actual values get initialized at runtime).
	write_uint32(buf, p_script->static_variables.size());

	// Constants.
	write_uint32(buf, p_script->constants.size());
	for (const KeyValue<StringName, Variant> &E : p_script->constants) {
		// Skip subclass constants (they'll be handled as subclasses).
		write_string_name(buf, E.key);
		write_variant(buf, E.value);
	}

	// Signals.
	write_uint32(buf, p_script->_signals.size());
	for (const KeyValue<StringName, MethodInfo> &E : p_script->_signals) {
		write_string_name(buf, E.key);
		write_method_info(buf, E.value);
	}

	// RPC config.
	write_variant(buf, p_script->rpc_config);

	// Member functions.
	write_uint32(buf, p_script->member_functions.size());
	for (const KeyValue<StringName, GDScriptFunction *> &E : p_script->member_functions) {
		write_string_name(buf, E.key);
		write_function(buf, E.value);
	}

	// Special functions that are not part of member_functions.
	write_uint8(buf, p_script->implicit_initializer != nullptr ? 1 : 0);
	if (p_script->implicit_initializer != nullptr) {
		write_function(buf, p_script->implicit_initializer);
	}

	write_uint8(buf, p_script->implicit_ready != nullptr ? 1 : 0);
	if (p_script->implicit_ready != nullptr) {
		write_function(buf, p_script->implicit_ready);
	}

	write_uint8(buf, p_script->static_initializer != nullptr ? 1 : 0);
	if (p_script->static_initializer != nullptr) {
		write_function(buf, p_script->static_initializer);
	}

	// Lambda info.
	write_uint32(buf, p_script->lambda_info.size());
	for (const KeyValue<GDScriptFunction *, GDScript::LambdaInfo> &E : p_script->lambda_info) {
		// Store the lambda function name to identify it later.
		write_string_name(buf, E.key->get_name());
		write_int32(buf, E.value.capture_count);
		write_uint8(buf, E.value.use_self ? 1 : 0);
	}

	// Special function name for the user-defined constructor living in member_functions.
	write_string_name(buf, p_script->initializer ? p_script->initializer->get_name() : StringName());

	// Subclasses (recursive).
	write_uint32(buf, p_script->subclasses.size());
	for (const KeyValue<StringName, Ref<GDScript>> &E : p_script->subclasses) {
		write_string_name(buf, E.key);
		write_script_data(buf, E.value.ptr());
	}
}

// =============================================================================
// Binary decoding helpers
// =============================================================================

uint8_t GDScriptBytecodeSerializer::read_uint8(const uint8_t *p_buf, int &ofs, int p_len) {
	ERR_FAIL_COND_V(ofs + 1 > p_len, 0);
	uint8_t val = p_buf[ofs];
	ofs += 1;
	return val;
}

uint32_t GDScriptBytecodeSerializer::read_uint32(const uint8_t *p_buf, int &ofs, int p_len) {
	ERR_FAIL_COND_V(ofs + 4 > p_len, 0);
	uint32_t val = decode_uint32(&p_buf[ofs]);
	ofs += 4;
	return val;
}

int32_t GDScriptBytecodeSerializer::read_int32(const uint8_t *p_buf, int &ofs, int p_len) {
	return (int32_t)read_uint32(p_buf, ofs, p_len);
}

String GDScriptBytecodeSerializer::read_string(const uint8_t *p_buf, int &ofs, int p_len) {
	uint32_t str_len = read_uint32(p_buf, ofs, p_len);
	if (str_len == 0) {
		return String();
	}
	ERR_FAIL_COND_V(ofs + (int)str_len > p_len, String());
	String result = String::utf8((const char *)&p_buf[ofs], str_len);
	ofs += str_len;
	return result;
}

StringName GDScriptBytecodeSerializer::read_string_name(const uint8_t *p_buf, int &ofs, int p_len) {
	return StringName(read_string(p_buf, ofs, p_len));
}

Variant GDScriptBytecodeSerializer::read_variant(const uint8_t *p_buf, int &ofs, int p_len) {
	uint8_t tag = read_uint8(p_buf, ofs, p_len);

	switch (tag) {
		case VAR_TAG_NATIVE_CLASS: {
			StringName class_name = read_string_name(p_buf, ofs, p_len);
			Ref<GDScriptNativeClass> nc = memnew(GDScriptNativeClass(class_name));
			return nc;
		}
		case VAR_TAG_GDSCRIPT: {
			String script_path = read_string(p_buf, ofs, p_len);
			String script_fqcn = read_string(p_buf, ofs, p_len);
			if (!script_path.is_empty()) {
				Ref<GDScriptBytecodeScriptReference> reference = memnew(GDScriptBytecodeScriptReference(script_path, script_fqcn));
				return reference;
			}
			return Variant();
		}
		case VAR_TAG_GLOBAL_OBJECT: {
			StringName global_name = read_string_name(p_buf, ofs, p_len);
			GDScriptLanguage *language = GDScriptLanguage::get_singleton();
			if (language != nullptr && language->get_global_map().has(global_name)) {
				int global_idx = language->get_global_map()[global_name];
				return language->get_global_array()[global_idx];
			}
			return Variant(static_cast<Object *>(nullptr));
		}
		case VAR_TAG_RESOURCE: {
			String resource_path = read_string(p_buf, ofs, p_len);
			if (!resource_path.is_empty()) {
				Error err = OK;
				Ref<Resource> resource = ResourceLoader::load(resource_path, "", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
				if (err == OK && resource.is_valid()) {
					return resource;
				}
			}
			return Variant(static_cast<Object *>(nullptr));
		}
		case VAR_TAG_NULL_OBJECT: {
			return Variant(static_cast<Object *>(nullptr));
		}
		case VAR_TAG_REGULAR:
		default: {
			uint32_t var_len = read_uint32(p_buf, ofs, p_len);
			ERR_FAIL_COND_V(ofs + (int)var_len > p_len, Variant());
			Variant v;
			int bytes_read = 0;
			decode_variant(v, &p_buf[ofs], var_len, &bytes_read, false);
			ofs += var_len;
			return v;
		}
	}
}

GDScriptDataType GDScriptBytecodeSerializer::read_data_type(const uint8_t *p_buf, int &ofs, int p_len) {
	GDScriptDataType dt;
	dt.kind = (GDScriptDataType::Kind)read_uint8(p_buf, ofs, p_len);
	dt.builtin_type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
	dt.native_type = read_string_name(p_buf, ofs, p_len);

	dt.serialized_script_path = read_string(p_buf, ofs, p_len);
	dt.serialized_script_fqcn = read_string(p_buf, ofs, p_len);

	uint32_t container_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < container_count; i++) {
		dt.container_element_types.push_back(read_data_type(p_buf, ofs, p_len));
	}

	return dt;
}

PropertyInfo GDScriptBytecodeSerializer::read_property_info(const uint8_t *p_buf, int &ofs, int p_len) {
	PropertyInfo pi;
	pi.name = read_string_name(p_buf, ofs, p_len);
	pi.type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
	pi.class_name = read_string_name(p_buf, ofs, p_len);
	pi.hint = (PropertyHint)read_uint32(p_buf, ofs, p_len);
	pi.hint_string = read_string(p_buf, ofs, p_len);
	pi.usage = read_uint32(p_buf, ofs, p_len);
	return pi;
}

MethodInfo GDScriptBytecodeSerializer::read_method_info(const uint8_t *p_buf, int &ofs, int p_len) {
	MethodInfo mi;
	mi.name = read_string_name(p_buf, ofs, p_len);
	mi.return_val = read_property_info(p_buf, ofs, p_len);
	mi.flags = read_uint32(p_buf, ofs, p_len);
	mi.id = read_int32(p_buf, ofs, p_len);
	mi.return_val_metadata = read_int32(p_buf, ofs, p_len);

	uint32_t arg_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < arg_count; i++) {
		mi.arguments.push_back(read_property_info(p_buf, ofs, p_len));
	}

	uint32_t def_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < def_count; i++) {
		mi.default_arguments.push_back(read_variant(p_buf, ofs, p_len));
	}

	uint32_t meta_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < meta_count; i++) {
		mi.arguments_metadata.push_back(read_int32(p_buf, ofs, p_len));
	}

	return mi;
}

GDScriptFunction *GDScriptBytecodeSerializer::read_function(const uint8_t *p_buf, int &ofs, int p_len, GDScript *p_script) {
	GDScriptFunction *fn = memnew(GDScriptFunction);
	fn->_script = p_script;

	// Metadata.
	fn->name = read_string_name(p_buf, ofs, p_len);
	fn->source = read_string_name(p_buf, ofs, p_len);
	fn->_static = read_uint8(p_buf, ofs, p_len) != 0;
	fn->_argument_count = read_int32(p_buf, ofs, p_len);
	fn->_vararg_index = read_int32(p_buf, ofs, p_len);
	fn->_stack_size = read_int32(p_buf, ofs, p_len);
	fn->_instruction_args_size = read_int32(p_buf, ofs, p_len);
	fn->_initial_line = read_int32(p_buf, ofs, p_len);

	// Argument types.
	uint32_t arg_type_count = read_uint32(p_buf, ofs, p_len);
	fn->argument_types.resize(arg_type_count);
	for (uint32_t i = 0; i < arg_type_count; i++) {
		fn->argument_types.write[i] = read_data_type(p_buf, ofs, p_len);
	}

	// Return type.
	fn->return_type = read_data_type(p_buf, ofs, p_len);

	// Method info.
	fn->method_info = read_method_info(p_buf, ofs, p_len);

	// RPC config.
	fn->rpc_config = read_variant(p_buf, ofs, p_len);

	// Default arguments.
	uint32_t def_arg_count = read_uint32(p_buf, ofs, p_len);
	fn->default_arguments.resize(def_arg_count);
	for (uint32_t i = 0; i < def_arg_count; i++) {
		fn->default_arguments.write[i] = read_int32(p_buf, ofs, p_len);
	}
	if (fn->default_arguments.size()) {
		fn->_default_arg_count = fn->default_arguments.size() - 1;
		fn->_default_arg_ptr = &fn->default_arguments[0];
	}

	// Code section.
	uint32_t code_size = read_uint32(p_buf, ofs, p_len);
	fn->code.resize(code_size);
	for (uint32_t i = 0; i < code_size; i++) {
		fn->code.write[i] = read_int32(p_buf, ofs, p_len);
	}
	if (fn->code.size()) {
		fn->_code_ptr = &fn->code.write[0];
		fn->_code_size = fn->code.size();
	}

	// Constants section.
	uint32_t const_count = read_uint32(p_buf, ofs, p_len);
	fn->constants.resize(const_count);
	fn->_constant_count = const_count;
	for (uint32_t i = 0; i < const_count; i++) {
		fn->constants.write[i] = read_variant(p_buf, ofs, p_len);
	}
	fn->_constants_ptr = const_count > 0 ? fn->constants.ptrw() : nullptr;

	// Global names section.
	uint32_t names_count = read_uint32(p_buf, ofs, p_len);
	fn->global_names.resize(names_count);
	fn->_global_names_count = names_count;
	for (uint32_t i = 0; i < names_count; i++) {
		fn->global_names.write[i] = read_string_name(p_buf, ofs, p_len);
	}
	fn->_global_names_ptr = names_count > 0 ? &fn->global_names[0] : nullptr;

	// Temporary slots.
	uint32_t temp_count = read_uint32(p_buf, ofs, p_len);
	fn->temporary_slots.resize(temp_count);
	for (uint32_t i = 0; i < temp_count; i++) {
		int key = read_int32(p_buf, ofs, p_len);
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		fn->temporary_slots[i] = Pair(key, type);
	}

	// Import table resolution.
	// Methods (MethodBind*).
	uint32_t method_count = read_uint32(p_buf, ofs, p_len);
	fn->methods.resize(method_count);
	fn->_methods_count = method_count;
	for (uint32_t i = 0; i < method_count; i++) {
		StringName class_name = read_string_name(p_buf, ofs, p_len);
		StringName method_name = read_string_name(p_buf, ofs, p_len);
		MethodBind *mb = ClassDB::get_method(class_name, method_name);
		ERR_FAIL_NULL_V_MSG(mb, fn, vformat("Failed to resolve method %s::%s", class_name, method_name));
		fn->methods.write[i] = mb;
	}
	fn->_methods_ptr = method_count > 0 ? fn->methods.ptrw() : nullptr;

	// Operators.
	uint32_t op_count = read_uint32(p_buf, ofs, p_len);
	fn->operator_funcs.resize(op_count);
	fn->_operator_funcs_count = op_count;
	for (uint32_t i = 0; i < op_count; i++) {
		Variant::Operator op = (Variant::Operator)read_uint32(p_buf, ofs, p_len);
		Variant::Type left = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		Variant::Type right = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		fn->operator_funcs.write[i] = Variant::get_validated_operator_evaluator(op, left, right);
	}
	fn->_operator_funcs_ptr = op_count > 0 ? fn->operator_funcs.ptr() : nullptr;

	// Setters.
	uint32_t setter_count = read_uint32(p_buf, ofs, p_len);
	fn->setters.resize(setter_count);
	fn->_setters_count = setter_count;
	for (uint32_t i = 0; i < setter_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		StringName member = read_string_name(p_buf, ofs, p_len);
		fn->setters.write[i] = Variant::get_member_validated_setter(type, member);
	}
	fn->_setters_ptr = setter_count > 0 ? fn->setters.ptr() : nullptr;

	// Getters.
	uint32_t getter_count = read_uint32(p_buf, ofs, p_len);
	fn->getters.resize(getter_count);
	fn->_getters_count = getter_count;
	for (uint32_t i = 0; i < getter_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		StringName member = read_string_name(p_buf, ofs, p_len);
		fn->getters.write[i] = Variant::get_member_validated_getter(type, member);
	}
	fn->_getters_ptr = getter_count > 0 ? fn->getters.ptr() : nullptr;

	// Keyed setters.
	uint32_t ks_count = read_uint32(p_buf, ofs, p_len);
	fn->keyed_setters.resize(ks_count);
	fn->_keyed_setters_count = ks_count;
	for (uint32_t i = 0; i < ks_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		fn->keyed_setters.write[i] = Variant::get_member_validated_keyed_setter(type);
	}
	fn->_keyed_setters_ptr = ks_count > 0 ? fn->keyed_setters.ptr() : nullptr;

	// Keyed getters.
	uint32_t kg_count = read_uint32(p_buf, ofs, p_len);
	fn->keyed_getters.resize(kg_count);
	fn->_keyed_getters_count = kg_count;
	for (uint32_t i = 0; i < kg_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		fn->keyed_getters.write[i] = Variant::get_member_validated_keyed_getter(type);
	}
	fn->_keyed_getters_ptr = kg_count > 0 ? fn->keyed_getters.ptr() : nullptr;

	// Indexed setters.
	uint32_t is_count = read_uint32(p_buf, ofs, p_len);
	fn->indexed_setters.resize(is_count);
	fn->_indexed_setters_count = is_count;
	for (uint32_t i = 0; i < is_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		fn->indexed_setters.write[i] = Variant::get_member_validated_indexed_setter(type);
	}
	fn->_indexed_setters_ptr = is_count > 0 ? fn->indexed_setters.ptr() : nullptr;

	// Indexed getters.
	uint32_t ig_count = read_uint32(p_buf, ofs, p_len);
	fn->indexed_getters.resize(ig_count);
	fn->_indexed_getters_count = ig_count;
	for (uint32_t i = 0; i < ig_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		fn->indexed_getters.write[i] = Variant::get_member_validated_indexed_getter(type);
	}
	fn->_indexed_getters_ptr = ig_count > 0 ? fn->indexed_getters.ptr() : nullptr;

	// Builtin methods.
	uint32_t bm_count = read_uint32(p_buf, ofs, p_len);
	fn->builtin_methods.resize(bm_count);
	fn->_builtin_methods_count = bm_count;
	for (uint32_t i = 0; i < bm_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		StringName method = read_string_name(p_buf, ofs, p_len);
		fn->builtin_methods.write[i] = Variant::get_validated_builtin_method(type, method);
	}
	fn->_builtin_methods_ptr = bm_count > 0 ? fn->builtin_methods.ptr() : nullptr;

	// Constructors.
	uint32_t ctor_count = read_uint32(p_buf, ofs, p_len);
	fn->constructors.resize(ctor_count);
	fn->_constructors_count = ctor_count;
	for (uint32_t i = 0; i < ctor_count; i++) {
		Variant::Type type = (Variant::Type)read_uint32(p_buf, ofs, p_len);
		int idx = read_int32(p_buf, ofs, p_len);
		fn->constructors.write[i] = Variant::get_validated_constructor(type, idx);
	}
	fn->_constructors_ptr = ctor_count > 0 ? fn->constructors.ptr() : nullptr;

	// Utilities.
	uint32_t util_count = read_uint32(p_buf, ofs, p_len);
	fn->utilities.resize(util_count);
	fn->_utilities_count = util_count;
	for (uint32_t i = 0; i < util_count; i++) {
		StringName name = read_string_name(p_buf, ofs, p_len);
		fn->utilities.write[i] = Variant::get_validated_utility_function(name);
	}
	fn->_utilities_ptr = util_count > 0 ? fn->utilities.ptr() : nullptr;

	// GDS utilities.
	uint32_t gds_count = read_uint32(p_buf, ofs, p_len);
	fn->gds_utilities.resize(gds_count);
	fn->_gds_utilities_count = gds_count;
	for (uint32_t i = 0; i < gds_count; i++) {
		StringName name = read_string_name(p_buf, ofs, p_len);
		fn->gds_utilities.write[i] = GDScriptUtilityFunctions::get_function(name);
	}
	fn->_gds_utilities_ptr = gds_count > 0 ? fn->gds_utilities.ptr() : nullptr;

	// Lambdas.
	uint32_t lambda_count = read_uint32(p_buf, ofs, p_len);
	fn->lambdas.resize(lambda_count);
	fn->_lambdas_count = lambda_count;
	for (uint32_t i = 0; i < lambda_count; i++) {
		fn->lambdas.write[i] = read_function(p_buf, ofs, p_len, p_script);
	}
	fn->_lambdas_ptr = lambda_count > 0 ? fn->lambdas.ptrw() : nullptr;

	return fn;
}

Error GDScriptBytecodeSerializer::read_script_data(const uint8_t *p_buf, int &ofs, int p_len, GDScript *p_script) {
	// Script metadata.
	p_script->tool = read_uint8(p_buf, ofs, p_len) != 0;
	p_script->_is_abstract = read_uint8(p_buf, ofs, p_len) != 0;
	p_script->fully_qualified_name = read_string(p_buf, ofs, p_len);
	p_script->local_name = read_string_name(p_buf, ofs, p_len);
	p_script->global_name = read_string_name(p_buf, ofs, p_len);
	p_script->simplified_icon_path = read_string(p_buf, ofs, p_len);

	// Native base class.
	StringName native_name = read_string_name(p_buf, ofs, p_len);
	if (native_name != StringName()) {
		p_script->native = Ref<GDScriptNativeClass>(memnew(GDScriptNativeClass(native_name)));
	}

	// Base GDScript path.
	String base_path = read_string(p_buf, ofs, p_len);
	if (!base_path.is_empty()) {
		Ref<GDScript> base_script = ResourceLoader::load(base_path);
		if (base_script.is_valid()) {
			p_script->base = base_script;
			// Inherit member indices from base.
			p_script->member_indices = base_script->member_indices;
		}
	}

	// Members (HashSet).
	uint32_t member_set_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < member_set_count; i++) {
		p_script->members.insert(read_string_name(p_buf, ofs, p_len));
	}

	// Member indices.
	uint32_t mi_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < mi_count; i++) {
		StringName name = read_string_name(p_buf, ofs, p_len);
		GDScript::MemberInfo mi;
		mi.index = read_int32(p_buf, ofs, p_len);
		mi.setter = read_string_name(p_buf, ofs, p_len);
		mi.getter = read_string_name(p_buf, ofs, p_len);
		mi.data_type = read_data_type(p_buf, ofs, p_len);
		mi.property_info = read_property_info(p_buf, ofs, p_len);
		p_script->member_indices[name] = mi;
	}

	// Static variable indices.
	uint32_t svi_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < svi_count; i++) {
		StringName name = read_string_name(p_buf, ofs, p_len);
		GDScript::MemberInfo mi;
		mi.index = read_int32(p_buf, ofs, p_len);
		mi.setter = read_string_name(p_buf, ofs, p_len);
		mi.getter = read_string_name(p_buf, ofs, p_len);
		mi.data_type = read_data_type(p_buf, ofs, p_len);
		mi.property_info = read_property_info(p_buf, ofs, p_len);
		p_script->static_variables_indices[name] = mi;
	}

	// Static variables.
	uint32_t sv_count = read_uint32(p_buf, ofs, p_len);
	p_script->static_variables.resize(sv_count);

	// Constants.
	uint32_t const_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < const_count; i++) {
		StringName name = read_string_name(p_buf, ofs, p_len);
		Variant value = read_variant(p_buf, ofs, p_len);
		p_script->constants[name] = value;
	}

	// Signals.
	uint32_t sig_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < sig_count; i++) {
		StringName name = read_string_name(p_buf, ofs, p_len);
		MethodInfo mi = read_method_info(p_buf, ofs, p_len);
		p_script->_signals[name] = mi;
	}

	// RPC config.
	p_script->rpc_config = read_variant(p_buf, ofs, p_len);

	// Member functions.
	uint32_t func_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < func_count; i++) {
		StringName func_name = read_string_name(p_buf, ofs, p_len);
		GDScriptFunction *fn = read_function(p_buf, ofs, p_len, p_script);
		// Delete old function if it exists.
		if (p_script->member_functions.has(func_name)) {
			memdelete(p_script->member_functions[func_name]);
		}
		p_script->member_functions[func_name] = fn;
	}

	// Special functions serialized outside of member_functions.
	if (read_uint8(p_buf, ofs, p_len) != 0) {
		p_script->implicit_initializer = read_function(p_buf, ofs, p_len, p_script);
	}
	if (read_uint8(p_buf, ofs, p_len) != 0) {
		p_script->implicit_ready = read_function(p_buf, ofs, p_len, p_script);
	}
	if (read_uint8(p_buf, ofs, p_len) != 0) {
		p_script->static_initializer = read_function(p_buf, ofs, p_len, p_script);
	}

	// Lambda info.
	uint32_t li_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < li_count; i++) {
		StringName fn_name = read_string_name(p_buf, ofs, p_len);
		GDScript::LambdaInfo li;
		li.capture_count = read_int32(p_buf, ofs, p_len);
		li.use_self = read_uint8(p_buf, ofs, p_len) != 0;
		// Find the lambda function pointer by searching all functions' lambda vectors.
		// This is a best-effort approach since lambda info mapping is complex.
		(void)fn_name;
		(void)li;
	}

	// Special function pointer for the user-defined constructor living in member_functions.
	StringName init_name = read_string_name(p_buf, ofs, p_len);

	if (init_name != StringName() && p_script->member_functions.has(init_name)) {
		p_script->initializer = p_script->member_functions[init_name];
	}

	// Subclasses (recursive).
	uint32_t sub_count = read_uint32(p_buf, ofs, p_len);
	for (uint32_t i = 0; i < sub_count; i++) {
		StringName sub_name = read_string_name(p_buf, ofs, p_len);
		Ref<GDScript> sub_script;
		sub_script.instantiate();
		sub_script->_owner = p_script;
		sub_script->path = p_script->path;
		Error err = read_script_data(p_buf, ofs, p_len, sub_script.ptr());
		ERR_FAIL_COND_V(err != OK, err);
		p_script->subclasses[sub_name] = sub_script;
		p_script->constants[sub_name] = sub_script;
	}

	p_script->valid = true;
	return OK;
}

// =============================================================================
// Public API
// =============================================================================

Vector<uint8_t> GDScriptBytecodeSerializer::serialize_script(const GDScript *p_script) {
	Vector<uint8_t> buf;

	// Header.
	write_uint32(buf, MAGIC);
	write_uint32(buf, FORMAT_VERSION);

	// Script data (recursive for subclasses).
	write_script_data(buf, p_script);

	return buf;
}

Error GDScriptBytecodeSerializer::deserialize_script(const Vector<uint8_t> &p_data, GDScript *p_script) {
	ERR_FAIL_COND_V(p_data.size() < 8, ERR_INVALID_DATA);

	const uint8_t *buf = p_data.ptr();
	int ofs = 0;
	int len = p_data.size();

	// Header.
	uint32_t magic = read_uint32(buf, ofs, len);
	ERR_FAIL_COND_V_MSG(magic != MAGIC, ERR_INVALID_DATA, "Invalid bytecode file magic.");

	uint32_t version = read_uint32(buf, ofs, len);
	ERR_FAIL_COND_V_MSG(version != FORMAT_VERSION, ERR_INVALID_DATA, vformat("Unsupported bytecode format version: %d", version));

	Error err = read_script_data(buf, ofs, len, p_script);
	ERR_FAIL_COND_V(err != OK, err);

	resolve_script_references(p_script);
	return OK;
}

// =============================================================================
// Human-readable bytecode dump
// =============================================================================

String GDScriptBytecodeSerializer::opcode_to_name(int p_opcode) {
	static const char *names[] = {
		"OPERATOR",
		"OPERATOR_VALIDATED",
		"TYPE_TEST_BUILTIN",
		"TYPE_TEST_ARRAY",
		"TYPE_TEST_DICTIONARY",
		"TYPE_TEST_NATIVE",
		"TYPE_TEST_SCRIPT",
		"SET_KEYED",
		"SET_KEYED_VALIDATED",
		"SET_INDEXED_VALIDATED",
		"GET_KEYED",
		"GET_KEYED_VALIDATED",
		"GET_INDEXED_VALIDATED",
		"SET_NAMED",
		"SET_NAMED_VALIDATED",
		"GET_NAMED",
		"GET_NAMED_VALIDATED",
		"SET_MEMBER",
		"GET_MEMBER",
		"SET_STATIC_VARIABLE",
		"GET_STATIC_VARIABLE",
		"ASSIGN",
		"ASSIGN_NULL",
		"ASSIGN_TRUE",
		"ASSIGN_FALSE",
		"ASSIGN_TYPED_BUILTIN",
		"ASSIGN_TYPED_ARRAY",
		"ASSIGN_TYPED_DICTIONARY",
		"ASSIGN_TYPED_NATIVE",
		"ASSIGN_TYPED_SCRIPT",
		"CAST_TO_BUILTIN",
		"CAST_TO_NATIVE",
		"CAST_TO_SCRIPT",
		"CONSTRUCT",
		"CONSTRUCT_VALIDATED",
		"CONSTRUCT_ARRAY",
		"CONSTRUCT_TYPED_ARRAY",
		"CONSTRUCT_DICTIONARY",
		"CONSTRUCT_TYPED_DICTIONARY",
		"CALL",
		"CALL_RETURN",
		"CALL_ASYNC",
		"CALL_UTILITY",
		"CALL_UTILITY_VALIDATED",
		"CALL_GDSCRIPT_UTILITY",
		"CALL_BUILTIN_TYPE_VALIDATED",
		"CALL_SELF_BASE",
		"CALL_METHOD_BIND",
		"CALL_METHOD_BIND_RET",
		"CALL_BUILTIN_STATIC",
		"CALL_NATIVE_STATIC",
		"CALL_NATIVE_STATIC_VALIDATED_RETURN",
		"CALL_NATIVE_STATIC_VALIDATED_NO_RETURN",
		"CALL_METHOD_BIND_VALIDATED_RETURN",
		"CALL_METHOD_BIND_VALIDATED_NO_RETURN",
		"AWAIT",
		"AWAIT_RESUME",
		"CREATE_LAMBDA",
		"CREATE_SELF_LAMBDA",
		"JUMP",
		"JUMP_IF",
		"JUMP_IF_NOT",
		"JUMP_TO_DEF_ARGUMENT",
		"JUMP_IF_SHARED",
		"RETURN",
		"RETURN_TYPED_BUILTIN",
		"RETURN_TYPED_ARRAY",
		"RETURN_TYPED_DICTIONARY",
		"RETURN_TYPED_NATIVE",
		"RETURN_TYPED_SCRIPT",
		"ITERATE_BEGIN",
		"ITERATE_BEGIN_INT",
		"ITERATE_BEGIN_FLOAT",
		"ITERATE_BEGIN_VECTOR2",
		"ITERATE_BEGIN_VECTOR2I",
		"ITERATE_BEGIN_VECTOR3",
		"ITERATE_BEGIN_VECTOR3I",
		"ITERATE_BEGIN_STRING",
		"ITERATE_BEGIN_DICTIONARY",
		"ITERATE_BEGIN_ARRAY",
		"ITERATE_BEGIN_PACKED_BYTE_ARRAY",
		"ITERATE_BEGIN_PACKED_INT32_ARRAY",
		"ITERATE_BEGIN_PACKED_INT64_ARRAY",
		"ITERATE_BEGIN_PACKED_FLOAT32_ARRAY",
		"ITERATE_BEGIN_PACKED_FLOAT64_ARRAY",
		"ITERATE_BEGIN_PACKED_STRING_ARRAY",
		"ITERATE_BEGIN_PACKED_VECTOR2_ARRAY",
		"ITERATE_BEGIN_PACKED_VECTOR3_ARRAY",
		"ITERATE_BEGIN_PACKED_COLOR_ARRAY",
		"ITERATE_BEGIN_PACKED_VECTOR4_ARRAY",
		"ITERATE_BEGIN_OBJECT",
		"ITERATE_BEGIN_RANGE",
		"ITERATE",
		"ITERATE_INT",
		"ITERATE_FLOAT",
		"ITERATE_VECTOR2",
		"ITERATE_VECTOR2I",
		"ITERATE_VECTOR3",
		"ITERATE_VECTOR3I",
		"ITERATE_STRING",
		"ITERATE_DICTIONARY",
		"ITERATE_ARRAY",
		"ITERATE_PACKED_BYTE_ARRAY",
		"ITERATE_PACKED_INT32_ARRAY",
		"ITERATE_PACKED_INT64_ARRAY",
		"ITERATE_PACKED_FLOAT32_ARRAY",
		"ITERATE_PACKED_FLOAT64_ARRAY",
		"ITERATE_PACKED_STRING_ARRAY",
		"ITERATE_PACKED_VECTOR2_ARRAY",
		"ITERATE_PACKED_VECTOR3_ARRAY",
		"ITERATE_PACKED_COLOR_ARRAY",
		"ITERATE_PACKED_VECTOR4_ARRAY",
		"ITERATE_OBJECT",
		"ITERATE_RANGE",
		"STORE_GLOBAL",
		"STORE_NAMED_GLOBAL",
		"TYPE_ADJUST_BOOL",
		"TYPE_ADJUST_INT",
		"TYPE_ADJUST_FLOAT",
		"TYPE_ADJUST_STRING",
		"TYPE_ADJUST_VECTOR2",
		"TYPE_ADJUST_VECTOR2I",
		"TYPE_ADJUST_RECT2",
		"TYPE_ADJUST_RECT2I",
		"TYPE_ADJUST_VECTOR3",
		"TYPE_ADJUST_VECTOR3I",
		"TYPE_ADJUST_TRANSFORM2D",
		"TYPE_ADJUST_VECTOR4",
		"TYPE_ADJUST_VECTOR4I",
		"TYPE_ADJUST_PLANE",
		"TYPE_ADJUST_QUATERNION",
		"TYPE_ADJUST_AABB",
		"TYPE_ADJUST_BASIS",
		"TYPE_ADJUST_TRANSFORM3D",
		"TYPE_ADJUST_PROJECTION",
		"TYPE_ADJUST_COLOR",
		"TYPE_ADJUST_STRING_NAME",
		"TYPE_ADJUST_NODE_PATH",
		"TYPE_ADJUST_RID",
		"TYPE_ADJUST_OBJECT",
		"TYPE_ADJUST_CALLABLE",
		"TYPE_ADJUST_SIGNAL",
		"TYPE_ADJUST_DICTIONARY",
		"TYPE_ADJUST_ARRAY",
		"TYPE_ADJUST_PACKED_BYTE_ARRAY",
		"TYPE_ADJUST_PACKED_INT32_ARRAY",
		"TYPE_ADJUST_PACKED_INT64_ARRAY",
		"TYPE_ADJUST_PACKED_FLOAT32_ARRAY",
		"TYPE_ADJUST_PACKED_FLOAT64_ARRAY",
		"TYPE_ADJUST_PACKED_STRING_ARRAY",
		"TYPE_ADJUST_PACKED_VECTOR2_ARRAY",
		"TYPE_ADJUST_PACKED_VECTOR3_ARRAY",
		"TYPE_ADJUST_PACKED_COLOR_ARRAY",
		"TYPE_ADJUST_PACKED_VECTOR4_ARRAY",
		"ASSERT",
		"BREAKPOINT",
		"LINE",
		"END",
	};
	constexpr int name_count = sizeof(names) / sizeof(names[0]);
	if (p_opcode >= 0 && p_opcode < name_count) {
		return names[p_opcode];
	}
	return vformat("UNKNOWN(%d)", p_opcode);
}

String GDScriptBytecodeSerializer::variant_to_text(const Variant &v) {
	if (v.get_type() == Variant::STRING) {
		return "\"" + String(v) + "\"";
	} else if (v.get_type() == Variant::STRING_NAME) {
		return "&\"" + String(v) + "\"";
	} else if (v.get_type() == Variant::NODE_PATH) {
		return "^\"" + String(v) + "\"";
	} else if (v.get_type() == Variant::OBJECT) {
		Object *obj = v;
		if (!obj) {
			return "null";
		}
		GDScriptNativeClass *nc = Object::cast_to<GDScriptNativeClass>(obj);
		if (nc) {
			return vformat("NativeClass(%s)", nc->get_name());
		}
		GDScript *gs = Object::cast_to<GDScript>(obj);
		if (gs) {
			return vformat("GDScript(%s)", gs->get_path());
		}
		return vformat("Object(%s)", obj->get_class());
	}
	return v.stringify();
}

String GDScriptBytecodeSerializer::data_type_to_text(const GDScriptDataType &dt) {
	switch (dt.kind) {
		case GDScriptDataType::VARIANT:
			return "Variant";
		case GDScriptDataType::BUILTIN:
			return Variant::get_type_name(dt.builtin_type);
		case GDScriptDataType::NATIVE:
			return String(dt.native_type);
		case GDScriptDataType::SCRIPT:
		case GDScriptDataType::GDSCRIPT: {
			String s = "Script(";
			if (dt.script_type_ref.is_valid()) {
				s += dt.script_type_ref->get_path();
			}
			s += ")";
			return s;
		}
	}
	return "?";
}

String GDScriptBytecodeSerializer::dump_function_text(const GDScriptFunction *fn, const String &p_indent) {
	String out;
	String ind = p_indent;

	out += ind + vformat("Function: %s\n", String(fn->name));
	out += ind + vformat("  source: %s\n", String(fn->source));
	out += ind + vformat("  static: %s\n", fn->_static ? "true" : "false");
	out += ind + vformat("  arguments: %d\n", fn->_argument_count);
	out += ind + vformat("  stack_size: %d\n", fn->_stack_size);
	out += ind + vformat("  instruction_args_size: %d\n", fn->_instruction_args_size);
	out += ind + vformat("  initial_line: %d\n", fn->_initial_line);

	if (fn->argument_types.size() > 0) {
		out += ind + "  argument_types:\n";
		for (int i = 0; i < fn->argument_types.size(); i++) {
			out += ind + vformat("    [%d] %s\n", i, data_type_to_text(fn->argument_types[i]));
		}
	}

	out += ind + vformat("  return_type: %s\n", data_type_to_text(fn->return_type));

	if (fn->default_arguments.size() > 0) {
		out += ind + vformat("  default_arguments (%d): [", fn->default_arguments.size());
		for (int i = 0; i < fn->default_arguments.size(); i++) {
			if (i > 0) {
				out += ", ";
			}
			out += itos(fn->default_arguments[i]);
		}
		out += "]\n";
	}

	if (fn->constants.size() > 0) {
		out += ind + vformat("  constants (%d):\n", fn->constants.size());
		for (int i = 0; i < fn->constants.size(); i++) {
			out += ind + vformat("    [%d] (%s) %s\n", i,
					Variant::get_type_name(fn->constants[i].get_type()),
					variant_to_text(fn->constants[i]));
		}
	}

	if (fn->global_names.size() > 0) {
		out += ind + vformat("  global_names (%d):\n", fn->global_names.size());
		for (int i = 0; i < fn->global_names.size(); i++) {
			out += ind + vformat("    [%d] %s\n", i, String(fn->global_names[i]));
		}
	}

	const GDScriptFunction::ImportTable &it = fn->import_table;

	if (it.methods.size() > 0) {
		out += ind + vformat("  import_methods (%d):\n", it.methods.size());
		for (int i = 0; i < it.methods.size(); i++) {
			out += ind + vformat("    [%d] %s::%s\n", i, String(it.methods[i].class_name), String(it.methods[i].method_name));
		}
	}

	if (it.operators.size() > 0) {
		out += ind + vformat("  import_operators (%d):\n", it.operators.size());
		for (int i = 0; i < it.operators.size(); i++) {
			out += ind + vformat("    [%d] op=%d left=%s right=%s\n", i,
					(int)it.operators[i].op,
					Variant::get_type_name(it.operators[i].left_type),
					Variant::get_type_name(it.operators[i].right_type));
		}
	}

	if (it.setters.size() > 0) {
		out += ind + vformat("  import_setters (%d):\n", it.setters.size());
		for (int i = 0; i < it.setters.size(); i++) {
			out += ind + vformat("    [%d] %s.%s\n", i, Variant::get_type_name(it.setters[i].type), String(it.setters[i].member));
		}
	}

	if (it.getters.size() > 0) {
		out += ind + vformat("  import_getters (%d):\n", it.getters.size());
		for (int i = 0; i < it.getters.size(); i++) {
			out += ind + vformat("    [%d] %s.%s\n", i, Variant::get_type_name(it.getters[i].type), String(it.getters[i].member));
		}
	}

	if (it.builtin_methods.size() > 0) {
		out += ind + vformat("  import_builtin_methods (%d):\n", it.builtin_methods.size());
		for (int i = 0; i < it.builtin_methods.size(); i++) {
			out += ind + vformat("    [%d] %s.%s\n", i, Variant::get_type_name(it.builtin_methods[i].type), String(it.builtin_methods[i].method));
		}
	}

	if (it.constructors.size() > 0) {
		out += ind + vformat("  import_constructors (%d):\n", it.constructors.size());
		for (int i = 0; i < it.constructors.size(); i++) {
			out += ind + vformat("    [%d] %s#%d\n", i, Variant::get_type_name(it.constructors[i].type), it.constructors[i].constructor_idx);
		}
	}

	if (it.utilities.size() > 0) {
		out += ind + vformat("  import_utilities (%d):\n", it.utilities.size());
		for (int i = 0; i < it.utilities.size(); i++) {
			out += ind + vformat("    [%d] %s\n", i, String(it.utilities[i].name));
		}
	}

	if (it.gds_utilities.size() > 0) {
		out += ind + vformat("  import_gds_utilities (%d):\n", it.gds_utilities.size());
		for (int i = 0; i < it.gds_utilities.size(); i++) {
			out += ind + vformat("    [%d] %s\n", i, String(it.gds_utilities[i].name));
		}
	}

	if (fn->code.size() > 0) {
		out += ind + vformat("  code (%d ints):\n", fn->code.size());

		for (int ip = 0; ip < fn->code.size(); ip += 8) {
			out += ind + vformat("    %04d:", ip);
			for (int j = 0; j < 8 && (ip + j) < fn->code.size(); j++) {
				out += vformat(" %08X", (uint32_t)fn->code[ip + j]);
			}
			out += "\n";
		}

		out += ind + "  opcode_scan (values matching known opcodes):\n";
		for (int ip = 0; ip < fn->code.size(); ip++) {
			int val = fn->code[ip];
			if (val >= 0 && val <= GDScriptFunction::OPCODE_END) {
				out += ind + vformat("    [%04d] %s (%d)\n", ip, opcode_to_name(val), val);
			}
		}
	}

	if (fn->lambdas.size() > 0) {
		out += ind + vformat("  lambdas (%d):\n", fn->lambdas.size());
		for (int i = 0; i < fn->lambdas.size(); i++) {
			out += dump_function_text(fn->lambdas[i], ind + "    ");
		}
	}

	return out;
}

String GDScriptBytecodeSerializer::dump_script_data_text(const GDScript *p_script, const String &p_indent) {
	String out;
	String ind = p_indent;

	out += ind + vformat("Script: %s\n", p_script->get_path());
	out += ind + vformat("  tool: %s\n", p_script->tool ? "true" : "false");
	out += ind + vformat("  abstract: %s\n", p_script->_is_abstract ? "true" : "false");

	if (p_script->native.is_valid()) {
		out += ind + vformat("  native_base: %s\n", String(p_script->native->get_name()));
	}

	if (p_script->base.is_valid()) {
		out += ind + vformat("  base_script: %s\n", p_script->base->get_path());
	}

	if (p_script->members.size() > 0) {
		out += ind + vformat("  members (%d):\n", p_script->members.size());
		for (const StringName &m : p_script->members) {
			out += ind + vformat("    - %s\n", String(m));
		}
	}

	if (p_script->member_indices.size() > 0) {
		out += ind + vformat("  member_indices (%d):\n", p_script->member_indices.size());
		for (const KeyValue<StringName, GDScript::MemberInfo> &E : p_script->member_indices) {
			out += ind + vformat("    %s: index=%d type=%s\n", String(E.key), E.value.index, data_type_to_text(E.value.data_type));
		}
	}

	if (p_script->static_variables_indices.size() > 0) {
		out += ind + vformat("  static_variable_indices (%d):\n", p_script->static_variables_indices.size());
		for (const KeyValue<StringName, GDScript::MemberInfo> &E : p_script->static_variables_indices) {
			out += ind + vformat("    %s: index=%d type=%s\n", String(E.key), E.value.index, data_type_to_text(E.value.data_type));
		}
	}

	if (p_script->constants.size() > 0) {
		out += ind + vformat("  constants (%d):\n", p_script->constants.size());
		for (const KeyValue<StringName, Variant> &E : p_script->constants) {
			out += ind + vformat("    %s = (%s) %s\n", String(E.key),
					Variant::get_type_name(E.value.get_type()),
					variant_to_text(E.value));
		}
	}

	if (p_script->_signals.size() > 0) {
		out += ind + vformat("  signals (%d):\n", p_script->_signals.size());
		for (const KeyValue<StringName, MethodInfo> &E : p_script->_signals) {
			out += ind + vformat("    %s (args: %d)\n", String(E.key), E.value.arguments.size());
		}
	}

	if (p_script->member_functions.size() > 0) {
		out += ind + vformat("  member_functions (%d):\n", p_script->member_functions.size());
		out += ind + "  ----------------------------------------\n";
		for (const KeyValue<StringName, GDScriptFunction *> &E : p_script->member_functions) {
			out += dump_function_text(E.value, ind + "  ");
			out += ind + "  ----------------------------------------\n";
		}
	}

	if (p_script->subclasses.size() > 0) {
		out += ind + vformat("  subclasses (%d):\n", p_script->subclasses.size());
		for (const KeyValue<StringName, Ref<GDScript>> &E : p_script->subclasses) {
			out += ind + vformat("  === Subclass: %s ===\n", String(E.key));
			out += dump_script_data_text(E.value.ptr(), ind + "    ");
		}
	}

	return out;
}

String GDScriptBytecodeSerializer::dump_script_text(const GDScript *p_script) {
	String out;
	out += "========================================\n";
	out += "GDScript Bytecode Dump\n";
	out += vformat("Format: GDBC v%d\n", FORMAT_VERSION);
	out += "========================================\n\n";
	out += dump_script_data_text(p_script, "");
	return out;
}
