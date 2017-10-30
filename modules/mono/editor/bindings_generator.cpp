/*************************************************************************/
/*  bindings_generator.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "bindings_generator.h"

#ifdef DEBUG_METHODS_ENABLED

#include "global_constants.h"
#include "io/compression.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "os/os.h"
#include "project_settings.h"
#include "ucaps.h"

#include "../glue/cs_compressed.gen.h"
#include "../godotsharp_defs.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../utils/path_utils.h"
#include "../utils/string_utils.h"
#include "csharp_project.h"
#include "net_solution.h"

#define CS_INDENT "    "

#define INDENT1 CS_INDENT
#define INDENT2 INDENT1 INDENT1
#define INDENT3 INDENT2 INDENT1
#define INDENT4 INDENT3 INDENT1
#define INDENT5 INDENT4 INDENT1

#define MEMBER_BEGIN "\n" INDENT2

#define OPEN_BLOCK "{\n"
#define CLOSE_BLOCK "}\n"

#define OPEN_BLOCK_L2 INDENT2 OPEN_BLOCK INDENT3
#define OPEN_BLOCK_L3 INDENT3 OPEN_BLOCK INDENT4
#define OPEN_BLOCK_L4 INDENT4 OPEN_BLOCK INDENT5
#define CLOSE_BLOCK_L2 INDENT2 CLOSE_BLOCK
#define CLOSE_BLOCK_L3 INDENT3 CLOSE_BLOCK
#define CLOSE_BLOCK_L4 INDENT4 CLOSE_BLOCK

#define LOCAL_RET "ret"

#define CS_CLASS_NATIVECALLS "NativeCalls"
#define CS_CLASS_NATIVECALLS_EDITOR "EditorNativeCalls"
#define CS_FIELD_MEMORYOWN "memoryOwn"
#define CS_PARAM_METHODBIND "method"
#define CS_PARAM_INSTANCE "ptr"
#define CS_SMETHOD_GETINSTANCE "GetPtr"
#define CS_FIELD_SINGLETON "instance"
#define CS_PROP_SINGLETON "Instance"
#define CS_CLASS_SIGNALAWAITER "SignalAwaiter"
#define CS_METHOD_CALL "Call"

#define GLUE_HEADER_FILE "glue_header.h"
#define ICALL_PREFIX "godot_icall_"
#define SINGLETON_ICALL_SUFFIX "_get_singleton"
#define ICALL_GET_METHODBIND ICALL_PREFIX "ClassDB_get_method"
#define ICALL_CONNECT_SIGNAL_AWAITER ICALL_PREFIX "Object_connect_signal_awaiter"
#define ICALL_OBJECT_DTOR ICALL_PREFIX "Object_Dtor"
#define C_LOCAL_PTRCALL_ARGS "call_args"
#define C_MACRO_OBJECT_CONSTRUCT "GODOTSHARP_INSTANCE_OBJECT"

#define C_NS_MONOUTILS "GDMonoUtils"
#define C_NS_MONOINTERNALS "GDMonoInternals"
#define C_METHOD_TIE_MANAGED_TO_UNMANAGED C_NS_MONOINTERNALS "::tie_managed_to_unmanaged"
#define C_METHOD_UNMANAGED_GET_MANAGED C_NS_MONOUTILS "::unmanaged_get_managed"

#define C_NS_MONOMARSHAL "GDMonoMarshal"
#define C_METHOD_MANAGED_TO_VARIANT C_NS_MONOMARSHAL "::mono_object_to_variant"
#define C_METHOD_MANAGED_FROM_VARIANT C_NS_MONOMARSHAL "::variant_to_mono_object"
#define C_METHOD_MONOSTR_TO_GODOT C_NS_MONOMARSHAL "::mono_string_to_godot"
#define C_METHOD_MONOSTR_FROM_GODOT C_NS_MONOMARSHAL "::mono_string_from_godot"
#define C_METHOD_MONOARRAY_TO(m_type) C_NS_MONOMARSHAL "::mono_array_to_" #m_type
#define C_METHOD_MONOARRAY_FROM(m_type) C_NS_MONOMARSHAL "::" #m_type "_to_mono_array"
#define C_METHOD_MANAGED_TO_DICT C_NS_MONOMARSHAL "::mono_object_to_Dictionary"
#define C_METHOD_MANAGED_FROM_DICT C_NS_MONOMARSHAL "::Dictionary_to_mono_object"

const char *BindingsGenerator::TypeInterface::DEFAULT_VARARG_C_IN = "\t%0 %1_in = %1;\n";

bool BindingsGenerator::verbose_output = false;

static String snake_to_pascal_case(const String &p_identifier) {

	String ret;
	Vector<String> parts = p_identifier.split("_", true);

	for (int i = 0; i < parts.size(); i++) {
		String part = parts[i];

		if (part.length()) {
			part[0] = _find_upper(part[0]);
			ret += part;
		} else {
			if (i == 0 || i == (parts.size() - 1)) {
				// Preserve underscores at the beginning and end
				ret += "_";
			} else {
				// Preserve contiguous underscores
				if (parts[i - 1].length()) {
					ret += "__";
				} else {
					ret += "_";
				}
			}
		}
	}

	return ret;
}

static String snake_to_camel_case(const String &p_identifier) {

	String ret;
	Vector<String> parts = p_identifier.split("_", true);

	for (int i = 0; i < parts.size(); i++) {
		String part = parts[i];

		if (part.length()) {
			if (i != 0)
				part[0] = _find_upper(part[0]);
			ret += part;
		} else {
			if (i == 0 || i == (parts.size() - 1)) {
				// Preserve underscores at the beginning and end
				ret += "_";
			} else {
				// Preserve contiguous underscores
				if (parts[i - 1].length()) {
					ret += "__";
				} else {
					ret += "_";
				}
			}
		}
	}

	return ret;
}

void BindingsGenerator::_generate_header_icalls() {

	core_custom_icalls.clear();

	core_custom_icalls.push_back(InternalCall(ICALL_GET_METHODBIND, "IntPtr", "string type, string method"));
	core_custom_icalls.push_back(InternalCall(ICALL_OBJECT_DTOR, "void", "IntPtr ptr"));

	core_custom_icalls.push_back(InternalCall(ICALL_CONNECT_SIGNAL_AWAITER, "Error",
			"IntPtr source, string signal, IntPtr target, " CS_CLASS_SIGNALAWAITER " awaiter"));

	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "NodePath_Ctor", "IntPtr", "string path"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "NodePath_Dtor", "void", "IntPtr ptr"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "NodePath_operator_String", "string", "IntPtr ptr"));

	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "RID_Ctor", "IntPtr", "IntPtr from"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "RID_Dtor", "void", "IntPtr ptr"));

	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "String_md5_buffer", "byte[]", "string str"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "String_md5_text", "string", "string str"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "String_rfind", "int", "string str, string what, int from"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "String_rfindn", "int", "string str, string what, int from"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "String_sha256_buffer", "byte[]", "string str"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "String_sha256_text", "string", "string str"));

	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_bytes2var", "object", "byte[] bytes"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_convert", "object", "object what, int type"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_hash", "int", "object var"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_instance_from_id", "Object", "int instance_id"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_print", "void", "object[] what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_printerr", "void", "object[] what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_printraw", "void", "object[] what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_prints", "void", "object[] what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_printt", "void", "object[] what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_seed", "void", "int seed"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_str", "string", "object[] what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_str2var", "object", "string str"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_type_exists", "bool", "string type"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_var2bytes", "byte[]", "object what"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_var2str", "string", "object var"));
	core_custom_icalls.push_back(InternalCall(ICALL_PREFIX "Godot_weakref", "WeakRef", "IntPtr obj"));
}

void BindingsGenerator::_generate_method_icalls(const TypeInterface &p_itype) {

	for (const List<MethodInterface>::Element *E = p_itype.methods.front(); E; E = E->next()) {
		const MethodInterface &imethod = E->get();

		if (imethod.is_virtual)
			continue;

		const TypeInterface *return_type = _get_type_by_name_or_placeholder(imethod.return_type);

		String im_sig = "IntPtr " CS_PARAM_METHODBIND ", IntPtr " CS_PARAM_INSTANCE;
		String im_unique_sig = imethod.return_type + ",IntPtr,IntPtr";

		// Get arguments information
		int i = 0;
		for (const List<ArgumentInterface>::Element *F = imethod.arguments.front(); F; F = F->next()) {
			const TypeInterface *arg_type = _get_type_by_name_or_placeholder(F->get().type);

			im_sig += ", ";
			im_sig += arg_type->im_type_in;
			im_sig += " arg";
			im_sig += itos(i + 1);

			im_unique_sig += ",";
			im_unique_sig += get_unique_sig(*arg_type);

			i++;
		}

		// godot_icall_{argc}_{icallcount}
		String icall_method = ICALL_PREFIX + itos(imethod.arguments.size()) + "_" + itos(method_icalls.size());

		InternalCall im_icall = InternalCall(p_itype.api_type, icall_method, return_type->im_type_out, im_sig, im_unique_sig);

		List<InternalCall>::Element *match = method_icalls.find(im_icall);

		if (match) {
			if (p_itype.api_type != ClassDB::API_EDITOR)
				match->get().editor_only = false;
			method_icalls_map.insert(&E->get(), &match->get());
		} else {
			List<InternalCall>::Element *added = method_icalls.push_back(im_icall);
			method_icalls_map.insert(&E->get(), &added->get());
		}
	}
}

Error BindingsGenerator::generate_cs_core_project(const String &p_output_dir, bool p_verbose_output) {

	verbose_output = p_verbose_output;

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);

	if (!DirAccess::exists(p_output_dir)) {
		Error err = da->make_dir_recursive(p_output_dir);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	da->change_dir(p_output_dir);
	da->make_dir("Core");
	da->make_dir("ObjectType");

	String core_dir = path_join(p_output_dir, "Core");
	String obj_type_dir = path_join(p_output_dir, "ObjectType");

	Vector<String> compile_items;

	NETSolution solution(API_ASSEMBLY_NAME);

	if (!solution.set_path(p_output_dir))
		return ERR_FILE_NOT_FOUND;

	for (Map<String, TypeInterface>::Element *E = obj_types.front(); E; E = E->next()) {
		const TypeInterface &itype = E->get();

		if (itype.api_type == ClassDB::API_EDITOR)
			continue;

		String output_file = path_join(obj_type_dir, E->get().proxy_name + ".cs");
		Error err = _generate_cs_type(E->get(), output_file);

		if (err == ERR_SKIP)
			continue;

		if (err != OK)
			return err;

		compile_items.push_back(output_file);
	}

#define GENERATE_BUILTIN_TYPE(m_name)                                       \
	{                                                                       \
		String output_file = path_join(core_dir, #m_name ".cs");            \
		Error err = _generate_cs_type(builtin_types[#m_name], output_file); \
		if (err != OK)                                                      \
			return err;                                                     \
		compile_items.push_back(output_file);                               \
	}

	GENERATE_BUILTIN_TYPE(NodePath);
	GENERATE_BUILTIN_TYPE(RID);

#undef GENERATE_BUILTIN_TYPE

	// Generate source for GlobalConstants

	String constants_source;
	int global_constants_count = GlobalConstants::get_global_constant_count();

	if (global_constants_count > 0) {
		Map<String, DocData::ClassDoc>::Element *match = EditorHelp::get_doc_data()->class_list.find("@Global Scope");

		ERR_EXPLAIN("Could not find `@Global Scope` in DocData");
		ERR_FAIL_COND_V(!match, ERR_BUG);

		const DocData::ClassDoc &global_scope_doc = match->value();

		for (int i = 0; i < global_constants_count; i++) {
			const DocData::ConstantDoc &const_doc = global_scope_doc.constants[i];

			if (i > 0)
				constants_source += MEMBER_BEGIN;

			if (const_doc.description.size()) {
				constants_source += "/// <summary>\n";

				Vector<String> description_lines = const_doc.description.split("\n");

				for (int i = 0; i < description_lines.size(); i++) {
					if (description_lines[i].size()) {
						constants_source += INDENT2 "/// ";
						constants_source += description_lines[i].strip_edges().xml_escape();
						constants_source += "\n";
					}
				}

				constants_source += INDENT2 "/// </summary>" MEMBER_BEGIN;
			}

			constants_source += "public const int ";
			constants_source += GlobalConstants::get_global_constant_name(i);
			constants_source += " = ";
			constants_source += itos(GlobalConstants::get_global_constant_value(i));
			constants_source += ";";
		}
	}

	// Generate sources from compressed files

	Map<String, CompressedFile> compressed_files;
	get_compressed_files(compressed_files);

	for (Map<String, CompressedFile>::Element *E = compressed_files.front(); E; E = E->next()) {
		const String &file_name = E->key();
		const CompressedFile &file_data = E->value();

		String output_file = path_join(core_dir, file_name);

		Vector<uint8_t> data;
		data.resize(file_data.uncompressed_size);
		Compression::decompress(data.ptr(), file_data.uncompressed_size, file_data.data, file_data.compressed_size, Compression::MODE_DEFLATE);

		if (file_name.get_basename() == BINDINGS_GLOBAL_SCOPE_CLASS) {
			// GD.cs must be formatted to include the generated global constants
			String data_str = String::utf8(reinterpret_cast<const char *>(data.ptr()), data.size());

			Dictionary format_keys;
			format_keys["GodotGlobalConstants"] = constants_source;
			data_str = data_str.format(format_keys, "/*{_}*/");

			CharString data_utf8 = data_str.utf8();
			data.resize(data_utf8.length());
			copymem(data.ptr(), reinterpret_cast<const uint8_t *>(data_utf8.get_data()), data_utf8.length());
		}

		FileAccessRef file = FileAccess::open(output_file, FileAccess::WRITE);
		ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);
		file->store_buffer(data.ptr(), data.size());
		file->close();

		compile_items.push_back(output_file);
	}

	List<String> cs_icalls_content;

	cs_icalls_content.push_back("using System;\n"
								"using System.Runtime.CompilerServices;\n"
								"using System.Collections.Generic;\n"
								"\n");
	cs_icalls_content.push_back("namespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);
	cs_icalls_content.push_back(INDENT1 "internal static class " CS_CLASS_NATIVECALLS "\n" INDENT1 OPEN_BLOCK);

#define ADD_INTERNAL_CALL(m_icall)                                                             \
	if (!m_icall.editor_only) {                                                                \
		cs_icalls_content.push_back(INDENT2 "[MethodImpl(MethodImplOptions.InternalCall)]\n"); \
		cs_icalls_content.push_back(INDENT2 "internal extern static ");                        \
		cs_icalls_content.push_back(m_icall.im_type_out + " ");                                \
		cs_icalls_content.push_back(m_icall.name + "(");                                       \
		cs_icalls_content.push_back(m_icall.im_sig + ");\n");                                  \
	}

	for (const List<InternalCall>::Element *E = core_custom_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());
	for (const List<InternalCall>::Element *E = method_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());

#undef ADD_INTERNAL_CALL

	cs_icalls_content.push_back(INDENT1 CLOSE_BLOCK CLOSE_BLOCK);

	String internal_methods_file = path_join(core_dir, CS_CLASS_NATIVECALLS ".cs");

	Error err = _save_file(internal_methods_file, cs_icalls_content);
	if (err != OK)
		return err;

	compile_items.push_back(internal_methods_file);

	String guid = CSharpProject::generate_core_api_project(p_output_dir, compile_items);

	solution.add_new_project(API_ASSEMBLY_NAME, guid);

	Error sln_error = solution.save();
	if (sln_error != OK) {
		ERR_PRINT("Could not to save .NET solution.");
		return sln_error;
	}

	if (verbose_output)
		OS::get_singleton()->print("The solution and C# project for the Core API was generated successfully\n");

	return OK;
}

Error BindingsGenerator::generate_cs_editor_project(const String &p_output_dir, const String &p_core_dll_path, bool p_verbose_output) {

	verbose_output = p_verbose_output;

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);

	if (!DirAccess::exists(p_output_dir)) {
		Error err = da->make_dir_recursive(p_output_dir);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	da->change_dir(p_output_dir);
	da->make_dir("Core");
	da->make_dir("ObjectType");

	String core_dir = path_join(p_output_dir, "Core");
	String obj_type_dir = path_join(p_output_dir, "ObjectType");

	Vector<String> compile_items;

	NETSolution solution(EDITOR_API_ASSEMBLY_NAME);

	if (!solution.set_path(p_output_dir))
		return ERR_FILE_NOT_FOUND;

	for (Map<String, TypeInterface>::Element *E = obj_types.front(); E; E = E->next()) {
		const TypeInterface &itype = E->get();

		if (itype.api_type != ClassDB::API_EDITOR)
			continue;

		String output_file = path_join(obj_type_dir, E->get().proxy_name + ".cs");
		Error err = _generate_cs_type(E->get(), output_file);

		if (err == ERR_SKIP)
			continue;

		if (err != OK)
			return err;

		compile_items.push_back(output_file);
	}

	List<String> cs_icalls_content;

	cs_icalls_content.push_back("using System;\n"
								"using System.Runtime.CompilerServices;\n"
								"using System.Collections.Generic;\n"
								"\n");
	cs_icalls_content.push_back("namespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);
	cs_icalls_content.push_back(INDENT1 "internal static class " CS_CLASS_NATIVECALLS_EDITOR "\n" INDENT1 OPEN_BLOCK);

#define ADD_INTERNAL_CALL(m_icall)                                                             \
	if (m_icall.editor_only) {                                                                 \
		cs_icalls_content.push_back(INDENT2 "[MethodImpl(MethodImplOptions.InternalCall)]\n"); \
		cs_icalls_content.push_back(INDENT2 "internal extern static ");                        \
		cs_icalls_content.push_back(m_icall.im_type_out + " ");                                \
		cs_icalls_content.push_back(m_icall.name + "(");                                       \
		cs_icalls_content.push_back(m_icall.im_sig + ");\n");                                  \
	}

	for (const List<InternalCall>::Element *E = editor_custom_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());
	for (const List<InternalCall>::Element *E = method_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL(E->get());

#undef ADD_INTERNAL_CALL

	cs_icalls_content.push_back(INDENT1 CLOSE_BLOCK CLOSE_BLOCK);

	String internal_methods_file = path_join(core_dir, CS_CLASS_NATIVECALLS_EDITOR ".cs");

	Error err = _save_file(internal_methods_file, cs_icalls_content);
	if (err != OK)
		return err;

	compile_items.push_back(internal_methods_file);

	String guid = CSharpProject::generate_editor_api_project(p_output_dir, p_core_dll_path, compile_items);

	solution.add_new_project(EDITOR_API_ASSEMBLY_NAME, guid);

	Error sln_error = solution.save();
	if (sln_error != OK) {
		ERR_PRINT("Could not to save .NET solution.");
		return sln_error;
	}

	if (verbose_output)
		OS::get_singleton()->print("The solution and C# project for the Editor API was generated successfully\n");

	return OK;
}

// TODO: there are constants that hide inherited members. must explicitly use `new` to avoid warnings
// e.g.: warning CS0108: 'SpriteBase3D.FLAG_MAX' hides inherited member 'GeometryInstance.FLAG_MAX'. Use the new keyword if hiding was intended.
Error BindingsGenerator::_generate_cs_type(const TypeInterface &itype, const String &p_output_file) {

	bool is_derived_type = itype.base_name.length();

	List<InternalCall> &custom_icalls = itype.api_type == ClassDB::API_EDITOR ? editor_custom_icalls : core_custom_icalls;

	if (verbose_output)
		OS::get_singleton()->print(String("Generating " + itype.proxy_name + ".cs...\n").utf8());

	String ctor_method(ICALL_PREFIX + itype.proxy_name + "_Ctor");

	List<String> output;

	output.push_back("using System;\n"); // IntPtr

	if (itype.requires_collections)
		output.push_back("using System.Collections.Generic;\n"); // Dictionary

	output.push_back("\nnamespace " BINDINGS_NAMESPACE "\n" OPEN_BLOCK);

	const DocData::ClassDoc *class_doc = itype.class_doc;

	if (class_doc && class_doc->description.size()) {
		output.push_back(INDENT1 "/// <summary>\n");

		Vector<String> description_lines = class_doc->description.split("\n");

		for (int i = 0; i < description_lines.size(); i++) {
			if (description_lines[i].size()) {
				output.push_back(INDENT1 "/// ");
				output.push_back(description_lines[i].strip_edges().xml_escape());
				output.push_back("\n");
			}
		}

		output.push_back(INDENT1 "/// </summary>\n");
	}

	output.push_back(INDENT1 "public ");
	output.push_back(itype.is_singleton ? "static class " : "class ");
	output.push_back(itype.proxy_name);

	if (itype.is_singleton || !itype.is_object_type) {
		output.push_back("\n");
	} else if (!is_derived_type) {
		output.push_back(" : IDisposable\n");
	} else if (obj_types.has(itype.base_name)) {
		output.push_back(" : ");
		output.push_back(obj_types[itype.base_name].proxy_name);
		output.push_back("\n");
	} else {
		ERR_PRINTS("Base type '" + itype.base_name + "' does not exist, for class " + itype.name);
		return ERR_INVALID_DATA;
	}

	output.push_back(INDENT1 "{");

	if (class_doc) {

		// Add constants

		for (int i = 0; i < class_doc->constants.size(); i++) {
			const DocData::ConstantDoc &const_doc = class_doc->constants[i];

			if (const_doc.description.size()) {
				output.push_back(MEMBER_BEGIN "/// <summary>\n");

				Vector<String> description_lines = const_doc.description.split("\n");

				for (int i = 0; i < description_lines.size(); i++) {
					if (description_lines[i].size()) {
						output.push_back(INDENT2 "/// ");
						output.push_back(description_lines[i].strip_edges().xml_escape());
						output.push_back("\n");
					}
				}

				output.push_back(INDENT2 "/// </summary>");
			}

			output.push_back(MEMBER_BEGIN "public const int ");
			output.push_back(const_doc.name);
			output.push_back(" = ");
			output.push_back(const_doc.value);
			output.push_back(";");
		}

		if (class_doc->constants.size())
			output.push_back("\n");

		// Add properties

		const Vector<DocData::PropertyDoc> &properties = class_doc->properties;

		for (int i = 0; i < properties.size(); i++) {
			const DocData::PropertyDoc &prop_doc = properties[i];
			Error prop_err = _generate_cs_property(itype, prop_doc, output);
			if (prop_err != OK) {
				ERR_EXPLAIN("Failed to generate property '" + prop_doc.name + "' for class '" + itype.name + "'");
				ERR_FAIL_V(prop_err);
			}
		}

		if (class_doc->properties.size())
			output.push_back("\n");
	}

	if (!itype.is_object_type) {
		output.push_back(MEMBER_BEGIN "private const string " BINDINGS_NATIVE_NAME_FIELD " = \"" + itype.name + "\";\n");
		output.push_back(MEMBER_BEGIN "private bool disposed = false;\n");
		output.push_back(MEMBER_BEGIN "internal IntPtr " BINDINGS_PTR_FIELD ";\n");

		output.push_back(MEMBER_BEGIN "internal static IntPtr " CS_SMETHOD_GETINSTANCE "(");
		output.push_back(itype.proxy_name);
		output.push_back(" instance)\n" OPEN_BLOCK_L2 "return instance == null ? IntPtr.Zero : instance." BINDINGS_PTR_FIELD ";\n" CLOSE_BLOCK_L2);

		// Add Destructor
		output.push_back(MEMBER_BEGIN "~");
		output.push_back(itype.proxy_name);
		output.push_back("()\n" OPEN_BLOCK_L2 "Dispose(false);\n" CLOSE_BLOCK_L2);

		// Add the Dispose from IDisposable
		output.push_back(MEMBER_BEGIN "public void Dispose()\n" OPEN_BLOCK_L2 "Dispose(true);\n" INDENT3 "GC.SuppressFinalize(this);\n" CLOSE_BLOCK_L2);

		// Add the virtual Dispose
		output.push_back(MEMBER_BEGIN "public virtual void Dispose(bool disposing)\n" OPEN_BLOCK_L2
									  "if (disposed) return;\n" INDENT3
									  "if (" BINDINGS_PTR_FIELD " != IntPtr.Zero)\n" OPEN_BLOCK_L3 "NativeCalls.godot_icall_");
		output.push_back(itype.proxy_name);
		output.push_back("_Dtor(" BINDINGS_PTR_FIELD ");\n" INDENT5 BINDINGS_PTR_FIELD " = IntPtr.Zero;\n" CLOSE_BLOCK_L3 INDENT3
						 "GC.SuppressFinalize(this);\n" INDENT3 "disposed = true;\n" CLOSE_BLOCK_L2);

		output.push_back(MEMBER_BEGIN "internal ");
		output.push_back(itype.proxy_name);
		output.push_back("(IntPtr " BINDINGS_PTR_FIELD ")\n" OPEN_BLOCK_L2 "this." BINDINGS_PTR_FIELD " = " BINDINGS_PTR_FIELD ";\n" CLOSE_BLOCK_L2);

		output.push_back(MEMBER_BEGIN "public IntPtr NativeInstance\n" OPEN_BLOCK_L2
									  "get { return " BINDINGS_PTR_FIELD "; }\n" CLOSE_BLOCK_L2);
	} else if (itype.is_singleton) {
		// Add the type name and the singleton pointer as static fields

		output.push_back(MEMBER_BEGIN "private const string " BINDINGS_NATIVE_NAME_FIELD " = \"");
		output.push_back(itype.name);
		output.push_back("\";\n");

		output.push_back(INDENT2 "internal static IntPtr " BINDINGS_PTR_FIELD " = ");
		output.push_back(itype.api_type == ClassDB::API_EDITOR ? CS_CLASS_NATIVECALLS_EDITOR : CS_CLASS_NATIVECALLS);
		output.push_back("." ICALL_PREFIX);
		output.push_back(itype.name);
		output.push_back(SINGLETON_ICALL_SUFFIX "();\n");
	} else {
		// Add member fields

		output.push_back(MEMBER_BEGIN "private const string " BINDINGS_NATIVE_NAME_FIELD " = \"");
		output.push_back(itype.name);
		output.push_back("\";\n");

		// Only the base class stores the pointer to the native object
		// This pointer is expected to be and must be of type Object*
		if (!is_derived_type) {
			output.push_back(MEMBER_BEGIN "private bool disposed = false;\n");
			output.push_back(INDENT2 "internal IntPtr " BINDINGS_PTR_FIELD ";\n");
			output.push_back(INDENT2 "internal bool " CS_FIELD_MEMORYOWN ";\n");
		}

		// Add default constructor
		if (itype.is_instantiable) {
			output.push_back(MEMBER_BEGIN "public ");
			output.push_back(itype.proxy_name);
			output.push_back("() : this(");
			output.push_back(itype.memory_own ? "true" : "false");

			// The default constructor may also be called by the engine when instancing existing native objects
			// The engine will initialize the pointer field of the managed side before calling the constructor
			// This is why we only allocate a new native object from the constructor if the pointer field is not set
			output.push_back(")\n" OPEN_BLOCK_L2 "if (" BINDINGS_PTR_FIELD " == IntPtr.Zero)\n" INDENT4 BINDINGS_PTR_FIELD " = ");
			output.push_back(itype.api_type == ClassDB::API_EDITOR ? CS_CLASS_NATIVECALLS_EDITOR : CS_CLASS_NATIVECALLS);
			output.push_back("." + ctor_method);
			output.push_back("(this);\n" CLOSE_BLOCK_L2);
		} else {
			// Hide the constructor
			output.push_back(MEMBER_BEGIN "internal ");
			output.push_back(itype.proxy_name);
			output.push_back("() {}\n");
		}

		// Add.. em.. trick constructor. Sort of.
		output.push_back(MEMBER_BEGIN "internal ");
		output.push_back(itype.proxy_name);
		if (is_derived_type) {
			output.push_back("(bool " CS_FIELD_MEMORYOWN ") : base(" CS_FIELD_MEMORYOWN ") {}\n");
		} else {
			output.push_back("(bool " CS_FIELD_MEMORYOWN ")\n" OPEN_BLOCK_L2
							 "this." CS_FIELD_MEMORYOWN " = " CS_FIELD_MEMORYOWN ";\n" CLOSE_BLOCK_L2);
		}

		// Add methods

		if (!is_derived_type) {
			output.push_back(MEMBER_BEGIN "public IntPtr NativeInstance\n" OPEN_BLOCK_L2
										  "get { return " BINDINGS_PTR_FIELD "; }\n" CLOSE_BLOCK_L2);

			output.push_back(MEMBER_BEGIN "internal static IntPtr " CS_SMETHOD_GETINSTANCE "(Object instance)\n" OPEN_BLOCK_L2
										  "return instance == null ? IntPtr.Zero : instance." BINDINGS_PTR_FIELD ";\n" CLOSE_BLOCK_L2);
		}

		if (!is_derived_type) {
			// Add destructor
			output.push_back(MEMBER_BEGIN "~");
			output.push_back(itype.proxy_name);
			output.push_back("()\n" OPEN_BLOCK_L2 "Dispose(false);\n" CLOSE_BLOCK_L2);

			// Add the Dispose from IDisposable
			output.push_back(MEMBER_BEGIN "public void Dispose()\n" OPEN_BLOCK_L2 "Dispose(true);\n" INDENT3 "GC.SuppressFinalize(this);\n" CLOSE_BLOCK_L2);

			// Add the virtual Dispose
			output.push_back(MEMBER_BEGIN "public virtual void Dispose(bool disposing)\n" OPEN_BLOCK_L2
										  "if (disposed) return;\n" INDENT3
										  "if (" BINDINGS_PTR_FIELD " != IntPtr.Zero)\n" OPEN_BLOCK_L3
										  "if (" CS_FIELD_MEMORYOWN ")\n" OPEN_BLOCK_L4 CS_FIELD_MEMORYOWN
										  " = false;\n" INDENT5 CS_CLASS_NATIVECALLS "." ICALL_OBJECT_DTOR
										  "(" BINDINGS_PTR_FIELD ");\n" INDENT5 BINDINGS_PTR_FIELD
										  " = IntPtr.Zero;\n" CLOSE_BLOCK_L4 CLOSE_BLOCK_L3 INDENT3
										  "GC.SuppressFinalize(this);\n" INDENT3 "disposed = true;\n" CLOSE_BLOCK_L2);

			Map<String, TypeInterface>::Element *array_itype = builtin_types.find("Array");

			if (!array_itype) {
				ERR_PRINT("BUG: Array type interface not found!");
				return ERR_BUG;
			}

			Map<String, TypeInterface>::Element *object_itype = obj_types.find("Object");

			if (!object_itype) {
				ERR_PRINT("BUG: Object type interface not found!");
				return ERR_BUG;
			}

			output.push_back(MEMBER_BEGIN "public " CS_CLASS_SIGNALAWAITER " ToSignal(");
			output.push_back(object_itype->get().cs_type);
			output.push_back(" source, string signal)\n" OPEN_BLOCK_L2
							 "return new " CS_CLASS_SIGNALAWAITER "(source, signal, this);\n" CLOSE_BLOCK_L2);
		}
	}

	Map<String, String>::Element *extra_member = extra_members.find(itype.name);
	if (extra_member)
		output.push_back(extra_member->get());

	int method_bind_count = 0;
	for (const List<MethodInterface>::Element *E = itype.methods.front(); E; E = E->next()) {
		const MethodInterface &imethod = E->get();
		Error method_err = _generate_cs_method(itype, imethod, method_bind_count, output);
		if (method_err != OK) {
			ERR_EXPLAIN("Failed to generate method '" + imethod.name + "' for class '" + itype.name + "'");
			ERR_FAIL_V(method_err);
		}
	}

	if (itype.is_singleton) {
		InternalCall singleton_icall = InternalCall(itype.api_type, ICALL_PREFIX + itype.name + SINGLETON_ICALL_SUFFIX, "IntPtr");

		if (!find_icall_by_name(singleton_icall.name, custom_icalls))
			custom_icalls.push_back(singleton_icall);
	}

	if (itype.is_instantiable) {
		InternalCall ctor_icall = InternalCall(itype.api_type, ctor_method, "IntPtr", itype.proxy_name + " obj");

		if (!find_icall_by_name(ctor_icall.name, custom_icalls))
			custom_icalls.push_back(ctor_icall);
	}

	output.push_back(INDENT1 CLOSE_BLOCK CLOSE_BLOCK);

	return _save_file(p_output_file, output);
}

Error BindingsGenerator::_generate_cs_property(const BindingsGenerator::TypeInterface &p_itype, const DocData::PropertyDoc &p_prop_doc, List<String> &p_output) {

	const MethodInterface *setter = p_itype.find_method_by_name(p_prop_doc.setter);

	// Search it in base types too
	const TypeInterface *current_type = &p_itype;
	while (!setter && current_type->base_name.length()) {
		Map<String, TypeInterface>::Element *base_match = obj_types.find(current_type->base_name);
		ERR_FAIL_NULL_V(base_match, ERR_BUG);
		current_type = &base_match->get();
		setter = current_type->find_method_by_name(p_prop_doc.setter);
	}

	const MethodInterface *getter = p_itype.find_method_by_name(p_prop_doc.getter);

	// Search it in base types too
	current_type = &p_itype;
	while (!getter && current_type->base_name.length()) {
		Map<String, TypeInterface>::Element *base_match = obj_types.find(current_type->base_name);
		ERR_FAIL_NULL_V(base_match, ERR_BUG);
		current_type = &base_match->get();
		getter = current_type->find_method_by_name(p_prop_doc.getter);
	}

	ERR_FAIL_COND_V(!setter && !getter, ERR_BUG);

	bool is_valid = false;
	int prop_index = ClassDB::get_property_index(p_itype.name, p_prop_doc.name, &is_valid);
	ERR_FAIL_COND_V(!is_valid, ERR_BUG);

	if (setter) {
		int setter_argc = prop_index != -1 ? 2 : 1;
		ERR_FAIL_COND_V(setter->arguments.size() != setter_argc, ERR_BUG);
	}

	if (getter) {
		int getter_argc = prop_index != -1 ? 1 : 0;
		ERR_FAIL_COND_V(getter->arguments.size() != getter_argc, ERR_BUG);
	}

	if (getter && setter) {
		ERR_FAIL_COND_V(getter->return_type != setter->arguments.back()->get().type, ERR_BUG);
	}

	// Let's not trust PropertyDoc::type
	String proptype_name = getter ? getter->return_type : setter->arguments.back()->get().type;

	const TypeInterface *prop_itype = _get_type_by_name_or_null(proptype_name);
	if (!prop_itype) {
		// Try with underscore prefix
		prop_itype = _get_type_by_name_or_null("_" + proptype_name);
	}

	ERR_FAIL_NULL_V(prop_itype, ERR_BUG);

	String prop_proxy_name = escape_csharp_keyword(snake_to_pascal_case(p_prop_doc.name));

	// Prevent property and enclosing type from sharing the same name
	if (prop_proxy_name == p_itype.proxy_name) {
		if (verbose_output) {
			WARN_PRINTS("Name of property `" + prop_proxy_name + "` is ambiguous with the name of its class `" +
						p_itype.proxy_name + "`. Renaming property to `" + prop_proxy_name + "_`");
		}

		prop_proxy_name += "_";
	}

	if (p_prop_doc.description.size()) {
		p_output.push_back(MEMBER_BEGIN "/// <summary>\n");

		Vector<String> description_lines = p_prop_doc.description.split("\n");

		for (int i = 0; i < description_lines.size(); i++) {
			if (description_lines[i].size()) {
				p_output.push_back(INDENT2 "/// ");
				p_output.push_back(description_lines[i].strip_edges().xml_escape());
				p_output.push_back("\n");
			}
		}

		p_output.push_back(INDENT2 "/// </summary>");
	}

	p_output.push_back(MEMBER_BEGIN "public ");

	if (p_itype.is_singleton)
		p_output.push_back("static ");

	p_output.push_back(prop_itype->cs_type);
	p_output.push_back(" ");
	p_output.push_back(prop_proxy_name.replace("/", "__"));
	p_output.push_back("\n" INDENT2 OPEN_BLOCK);

	if (getter) {
		p_output.push_back(INDENT3 "get\n" OPEN_BLOCK_L3);
		p_output.push_back("return ");
		p_output.push_back(getter->proxy_name + "(");
		if (prop_index != -1)
			p_output.push_back(itos(prop_index));
		p_output.push_back(");\n" CLOSE_BLOCK_L3);
	}

	if (setter) {
		p_output.push_back(INDENT3 "set\n" OPEN_BLOCK_L3);
		p_output.push_back(setter->proxy_name + "(");
		if (prop_index != -1)
			p_output.push_back(itos(prop_index) + ", ");
		p_output.push_back("value);\n" CLOSE_BLOCK_L3);
	}

	p_output.push_back(CLOSE_BLOCK_L2);

	return OK;
}

Error BindingsGenerator::_generate_cs_method(const BindingsGenerator::TypeInterface &p_itype, const BindingsGenerator::MethodInterface &p_imethod, int &p_method_bind_count, List<String> &p_output) {

	const TypeInterface *return_type = _get_type_by_name_or_placeholder(p_imethod.return_type);

	String method_bind_field = "method_bind_" + itos(p_method_bind_count);

	String icall_params = method_bind_field + ", " + sformat(p_itype.cs_in, "this");
	String arguments_sig;
	String cs_in_statements;

	List<String> default_args_doc;

	// Retrieve information from the arguments
	for (const List<ArgumentInterface>::Element *F = p_imethod.arguments.front(); F; F = F->next()) {
		const ArgumentInterface &iarg = F->get();
		const TypeInterface *arg_type = _get_type_by_name_or_placeholder(iarg.type);

		// Add the current arguments to the signature
		// If the argument has a default value which is not a constant, we will make it Nullable
		{
			if (F != p_imethod.arguments.front())
				arguments_sig += ", ";

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				arguments_sig += "Nullable<";

			arguments_sig += arg_type->cs_type;

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				arguments_sig += "> ";
			else
				arguments_sig += " ";

			arguments_sig += iarg.name;

			if (iarg.default_argument.size()) {
				if (iarg.def_param_mode != ArgumentInterface::CONSTANT)
					arguments_sig += " = null";
				else
					arguments_sig += " = " + sformat(iarg.default_argument, arg_type->cs_type);
			}
		}

		icall_params += ", ";

		if (iarg.default_argument.size() && iarg.def_param_mode != ArgumentInterface::CONSTANT) {
			// The default value of an argument must be constant. Otherwise we make it Nullable and do the following:
			// Type arg_in = arg.HasValue ? arg.Value : <non-const default value>;
			String arg_in = iarg.name;
			arg_in += "_in";

			cs_in_statements += arg_type->cs_type;
			cs_in_statements += " ";
			cs_in_statements += arg_in;
			cs_in_statements += " = ";
			cs_in_statements += iarg.name;

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				cs_in_statements += ".HasValue ? ";
			else
				cs_in_statements += " != null ? ";

			cs_in_statements += iarg.name;

			if (iarg.def_param_mode == ArgumentInterface::NULLABLE_VAL)
				cs_in_statements += ".Value : ";
			else
				cs_in_statements += " : ";

			String def_arg = sformat(iarg.default_argument, arg_type->cs_type);

			cs_in_statements += def_arg;
			cs_in_statements += ";\n" INDENT3;

			icall_params += arg_type->cs_in.empty() ? arg_in : sformat(arg_type->cs_in, arg_in);

			default_args_doc.push_back(INDENT2 "/// <param name=\"" + iarg.name + "\">If the param is null, then the default value is " + def_arg + "</param>\n");
		} else {
			icall_params += arg_type->cs_in.empty() ? iarg.name : sformat(arg_type->cs_in, iarg.name);
		}
	}

	// Generate method
	{
		if (!p_imethod.is_virtual && !p_imethod.requires_object_call) {
			p_output.push_back(MEMBER_BEGIN "private ");
			p_output.push_back(p_itype.is_singleton ? "static IntPtr " : "IntPtr ");
			p_output.push_back(method_bind_field + " = " CS_CLASS_NATIVECALLS "." ICALL_GET_METHODBIND "(" BINDINGS_NATIVE_NAME_FIELD ", \"");
			p_output.push_back(p_imethod.name);
			p_output.push_back("\");\n");
		}

		if (p_imethod.method_doc && p_imethod.method_doc->description.size()) {
			p_output.push_back(MEMBER_BEGIN "/// <summary>\n");

			Vector<String> description_lines = p_imethod.method_doc->description.split("\n");

			for (int i = 0; i < description_lines.size(); i++) {
				if (description_lines[i].size()) {
					p_output.push_back(INDENT2 "/// ");
					p_output.push_back(description_lines[i].strip_edges().xml_escape());
					p_output.push_back("\n");
				}
			}

			for (List<String>::Element *E = default_args_doc.front(); E; E = E->next()) {
				p_output.push_back(E->get().xml_escape());
			}

			p_output.push_back(INDENT2 "/// </summary>");
		}

		if (!p_imethod.is_internal) {
			p_output.push_back(MEMBER_BEGIN "[GodotMethod(\"");
			p_output.push_back(p_imethod.name);
			p_output.push_back("\")]");
		}

		p_output.push_back(MEMBER_BEGIN);
		p_output.push_back(p_imethod.is_internal ? "internal " : "public ");

		if (p_itype.is_singleton) {
			p_output.push_back("static ");
		} else if (p_imethod.is_virtual) {
			p_output.push_back("virtual ");
		}

		p_output.push_back(return_type->cs_type + " ");
		p_output.push_back(p_imethod.proxy_name + "(");
		p_output.push_back(arguments_sig + ")\n" OPEN_BLOCK_L2);

		if (p_imethod.is_virtual) {
			// Godot virtual method must be overridden, therefore we return a default value by default.

			if (return_type->name == "void") {
				p_output.push_back("return;\n" CLOSE_BLOCK_L2);
			} else {
				p_output.push_back("return default(");
				p_output.push_back(return_type->cs_type);
				p_output.push_back(");\n" CLOSE_BLOCK_L2);
			}

			return OK; // Won't increment method bind count
		}

		if (p_imethod.requires_object_call) {
			// Fallback to Godot's object.Call(string, params)

			p_output.push_back(CS_METHOD_CALL "(\"");
			p_output.push_back(p_imethod.name);
			p_output.push_back("\"");

			for (const List<ArgumentInterface>::Element *F = p_imethod.arguments.front(); F; F = F->next()) {
				p_output.push_back(", ");
				p_output.push_back(F->get().name);
			}

			p_output.push_back(");\n" CLOSE_BLOCK_L2);

			return OK; // Won't increment method bind count
		}

		const Map<const MethodInterface *, const InternalCall *>::Element *match = method_icalls_map.find(&p_imethod);
		ERR_FAIL_NULL_V(match, ERR_BUG);

		const InternalCall *im_icall = match->value();

		String im_call = im_icall->editor_only ? CS_CLASS_NATIVECALLS_EDITOR : CS_CLASS_NATIVECALLS;
		im_call += "." + im_icall->name + "(" + icall_params + ");\n";

		if (p_imethod.arguments.size())
			p_output.push_back(cs_in_statements);

		if (return_type->name == "void") {
			p_output.push_back(im_call);
		} else if (return_type->cs_out.empty()) {
			p_output.push_back("return " + im_call);
		} else {
			p_output.push_back(return_type->im_type_out);
			p_output.push_back(" " LOCAL_RET " = ");
			p_output.push_back(im_call);
			p_output.push_back(INDENT3);
			p_output.push_back(sformat(return_type->cs_out, LOCAL_RET) + "\n");
		}

		p_output.push_back(CLOSE_BLOCK_L2);
	}

	p_method_bind_count++;
	return OK;
}

Error BindingsGenerator::generate_glue(const String &p_output_dir) {

	verbose_output = true;

	bool dir_exists = DirAccess::exists(p_output_dir);
	ERR_EXPLAIN("The output directory does not exist.");
	ERR_FAIL_COND_V(!dir_exists, ERR_FILE_BAD_PATH);

	List<String> output;

	output.push_back("#include \"" GLUE_HEADER_FILE "\"\n"
					 "\n");

	generated_icall_funcs.clear();

	for (Map<String, TypeInterface>::Element *type_elem = obj_types.front(); type_elem; type_elem = type_elem->next()) {
		const TypeInterface &itype = type_elem->get();

		List<InternalCall> &custom_icalls = itype.api_type == ClassDB::API_EDITOR ? editor_custom_icalls : core_custom_icalls;

		OS::get_singleton()->print(String("Generating " + itype.name + "...\n").utf8());

		String ctor_method(ICALL_PREFIX + itype.proxy_name + "_Ctor");

		for (const List<MethodInterface>::Element *E = itype.methods.front(); E; E = E->next()) {
			const MethodInterface &imethod = E->get();
			Error method_err = _generate_glue_method(itype, imethod, output);
			if (method_err != OK) {
				ERR_EXPLAIN("Failed to generate method '" + imethod.name + "' for class '" + itype.name + "'");
				ERR_FAIL_V(method_err);
			}
		}

		if (itype.is_singleton) {
			String singleton_icall_name = ICALL_PREFIX + itype.name + SINGLETON_ICALL_SUFFIX;
			InternalCall singleton_icall = InternalCall(itype.api_type, singleton_icall_name, "IntPtr");

			if (!find_icall_by_name(singleton_icall.name, custom_icalls))
				custom_icalls.push_back(singleton_icall);

			output.push_back("Object* ");
			output.push_back(singleton_icall_name);
			output.push_back("() " OPEN_BLOCK "\treturn ProjectSettings::get_singleton()->get_singleton_object(\"");
			output.push_back(itype.proxy_name);
			output.push_back("\");\n" CLOSE_BLOCK "\n");
		}

		if (itype.is_instantiable) {
			InternalCall ctor_icall = InternalCall(itype.api_type, ctor_method, "IntPtr", itype.proxy_name + " obj");

			if (!find_icall_by_name(ctor_icall.name, custom_icalls))
				custom_icalls.push_back(ctor_icall);

			output.push_back("Object* ");
			output.push_back(ctor_method);
			output.push_back("(MonoObject* obj) " OPEN_BLOCK
							 "\t" C_MACRO_OBJECT_CONSTRUCT "(instance, \"");
			output.push_back(itype.name);
			output.push_back("\");\n"
							 "\t" C_METHOD_TIE_MANAGED_TO_UNMANAGED "(obj, instance);\n"
							 "\treturn instance;\n" CLOSE_BLOCK "\n");
		}
	}

	output.push_back("namespace GodotSharpBindings\n" OPEN_BLOCK);
	output.push_back("uint64_t get_core_api_hash() { return ");
	output.push_back(itos(GDMono::get_singleton()->get_api_core_hash()) + "; }\n");
	output.push_back("#ifdef TOOLS_ENABLED\n"
					 "uint64_t get_editor_api_hash() { return ");
	output.push_back(itos(GDMono::get_singleton()->get_api_editor_hash()) +
					 "; }\n#endif // TOOLS_ENABLED\n");
	output.push_back("void register_generated_icalls() " OPEN_BLOCK);

#define ADD_INTERNAL_CALL_REGISTRATION(m_icall)                                                     \
	{                                                                                               \
		output.push_back("\tmono_add_internal_call(");                                              \
		output.push_back("\"" BINDINGS_NAMESPACE ".");                                              \
		output.push_back(m_icall.editor_only ? CS_CLASS_NATIVECALLS_EDITOR : CS_CLASS_NATIVECALLS); \
		output.push_back("::");                                                                     \
		output.push_back(m_icall.name);                                                             \
		output.push_back("\", (void*)");                                                            \
		output.push_back(m_icall.name);                                                             \
		output.push_back(");\n");                                                                   \
	}

	bool tools_sequence = false;
	for (const List<InternalCall>::Element *E = core_custom_icalls.front(); E; E = E->next()) {

		if (tools_sequence) {
			if (!E->get().editor_only) {
				tools_sequence = false;
				output.push_back("#endif\n");
			}
		} else {
			if (E->get().editor_only) {
				output.push_back("#ifdef TOOLS_ENABLED\n");
				tools_sequence = true;
			}
		}

		ADD_INTERNAL_CALL_REGISTRATION(E->get());
	}

	if (tools_sequence) {
		tools_sequence = false;
		output.push_back("#endif\n");
	}

	output.push_back("#ifdef TOOLS_ENABLED\n");
	for (const List<InternalCall>::Element *E = editor_custom_icalls.front(); E; E = E->next())
		ADD_INTERNAL_CALL_REGISTRATION(E->get());
	output.push_back("#endif // TOOLS_ENABLED\n");

	for (const List<InternalCall>::Element *E = method_icalls.front(); E; E = E->next()) {
		if (tools_sequence) {
			if (!E->get().editor_only) {
				tools_sequence = false;
				output.push_back("#endif\n");
			}
		} else {
			if (E->get().editor_only) {
				output.push_back("#ifdef TOOLS_ENABLED\n");
				tools_sequence = true;
			}
		}

		ADD_INTERNAL_CALL_REGISTRATION(E->get());
	}

	if (tools_sequence) {
		tools_sequence = false;
		output.push_back("#endif\n");
	}

#undef ADD_INTERNAL_CALL_REGISTRATION

	output.push_back(CLOSE_BLOCK "}\n");

	Error save_err = _save_file(path_join(p_output_dir, "mono_glue.gen.cpp"), output);
	if (save_err != OK)
		return save_err;

	OS::get_singleton()->print("Mono glue generated successfully\n");

	return OK;
}

Error BindingsGenerator::_save_file(const String &p_path, const List<String> &p_content) {

	FileAccessRef file = FileAccess::open(p_path, FileAccess::WRITE);

	ERR_EXPLAIN("Cannot open file: " + p_path);
	ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);

	for (const List<String>::Element *E = p_content.front(); E; E = E->next()) {
		file->store_string(E->get());
	}

	file->close();

	return OK;
}

Error BindingsGenerator::_generate_glue_method(const BindingsGenerator::TypeInterface &p_itype, const BindingsGenerator::MethodInterface &p_imethod, List<String> &p_output) {

	if (p_imethod.is_virtual)
		return OK; // Ignore

	bool ret_void = p_imethod.return_type == "void";

	const TypeInterface *return_type = _get_type_by_name_or_placeholder(p_imethod.return_type);

	String argc_str = itos(p_imethod.arguments.size());

	String c_func_sig = "MethodBind* " CS_PARAM_METHODBIND ", " + p_itype.c_type_in + " " CS_PARAM_INSTANCE;
	String c_in_statements;
	String c_args_var_content;

	// Get arguments information
	int i = 0;
	for (const List<ArgumentInterface>::Element *F = p_imethod.arguments.front(); F; F = F->next()) {
		const ArgumentInterface &iarg = F->get();
		const TypeInterface *arg_type = _get_type_by_name_or_placeholder(iarg.type);

		String c_param_name = "arg" + itos(i + 1);

		if (p_imethod.is_vararg) {
			if (i < p_imethod.arguments.size() - 1) {
				c_in_statements += sformat(arg_type->c_in.size() ? arg_type->c_in : TypeInterface::DEFAULT_VARARG_C_IN, "Variant", c_param_name);
				c_in_statements += "\t" C_LOCAL_PTRCALL_ARGS ".set(0, ";
				c_in_statements += sformat("&%s_in", c_param_name);
				c_in_statements += ");\n";
			}
		} else {
			if (i > 0)
				c_args_var_content += ", ";
			if (arg_type->c_in.size())
				c_in_statements += sformat(arg_type->c_in, arg_type->c_type, c_param_name);
			c_args_var_content += sformat(arg_type->c_arg_in, c_param_name);
		}

		c_func_sig += ", ";
		c_func_sig += arg_type->c_type_in;
		c_func_sig += " ";
		c_func_sig += c_param_name;

		i++;
	}

	const Map<const MethodInterface *, const InternalCall *>::Element *match = method_icalls_map.find(&p_imethod);
	ERR_FAIL_NULL_V(match, ERR_BUG);

	const InternalCall *im_icall = match->value();
	String icall_method = im_icall->name;

	if (!generated_icall_funcs.find(im_icall)) {
		generated_icall_funcs.push_back(im_icall);

		if (im_icall->editor_only)
			p_output.push_back("#ifdef TOOLS_ENABLED\n");

		// Generate icall function

		p_output.push_back(ret_void ? "void " : return_type->c_type_out + " ");
		p_output.push_back(icall_method);
		p_output.push_back("(");
		p_output.push_back(c_func_sig);
		p_output.push_back(") " OPEN_BLOCK);

		String fail_ret = ret_void ? "" : ", " + (return_type->c_type_out.ends_with("*") ? "NULL" : return_type->c_type_out + "()");

		if (!ret_void) {
			String ptrcall_return_type;
			String initialization;

			if (return_type->is_object_type) {
				ptrcall_return_type = return_type->is_reference ? "Ref<Reference>" : return_type->c_type;
				initialization = return_type->is_reference ? "" : " = NULL";
			} else {
				ptrcall_return_type = return_type->c_type;
			}

			p_output.push_back("\t" + ptrcall_return_type);
			p_output.push_back(" " LOCAL_RET);
			p_output.push_back(initialization + ";\n");
			p_output.push_back("\tERR_FAIL_NULL_V(" CS_PARAM_INSTANCE);
			p_output.push_back(fail_ret);
			p_output.push_back(");\n");
		} else {
			p_output.push_back("\tERR_FAIL_NULL(" CS_PARAM_INSTANCE ");\n");
		}

		if (p_imethod.arguments.size()) {
			if (p_imethod.is_vararg) {
				String err_fail_macro = ret_void ? "ERR_FAIL_COND" : "ERR_FAIL_COND_V";
				String vararg_arg = "arg" + argc_str;
				String real_argc_str = itos(p_imethod.arguments.size() - 1); // Arguments count without vararg

				p_output.push_back("\tVector<Variant> varargs;\n"
								   "\tint vararg_length = mono_array_length(");
				p_output.push_back(vararg_arg);
				p_output.push_back(");\n\tint total_length = ");
				p_output.push_back(real_argc_str);
				p_output.push_back(" + vararg_length;\n\t");
				p_output.push_back(err_fail_macro);
				p_output.push_back("(varargs.resize(vararg_length) != OK");
				p_output.push_back(fail_ret);
				p_output.push_back(");\n\tVector<Variant*> " C_LOCAL_PTRCALL_ARGS ";\n\t");
				p_output.push_back(err_fail_macro);
				p_output.push_back("(call_args.resize(total_length) != OK");
				p_output.push_back(fail_ret);
				p_output.push_back(");\n");
				p_output.push_back(c_in_statements);
				p_output.push_back("\tfor (int i = 0; i < vararg_length; i++) " OPEN_BLOCK
								   "\t\tMonoObject* elem = mono_array_get(");
				p_output.push_back(vararg_arg);
				p_output.push_back(", MonoObject*, i);\n"
								   "\t\tvarargs.set(i, GDMonoMarshal::mono_object_to_variant(elem));\n"
								   "\t\t" C_LOCAL_PTRCALL_ARGS ".set(");
				p_output.push_back(real_argc_str);
				p_output.push_back(" + i, &varargs[i]);\n\t" CLOSE_BLOCK);
			} else {
				p_output.push_back(c_in_statements);
				p_output.push_back("\tconst void* " C_LOCAL_PTRCALL_ARGS "[");
				p_output.push_back(argc_str + "] = { ");
				p_output.push_back(c_args_var_content + " };\n");
			}
		}

		if (p_imethod.is_vararg) {
			p_output.push_back("\tVariant::CallError vcall_error;\n\t");

			if (!ret_void)
				p_output.push_back(LOCAL_RET " = ");

			p_output.push_back(CS_PARAM_METHODBIND "->call(" CS_PARAM_INSTANCE ", ");
			p_output.push_back(p_imethod.arguments.size() ? "(const Variant**)" C_LOCAL_PTRCALL_ARGS ".ptr()" : "NULL");
			p_output.push_back(", total_length, vcall_error);\n");
		} else {
			p_output.push_back("\t" CS_PARAM_METHODBIND "->ptrcall(" CS_PARAM_INSTANCE ", ");
			p_output.push_back(p_imethod.arguments.size() ? C_LOCAL_PTRCALL_ARGS ", " : "NULL, ");
			p_output.push_back(!ret_void ? "&" LOCAL_RET ");\n" : "NULL);\n");
		}

		if (!ret_void) {
			if (return_type->c_out.empty())
				p_output.push_back("\treturn " LOCAL_RET ";\n");
			else
				p_output.push_back(sformat(return_type->c_out, return_type->c_type_out, LOCAL_RET, return_type->name));
		}

		p_output.push_back(CLOSE_BLOCK "\n");

		if (im_icall->editor_only)
			p_output.push_back("#endif // TOOLS_ENABLED\n");
	}

	return OK;
}

const BindingsGenerator::TypeInterface *BindingsGenerator::_get_type_by_name_or_null(const String &p_name) {

	const Map<String, TypeInterface>::Element *match = builtin_types.find(p_name);

	if (match)
		return &match->get();

	match = obj_types.find(p_name);

	if (match)
		return &match->get();

	return NULL;
}

const BindingsGenerator::TypeInterface *BindingsGenerator::_get_type_by_name_or_placeholder(const String &p_name) {

	const TypeInterface *found = _get_type_by_name_or_null(p_name);

	if (found)
		return found;

	ERR_PRINTS(String() + "Type not found. Creating placeholder: " + p_name);

	const Map<String, TypeInterface>::Element *match = placeholder_types.find(p_name);

	if (match)
		return &match->get();

	TypeInterface placeholder;
	TypeInterface::create_placeholder_type(placeholder, p_name);

	return &placeholder_types.insert(placeholder.name, placeholder)->get();
}

void BindingsGenerator::_populate_object_type_interfaces() {

	obj_types.clear();

	List<StringName> class_list;
	ClassDB::get_class_list(&class_list);
	class_list.sort_custom<StringName::AlphCompare>();

	StringName refclass_name = String("Reference");

	while (class_list.size()) {
		StringName type_cname = class_list.front()->get();

		ClassDB::APIType api_type = ClassDB::get_api_type(type_cname);

		if (api_type == ClassDB::API_NONE) {
			class_list.pop_front();
			continue;
		}

		TypeInterface itype = TypeInterface::create_object_type(type_cname, api_type);

		itype.base_name = ClassDB::get_parent_class(type_cname);
		itype.is_singleton = ProjectSettings::get_singleton()->has_singleton(itype.proxy_name);
		itype.is_instantiable = ClassDB::can_instance(type_cname) && !itype.is_singleton;
		itype.is_reference = ClassDB::is_parent_class(type_cname, refclass_name);
		itype.memory_own = itype.is_reference;

		if (!ClassDB::is_class_exposed(type_cname)) {
			if (verbose_output)
				WARN_PRINTS("Ignoring type " + String(type_cname) + " because it's not exposed");
			class_list.pop_front();
			continue;
		}

		itype.c_out = "\treturn ";
		itype.c_out += C_METHOD_UNMANAGED_GET_MANAGED;
		itype.c_out += itype.is_reference ? "(%1.ptr());\n" : "(%1);\n";

		itype.cs_in = itype.is_singleton ? BINDINGS_PTR_FIELD : "Object." CS_SMETHOD_GETINSTANCE "(%0)";

		itype.c_type = "Object*";
		itype.c_type_in = itype.c_type;
		itype.c_type_out = "MonoObject*";
		itype.cs_type = itype.proxy_name;
		itype.im_type_in = "IntPtr";
		itype.im_type_out = itype.proxy_name;

		List<MethodInfo> virtual_method_list;
		ClassDB::get_virtual_methods(type_cname, &virtual_method_list, true);

		List<MethodInfo> method_list;
		ClassDB::get_method_list(type_cname, &method_list, true);
		method_list.sort();

		for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {
			const MethodInfo &method_info = E->get();

			int argc = method_info.arguments.size();

			if (method_info.name.empty())
				continue;

			MethodInterface imethod;
			imethod.name = method_info.name;

			if (method_info.flags & METHOD_FLAG_VIRTUAL)
				imethod.is_virtual = true;

			PropertyInfo return_info = method_info.return_val;

			MethodBind *m = imethod.is_virtual ? NULL : ClassDB::get_method(type_cname, method_info.name);

			imethod.is_vararg = m && m->is_vararg();

			if (!m && !imethod.is_virtual) {
				if (virtual_method_list.find(method_info)) {
					// A virtual method without the virtual flag. This is a special case.

					// This type of method can only be found in Object derived types.
					ERR_FAIL_COND(!itype.is_object_type);

					// There is no method bind, so let's fallback to Godot's object.Call(string, params)
					imethod.requires_object_call = true;

					// The method Object.free is registered as a virtual method, but without the virtual flag.
					// This is because this method is not supposed to be overridden, but called.
					// We assume the return type is void.
					imethod.return_type = "void";

					// Actually, more methods like this may be added in the future,
					// which could actually will return something differnet.
					// Let's put this to notify us if that ever happens.
					if (itype.name != "Object" || imethod.name != "free") {
						if (verbose_output) {
							WARN_PRINTS("Notification: New unexpected virtual non-overridable method found.\n"
										"We only expected Object.free, but found " +
										itype.name + "." + imethod.name);
						}
					}
				} else {
					ERR_PRINTS("Missing MethodBind for non-virtual method: " + itype.name + "." + imethod.name);
				}
			} else if (return_info.type == Variant::INT && return_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
				//imethod.return_type = return_info.class_name;
				imethod.return_type = "int";
			} else if (return_info.class_name != StringName()) {
				imethod.return_type = return_info.class_name;
			} else if (return_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
				imethod.return_type = return_info.hint_string;
			} else if (return_info.type == Variant::NIL && return_info.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
				imethod.return_type = "Variant";
			} else if (return_info.type == Variant::NIL) {
				imethod.return_type = "void";
			} else {
				imethod.return_type = Variant::get_type_name(return_info.type);
			}

			if (!itype.requires_collections && imethod.return_type == "Dictionary")
				itype.requires_collections = true;

			for (int i = 0; i < argc; i++) {
				PropertyInfo arginfo = method_info.arguments[i];

				ArgumentInterface iarg;
				iarg.name = arginfo.name;

				if (arginfo.type == Variant::INT && arginfo.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
					//iarg.type = arginfo.class_name;
					iarg.type = "int";
				} else if (arginfo.class_name != StringName()) {
					iarg.type = arginfo.class_name;
				} else if (arginfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
					iarg.type = arginfo.hint_string;
				} else if (arginfo.type == Variant::NIL) {
					iarg.type = "Variant";
				} else {
					iarg.type = Variant::get_type_name(arginfo.type);
				}

				iarg.name = escape_csharp_keyword(snake_to_camel_case(iarg.name));

				if (!itype.requires_collections && iarg.type == "Dictionary")
					itype.requires_collections = true;

				if (m && m->has_default_argument(i)) {
					_default_argument_from_variant(m->get_default_argument(i), iarg);
				}

				imethod.add_argument(iarg);
			}

			if (imethod.is_vararg) {
				ArgumentInterface ivararg;
				ivararg.type = "VarArg";
				ivararg.name = "@args";
				imethod.add_argument(ivararg);
			}

			imethod.proxy_name = escape_csharp_keyword(snake_to_pascal_case(imethod.name));

			// Prevent naming the property and its enclosing type from sharing the same name
			if (imethod.proxy_name == itype.proxy_name) {
				if (verbose_output) {
					WARN_PRINTS("Name of method `" + imethod.proxy_name + "` is ambiguous with the name of its class `" +
								itype.proxy_name + "`. Renaming method to `" + imethod.proxy_name + "_`");
				}

				imethod.proxy_name += "_";
			}

			if (itype.class_doc) {
				for (int i = 0; i < itype.class_doc->methods.size(); i++) {
					if (itype.class_doc->methods[i].name == imethod.name) {
						imethod.method_doc = &itype.class_doc->methods[i];
						break;
					}
				}
			}

			if (!imethod.is_virtual && imethod.name[0] == '_') {
				const Vector<DocData::PropertyDoc> &properties = itype.class_doc->properties;

				for (int i = 0; i < properties.size(); i++) {
					const DocData::PropertyDoc &prop_doc = properties[i];

					if (prop_doc.getter == imethod.name || prop_doc.setter == imethod.name) {
						imethod.is_internal = true;
						itype.methods.push_back(imethod);
						break;
					}
				}
			} else {
				itype.methods.push_back(imethod);
			}
		}

		obj_types.insert(itype.name, itype);

		class_list.pop_front();
	}
}

void BindingsGenerator::_default_argument_from_variant(const Variant &p_val, ArgumentInterface &r_iarg) {

	r_iarg.default_argument = p_val;

	switch (p_val.get_type()) {
		case Variant::NIL:
			if (ClassDB::class_exists(r_iarg.type)) {
				// Object type
				r_iarg.default_argument = "null";
			} else {
				// Variant
				r_iarg.default_argument = "null";
			}
			break;
		// Atomic types
		case Variant::BOOL:
			r_iarg.default_argument = bool(p_val) ? "true" : "false";
			break;
		case Variant::INT:
			break; // Keep it
		case Variant::REAL:
#ifndef REAL_T_IS_DOUBLE
			r_iarg.default_argument += "f";
#endif
			break;
		case Variant::STRING:
		case Variant::NODE_PATH:
			r_iarg.default_argument = "\"" + r_iarg.default_argument + "\"";
			break;
		case Variant::TRANSFORM:
			if (p_val.operator Transform() == Transform())
				r_iarg.default_argument.clear();
			r_iarg.default_argument = "new %s(" + r_iarg.default_argument + ")";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		case Variant::PLANE:
		case Variant::RECT3:
		case Variant::COLOR:
			r_iarg.default_argument = "new Color(1, 1, 1, 1)";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
			r_iarg.default_argument = "new %s" + r_iarg.default_argument;
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		case Variant::OBJECT:
			if (p_val.is_zero()) {
				r_iarg.default_argument = "null";
				break;
			}
		case Variant::DICTIONARY:
		case Variant::_RID:
			r_iarg.default_argument = "new %s()";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_REF;
			break;
		case Variant::ARRAY:
		case Variant::POOL_BYTE_ARRAY:
		case Variant::POOL_INT_ARRAY:
		case Variant::POOL_REAL_ARRAY:
		case Variant::POOL_STRING_ARRAY:
		case Variant::POOL_VECTOR2_ARRAY:
		case Variant::POOL_VECTOR3_ARRAY:
		case Variant::POOL_COLOR_ARRAY:
			r_iarg.default_argument = "new %s {}";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_REF;
			break;
		case Variant::TRANSFORM2D:
		case Variant::BASIS:
		case Variant::QUAT:
			r_iarg.default_argument = Variant::get_type_name(p_val.get_type()) + ".Identity";
			r_iarg.def_param_mode = ArgumentInterface::NULLABLE_VAL;
			break;
		default: {}
	}

	if (r_iarg.def_param_mode == ArgumentInterface::CONSTANT && r_iarg.type == "Variant" && r_iarg.default_argument != "null")
		r_iarg.def_param_mode = ArgumentInterface::NULLABLE_REF;
}

void BindingsGenerator::_populate_builtin_type_interfaces() {

	builtin_types.clear();

	TypeInterface itype;

#define INSERT_STRUCT_TYPE(m_type, m_type_in)                                                         \
	{                                                                                                 \
		itype = TypeInterface::create_value_type(#m_type);                                            \
		itype.c_in = "\tMARSHALLED_IN(" #m_type ", %1, %1_in);\n";                                    \
		itype.c_out = "\tMARSHALLED_OUT(" #m_type ", %1, ret_out)\n"                                  \
					  "\treturn mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(%2), ret_out);\n"; \
		itype.c_arg_in = "&%s_in";                                                                    \
		itype.c_type_in = m_type_in;                                                                  \
		itype.cs_in = "ref %s";                                                                       \
		itype.cs_out = "return (" #m_type ")%0;";                                                     \
		itype.im_type_out = "object";                                                                 \
		builtin_types.insert(#m_type, itype);                                                         \
	}

	INSERT_STRUCT_TYPE(Vector2, "real_t*")
	INSERT_STRUCT_TYPE(Rect2, "real_t*")
	INSERT_STRUCT_TYPE(Transform2D, "real_t*")
	INSERT_STRUCT_TYPE(Vector3, "real_t*")
	INSERT_STRUCT_TYPE(Basis, "real_t*")
	INSERT_STRUCT_TYPE(Quat, "real_t*")
	INSERT_STRUCT_TYPE(Transform, "real_t*")
	INSERT_STRUCT_TYPE(Rect3, "real_t*")
	INSERT_STRUCT_TYPE(Color, "real_t*")
	INSERT_STRUCT_TYPE(Plane, "real_t*")

#undef INSERT_STRUCT_TYPE

#define INSERT_PRIMITIVE_TYPE(m_type)                      \
	{                                                      \
		itype = TypeInterface::create_value_type(#m_type); \
		itype.c_arg_in = "&%s";                            \
		itype.c_type_in = #m_type;                         \
		itype.c_type_out = #m_type;                        \
		itype.im_type_in = #m_type;                        \
		itype.im_type_out = #m_type;                       \
		builtin_types.insert(#m_type, itype);              \
	}

	INSERT_PRIMITIVE_TYPE(bool)
	//INSERT_PRIMITIVE_TYPE(int)

	// int
	itype = TypeInterface::create_value_type("int");
	itype.c_arg_in = "&%s_in";
	//* ptrcall only supports int64_t and uint64_t
	itype.c_in = "\t%0 %1_in = (%0)%1;\n";
	itype.c_out = "\treturn (%0)%1;\n";
	itype.c_type = "int64_t";
	//*/
	itype.c_type_in = itype.name;
	itype.c_type_out = itype.name;
	itype.im_type_in = itype.name;
	itype.im_type_out = itype.name;
	builtin_types.insert(itype.name, itype);

#undef INSERT_PRIMITIVE_TYPE

	// real_t
	itype = TypeInterface();
#ifdef REAL_T_IS_DOUBLE
	itype.name = "double";
#else
	itype.name = "float";
#endif
	itype.proxy_name = itype.name;
	itype.c_arg_in = "&%s_in";
	//* ptrcall only supports double
	itype.c_in = "\t%0 %1_in = (%0)%1;\n";
	itype.c_out = "\treturn (%0)%1;\n";
	itype.c_type = "double";
	//*/
	itype.c_type_in = "real_t";
	itype.c_type_out = "real_t";
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.name, itype);

	// String
	itype = TypeInterface();
	itype.name = "String";
	itype.proxy_name = "string";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MONOSTR_TO_GODOT "(%1);\n";
	itype.c_out = "\treturn " C_METHOD_MONOSTR_FROM_GODOT "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = itype.name;
	itype.c_type_in = "MonoString*";
	itype.c_type_out = "MonoString*";
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.name, itype);

	// NodePath
	itype = TypeInterface();
	itype.name = "NodePath";
	itype.proxy_name = "NodePath";
	itype.c_out = "\treturn memnew(NodePath(%1));\n";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type + "*";
	itype.c_type_out = itype.c_type + "*";
	itype.cs_type = itype.proxy_name;
	itype.cs_in = "NodePath." CS_SMETHOD_GETINSTANCE "(%0)";
	itype.cs_out = "return new NodePath(%0);";
	itype.im_type_in = "IntPtr";
	itype.im_type_out = "IntPtr";
	_populate_builtin_type(itype, Variant::NODE_PATH);
	extra_members.insert(itype.name, MEMBER_BEGIN "public NodePath() : this(string.Empty) {}\n" MEMBER_BEGIN "public NodePath(string path)\n" OPEN_BLOCK_L2
												  "this." BINDINGS_PTR_FIELD " = NativeCalls.godot_icall_NodePath_Ctor(path);\n" CLOSE_BLOCK_L2
														  MEMBER_BEGIN "public static implicit operator NodePath(string from)\n" OPEN_BLOCK_L2 "return new NodePath(from);\n" CLOSE_BLOCK_L2
																  MEMBER_BEGIN "public static implicit operator string(NodePath from)\n" OPEN_BLOCK_L2
												  "return NativeCalls." ICALL_PREFIX "NodePath_operator_String(NodePath." CS_SMETHOD_GETINSTANCE "(from));\n" CLOSE_BLOCK_L2);
	builtin_types.insert(itype.name, itype);

	// RID
	itype = TypeInterface();
	itype.name = "RID";
	itype.proxy_name = "RID";
	itype.c_out = "\treturn memnew(RID(%1));\n";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type + "*";
	itype.c_type_out = itype.c_type + "*";
	itype.cs_type = itype.proxy_name;
	itype.cs_in = "RID." CS_SMETHOD_GETINSTANCE "(%0)";
	itype.cs_out = "return new RID(%0);";
	itype.im_type_in = "IntPtr";
	itype.im_type_out = "IntPtr";
	_populate_builtin_type(itype, Variant::_RID);
	extra_members.insert(itype.name, MEMBER_BEGIN "internal RID()\n" OPEN_BLOCK_L2
												  "this." BINDINGS_PTR_FIELD " = IntPtr.Zero;\n" CLOSE_BLOCK_L2);
	builtin_types.insert(itype.name, itype);

	// Variant
	itype = TypeInterface();
	itype.name = "Variant";
	itype.proxy_name = "object";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MANAGED_TO_VARIANT "(%1);\n";
	itype.c_out = "\treturn " C_METHOD_MANAGED_FROM_VARIANT "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = itype.name;
	itype.c_type_in = "MonoObject*";
	itype.c_type_out = "MonoObject*";
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = "object";
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.name, itype);

	// VarArg (fictitious type to represent variable arguments)
	itype = TypeInterface();
	itype.name = "VarArg";
	itype.proxy_name = "object[]";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MONOARRAY_TO(Array) "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = "Array";
	itype.c_type_in = "MonoArray*";
	itype.cs_type = "params object[]";
	itype.im_type_in = "object[]";
	builtin_types.insert(itype.name, itype);

#define INSERT_ARRAY_FULL(m_name, m_type, m_proxy_t)                          \
	{                                                                         \
		itype = TypeInterface();                                              \
		itype.name = #m_name;                                                 \
		itype.proxy_name = #m_proxy_t "[]";                                   \
		itype.c_in = "\t%0 %1_in = " C_METHOD_MONOARRAY_TO(m_type) "(%1);\n"; \
		itype.c_out = "\treturn " C_METHOD_MONOARRAY_FROM(m_type) "(%1);\n";  \
		itype.c_arg_in = "&%s_in";                                            \
		itype.c_type = #m_type;                                               \
		itype.c_type_in = "MonoArray*";                                       \
		itype.c_type_out = "MonoArray*";                                      \
		itype.cs_type = itype.proxy_name;                                     \
		itype.im_type_in = itype.proxy_name;                                  \
		itype.im_type_out = itype.proxy_name;                                 \
		builtin_types.insert(itype.name, itype);                              \
	}

#define INSERT_ARRAY(m_type, m_proxy_t) INSERT_ARRAY_FULL(m_type, m_type, m_proxy_t)

	INSERT_ARRAY(Array, object);
	INSERT_ARRAY(PoolIntArray, int);
	INSERT_ARRAY_FULL(PoolByteArray, PoolByteArray, byte);

#ifdef REAL_T_IS_DOUBLE
	INSERT_ARRAY(PoolRealArray, double);
#else
	INSERT_ARRAY(PoolRealArray, float);
#endif

	INSERT_ARRAY(PoolStringArray, string);

	INSERT_ARRAY(PoolColorArray, Color);
	INSERT_ARRAY(PoolVector2Array, Vector2);
	INSERT_ARRAY(PoolVector3Array, Vector3);

#undef INSERT_ARRAY

	// Dictionary
	itype = TypeInterface();
	itype.name = "Dictionary";
	itype.proxy_name = "Dictionary<object, object>";
	itype.c_in = "\t%0 %1_in = " C_METHOD_MANAGED_TO_DICT "(%1);\n";
	itype.c_out = "\treturn " C_METHOD_MANAGED_FROM_DICT "(%1);\n";
	itype.c_arg_in = "&%s_in";
	itype.c_type = itype.name;
	itype.c_type_in = "MonoObject*";
	itype.c_type_out = "MonoObject*";
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.name, itype);

	// void (fictitious type to represent the return type of methods that do not return anything)
	itype = TypeInterface();
	itype.name = "void";
	itype.proxy_name = itype.name;
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type;
	itype.c_type_out = itype.c_type;
	itype.cs_type = itype.proxy_name;
	itype.im_type_in = itype.proxy_name;
	itype.im_type_out = itype.proxy_name;
	builtin_types.insert(itype.name, itype);

	// Error
	itype = TypeInterface();
	itype.name = "Error";
	itype.proxy_name = "Error";
	itype.c_type = itype.name;
	itype.c_type_in = itype.c_type;
	itype.c_type_out = itype.c_type;
	itype.cs_type = itype.proxy_name;
	itype.cs_in = "(int)%0";
	itype.cs_out = "return (Error)%s;";
	itype.im_type_in = "int";
	itype.im_type_out = "int";
	builtin_types.insert(itype.name, itype);
}

void BindingsGenerator::_populate_builtin_type(TypeInterface &r_itype, Variant::Type vtype) {

	Variant::CallError cerror;
	Variant v = Variant::construct(vtype, NULL, 0, cerror);

	List<MethodInfo> method_list;
	v.get_method_list(&method_list);
	method_list.sort();

	for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {
		MethodInfo &mi = E->get();
		MethodInterface imethod;

		imethod.name = mi.name;
		imethod.proxy_name = mi.name;

		for (int i = 0; i < mi.arguments.size(); i++) {
			ArgumentInterface iarg;
			PropertyInfo pi = mi.arguments[i];

			iarg.name = pi.name;

			if (pi.type == Variant::NIL)
				iarg.type = "Variant";
			else
				iarg.type = Variant::get_type_name(pi.type);

			if (!r_itype.requires_collections && iarg.type == "Dictionary")
				r_itype.requires_collections = true;

			if ((mi.default_arguments.size() - mi.arguments.size() + i) >= 0)
				_default_argument_from_variant(Variant::construct(pi.type, NULL, 0, cerror), iarg);

			imethod.add_argument(iarg);
		}

		if (mi.return_val.type == Variant::NIL) {
			if (mi.return_val.name != "")
				imethod.return_type = "Variant";
		} else {
			imethod.return_type = Variant::get_type_name(mi.return_val.type);
		}

		if (!r_itype.requires_collections && imethod.return_type == "Dictionary")
			r_itype.requires_collections = true;

		if (r_itype.class_doc) {
			for (int i = 0; i < r_itype.class_doc->methods.size(); i++) {
				if (r_itype.class_doc->methods[i].name == imethod.name) {
					imethod.method_doc = &r_itype.class_doc->methods[i];
					break;
				}
			}
		}

		r_itype.methods.push_back(imethod);
	}
}

BindingsGenerator::BindingsGenerator() {

	EditorHelp::generate_doc();

	_populate_object_type_interfaces();
	_populate_builtin_type_interfaces();
	_generate_header_icalls();

	for (Map<String, TypeInterface>::Element *E = obj_types.front(); E; E = E->next())
		_generate_method_icalls(E->get());

	_generate_method_icalls(builtin_types["NodePath"]);
	_generate_method_icalls(builtin_types["RID"]);
}

void BindingsGenerator::handle_cmdline_args(const List<String> &p_cmdline_args) {

	const int NUM_OPTIONS = 3;
	int options_left = NUM_OPTIONS;

	String mono_glue_option = "--generate-mono-glue";
	String cs_core_api_option = "--generate-cs-core-api";
	String cs_editor_api_option = "--generate-cs-editor-api";

	verbose_output = true;

	const List<String>::Element *elem = p_cmdline_args.front();

	while (elem && options_left) {

		if (elem->get() == mono_glue_option) {

			const List<String>::Element *path_elem = elem->next();

			if (path_elem) {
				if (get_singleton().generate_glue(path_elem->get()) != OK)
					ERR_PRINT("Mono glue generation failed");
				elem = elem->next();
			} else {
				ERR_PRINTS("--generate-mono-glue: No output directory specified");
			}

			--options_left;

		} else if (elem->get() == cs_core_api_option) {

			const List<String>::Element *path_elem = elem->next();

			if (path_elem) {
				if (get_singleton().generate_cs_core_project(path_elem->get()) != OK)
					ERR_PRINT("Generation of solution and C# project for the Core API failed");
				elem = elem->next();
			} else {
				ERR_PRINTS(cs_core_api_option + ": No output directory specified");
			}

			--options_left;

		} else if (elem->get() == cs_editor_api_option) {

			const List<String>::Element *path_elem = elem->next();

			if (path_elem) {
				if (path_elem->next()) {
					if (get_singleton().generate_cs_editor_project(path_elem->get(), path_elem->next()->get()) != OK)
						ERR_PRINT("Generation of solution and C# project for the Editor API failed");
					elem = path_elem->next();
				} else {
					ERR_PRINTS(cs_editor_api_option + ": No hint path for the Core API dll specified");
				}
			} else {
				ERR_PRINTS(cs_editor_api_option + ": No output directory specified");
			}

			--options_left;
		}

		elem = elem->next();
	}

	verbose_output = false;

	if (options_left != NUM_OPTIONS)
		exit(0);
}

#endif
