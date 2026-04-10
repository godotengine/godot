/**************************************************************************/
/*  gdscript_bytecode_serializer.h                                        */
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

#include "gdscript.h"
#include "gdscript_function.h"

class GDScriptBytecodeSerializer {
	static constexpr uint32_t MAGIC = 0x43424447; // "GDBC" little-endian
	static constexpr uint32_t FORMAT_VERSION = 3;

	// Binary encoding helpers.
	static void write_uint8(Vector<uint8_t> &buf, uint8_t val);
	static void write_uint32(Vector<uint8_t> &buf, uint32_t val);
	static void write_int32(Vector<uint8_t> &buf, int32_t val);
	static void write_string(Vector<uint8_t> &buf, const String &s);
	static void write_string_name(Vector<uint8_t> &buf, const StringName &s);
	static void write_variant(Vector<uint8_t> &buf, const Variant &v);
	static void write_data_type(Vector<uint8_t> &buf, const GDScriptDataType &dt);
	static void write_property_info(Vector<uint8_t> &buf, const PropertyInfo &pi);
	static void write_method_info(Vector<uint8_t> &buf, const MethodInfo &mi);
	static void write_function(Vector<uint8_t> &buf, const GDScriptFunction *fn);

	// Binary decoding helpers.
	static uint8_t read_uint8(const uint8_t *p_buf, int &ofs, int p_len);
	static uint32_t read_uint32(const uint8_t *p_buf, int &ofs, int p_len);
	static int32_t read_int32(const uint8_t *p_buf, int &ofs, int p_len);
	static String read_string(const uint8_t *p_buf, int &ofs, int p_len);
	static StringName read_string_name(const uint8_t *p_buf, int &ofs, int p_len);
	static Variant read_variant(const uint8_t *p_buf, int &ofs, int p_len);
	static GDScriptDataType read_data_type(const uint8_t *p_buf, int &ofs, int p_len);
	static PropertyInfo read_property_info(const uint8_t *p_buf, int &ofs, int p_len);
	static MethodInfo read_method_info(const uint8_t *p_buf, int &ofs, int p_len);
	static GDScriptFunction *read_function(const uint8_t *p_buf, int &ofs, int p_len, GDScript *p_script);
	static Ref<Script> resolve_script_reference(GDScript *p_script, const String &p_path, const String &p_fqcn);
	static void resolve_variant_script_references(GDScript *p_script, Variant &r_variant);
	static void resolve_data_type_script_references(GDScript *p_script, GDScriptDataType &r_data_type);
	static void resolve_method_info_script_references(GDScript *p_script, MethodInfo &r_method_info);
	static void resolve_function_script_references(GDScript *p_script, GDScriptFunction *p_function);
	static void resolve_script_references(GDScript *p_script);

	static void write_script_data(Vector<uint8_t> &buf, const GDScript *p_script);
	static Error read_script_data(const uint8_t *p_buf, int &ofs, int p_len, GDScript *p_script);

	// Text dump helpers.
	static String opcode_to_name(int p_opcode);
	static String variant_to_text(const Variant &v);
	static String data_type_to_text(const GDScriptDataType &dt);
	static String dump_function_text(const GDScriptFunction *fn, const String &p_indent);
	static String dump_script_data_text(const GDScript *p_script, const String &p_indent);

public:
	static Vector<uint8_t> serialize_script(const GDScript *p_script);
	static Error deserialize_script(const Vector<uint8_t> &p_data, GDScript *p_script);

	// Produce a human-readable text dump of the script's bytecode.
	static String dump_script_text(const GDScript *p_script);
};
