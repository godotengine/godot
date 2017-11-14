/*************************************************************************/
/*  glue_header.h                                                        */
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
#include "../csharp_script.h"
#include "../mono_gd/gd_mono_class.h"
#include "../mono_gd/gd_mono_internals.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../signal_awaiter_utils.h"

#include "bind/core_bind.h"
#include "class_db.h"
#include "engine.h"
#include "io/marshalls.h"
#include "object.h"
#include "os/os.h"
#include "reference.h"
#include "variant_parser.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

#define GODOTSHARP_INSTANCE_OBJECT(m_instance, m_type) \
	static ClassDB::ClassInfo *ci = NULL;              \
	if (!ci) {                                         \
		ci = ClassDB::classes.getptr(m_type);          \
	}                                                  \
	Object *m_instance = ci->creation_func();

void godot_icall_Object_Dtor(Object *ptr) {
	ERR_FAIL_NULL(ptr);
	_GodotSharp::get_singleton()->queue_dispose(ptr);
}

// -- ClassDB --

MethodBind *godot_icall_ClassDB_get_method(MonoString *p_type, MonoString *p_method) {
	StringName type(GDMonoMarshal::mono_string_to_godot(p_type));
	StringName method(GDMonoMarshal::mono_string_to_godot(p_method));
	return ClassDB::get_method(type, method);
}

// -- SignalAwaiter --

Error godot_icall_Object_connect_signal_awaiter(Object *p_source, MonoString *p_signal, Object *p_target, MonoObject *p_awaiter) {
	String signal = GDMonoMarshal::mono_string_to_godot(p_signal);
	return SignalAwaiterUtils::connect_signal_awaiter(p_source, signal, p_target, p_awaiter);
}

// -- NodePath --

NodePath *godot_icall_NodePath_Ctor(MonoString *p_path) {
	return memnew(NodePath(GDMonoMarshal::mono_string_to_godot(p_path)));
}

void godot_icall_NodePath_Dtor(NodePath *p_ptr) {
	ERR_FAIL_NULL(p_ptr);
	_GodotSharp::get_singleton()->queue_dispose(p_ptr);
}

MonoString *godot_icall_NodePath_operator_String(NodePath *p_np) {
	return GDMonoMarshal::mono_string_from_godot(p_np->operator String());
}

MonoArray *godot_icall_String_md5_buffer(MonoString *p_str) {
	Vector<uint8_t> ret = GDMonoMarshal::mono_string_to_godot(p_str).md5_buffer();
	// TODO Check possible Array/Vector<uint8_t> problem?
	return GDMonoMarshal::Array_to_mono_array(Variant(ret));
}

// -- RID --

RID *godot_icall_RID_Ctor(Object *p_from) {
	Resource *res_from = Object::cast_to<Resource>(p_from);

	if (res_from)
		return memnew(RID(res_from->get_rid()));

	return memnew(RID);
}

void godot_icall_RID_Dtor(RID *p_ptr) {
	ERR_FAIL_NULL(p_ptr);
	_GodotSharp::get_singleton()->queue_dispose(p_ptr);
}

// -- String --

MonoString *godot_icall_String_md5_text(MonoString *p_str) {
	String ret = GDMonoMarshal::mono_string_to_godot(p_str).md5_text();
	return GDMonoMarshal::mono_string_from_godot(ret);
}

int godot_icall_String_rfind(MonoString *p_str, MonoString *p_what, int p_from) {
	String what = GDMonoMarshal::mono_string_to_godot(p_what);
	return GDMonoMarshal::mono_string_to_godot(p_str).rfind(what, p_from);
}

int godot_icall_String_rfindn(MonoString *p_str, MonoString *p_what, int p_from) {
	String what = GDMonoMarshal::mono_string_to_godot(p_what);
	return GDMonoMarshal::mono_string_to_godot(p_str).rfindn(what, p_from);
}

MonoArray *godot_icall_String_sha256_buffer(MonoString *p_str) {
	Vector<uint8_t> ret = GDMonoMarshal::mono_string_to_godot(p_str).sha256_buffer();
	return GDMonoMarshal::Array_to_mono_array(Variant(ret));
}

MonoString *godot_icall_String_sha256_text(MonoString *p_str) {
	String ret = GDMonoMarshal::mono_string_to_godot(p_str).sha256_text();
	return GDMonoMarshal::mono_string_from_godot(ret);
}

// -- Global Scope --

MonoObject *godot_icall_Godot_bytes2var(MonoArray *p_bytes) {
	Variant ret;
	PoolByteArray varr = GDMonoMarshal::mono_array_to_PoolByteArray(p_bytes);
	PoolByteArray::Read r = varr.read();
	Error err = decode_variant(ret, r.ptr(), varr.size(), NULL);
	if (err != OK) {
		ret = RTR("Not enough bytes for decoding bytes, or invalid format.");
	}
	return GDMonoMarshal::variant_to_mono_object(ret);
}

MonoObject *godot_icall_Godot_convert(MonoObject *p_what, int p_type) {
	Variant what = GDMonoMarshal::mono_object_to_variant(p_what);
	const Variant *args[1] = { &what };
	Variant::CallError ce;
	Variant ret = Variant::construct(Variant::Type(p_type), args, 1, ce);
	ERR_FAIL_COND_V(ce.error != Variant::CallError::CALL_OK, NULL);
	return GDMonoMarshal::variant_to_mono_object(ret);
}

int godot_icall_Godot_hash(MonoObject *p_var) {
	return GDMonoMarshal::mono_object_to_variant(p_var).hash();
}

MonoObject *godot_icall_Godot_instance_from_id(int p_instance_id) {
	return GDMonoUtils::unmanaged_get_managed(ObjectDB::get_instance(p_instance_id));
}

void godot_icall_Godot_print(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++)
		str += what[i].operator String();
	print_line(str);
}

void godot_icall_Godot_printerr(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++)
		str += what[i].operator String();
	OS::get_singleton()->printerr("%s\n", str.utf8().get_data());
}

void godot_icall_Godot_printraw(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++)
		str += what[i].operator String();
	OS::get_singleton()->print("%s", str.utf8().get_data());
}

void godot_icall_Godot_prints(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++) {
		if (i)
			str += " ";
		str += what[i].operator String();
	}
	print_line(str);
}

void godot_icall_Godot_printt(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++) {
		if (i)
			str += "\t";
		str += what[i].operator String();
	}
	print_line(str);
}

void godot_icall_Godot_seed(int p_seed) {
	Math::seed(p_seed);
}

MonoString *godot_icall_Godot_str(MonoArray *p_what) {
	String str;
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);

	for (int i = 0; i < what.size(); i++) {
		String os = what[i].operator String();

		if (i == 0)
			str = os;
		else
			str += os;
	}

	return GDMonoMarshal::mono_string_from_godot(str);
}

MonoObject *godot_icall_Godot_str2var(MonoString *p_str) {
	Variant ret;

	VariantParser::StreamString ss;
	ss.s = GDMonoMarshal::mono_string_to_godot(p_str);

	String errs;
	int line;
	Error err = VariantParser::parse(&ss, ret, errs, line);
	if (err != OK) {
		String err_str = "Parse error at line " + itos(line) + ": " + errs;
		ERR_PRINTS(err_str);
		ret = err_str;
	}

	return GDMonoMarshal::variant_to_mono_object(ret);
}

bool godot_icall_Godot_type_exists(MonoString *p_type) {
	return ClassDB::class_exists(GDMonoMarshal::mono_string_to_godot(p_type));
}

MonoArray *godot_icall_Godot_var2bytes(MonoObject *p_var) {
	Variant var = GDMonoMarshal::mono_object_to_variant(p_var);

	PoolByteArray barr;
	int len;
	Error err = encode_variant(var, NULL, len);
	ERR_EXPLAIN("Unexpected error encoding variable to bytes, likely unserializable type found (Object or RID).");
	ERR_FAIL_COND_V(err != OK, NULL);

	barr.resize(len);
	{
		PoolByteArray::Write w = barr.write();
		encode_variant(var, w.ptr(), len);
	}

	return GDMonoMarshal::PoolByteArray_to_mono_array(barr);
}

MonoString *godot_icall_Godot_var2str(MonoObject *p_var) {
	String vars;
	VariantWriter::write_to_string(GDMonoMarshal::mono_object_to_variant(p_var), vars);
	return GDMonoMarshal::mono_string_from_godot(vars);
}

MonoObject *godot_icall_Godot_weakref(Object *p_obj) {
	if (!p_obj)
		return NULL;

	Ref<WeakRef> wref;
	Reference *ref = Object::cast_to<Reference>(p_obj);

	if (ref) {
		REF r = ref;
		if (!r.is_valid())
			return NULL;

		wref.instance();
		wref->set_ref(r);
	} else {
		wref.instance();
		wref->set_obj(p_obj);
	}

	return GDMonoUtils::create_managed_for_godot_object(CACHED_CLASS(WeakRef), Reference::get_class_static(), Object::cast_to<Object>(wref.ptr()));
}
