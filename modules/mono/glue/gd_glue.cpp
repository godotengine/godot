/*************************************************************************/
/*  gd_glue.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "gd_glue.h"

#ifdef MONO_GLUE_ENABLED

#include "core/array.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include "core/variant.h"
#include "core/variant_parser.h"

#include "../mono_gd/gd_mono_utils.h"

MonoObject *godot_icall_GD_bytes2var(MonoArray *p_bytes, MonoBoolean p_allow_objects) {
	Variant ret;
	PoolByteArray varr = GDMonoMarshal::mono_array_to_PoolByteArray(p_bytes);
	PoolByteArray::Read r = varr.read();
	Error err = decode_variant(ret, r.ptr(), varr.size(), NULL, p_allow_objects);
	if (err != OK) {
		ret = RTR("Not enough bytes for decoding bytes, or invalid format.");
	}
	return GDMonoMarshal::variant_to_mono_object(ret);
}

MonoObject *godot_icall_GD_convert(MonoObject *p_what, int32_t p_type) {
	Variant what = GDMonoMarshal::mono_object_to_variant(p_what);
	const Variant *args[1] = { &what };
	Variant::CallError ce;
	Variant ret = Variant::construct(Variant::Type(p_type), args, 1, ce);
	ERR_FAIL_COND_V(ce.error != Variant::CallError::CALL_OK, NULL);
	return GDMonoMarshal::variant_to_mono_object(ret);
}

int godot_icall_GD_hash(MonoObject *p_var) {
	return GDMonoMarshal::mono_object_to_variant(p_var).hash();
}

MonoObject *godot_icall_GD_instance_from_id(uint64_t p_instance_id) {
	return GDMonoUtils::unmanaged_get_managed(ObjectDB::get_instance(p_instance_id));
}

void godot_icall_GD_print(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++)
		str += what[i].operator String();
	print_line(str);
}

void godot_icall_GD_printerr(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++)
		str += what[i].operator String();
	OS::get_singleton()->printerr("%s\n", str.utf8().get_data());
}

void godot_icall_GD_printraw(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++)
		str += what[i].operator String();
	OS::get_singleton()->print("%s", str.utf8().get_data());
}

void godot_icall_GD_prints(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++) {
		if (i)
			str += " ";
		str += what[i].operator String();
	}
	print_line(str);
}

void godot_icall_GD_printt(MonoArray *p_what) {
	Array what = GDMonoMarshal::mono_array_to_Array(p_what);
	String str;
	for (int i = 0; i < what.size(); i++) {
		if (i)
			str += "\t";
		str += what[i].operator String();
	}
	print_line(str);
}

float godot_icall_GD_randf() {
	return Math::randf();
}

uint32_t godot_icall_GD_randi() {
	return Math::rand();
}

void godot_icall_GD_randomize() {
	Math::randomize();
}

double godot_icall_GD_rand_range(double from, double to) {
	return Math::random(from, to);
}

uint32_t godot_icall_GD_rand_seed(uint64_t seed, uint64_t *newSeed) {
	int ret = Math::rand_from_seed(&seed);
	*newSeed = seed;
	return ret;
}

void godot_icall_GD_seed(uint64_t p_seed) {
	Math::seed(p_seed);
}

MonoString *godot_icall_GD_str(MonoArray *p_what) {
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

MonoObject *godot_icall_GD_str2var(MonoString *p_str) {
	Variant ret;

	VariantParser::StreamString ss;
	ss.s = GDMonoMarshal::mono_string_to_godot(p_str);

	String errs;
	int line;
	Error err = VariantParser::parse(&ss, ret, errs, line);
	if (err != OK) {
		String err_str = "Parse error at line " + itos(line) + ": " + errs + ".";
		ERR_PRINTS(err_str);
		ret = err_str;
	}

	return GDMonoMarshal::variant_to_mono_object(ret);
}

MonoBoolean godot_icall_GD_type_exists(MonoString *p_type) {
	return ClassDB::class_exists(GDMonoMarshal::mono_string_to_godot(p_type));
}

void godot_icall_GD_pusherror(MonoString *p_str) {
	ERR_PRINTS(GDMonoMarshal::mono_string_to_godot(p_str));
}

void godot_icall_GD_pushwarning(MonoString *p_str) {
	WARN_PRINTS(GDMonoMarshal::mono_string_to_godot(p_str));
}

MonoArray *godot_icall_GD_var2bytes(MonoObject *p_var, MonoBoolean p_full_objects) {
	Variant var = GDMonoMarshal::mono_object_to_variant(p_var);

	PoolByteArray barr;
	int len;
	Error err = encode_variant(var, NULL, len, p_full_objects);
	ERR_FAIL_COND_V_MSG(err != OK, NULL, "Unexpected error encoding variable to bytes, likely unserializable type found (Object or RID).");

	barr.resize(len);
	{
		PoolByteArray::Write w = barr.write();
		encode_variant(var, w.ptr(), len, p_full_objects);
	}

	return GDMonoMarshal::PoolByteArray_to_mono_array(barr);
}

MonoString *godot_icall_GD_var2str(MonoObject *p_var) {
	String vars;
	VariantWriter::write_to_string(GDMonoMarshal::mono_object_to_variant(p_var), vars);
	return GDMonoMarshal::mono_string_from_godot(vars);
}

MonoObject *godot_icall_DefaultGodotTaskScheduler() {
	return GDMonoUtils::mono_cache.task_scheduler_handle->get_target();
}

void godot_register_gd_icalls() {
	mono_add_internal_call("Godot.GD::godot_icall_GD_bytes2var", (void *)godot_icall_GD_bytes2var);
	mono_add_internal_call("Godot.GD::godot_icall_GD_convert", (void *)godot_icall_GD_convert);
	mono_add_internal_call("Godot.GD::godot_icall_GD_hash", (void *)godot_icall_GD_hash);
	mono_add_internal_call("Godot.GD::godot_icall_GD_instance_from_id", (void *)godot_icall_GD_instance_from_id);
	mono_add_internal_call("Godot.GD::godot_icall_GD_pusherror", (void *)godot_icall_GD_pusherror);
	mono_add_internal_call("Godot.GD::godot_icall_GD_pushwarning", (void *)godot_icall_GD_pushwarning);
	mono_add_internal_call("Godot.GD::godot_icall_GD_print", (void *)godot_icall_GD_print);
	mono_add_internal_call("Godot.GD::godot_icall_GD_printerr", (void *)godot_icall_GD_printerr);
	mono_add_internal_call("Godot.GD::godot_icall_GD_printraw", (void *)godot_icall_GD_printraw);
	mono_add_internal_call("Godot.GD::godot_icall_GD_prints", (void *)godot_icall_GD_prints);
	mono_add_internal_call("Godot.GD::godot_icall_GD_printt", (void *)godot_icall_GD_printt);
	mono_add_internal_call("Godot.GD::godot_icall_GD_randf", (void *)godot_icall_GD_randf);
	mono_add_internal_call("Godot.GD::godot_icall_GD_randi", (void *)godot_icall_GD_randi);
	mono_add_internal_call("Godot.GD::godot_icall_GD_randomize", (void *)godot_icall_GD_randomize);
	mono_add_internal_call("Godot.GD::godot_icall_GD_rand_range", (void *)godot_icall_GD_rand_range);
	mono_add_internal_call("Godot.GD::godot_icall_GD_rand_seed", (void *)godot_icall_GD_rand_seed);
	mono_add_internal_call("Godot.GD::godot_icall_GD_seed", (void *)godot_icall_GD_seed);
	mono_add_internal_call("Godot.GD::godot_icall_GD_str", (void *)godot_icall_GD_str);
	mono_add_internal_call("Godot.GD::godot_icall_GD_str2var", (void *)godot_icall_GD_str2var);
	mono_add_internal_call("Godot.GD::godot_icall_GD_type_exists", (void *)godot_icall_GD_type_exists);
	mono_add_internal_call("Godot.GD::godot_icall_GD_var2bytes", (void *)godot_icall_GD_var2bytes);
	mono_add_internal_call("Godot.GD::godot_icall_GD_var2str", (void *)godot_icall_GD_var2str);

	// Dispatcher
	mono_add_internal_call("Godot.Dispatcher::godot_icall_DefaultGodotTaskScheduler", (void *)godot_icall_DefaultGodotTaskScheduler);
}

#endif // MONO_GLUE_ENABLED
