/*************************************************************************/
/*  gdscript_utility_functions.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdscript_utility_functions.h"

#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/object/object.h"
#include "core/templates/oa_hash_map.h"
#include "core/templates/vector.h"
#include "gdscript.h"

#ifdef DEBUG_ENABLED

#define VALIDATE_ARG_COUNT(m_count)                                         \
	if (p_arg_count < m_count) {                                            \
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;  \
		r_error.argument = m_count;                                         \
		r_error.expected = m_count;                                         \
		*r_ret = Variant();                                                 \
		return;                                                             \
	}                                                                       \
	if (p_arg_count > m_count) {                                            \
		r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS; \
		r_error.argument = m_count;                                         \
		r_error.expected = m_count;                                         \
		*r_ret = Variant();                                                 \
		return;                                                             \
	}

#define VALIDATE_ARG_INT(m_arg)                                           \
	if (p_args[m_arg]->get_type() != Variant::INT) {                      \
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT; \
		r_error.argument = m_arg;                                         \
		r_error.expected = Variant::INT;                                  \
		*r_ret = Variant();                                               \
		return;                                                           \
	}

#define VALIDATE_ARG_NUM(m_arg)                                           \
	if (!p_args[m_arg]->is_num()) {                                       \
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT; \
		r_error.argument = m_arg;                                         \
		r_error.expected = Variant::FLOAT;                                \
		*r_ret = Variant();                                               \
		return;                                                           \
	}

#else

#define VALIDATE_ARG_COUNT(m_count)
#define VALIDATE_ARG_INT(m_arg)
#define VALIDATE_ARG_NUM(m_arg)

#endif

struct GDScriptUtilityFunctionsDefinitions {
	static inline void convert(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(2);
		VALIDATE_ARG_INT(1);
		int type = *p_args[1];
		if (type < 0 || type >= Variant::VARIANT_MAX) {
			*r_ret = RTR("Invalid type argument to convert(), use TYPE_* constants.");
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::INT;
			return;

		} else {
			Variant::construct(Variant::Type(type), *r_ret, p_args, 1, r_error);
		}
	}

	static inline void type_exists(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(1);
		*r_ret = ClassDB::class_exists(*p_args[0]);
	}

	static inline void _char(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(1);
		VALIDATE_ARG_INT(0);
		char32_t result[2] = { *p_args[0], 0 };
		*r_ret = String(result);
	}

	static inline void str(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		if (p_arg_count < 1) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = 1;
			*r_ret = Variant();
			return;
		}

		String str;
		for (int i = 0; i < p_arg_count; i++) {
			String os = p_args[i]->operator String();

			if (i == 0) {
				str = os;
			} else {
				str += os;
			}
		}
		*r_ret = str;
	}

	static inline void range(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		switch (p_arg_count) {
			case 0: {
				r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
				r_error.argument = 1;
				r_error.expected = 1;
				*r_ret = Variant();
			} break;
			case 1: {
				VALIDATE_ARG_NUM(0);
				int count = *p_args[0];
				Array arr;
				if (count <= 0) {
					*r_ret = arr;
					return;
				}
				Error err = arr.resize(count);
				if (err != OK) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					*r_ret = Variant();
					return;
				}

				for (int i = 0; i < count; i++) {
					arr[i] = i;
				}

				*r_ret = arr;
			} break;
			case 2: {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);

				int from = *p_args[0];
				int to = *p_args[1];

				Array arr;
				if (from >= to) {
					*r_ret = arr;
					return;
				}
				Error err = arr.resize(to - from);
				if (err != OK) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					*r_ret = Variant();
					return;
				}
				for (int i = from; i < to; i++) {
					arr[i - from] = i;
				}
				*r_ret = arr;
			} break;
			case 3: {
				VALIDATE_ARG_NUM(0);
				VALIDATE_ARG_NUM(1);
				VALIDATE_ARG_NUM(2);

				int from = *p_args[0];
				int to = *p_args[1];
				int incr = *p_args[2];
				if (incr == 0) {
					*r_ret = RTR("Step argument is zero!");
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					return;
				}

				Array arr;
				if (from >= to && incr > 0) {
					*r_ret = arr;
					return;
				}
				if (from <= to && incr < 0) {
					*r_ret = arr;
					return;
				}

				// Calculate how many.
				int count = 0;
				if (incr > 0) {
					count = ((to - from - 1) / incr) + 1;
				} else {
					count = ((from - to - 1) / -incr) + 1;
				}

				Error err = arr.resize(count);

				if (err != OK) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					*r_ret = Variant();
					return;
				}

				if (incr > 0) {
					int idx = 0;
					for (int i = from; i < to; i += incr) {
						arr[idx++] = i;
					}
				} else {
					int idx = 0;
					for (int i = from; i > to; i += incr) {
						arr[idx++] = i;
					}
				}

				*r_ret = arr;
			} break;
			default: {
				r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
				r_error.argument = 3;
				r_error.expected = 3;
				*r_ret = Variant();

			} break;
		}
	}

	static inline void load(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(1);
		if (p_args[0]->get_type() != Variant::STRING) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::STRING;
			*r_ret = Variant();
		} else {
			*r_ret = ResourceLoader::load(*p_args[0]);
		}
	}

	static inline void inst2dict(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(1);

		if (p_args[0]->get_type() == Variant::NIL) {
			*r_ret = Variant();
		} else if (p_args[0]->get_type() != Variant::OBJECT) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			*r_ret = Variant();
		} else {
			Object *obj = *p_args[0];
			if (!obj) {
				*r_ret = Variant();

			} else if (!obj->get_script_instance() || obj->get_script_instance()->get_language() != GDScriptLanguage::get_singleton()) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::DICTIONARY;
				*r_ret = RTR("Not a script with an instance");
				return;
			} else {
				GDScriptInstance *ins = static_cast<GDScriptInstance *>(obj->get_script_instance());
				Ref<GDScript> base = ins->get_script();
				if (base.is_null()) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = 0;
					r_error.expected = Variant::DICTIONARY;
					*r_ret = RTR("Not based on a script");
					return;
				}

				GDScript *p = base.ptr();
				Vector<StringName> sname;

				while (p->_owner) {
					sname.push_back(p->name);
					p = p->_owner;
				}
				sname.reverse();

				if (!p->path.is_resource_file()) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = 0;
					r_error.expected = Variant::DICTIONARY;
					*r_ret = Variant();

					*r_ret = RTR("Not based on a resource file");

					return;
				}

				NodePath cp(sname, Vector<StringName>(), false);

				Dictionary d;
				d["@subpath"] = cp;
				d["@path"] = p->get_path();

				for (const KeyValue<StringName, GDScript::MemberInfo> &E : base->member_indices) {
					if (!d.has(E.key)) {
						d[E.key] = ins->members[E.value.index];
					}
				}
				*r_ret = d;
			}
		}
	}

	static inline void dict2inst(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(1);

		if (p_args[0]->get_type() != Variant::DICTIONARY) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::DICTIONARY;
			*r_ret = Variant();

			return;
		}

		Dictionary d = *p_args[0];

		if (!d.has("@path")) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
			*r_ret = RTR("Invalid instance dictionary format (missing @path)");

			return;
		}

		Ref<Script> scr = ResourceLoader::load(d["@path"]);
		if (!scr.is_valid()) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
			*r_ret = RTR("Invalid instance dictionary format (can't load script at @path)");
			return;
		}

		Ref<GDScript> gdscr = scr;

		if (!gdscr.is_valid()) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
			*r_ret = Variant();
			*r_ret = RTR("Invalid instance dictionary format (invalid script at @path)");
			return;
		}

		NodePath sub;
		if (d.has("@subpath")) {
			sub = d["@subpath"];
		}

		for (int i = 0; i < sub.get_name_count(); i++) {
			gdscr = gdscr->subclasses[sub.get_name(i)];
			if (!gdscr.is_valid()) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::OBJECT;
				*r_ret = Variant();
				*r_ret = RTR("Invalid instance dictionary (invalid subclasses)");
				return;
			}
		}
		*r_ret = gdscr->_new(nullptr, -1 /*skip initializer*/, r_error);

		if (r_error.error != Callable::CallError::CALL_OK) {
			*r_ret = Variant();
			return;
		}

		GDScriptInstance *ins = static_cast<GDScriptInstance *>(static_cast<Object *>(*r_ret)->get_script_instance());
		Ref<GDScript> gd_ref = ins->get_script();

		for (KeyValue<StringName, GDScript::MemberInfo> &E : gd_ref->member_indices) {
			if (d.has(E.key)) {
				ins->members.write[E.value.index] = d[E.key];
			}
		}
	}

	static inline void Color8(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		if (p_arg_count < 3) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument = 3;
			*r_ret = Variant();
			return;
		}
		if (p_arg_count > 4) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.argument = 4;
			*r_ret = Variant();
			return;
		}

		VALIDATE_ARG_INT(0);
		VALIDATE_ARG_INT(1);
		VALIDATE_ARG_INT(2);

		Color color((int64_t)*p_args[0] / 255.0f, (int64_t)*p_args[1] / 255.0f, (int64_t)*p_args[2] / 255.0f);

		if (p_arg_count == 4) {
			VALIDATE_ARG_INT(3);
			color.a = (int64_t)*p_args[3] / 255.0f;
		}

		*r_ret = color;
	}

	static inline void print_debug(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		String str;
		for (int i = 0; i < p_arg_count; i++) {
			str += p_args[i]->operator String();
		}

		if (Thread::get_caller_id() == Thread::get_main_id()) {
			ScriptLanguage *script = GDScriptLanguage::get_singleton();
			if (script->debug_get_stack_level_count() > 0) {
				str += "\n   At: " + script->debug_get_stack_level_source(0) + ":" + itos(script->debug_get_stack_level_line(0)) + ":" + script->debug_get_stack_level_function(0) + "()";
			}
		} else {
			str += "\n   At: Cannot retrieve debug info outside the main thread. Thread ID: " + itos(Thread::get_caller_id());
		}

		print_line(str);
		*r_ret = Variant();
	}

	static inline void print_stack(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(0);
		if (Thread::get_caller_id() != Thread::get_main_id()) {
			print_line("Cannot retrieve debug info outside the main thread. Thread ID: " + itos(Thread::get_caller_id()));
			return;
		}

		ScriptLanguage *script = GDScriptLanguage::get_singleton();
		for (int i = 0; i < script->debug_get_stack_level_count(); i++) {
			print_line("Frame " + itos(i) + " - " + script->debug_get_stack_level_source(i) + ":" + itos(script->debug_get_stack_level_line(i)) + " in function '" + script->debug_get_stack_level_function(i) + "'");
		};
		*r_ret = Variant();
	}

	static inline void get_stack(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(0);
		if (Thread::get_caller_id() != Thread::get_main_id()) {
			*r_ret = Array();
			return;
		}

		ScriptLanguage *script = GDScriptLanguage::get_singleton();
		Array ret;
		for (int i = 0; i < script->debug_get_stack_level_count(); i++) {
			Dictionary frame;
			frame["source"] = script->debug_get_stack_level_source(i);
			frame["function"] = script->debug_get_stack_level_function(i);
			frame["line"] = script->debug_get_stack_level_line(i);
			ret.push_back(frame);
		};
		*r_ret = ret;
	}

	static inline void len(Variant *r_ret, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		VALIDATE_ARG_COUNT(1);
		switch (p_args[0]->get_type()) {
			case Variant::STRING: {
				String d = *p_args[0];
				*r_ret = d.length();
			} break;
			case Variant::DICTIONARY: {
				Dictionary d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::ARRAY: {
				Array d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_BYTE_ARRAY: {
				Vector<uint8_t> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_INT32_ARRAY: {
				Vector<int32_t> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_INT64_ARRAY: {
				Vector<int64_t> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_FLOAT32_ARRAY: {
				Vector<float> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_FLOAT64_ARRAY: {
				Vector<double> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_STRING_ARRAY: {
				Vector<String> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_VECTOR2_ARRAY: {
				Vector<Vector2> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_VECTOR3_ARRAY: {
				Vector<Vector3> d = *p_args[0];
				*r_ret = d.size();
			} break;
			case Variant::PACKED_COLOR_ARRAY: {
				Vector<Color> d = *p_args[0];
				*r_ret = d.size();
			} break;
			default: {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = 0;
				r_error.expected = Variant::NIL;
				*r_ret = vformat(RTR("Value of type '%s' can't provide a length."), Variant::get_type_name(p_args[0]->get_type()));
			}
		}
	}
};

struct GDScriptUtilityFunctionInfo {
	GDScriptUtilityFunctions::FunctionPtr function;
	MethodInfo info;
	bool is_constant = false;
};

static OAHashMap<StringName, GDScriptUtilityFunctionInfo> utility_function_table;
static List<StringName> utility_function_name_table;

static void _register_function(const String &p_name, const MethodInfo &p_method_info, GDScriptUtilityFunctions::FunctionPtr p_function, bool p_is_const) {
	StringName sname(p_name);

	ERR_FAIL_COND(utility_function_table.has(sname));

	GDScriptUtilityFunctionInfo function;
	function.function = p_function;
	function.info = p_method_info;
	function.is_constant = p_is_const;

	utility_function_table.insert(sname, function);
	utility_function_name_table.push_back(sname);
}

#define REGISTER_FUNC(m_func, m_is_const, m_return_type, ...)                                    \
	{                                                                                            \
		String name(#m_func);                                                                    \
		if (name.begins_with("_")) {                                                             \
			name = name.substr(1, name.length() - 1);                                            \
		}                                                                                        \
		MethodInfo info = MethodInfo(name, __VA_ARGS__);                                         \
		info.return_val.type = m_return_type;                                                    \
		_register_function(name, info, GDScriptUtilityFunctionsDefinitions::m_func, m_is_const); \
	}

#define REGISTER_FUNC_NO_ARGS(m_func, m_is_const, m_return_type)                                 \
	{                                                                                            \
		String name(#m_func);                                                                    \
		if (name.begins_with("_")) {                                                             \
			name = name.substr(1, name.length() - 1);                                            \
		}                                                                                        \
		MethodInfo info = MethodInfo(name);                                                      \
		info.return_val.type = m_return_type;                                                    \
		_register_function(name, info, GDScriptUtilityFunctionsDefinitions::m_func, m_is_const); \
	}

#define REGISTER_VARARG_FUNC(m_func, m_is_const, m_return_type)                                  \
	{                                                                                            \
		String name(#m_func);                                                                    \
		if (name.begins_with("_")) {                                                             \
			name = name.substr(1, name.length() - 1);                                            \
		}                                                                                        \
		MethodInfo info = MethodInfo(name);                                                      \
		info.return_val.type = m_return_type;                                                    \
		info.flags |= METHOD_FLAG_VARARG;                                                        \
		_register_function(name, info, GDScriptUtilityFunctionsDefinitions::m_func, m_is_const); \
	}

#define REGISTER_VARIANT_FUNC(m_func, m_is_const, ...)                                           \
	{                                                                                            \
		String name(#m_func);                                                                    \
		if (name.begins_with("_")) {                                                             \
			name = name.substr(1, name.length() - 1);                                            \
		}                                                                                        \
		MethodInfo info = MethodInfo(name, __VA_ARGS__);                                         \
		info.return_val.type = Variant::NIL;                                                     \
		info.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;                                  \
		_register_function(name, info, GDScriptUtilityFunctionsDefinitions::m_func, m_is_const); \
	}

#define REGISTER_CLASS_FUNC(m_func, m_is_const, m_return_type, ...)                              \
	{                                                                                            \
		String name(#m_func);                                                                    \
		if (name.begins_with("_")) {                                                             \
			name = name.substr(1, name.length() - 1);                                            \
		}                                                                                        \
		MethodInfo info = MethodInfo(name, __VA_ARGS__);                                         \
		info.return_val.type = Variant::OBJECT;                                                  \
		info.return_val.hint = PROPERTY_HINT_RESOURCE_TYPE;                                      \
		info.return_val.class_name = m_return_type;                                              \
		_register_function(name, info, GDScriptUtilityFunctionsDefinitions::m_func, m_is_const); \
	}

#define REGISTER_FUNC_DEF(m_func, m_is_const, m_default, m_return_type, ...)                     \
	{                                                                                            \
		String name(#m_func);                                                                    \
		if (name.begins_with("_")) {                                                             \
			name = name.substr(1, name.length() - 1);                                            \
		}                                                                                        \
		MethodInfo info = MethodInfo(name, __VA_ARGS__);                                         \
		info.return_val.type = m_return_type;                                                    \
		info.default_arguments.push_back(m_default);                                             \
		_register_function(name, info, GDScriptUtilityFunctionsDefinitions::m_func, m_is_const); \
	}

#define ARG(m_name, m_type) \
	PropertyInfo(m_type, m_name)

#define VARARG(m_name) \
	PropertyInfo(Variant::NIL, m_name, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT)

void GDScriptUtilityFunctions::register_functions() {
	REGISTER_VARIANT_FUNC(convert, true, VARARG("what"), ARG("type", Variant::INT));
	REGISTER_FUNC(type_exists, true, Variant::BOOL, ARG("type", Variant::STRING_NAME));
	REGISTER_FUNC(_char, true, Variant::STRING, ARG("char", Variant::INT));
	REGISTER_VARARG_FUNC(str, true, Variant::STRING);
	REGISTER_VARARG_FUNC(range, false, Variant::ARRAY);
	REGISTER_CLASS_FUNC(load, false, "Resource", ARG("path", Variant::STRING));
	REGISTER_FUNC(inst2dict, false, Variant::DICTIONARY, ARG("instance", Variant::OBJECT));
	REGISTER_FUNC(dict2inst, false, Variant::OBJECT, ARG("dictionary", Variant::DICTIONARY));
	REGISTER_FUNC_DEF(Color8, true, 255, Variant::COLOR, ARG("r8", Variant::INT), ARG("g8", Variant::INT), ARG("b8", Variant::INT), ARG("a8", Variant::INT));
	REGISTER_VARARG_FUNC(print_debug, false, Variant::NIL);
	REGISTER_FUNC_NO_ARGS(print_stack, false, Variant::NIL);
	REGISTER_FUNC_NO_ARGS(get_stack, false, Variant::ARRAY);
	REGISTER_FUNC(len, true, Variant::INT, VARARG("var"));
}

void GDScriptUtilityFunctions::unregister_functions() {
	utility_function_name_table.clear();
	utility_function_table.clear();
}

GDScriptUtilityFunctions::FunctionPtr GDScriptUtilityFunctions::get_function(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, nullptr);
	return info->function;
}

bool GDScriptUtilityFunctions::has_function_return_value(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, false);
	return info->info.return_val.type != Variant::NIL || bool(info->info.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT);
}

Variant::Type GDScriptUtilityFunctions::get_function_return_type(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, Variant::NIL);
	return info->info.return_val.type;
}

StringName GDScriptUtilityFunctions::get_function_return_class(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, StringName());
	return info->info.return_val.class_name;
}

Variant::Type GDScriptUtilityFunctions::get_function_argument_type(const StringName &p_function, int p_arg) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, Variant::NIL);
	ERR_FAIL_COND_V(p_arg >= info->info.arguments.size(), Variant::NIL);
	return info->info.arguments[p_arg].type;
}

int GDScriptUtilityFunctions::get_function_argument_count(const StringName &p_function, int p_arg) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, 0);
	return info->info.arguments.size();
}

bool GDScriptUtilityFunctions::is_function_vararg(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, false);
	return (bool)(info->info.flags & METHOD_FLAG_VARARG);
}

bool GDScriptUtilityFunctions::is_function_constant(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, false);
	return info->is_constant;
}

bool GDScriptUtilityFunctions::function_exists(const StringName &p_function) {
	return utility_function_table.has(p_function);
}

void GDScriptUtilityFunctions::get_function_list(List<StringName> *r_functions) {
	for (const StringName &E : utility_function_name_table) {
		r_functions->push_back(E);
	}
}

MethodInfo GDScriptUtilityFunctions::get_function_info(const StringName &p_function) {
	GDScriptUtilityFunctionInfo *info = utility_function_table.lookup_ptr(p_function);
	ERR_FAIL_COND_V(!info, MethodInfo());
	return info->info;
}
