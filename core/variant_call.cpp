/*************************************************************************/
/*  variant_call.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "variant.h"

#include "core/color_names.inc"
#include "core/core_string_names.h"
#include "core/crypto/crypto_core.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/compression.h"
#include "core/object.h"
#include "core/os/os.h"

typedef void (*VariantFunc)(Variant &r_ret, Variant &p_self, const Variant **p_args);
typedef void (*VariantConstructFunc)(Variant &r_ret, const Variant **p_args);

struct _VariantCall {

	static void Vector3_dot(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		r_ret = reinterpret_cast<Vector3 *>(p_self._data._mem)->dot(*reinterpret_cast<const Vector3 *>(p_args[0]->_data._mem));
	}

	struct FuncData {

		int arg_count;
		Vector<Variant> default_args;
		Vector<Variant::Type> arg_types;
		Vector<StringName> arg_names;
		Variant::Type return_type;

		bool _const;
		bool returns;

		VariantFunc func;

		_FORCE_INLINE_ bool verify_arguments(const Variant **p_args, Callable::CallError &r_error) {

			if (arg_count == 0)
				return true;

			const Variant::Type *tptr = &arg_types[0];

			for (int i = 0; i < arg_count; i++) {

				if (tptr[i] == Variant::NIL || tptr[i] == p_args[i]->type)
					continue; // all good
				if (!Variant::can_convert(p_args[i]->type, tptr[i])) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.argument = i;
					r_error.expected = tptr[i];
					return false;
				}
			}
			return true;
		}

		_FORCE_INLINE_ void call(Variant &r_ret, Variant &p_self, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
#ifdef DEBUG_ENABLED
			if (p_argcount > arg_count) {
				r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
				r_error.argument = arg_count;
				return;
			} else
#endif
					if (p_argcount < arg_count) {
				int def_argcount = default_args.size();
#ifdef DEBUG_ENABLED
				if (p_argcount < (arg_count - def_argcount)) {
					r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
					r_error.argument = arg_count - def_argcount;
					return;
				}

#endif
				ERR_FAIL_COND(p_argcount > VARIANT_ARG_MAX);
				const Variant *newargs[VARIANT_ARG_MAX];
				for (int i = 0; i < p_argcount; i++)
					newargs[i] = p_args[i];
				// fill in any remaining parameters with defaults
				int first_default_arg = arg_count - def_argcount;
				for (int i = p_argcount; i < arg_count; i++)
					newargs[i] = &default_args[i - first_default_arg];
#ifdef DEBUG_ENABLED
				if (!verify_arguments(newargs, r_error))
					return;
#endif
				func(r_ret, p_self, newargs);
			} else {
#ifdef DEBUG_ENABLED
				if (!verify_arguments(p_args, r_error))
					return;
#endif
				func(r_ret, p_self, p_args);
			}
		}
	};

	struct TypeFunc {

		Map<StringName, FuncData> functions;
	};

	static TypeFunc *type_funcs;

	struct Arg {
		StringName name;
		Variant::Type type;
		Arg() { type = Variant::NIL; }
		Arg(Variant::Type p_type, const StringName &p_name) :
				name(p_name),
				type(p_type) {
		}
	};

	//void addfunc(Variant::Type p_type, const StringName& p_name,VariantFunc p_func);

	static void make_func_return_variant(Variant::Type p_type, const StringName &p_name) {

#ifdef DEBUG_ENABLED
		type_funcs[p_type].functions[p_name].returns = true;
#endif
	}

	static void addfunc(bool p_const, Variant::Type p_type, Variant::Type p_return, bool p_has_return, const StringName &p_name, VariantFunc p_func, const Vector<Variant> &p_defaultarg, const Arg &p_argtype1 = Arg(), const Arg &p_argtype2 = Arg(), const Arg &p_argtype3 = Arg(), const Arg &p_argtype4 = Arg(), const Arg &p_argtype5 = Arg()) {

		FuncData funcdata;
		funcdata.func = p_func;
		funcdata.default_args = p_defaultarg;
		funcdata._const = p_const;
		funcdata.returns = p_has_return;
		funcdata.return_type = p_return;

		if (p_argtype1.name) {
			funcdata.arg_types.push_back(p_argtype1.type);
#ifdef DEBUG_ENABLED
			funcdata.arg_names.push_back(p_argtype1.name);
#endif

		} else
			goto end;

		if (p_argtype2.name) {
			funcdata.arg_types.push_back(p_argtype2.type);
#ifdef DEBUG_ENABLED
			funcdata.arg_names.push_back(p_argtype2.name);
#endif

		} else
			goto end;

		if (p_argtype3.name) {
			funcdata.arg_types.push_back(p_argtype3.type);
#ifdef DEBUG_ENABLED
			funcdata.arg_names.push_back(p_argtype3.name);
#endif

		} else
			goto end;

		if (p_argtype4.name) {
			funcdata.arg_types.push_back(p_argtype4.type);
#ifdef DEBUG_ENABLED
			funcdata.arg_names.push_back(p_argtype4.name);
#endif
		} else
			goto end;

		if (p_argtype5.name) {
			funcdata.arg_types.push_back(p_argtype5.type);
#ifdef DEBUG_ENABLED
			funcdata.arg_names.push_back(p_argtype5.name);
#endif
		} else
			goto end;

	end:

		funcdata.arg_count = funcdata.arg_types.size();
		type_funcs[p_type].functions[p_name] = funcdata;
	}

#define VCALL_LOCALMEM0(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._mem)->m_method(); }
#define VCALL_LOCALMEM0R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._mem)->m_method(); }
#define VCALL_LOCALMEM1(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0]); }
#define VCALL_LOCALMEM1R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0]); }
#define VCALL_LOCALMEM2(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1]); }
#define VCALL_LOCALMEM2R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1]); }
#define VCALL_LOCALMEM3(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1], *p_args[2]); }
#define VCALL_LOCALMEM3R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1], *p_args[2]); }
#define VCALL_LOCALMEM4(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3]); }
#define VCALL_LOCALMEM4R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3]); }
#define VCALL_LOCALMEM5(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3], *p_args[4]); }
#define VCALL_LOCALMEM5R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._mem)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3], *p_args[4]); }

	// built-in functions of localmem based types

	VCALL_LOCALMEM1R(String, casecmp_to);
	VCALL_LOCALMEM1R(String, nocasecmp_to);
	VCALL_LOCALMEM0R(String, length);
	VCALL_LOCALMEM3R(String, count);
	VCALL_LOCALMEM3R(String, countn);
	VCALL_LOCALMEM2R(String, substr);
	VCALL_LOCALMEM2R(String, find);
	VCALL_LOCALMEM1R(String, find_last);
	VCALL_LOCALMEM2R(String, findn);
	VCALL_LOCALMEM2R(String, rfind);
	VCALL_LOCALMEM2R(String, rfindn);
	VCALL_LOCALMEM1R(String, match);
	VCALL_LOCALMEM1R(String, matchn);
	VCALL_LOCALMEM1R(String, begins_with);
	VCALL_LOCALMEM1R(String, ends_with);
	VCALL_LOCALMEM1R(String, is_subsequence_of);
	VCALL_LOCALMEM1R(String, is_subsequence_ofi);
	VCALL_LOCALMEM0R(String, bigrams);
	VCALL_LOCALMEM1R(String, similarity);
	VCALL_LOCALMEM2R(String, format);
	VCALL_LOCALMEM2R(String, replace);
	VCALL_LOCALMEM2R(String, replacen);
	VCALL_LOCALMEM1R(String, repeat);
	VCALL_LOCALMEM2R(String, insert);
	VCALL_LOCALMEM0R(String, capitalize);
	VCALL_LOCALMEM3R(String, split);
	VCALL_LOCALMEM3R(String, rsplit);
	VCALL_LOCALMEM2R(String, split_floats);
	VCALL_LOCALMEM0R(String, to_upper);
	VCALL_LOCALMEM0R(String, to_lower);
	VCALL_LOCALMEM1R(String, left);
	VCALL_LOCALMEM1R(String, right);
	VCALL_LOCALMEM0R(String, dedent);
	VCALL_LOCALMEM2R(String, strip_edges);
	VCALL_LOCALMEM0R(String, strip_escapes);
	VCALL_LOCALMEM1R(String, lstrip);
	VCALL_LOCALMEM1R(String, rstrip);
	VCALL_LOCALMEM0R(String, get_extension);
	VCALL_LOCALMEM0R(String, get_basename);
	VCALL_LOCALMEM1R(String, plus_file);
	VCALL_LOCALMEM1R(String, ord_at);
	VCALL_LOCALMEM2(String, erase);
	VCALL_LOCALMEM0R(String, hash);
	VCALL_LOCALMEM0R(String, md5_text);
	VCALL_LOCALMEM0R(String, sha1_text);
	VCALL_LOCALMEM0R(String, sha256_text);
	VCALL_LOCALMEM0R(String, md5_buffer);
	VCALL_LOCALMEM0R(String, sha1_buffer);
	VCALL_LOCALMEM0R(String, sha256_buffer);
	VCALL_LOCALMEM0R(String, empty);
	VCALL_LOCALMEM1R(String, humanize_size);
	VCALL_LOCALMEM0R(String, is_abs_path);
	VCALL_LOCALMEM0R(String, is_rel_path);
	VCALL_LOCALMEM0R(String, get_base_dir);
	VCALL_LOCALMEM0R(String, get_file);
	VCALL_LOCALMEM0R(String, xml_escape);
	VCALL_LOCALMEM0R(String, xml_unescape);
	VCALL_LOCALMEM0R(String, http_escape);
	VCALL_LOCALMEM0R(String, http_unescape);
	VCALL_LOCALMEM0R(String, c_escape);
	VCALL_LOCALMEM0R(String, c_unescape);
	VCALL_LOCALMEM0R(String, json_escape);
	VCALL_LOCALMEM0R(String, percent_encode);
	VCALL_LOCALMEM0R(String, percent_decode);
	VCALL_LOCALMEM0R(String, is_valid_identifier);
	VCALL_LOCALMEM0R(String, is_valid_integer);
	VCALL_LOCALMEM0R(String, is_valid_float);
	VCALL_LOCALMEM1R(String, is_valid_hex_number);
	VCALL_LOCALMEM0R(String, is_valid_html_color);
	VCALL_LOCALMEM0R(String, is_valid_ip_address);
	VCALL_LOCALMEM0R(String, is_valid_filename);
	VCALL_LOCALMEM0R(String, to_int);
	VCALL_LOCALMEM0R(String, to_float);
	VCALL_LOCALMEM0R(String, hex_to_int);
	VCALL_LOCALMEM1R(String, pad_decimals);
	VCALL_LOCALMEM1R(String, pad_zeros);
	VCALL_LOCALMEM1R(String, trim_prefix);
	VCALL_LOCALMEM1R(String, trim_suffix);

	static void _call_String_to_ascii(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		String *s = reinterpret_cast<String *>(p_self._data._mem);
		if (s->empty()) {
			r_ret = PackedByteArray();
			return;
		}
		CharString charstr = s->ascii();

		PackedByteArray retval;
		size_t len = charstr.length();
		retval.resize(len);
		uint8_t *w = retval.ptrw();
		copymem(w, charstr.ptr(), len);

		r_ret = retval;
	}

	static void _call_String_to_utf8(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		String *s = reinterpret_cast<String *>(p_self._data._mem);
		if (s->empty()) {
			r_ret = PackedByteArray();
			return;
		}
		CharString charstr = s->utf8();

		PackedByteArray retval;
		size_t len = charstr.length();
		retval.resize(len);
		uint8_t *w = retval.ptrw();
		copymem(w, charstr.ptr(), len);

		r_ret = retval;
	}

	VCALL_LOCALMEM1R(Vector2, distance_to);
	VCALL_LOCALMEM1R(Vector2, distance_squared_to);
	VCALL_LOCALMEM0R(Vector2, length);
	VCALL_LOCALMEM0R(Vector2, length_squared);
	VCALL_LOCALMEM0R(Vector2, normalized);
	VCALL_LOCALMEM0R(Vector2, is_normalized);
	VCALL_LOCALMEM1R(Vector2, is_equal_approx);
	VCALL_LOCALMEM1R(Vector2, posmod);
	VCALL_LOCALMEM1R(Vector2, posmodv);
	VCALL_LOCALMEM1R(Vector2, project);
	VCALL_LOCALMEM1R(Vector2, angle_to);
	VCALL_LOCALMEM1R(Vector2, angle_to_point);
	VCALL_LOCALMEM1R(Vector2, direction_to);
	VCALL_LOCALMEM2R(Vector2, linear_interpolate);
	VCALL_LOCALMEM2R(Vector2, slerp);
	VCALL_LOCALMEM4R(Vector2, cubic_interpolate);
	VCALL_LOCALMEM2R(Vector2, move_toward);
	VCALL_LOCALMEM1R(Vector2, rotated);
	VCALL_LOCALMEM0R(Vector2, tangent);
	VCALL_LOCALMEM0R(Vector2, floor);
	VCALL_LOCALMEM0R(Vector2, ceil);
	VCALL_LOCALMEM0R(Vector2, round);
	VCALL_LOCALMEM1R(Vector2, snapped);
	VCALL_LOCALMEM0R(Vector2, aspect);
	VCALL_LOCALMEM1R(Vector2, dot);
	VCALL_LOCALMEM1R(Vector2, slide);
	VCALL_LOCALMEM1R(Vector2, bounce);
	VCALL_LOCALMEM1R(Vector2, reflect);
	VCALL_LOCALMEM0R(Vector2, angle);
	VCALL_LOCALMEM1R(Vector2, cross);
	VCALL_LOCALMEM0R(Vector2, abs);
	VCALL_LOCALMEM1R(Vector2, clamped);
	VCALL_LOCALMEM0R(Vector2, sign);

	VCALL_LOCALMEM0R(Vector2i, aspect);
	VCALL_LOCALMEM0R(Vector2i, sign);
	VCALL_LOCALMEM0R(Vector2i, abs);

	VCALL_LOCALMEM0R(Rect2, get_area);
	VCALL_LOCALMEM0R(Rect2, has_no_area);
	VCALL_LOCALMEM1R(Rect2, has_point);
	VCALL_LOCALMEM1R(Rect2, is_equal_approx);
	VCALL_LOCALMEM2R(Rect2, intersects);
	VCALL_LOCALMEM1R(Rect2, encloses);
	VCALL_LOCALMEM1R(Rect2, clip);
	VCALL_LOCALMEM1R(Rect2, merge);
	VCALL_LOCALMEM1R(Rect2, expand);
	VCALL_LOCALMEM1R(Rect2, grow);
	VCALL_LOCALMEM2R(Rect2, grow_margin);
	VCALL_LOCALMEM4R(Rect2, grow_individual);
	VCALL_LOCALMEM0R(Rect2, abs);

	VCALL_LOCALMEM0R(Rect2i, get_area);
	VCALL_LOCALMEM0R(Rect2i, has_no_area);
	VCALL_LOCALMEM1R(Rect2i, has_point);
	VCALL_LOCALMEM1R(Rect2i, intersects);
	VCALL_LOCALMEM1R(Rect2i, encloses);
	VCALL_LOCALMEM1R(Rect2i, clip);
	VCALL_LOCALMEM1R(Rect2i, merge);
	VCALL_LOCALMEM1R(Rect2i, expand);
	VCALL_LOCALMEM1R(Rect2i, grow);
	VCALL_LOCALMEM2R(Rect2i, grow_margin);
	VCALL_LOCALMEM4R(Rect2i, grow_individual);
	VCALL_LOCALMEM0R(Rect2i, abs);

	VCALL_LOCALMEM0R(Vector3, min_axis);
	VCALL_LOCALMEM0R(Vector3, max_axis);
	VCALL_LOCALMEM1R(Vector3, distance_to);
	VCALL_LOCALMEM1R(Vector3, distance_squared_to);
	VCALL_LOCALMEM0R(Vector3, length);
	VCALL_LOCALMEM0R(Vector3, length_squared);
	VCALL_LOCALMEM0R(Vector3, normalized);
	VCALL_LOCALMEM0R(Vector3, is_normalized);
	VCALL_LOCALMEM1R(Vector3, is_equal_approx);
	VCALL_LOCALMEM0R(Vector3, inverse);
	VCALL_LOCALMEM1R(Vector3, snapped);
	VCALL_LOCALMEM2R(Vector3, rotated);
	VCALL_LOCALMEM2R(Vector3, linear_interpolate);
	VCALL_LOCALMEM2R(Vector3, slerp);
	VCALL_LOCALMEM4R(Vector3, cubic_interpolate);
	VCALL_LOCALMEM2R(Vector3, move_toward);
	VCALL_LOCALMEM1R(Vector3, dot);
	VCALL_LOCALMEM1R(Vector3, cross);
	VCALL_LOCALMEM1R(Vector3, outer);
	VCALL_LOCALMEM0R(Vector3, to_diagonal_matrix);
	VCALL_LOCALMEM0R(Vector3, abs);
	VCALL_LOCALMEM0R(Vector3, floor);
	VCALL_LOCALMEM0R(Vector3, ceil);
	VCALL_LOCALMEM0R(Vector3, round);
	VCALL_LOCALMEM1R(Vector3, posmod);
	VCALL_LOCALMEM1R(Vector3, posmodv);
	VCALL_LOCALMEM1R(Vector3, project);
	VCALL_LOCALMEM1R(Vector3, angle_to);
	VCALL_LOCALMEM1R(Vector3, direction_to);
	VCALL_LOCALMEM1R(Vector3, slide);
	VCALL_LOCALMEM1R(Vector3, bounce);
	VCALL_LOCALMEM1R(Vector3, reflect);
	VCALL_LOCALMEM0R(Vector3, sign);

	VCALL_LOCALMEM0R(Vector3i, min_axis);
	VCALL_LOCALMEM0R(Vector3i, max_axis);
	VCALL_LOCALMEM0R(Vector3i, sign);

	VCALL_LOCALMEM0R(Plane, normalized);
	VCALL_LOCALMEM0R(Plane, center);
	VCALL_LOCALMEM0R(Plane, get_any_point);
	VCALL_LOCALMEM1R(Plane, is_equal_approx);
	VCALL_LOCALMEM1R(Plane, is_point_over);
	VCALL_LOCALMEM1R(Plane, distance_to);
	VCALL_LOCALMEM2R(Plane, has_point);
	VCALL_LOCALMEM1R(Plane, project);

	//return vector3 if intersected, nil if not
	static void _call_Plane_intersect_3(Variant &r_ret, Variant &p_self, const Variant **p_args) {
		Vector3 result;
		if (reinterpret_cast<Plane *>(p_self._data._mem)->intersect_3(*p_args[0], *p_args[1], &result))
			r_ret = result;
		else
			r_ret = Variant();
	}

	static void _call_Plane_intersects_ray(Variant &r_ret, Variant &p_self, const Variant **p_args) {
		Vector3 result;
		if (reinterpret_cast<Plane *>(p_self._data._mem)->intersects_ray(*p_args[0], *p_args[1], &result))
			r_ret = result;
		else
			r_ret = Variant();
	}

	static void _call_Plane_intersects_segment(Variant &r_ret, Variant &p_self, const Variant **p_args) {
		Vector3 result;
		if (reinterpret_cast<Plane *>(p_self._data._mem)->intersects_segment(*p_args[0], *p_args[1], &result))
			r_ret = result;
		else
			r_ret = Variant();
	}

	VCALL_LOCALMEM0R(Quat, length);
	VCALL_LOCALMEM0R(Quat, length_squared);
	VCALL_LOCALMEM0R(Quat, normalized);
	VCALL_LOCALMEM0R(Quat, is_normalized);
	VCALL_LOCALMEM1R(Quat, is_equal_approx);
	VCALL_LOCALMEM0R(Quat, inverse);
	VCALL_LOCALMEM1R(Quat, dot);
	VCALL_LOCALMEM1R(Quat, xform);
	VCALL_LOCALMEM2R(Quat, slerp);
	VCALL_LOCALMEM2R(Quat, slerpni);
	VCALL_LOCALMEM4R(Quat, cubic_slerp);
	VCALL_LOCALMEM0R(Quat, get_euler);
	VCALL_LOCALMEM1(Quat, set_euler);
	VCALL_LOCALMEM2(Quat, set_axis_angle);

	VCALL_LOCALMEM0R(Color, to_argb32);
	VCALL_LOCALMEM0R(Color, to_abgr32);
	VCALL_LOCALMEM0R(Color, to_rgba32);
	VCALL_LOCALMEM0R(Color, to_argb64);
	VCALL_LOCALMEM0R(Color, to_abgr64);
	VCALL_LOCALMEM0R(Color, to_rgba64);
	VCALL_LOCALMEM0R(Color, inverted);
	VCALL_LOCALMEM0R(Color, contrasted);
	VCALL_LOCALMEM2R(Color, linear_interpolate);
	VCALL_LOCALMEM1R(Color, blend);
	VCALL_LOCALMEM1R(Color, lightened);
	VCALL_LOCALMEM1R(Color, darkened);
	VCALL_LOCALMEM1R(Color, to_html);
	VCALL_LOCALMEM4R(Color, from_hsv);
	VCALL_LOCALMEM1R(Color, is_equal_approx);

	VCALL_LOCALMEM0R(RID, get_id);

	VCALL_LOCALMEM0R(NodePath, is_absolute);
	VCALL_LOCALMEM0R(NodePath, get_name_count);
	VCALL_LOCALMEM1R(NodePath, get_name);
	VCALL_LOCALMEM0R(NodePath, get_subname_count);
	VCALL_LOCALMEM1R(NodePath, get_subname);
	VCALL_LOCALMEM0R(NodePath, get_concatenated_subnames);
	VCALL_LOCALMEM0R(NodePath, get_as_property_path);
	VCALL_LOCALMEM0R(NodePath, is_empty);

	VCALL_LOCALMEM0R(Dictionary, size);
	VCALL_LOCALMEM0R(Dictionary, empty);
	VCALL_LOCALMEM0(Dictionary, clear);
	VCALL_LOCALMEM1R(Dictionary, has);
	VCALL_LOCALMEM1R(Dictionary, has_all);
	VCALL_LOCALMEM1R(Dictionary, erase);
	VCALL_LOCALMEM0R(Dictionary, hash);
	VCALL_LOCALMEM0R(Dictionary, keys);
	VCALL_LOCALMEM0R(Dictionary, values);
	VCALL_LOCALMEM1R(Dictionary, duplicate);
	VCALL_LOCALMEM2R(Dictionary, get);

	VCALL_LOCALMEM0R(Callable, is_null);
	VCALL_LOCALMEM0R(Callable, is_custom);
	VCALL_LOCALMEM0(Callable, is_standard);
	VCALL_LOCALMEM0(Callable, get_object);
	VCALL_LOCALMEM0(Callable, get_object_id);
	VCALL_LOCALMEM0(Callable, get_method);
	VCALL_LOCALMEM0(Callable, hash);

	VCALL_LOCALMEM0R(Signal, is_null);
	VCALL_LOCALMEM0R(Signal, get_object);
	VCALL_LOCALMEM0R(Signal, get_object_id);
	VCALL_LOCALMEM0R(Signal, get_name);
	VCALL_LOCALMEM3R(Signal, connect);
	VCALL_LOCALMEM1(Signal, disconnect);
	VCALL_LOCALMEM1R(Signal, is_connected);
	VCALL_LOCALMEM0R(Signal, get_connections);

	VCALL_LOCALMEM2(Array, set);
	VCALL_LOCALMEM1R(Array, get);
	VCALL_LOCALMEM0R(Array, size);
	VCALL_LOCALMEM0R(Array, empty);
	VCALL_LOCALMEM0(Array, clear);
	VCALL_LOCALMEM0R(Array, hash);
	VCALL_LOCALMEM1(Array, push_back);
	VCALL_LOCALMEM1(Array, push_front);
	VCALL_LOCALMEM0R(Array, pop_back);
	VCALL_LOCALMEM0R(Array, pop_front);
	VCALL_LOCALMEM1(Array, append);
	VCALL_LOCALMEM1(Array, resize);
	VCALL_LOCALMEM2(Array, insert);
	VCALL_LOCALMEM1(Array, remove);
	VCALL_LOCALMEM0R(Array, front);
	VCALL_LOCALMEM0R(Array, back);
	VCALL_LOCALMEM2R(Array, find);
	VCALL_LOCALMEM2R(Array, rfind);
	VCALL_LOCALMEM1R(Array, find_last);
	VCALL_LOCALMEM1R(Array, count);
	VCALL_LOCALMEM1R(Array, has);
	VCALL_LOCALMEM1(Array, erase);
	VCALL_LOCALMEM0(Array, sort);
	VCALL_LOCALMEM2(Array, sort_custom);
	VCALL_LOCALMEM0(Array, shuffle);
	VCALL_LOCALMEM2R(Array, bsearch);
	VCALL_LOCALMEM4R(Array, bsearch_custom);
	VCALL_LOCALMEM1R(Array, duplicate);
	VCALL_LOCALMEM4R(Array, slice);
	VCALL_LOCALMEM0(Array, invert);
	VCALL_LOCALMEM0R(Array, max);
	VCALL_LOCALMEM0R(Array, min);

	static void _call_PackedByteArray_get_string_from_ascii(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		PackedByteArray *ba = reinterpret_cast<PackedByteArray *>(p_self._data._mem);
		String s;
		if (ba->size() > 0) {
			const uint8_t *r = ba->ptr();
			CharString cs;
			cs.resize(ba->size() + 1);
			copymem(cs.ptrw(), r, ba->size());
			cs[ba->size()] = 0;

			s = cs.get_data();
		}
		r_ret = s;
	}

	static void _call_PackedByteArray_get_string_from_utf8(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		PackedByteArray *ba = reinterpret_cast<PackedByteArray *>(p_self._data._mem);
		String s;
		if (ba->size() > 0) {
			const uint8_t *r = ba->ptr();
			s.parse_utf8((const char *)r, ba->size());
		}
		r_ret = s;
	}

	static void _call_PackedByteArray_compress(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		PackedByteArray *ba = reinterpret_cast<PackedByteArray *>(p_self._data._mem);
		PackedByteArray compressed;
		if (ba->size() > 0) {
			Compression::Mode mode = (Compression::Mode)(int)(*p_args[0]);

			compressed.resize(Compression::get_max_compressed_buffer_size(ba->size(), mode));
			int result = Compression::compress(compressed.ptrw(), ba->ptr(), ba->size(), mode);

			result = result >= 0 ? result : 0;
			compressed.resize(result);
		}
		r_ret = compressed;
	}

	static void _call_PackedByteArray_decompress(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		PackedByteArray *ba = reinterpret_cast<PackedByteArray *>(p_self._data._mem);
		PackedByteArray decompressed;
		Compression::Mode mode = (Compression::Mode)(int)(*p_args[1]);

		int buffer_size = (int)(*p_args[0]);

		if (buffer_size <= 0) {
			r_ret = decompressed;
			ERR_FAIL_MSG("Decompression buffer size must be greater than zero.");
		}

		decompressed.resize(buffer_size);
		int result = Compression::decompress(decompressed.ptrw(), buffer_size, ba->ptr(), ba->size(), mode);

		result = result >= 0 ? result : 0;
		decompressed.resize(result);

		r_ret = decompressed;
	}

	static void _call_PackedByteArray_hex_encode(Variant &r_ret, Variant &p_self, const Variant **p_args) {
		PackedByteArray *ba = reinterpret_cast<PackedByteArray *>(p_self._data._mem);
		if (ba->size() == 0) {
			r_ret = String();
			return;
		}
		const uint8_t *r = ba->ptr();
		String s = String::hex_encode_buffer(&r[0], ba->size());
		r_ret = s;
	}

#define VCALL_PARRMEM0(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(); }
#define VCALL_PARRMEM0R(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(); }
#define VCALL_PARRMEM1(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0]); }
#define VCALL_PARRMEM1R(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0]); }
#define VCALL_PARRMEM2(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1]); }
#define VCALL_PARRMEM2R(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1]); }
#define VCALL_PARRMEM3(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1], *p_args[2]); }
#define VCALL_PARRMEM3R(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1], *p_args[2]); }
#define VCALL_PARRMEM4(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3]); }
#define VCALL_PARRMEM4R(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3]); }
#define VCALL_PARRMEM5(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3], *p_args[4]); }
#define VCALL_PARRMEM5R(m_type, m_elemtype, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = Variant::PackedArrayRef<m_elemtype>::get_array_ptr(p_self._data.packed_array)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3], *p_args[4]); }

	VCALL_PARRMEM0R(PackedByteArray, uint8_t, size);
	VCALL_PARRMEM0R(PackedByteArray, uint8_t, empty);
	VCALL_PARRMEM2(PackedByteArray, uint8_t, set);
	VCALL_PARRMEM1R(PackedByteArray, uint8_t, get);
	VCALL_PARRMEM1(PackedByteArray, uint8_t, push_back);
	VCALL_PARRMEM1(PackedByteArray, uint8_t, resize);
	VCALL_PARRMEM2R(PackedByteArray, uint8_t, insert);
	VCALL_PARRMEM1(PackedByteArray, uint8_t, remove);
	VCALL_PARRMEM1(PackedByteArray, uint8_t, append);
	VCALL_PARRMEM1(PackedByteArray, uint8_t, append_array);
	VCALL_PARRMEM0(PackedByteArray, uint8_t, invert);
	VCALL_PARRMEM2R(PackedByteArray, uint8_t, subarray);

	VCALL_PARRMEM0R(PackedInt32Array, int32_t, size);
	VCALL_PARRMEM0R(PackedInt32Array, int32_t, empty);
	VCALL_PARRMEM2(PackedInt32Array, int32_t, set);
	VCALL_PARRMEM1R(PackedInt32Array, int32_t, get);
	VCALL_PARRMEM1(PackedInt32Array, int32_t, push_back);
	VCALL_PARRMEM1(PackedInt32Array, int32_t, resize);
	VCALL_PARRMEM2R(PackedInt32Array, int32_t, insert);
	VCALL_PARRMEM1(PackedInt32Array, int32_t, remove);
	VCALL_PARRMEM1(PackedInt32Array, int32_t, append);
	VCALL_PARRMEM1(PackedInt32Array, int32_t, append_array);
	VCALL_PARRMEM0(PackedInt32Array, int32_t, invert);

	VCALL_PARRMEM0R(PackedInt64Array, int64_t, size);
	VCALL_PARRMEM0R(PackedInt64Array, int64_t, empty);
	VCALL_PARRMEM2(PackedInt64Array, int64_t, set);
	VCALL_PARRMEM1R(PackedInt64Array, int64_t, get);
	VCALL_PARRMEM1(PackedInt64Array, int64_t, push_back);
	VCALL_PARRMEM1(PackedInt64Array, int64_t, resize);
	VCALL_PARRMEM2R(PackedInt64Array, int64_t, insert);
	VCALL_PARRMEM1(PackedInt64Array, int64_t, remove);
	VCALL_PARRMEM1(PackedInt64Array, int64_t, append);
	VCALL_PARRMEM1(PackedInt64Array, int64_t, append_array);
	VCALL_PARRMEM0(PackedInt64Array, int64_t, invert);

	VCALL_PARRMEM0R(PackedFloat32Array, float, size);
	VCALL_PARRMEM0R(PackedFloat32Array, float, empty);
	VCALL_PARRMEM2(PackedFloat32Array, float, set);
	VCALL_PARRMEM1R(PackedFloat32Array, float, get);
	VCALL_PARRMEM1(PackedFloat32Array, float, push_back);
	VCALL_PARRMEM1(PackedFloat32Array, float, resize);
	VCALL_PARRMEM2R(PackedFloat32Array, float, insert);
	VCALL_PARRMEM1(PackedFloat32Array, float, remove);
	VCALL_PARRMEM1(PackedFloat32Array, float, append);
	VCALL_PARRMEM1(PackedFloat32Array, float, append_array);
	VCALL_PARRMEM0(PackedFloat32Array, float, invert);

	VCALL_PARRMEM0R(PackedFloat64Array, double, size);
	VCALL_PARRMEM0R(PackedFloat64Array, double, empty);
	VCALL_PARRMEM2(PackedFloat64Array, double, set);
	VCALL_PARRMEM1R(PackedFloat64Array, double, get);
	VCALL_PARRMEM1(PackedFloat64Array, double, push_back);
	VCALL_PARRMEM1(PackedFloat64Array, double, resize);
	VCALL_PARRMEM2R(PackedFloat64Array, double, insert);
	VCALL_PARRMEM1(PackedFloat64Array, double, remove);
	VCALL_PARRMEM1(PackedFloat64Array, double, append);
	VCALL_PARRMEM1(PackedFloat64Array, double, append_array);
	VCALL_PARRMEM0(PackedFloat64Array, double, invert);

	VCALL_PARRMEM0R(PackedStringArray, String, size);
	VCALL_PARRMEM0R(PackedStringArray, String, empty);
	VCALL_PARRMEM2(PackedStringArray, String, set);
	VCALL_PARRMEM1R(PackedStringArray, String, get);
	VCALL_PARRMEM1(PackedStringArray, String, push_back);
	VCALL_PARRMEM1(PackedStringArray, String, resize);
	VCALL_PARRMEM2R(PackedStringArray, String, insert);
	VCALL_PARRMEM1(PackedStringArray, String, remove);
	VCALL_PARRMEM1(PackedStringArray, String, append);
	VCALL_PARRMEM1(PackedStringArray, String, append_array);
	VCALL_PARRMEM0(PackedStringArray, String, invert);

	VCALL_PARRMEM0R(PackedVector2Array, Vector2, size);
	VCALL_PARRMEM0R(PackedVector2Array, Vector2, empty);
	VCALL_PARRMEM2(PackedVector2Array, Vector2, set);
	VCALL_PARRMEM1R(PackedVector2Array, Vector2, get);
	VCALL_PARRMEM1(PackedVector2Array, Vector2, push_back);
	VCALL_PARRMEM1(PackedVector2Array, Vector2, resize);
	VCALL_PARRMEM2R(PackedVector2Array, Vector2, insert);
	VCALL_PARRMEM1(PackedVector2Array, Vector2, remove);
	VCALL_PARRMEM1(PackedVector2Array, Vector2, append);
	VCALL_PARRMEM1(PackedVector2Array, Vector2, append_array);
	VCALL_PARRMEM0(PackedVector2Array, Vector2, invert);

	VCALL_PARRMEM0R(PackedVector3Array, Vector3, size);
	VCALL_PARRMEM0R(PackedVector3Array, Vector3, empty);
	VCALL_PARRMEM2(PackedVector3Array, Vector3, set);
	VCALL_PARRMEM1R(PackedVector3Array, Vector3, get);
	VCALL_PARRMEM1(PackedVector3Array, Vector3, push_back);
	VCALL_PARRMEM1(PackedVector3Array, Vector3, resize);
	VCALL_PARRMEM2R(PackedVector3Array, Vector3, insert);
	VCALL_PARRMEM1(PackedVector3Array, Vector3, remove);
	VCALL_PARRMEM1(PackedVector3Array, Vector3, append);
	VCALL_PARRMEM1(PackedVector3Array, Vector3, append_array);
	VCALL_PARRMEM0(PackedVector3Array, Vector3, invert);

	VCALL_PARRMEM0R(PackedColorArray, Color, size);
	VCALL_PARRMEM0R(PackedColorArray, Color, empty);
	VCALL_PARRMEM2(PackedColorArray, Color, set);
	VCALL_PARRMEM1R(PackedColorArray, Color, get);
	VCALL_PARRMEM1(PackedColorArray, Color, push_back);
	VCALL_PARRMEM1(PackedColorArray, Color, resize);
	VCALL_PARRMEM2R(PackedColorArray, Color, insert);
	VCALL_PARRMEM1(PackedColorArray, Color, remove);
	VCALL_PARRMEM1(PackedColorArray, Color, append);
	VCALL_PARRMEM1(PackedColorArray, Color, append_array);
	VCALL_PARRMEM0(PackedColorArray, Color, invert);

#define VCALL_PTR0(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(); }
#define VCALL_PTR0R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(); }
#define VCALL_PTR1(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0]); }
#define VCALL_PTR1R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0]); }
#define VCALL_PTR2(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1]); }
#define VCALL_PTR2R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1]); }
#define VCALL_PTR3(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1], *p_args[2]); }
#define VCALL_PTR3R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1], *p_args[2]); }
#define VCALL_PTR4(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3]); }
#define VCALL_PTR4R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3]); }
#define VCALL_PTR5(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3], *p_args[4]); }
#define VCALL_PTR5R(m_type, m_method) \
	static void _call_##m_type##_##m_method(Variant &r_ret, Variant &p_self, const Variant **p_args) { r_ret = reinterpret_cast<m_type *>(p_self._data._ptr)->m_method(*p_args[0], *p_args[1], *p_args[2], *p_args[3], *p_args[4]); }

	VCALL_PTR0R(AABB, get_area);
	VCALL_PTR0R(AABB, has_no_area);
	VCALL_PTR0R(AABB, has_no_surface);
	VCALL_PTR1R(AABB, has_point);
	VCALL_PTR1R(AABB, is_equal_approx);
	VCALL_PTR1R(AABB, intersects);
	VCALL_PTR1R(AABB, encloses);
	VCALL_PTR1R(AABB, intersects_plane);
	VCALL_PTR2R(AABB, intersects_segment);
	VCALL_PTR1R(AABB, intersection);
	VCALL_PTR1R(AABB, merge);
	VCALL_PTR1R(AABB, expand);
	VCALL_PTR1R(AABB, grow);
	VCALL_PTR1R(AABB, get_support);
	VCALL_PTR0R(AABB, get_longest_axis);
	VCALL_PTR0R(AABB, get_longest_axis_index);
	VCALL_PTR0R(AABB, get_longest_axis_size);
	VCALL_PTR0R(AABB, get_shortest_axis);
	VCALL_PTR0R(AABB, get_shortest_axis_index);
	VCALL_PTR0R(AABB, get_shortest_axis_size);
	VCALL_PTR1R(AABB, get_endpoint);

	VCALL_PTR0R(Transform2D, inverse);
	VCALL_PTR0R(Transform2D, affine_inverse);
	VCALL_PTR0R(Transform2D, get_rotation);
	VCALL_PTR0R(Transform2D, get_origin);
	VCALL_PTR0R(Transform2D, get_scale);
	VCALL_PTR0R(Transform2D, orthonormalized);
	VCALL_PTR1R(Transform2D, rotated);
	VCALL_PTR1R(Transform2D, scaled);
	VCALL_PTR1R(Transform2D, translated);
	VCALL_PTR2R(Transform2D, interpolate_with);
	VCALL_PTR1R(Transform2D, is_equal_approx);

	static void _call_Transform2D_xform(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		switch (p_args[0]->type) {

			case Variant::VECTOR2: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->xform(p_args[0]->operator Vector2()); return;
			case Variant::RECT2: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->xform(p_args[0]->operator Rect2()); return;
			case Variant::PACKED_VECTOR2_ARRAY: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->xform(p_args[0]->operator PackedVector2Array()); return;
			default: r_ret = Variant();
		}
	}

	static void _call_Transform2D_xform_inv(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		switch (p_args[0]->type) {

			case Variant::VECTOR2: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->xform_inv(p_args[0]->operator Vector2()); return;
			case Variant::RECT2: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->xform_inv(p_args[0]->operator Rect2()); return;
			case Variant::PACKED_VECTOR2_ARRAY: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->xform_inv(p_args[0]->operator PackedVector2Array()); return;
			default: r_ret = Variant();
		}
	}

	static void _call_Transform2D_basis_xform(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		switch (p_args[0]->type) {

			case Variant::VECTOR2: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->basis_xform(p_args[0]->operator Vector2()); return;
			default: r_ret = Variant();
		}
	}

	static void _call_Transform2D_basis_xform_inv(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		switch (p_args[0]->type) {

			case Variant::VECTOR2: r_ret = reinterpret_cast<Transform2D *>(p_self._data._ptr)->basis_xform_inv(p_args[0]->operator Vector2()); return;
			default: r_ret = Variant();
		}
	}

	VCALL_PTR0R(Basis, inverse);
	VCALL_PTR0R(Basis, transposed);
	VCALL_PTR0R(Basis, determinant);
	VCALL_PTR2R(Basis, rotated);
	VCALL_PTR1R(Basis, scaled);
	VCALL_PTR0R(Basis, get_scale);
	VCALL_PTR0R(Basis, get_euler);
	VCALL_PTR1R(Basis, tdotx);
	VCALL_PTR1R(Basis, tdoty);
	VCALL_PTR1R(Basis, tdotz);
	VCALL_PTR1R(Basis, xform);
	VCALL_PTR1R(Basis, xform_inv);
	VCALL_PTR0R(Basis, get_orthogonal_index);
	VCALL_PTR0R(Basis, orthonormalized);
	VCALL_PTR2R(Basis, slerp);
	VCALL_PTR2R(Basis, is_equal_approx); // TODO: Break compatibility in 4.0 to change this to an instance method (a.is_equal_approx(b) as VCALL_PTR1R) for consistency.
	VCALL_PTR0R(Basis, get_rotation_quat);

	VCALL_PTR0R(Transform, inverse);
	VCALL_PTR0R(Transform, affine_inverse);
	VCALL_PTR2R(Transform, rotated);
	VCALL_PTR1R(Transform, scaled);
	VCALL_PTR1R(Transform, translated);
	VCALL_PTR0R(Transform, orthonormalized);
	VCALL_PTR2R(Transform, looking_at);
	VCALL_PTR2R(Transform, interpolate_with);
	VCALL_PTR1R(Transform, is_equal_approx);

	static void _call_Transform_xform(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		switch (p_args[0]->type) {

			case Variant::VECTOR3: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform(p_args[0]->operator Vector3()); return;
			case Variant::PLANE: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform(p_args[0]->operator Plane()); return;
			case Variant::AABB: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform(p_args[0]->operator ::AABB()); return;
			case Variant::PACKED_VECTOR3_ARRAY: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform(p_args[0]->operator ::PackedVector3Array()); return;
			default: r_ret = Variant();
		}
	}

	static void _call_Transform_xform_inv(Variant &r_ret, Variant &p_self, const Variant **p_args) {

		switch (p_args[0]->type) {

			case Variant::VECTOR3: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform_inv(p_args[0]->operator Vector3()); return;
			case Variant::PLANE: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform_inv(p_args[0]->operator Plane()); return;
			case Variant::AABB: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform_inv(p_args[0]->operator ::AABB()); return;
			case Variant::PACKED_VECTOR3_ARRAY: r_ret = reinterpret_cast<Transform *>(p_self._data._ptr)->xform_inv(p_args[0]->operator ::PackedVector3Array()); return;
			default: r_ret = Variant();
		}
	}

	/*
	VCALL_PTR0( Transform, invert );
	VCALL_PTR0( Transform, affine_invert );
	VCALL_PTR2( Transform, rotate );
	VCALL_PTR1( Transform, scale );
	VCALL_PTR1( Transform, translate );
	VCALL_PTR0( Transform, orthonormalize ); */

	struct ConstructData {

		int arg_count;
		Vector<Variant::Type> arg_types;
		Vector<String> arg_names;
		VariantConstructFunc func;
	};

	struct ConstructFunc {

		List<ConstructData> constructors;
	};

	static ConstructFunc *construct_funcs;

	static void Vector2_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Vector2(*p_args[0], *p_args[1]);
	}

	static void Vector2i_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Vector2i(*p_args[0], *p_args[1]);
	}

	static void Rect2_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Rect2(*p_args[0], *p_args[1]);
	}

	static void Rect2_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Rect2(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Rect2i_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Rect2i(*p_args[0], *p_args[1]);
	}

	static void Rect2i_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Rect2i(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Transform2D_init2(Variant &r_ret, const Variant **p_args) {

		Transform2D m(*p_args[0], *p_args[1]);
		r_ret = m;
	}

	static void Transform2D_init3(Variant &r_ret, const Variant **p_args) {

		Transform2D m;
		m[0] = *p_args[0];
		m[1] = *p_args[1];
		m[2] = *p_args[2];
		r_ret = m;
	}

	static void Vector3_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Vector3(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Vector3i_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Vector3i(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Plane_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Plane(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Plane_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Plane(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Plane_init3(Variant &r_ret, const Variant **p_args) {

		r_ret = Plane(p_args[0]->operator Vector3(), p_args[1]->operator real_t());
	}
	static void Plane_init4(Variant &r_ret, const Variant **p_args) {

		r_ret = Plane(p_args[0]->operator Vector3(), p_args[1]->operator Vector3());
	}

	static void Quat_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Quat(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Quat_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Quat(((Vector3)(*p_args[0])), ((real_t)(*p_args[1])));
	}

	static void Quat_init3(Variant &r_ret, const Variant **p_args) {

		r_ret = Quat(((Vector3)(*p_args[0])));
	}

	static void Color_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = Color(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Color_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Color(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Color_init3(Variant &r_ret, const Variant **p_args) {

		r_ret = Color::html(*p_args[0]);
	}

	static void Color_init4(Variant &r_ret, const Variant **p_args) {

		r_ret = Color::hex(*p_args[0]);
	}

	static void AABB_init1(Variant &r_ret, const Variant **p_args) {

		r_ret = ::AABB(*p_args[0], *p_args[1]);
	}

	static void Basis_init1(Variant &r_ret, const Variant **p_args) {

		Basis m;
		m.set_axis(0, *p_args[0]);
		m.set_axis(1, *p_args[1]);
		m.set_axis(2, *p_args[2]);
		r_ret = m;
	}

	static void Basis_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Basis(p_args[0]->operator Vector3(), p_args[1]->operator real_t());
	}

	static void Transform_init1(Variant &r_ret, const Variant **p_args) {

		Transform t;
		t.basis.set_axis(0, *p_args[0]);
		t.basis.set_axis(1, *p_args[1]);
		t.basis.set_axis(2, *p_args[2]);
		t.origin = *p_args[3];
		r_ret = t;
	}

	static void Transform_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Transform(p_args[0]->operator Basis(), p_args[1]->operator Vector3());
	}

	static void Callable_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Callable(p_args[0]->operator ObjectID(), p_args[1]->operator String());
	}

	static void Signal_init2(Variant &r_ret, const Variant **p_args) {

		r_ret = Signal(p_args[0]->operator ObjectID(), p_args[1]->operator String());
	}

	static void add_constructor(VariantConstructFunc p_func, const Variant::Type p_type,
			const String &p_name1 = "", const Variant::Type p_type1 = Variant::NIL,
			const String &p_name2 = "", const Variant::Type p_type2 = Variant::NIL,
			const String &p_name3 = "", const Variant::Type p_type3 = Variant::NIL,
			const String &p_name4 = "", const Variant::Type p_type4 = Variant::NIL) {

		ConstructData cd;
		cd.func = p_func;
		cd.arg_count = 0;

		if (p_name1 == "")
			goto end;
		cd.arg_count++;
		cd.arg_names.push_back(p_name1);
		cd.arg_types.push_back(p_type1);

		if (p_name2 == "")
			goto end;
		cd.arg_count++;
		cd.arg_names.push_back(p_name2);
		cd.arg_types.push_back(p_type2);

		if (p_name3 == "")
			goto end;
		cd.arg_count++;
		cd.arg_names.push_back(p_name3);
		cd.arg_types.push_back(p_type3);

		if (p_name4 == "")
			goto end;
		cd.arg_count++;
		cd.arg_names.push_back(p_name4);
		cd.arg_types.push_back(p_type4);

	end:

		construct_funcs[p_type].constructors.push_back(cd);
	}

	struct ConstantData {

		Map<StringName, int> value;
#ifdef DEBUG_ENABLED
		List<StringName> value_ordered;
#endif
		Map<StringName, Variant> variant_value;
	};

	static ConstantData *constant_data;

	static void add_constant(int p_type, StringName p_constant_name, int p_constant_value) {

		constant_data[p_type].value[p_constant_name] = p_constant_value;
#ifdef DEBUG_ENABLED
		constant_data[p_type].value_ordered.push_back(p_constant_name);
#endif
	}

	static void add_variant_constant(int p_type, StringName p_constant_name, const Variant &p_constant_value) {

		constant_data[p_type].variant_value[p_constant_name] = p_constant_value;
	}
};

_VariantCall::TypeFunc *_VariantCall::type_funcs = nullptr;
_VariantCall::ConstructFunc *_VariantCall::construct_funcs = nullptr;
_VariantCall::ConstantData *_VariantCall::constant_data = nullptr;

Variant Variant::call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {

	Variant ret;
	call_ptr(p_method, p_args, p_argcount, &ret, r_error);
	return ret;
}

void Variant::call_ptr(const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Callable::CallError &r_error) {
	Variant ret;

	if (type == Variant::OBJECT) {
		//call object
		Object *obj = _get_obj().obj;
		if (!obj) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}
#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}

#endif
		ret = _get_obj().obj->call(p_method, p_args, p_argcount, r_error);

		//else if (type==Variant::METHOD) {

	} else {

		r_error.error = Callable::CallError::CALL_OK;

		Map<StringName, _VariantCall::FuncData>::Element *E = _VariantCall::type_funcs[type].functions.find(p_method);

		if (E) {

			_VariantCall::FuncData &funcdata = E->get();
			funcdata.call(ret, *this, p_args, p_argcount, r_error);

		} else {
			//handle vararg functions manually
			bool valid = false;
			if (type == CALLABLE) {
				if (p_method == CoreStringNames::get_singleton()->call) {

					reinterpret_cast<const Callable *>(_data._mem)->call(p_args, p_argcount, ret, r_error);
					valid = true;
				}
				if (p_method == CoreStringNames::get_singleton()->call_deferred) {
					reinterpret_cast<const Callable *>(_data._mem)->call_deferred(p_args, p_argcount);
					valid = true;
				}
			} else if (type == SIGNAL) {
				if (p_method == CoreStringNames::get_singleton()->emit) {
					if (r_ret) {
						*r_ret = Variant();
					}
					reinterpret_cast<const Signal *>(_data._mem)->emit(p_args, p_argcount);
					valid = true;
				}
			}
			if (!valid) {
				//ok fail because not found
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				return;
			}
		}
	}

	if (r_error.error == Callable::CallError::CALL_OK && r_ret)
		*r_ret = ret;
}

#define VCALL(m_type, m_method) _VariantCall::_call_##m_type##_##m_method

Variant Variant::construct(const Variant::Type p_type, const Variant **p_args, int p_argcount, Callable::CallError &r_error, bool p_strict) {

	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	ERR_FAIL_INDEX_V(p_type, VARIANT_MAX, Variant());

	r_error.error = Callable::CallError::CALL_OK;
	if (p_argcount == 0) { //generic construct

		switch (p_type) {
			case NIL:
				return Variant();

			// atomic types
			case BOOL: return Variant(false);
			case INT: return 0;
			case FLOAT: return 0.0f;
			case STRING:
				return String();

			// math types
			case VECTOR2:
				return Vector2();
			case RECT2: return Rect2();
			case VECTOR3: return Vector3();
			case TRANSFORM2D: return Transform2D();
			case PLANE: return Plane();
			case QUAT: return Quat();
			case AABB:
				return ::AABB();
			case BASIS: return Basis();
			case TRANSFORM:
				return Transform();

			// misc types
			case COLOR: return Color();
			case STRING_NAME:
				return StringName();
			case NODE_PATH:
				return NodePath();
			case _RID: return RID();
			case OBJECT: return (Object *)nullptr;
			case CALLABLE: return Callable();
			case SIGNAL: return Signal();
			case DICTIONARY: return Dictionary();
			case ARRAY:
				return Array();
			case PACKED_BYTE_ARRAY: return PackedByteArray();
			case PACKED_INT32_ARRAY: return PackedInt32Array();
			case PACKED_INT64_ARRAY: return PackedInt64Array();
			case PACKED_FLOAT32_ARRAY: return PackedFloat32Array();
			case PACKED_FLOAT64_ARRAY: return PackedFloat64Array();
			case PACKED_STRING_ARRAY: return PackedStringArray();
			case PACKED_VECTOR2_ARRAY:
				return PackedVector2Array();
			case PACKED_VECTOR3_ARRAY: return PackedVector3Array();
			case PACKED_COLOR_ARRAY: return PackedColorArray();
			default: return Variant();
		}

	} else if (p_argcount == 1 && p_args[0]->type == p_type) {
		return *p_args[0]; //copy construct
	} else if (p_argcount == 1 && (!p_strict || Variant::can_convert(p_args[0]->type, p_type))) {
		//near match construct

		switch (p_type) {
			case NIL: {

				return Variant();
			} break;
			case BOOL: {
				return Variant(bool(*p_args[0]));
			}
			case INT: {
				return (int64_t(*p_args[0]));
			}
			case FLOAT: {
				return real_t(*p_args[0]);
			}
			case STRING: {
				return String(*p_args[0]);
			}
			case VECTOR2: {
				return Vector2(*p_args[0]);
			}
			case VECTOR2I: {
				return Vector2i(*p_args[0]);
			}
			case RECT2: return (Rect2(*p_args[0]));
			case RECT2I: return (Rect2i(*p_args[0]));
			case VECTOR3: return (Vector3(*p_args[0]));
			case VECTOR3I: return (Vector3i(*p_args[0]));
			case PLANE: return (Plane(*p_args[0]));
			case QUAT: return (p_args[0]->operator Quat());
			case AABB:
				return (::AABB(*p_args[0]));
			case BASIS: return (Basis(p_args[0]->operator Basis()));
			case TRANSFORM:
				return (Transform(p_args[0]->operator Transform()));

			// misc types
			case COLOR: return p_args[0]->type == Variant::STRING ? Color::html(*p_args[0]) : Color::hex(*p_args[0]);
			case STRING_NAME:
				return (StringName(p_args[0]->operator StringName()));
			case NODE_PATH:
				return (NodePath(p_args[0]->operator NodePath()));
			case _RID: return (RID(*p_args[0]));
			case OBJECT: return ((Object *)(p_args[0]->operator Object *()));
			case CALLABLE: return ((Callable)(p_args[0]->operator Callable()));
			case SIGNAL: return ((Signal)(p_args[0]->operator Signal()));
			case DICTIONARY: return p_args[0]->operator Dictionary();
			case ARRAY:
				return p_args[0]->operator Array();

			// arrays
			case PACKED_BYTE_ARRAY: return (PackedByteArray(*p_args[0]));
			case PACKED_INT32_ARRAY: return (PackedInt32Array(*p_args[0]));
			case PACKED_INT64_ARRAY: return (PackedInt64Array(*p_args[0]));
			case PACKED_FLOAT32_ARRAY: return (PackedFloat32Array(*p_args[0]));
			case PACKED_FLOAT64_ARRAY: return (PackedFloat64Array(*p_args[0]));
			case PACKED_STRING_ARRAY: return (PackedStringArray(*p_args[0]));
			case PACKED_VECTOR2_ARRAY:
				return (PackedVector2Array(*p_args[0]));
			case PACKED_VECTOR3_ARRAY: return (PackedVector3Array(*p_args[0]));
			case PACKED_COLOR_ARRAY: return (PackedColorArray(*p_args[0]));
			default: return Variant();
		}
	} else if (p_argcount >= 1) {

		_VariantCall::ConstructFunc &c = _VariantCall::construct_funcs[p_type];

		for (List<_VariantCall::ConstructData>::Element *E = c.constructors.front(); E; E = E->next()) {
			const _VariantCall::ConstructData &cd = E->get();

			if (cd.arg_count != p_argcount)
				continue;

			//validate parameters
			for (int i = 0; i < cd.arg_count; i++) {
				if (!Variant::can_convert(p_args[i]->type, cd.arg_types[i])) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT; //no such constructor
					r_error.argument = i;
					r_error.expected = cd.arg_types[i];
					return Variant();
				}
			}

			Variant v;
			cd.func(v, p_args);
			return v;
		}
	}
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; //no such constructor
	return Variant();
}

bool Variant::has_method(const StringName &p_method) const {

	if (type == OBJECT) {
		Object *obj = get_validated_object();
		if (!obj)
			return false;

		return obj->has_method(p_method);
	}

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[type];
	return tf.functions.has(p_method);
}

Vector<Variant::Type> Variant::get_method_argument_types(Variant::Type p_type, const StringName &p_method) {

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[p_type];

	const Map<StringName, _VariantCall::FuncData>::Element *E = tf.functions.find(p_method);
	if (!E)
		return Vector<Variant::Type>();

	return E->get().arg_types;
}

bool Variant::is_method_const(Variant::Type p_type, const StringName &p_method) {

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[p_type];

	const Map<StringName, _VariantCall::FuncData>::Element *E = tf.functions.find(p_method);
	if (!E)
		return false;

	return E->get()._const;
}

Vector<StringName> Variant::get_method_argument_names(Variant::Type p_type, const StringName &p_method) {

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[p_type];

	const Map<StringName, _VariantCall::FuncData>::Element *E = tf.functions.find(p_method);
	if (!E)
		return Vector<StringName>();

	return E->get().arg_names;
}

Variant::Type Variant::get_method_return_type(Variant::Type p_type, const StringName &p_method, bool *r_has_return) {

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[p_type];

	const Map<StringName, _VariantCall::FuncData>::Element *E = tf.functions.find(p_method);
	if (!E)
		return Variant::NIL;

	if (r_has_return)
		*r_has_return = E->get().returns;

	return E->get().return_type;
}

Vector<Variant> Variant::get_method_default_arguments(Variant::Type p_type, const StringName &p_method) {

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[p_type];

	const Map<StringName, _VariantCall::FuncData>::Element *E = tf.functions.find(p_method);
	if (!E)
		return Vector<Variant>();

	return E->get().default_args;
}

void Variant::get_method_list(List<MethodInfo> *p_list) const {

	const _VariantCall::TypeFunc &tf = _VariantCall::type_funcs[type];

	for (const Map<StringName, _VariantCall::FuncData>::Element *E = tf.functions.front(); E; E = E->next()) {

		const _VariantCall::FuncData &fd = E->get();

		MethodInfo mi;
		mi.name = E->key();

		if (fd._const) {
			mi.flags |= METHOD_FLAG_CONST;
		}

		for (int i = 0; i < fd.arg_types.size(); i++) {

			PropertyInfo pi;
			pi.type = fd.arg_types[i];
#ifdef DEBUG_ENABLED
			pi.name = fd.arg_names[i];
#endif
			mi.arguments.push_back(pi);
		}

		mi.default_arguments = fd.default_args;
		PropertyInfo ret;
#ifdef DEBUG_ENABLED
		ret.type = fd.return_type;
		if (fd.returns) {
			ret.name = "ret";
			if (fd.return_type == Variant::NIL)
				ret.usage = PROPERTY_USAGE_NIL_IS_VARIANT;
		}
		mi.return_val = ret;
#endif

		p_list->push_back(mi);
	}

	if (type == CALLABLE) {

		MethodInfo mi;
		mi.name = "call";
		mi.return_val.usage = PROPERTY_USAGE_NIL_IS_VARIANT;
		mi.flags |= METHOD_FLAG_VARARG;

		p_list->push_back(mi);

		mi.name = "call_deferred";
		mi.return_val.usage = 0;

		p_list->push_back(mi);
	}

	if (type == SIGNAL) {

		MethodInfo mi;
		mi.name = "emit";
		mi.flags |= METHOD_FLAG_VARARG;

		p_list->push_back(mi);
	}
}

void Variant::get_constructor_list(Variant::Type p_type, List<MethodInfo> *p_list) {

	ERR_FAIL_INDEX(p_type, VARIANT_MAX);

	//custom constructors
	for (const List<_VariantCall::ConstructData>::Element *E = _VariantCall::construct_funcs[p_type].constructors.front(); E; E = E->next()) {

		const _VariantCall::ConstructData &cd = E->get();
		MethodInfo mi;
		mi.name = Variant::get_type_name(p_type);
		mi.return_val.type = p_type;
		for (int i = 0; i < cd.arg_count; i++) {

			PropertyInfo pi;
			pi.name = cd.arg_names[i];
			pi.type = cd.arg_types[i];
			mi.arguments.push_back(pi);
		}
		p_list->push_back(mi);
	}
	//default constructors
	for (int i = 0; i < VARIANT_MAX; i++) {
		if (i == p_type)
			continue;
		if (!Variant::can_convert(Variant::Type(i), p_type))
			continue;

		MethodInfo mi;
		mi.name = Variant::get_type_name(p_type);
		PropertyInfo pi;
		pi.name = "from";
		pi.type = Variant::Type(i);
		mi.arguments.push_back(pi);
		mi.return_val.type = p_type;
		p_list->push_back(mi);
	}
}

void Variant::get_constants_for_type(Variant::Type p_type, List<StringName> *p_constants) {

	ERR_FAIL_INDEX(p_type, Variant::VARIANT_MAX);

	_VariantCall::ConstantData &cd = _VariantCall::constant_data[p_type];

#ifdef DEBUG_ENABLED
	for (List<StringName>::Element *E = cd.value_ordered.front(); E; E = E->next()) {

		p_constants->push_back(E->get());
#else
	for (Map<StringName, int>::Element *E = cd.value.front(); E; E = E->next()) {

		p_constants->push_back(E->key());
#endif
	}

	for (Map<StringName, Variant>::Element *E = cd.variant_value.front(); E; E = E->next()) {

		p_constants->push_back(E->key());
	}
}

bool Variant::has_constant(Variant::Type p_type, const StringName &p_value) {

	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, false);
	_VariantCall::ConstantData &cd = _VariantCall::constant_data[p_type];
	return cd.value.has(p_value) || cd.variant_value.has(p_value);
}

Variant Variant::get_constant_value(Variant::Type p_type, const StringName &p_value, bool *r_valid) {

	if (r_valid)
		*r_valid = false;

	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, 0);
	_VariantCall::ConstantData &cd = _VariantCall::constant_data[p_type];

	Map<StringName, int>::Element *E = cd.value.find(p_value);
	if (!E) {
		Map<StringName, Variant>::Element *F = cd.variant_value.find(p_value);
		if (F) {
			if (r_valid)
				*r_valid = true;
			return F->get();
		} else {
			return -1;
		}
	}
	if (r_valid)
		*r_valid = true;

	return E->get();
}

void register_variant_methods() {

	_VariantCall::type_funcs = memnew_arr(_VariantCall::TypeFunc, Variant::VARIANT_MAX);

	_VariantCall::construct_funcs = memnew_arr(_VariantCall::ConstructFunc, Variant::VARIANT_MAX);
	_VariantCall::constant_data = memnew_arr(_VariantCall::ConstantData, Variant::VARIANT_MAX);

#define ADDFUNC0R(m_vtype, m_ret, m_class, m_method, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg);
#define ADDFUNC1R(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)));
#define ADDFUNC2R(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)));
#define ADDFUNC3R(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)));
#define ADDFUNC4R(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_arg4, m_argname4, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)), _VariantCall::Arg(Variant::m_arg4, _scs_create(m_argname4)));

#define ADDFUNC0RNC(m_vtype, m_ret, m_class, m_method, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg);
#define ADDFUNC1RNC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)));
#define ADDFUNC2RNC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)));
#define ADDFUNC3RNC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)));
#define ADDFUNC4RNC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_arg4, m_argname4, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, true, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)), _VariantCall::Arg(Variant::m_arg4, _scs_create(m_argname4)));

#define ADDFUNC0(m_vtype, m_ret, m_class, m_method, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg);
#define ADDFUNC1(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)));
#define ADDFUNC2(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)));
#define ADDFUNC3(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)));
#define ADDFUNC4(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_arg4, m_argname4, m_defarg) \
	_VariantCall::addfunc(true, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)), _VariantCall::Arg(Variant::m_arg4, _scs_create(m_argname4)));

#define ADDFUNC0NC(m_vtype, m_ret, m_class, m_method, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg);
#define ADDFUNC1NC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)));
#define ADDFUNC2NC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)));
#define ADDFUNC3NC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)));
#define ADDFUNC4NC(m_vtype, m_ret, m_class, m_method, m_arg1, m_argname1, m_arg2, m_argname2, m_arg3, m_argname3, m_arg4, m_argname4, m_defarg) \
	_VariantCall::addfunc(false, Variant::m_vtype, Variant::m_ret, false, _scs_create(#m_method), VCALL(m_class, m_method), m_defarg, _VariantCall::Arg(Variant::m_arg1, _scs_create(m_argname1)), _VariantCall::Arg(Variant::m_arg2, _scs_create(m_argname2)), _VariantCall::Arg(Variant::m_arg3, _scs_create(m_argname3)), _VariantCall::Arg(Variant::m_arg4, _scs_create(m_argname4)));

	/* STRING */
	ADDFUNC1R(STRING, INT, String, casecmp_to, STRING, "to", varray());
	ADDFUNC1R(STRING, INT, String, nocasecmp_to, STRING, "to", varray());
	ADDFUNC0R(STRING, INT, String, length, varray());
	ADDFUNC2R(STRING, STRING, String, substr, INT, "from", INT, "len", varray(-1));

	ADDFUNC2R(STRING, INT, String, find, STRING, "what", INT, "from", varray(0));

	ADDFUNC3R(STRING, INT, String, count, STRING, "what", INT, "from", INT, "to", varray(0, 0));
	ADDFUNC3R(STRING, INT, String, countn, STRING, "what", INT, "from", INT, "to", varray(0, 0));

	ADDFUNC1R(STRING, INT, String, find_last, STRING, "what", varray());
	ADDFUNC2R(STRING, INT, String, findn, STRING, "what", INT, "from", varray(0));
	ADDFUNC2R(STRING, INT, String, rfind, STRING, "what", INT, "from", varray(-1));
	ADDFUNC2R(STRING, INT, String, rfindn, STRING, "what", INT, "from", varray(-1));
	ADDFUNC1R(STRING, BOOL, String, match, STRING, "expr", varray());
	ADDFUNC1R(STRING, BOOL, String, matchn, STRING, "expr", varray());
	ADDFUNC1R(STRING, BOOL, String, begins_with, STRING, "text", varray());
	ADDFUNC1R(STRING, BOOL, String, ends_with, STRING, "text", varray());
	ADDFUNC1R(STRING, BOOL, String, is_subsequence_of, STRING, "text", varray());
	ADDFUNC1R(STRING, BOOL, String, is_subsequence_ofi, STRING, "text", varray());
	ADDFUNC0R(STRING, PACKED_STRING_ARRAY, String, bigrams, varray());
	ADDFUNC1R(STRING, FLOAT, String, similarity, STRING, "text", varray());

	ADDFUNC2R(STRING, STRING, String, format, NIL, "values", STRING, "placeholder", varray("{_}"));
	ADDFUNC2R(STRING, STRING, String, replace, STRING, "what", STRING, "forwhat", varray());
	ADDFUNC2R(STRING, STRING, String, replacen, STRING, "what", STRING, "forwhat", varray());
	ADDFUNC1R(STRING, STRING, String, repeat, INT, "count", varray());
	ADDFUNC2R(STRING, STRING, String, insert, INT, "position", STRING, "what", varray());
	ADDFUNC0R(STRING, STRING, String, capitalize, varray());
	ADDFUNC3R(STRING, PACKED_STRING_ARRAY, String, split, STRING, "delimiter", BOOL, "allow_empty", INT, "maxsplit", varray(true, 0));
	ADDFUNC3R(STRING, PACKED_STRING_ARRAY, String, rsplit, STRING, "delimiter", BOOL, "allow_empty", INT, "maxsplit", varray(true, 0));
	ADDFUNC2R(STRING, PACKED_FLOAT32_ARRAY, String, split_floats, STRING, "delimiter", BOOL, "allow_empty", varray(true));

	ADDFUNC0R(STRING, STRING, String, to_upper, varray());
	ADDFUNC0R(STRING, STRING, String, to_lower, varray());

	ADDFUNC1R(STRING, STRING, String, left, INT, "position", varray());
	ADDFUNC1R(STRING, STRING, String, right, INT, "position", varray());
	ADDFUNC2R(STRING, STRING, String, strip_edges, BOOL, "left", BOOL, "right", varray(true, true));
	ADDFUNC0R(STRING, STRING, String, strip_escapes, varray());
	ADDFUNC1R(STRING, STRING, String, lstrip, STRING, "chars", varray());
	ADDFUNC1R(STRING, STRING, String, rstrip, STRING, "chars", varray());
	ADDFUNC0R(STRING, STRING, String, get_extension, varray());
	ADDFUNC0R(STRING, STRING, String, get_basename, varray());
	ADDFUNC1R(STRING, STRING, String, plus_file, STRING, "file", varray());
	ADDFUNC1R(STRING, INT, String, ord_at, INT, "at", varray());
	ADDFUNC0R(STRING, STRING, String, dedent, varray());
	ADDFUNC2(STRING, NIL, String, erase, INT, "position", INT, "chars", varray());
	ADDFUNC0R(STRING, INT, String, hash, varray());
	ADDFUNC0R(STRING, STRING, String, md5_text, varray());
	ADDFUNC0R(STRING, STRING, String, sha1_text, varray());
	ADDFUNC0R(STRING, STRING, String, sha256_text, varray());
	ADDFUNC0R(STRING, PACKED_BYTE_ARRAY, String, md5_buffer, varray());
	ADDFUNC0R(STRING, PACKED_BYTE_ARRAY, String, sha1_buffer, varray());
	ADDFUNC0R(STRING, PACKED_BYTE_ARRAY, String, sha256_buffer, varray());
	ADDFUNC0R(STRING, BOOL, String, empty, varray());
	ADDFUNC1R(STRING, STRING, String, humanize_size, INT, "size", varray());
	ADDFUNC0R(STRING, BOOL, String, is_abs_path, varray());
	ADDFUNC0R(STRING, BOOL, String, is_rel_path, varray());
	ADDFUNC0R(STRING, STRING, String, get_base_dir, varray());
	ADDFUNC0R(STRING, STRING, String, get_file, varray());
	ADDFUNC0R(STRING, STRING, String, xml_escape, varray());
	ADDFUNC0R(STRING, STRING, String, xml_unescape, varray());
	ADDFUNC0R(STRING, STRING, String, http_escape, varray());
	ADDFUNC0R(STRING, STRING, String, http_unescape, varray());
	ADDFUNC0R(STRING, STRING, String, c_escape, varray());
	ADDFUNC0R(STRING, STRING, String, c_unescape, varray());
	ADDFUNC0R(STRING, STRING, String, json_escape, varray());
	ADDFUNC0R(STRING, STRING, String, percent_encode, varray());
	ADDFUNC0R(STRING, STRING, String, percent_decode, varray());
	ADDFUNC0R(STRING, BOOL, String, is_valid_identifier, varray());
	ADDFUNC0R(STRING, BOOL, String, is_valid_integer, varray());
	ADDFUNC0R(STRING, BOOL, String, is_valid_float, varray());
	ADDFUNC1R(STRING, BOOL, String, is_valid_hex_number, BOOL, "with_prefix", varray(false));
	ADDFUNC0R(STRING, BOOL, String, is_valid_html_color, varray());
	ADDFUNC0R(STRING, BOOL, String, is_valid_ip_address, varray());
	ADDFUNC0R(STRING, BOOL, String, is_valid_filename, varray());
	ADDFUNC0R(STRING, INT, String, to_int, varray());
	ADDFUNC0R(STRING, FLOAT, String, to_float, varray());
	ADDFUNC0R(STRING, INT, String, hex_to_int, varray());
	ADDFUNC1R(STRING, STRING, String, pad_decimals, INT, "digits", varray());
	ADDFUNC1R(STRING, STRING, String, pad_zeros, INT, "digits", varray());
	ADDFUNC1R(STRING, STRING, String, trim_prefix, STRING, "prefix", varray());
	ADDFUNC1R(STRING, STRING, String, trim_suffix, STRING, "suffix", varray());

	ADDFUNC0R(STRING, PACKED_BYTE_ARRAY, String, to_ascii, varray());
	ADDFUNC0R(STRING, PACKED_BYTE_ARRAY, String, to_utf8, varray());

	ADDFUNC0R(VECTOR2, FLOAT, Vector2, angle, varray());
	ADDFUNC1R(VECTOR2, FLOAT, Vector2, angle_to, VECTOR2, "to", varray());
	ADDFUNC1R(VECTOR2, FLOAT, Vector2, angle_to_point, VECTOR2, "to", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, direction_to, VECTOR2, "b", varray());
	ADDFUNC1R(VECTOR2, FLOAT, Vector2, distance_to, VECTOR2, "to", varray());
	ADDFUNC1R(VECTOR2, FLOAT, Vector2, distance_squared_to, VECTOR2, "to", varray());
	ADDFUNC0R(VECTOR2, FLOAT, Vector2, length, varray());
	ADDFUNC0R(VECTOR2, FLOAT, Vector2, length_squared, varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, normalized, varray());
	ADDFUNC0R(VECTOR2, BOOL, Vector2, is_normalized, varray());
	ADDFUNC1R(VECTOR2, BOOL, Vector2, is_equal_approx, VECTOR2, "v", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, posmod, FLOAT, "mod", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, posmodv, VECTOR2, "modv", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, project, VECTOR2, "b", varray());
	ADDFUNC2R(VECTOR2, VECTOR2, Vector2, linear_interpolate, VECTOR2, "b", FLOAT, "t", varray());
	ADDFUNC2R(VECTOR2, VECTOR2, Vector2, slerp, VECTOR2, "b", FLOAT, "t", varray());
	ADDFUNC4R(VECTOR2, VECTOR2, Vector2, cubic_interpolate, VECTOR2, "b", VECTOR2, "pre_a", VECTOR2, "post_b", FLOAT, "t", varray());
	ADDFUNC2R(VECTOR2, VECTOR2, Vector2, move_toward, VECTOR2, "to", FLOAT, "delta", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, rotated, FLOAT, "phi", varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, tangent, varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, floor, varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, ceil, varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, round, varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, snapped, VECTOR2, "by", varray());
	ADDFUNC0R(VECTOR2, FLOAT, Vector2, aspect, varray());
	ADDFUNC1R(VECTOR2, FLOAT, Vector2, dot, VECTOR2, "with", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, slide, VECTOR2, "n", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, bounce, VECTOR2, "n", varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, reflect, VECTOR2, "n", varray());
	ADDFUNC1R(VECTOR2, FLOAT, Vector2, cross, VECTOR2, "with", varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, abs, varray());
	ADDFUNC1R(VECTOR2, VECTOR2, Vector2, clamped, FLOAT, "length", varray());
	ADDFUNC0R(VECTOR2, VECTOR2, Vector2, sign, varray());

	ADDFUNC0R(VECTOR2I, FLOAT, Vector2i, aspect, varray());
	ADDFUNC0R(VECTOR2I, VECTOR2I, Vector2i, sign, varray());
	ADDFUNC0R(VECTOR2I, VECTOR2I, Vector2i, abs, varray());

	ADDFUNC0R(RECT2, FLOAT, Rect2, get_area, varray());
	ADDFUNC0R(RECT2, BOOL, Rect2, has_no_area, varray());
	ADDFUNC1R(RECT2, BOOL, Rect2, has_point, VECTOR2, "point", varray());
	ADDFUNC1R(RECT2, BOOL, Rect2, is_equal_approx, RECT2, "rect", varray());
	ADDFUNC2R(RECT2, BOOL, Rect2, intersects, RECT2, "b", BOOL, "include_borders", varray(false));
	ADDFUNC1R(RECT2, BOOL, Rect2, encloses, RECT2, "b", varray());
	ADDFUNC1R(RECT2, RECT2, Rect2, clip, RECT2, "b", varray());
	ADDFUNC1R(RECT2, RECT2, Rect2, merge, RECT2, "b", varray());
	ADDFUNC1R(RECT2, RECT2, Rect2, expand, VECTOR2, "to", varray());
	ADDFUNC1R(RECT2, RECT2, Rect2, grow, FLOAT, "by", varray());
	ADDFUNC2R(RECT2, RECT2, Rect2, grow_margin, INT, "margin", FLOAT, "by", varray());
	ADDFUNC4R(RECT2, RECT2, Rect2, grow_individual, FLOAT, "left", FLOAT, "top", FLOAT, "right", FLOAT, " bottom", varray());
	ADDFUNC0R(RECT2, RECT2, Rect2, abs, varray());

	ADDFUNC0R(RECT2I, INT, Rect2i, get_area, varray());
	ADDFUNC0R(RECT2I, BOOL, Rect2i, has_no_area, varray());
	ADDFUNC1R(RECT2I, BOOL, Rect2i, has_point, VECTOR2I, "point", varray());
	ADDFUNC1R(RECT2I, BOOL, Rect2i, intersects, RECT2I, "b", varray());
	ADDFUNC1R(RECT2I, BOOL, Rect2i, encloses, RECT2I, "b", varray());
	ADDFUNC1R(RECT2I, RECT2I, Rect2i, clip, RECT2I, "b", varray());
	ADDFUNC1R(RECT2I, RECT2I, Rect2i, merge, RECT2I, "b", varray());
	ADDFUNC1R(RECT2I, RECT2I, Rect2i, expand, VECTOR2I, "to", varray());
	ADDFUNC1R(RECT2I, RECT2I, Rect2i, grow, INT, "by", varray());
	ADDFUNC2R(RECT2I, RECT2I, Rect2i, grow_margin, INT, "margin", INT, "by", varray());
	ADDFUNC4R(RECT2I, RECT2I, Rect2i, grow_individual, INT, "left", INT, "top", INT, "right", INT, " bottom", varray());
	ADDFUNC0R(RECT2I, RECT2I, Rect2i, abs, varray());

	ADDFUNC0R(VECTOR3, INT, Vector3, min_axis, varray());
	ADDFUNC0R(VECTOR3, INT, Vector3, max_axis, varray());
	ADDFUNC1R(VECTOR3, FLOAT, Vector3, angle_to, VECTOR3, "to", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, direction_to, VECTOR3, "b", varray());
	ADDFUNC1R(VECTOR3, FLOAT, Vector3, distance_to, VECTOR3, "b", varray());
	ADDFUNC1R(VECTOR3, FLOAT, Vector3, distance_squared_to, VECTOR3, "b", varray());
	ADDFUNC0R(VECTOR3, FLOAT, Vector3, length, varray());
	ADDFUNC0R(VECTOR3, FLOAT, Vector3, length_squared, varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, normalized, varray());
	ADDFUNC0R(VECTOR3, BOOL, Vector3, is_normalized, varray());
	ADDFUNC1R(VECTOR3, BOOL, Vector3, is_equal_approx, VECTOR3, "v", varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, inverse, varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, snapped, VECTOR3, "by", varray());
	ADDFUNC2R(VECTOR3, VECTOR3, Vector3, rotated, VECTOR3, "axis", FLOAT, "phi", varray());
	ADDFUNC2R(VECTOR3, VECTOR3, Vector3, linear_interpolate, VECTOR3, "b", FLOAT, "t", varray());
	ADDFUNC2R(VECTOR3, VECTOR3, Vector3, slerp, VECTOR3, "b", FLOAT, "t", varray());
	ADDFUNC4R(VECTOR3, VECTOR3, Vector3, cubic_interpolate, VECTOR3, "b", VECTOR3, "pre_a", VECTOR3, "post_b", FLOAT, "t", varray());
	ADDFUNC2R(VECTOR3, VECTOR3, Vector3, move_toward, VECTOR3, "to", FLOAT, "delta", varray());
	ADDFUNC1R(VECTOR3, FLOAT, Vector3, dot, VECTOR3, "b", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, cross, VECTOR3, "b", varray());
	ADDFUNC1R(VECTOR3, BASIS, Vector3, outer, VECTOR3, "b", varray());
	ADDFUNC0R(VECTOR3, BASIS, Vector3, to_diagonal_matrix, varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, abs, varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, floor, varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, ceil, varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, round, varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, posmod, FLOAT, "mod", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, posmodv, VECTOR3, "modv", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, project, VECTOR3, "b", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, slide, VECTOR3, "n", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, bounce, VECTOR3, "n", varray());
	ADDFUNC1R(VECTOR3, VECTOR3, Vector3, reflect, VECTOR3, "n", varray());
	ADDFUNC0R(VECTOR3, VECTOR3, Vector3, sign, varray());

	ADDFUNC0R(VECTOR3I, INT, Vector3i, min_axis, varray());
	ADDFUNC0R(VECTOR3I, INT, Vector3i, max_axis, varray());
	ADDFUNC0R(VECTOR3I, VECTOR3I, Vector3i, sign, varray());

	ADDFUNC0R(PLANE, PLANE, Plane, normalized, varray());
	ADDFUNC0R(PLANE, VECTOR3, Plane, center, varray());
	ADDFUNC0R(PLANE, VECTOR3, Plane, get_any_point, varray());
	ADDFUNC1R(PLANE, BOOL, Plane, is_equal_approx, PLANE, "plane", varray());
	ADDFUNC1R(PLANE, BOOL, Plane, is_point_over, VECTOR3, "point", varray());
	ADDFUNC1R(PLANE, FLOAT, Plane, distance_to, VECTOR3, "point", varray());
	ADDFUNC2R(PLANE, BOOL, Plane, has_point, VECTOR3, "point", FLOAT, "epsilon", varray(CMP_EPSILON));
	ADDFUNC1R(PLANE, VECTOR3, Plane, project, VECTOR3, "point", varray());
	ADDFUNC2R(PLANE, VECTOR3, Plane, intersect_3, PLANE, "b", PLANE, "c", varray());
	ADDFUNC2R(PLANE, VECTOR3, Plane, intersects_ray, VECTOR3, "from", VECTOR3, "dir", varray());
	ADDFUNC2R(PLANE, VECTOR3, Plane, intersects_segment, VECTOR3, "begin", VECTOR3, "end", varray());

	ADDFUNC0R(QUAT, FLOAT, Quat, length, varray());
	ADDFUNC0R(QUAT, FLOAT, Quat, length_squared, varray());
	ADDFUNC0R(QUAT, QUAT, Quat, normalized, varray());
	ADDFUNC0R(QUAT, BOOL, Quat, is_normalized, varray());
	ADDFUNC1R(QUAT, BOOL, Quat, is_equal_approx, QUAT, "quat", varray());
	ADDFUNC0R(QUAT, QUAT, Quat, inverse, varray());
	ADDFUNC1R(QUAT, FLOAT, Quat, dot, QUAT, "b", varray());
	ADDFUNC1R(QUAT, VECTOR3, Quat, xform, VECTOR3, "v", varray());
	ADDFUNC2R(QUAT, QUAT, Quat, slerp, QUAT, "b", FLOAT, "t", varray());
	ADDFUNC2R(QUAT, QUAT, Quat, slerpni, QUAT, "b", FLOAT, "t", varray());
	ADDFUNC4R(QUAT, QUAT, Quat, cubic_slerp, QUAT, "b", QUAT, "pre_a", QUAT, "post_b", FLOAT, "t", varray());
	ADDFUNC0R(QUAT, VECTOR3, Quat, get_euler, varray());
	ADDFUNC1(QUAT, NIL, Quat, set_euler, VECTOR3, "euler", varray());
	ADDFUNC2(QUAT, NIL, Quat, set_axis_angle, VECTOR3, "axis", FLOAT, "angle", varray());

	ADDFUNC0R(COLOR, INT, Color, to_argb32, varray());
	ADDFUNC0R(COLOR, INT, Color, to_abgr32, varray());
	ADDFUNC0R(COLOR, INT, Color, to_rgba32, varray());
	ADDFUNC0R(COLOR, INT, Color, to_argb64, varray());
	ADDFUNC0R(COLOR, INT, Color, to_abgr64, varray());
	ADDFUNC0R(COLOR, INT, Color, to_rgba64, varray());
	ADDFUNC0R(COLOR, COLOR, Color, inverted, varray());
	ADDFUNC0R(COLOR, COLOR, Color, contrasted, varray());
	ADDFUNC2R(COLOR, COLOR, Color, linear_interpolate, COLOR, "b", FLOAT, "t", varray());
	ADDFUNC1R(COLOR, COLOR, Color, blend, COLOR, "over", varray());
	ADDFUNC1R(COLOR, COLOR, Color, lightened, FLOAT, "amount", varray());
	ADDFUNC1R(COLOR, COLOR, Color, darkened, FLOAT, "amount", varray());
	ADDFUNC1R(COLOR, STRING, Color, to_html, BOOL, "with_alpha", varray(true));
	ADDFUNC4R(COLOR, COLOR, Color, from_hsv, FLOAT, "h", FLOAT, "s", FLOAT, "v", FLOAT, "a", varray(1.0));
	ADDFUNC1R(COLOR, BOOL, Color, is_equal_approx, COLOR, "color", varray());

	ADDFUNC0R(_RID, INT, RID, get_id, varray());

	ADDFUNC0R(NODE_PATH, BOOL, NodePath, is_absolute, varray());
	ADDFUNC0R(NODE_PATH, INT, NodePath, get_name_count, varray());
	ADDFUNC1R(NODE_PATH, STRING, NodePath, get_name, INT, "idx", varray());
	ADDFUNC0R(NODE_PATH, INT, NodePath, get_subname_count, varray());
	ADDFUNC1R(NODE_PATH, STRING, NodePath, get_subname, INT, "idx", varray());
	ADDFUNC0R(NODE_PATH, STRING, NodePath, get_concatenated_subnames, varray());
	ADDFUNC0R(NODE_PATH, NODE_PATH, NodePath, get_as_property_path, varray());
	ADDFUNC0R(NODE_PATH, BOOL, NodePath, is_empty, varray());

	ADDFUNC0R(DICTIONARY, INT, Dictionary, size, varray());
	ADDFUNC0R(DICTIONARY, BOOL, Dictionary, empty, varray());
	ADDFUNC0NC(DICTIONARY, NIL, Dictionary, clear, varray());
	ADDFUNC1R(DICTIONARY, BOOL, Dictionary, has, NIL, "key", varray());
	ADDFUNC1R(DICTIONARY, BOOL, Dictionary, has_all, ARRAY, "keys", varray());
	ADDFUNC1RNC(DICTIONARY, BOOL, Dictionary, erase, NIL, "key", varray());
	ADDFUNC0R(DICTIONARY, INT, Dictionary, hash, varray());
	ADDFUNC0R(DICTIONARY, ARRAY, Dictionary, keys, varray());
	ADDFUNC0R(DICTIONARY, ARRAY, Dictionary, values, varray());
	ADDFUNC1R(DICTIONARY, DICTIONARY, Dictionary, duplicate, BOOL, "deep", varray(false));
	ADDFUNC2R(DICTIONARY, NIL, Dictionary, get, NIL, "key", NIL, "default", varray(Variant()));

	ADDFUNC0R(CALLABLE, BOOL, Callable, is_null, varray());
	ADDFUNC0R(CALLABLE, BOOL, Callable, is_custom, varray());
	ADDFUNC0R(CALLABLE, BOOL, Callable, is_standard, varray());
	ADDFUNC0R(CALLABLE, OBJECT, Callable, get_object, varray());
	ADDFUNC0R(CALLABLE, INT, Callable, get_object_id, varray());
	ADDFUNC0R(CALLABLE, STRING_NAME, Callable, get_method, varray());
	ADDFUNC0R(CALLABLE, INT, Callable, hash, varray());

	ADDFUNC0R(SIGNAL, BOOL, Signal, is_null, varray());
	ADDFUNC0R(SIGNAL, OBJECT, Signal, get_object, varray());
	ADDFUNC0R(SIGNAL, INT, Signal, get_object_id, varray());
	ADDFUNC0R(SIGNAL, STRING_NAME, Signal, get_name, varray());

	ADDFUNC3R(SIGNAL, INT, Signal, connect, CALLABLE, "callable", ARRAY, "binds", INT, "flags", varray(Array(), 0));

	ADDFUNC1R(SIGNAL, NIL, Signal, disconnect, CALLABLE, "callable", varray());
	ADDFUNC1R(SIGNAL, BOOL, Signal, is_connected, CALLABLE, "callable", varray());
	ADDFUNC0R(SIGNAL, ARRAY, Signal, get_connections, varray());

	ADDFUNC0R(ARRAY, INT, Array, size, varray());
	ADDFUNC0R(ARRAY, BOOL, Array, empty, varray());
	ADDFUNC0NC(ARRAY, NIL, Array, clear, varray());
	ADDFUNC0R(ARRAY, INT, Array, hash, varray());
	ADDFUNC1NC(ARRAY, NIL, Array, push_back, NIL, "value", varray());
	ADDFUNC1NC(ARRAY, NIL, Array, push_front, NIL, "value", varray());
	ADDFUNC1NC(ARRAY, NIL, Array, append, NIL, "value", varray());
	ADDFUNC1NC(ARRAY, NIL, Array, resize, INT, "size", varray());
	ADDFUNC2NC(ARRAY, NIL, Array, insert, INT, "position", NIL, "value", varray());
	ADDFUNC1NC(ARRAY, NIL, Array, remove, INT, "position", varray());
	ADDFUNC1NC(ARRAY, NIL, Array, erase, NIL, "value", varray());
	ADDFUNC0R(ARRAY, NIL, Array, front, varray());
	ADDFUNC0R(ARRAY, NIL, Array, back, varray());
	ADDFUNC2R(ARRAY, INT, Array, find, NIL, "what", INT, "from", varray(0));
	ADDFUNC2R(ARRAY, INT, Array, rfind, NIL, "what", INT, "from", varray(-1));
	ADDFUNC1R(ARRAY, INT, Array, find_last, NIL, "value", varray());
	ADDFUNC1R(ARRAY, INT, Array, count, NIL, "value", varray());
	ADDFUNC1R(ARRAY, BOOL, Array, has, NIL, "value", varray());
	ADDFUNC0RNC(ARRAY, NIL, Array, pop_back, varray());
	ADDFUNC0RNC(ARRAY, NIL, Array, pop_front, varray());
	ADDFUNC0NC(ARRAY, NIL, Array, sort, varray());
	ADDFUNC2NC(ARRAY, NIL, Array, sort_custom, OBJECT, "obj", STRING, "func", varray());
	ADDFUNC0NC(ARRAY, NIL, Array, shuffle, varray());
	ADDFUNC2R(ARRAY, INT, Array, bsearch, NIL, "value", BOOL, "before", varray(true));
	ADDFUNC4R(ARRAY, INT, Array, bsearch_custom, NIL, "value", OBJECT, "obj", STRING, "func", BOOL, "before", varray(true));
	ADDFUNC0NC(ARRAY, NIL, Array, invert, varray());
	ADDFUNC1R(ARRAY, ARRAY, Array, duplicate, BOOL, "deep", varray(false));
	ADDFUNC4R(ARRAY, ARRAY, Array, slice, INT, "begin", INT, "end", INT, "step", BOOL, "deep", varray(1, false));
	ADDFUNC0R(ARRAY, NIL, Array, max, varray());
	ADDFUNC0R(ARRAY, NIL, Array, min, varray());

	ADDFUNC0R(PACKED_BYTE_ARRAY, INT, PackedByteArray, size, varray());
	ADDFUNC0R(PACKED_BYTE_ARRAY, BOOL, PackedByteArray, empty, varray());
	ADDFUNC2(PACKED_BYTE_ARRAY, NIL, PackedByteArray, set, INT, "idx", INT, "byte", varray());
	ADDFUNC1(PACKED_BYTE_ARRAY, NIL, PackedByteArray, push_back, INT, "byte", varray());
	ADDFUNC1(PACKED_BYTE_ARRAY, NIL, PackedByteArray, append, INT, "byte", varray());
	ADDFUNC1(PACKED_BYTE_ARRAY, NIL, PackedByteArray, append_array, PACKED_BYTE_ARRAY, "array", varray());
	ADDFUNC1(PACKED_BYTE_ARRAY, NIL, PackedByteArray, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_BYTE_ARRAY, INT, PackedByteArray, insert, INT, "idx", INT, "byte", varray());
	ADDFUNC1(PACKED_BYTE_ARRAY, NIL, PackedByteArray, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_BYTE_ARRAY, NIL, PackedByteArray, invert, varray());
	ADDFUNC2R(PACKED_BYTE_ARRAY, PACKED_BYTE_ARRAY, PackedByteArray, subarray, INT, "from", INT, "to", varray());

	ADDFUNC0R(PACKED_BYTE_ARRAY, STRING, PackedByteArray, get_string_from_ascii, varray());
	ADDFUNC0R(PACKED_BYTE_ARRAY, STRING, PackedByteArray, get_string_from_utf8, varray());
	ADDFUNC0R(PACKED_BYTE_ARRAY, STRING, PackedByteArray, hex_encode, varray());
	ADDFUNC1R(PACKED_BYTE_ARRAY, PACKED_BYTE_ARRAY, PackedByteArray, compress, INT, "compression_mode", varray(0));
	ADDFUNC2R(PACKED_BYTE_ARRAY, PACKED_BYTE_ARRAY, PackedByteArray, decompress, INT, "buffer_size", INT, "compression_mode", varray(0));

	ADDFUNC0R(PACKED_INT32_ARRAY, INT, PackedInt32Array, size, varray());
	ADDFUNC0R(PACKED_INT32_ARRAY, BOOL, PackedInt32Array, empty, varray());
	ADDFUNC2(PACKED_INT32_ARRAY, NIL, PackedInt32Array, set, INT, "idx", INT, "integer", varray());
	ADDFUNC1(PACKED_INT32_ARRAY, NIL, PackedInt32Array, push_back, INT, "integer", varray());
	ADDFUNC1(PACKED_INT32_ARRAY, NIL, PackedInt32Array, append, INT, "integer", varray());
	ADDFUNC1(PACKED_INT32_ARRAY, NIL, PackedInt32Array, append_array, PACKED_INT32_ARRAY, "array", varray());
	ADDFUNC1(PACKED_INT32_ARRAY, NIL, PackedInt32Array, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_INT32_ARRAY, INT, PackedInt32Array, insert, INT, "idx", INT, "integer", varray());
	ADDFUNC1(PACKED_INT32_ARRAY, NIL, PackedInt32Array, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_INT32_ARRAY, NIL, PackedInt32Array, invert, varray());

	ADDFUNC0R(PACKED_INT64_ARRAY, INT, PackedInt64Array, size, varray());
	ADDFUNC0R(PACKED_INT64_ARRAY, BOOL, PackedInt64Array, empty, varray());
	ADDFUNC2(PACKED_INT64_ARRAY, NIL, PackedInt64Array, set, INT, "idx", INT, "integer", varray());
	ADDFUNC1(PACKED_INT64_ARRAY, NIL, PackedInt64Array, push_back, INT, "integer", varray());
	ADDFUNC1(PACKED_INT64_ARRAY, NIL, PackedInt64Array, append, INT, "integer", varray());
	ADDFUNC1(PACKED_INT64_ARRAY, NIL, PackedInt64Array, append_array, PACKED_INT64_ARRAY, "array", varray());
	ADDFUNC1(PACKED_INT64_ARRAY, NIL, PackedInt64Array, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_INT64_ARRAY, INT, PackedInt64Array, insert, INT, "idx", INT, "integer", varray());
	ADDFUNC1(PACKED_INT64_ARRAY, NIL, PackedInt64Array, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_INT64_ARRAY, NIL, PackedInt64Array, invert, varray());

	ADDFUNC0R(PACKED_FLOAT32_ARRAY, INT, PackedFloat32Array, size, varray());
	ADDFUNC0R(PACKED_FLOAT32_ARRAY, BOOL, PackedFloat32Array, empty, varray());
	ADDFUNC2(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, set, INT, "idx", FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, push_back, FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, append, FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, append_array, PACKED_FLOAT32_ARRAY, "array", varray());
	ADDFUNC1(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_FLOAT32_ARRAY, INT, PackedFloat32Array, insert, INT, "idx", FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_FLOAT32_ARRAY, NIL, PackedFloat32Array, invert, varray());

	ADDFUNC0R(PACKED_FLOAT64_ARRAY, INT, PackedFloat64Array, size, varray());
	ADDFUNC0R(PACKED_FLOAT64_ARRAY, BOOL, PackedFloat64Array, empty, varray());
	ADDFUNC2(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, set, INT, "idx", FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, push_back, FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, append, FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, append_array, PACKED_FLOAT64_ARRAY, "array", varray());
	ADDFUNC1(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_FLOAT64_ARRAY, INT, PackedFloat64Array, insert, INT, "idx", FLOAT, "value", varray());
	ADDFUNC1(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_FLOAT64_ARRAY, NIL, PackedFloat64Array, invert, varray());

	ADDFUNC0R(PACKED_STRING_ARRAY, INT, PackedStringArray, size, varray());
	ADDFUNC0R(PACKED_STRING_ARRAY, BOOL, PackedStringArray, empty, varray());
	ADDFUNC2(PACKED_STRING_ARRAY, NIL, PackedStringArray, set, INT, "idx", STRING, "string", varray());
	ADDFUNC1(PACKED_STRING_ARRAY, NIL, PackedStringArray, push_back, STRING, "string", varray());
	ADDFUNC1(PACKED_STRING_ARRAY, NIL, PackedStringArray, append, STRING, "string", varray());
	ADDFUNC1(PACKED_STRING_ARRAY, NIL, PackedStringArray, append_array, PACKED_STRING_ARRAY, "array", varray());
	ADDFUNC1(PACKED_STRING_ARRAY, NIL, PackedStringArray, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_STRING_ARRAY, INT, PackedStringArray, insert, INT, "idx", STRING, "string", varray());
	ADDFUNC1(PACKED_STRING_ARRAY, NIL, PackedStringArray, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_STRING_ARRAY, NIL, PackedStringArray, invert, varray());

	ADDFUNC0R(PACKED_VECTOR2_ARRAY, INT, PackedVector2Array, size, varray());
	ADDFUNC0R(PACKED_VECTOR2_ARRAY, BOOL, PackedVector2Array, empty, varray());
	ADDFUNC2(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, set, INT, "idx", VECTOR2, "vector2", varray());
	ADDFUNC1(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, push_back, VECTOR2, "vector2", varray());
	ADDFUNC1(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, append, VECTOR2, "vector2", varray());
	ADDFUNC1(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, append_array, PACKED_VECTOR2_ARRAY, "array", varray());
	ADDFUNC1(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_VECTOR2_ARRAY, INT, PackedVector2Array, insert, INT, "idx", VECTOR2, "vector2", varray());
	ADDFUNC1(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_VECTOR2_ARRAY, NIL, PackedVector2Array, invert, varray());

	ADDFUNC0R(PACKED_VECTOR3_ARRAY, INT, PackedVector3Array, size, varray());
	ADDFUNC0R(PACKED_VECTOR3_ARRAY, BOOL, PackedVector3Array, empty, varray());
	ADDFUNC2(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, set, INT, "idx", VECTOR3, "vector3", varray());
	ADDFUNC1(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, push_back, VECTOR3, "vector3", varray());
	ADDFUNC1(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, append, VECTOR3, "vector3", varray());
	ADDFUNC1(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, append_array, PACKED_VECTOR3_ARRAY, "array", varray());
	ADDFUNC1(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_VECTOR3_ARRAY, INT, PackedVector3Array, insert, INT, "idx", VECTOR3, "vector3", varray());
	ADDFUNC1(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_VECTOR3_ARRAY, NIL, PackedVector3Array, invert, varray());

	ADDFUNC0R(PACKED_COLOR_ARRAY, INT, PackedColorArray, size, varray());
	ADDFUNC0R(PACKED_COLOR_ARRAY, BOOL, PackedColorArray, empty, varray());
	ADDFUNC2(PACKED_COLOR_ARRAY, NIL, PackedColorArray, set, INT, "idx", COLOR, "color", varray());
	ADDFUNC1(PACKED_COLOR_ARRAY, NIL, PackedColorArray, push_back, COLOR, "color", varray());
	ADDFUNC1(PACKED_COLOR_ARRAY, NIL, PackedColorArray, append, COLOR, "color", varray());
	ADDFUNC1(PACKED_COLOR_ARRAY, NIL, PackedColorArray, append_array, PACKED_COLOR_ARRAY, "array", varray());
	ADDFUNC1(PACKED_COLOR_ARRAY, NIL, PackedColorArray, remove, INT, "idx", varray());
	ADDFUNC2R(PACKED_COLOR_ARRAY, INT, PackedColorArray, insert, INT, "idx", COLOR, "color", varray());
	ADDFUNC1(PACKED_COLOR_ARRAY, NIL, PackedColorArray, resize, INT, "idx", varray());
	ADDFUNC0(PACKED_COLOR_ARRAY, NIL, PackedColorArray, invert, varray());

	//pointerbased

	ADDFUNC0R(AABB, FLOAT, AABB, get_area, varray());
	ADDFUNC0R(AABB, BOOL, AABB, has_no_area, varray());
	ADDFUNC0R(AABB, BOOL, AABB, has_no_surface, varray());
	ADDFUNC1R(AABB, BOOL, AABB, has_point, VECTOR3, "point", varray());
	ADDFUNC1R(AABB, BOOL, AABB, is_equal_approx, AABB, "aabb", varray());
	ADDFUNC1R(AABB, BOOL, AABB, intersects, AABB, "with", varray());
	ADDFUNC1R(AABB, BOOL, AABB, encloses, AABB, "with", varray());
	ADDFUNC1R(AABB, BOOL, AABB, intersects_plane, PLANE, "plane", varray());
	ADDFUNC2R(AABB, BOOL, AABB, intersects_segment, VECTOR3, "from", VECTOR3, "to", varray());
	ADDFUNC1R(AABB, AABB, AABB, intersection, AABB, "with", varray());
	ADDFUNC1R(AABB, AABB, AABB, merge, AABB, "with", varray());
	ADDFUNC1R(AABB, AABB, AABB, expand, VECTOR3, "to_point", varray());
	ADDFUNC1R(AABB, AABB, AABB, grow, FLOAT, "by", varray());
	ADDFUNC1R(AABB, VECTOR3, AABB, get_support, VECTOR3, "dir", varray());
	ADDFUNC0R(AABB, VECTOR3, AABB, get_longest_axis, varray());
	ADDFUNC0R(AABB, INT, AABB, get_longest_axis_index, varray());
	ADDFUNC0R(AABB, FLOAT, AABB, get_longest_axis_size, varray());
	ADDFUNC0R(AABB, VECTOR3, AABB, get_shortest_axis, varray());
	ADDFUNC0R(AABB, INT, AABB, get_shortest_axis_index, varray());
	ADDFUNC0R(AABB, FLOAT, AABB, get_shortest_axis_size, varray());
	ADDFUNC1R(AABB, VECTOR3, AABB, get_endpoint, INT, "idx", varray());

	ADDFUNC0R(TRANSFORM2D, TRANSFORM2D, Transform2D, inverse, varray());
	ADDFUNC0R(TRANSFORM2D, TRANSFORM2D, Transform2D, affine_inverse, varray());
	ADDFUNC0R(TRANSFORM2D, FLOAT, Transform2D, get_rotation, varray());
	ADDFUNC0R(TRANSFORM2D, VECTOR2, Transform2D, get_origin, varray());
	ADDFUNC0R(TRANSFORM2D, VECTOR2, Transform2D, get_scale, varray());
	ADDFUNC0R(TRANSFORM2D, TRANSFORM2D, Transform2D, orthonormalized, varray());
	ADDFUNC1R(TRANSFORM2D, TRANSFORM2D, Transform2D, rotated, FLOAT, "phi", varray());
	ADDFUNC1R(TRANSFORM2D, TRANSFORM2D, Transform2D, scaled, VECTOR2, "scale", varray());
	ADDFUNC1R(TRANSFORM2D, TRANSFORM2D, Transform2D, translated, VECTOR2, "offset", varray());
	ADDFUNC1R(TRANSFORM2D, NIL, Transform2D, xform, NIL, "v", varray());
	ADDFUNC1R(TRANSFORM2D, NIL, Transform2D, xform_inv, NIL, "v", varray());
	ADDFUNC1R(TRANSFORM2D, VECTOR2, Transform2D, basis_xform, VECTOR2, "v", varray());
	ADDFUNC1R(TRANSFORM2D, VECTOR2, Transform2D, basis_xform_inv, VECTOR2, "v", varray());
	ADDFUNC2R(TRANSFORM2D, TRANSFORM2D, Transform2D, interpolate_with, TRANSFORM2D, "transform", FLOAT, "weight", varray());
	ADDFUNC1R(TRANSFORM2D, BOOL, Transform2D, is_equal_approx, TRANSFORM2D, "transform", varray());

	ADDFUNC0R(BASIS, BASIS, Basis, inverse, varray());
	ADDFUNC0R(BASIS, BASIS, Basis, transposed, varray());
	ADDFUNC0R(BASIS, BASIS, Basis, orthonormalized, varray());
	ADDFUNC0R(BASIS, FLOAT, Basis, determinant, varray());
	ADDFUNC2R(BASIS, BASIS, Basis, rotated, VECTOR3, "axis", FLOAT, "phi", varray());
	ADDFUNC1R(BASIS, BASIS, Basis, scaled, VECTOR3, "scale", varray());
	ADDFUNC0R(BASIS, VECTOR3, Basis, get_scale, varray());
	ADDFUNC0R(BASIS, VECTOR3, Basis, get_euler, varray());
	ADDFUNC1R(BASIS, FLOAT, Basis, tdotx, VECTOR3, "with", varray());
	ADDFUNC1R(BASIS, FLOAT, Basis, tdoty, VECTOR3, "with", varray());
	ADDFUNC1R(BASIS, FLOAT, Basis, tdotz, VECTOR3, "with", varray());
	ADDFUNC1R(BASIS, VECTOR3, Basis, xform, VECTOR3, "v", varray());
	ADDFUNC1R(BASIS, VECTOR3, Basis, xform_inv, VECTOR3, "v", varray());
	ADDFUNC0R(BASIS, INT, Basis, get_orthogonal_index, varray());
	ADDFUNC2R(BASIS, BASIS, Basis, slerp, BASIS, "b", FLOAT, "t", varray());
	ADDFUNC2R(BASIS, BOOL, Basis, is_equal_approx, BASIS, "b", FLOAT, "epsilon", varray(CMP_EPSILON)); // TODO: Replace in 4.0, see other TODO.
	ADDFUNC0R(BASIS, QUAT, Basis, get_rotation_quat, varray());

	ADDFUNC0R(TRANSFORM, TRANSFORM, Transform, inverse, varray());
	ADDFUNC0R(TRANSFORM, TRANSFORM, Transform, affine_inverse, varray());
	ADDFUNC0R(TRANSFORM, TRANSFORM, Transform, orthonormalized, varray());
	ADDFUNC2R(TRANSFORM, TRANSFORM, Transform, rotated, VECTOR3, "axis", FLOAT, "phi", varray());
	ADDFUNC1R(TRANSFORM, TRANSFORM, Transform, scaled, VECTOR3, "scale", varray());
	ADDFUNC1R(TRANSFORM, TRANSFORM, Transform, translated, VECTOR3, "offset", varray());
	ADDFUNC2R(TRANSFORM, TRANSFORM, Transform, looking_at, VECTOR3, "target", VECTOR3, "up", varray());
	ADDFUNC2R(TRANSFORM, TRANSFORM, Transform, interpolate_with, TRANSFORM, "transform", FLOAT, "weight", varray());
	ADDFUNC1R(TRANSFORM, BOOL, Transform, is_equal_approx, TRANSFORM, "transform", varray());
	ADDFUNC1R(TRANSFORM, NIL, Transform, xform, NIL, "v", varray());
	ADDFUNC1R(TRANSFORM, NIL, Transform, xform_inv, NIL, "v", varray());

	/* REGISTER CONSTRUCTORS */

	_VariantCall::add_constructor(_VariantCall::Vector2_init1, Variant::VECTOR2, "x", Variant::FLOAT, "y", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Vector2i_init1, Variant::VECTOR2I, "x", Variant::INT, "y", Variant::INT);

	_VariantCall::add_constructor(_VariantCall::Rect2_init1, Variant::RECT2, "position", Variant::VECTOR2, "size", Variant::VECTOR2);
	_VariantCall::add_constructor(_VariantCall::Rect2_init2, Variant::RECT2, "x", Variant::FLOAT, "y", Variant::FLOAT, "width", Variant::FLOAT, "height", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::Rect2i_init1, Variant::RECT2I, "position", Variant::VECTOR2, "size", Variant::VECTOR2);
	_VariantCall::add_constructor(_VariantCall::Rect2i_init2, Variant::RECT2I, "x", Variant::INT, "y", Variant::INT, "width", Variant::INT, "height", Variant::INT);

	_VariantCall::add_constructor(_VariantCall::Transform2D_init2, Variant::TRANSFORM2D, "rotation", Variant::FLOAT, "position", Variant::VECTOR2);
	_VariantCall::add_constructor(_VariantCall::Transform2D_init3, Variant::TRANSFORM2D, "x_axis", Variant::VECTOR2, "y_axis", Variant::VECTOR2, "origin", Variant::VECTOR2);

	_VariantCall::add_constructor(_VariantCall::Vector3_init1, Variant::VECTOR3, "x", Variant::FLOAT, "y", Variant::FLOAT, "z", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Vector3i_init1, Variant::VECTOR3I, "x", Variant::INT, "y", Variant::INT, "z", Variant::INT);

	_VariantCall::add_constructor(_VariantCall::Plane_init1, Variant::PLANE, "a", Variant::FLOAT, "b", Variant::FLOAT, "c", Variant::FLOAT, "d", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Plane_init2, Variant::PLANE, "v1", Variant::VECTOR3, "v2", Variant::VECTOR3, "v3", Variant::VECTOR3);
	_VariantCall::add_constructor(_VariantCall::Plane_init3, Variant::PLANE, "normal", Variant::VECTOR3, "d", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::Quat_init1, Variant::QUAT, "x", Variant::FLOAT, "y", Variant::FLOAT, "z", Variant::FLOAT, "w", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Quat_init2, Variant::QUAT, "axis", Variant::VECTOR3, "angle", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Quat_init3, Variant::QUAT, "euler", Variant::VECTOR3);

	_VariantCall::add_constructor(_VariantCall::Color_init1, Variant::COLOR, "r", Variant::FLOAT, "g", Variant::FLOAT, "b", Variant::FLOAT, "a", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Color_init2, Variant::COLOR, "r", Variant::FLOAT, "g", Variant::FLOAT, "b", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::AABB_init1, Variant::AABB, "position", Variant::VECTOR3, "size", Variant::VECTOR3);

	_VariantCall::add_constructor(_VariantCall::Basis_init1, Variant::BASIS, "x_axis", Variant::VECTOR3, "y_axis", Variant::VECTOR3, "z_axis", Variant::VECTOR3);
	_VariantCall::add_constructor(_VariantCall::Basis_init2, Variant::BASIS, "axis", Variant::VECTOR3, "phi", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::Transform_init1, Variant::TRANSFORM, "x_axis", Variant::VECTOR3, "y_axis", Variant::VECTOR3, "z_axis", Variant::VECTOR3, "origin", Variant::VECTOR3);
	_VariantCall::add_constructor(_VariantCall::Transform_init2, Variant::TRANSFORM, "basis", Variant::BASIS, "origin", Variant::VECTOR3);

	_VariantCall::add_constructor(_VariantCall::Callable_init2, Variant::CALLABLE, "object", Variant::OBJECT, "method_name", Variant::STRING_NAME);
	_VariantCall::add_constructor(_VariantCall::Signal_init2, Variant::SIGNAL, "object", Variant::OBJECT, "signal_name", Variant::STRING_NAME);

	/* REGISTER CONSTANTS */

	_populate_named_colors();
	for (Map<String, Color>::Element *color = _named_colors.front(); color; color = color->next()) {
		_VariantCall::add_variant_constant(Variant::COLOR, color->key(), color->value());
	}

	_VariantCall::add_constant(Variant::VECTOR3, "AXIS_X", Vector3::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR3, "AXIS_Y", Vector3::AXIS_Y);
	_VariantCall::add_constant(Variant::VECTOR3, "AXIS_Z", Vector3::AXIS_Z);

	_VariantCall::add_variant_constant(Variant::VECTOR3, "ZERO", Vector3(0, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "ONE", Vector3(1, 1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "INF", Vector3(Math_INF, Math_INF, Math_INF));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "LEFT", Vector3(-1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "RIGHT", Vector3(1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "UP", Vector3(0, 1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "DOWN", Vector3(0, -1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "FORWARD", Vector3(0, 0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "BACK", Vector3(0, 0, 1));

	_VariantCall::add_constant(Variant::VECTOR3I, "AXIS_X", Vector3::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR3I, "AXIS_Y", Vector3::AXIS_Y);
	_VariantCall::add_constant(Variant::VECTOR3I, "AXIS_Z", Vector3::AXIS_Z);

	_VariantCall::add_variant_constant(Variant::VECTOR3I, "ZERO", Vector3i(0, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "ONE", Vector3i(1, 1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "LEFT", Vector3i(-1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "RIGHT", Vector3i(1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "UP", Vector3i(0, 1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "DOWN", Vector3i(0, -1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "FORWARD", Vector3i(0, 0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "BACK", Vector3i(0, 0, 1));

	_VariantCall::add_constant(Variant::VECTOR2, "AXIS_X", Vector2::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR2, "AXIS_Y", Vector2::AXIS_Y);

	_VariantCall::add_constant(Variant::VECTOR2I, "AXIS_X", Vector2::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR2I, "AXIS_Y", Vector2::AXIS_Y);

	_VariantCall::add_variant_constant(Variant::VECTOR2, "ZERO", Vector2(0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "ONE", Vector2(1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "INF", Vector2(Math_INF, Math_INF));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "LEFT", Vector2(-1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "RIGHT", Vector2(1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "UP", Vector2(0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "DOWN", Vector2(0, 1));

	_VariantCall::add_variant_constant(Variant::VECTOR2I, "ZERO", Vector2i(0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "ONE", Vector2i(1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "LEFT", Vector2i(-1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "RIGHT", Vector2i(1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "UP", Vector2i(0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "DOWN", Vector2i(0, 1));

	_VariantCall::add_variant_constant(Variant::TRANSFORM2D, "IDENTITY", Transform2D());
	_VariantCall::add_variant_constant(Variant::TRANSFORM2D, "FLIP_X", Transform2D(-1, 0, 0, 1, 0, 0));
	_VariantCall::add_variant_constant(Variant::TRANSFORM2D, "FLIP_Y", Transform2D(1, 0, 0, -1, 0, 0));

	Transform identity_transform = Transform();
	Transform flip_x_transform = Transform(-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
	Transform flip_y_transform = Transform(1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0);
	Transform flip_z_transform = Transform(1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "IDENTITY", identity_transform);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "FLIP_X", flip_x_transform);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "FLIP_Y", flip_y_transform);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "FLIP_Z", flip_z_transform);

	Basis identity_basis = Basis();
	Basis flip_x_basis = Basis(-1, 0, 0, 0, 1, 0, 0, 0, 1);
	Basis flip_y_basis = Basis(1, 0, 0, 0, -1, 0, 0, 0, 1);
	Basis flip_z_basis = Basis(1, 0, 0, 0, 1, 0, 0, 0, -1);
	_VariantCall::add_variant_constant(Variant::BASIS, "IDENTITY", identity_basis);
	_VariantCall::add_variant_constant(Variant::BASIS, "FLIP_X", flip_x_basis);
	_VariantCall::add_variant_constant(Variant::BASIS, "FLIP_Y", flip_y_basis);
	_VariantCall::add_variant_constant(Variant::BASIS, "FLIP_Z", flip_z_basis);

	_VariantCall::add_variant_constant(Variant::PLANE, "PLANE_YZ", Plane(Vector3(1, 0, 0), 0));
	_VariantCall::add_variant_constant(Variant::PLANE, "PLANE_XZ", Plane(Vector3(0, 1, 0), 0));
	_VariantCall::add_variant_constant(Variant::PLANE, "PLANE_XY", Plane(Vector3(0, 0, 1), 0));

	_VariantCall::add_variant_constant(Variant::QUAT, "IDENTITY", Quat(0, 0, 0, 1));
}

void unregister_variant_methods() {

	memdelete_arr(_VariantCall::type_funcs);
	memdelete_arr(_VariantCall::construct_funcs);
	memdelete_arr(_VariantCall::constant_data);
}
