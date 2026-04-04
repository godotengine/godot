/**************************************************************************/
/*  string.cpp                                                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_vector4_array.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/signal.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>
#include <godot_cpp/variant/vector4.hpp>
#include <godot_cpp/variant/vector4i.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

String::_MethodBindings String::_method_bindings;

void String::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_STRING, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_STRING, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_STRING, 2);
	_method_bindings.constructor_3 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_STRING, 3);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_STRING);
}
void String::init_bindings() {
	String::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("casecmp_to");
	_method_bindings.method_casecmp_to = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("nocasecmp_to");
	_method_bindings.method_nocasecmp_to = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("naturalcasecmp_to");
	_method_bindings.method_naturalcasecmp_to = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("naturalnocasecmp_to");
	_method_bindings.method_naturalnocasecmp_to = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("filecasecmp_to");
	_method_bindings.method_filecasecmp_to = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("filenocasecmp_to");
	_method_bindings.method_filenocasecmp_to = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("length");
	_method_bindings.method_length = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("substr");
	_method_bindings.method_substr = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 787537301);
	_gde_name = StringName("get_slice");
	_method_bindings.method_get_slice = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3535100402);
	_gde_name = StringName("get_slicec");
	_method_bindings.method_get_slicec = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 787537301);
	_gde_name = StringName("get_slice_count");
	_method_bindings.method_get_slice_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2920860731);
	_gde_name = StringName("find");
	_method_bindings.method_find = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1760645412);
	_gde_name = StringName("findn");
	_method_bindings.method_findn = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1760645412);
	_gde_name = StringName("count");
	_method_bindings.method_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2343087891);
	_gde_name = StringName("countn");
	_method_bindings.method_countn = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2343087891);
	_gde_name = StringName("rfind");
	_method_bindings.method_rfind = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1760645412);
	_gde_name = StringName("rfindn");
	_method_bindings.method_rfindn = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1760645412);
	_gde_name = StringName("match");
	_method_bindings.method_match = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("matchn");
	_method_bindings.method_matchn = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("begins_with");
	_method_bindings.method_begins_with = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("ends_with");
	_method_bindings.method_ends_with = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("is_subsequence_of");
	_method_bindings.method_is_subsequence_of = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("is_subsequence_ofn");
	_method_bindings.method_is_subsequence_ofn = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("bigrams");
	_method_bindings.method_bigrams = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 747180633);
	_gde_name = StringName("similarity");
	_method_bindings.method_similarity = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2697460964);
	_gde_name = StringName("format");
	_method_bindings.method_format = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3212199029);
	_gde_name = StringName("replace");
	_method_bindings.method_replace = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1340436205);
	_gde_name = StringName("replacen");
	_method_bindings.method_replacen = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1340436205);
	_gde_name = StringName("replace_char");
	_method_bindings.method_replace_char = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 787537301);
	_gde_name = StringName("replace_chars");
	_method_bindings.method_replace_chars = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3535100402);
	_gde_name = StringName("remove_char");
	_method_bindings.method_remove_char = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2162347432);
	_gde_name = StringName("remove_chars");
	_method_bindings.method_remove_chars = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("repeat");
	_method_bindings.method_repeat = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2162347432);
	_gde_name = StringName("reverse");
	_method_bindings.method_reverse = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("insert");
	_method_bindings.method_insert = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 248737229);
	_gde_name = StringName("erase");
	_method_bindings.method_erase = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 787537301);
	_gde_name = StringName("capitalize");
	_method_bindings.method_capitalize = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("to_camel_case");
	_method_bindings.method_to_camel_case = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("to_pascal_case");
	_method_bindings.method_to_pascal_case = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("to_snake_case");
	_method_bindings.method_to_snake_case = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("to_kebab_case");
	_method_bindings.method_to_kebab_case = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("split");
	_method_bindings.method_split = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1252735785);
	_gde_name = StringName("rsplit");
	_method_bindings.method_rsplit = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1252735785);
	_gde_name = StringName("split_floats");
	_method_bindings.method_split_floats = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2092079095);
	_gde_name = StringName("join");
	_method_bindings.method_join = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3595973238);
	_gde_name = StringName("to_upper");
	_method_bindings.method_to_upper = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("to_lower");
	_method_bindings.method_to_lower = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("left");
	_method_bindings.method_left = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2162347432);
	_gde_name = StringName("right");
	_method_bindings.method_right = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2162347432);
	_gde_name = StringName("strip_edges");
	_method_bindings.method_strip_edges = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 907855311);
	_gde_name = StringName("strip_escapes");
	_method_bindings.method_strip_escapes = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("lstrip");
	_method_bindings.method_lstrip = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("rstrip");
	_method_bindings.method_rstrip = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("get_extension");
	_method_bindings.method_get_extension = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_basename");
	_method_bindings.method_get_basename = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("path_join");
	_method_bindings.method_path_join = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("unicode_at");
	_method_bindings.method_unicode_at = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("indent");
	_method_bindings.method_indent = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("dedent");
	_method_bindings.method_dedent = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("hash");
	_method_bindings.method_hash = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("md5_text");
	_method_bindings.method_md5_text = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("sha1_text");
	_method_bindings.method_sha1_text = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("sha256_text");
	_method_bindings.method_sha256_text = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("md5_buffer");
	_method_bindings.method_md5_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("sha1_buffer");
	_method_bindings.method_sha1_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("sha256_buffer");
	_method_bindings.method_sha256_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("is_empty");
	_method_bindings.method_is_empty = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("contains");
	_method_bindings.method_contains = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("containsn");
	_method_bindings.method_containsn = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2566493496);
	_gde_name = StringName("is_absolute_path");
	_method_bindings.method_is_absolute_path = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_relative_path");
	_method_bindings.method_is_relative_path = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("simplify_path");
	_method_bindings.method_simplify_path = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_base_dir");
	_method_bindings.method_get_base_dir = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_file");
	_method_bindings.method_get_file = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("xml_escape");
	_method_bindings.method_xml_escape = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3429816538);
	_gde_name = StringName("xml_unescape");
	_method_bindings.method_xml_unescape = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("uri_encode");
	_method_bindings.method_uri_encode = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("uri_decode");
	_method_bindings.method_uri_decode = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("uri_file_decode");
	_method_bindings.method_uri_file_decode = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("c_escape");
	_method_bindings.method_c_escape = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("c_unescape");
	_method_bindings.method_c_unescape = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("json_escape");
	_method_bindings.method_json_escape = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("validate_node_name");
	_method_bindings.method_validate_node_name = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("validate_filename");
	_method_bindings.method_validate_filename = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("is_valid_ascii_identifier");
	_method_bindings.method_is_valid_ascii_identifier = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_unicode_identifier");
	_method_bindings.method_is_valid_unicode_identifier = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_identifier");
	_method_bindings.method_is_valid_identifier = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_int");
	_method_bindings.method_is_valid_int = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_float");
	_method_bindings.method_is_valid_float = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_hex_number");
	_method_bindings.method_is_valid_hex_number = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 593672999);
	_gde_name = StringName("is_valid_html_color");
	_method_bindings.method_is_valid_html_color = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_ip_address");
	_method_bindings.method_is_valid_ip_address = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid_filename");
	_method_bindings.method_is_valid_filename = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("to_int");
	_method_bindings.method_to_int = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("to_float");
	_method_bindings.method_to_float = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 466405837);
	_gde_name = StringName("hex_to_int");
	_method_bindings.method_hex_to_int = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("bin_to_int");
	_method_bindings.method_bin_to_int = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("lpad");
	_method_bindings.method_lpad = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 248737229);
	_gde_name = StringName("rpad");
	_method_bindings.method_rpad = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 248737229);
	_gde_name = StringName("pad_decimals");
	_method_bindings.method_pad_decimals = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2162347432);
	_gde_name = StringName("pad_zeros");
	_method_bindings.method_pad_zeros = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2162347432);
	_gde_name = StringName("trim_prefix");
	_method_bindings.method_trim_prefix = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("trim_suffix");
	_method_bindings.method_trim_suffix = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("to_ascii_buffer");
	_method_bindings.method_to_ascii_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("to_utf8_buffer");
	_method_bindings.method_to_utf8_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("to_utf16_buffer");
	_method_bindings.method_to_utf16_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("to_utf32_buffer");
	_method_bindings.method_to_utf32_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("to_wchar_buffer");
	_method_bindings.method_to_wchar_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("to_multibyte_char_buffer");
	_method_bindings.method_to_multibyte_char_buffer = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 3055765187);
	_gde_name = StringName("hex_decode");
	_method_bindings.method_hex_decode = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("num_scientific");
	_method_bindings.method_num_scientific = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2710373411);
	_gde_name = StringName("num");
	_method_bindings.method_num = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 1555901022);
	_gde_name = StringName("num_int64");
	_method_bindings.method_num_int64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2111271071);
	_gde_name = StringName("num_uint64");
	_method_bindings.method_num_uint64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 2111271071);
	_gde_name = StringName("chr");
	_method_bindings.method_chr = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 897497541);
	_gde_name = StringName("humanize_size");
	_method_bindings.method_humanize_size = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_STRING, _gde_name._native_ptr(), 897497541);
	_method_bindings.indexed_setter = ::godot::gdextension_interface::variant_get_ptr_indexed_setter(GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.indexed_getter = ::godot::gdextension_interface::variant_get_ptr_indexed_getter(GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_module_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_module_bool = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_BOOL);
	_method_bindings.operator_module_int = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_INT);
	_method_bindings.operator_module_float = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_FLOAT);
	_method_bindings.operator_equal_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_not_equal_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_less_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_LESS, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_less_equal_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_LESS_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_greater_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_GREATER, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_greater_equal_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_GREATER_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_add_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_ADD, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_module_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_in_String = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING);
	_method_bindings.operator_module_Vector2 = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_VECTOR2);
	_method_bindings.operator_module_Vector2i = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_VECTOR2I);
	_method_bindings.operator_module_Rect2 = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_RECT2);
	_method_bindings.operator_module_Rect2i = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_RECT2I);
	_method_bindings.operator_module_Vector3 = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_VECTOR3);
	_method_bindings.operator_module_Vector3i = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_VECTOR3I);
	_method_bindings.operator_module_Transform2D = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_TRANSFORM2D);
	_method_bindings.operator_module_Vector4 = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_VECTOR4);
	_method_bindings.operator_module_Vector4i = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_VECTOR4I);
	_method_bindings.operator_module_Plane = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PLANE);
	_method_bindings.operator_module_Quaternion = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_QUATERNION);
	_method_bindings.operator_module_AABB = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_AABB);
	_method_bindings.operator_module_Basis = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_BASIS);
	_method_bindings.operator_module_Transform3D = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_TRANSFORM3D);
	_method_bindings.operator_module_Projection = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PROJECTION);
	_method_bindings.operator_module_Color = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_COLOR);
	_method_bindings.operator_equal_StringName = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING_NAME);
	_method_bindings.operator_not_equal_StringName = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING_NAME);
	_method_bindings.operator_add_StringName = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_ADD, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING_NAME);
	_method_bindings.operator_module_StringName = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING_NAME);
	_method_bindings.operator_in_StringName = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_STRING_NAME);
	_method_bindings.operator_module_NodePath = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_NODE_PATH);
	_method_bindings.operator_module_RID = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_RID);
	_method_bindings.operator_module_Object = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_OBJECT);
	_method_bindings.operator_in_Object = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_OBJECT);
	_method_bindings.operator_module_Callable = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_CALLABLE);
	_method_bindings.operator_module_Signal = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_SIGNAL);
	_method_bindings.operator_module_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_module_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_module_PackedByteArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
	_method_bindings.operator_module_PackedInt32Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_INT32_ARRAY);
	_method_bindings.operator_module_PackedInt64Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_INT64_ARRAY);
	_method_bindings.operator_module_PackedFloat32Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
	_method_bindings.operator_module_PackedFloat64Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT64_ARRAY);
	_method_bindings.operator_module_PackedStringArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_STRING_ARRAY);
	_method_bindings.operator_in_PackedStringArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_STRING_ARRAY);
	_method_bindings.operator_module_PackedVector2Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR2_ARRAY);
	_method_bindings.operator_module_PackedVector3Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR3_ARRAY);
	_method_bindings.operator_module_PackedColorArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_COLOR_ARRAY);
	_method_bindings.operator_module_PackedVector4Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_MODULE, GDEXTENSION_VARIANT_TYPE_STRING, GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR4_ARRAY);
}

String::String(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

String::String() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

String::String(const String &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

String::String(const StringName &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, &p_from);
}

String::String(const NodePath &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_3, &opaque, &p_from);
}

String::String(String &&p_other) {
	std::swap(opaque, p_other.opaque);
}

String::~String() {
	_method_bindings.destructor(&opaque);
}

int64_t String::casecmp_to(const String &p_to) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_casecmp_to, (GDExtensionTypePtr)&opaque, &p_to);
}

int64_t String::nocasecmp_to(const String &p_to) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_nocasecmp_to, (GDExtensionTypePtr)&opaque, &p_to);
}

int64_t String::naturalcasecmp_to(const String &p_to) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_naturalcasecmp_to, (GDExtensionTypePtr)&opaque, &p_to);
}

int64_t String::naturalnocasecmp_to(const String &p_to) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_naturalnocasecmp_to, (GDExtensionTypePtr)&opaque, &p_to);
}

int64_t String::filecasecmp_to(const String &p_to) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_filecasecmp_to, (GDExtensionTypePtr)&opaque, &p_to);
}

int64_t String::filenocasecmp_to(const String &p_to) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_filenocasecmp_to, (GDExtensionTypePtr)&opaque, &p_to);
}

int64_t String::length() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_length, (GDExtensionTypePtr)&opaque);
}

String String::substr(int64_t p_from, int64_t p_len) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	int64_t p_len_encoded;
	PtrToArg<int64_t>::encode(p_len, &p_len_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_substr, (GDExtensionTypePtr)&opaque, &p_from_encoded, &p_len_encoded);
}

String String::get_slice(const String &p_delimiter, int64_t p_slice) const {
	int64_t p_slice_encoded;
	PtrToArg<int64_t>::encode(p_slice, &p_slice_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_slice, (GDExtensionTypePtr)&opaque, &p_delimiter, &p_slice_encoded);
}

String String::get_slicec(int64_t p_delimiter, int64_t p_slice) const {
	int64_t p_delimiter_encoded;
	PtrToArg<int64_t>::encode(p_delimiter, &p_delimiter_encoded);
	int64_t p_slice_encoded;
	PtrToArg<int64_t>::encode(p_slice, &p_slice_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_slicec, (GDExtensionTypePtr)&opaque, &p_delimiter_encoded, &p_slice_encoded);
}

int64_t String::get_slice_count(const String &p_delimiter) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_slice_count, (GDExtensionTypePtr)&opaque, &p_delimiter);
}

int64_t String::find(const String &p_what, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_find, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded);
}

int64_t String::findn(const String &p_what, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_findn, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded);
}

int64_t String::count(const String &p_what, int64_t p_from, int64_t p_to) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	int64_t p_to_encoded;
	PtrToArg<int64_t>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_count, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded, &p_to_encoded);
}

int64_t String::countn(const String &p_what, int64_t p_from, int64_t p_to) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	int64_t p_to_encoded;
	PtrToArg<int64_t>::encode(p_to, &p_to_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_countn, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded, &p_to_encoded);
}

int64_t String::rfind(const String &p_what, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_rfind, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded);
}

int64_t String::rfindn(const String &p_what, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_rfindn, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded);
}

bool String::match(const String &p_expr) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_match, (GDExtensionTypePtr)&opaque, &p_expr);
}

bool String::matchn(const String &p_expr) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_matchn, (GDExtensionTypePtr)&opaque, &p_expr);
}

bool String::begins_with(const String &p_text) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_begins_with, (GDExtensionTypePtr)&opaque, &p_text);
}

bool String::ends_with(const String &p_text) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_ends_with, (GDExtensionTypePtr)&opaque, &p_text);
}

bool String::is_subsequence_of(const String &p_text) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_subsequence_of, (GDExtensionTypePtr)&opaque, &p_text);
}

bool String::is_subsequence_ofn(const String &p_text) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_subsequence_ofn, (GDExtensionTypePtr)&opaque, &p_text);
}

PackedStringArray String::bigrams() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedStringArray>(_method_bindings.method_bigrams, (GDExtensionTypePtr)&opaque);
}

double String::similarity(const String &p_text) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<double>(_method_bindings.method_similarity, (GDExtensionTypePtr)&opaque, &p_text);
}

String String::format(const Variant &p_values, const String &p_placeholder) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_format, (GDExtensionTypePtr)&opaque, &p_values, &p_placeholder);
}

String String::replace(const String &p_what, const String &p_forwhat) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_replace, (GDExtensionTypePtr)&opaque, &p_what, &p_forwhat);
}

String String::replacen(const String &p_what, const String &p_forwhat) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_replacen, (GDExtensionTypePtr)&opaque, &p_what, &p_forwhat);
}

String String::replace_char(int64_t p_key, int64_t p_with) const {
	int64_t p_key_encoded;
	PtrToArg<int64_t>::encode(p_key, &p_key_encoded);
	int64_t p_with_encoded;
	PtrToArg<int64_t>::encode(p_with, &p_with_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_replace_char, (GDExtensionTypePtr)&opaque, &p_key_encoded, &p_with_encoded);
}

String String::replace_chars(const String &p_keys, int64_t p_with) const {
	int64_t p_with_encoded;
	PtrToArg<int64_t>::encode(p_with, &p_with_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_replace_chars, (GDExtensionTypePtr)&opaque, &p_keys, &p_with_encoded);
}

String String::remove_char(int64_t p_what) const {
	int64_t p_what_encoded;
	PtrToArg<int64_t>::encode(p_what, &p_what_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_remove_char, (GDExtensionTypePtr)&opaque, &p_what_encoded);
}

String String::remove_chars(const String &p_chars) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_remove_chars, (GDExtensionTypePtr)&opaque, &p_chars);
}

String String::repeat(int64_t p_count) const {
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_repeat, (GDExtensionTypePtr)&opaque, &p_count_encoded);
}

String String::reverse() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_reverse, (GDExtensionTypePtr)&opaque);
}

String String::insert(int64_t p_position, const String &p_what) const {
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_insert, (GDExtensionTypePtr)&opaque, &p_position_encoded, &p_what);
}

String String::erase(int64_t p_position, int64_t p_chars) const {
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	int64_t p_chars_encoded;
	PtrToArg<int64_t>::encode(p_chars, &p_chars_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_erase, (GDExtensionTypePtr)&opaque, &p_position_encoded, &p_chars_encoded);
}

String String::capitalize() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_capitalize, (GDExtensionTypePtr)&opaque);
}

String String::to_camel_case() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_to_camel_case, (GDExtensionTypePtr)&opaque);
}

String String::to_pascal_case() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_to_pascal_case, (GDExtensionTypePtr)&opaque);
}

String String::to_snake_case() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_to_snake_case, (GDExtensionTypePtr)&opaque);
}

String String::to_kebab_case() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_to_kebab_case, (GDExtensionTypePtr)&opaque);
}

PackedStringArray String::split(const String &p_delimiter, bool p_allow_empty, int64_t p_maxsplit) const {
	int8_t p_allow_empty_encoded;
	PtrToArg<bool>::encode(p_allow_empty, &p_allow_empty_encoded);
	int64_t p_maxsplit_encoded;
	PtrToArg<int64_t>::encode(p_maxsplit, &p_maxsplit_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedStringArray>(_method_bindings.method_split, (GDExtensionTypePtr)&opaque, &p_delimiter, &p_allow_empty_encoded, &p_maxsplit_encoded);
}

PackedStringArray String::rsplit(const String &p_delimiter, bool p_allow_empty, int64_t p_maxsplit) const {
	int8_t p_allow_empty_encoded;
	PtrToArg<bool>::encode(p_allow_empty, &p_allow_empty_encoded);
	int64_t p_maxsplit_encoded;
	PtrToArg<int64_t>::encode(p_maxsplit, &p_maxsplit_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedStringArray>(_method_bindings.method_rsplit, (GDExtensionTypePtr)&opaque, &p_delimiter, &p_allow_empty_encoded, &p_maxsplit_encoded);
}

PackedFloat64Array String::split_floats(const String &p_delimiter, bool p_allow_empty) const {
	int8_t p_allow_empty_encoded;
	PtrToArg<bool>::encode(p_allow_empty, &p_allow_empty_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedFloat64Array>(_method_bindings.method_split_floats, (GDExtensionTypePtr)&opaque, &p_delimiter, &p_allow_empty_encoded);
}

String String::join(const PackedStringArray &p_parts) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_join, (GDExtensionTypePtr)&opaque, &p_parts);
}

String String::to_upper() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_to_upper, (GDExtensionTypePtr)&opaque);
}

String String::to_lower() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_to_lower, (GDExtensionTypePtr)&opaque);
}

String String::left(int64_t p_length) const {
	int64_t p_length_encoded;
	PtrToArg<int64_t>::encode(p_length, &p_length_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_left, (GDExtensionTypePtr)&opaque, &p_length_encoded);
}

String String::right(int64_t p_length) const {
	int64_t p_length_encoded;
	PtrToArg<int64_t>::encode(p_length, &p_length_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_right, (GDExtensionTypePtr)&opaque, &p_length_encoded);
}

String String::strip_edges(bool p_left, bool p_right) const {
	int8_t p_left_encoded;
	PtrToArg<bool>::encode(p_left, &p_left_encoded);
	int8_t p_right_encoded;
	PtrToArg<bool>::encode(p_right, &p_right_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_strip_edges, (GDExtensionTypePtr)&opaque, &p_left_encoded, &p_right_encoded);
}

String String::strip_escapes() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_strip_escapes, (GDExtensionTypePtr)&opaque);
}

String String::lstrip(const String &p_chars) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_lstrip, (GDExtensionTypePtr)&opaque, &p_chars);
}

String String::rstrip(const String &p_chars) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_rstrip, (GDExtensionTypePtr)&opaque, &p_chars);
}

String String::get_extension() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_extension, (GDExtensionTypePtr)&opaque);
}

String String::get_basename() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_basename, (GDExtensionTypePtr)&opaque);
}

String String::path_join(const String &p_path) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_path_join, (GDExtensionTypePtr)&opaque, &p_path);
}

int64_t String::unicode_at(int64_t p_at) const {
	int64_t p_at_encoded;
	PtrToArg<int64_t>::encode(p_at, &p_at_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_unicode_at, (GDExtensionTypePtr)&opaque, &p_at_encoded);
}

String String::indent(const String &p_prefix) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_indent, (GDExtensionTypePtr)&opaque, &p_prefix);
}

String String::dedent() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_dedent, (GDExtensionTypePtr)&opaque);
}

int64_t String::hash() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_hash, (GDExtensionTypePtr)&opaque);
}

String String::md5_text() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_md5_text, (GDExtensionTypePtr)&opaque);
}

String String::sha1_text() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_sha1_text, (GDExtensionTypePtr)&opaque);
}

String String::sha256_text() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_sha256_text, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::md5_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_md5_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::sha1_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_sha1_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::sha256_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_sha256_buffer, (GDExtensionTypePtr)&opaque);
}

bool String::is_empty() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_empty, (GDExtensionTypePtr)&opaque);
}

bool String::contains(const String &p_what) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_contains, (GDExtensionTypePtr)&opaque, &p_what);
}

bool String::containsn(const String &p_what) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_containsn, (GDExtensionTypePtr)&opaque, &p_what);
}

bool String::is_absolute_path() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_absolute_path, (GDExtensionTypePtr)&opaque);
}

bool String::is_relative_path() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_relative_path, (GDExtensionTypePtr)&opaque);
}

String String::simplify_path() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_simplify_path, (GDExtensionTypePtr)&opaque);
}

String String::get_base_dir() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_base_dir, (GDExtensionTypePtr)&opaque);
}

String String::get_file() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_file, (GDExtensionTypePtr)&opaque);
}

String String::xml_escape(bool p_escape_quotes) const {
	int8_t p_escape_quotes_encoded;
	PtrToArg<bool>::encode(p_escape_quotes, &p_escape_quotes_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_xml_escape, (GDExtensionTypePtr)&opaque, &p_escape_quotes_encoded);
}

String String::xml_unescape() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_xml_unescape, (GDExtensionTypePtr)&opaque);
}

String String::uri_encode() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_uri_encode, (GDExtensionTypePtr)&opaque);
}

String String::uri_decode() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_uri_decode, (GDExtensionTypePtr)&opaque);
}

String String::uri_file_decode() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_uri_file_decode, (GDExtensionTypePtr)&opaque);
}

String String::c_escape() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_c_escape, (GDExtensionTypePtr)&opaque);
}

String String::c_unescape() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_c_unescape, (GDExtensionTypePtr)&opaque);
}

String String::json_escape() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_json_escape, (GDExtensionTypePtr)&opaque);
}

String String::validate_node_name() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_validate_node_name, (GDExtensionTypePtr)&opaque);
}

String String::validate_filename() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_validate_filename, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_ascii_identifier() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_ascii_identifier, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_unicode_identifier() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_unicode_identifier, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_identifier() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_identifier, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_int() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_int, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_float() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_float, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_hex_number(bool p_with_prefix) const {
	int8_t p_with_prefix_encoded;
	PtrToArg<bool>::encode(p_with_prefix, &p_with_prefix_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_hex_number, (GDExtensionTypePtr)&opaque, &p_with_prefix_encoded);
}

bool String::is_valid_html_color() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_html_color, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_ip_address() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_ip_address, (GDExtensionTypePtr)&opaque);
}

bool String::is_valid_filename() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid_filename, (GDExtensionTypePtr)&opaque);
}

int64_t String::to_int() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_to_int, (GDExtensionTypePtr)&opaque);
}

double String::to_float() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<double>(_method_bindings.method_to_float, (GDExtensionTypePtr)&opaque);
}

int64_t String::hex_to_int() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_hex_to_int, (GDExtensionTypePtr)&opaque);
}

int64_t String::bin_to_int() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_bin_to_int, (GDExtensionTypePtr)&opaque);
}

String String::lpad(int64_t p_min_length, const String &p_character) const {
	int64_t p_min_length_encoded;
	PtrToArg<int64_t>::encode(p_min_length, &p_min_length_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_lpad, (GDExtensionTypePtr)&opaque, &p_min_length_encoded, &p_character);
}

String String::rpad(int64_t p_min_length, const String &p_character) const {
	int64_t p_min_length_encoded;
	PtrToArg<int64_t>::encode(p_min_length, &p_min_length_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_rpad, (GDExtensionTypePtr)&opaque, &p_min_length_encoded, &p_character);
}

String String::pad_decimals(int64_t p_digits) const {
	int64_t p_digits_encoded;
	PtrToArg<int64_t>::encode(p_digits, &p_digits_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_pad_decimals, (GDExtensionTypePtr)&opaque, &p_digits_encoded);
}

String String::pad_zeros(int64_t p_digits) const {
	int64_t p_digits_encoded;
	PtrToArg<int64_t>::encode(p_digits, &p_digits_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_pad_zeros, (GDExtensionTypePtr)&opaque, &p_digits_encoded);
}

String String::trim_prefix(const String &p_prefix) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_trim_prefix, (GDExtensionTypePtr)&opaque, &p_prefix);
}

String String::trim_suffix(const String &p_suffix) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_trim_suffix, (GDExtensionTypePtr)&opaque, &p_suffix);
}

PackedByteArray String::to_ascii_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_ascii_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::to_utf8_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_utf8_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::to_utf16_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_utf16_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::to_utf32_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_utf32_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::to_wchar_buffer() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_wchar_buffer, (GDExtensionTypePtr)&opaque);
}

PackedByteArray String::to_multibyte_char_buffer(const String &p_encoding) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_multibyte_char_buffer, (GDExtensionTypePtr)&opaque, &p_encoding);
}

PackedByteArray String::hex_decode() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_hex_decode, (GDExtensionTypePtr)&opaque);
}

String String::num_scientific(double p_number) {
	double p_number_encoded;
	PtrToArg<double>::encode(p_number, &p_number_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_num_scientific, nullptr, &p_number_encoded);
}

String String::num(double p_number, int64_t p_decimals) {
	double p_number_encoded;
	PtrToArg<double>::encode(p_number, &p_number_encoded);
	int64_t p_decimals_encoded;
	PtrToArg<int64_t>::encode(p_decimals, &p_decimals_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_num, nullptr, &p_number_encoded, &p_decimals_encoded);
}

String String::num_int64(int64_t p_number, int64_t p_base, bool p_capitalize_hex) {
	int64_t p_number_encoded;
	PtrToArg<int64_t>::encode(p_number, &p_number_encoded);
	int64_t p_base_encoded;
	PtrToArg<int64_t>::encode(p_base, &p_base_encoded);
	int8_t p_capitalize_hex_encoded;
	PtrToArg<bool>::encode(p_capitalize_hex, &p_capitalize_hex_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_num_int64, nullptr, &p_number_encoded, &p_base_encoded, &p_capitalize_hex_encoded);
}

String String::num_uint64(int64_t p_number, int64_t p_base, bool p_capitalize_hex) {
	int64_t p_number_encoded;
	PtrToArg<int64_t>::encode(p_number, &p_number_encoded);
	int64_t p_base_encoded;
	PtrToArg<int64_t>::encode(p_base, &p_base_encoded);
	int8_t p_capitalize_hex_encoded;
	PtrToArg<bool>::encode(p_capitalize_hex, &p_capitalize_hex_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_num_uint64, nullptr, &p_number_encoded, &p_base_encoded, &p_capitalize_hex_encoded);
}

String String::chr(int64_t p_code) {
	int64_t p_code_encoded;
	PtrToArg<int64_t>::encode(p_code, &p_code_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_chr, nullptr, &p_code_encoded);
}

String String::humanize_size(int64_t p_size) {
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_humanize_size, nullptr, &p_size_encoded);
}

bool String::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

String String::operator%(bool p_other) const {
	int8_t p_other_encoded;
	PtrToArg<bool>::encode(p_other, &p_other_encoded);
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_bool, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other_encoded);
}

String String::operator%(int64_t p_other) const {
	int64_t p_other_encoded;
	PtrToArg<int64_t>::encode(p_other, &p_other_encoded);
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_int, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other_encoded);
}

String String::operator%(double p_other) const {
	double p_other_encoded;
	PtrToArg<double>::encode(p_other, &p_other_encoded);
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_float, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other_encoded);
}

bool String::operator==(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator!=(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator<(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_less_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator<=(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_less_equal_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator>(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_greater_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator>=(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_greater_equal_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator+(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_add_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const String &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_String, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Vector2 &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Vector2, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Vector2i &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Vector2i, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Rect2 &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Rect2, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Rect2i &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Rect2i, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Vector3 &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Vector3, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Vector3i &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Vector3i, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Transform2D &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Transform2D, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Vector4 &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Vector4, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Vector4i &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Vector4i, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Plane &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Plane, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Quaternion &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Quaternion, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const AABB &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_AABB, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Basis &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Basis, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Transform3D &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Transform3D, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Projection &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Projection, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Color &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Color, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator==(const StringName &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_StringName, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool String::operator!=(const StringName &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_StringName, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator+(const StringName &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_add_StringName, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const StringName &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_StringName, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const NodePath &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_NodePath, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const RID &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_RID, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(Object *p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Object, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)(p_other != nullptr ? &p_other->_owner : nullptr));
}

String String::operator%(const Callable &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Callable, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Signal &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Signal, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Dictionary &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Dictionary, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedByteArray &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedByteArray, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedInt32Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedInt32Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedInt64Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedInt64Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedFloat32Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedFloat32Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedFloat64Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedFloat64Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedStringArray &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedStringArray, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedVector2Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedVector2Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedVector3Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedVector3Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedColorArray &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedColorArray, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String String::operator%(const PackedVector4Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<String>(_method_bindings.operator_module_PackedVector4Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

String &String::operator=(const String &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

String &String::operator=(String &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
