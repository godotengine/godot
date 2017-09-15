/*************************************************************************/
/*  gdnative.cpp                                                         */
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
#include "gdnative.h"

#include "global_constants.h"
#include "io/file_access_encrypted.h"
#include "os/file_access.h"
#include "os/os.h"
#include "project_settings.h"

#include "scene/main/scene_tree.h"

const String init_symbol = "godot_gdnative_init";
const String terminate_symbol = "godot_gdnative_terminate";
const godot_gdnative_api_struct api_struct = {
	.godot_color_new_rgba = godot_color_new_rgba,
	.godot_color_new_rgb = godot_color_new_rgb,
	.godot_color_get_r = godot_color_get_r,
	.godot_color_set_r = godot_color_set_r,
	.godot_color_get_g = godot_color_get_g,
	.godot_color_set_g = godot_color_set_g,
	.godot_color_get_b = godot_color_get_b,
	.godot_color_set_b = godot_color_set_b,
	.godot_color_get_a = godot_color_get_a,
	.godot_color_set_a = godot_color_set_a,
	.godot_color_get_h = godot_color_get_h,
	.godot_color_get_s = godot_color_get_s,
	.godot_color_get_v = godot_color_get_v,
	.godot_color_as_string = godot_color_as_string,
	.godot_color_to_rgba32 = godot_color_to_rgba32,
	.godot_color_to_argb32 = godot_color_to_argb32,
	.godot_color_gray = godot_color_gray,
	.godot_color_inverted = godot_color_inverted,
	.godot_color_contrasted = godot_color_contrasted,
	.godot_color_linear_interpolate = godot_color_linear_interpolate,
	.godot_color_blend = godot_color_blend,
	.godot_color_to_html = godot_color_to_html,
	.godot_color_operator_equal = godot_color_operator_equal,
	.godot_color_operator_less = godot_color_operator_less,
	.godot_vector2_new = godot_vector2_new,
	.godot_vector2_as_string = godot_vector2_as_string,
	.godot_vector2_normalized = godot_vector2_normalized,
	.godot_vector2_length = godot_vector2_length,
	.godot_vector2_angle = godot_vector2_angle,
	.godot_vector2_length_squared = godot_vector2_length_squared,
	.godot_vector2_is_normalized = godot_vector2_is_normalized,
	.godot_vector2_distance_to = godot_vector2_distance_to,
	.godot_vector2_distance_squared_to = godot_vector2_distance_squared_to,
	.godot_vector2_angle_to = godot_vector2_angle_to,
	.godot_vector2_angle_to_point = godot_vector2_angle_to_point,
	.godot_vector2_linear_interpolate = godot_vector2_linear_interpolate,
	.godot_vector2_cubic_interpolate = godot_vector2_cubic_interpolate,
	.godot_vector2_rotated = godot_vector2_rotated,
	.godot_vector2_tangent = godot_vector2_tangent,
	.godot_vector2_floor = godot_vector2_floor,
	.godot_vector2_snapped = godot_vector2_snapped,
	.godot_vector2_aspect = godot_vector2_aspect,
	.godot_vector2_dot = godot_vector2_dot,
	.godot_vector2_slide = godot_vector2_slide,
	.godot_vector2_bounce = godot_vector2_bounce,
	.godot_vector2_reflect = godot_vector2_reflect,
	.godot_vector2_abs = godot_vector2_abs,
	.godot_vector2_clamped = godot_vector2_clamped,
	.godot_vector2_operator_add = godot_vector2_operator_add,
	.godot_vector2_operator_substract = godot_vector2_operator_substract,
	.godot_vector2_operator_multiply_vector = godot_vector2_operator_multiply_vector,
	.godot_vector2_operator_multiply_scalar = godot_vector2_operator_multiply_scalar,
	.godot_vector2_operator_divide_vector = godot_vector2_operator_divide_vector,
	.godot_vector2_operator_divide_scalar = godot_vector2_operator_divide_scalar,
	.godot_vector2_operator_equal = godot_vector2_operator_equal,
	.godot_vector2_operator_less = godot_vector2_operator_less,
	.godot_vector2_operator_neg = godot_vector2_operator_neg,
	.godot_vector2_set_x = godot_vector2_set_x,
	.godot_vector2_set_y = godot_vector2_set_y,
	.godot_vector2_get_x = godot_vector2_get_x,
	.godot_vector2_get_y = godot_vector2_get_y,
	.godot_quat_new = godot_quat_new,
	.godot_quat_new_with_axis_angle = godot_quat_new_with_axis_angle,
	.godot_quat_get_x = godot_quat_get_x,
	.godot_quat_set_x = godot_quat_set_x,
	.godot_quat_get_y = godot_quat_get_y,
	.godot_quat_set_y = godot_quat_set_y,
	.godot_quat_get_z = godot_quat_get_z,
	.godot_quat_set_z = godot_quat_set_z,
	.godot_quat_get_w = godot_quat_get_w,
	.godot_quat_set_w = godot_quat_set_w,
	.godot_quat_as_string = godot_quat_as_string,
	.godot_quat_length = godot_quat_length,
	.godot_quat_length_squared = godot_quat_length_squared,
	.godot_quat_normalized = godot_quat_normalized,
	.godot_quat_is_normalized = godot_quat_is_normalized,
	.godot_quat_inverse = godot_quat_inverse,
	.godot_quat_dot = godot_quat_dot,
	.godot_quat_xform = godot_quat_xform,
	.godot_quat_slerp = godot_quat_slerp,
	.godot_quat_slerpni = godot_quat_slerpni,
	.godot_quat_cubic_slerp = godot_quat_cubic_slerp,
	.godot_quat_operator_multiply = godot_quat_operator_multiply,
	.godot_quat_operator_add = godot_quat_operator_add,
	.godot_quat_operator_substract = godot_quat_operator_substract,
	.godot_quat_operator_divide = godot_quat_operator_divide,
	.godot_quat_operator_equal = godot_quat_operator_equal,
	.godot_quat_operator_neg = godot_quat_operator_neg,
	.godot_basis_new_with_rows = godot_basis_new_with_rows,
	.godot_basis_new_with_axis_and_angle = godot_basis_new_with_axis_and_angle,
	.godot_basis_new_with_euler = godot_basis_new_with_euler,
	.godot_basis_as_string = godot_basis_as_string,
	.godot_basis_inverse = godot_basis_inverse,
	.godot_basis_transposed = godot_basis_transposed,
	.godot_basis_orthonormalized = godot_basis_orthonormalized,
	.godot_basis_determinant = godot_basis_determinant,
	.godot_basis_rotated = godot_basis_rotated,
	.godot_basis_scaled = godot_basis_scaled,
	.godot_basis_get_scale = godot_basis_get_scale,
	.godot_basis_get_euler = godot_basis_get_euler,
	.godot_basis_tdotx = godot_basis_tdotx,
	.godot_basis_tdoty = godot_basis_tdoty,
	.godot_basis_tdotz = godot_basis_tdotz,
	.godot_basis_xform = godot_basis_xform,
	.godot_basis_xform_inv = godot_basis_xform_inv,
	.godot_basis_get_orthogonal_index = godot_basis_get_orthogonal_index,
	.godot_basis_new = godot_basis_new,
	.godot_basis_new_with_euler_quat = godot_basis_new_with_euler_quat,
	.godot_basis_get_elements = godot_basis_get_elements,
	.godot_basis_get_axis = godot_basis_get_axis,
	.godot_basis_set_axis = godot_basis_set_axis,
	.godot_basis_get_row = godot_basis_get_row,
	.godot_basis_set_row = godot_basis_set_row,
	.godot_basis_operator_equal = godot_basis_operator_equal,
	.godot_basis_operator_add = godot_basis_operator_add,
	.godot_basis_operator_substract = godot_basis_operator_substract,
	.godot_basis_operator_multiply_vector = godot_basis_operator_multiply_vector,
	.godot_basis_operator_multiply_scalar = godot_basis_operator_multiply_scalar,
	.godot_vector3_new = godot_vector3_new,
	.godot_vector3_as_string = godot_vector3_as_string,
	.godot_vector3_min_axis = godot_vector3_min_axis,
	.godot_vector3_max_axis = godot_vector3_max_axis,
	.godot_vector3_length = godot_vector3_length,
	.godot_vector3_length_squared = godot_vector3_length_squared,
	.godot_vector3_is_normalized = godot_vector3_is_normalized,
	.godot_vector3_normalized = godot_vector3_normalized,
	.godot_vector3_inverse = godot_vector3_inverse,
	.godot_vector3_snapped = godot_vector3_snapped,
	.godot_vector3_rotated = godot_vector3_rotated,
	.godot_vector3_linear_interpolate = godot_vector3_linear_interpolate,
	.godot_vector3_cubic_interpolate = godot_vector3_cubic_interpolate,
	.godot_vector3_dot = godot_vector3_dot,
	.godot_vector3_cross = godot_vector3_cross,
	.godot_vector3_outer = godot_vector3_outer,
	.godot_vector3_to_diagonal_matrix = godot_vector3_to_diagonal_matrix,
	.godot_vector3_abs = godot_vector3_abs,
	.godot_vector3_floor = godot_vector3_floor,
	.godot_vector3_ceil = godot_vector3_ceil,
	.godot_vector3_distance_to = godot_vector3_distance_to,
	.godot_vector3_distance_squared_to = godot_vector3_distance_squared_to,
	.godot_vector3_angle_to = godot_vector3_angle_to,
	.godot_vector3_slide = godot_vector3_slide,
	.godot_vector3_bounce = godot_vector3_bounce,
	.godot_vector3_reflect = godot_vector3_reflect,
	.godot_vector3_operator_add = godot_vector3_operator_add,
	.godot_vector3_operator_substract = godot_vector3_operator_substract,
	.godot_vector3_operator_multiply_vector = godot_vector3_operator_multiply_vector,
	.godot_vector3_operator_multiply_scalar = godot_vector3_operator_multiply_scalar,
	.godot_vector3_operator_divide_vector = godot_vector3_operator_divide_vector,
	.godot_vector3_operator_divide_scalar = godot_vector3_operator_divide_scalar,
	.godot_vector3_operator_equal = godot_vector3_operator_equal,
	.godot_vector3_operator_less = godot_vector3_operator_less,
	.godot_vector3_operator_neg = godot_vector3_operator_neg,
	.godot_vector3_set_axis = godot_vector3_set_axis,
	.godot_vector3_get_axis = godot_vector3_get_axis,
	.godot_pool_byte_array_new = godot_pool_byte_array_new,
	.godot_pool_byte_array_new_copy = godot_pool_byte_array_new_copy,
	.godot_pool_byte_array_new_with_array = godot_pool_byte_array_new_with_array,
	.godot_pool_byte_array_append = godot_pool_byte_array_append,
	.godot_pool_byte_array_append_array = godot_pool_byte_array_append_array,
	.godot_pool_byte_array_insert = godot_pool_byte_array_insert,
	.godot_pool_byte_array_invert = godot_pool_byte_array_invert,
	.godot_pool_byte_array_push_back = godot_pool_byte_array_push_back,
	.godot_pool_byte_array_remove = godot_pool_byte_array_remove,
	.godot_pool_byte_array_resize = godot_pool_byte_array_resize,
	.godot_pool_byte_array_set = godot_pool_byte_array_set,
	.godot_pool_byte_array_get = godot_pool_byte_array_get,
	.godot_pool_byte_array_size = godot_pool_byte_array_size,
	.godot_pool_byte_array_destroy = godot_pool_byte_array_destroy,
	.godot_pool_int_array_new = godot_pool_int_array_new,
	.godot_pool_int_array_new_copy = godot_pool_int_array_new_copy,
	.godot_pool_int_array_new_with_array = godot_pool_int_array_new_with_array,
	.godot_pool_int_array_append = godot_pool_int_array_append,
	.godot_pool_int_array_append_array = godot_pool_int_array_append_array,
	.godot_pool_int_array_insert = godot_pool_int_array_insert,
	.godot_pool_int_array_invert = godot_pool_int_array_invert,
	.godot_pool_int_array_push_back = godot_pool_int_array_push_back,
	.godot_pool_int_array_remove = godot_pool_int_array_remove,
	.godot_pool_int_array_resize = godot_pool_int_array_resize,
	.godot_pool_int_array_set = godot_pool_int_array_set,
	.godot_pool_int_array_get = godot_pool_int_array_get,
	.godot_pool_int_array_size = godot_pool_int_array_size,
	.godot_pool_int_array_destroy = godot_pool_int_array_destroy,
	.godot_pool_real_array_new = godot_pool_real_array_new,
	.godot_pool_real_array_new_copy = godot_pool_real_array_new_copy,
	.godot_pool_real_array_new_with_array = godot_pool_real_array_new_with_array,
	.godot_pool_real_array_append = godot_pool_real_array_append,
	.godot_pool_real_array_append_array = godot_pool_real_array_append_array,
	.godot_pool_real_array_insert = godot_pool_real_array_insert,
	.godot_pool_real_array_invert = godot_pool_real_array_invert,
	.godot_pool_real_array_push_back = godot_pool_real_array_push_back,
	.godot_pool_real_array_remove = godot_pool_real_array_remove,
	.godot_pool_real_array_resize = godot_pool_real_array_resize,
	.godot_pool_real_array_set = godot_pool_real_array_set,
	.godot_pool_real_array_get = godot_pool_real_array_get,
	.godot_pool_real_array_size = godot_pool_real_array_size,
	.godot_pool_real_array_destroy = godot_pool_real_array_destroy,
	.godot_pool_string_array_new = godot_pool_string_array_new,
	.godot_pool_string_array_new_copy = godot_pool_string_array_new_copy,
	.godot_pool_string_array_new_with_array = godot_pool_string_array_new_with_array,
	.godot_pool_string_array_append = godot_pool_string_array_append,
	.godot_pool_string_array_append_array = godot_pool_string_array_append_array,
	.godot_pool_string_array_insert = godot_pool_string_array_insert,
	.godot_pool_string_array_invert = godot_pool_string_array_invert,
	.godot_pool_string_array_push_back = godot_pool_string_array_push_back,
	.godot_pool_string_array_remove = godot_pool_string_array_remove,
	.godot_pool_string_array_resize = godot_pool_string_array_resize,
	.godot_pool_string_array_set = godot_pool_string_array_set,
	.godot_pool_string_array_get = godot_pool_string_array_get,
	.godot_pool_string_array_size = godot_pool_string_array_size,
	.godot_pool_string_array_destroy = godot_pool_string_array_destroy,
	.godot_pool_vector2_array_new = godot_pool_vector2_array_new,
	.godot_pool_vector2_array_new_copy = godot_pool_vector2_array_new_copy,
	.godot_pool_vector2_array_new_with_array = godot_pool_vector2_array_new_with_array,
	.godot_pool_vector2_array_append = godot_pool_vector2_array_append,
	.godot_pool_vector2_array_append_array = godot_pool_vector2_array_append_array,
	.godot_pool_vector2_array_insert = godot_pool_vector2_array_insert,
	.godot_pool_vector2_array_invert = godot_pool_vector2_array_invert,
	.godot_pool_vector2_array_push_back = godot_pool_vector2_array_push_back,
	.godot_pool_vector2_array_remove = godot_pool_vector2_array_remove,
	.godot_pool_vector2_array_resize = godot_pool_vector2_array_resize,
	.godot_pool_vector2_array_set = godot_pool_vector2_array_set,
	.godot_pool_vector2_array_get = godot_pool_vector2_array_get,
	.godot_pool_vector2_array_size = godot_pool_vector2_array_size,
	.godot_pool_vector2_array_destroy = godot_pool_vector2_array_destroy,
	.godot_pool_vector3_array_new = godot_pool_vector3_array_new,
	.godot_pool_vector3_array_new_copy = godot_pool_vector3_array_new_copy,
	.godot_pool_vector3_array_new_with_array = godot_pool_vector3_array_new_with_array,
	.godot_pool_vector3_array_append = godot_pool_vector3_array_append,
	.godot_pool_vector3_array_append_array = godot_pool_vector3_array_append_array,
	.godot_pool_vector3_array_insert = godot_pool_vector3_array_insert,
	.godot_pool_vector3_array_invert = godot_pool_vector3_array_invert,
	.godot_pool_vector3_array_push_back = godot_pool_vector3_array_push_back,
	.godot_pool_vector3_array_remove = godot_pool_vector3_array_remove,
	.godot_pool_vector3_array_resize = godot_pool_vector3_array_resize,
	.godot_pool_vector3_array_set = godot_pool_vector3_array_set,
	.godot_pool_vector3_array_get = godot_pool_vector3_array_get,
	.godot_pool_vector3_array_size = godot_pool_vector3_array_size,
	.godot_pool_vector3_array_destroy = godot_pool_vector3_array_destroy,
	.godot_pool_color_array_new = godot_pool_color_array_new,
	.godot_pool_color_array_new_copy = godot_pool_color_array_new_copy,
	.godot_pool_color_array_new_with_array = godot_pool_color_array_new_with_array,
	.godot_pool_color_array_append = godot_pool_color_array_append,
	.godot_pool_color_array_append_array = godot_pool_color_array_append_array,
	.godot_pool_color_array_insert = godot_pool_color_array_insert,
	.godot_pool_color_array_invert = godot_pool_color_array_invert,
	.godot_pool_color_array_push_back = godot_pool_color_array_push_back,
	.godot_pool_color_array_remove = godot_pool_color_array_remove,
	.godot_pool_color_array_resize = godot_pool_color_array_resize,
	.godot_pool_color_array_set = godot_pool_color_array_set,
	.godot_pool_color_array_get = godot_pool_color_array_get,
	.godot_pool_color_array_size = godot_pool_color_array_size,
	.godot_pool_color_array_destroy = godot_pool_color_array_destroy,
	.godot_array_new = godot_array_new,
	.godot_array_new_copy = godot_array_new_copy,
	.godot_array_new_pool_color_array = godot_array_new_pool_color_array,
	.godot_array_new_pool_vector3_array = godot_array_new_pool_vector3_array,
	.godot_array_new_pool_vector2_array = godot_array_new_pool_vector2_array,
	.godot_array_new_pool_string_array = godot_array_new_pool_string_array,
	.godot_array_new_pool_real_array = godot_array_new_pool_real_array,
	.godot_array_new_pool_int_array = godot_array_new_pool_int_array,
	.godot_array_new_pool_byte_array = godot_array_new_pool_byte_array,
	.godot_array_set = godot_array_set,
	.godot_array_get = godot_array_get,
	.godot_array_operator_index = godot_array_operator_index,
	.godot_array_append = godot_array_append,
	.godot_array_clear = godot_array_clear,
	.godot_array_count = godot_array_count,
	.godot_array_empty = godot_array_empty,
	.godot_array_erase = godot_array_erase,
	.godot_array_front = godot_array_front,
	.godot_array_back = godot_array_back,
	.godot_array_find = godot_array_find,
	.godot_array_find_last = godot_array_find_last,
	.godot_array_has = godot_array_has,
	.godot_array_hash = godot_array_hash,
	.godot_array_insert = godot_array_insert,
	.godot_array_invert = godot_array_invert,
	.godot_array_pop_back = godot_array_pop_back,
	.godot_array_pop_front = godot_array_pop_front,
	.godot_array_push_back = godot_array_push_back,
	.godot_array_push_front = godot_array_push_front,
	.godot_array_remove = godot_array_remove,
	.godot_array_resize = godot_array_resize,
	.godot_array_rfind = godot_array_rfind,
	.godot_array_size = godot_array_size,
	.godot_array_sort = godot_array_sort,
	.godot_array_sort_custom = godot_array_sort_custom,
	.godot_array_destroy = godot_array_destroy,
	.godot_dictionary_new = godot_dictionary_new,
	.godot_dictionary_new_copy = godot_dictionary_new_copy,
	.godot_dictionary_destroy = godot_dictionary_destroy,
	.godot_dictionary_size = godot_dictionary_size,
	.godot_dictionary_empty = godot_dictionary_empty,
	.godot_dictionary_clear = godot_dictionary_clear,
	.godot_dictionary_has = godot_dictionary_has,
	.godot_dictionary_has_all = godot_dictionary_has_all,
	.godot_dictionary_erase = godot_dictionary_erase,
	.godot_dictionary_hash = godot_dictionary_hash,
	.godot_dictionary_keys = godot_dictionary_keys,
	.godot_dictionary_values = godot_dictionary_values,
	.godot_dictionary_get = godot_dictionary_get,
	.godot_dictionary_set = godot_dictionary_set,
	.godot_dictionary_operator_index = godot_dictionary_operator_index,
	.godot_dictionary_next = godot_dictionary_next,
	.godot_dictionary_operator_equal = godot_dictionary_operator_equal,
	.godot_dictionary_to_json = godot_dictionary_to_json,
	.godot_node_path_new = godot_node_path_new,
	.godot_node_path_new_copy = godot_node_path_new_copy,
	.godot_node_path_destroy = godot_node_path_destroy,
	.godot_node_path_as_string = godot_node_path_as_string,
	.godot_node_path_is_absolute = godot_node_path_is_absolute,
	.godot_node_path_get_name_count = godot_node_path_get_name_count,
	.godot_node_path_get_name = godot_node_path_get_name,
	.godot_node_path_get_subname_count = godot_node_path_get_subname_count,
	.godot_node_path_get_subname = godot_node_path_get_subname,
	.godot_node_path_get_property = godot_node_path_get_property,
	.godot_node_path_is_empty = godot_node_path_is_empty,
	.godot_node_path_operator_equal = godot_node_path_operator_equal,
	.godot_plane_new_with_reals = godot_plane_new_with_reals,
	.godot_plane_new_with_vectors = godot_plane_new_with_vectors,
	.godot_plane_new_with_normal = godot_plane_new_with_normal,
	.godot_plane_as_string = godot_plane_as_string,
	.godot_plane_normalized = godot_plane_normalized,
	.godot_plane_center = godot_plane_center,
	.godot_plane_get_any_point = godot_plane_get_any_point,
	.godot_plane_is_point_over = godot_plane_is_point_over,
	.godot_plane_distance_to = godot_plane_distance_to,
	.godot_plane_has_point = godot_plane_has_point,
	.godot_plane_project = godot_plane_project,
	.godot_plane_intersect_3 = godot_plane_intersect_3,
	.godot_plane_intersects_ray = godot_plane_intersects_ray,
	.godot_plane_intersects_segment = godot_plane_intersects_segment,
	.godot_plane_operator_neg = godot_plane_operator_neg,
	.godot_plane_operator_equal = godot_plane_operator_equal,
	.godot_plane_set_normal = godot_plane_set_normal,
	.godot_plane_get_normal = godot_plane_get_normal,
	.godot_plane_get_d = godot_plane_get_d,
	.godot_plane_set_d = godot_plane_set_d,
	.godot_rect2_new_with_position_and_size = godot_rect2_new_with_position_and_size,
	.godot_rect2_new = godot_rect2_new,
	.godot_rect2_as_string = godot_rect2_as_string,
	.godot_rect2_get_area = godot_rect2_get_area,
	.godot_rect2_intersects = godot_rect2_intersects,
	.godot_rect2_encloses = godot_rect2_encloses,
	.godot_rect2_has_no_area = godot_rect2_has_no_area,
	.godot_rect2_clip = godot_rect2_clip,
	.godot_rect2_merge = godot_rect2_merge,
	.godot_rect2_has_point = godot_rect2_has_point,
	.godot_rect2_grow = godot_rect2_grow,
	.godot_rect2_expand = godot_rect2_expand,
	.godot_rect2_operator_equal = godot_rect2_operator_equal,
	.godot_rect2_get_position = godot_rect2_get_position,
	.godot_rect2_get_size = godot_rect2_get_size,
	.godot_rect2_set_position = godot_rect2_set_position,
	.godot_rect2_set_size = godot_rect2_set_size,
	.godot_rect3_new = godot_rect3_new,
	.godot_rect3_get_position = godot_rect3_get_position,
	.godot_rect3_set_position = godot_rect3_set_position,
	.godot_rect3_get_size = godot_rect3_get_size,
	.godot_rect3_set_size = godot_rect3_set_size,
	.godot_rect3_as_string = godot_rect3_as_string,
	.godot_rect3_get_area = godot_rect3_get_area,
	.godot_rect3_has_no_area = godot_rect3_has_no_area,
	.godot_rect3_has_no_surface = godot_rect3_has_no_surface,
	.godot_rect3_intersects = godot_rect3_intersects,
	.godot_rect3_encloses = godot_rect3_encloses,
	.godot_rect3_merge = godot_rect3_merge,
	.godot_rect3_intersection = godot_rect3_intersection,
	.godot_rect3_intersects_plane = godot_rect3_intersects_plane,
	.godot_rect3_intersects_segment = godot_rect3_intersects_segment,
	.godot_rect3_has_point = godot_rect3_has_point,
	.godot_rect3_get_support = godot_rect3_get_support,
	.godot_rect3_get_longest_axis = godot_rect3_get_longest_axis,
	.godot_rect3_get_longest_axis_index = godot_rect3_get_longest_axis_index,
	.godot_rect3_get_longest_axis_size = godot_rect3_get_longest_axis_size,
	.godot_rect3_get_shortest_axis = godot_rect3_get_shortest_axis,
	.godot_rect3_get_shortest_axis_index = godot_rect3_get_shortest_axis_index,
	.godot_rect3_get_shortest_axis_size = godot_rect3_get_shortest_axis_size,
	.godot_rect3_expand = godot_rect3_expand,
	.godot_rect3_grow = godot_rect3_grow,
	.godot_rect3_get_endpoint = godot_rect3_get_endpoint,
	.godot_rect3_operator_equal = godot_rect3_operator_equal,
	.godot_rid_new = godot_rid_new,
	.godot_rid_get_id = godot_rid_get_id,
	.godot_rid_new_with_resource = godot_rid_new_with_resource,
	.godot_rid_operator_equal = godot_rid_operator_equal,
	.godot_rid_operator_less = godot_rid_operator_less,
	.godot_transform_new_with_axis_origin = godot_transform_new_with_axis_origin,
	.godot_transform_new = godot_transform_new,
	.godot_transform_get_basis = godot_transform_get_basis,
	.godot_transform_set_basis = godot_transform_set_basis,
	.godot_transform_get_origin = godot_transform_get_origin,
	.godot_transform_set_origin = godot_transform_set_origin,
	.godot_transform_as_string = godot_transform_as_string,
	.godot_transform_inverse = godot_transform_inverse,
	.godot_transform_affine_inverse = godot_transform_affine_inverse,
	.godot_transform_orthonormalized = godot_transform_orthonormalized,
	.godot_transform_rotated = godot_transform_rotated,
	.godot_transform_scaled = godot_transform_scaled,
	.godot_transform_translated = godot_transform_translated,
	.godot_transform_looking_at = godot_transform_looking_at,
	.godot_transform_xform_plane = godot_transform_xform_plane,
	.godot_transform_xform_inv_plane = godot_transform_xform_inv_plane,
	.godot_transform_new_identity = godot_transform_new_identity,
	.godot_transform_operator_equal = godot_transform_operator_equal,
	.godot_transform_operator_multiply = godot_transform_operator_multiply,
	.godot_transform_xform_vector3 = godot_transform_xform_vector3,
	.godot_transform_xform_inv_vector3 = godot_transform_xform_inv_vector3,
	.godot_transform_xform_rect3 = godot_transform_xform_rect3,
	.godot_transform_xform_inv_rect3 = godot_transform_xform_inv_rect3,
	.godot_transform2d_new = godot_transform2d_new,
	.godot_transform2d_new_axis_origin = godot_transform2d_new_axis_origin,
	.godot_transform2d_as_string = godot_transform2d_as_string,
	.godot_transform2d_inverse = godot_transform2d_inverse,
	.godot_transform2d_affine_inverse = godot_transform2d_affine_inverse,
	.godot_transform2d_get_rotation = godot_transform2d_get_rotation,
	.godot_transform2d_get_origin = godot_transform2d_get_origin,
	.godot_transform2d_get_scale = godot_transform2d_get_scale,
	.godot_transform2d_orthonormalized = godot_transform2d_orthonormalized,
	.godot_transform2d_rotated = godot_transform2d_rotated,
	.godot_transform2d_scaled = godot_transform2d_scaled,
	.godot_transform2d_translated = godot_transform2d_translated,
	.godot_transform2d_xform_vector2 = godot_transform2d_xform_vector2,
	.godot_transform2d_xform_inv_vector2 = godot_transform2d_xform_inv_vector2,
	.godot_transform2d_basis_xform_vector2 = godot_transform2d_basis_xform_vector2,
	.godot_transform2d_basis_xform_inv_vector2 = godot_transform2d_basis_xform_inv_vector2,
	.godot_transform2d_interpolate_with = godot_transform2d_interpolate_with,
	.godot_transform2d_operator_equal = godot_transform2d_operator_equal,
	.godot_transform2d_operator_multiply = godot_transform2d_operator_multiply,
	.godot_transform2d_new_identity = godot_transform2d_new_identity,
	.godot_transform2d_xform_rect2 = godot_transform2d_xform_rect2,
	.godot_transform2d_xform_inv_rect2 = godot_transform2d_xform_inv_rect2,
	.godot_variant_get_type = godot_variant_get_type,
	.godot_variant_new_copy = godot_variant_new_copy,
	.godot_variant_new_nil = godot_variant_new_nil,
	.godot_variant_new_bool = godot_variant_new_bool,
	.godot_variant_new_uint = godot_variant_new_uint,
	.godot_variant_new_int = godot_variant_new_int,
	.godot_variant_new_real = godot_variant_new_real,
	.godot_variant_new_string = godot_variant_new_string,
	.godot_variant_new_vector2 = godot_variant_new_vector2,
	.godot_variant_new_rect2 = godot_variant_new_rect2,
	.godot_variant_new_vector3 = godot_variant_new_vector3,
	.godot_variant_new_transform2d = godot_variant_new_transform2d,
	.godot_variant_new_plane = godot_variant_new_plane,
	.godot_variant_new_quat = godot_variant_new_quat,
	.godot_variant_new_rect3 = godot_variant_new_rect3,
	.godot_variant_new_basis = godot_variant_new_basis,
	.godot_variant_new_transform = godot_variant_new_transform,
	.godot_variant_new_color = godot_variant_new_color,
	.godot_variant_new_node_path = godot_variant_new_node_path,
	.godot_variant_new_rid = godot_variant_new_rid,
	.godot_variant_new_object = godot_variant_new_object,
	.godot_variant_new_dictionary = godot_variant_new_dictionary,
	.godot_variant_new_array = godot_variant_new_array,
	.godot_variant_new_pool_byte_array = godot_variant_new_pool_byte_array,
	.godot_variant_new_pool_int_array = godot_variant_new_pool_int_array,
	.godot_variant_new_pool_real_array = godot_variant_new_pool_real_array,
	.godot_variant_new_pool_string_array = godot_variant_new_pool_string_array,
	.godot_variant_new_pool_vector2_array = godot_variant_new_pool_vector2_array,
	.godot_variant_new_pool_vector3_array = godot_variant_new_pool_vector3_array,
	.godot_variant_new_pool_color_array = godot_variant_new_pool_color_array,
	.godot_variant_as_bool = godot_variant_as_bool,
	.godot_variant_as_uint = godot_variant_as_uint,
	.godot_variant_as_int = godot_variant_as_int,
	.godot_variant_as_real = godot_variant_as_real,
	.godot_variant_as_string = godot_variant_as_string,
	.godot_variant_as_vector2 = godot_variant_as_vector2,
	.godot_variant_as_rect2 = godot_variant_as_rect2,
	.godot_variant_as_vector3 = godot_variant_as_vector3,
	.godot_variant_as_transform2d = godot_variant_as_transform2d,
	.godot_variant_as_plane = godot_variant_as_plane,
	.godot_variant_as_quat = godot_variant_as_quat,
	.godot_variant_as_rect3 = godot_variant_as_rect3,
	.godot_variant_as_basis = godot_variant_as_basis,
	.godot_variant_as_transform = godot_variant_as_transform,
	.godot_variant_as_color = godot_variant_as_color,
	.godot_variant_as_node_path = godot_variant_as_node_path,
	.godot_variant_as_rid = godot_variant_as_rid,
	.godot_variant_as_object = godot_variant_as_object,
	.godot_variant_as_dictionary = godot_variant_as_dictionary,
	.godot_variant_as_array = godot_variant_as_array,
	.godot_variant_as_pool_byte_array = godot_variant_as_pool_byte_array,
	.godot_variant_as_pool_int_array = godot_variant_as_pool_int_array,
	.godot_variant_as_pool_real_array = godot_variant_as_pool_real_array,
	.godot_variant_as_pool_string_array = godot_variant_as_pool_string_array,
	.godot_variant_as_pool_vector2_array = godot_variant_as_pool_vector2_array,
	.godot_variant_as_pool_vector3_array = godot_variant_as_pool_vector3_array,
	.godot_variant_as_pool_color_array = godot_variant_as_pool_color_array,
	.godot_variant_call = godot_variant_call,
	.godot_variant_has_method = godot_variant_has_method,
	.godot_variant_operator_equal = godot_variant_operator_equal,
	.godot_variant_operator_less = godot_variant_operator_less,
	.godot_variant_hash_compare = godot_variant_hash_compare,
	.godot_variant_booleanize = godot_variant_booleanize,
	.godot_variant_destroy = godot_variant_destroy,
	.godot_string_new = godot_string_new,
	.godot_string_new_copy = godot_string_new_copy,
	.godot_string_new_data = godot_string_new_data,
	.godot_string_new_unicode_data = godot_string_new_unicode_data,
	.godot_string_get_data = godot_string_get_data,
	.godot_string_operator_index = godot_string_operator_index,
	.godot_string_c_str = godot_string_c_str,
	.godot_string_unicode_str = godot_string_unicode_str,
	.godot_string_operator_equal = godot_string_operator_equal,
	.godot_string_operator_less = godot_string_operator_less,
	.godot_string_operator_plus = godot_string_operator_plus,
	.godot_string_length = godot_string_length,
	.godot_string_begins_with = godot_string_begins_with,
	.godot_string_begins_with_char_array = godot_string_begins_with_char_array,
	.godot_string_bigrams = godot_string_bigrams,
	.godot_string_chr = godot_string_chr,
	.godot_string_ends_with = godot_string_ends_with,
	.godot_string_find = godot_string_find,
	.godot_string_find_from = godot_string_find_from,
	.godot_string_findmk = godot_string_findmk,
	.godot_string_findmk_from = godot_string_findmk_from,
	.godot_string_findmk_from_in_place = godot_string_findmk_from_in_place,
	.godot_string_findn = godot_string_findn,
	.godot_string_findn_from = godot_string_findn_from,
	.godot_string_find_last = godot_string_find_last,
	.godot_string_format = godot_string_format,
	.godot_string_format_with_custom_placeholder = godot_string_format_with_custom_placeholder,
	.godot_string_hex_encode_buffer = godot_string_hex_encode_buffer,
	.godot_string_hex_to_int = godot_string_hex_to_int,
	.godot_string_hex_to_int_without_prefix = godot_string_hex_to_int_without_prefix,
	.godot_string_insert = godot_string_insert,
	.godot_string_is_numeric = godot_string_is_numeric,
	.godot_string_is_subsequence_of = godot_string_is_subsequence_of,
	.godot_string_is_subsequence_ofi = godot_string_is_subsequence_ofi,
	.godot_string_lpad = godot_string_lpad,
	.godot_string_lpad_with_custom_character = godot_string_lpad_with_custom_character,
	.godot_string_match = godot_string_match,
	.godot_string_matchn = godot_string_matchn,
	.godot_string_md5 = godot_string_md5,
	.godot_string_num = godot_string_num,
	.godot_string_num_int64 = godot_string_num_int64,
	.godot_string_num_int64_capitalized = godot_string_num_int64_capitalized,
	.godot_string_num_real = godot_string_num_real,
	.godot_string_num_scientific = godot_string_num_scientific,
	.godot_string_num_with_decimals = godot_string_num_with_decimals,
	.godot_string_pad_decimals = godot_string_pad_decimals,
	.godot_string_pad_zeros = godot_string_pad_zeros,
	.godot_string_replace_first = godot_string_replace_first,
	.godot_string_replace = godot_string_replace,
	.godot_string_replacen = godot_string_replacen,
	.godot_string_rfind = godot_string_rfind,
	.godot_string_rfindn = godot_string_rfindn,
	.godot_string_rfind_from = godot_string_rfind_from,
	.godot_string_rfindn_from = godot_string_rfindn_from,
	.godot_string_rpad = godot_string_rpad,
	.godot_string_rpad_with_custom_character = godot_string_rpad_with_custom_character,
	.godot_string_similarity = godot_string_similarity,
	.godot_string_sprintf = godot_string_sprintf,
	.godot_string_substr = godot_string_substr,
	.godot_string_to_double = godot_string_to_double,
	.godot_string_to_float = godot_string_to_float,
	.godot_string_to_int = godot_string_to_int,
	.godot_string_camelcase_to_underscore = godot_string_camelcase_to_underscore,
	.godot_string_camelcase_to_underscore_lowercased = godot_string_camelcase_to_underscore_lowercased,
	.godot_string_capitalize = godot_string_capitalize,
	.godot_string_char_to_double = godot_string_char_to_double,
	.godot_string_char_to_int = godot_string_char_to_int,
	.godot_string_wchar_to_int = godot_string_wchar_to_int,
	.godot_string_char_to_int_with_len = godot_string_char_to_int_with_len,
	.godot_string_char_to_int64_with_len = godot_string_char_to_int64_with_len,
	.godot_string_hex_to_int64 = godot_string_hex_to_int64,
	.godot_string_hex_to_int64_with_prefix = godot_string_hex_to_int64_with_prefix,
	.godot_string_to_int64 = godot_string_to_int64,
	.godot_string_unicode_char_to_double = godot_string_unicode_char_to_double,
	.godot_string_get_slice_count = godot_string_get_slice_count,
	.godot_string_get_slice = godot_string_get_slice,
	.godot_string_get_slicec = godot_string_get_slicec,
	.godot_string_split = godot_string_split,
	.godot_string_split_allow_empty = godot_string_split_allow_empty,
	.godot_string_split_floats = godot_string_split_floats,
	.godot_string_split_floats_allows_empty = godot_string_split_floats_allows_empty,
	.godot_string_split_floats_mk = godot_string_split_floats_mk,
	.godot_string_split_floats_mk_allows_empty = godot_string_split_floats_mk_allows_empty,
	.godot_string_split_ints = godot_string_split_ints,
	.godot_string_split_ints_allows_empty = godot_string_split_ints_allows_empty,
	.godot_string_split_ints_mk = godot_string_split_ints_mk,
	.godot_string_split_ints_mk_allows_empty = godot_string_split_ints_mk_allows_empty,
	.godot_string_split_spaces = godot_string_split_spaces,
	.godot_string_char_lowercase = godot_string_char_lowercase,
	.godot_string_char_uppercase = godot_string_char_uppercase,
	.godot_string_to_lower = godot_string_to_lower,
	.godot_string_to_upper = godot_string_to_upper,
	.godot_string_get_basename = godot_string_get_basename,
	.godot_string_get_extension = godot_string_get_extension,
	.godot_string_left = godot_string_left,
	.godot_string_ord_at = godot_string_ord_at,
	.godot_string_plus_file = godot_string_plus_file,
	.godot_string_right = godot_string_right,
	.godot_string_strip_edges = godot_string_strip_edges,
	.godot_string_strip_escapes = godot_string_strip_escapes,
	.godot_string_erase = godot_string_erase,
	.godot_string_ascii = godot_string_ascii,
	.godot_string_ascii_extended = godot_string_ascii_extended,
	.godot_string_utf8 = godot_string_utf8,
	.godot_string_parse_utf8 = godot_string_parse_utf8,
	.godot_string_parse_utf8_with_len = godot_string_parse_utf8_with_len,
	.godot_string_chars_to_utf8 = godot_string_chars_to_utf8,
	.godot_string_chars_to_utf8_with_len = godot_string_chars_to_utf8_with_len,
	.godot_string_hash = godot_string_hash,
	.godot_string_hash64 = godot_string_hash64,
	.godot_string_hash_chars = godot_string_hash_chars,
	.godot_string_hash_chars_with_len = godot_string_hash_chars_with_len,
	.godot_string_hash_utf8_chars = godot_string_hash_utf8_chars,
	.godot_string_hash_utf8_chars_with_len = godot_string_hash_utf8_chars_with_len,
	.godot_string_md5_buffer = godot_string_md5_buffer,
	.godot_string_md5_text = godot_string_md5_text,
	.godot_string_sha256_buffer = godot_string_sha256_buffer,
	.godot_string_sha256_text = godot_string_sha256_text,
	.godot_string_empty = godot_string_empty,
	.godot_string_get_base_dir = godot_string_get_base_dir,
	.godot_string_get_file = godot_string_get_file,
	.godot_string_humanize_size = godot_string_humanize_size,
	.godot_string_is_abs_path = godot_string_is_abs_path,
	.godot_string_is_rel_path = godot_string_is_rel_path,
	.godot_string_is_resource_file = godot_string_is_resource_file,
	.godot_string_path_to = godot_string_path_to,
	.godot_string_path_to_file = godot_string_path_to_file,
	.godot_string_simplify_path = godot_string_simplify_path,
	.godot_string_c_escape = godot_string_c_escape,
	.godot_string_c_escape_multiline = godot_string_c_escape_multiline,
	.godot_string_c_unescape = godot_string_c_unescape,
	.godot_string_http_escape = godot_string_http_escape,
	.godot_string_http_unescape = godot_string_http_unescape,
	.godot_string_json_escape = godot_string_json_escape,
	.godot_string_word_wrap = godot_string_word_wrap,
	.godot_string_xml_escape = godot_string_xml_escape,
	.godot_string_xml_escape_with_quotes = godot_string_xml_escape_with_quotes,
	.godot_string_xml_unescape = godot_string_xml_unescape,
	.godot_string_percent_decode = godot_string_percent_decode,
	.godot_string_percent_encode = godot_string_percent_encode,
	.godot_string_is_valid_float = godot_string_is_valid_float,
	.godot_string_is_valid_hex_number = godot_string_is_valid_hex_number,
	.godot_string_is_valid_html_color = godot_string_is_valid_html_color,
	.godot_string_is_valid_identifier = godot_string_is_valid_identifier,
	.godot_string_is_valid_integer = godot_string_is_valid_integer,
	.godot_string_is_valid_ip_address = godot_string_is_valid_ip_address,
	.godot_string_destroy = godot_string_destroy,
	.godot_object_destroy = godot_object_destroy,
	.godot_global_get_singleton = godot_global_get_singleton,
	.godot_get_stack_bottom = godot_get_stack_bottom,
	.godot_method_bind_get_method = godot_method_bind_get_method,
	.godot_method_bind_ptrcall = godot_method_bind_ptrcall,
	.godot_method_bind_call = godot_method_bind_call,
	.godot_get_class_constructor = godot_get_class_constructor,
	.godot_get_global_constants = godot_get_global_constants,
	.godot_alloc = godot_alloc,
	.godot_realloc = godot_realloc,
	.godot_free = godot_free,
	.godot_print_error = godot_print_error,
	.godot_print_warning = godot_print_warning,
	.godot_print = godot_print,
	.godot_nativescript_register_class = godot_nativescript_register_class,
	.godot_nativescript_register_tool_class = godot_nativescript_register_tool_class,
	.godot_nativescript_register_method = godot_nativescript_register_method,
	.godot_nativescript_register_property = godot_nativescript_register_property,
	.godot_nativescript_register_signal = godot_nativescript_register_signal,
	.godot_nativescript_get_userdata = godot_nativescript_get_userdata,
};

String GDNativeLibrary::platform_names[NUM_PLATFORMS + 1] = {
	"X11_32bit",
	"X11_64bit",
	"Windows_32bit",
	"Windows_64bit",
	"OSX",

	"Android",

	"iOS_32bit",
	"iOS_64bit",

	"WebAssembly",

	""
};
String GDNativeLibrary::platform_lib_ext[NUM_PLATFORMS + 1] = {
	"so",
	"so",
	"dll",
	"dll",
	"dylib",

	"so",

	"dylib",
	"dylib",

	"wasm",

	""
};

GDNativeLibrary::Platform GDNativeLibrary::current_platform =
#if defined(X11_ENABLED)
		(sizeof(void *) == 8 ? X11_64BIT : X11_32BIT);
#elif defined(WINDOWS_ENABLED)
		(sizeof(void *) == 8 ? WINDOWS_64BIT : WINDOWS_32BIT);
#elif defined(OSX_ENABLED)
		OSX;
#elif defined(IPHONE_ENABLED)
		(sizeof(void *) == 8 ? IOS_64BIT : IOS_32BIT);
#elif defined(ANDROID_ENABLED)
		ANDROID;
#elif defined(JAVASCRIPT_ENABLED)
		WASM;
#else
		NUM_PLATFORMS;
#endif

GDNativeLibrary::GDNativeLibrary()
	: library_paths(), singleton_gdnative(false) {
}

GDNativeLibrary::~GDNativeLibrary() {
}

void GDNativeLibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_library_path", "platform", "path"), &GDNativeLibrary::set_library_path);
	ClassDB::bind_method(D_METHOD("get_library_path", "platform"), &GDNativeLibrary::get_library_path);

	ClassDB::bind_method(D_METHOD("is_singleton_gdnative"), &GDNativeLibrary::is_singleton_gdnative);
	ClassDB::bind_method(D_METHOD("set_singleton_gdnative", "singleton"), &GDNativeLibrary::set_singleton_gdnative);

	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "singleton_gdnative"), "set_singleton_gdnative", "is_singleton_gdnative");
}

bool GDNativeLibrary::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name.begins_with("platform/")) {
		set_library_path(name.get_slice("/", 1), p_value);
		return true;
	}
	return false;
}

bool GDNativeLibrary::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name.begins_with("platform/")) {
		r_ret = get_library_path(name.get_slice("/", 1));
		return true;
	}
	return false;
}

void GDNativeLibrary::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < NUM_PLATFORMS; i++) {
		p_list->push_back(PropertyInfo(Variant::STRING,
				"platform/" + platform_names[i],
				PROPERTY_HINT_FILE,
				"*." + platform_lib_ext[i]));
	}
}

void GDNativeLibrary::set_library_path(StringName p_platform, String p_path) {
	int i;
	for (i = 0; i <= NUM_PLATFORMS; i++) {
		if (i == NUM_PLATFORMS) break;
		if (platform_names[i] == p_platform) {
			break;
		}
	}

	if (i == NUM_PLATFORMS) {
		ERR_EXPLAIN(String("No such platform: ") + p_platform);
		ERR_FAIL();
	}

	library_paths[i] = p_path;
}

String GDNativeLibrary::get_library_path(StringName p_platform) const {
	int i;
	for (i = 0; i <= NUM_PLATFORMS; i++) {
		if (i == NUM_PLATFORMS) break;
		if (platform_names[i] == p_platform) {
			break;
		}
	}

	if (i == NUM_PLATFORMS) {
		ERR_EXPLAIN(String("No such platform: ") + p_platform);
		ERR_FAIL_V("");
	}

	return library_paths[i];
}

String GDNativeLibrary::get_active_library_path() const {
	if (GDNativeLibrary::current_platform != NUM_PLATFORMS) {
		return library_paths[GDNativeLibrary::current_platform];
	}
	return "";
}

GDNative::GDNative() {
	native_handle = NULL;
}

GDNative::~GDNative() {
}

extern "C" void _api_anchor();

void GDNative::_compile_dummy_for_api() {
	_api_anchor();
}

void GDNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_library", "library"), &GDNative::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &GDNative::get_library);

	ClassDB::bind_method(D_METHOD("initialize"), &GDNative::initialize);
	ClassDB::bind_method(D_METHOD("terminate"), &GDNative::terminate);

	// TODO(karroffel): get_native_(raw_)call_types binding?

	// TODO(karroffel): make this a varargs function?
	ClassDB::bind_method(D_METHOD("call_native", "procedure_name", "arguments"), &GDNative::call_native);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "library", PROPERTY_HINT_RESOURCE_TYPE, "GDNativeLibrary"), "set_library", "get_library");
}

void GDNative::set_library(Ref<GDNativeLibrary> p_library) {
	ERR_EXPLAIN("Tried to change library of GDNative when it is already set");
	ERR_FAIL_COND(library.is_valid());
	library = p_library;
}

Ref<GDNativeLibrary> GDNative::get_library() {
	return library;
}

bool GDNative::initialize() {
	if (library.is_null()) {
		ERR_PRINT("No library set, can't initialize GDNative object");
		return false;
	}

	String lib_path = library->get_active_library_path();
	if (lib_path.empty()) {
		ERR_PRINT("No library set for this platform");
		return false;
	}

	String path = ProjectSettings::get_singleton()->globalize_path(lib_path);
	Error err = OS::get_singleton()->open_dynamic_library(path, native_handle);
	if (err != OK) {
		return false;
	}

	void *library_init;
	err = OS::get_singleton()->get_dynamic_library_symbol_handle(
			native_handle,
			init_symbol,
			library_init);

	if (err || !library_init) {
		OS::get_singleton()->close_dynamic_library(native_handle);
		native_handle = NULL;
		ERR_PRINT("Failed to obtain godot_gdnative_init symbol");
		return false;
	}

	godot_gdnative_init_fn library_init_fpointer;
	library_init_fpointer = (godot_gdnative_init_fn)library_init;

	godot_gdnative_init_options options;

	options.api_struct = &api_struct;
	options.in_editor = Engine::get_singleton()->is_editor_hint();
	options.core_api_hash = ClassDB::get_api_hash(ClassDB::API_CORE);
	options.editor_api_hash = ClassDB::get_api_hash(ClassDB::API_EDITOR);
	options.no_api_hash = ClassDB::get_api_hash(ClassDB::API_NONE);
	options.gd_native_library = (godot_object *)(get_library().ptr());

	library_init_fpointer(&options);

	return true;
}

bool GDNative::terminate() {

	if (native_handle == NULL) {
		ERR_PRINT("No valid library handle, can't terminate GDNative object");
		return false;
	}

	void *library_terminate;
	Error error = OS::get_singleton()->get_dynamic_library_symbol_handle(
			native_handle,
			terminate_symbol,
			library_terminate);
	if (error) {
		OS::get_singleton()->close_dynamic_library(native_handle);
		native_handle = NULL;
		return true;
	}

	godot_gdnative_terminate_fn library_terminate_pointer;
	library_terminate_pointer = (godot_gdnative_terminate_fn)library_terminate;

	// TODO(karroffel): remove this? Should be part of NativeScript, not
	// GDNative IMO
	godot_gdnative_terminate_options options;
	options.in_editor = Engine::get_singleton()->is_editor_hint();

	library_terminate_pointer(&options);

	// GDNativeScriptLanguage::get_singleton()->initialized_libraries.erase(p_native_lib->path);

	OS::get_singleton()->close_dynamic_library(native_handle);
	native_handle = NULL;

	return true;
}

bool GDNative::is_initialized() {
	return (native_handle != NULL);
}

void GDNativeCallRegistry::register_native_call_type(StringName p_call_type, native_call_cb p_callback) {
	native_calls.insert(p_call_type, p_callback);
}

void GDNativeCallRegistry::register_native_raw_call_type(StringName p_raw_call_type, native_raw_call_cb p_callback) {
	native_raw_calls.insert(p_raw_call_type, p_callback);
}

Vector<StringName> GDNativeCallRegistry::get_native_call_types() {
	Vector<StringName> call_types;
	call_types.resize(native_calls.size());

	size_t idx = 0;
	for (Map<StringName, native_call_cb>::Element *E = native_calls.front(); E; E = E->next(), idx++) {
		call_types[idx] = E->key();
	}

	return call_types;
}

Vector<StringName> GDNativeCallRegistry::get_native_raw_call_types() {
	Vector<StringName> call_types;
	call_types.resize(native_raw_calls.size());

	size_t idx = 0;
	for (Map<StringName, native_raw_call_cb>::Element *E = native_raw_calls.front(); E; E = E->next(), idx++) {
		call_types[idx] = E->key();
	}

	return call_types;
}

Variant GDNative::call_native(StringName p_native_call_type, StringName p_procedure_name, Array p_arguments) {

	Map<StringName, native_call_cb>::Element *E = GDNativeCallRegistry::singleton->native_calls.find(p_native_call_type);
	if (!E) {
		ERR_PRINT((String("No handler for native call type \"" + p_native_call_type) + "\" found").utf8().get_data());
		return Variant();
	}

	String procedure_name = p_procedure_name;
	godot_variant result = E->get()(native_handle, (godot_string *)&procedure_name, (godot_array *)&p_arguments);

	return *(Variant *)&result;
}

void GDNative::call_native_raw(StringName p_raw_call_type, StringName p_procedure_name, void *data, int num_args, void **args, void *r_return) {

	Map<StringName, native_raw_call_cb>::Element *E = GDNativeCallRegistry::singleton->native_raw_calls.find(p_raw_call_type);
	if (!E) {
		ERR_PRINT((String("No handler for native raw call type \"" + p_raw_call_type) + "\" found").utf8().get_data());
		return;
	}

	String procedure_name = p_procedure_name;
	E->get()(native_handle, (godot_string *)&procedure_name, data, num_args, args, r_return);
}
