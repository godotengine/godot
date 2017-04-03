#ifndef GODOT_VECTOR3_H
#define GODOT_VECTOR3_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_VECTOR3_TYPE_DEFINED
typedef struct godot_vector3 {
	uint8_t _dont_touch_that[12];
} godot_vector3;
#endif

#include "../godot.h"

void GDAPI godot_vector3_new(godot_vector3 *p_v, const godot_real p_x, const godot_real p_y, const godot_real p_z);

void GDAPI godot_vector3_set_axis(godot_vector3 *p_v, const godot_int p_axis, const godot_real p_val);
godot_real GDAPI godot_vector3_get_axis(const godot_vector3 *p_v, const godot_int p_axis);

godot_int GDAPI godot_vector3_min_axis(const godot_vector3 *p_v);
godot_int GDAPI godot_vector3_max_axis(const godot_vector3 *p_v);

godot_real GDAPI godot_vector3_length(const godot_vector3 *p_v);
godot_real GDAPI godot_vector3_length_squared(const godot_vector3 *p_v);

void GDAPI godot_vector3_normalize(godot_vector3 *p_v);
void GDAPI godot_vector3_normalized(godot_vector3 *p_dest, const godot_vector3 *p_src);

// @Incomplete

/*
 * inverse
 * zero
 * snap
 * snapped
 * rotate
 * rotated
 *
 *
 * linear_interpolate
 * cubic_interpolate
 * cubic_interpolaten
 * cross
 * dot
 * outer
 * to_diagonal_matrix
 * abs
 * floor
 * ceil
 */

godot_real GDAPI godot_vector3_distance_to(const godot_vector3 *p_a, const godot_vector3 *p_b);
godot_real GDAPI godot_vector3_distance_squared_to(const godot_vector3 *p_a, const godot_vector3 *p_b);

// @Incomplete
/*
 * slide
 * reflect
 */

void GDAPI godot_vector3_operator_add(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_subtract(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_multiply_vector(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_multiply_scalar(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_real p_b);
void GDAPI godot_vector3_operator_divide_vector(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_divide_scalar(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_real p_b);

godot_bool GDAPI godot_vector3_operator_equal(const godot_vector3 *p_a, const godot_vector3 *p_b);
godot_bool GDAPI godot_vector3_operator_less(const godot_vector3 *p_a, const godot_vector3 *p_b);

/*
 * to_string
 */

#ifdef __cplusplus
}
#endif

#endif // GODOT_VECTOR3_H
