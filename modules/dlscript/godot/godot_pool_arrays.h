#ifndef GODOT_DLSCRIPT_POOL_ARRAYS_H
#define GODOT_DLSCRIPT_POOL_ARRAYS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/////// PoolByteArray

#ifndef GODOT_CORE_API_GODOT_POOL_BYTE_ARRAY_TYPE_DEFINED
typedef struct godot_pool_byte_array {
	uint8_t _dont_touch_that[8];
} godot_pool_byte_array;
#endif

/////// PoolIntArray

#ifndef GODOT_CORE_API_GODOT_POOL_INT_ARRAY_TYPE_DEFINED
typedef struct godot_pool_int_array {
	uint8_t _dont_touch_that[8];
} godot_pool_int_array;
#endif

/////// PoolRealArray

#ifndef GODOT_CORE_API_GODOT_POOL_REAL_ARRAY_TYPE_DEFINED
typedef struct godot_pool_real_array {
	uint8_t _dont_touch_that[8];
} godot_pool_real_array;
#endif

/////// PoolStringArray

#ifndef GODOT_CORE_API_GODOT_POOL_STRING_ARRAY_TYPE_DEFINED
typedef struct godot_pool_string_array {
	uint8_t _dont_touch_that[8];
} godot_pool_string_array;
#endif

/////// PoolVector2Array

#ifndef GODOT_CORE_API_GODOT_POOL_VECTOR2_ARRAY_TYPE_DEFINED
typedef struct godot_pool_vector2_array {
	uint8_t _dont_touch_that[8];
} godot_pool_vector2_array;
#endif

/////// PoolVector3Array

#ifndef GODOT_CORE_API_GODOT_POOL_VECTOR3_ARRAY_TYPE_DEFINED
typedef struct godot_pool_vector3_array {
	uint8_t _dont_touch_that[8];
} godot_pool_vector3_array;
#endif

/////// PoolColorArray

#ifndef GODOT_CORE_API_GODOT_POOL_COLOR_ARRAY_TYPE_DEFINED
typedef struct godot_pool_color_array {
	uint8_t _dont_touch_that[8];
} godot_pool_color_array;
#endif

#include "../godot.h"

#include "godot_array.h"

// byte

void GDAPI godot_pool_byte_array_new(godot_pool_byte_array *p_pba);
void GDAPI godot_pool_byte_array_new_with_array(godot_pool_byte_array *p_pba, const godot_array *p_a);

void GDAPI godot_pool_byte_array_append(godot_pool_byte_array *p_pba, const uint8_t p_data);

void GDAPI godot_pool_byte_array_append_array(godot_pool_byte_array *p_pba, const godot_pool_byte_array *p_array);

int GDAPI godot_pool_byte_array_insert(godot_pool_byte_array *p_pba, const godot_int p_idx, const uint8_t p_data);

void GDAPI godot_pool_byte_array_invert(godot_pool_byte_array *p_pba);

void GDAPI godot_pool_byte_array_push_back(godot_pool_byte_array *p_pba, const uint8_t p_data);

void GDAPI godot_pool_byte_array_remove(godot_pool_byte_array *p_pba, const godot_int p_idx);

void GDAPI godot_pool_byte_array_resize(godot_pool_byte_array *p_pba, const godot_int p_size);

void GDAPI godot_pool_byte_array_set(godot_pool_byte_array *p_pba, const godot_int p_idx, const uint8_t p_data);
uint8_t GDAPI godot_pool_byte_array_get(godot_pool_byte_array *p_pba, const godot_int p_idx);

godot_int GDAPI godot_pool_byte_array_size(godot_pool_byte_array *p_pba);

void GDAPI godot_pool_byte_array_destroy(godot_pool_byte_array *p_pba);

// int

void GDAPI godot_pool_int_array_new(godot_pool_int_array *p_pia);
void GDAPI godot_pool_int_array_new_with_array(godot_pool_int_array *p_pia, const godot_array *p_a);

void GDAPI godot_pool_int_array_append(godot_pool_int_array *p_pia, const godot_int p_data);

void GDAPI godot_pool_int_array_append_array(godot_pool_int_array *p_pia, const godot_pool_int_array *p_array);

int GDAPI godot_pool_int_array_insert(godot_pool_int_array *p_pia, const godot_int p_idx, const godot_int p_data);

void GDAPI godot_pool_int_array_invert(godot_pool_int_array *p_pia);

void GDAPI godot_pool_int_array_push_back(godot_pool_int_array *p_pia, const godot_int p_data);

void GDAPI godot_pool_int_array_remove(godot_pool_int_array *p_pia, const godot_int p_idx);

void GDAPI godot_pool_int_array_resize(godot_pool_int_array *p_pia, const godot_int p_size);

void GDAPI godot_pool_int_array_set(godot_pool_int_array *p_pia, const godot_int p_idx, const godot_int p_data);
godot_int GDAPI godot_pool_int_array_get(godot_pool_int_array *p_pia, const godot_int p_idx);

godot_int GDAPI godot_pool_int_array_size(godot_pool_int_array *p_pia);

void GDAPI godot_pool_int_array_destroy(godot_pool_int_array *p_pia);

// real

void GDAPI godot_pool_real_array_new(godot_pool_real_array *p_pra);
void GDAPI godot_pool_real_array_new_with_array(godot_pool_real_array *p_pra, const godot_array *p_a);

void GDAPI godot_pool_real_array_append(godot_pool_real_array *p_pra, const godot_real p_data);

void GDAPI godot_pool_real_array_append_array(godot_pool_real_array *p_pra, const godot_pool_real_array *p_array);

int GDAPI godot_pool_real_array_insert(godot_pool_real_array *p_pra, const godot_int p_idx, const godot_real p_data);

void GDAPI godot_pool_real_array_invert(godot_pool_real_array *p_pra);

void GDAPI godot_pool_real_array_push_back(godot_pool_real_array *p_pra, const godot_real p_data);

void GDAPI godot_pool_real_array_remove(godot_pool_real_array *p_pra, const godot_int p_idx);

void GDAPI godot_pool_real_array_resize(godot_pool_real_array *p_pra, const godot_int p_size);

void GDAPI godot_pool_real_array_set(godot_pool_real_array *p_pra, const godot_int p_idx, const godot_real p_data);
godot_real GDAPI godot_pool_real_array_get(godot_pool_real_array *p_pra, const godot_int p_idx);

godot_int GDAPI godot_pool_real_array_size(godot_pool_real_array *p_pra);

void GDAPI godot_pool_real_array_destroy(godot_pool_real_array *p_pra);

// string

void GDAPI godot_pool_string_array_new(godot_pool_string_array *p_psa);
void GDAPI godot_pool_string_array_new_with_array(godot_pool_string_array *p_psa, const godot_array *p_a);

void GDAPI godot_pool_string_array_append(godot_pool_string_array *p_psa, const godot_string *p_data);

void GDAPI godot_pool_string_array_append_array(godot_pool_string_array *p_psa, const godot_pool_string_array *p_array);

int GDAPI godot_pool_string_array_insert(godot_pool_string_array *p_psa, const godot_int p_idx, const godot_string *p_data);

void GDAPI godot_pool_string_array_invert(godot_pool_string_array *p_psa);

void GDAPI godot_pool_string_array_push_back(godot_pool_string_array *p_psa, const godot_string *p_data);

void GDAPI godot_pool_string_array_remove(godot_pool_string_array *p_psa, const godot_int p_idx);

void GDAPI godot_pool_string_array_resize(godot_pool_string_array *p_psa, const godot_int p_size);

void GDAPI godot_pool_string_array_set(godot_pool_string_array *p_psa, const godot_int p_idx, const godot_string *p_data);
godot_string GDAPI godot_pool_string_array_get(godot_pool_string_array *p_psa, const godot_int p_idx);

godot_int GDAPI godot_pool_string_array_size(godot_pool_string_array *p_psa);

void GDAPI godot_pool_string_array_destroy(godot_pool_string_array *p_psa);

// vector2

void GDAPI godot_pool_vector2_array_new(godot_pool_vector2_array *p_pv2a);
void GDAPI godot_pool_vector2_array_new_with_array(godot_pool_vector2_array *p_pv2a, const godot_array *p_a);

void GDAPI godot_pool_vector2_array_append(godot_pool_vector2_array *p_pv2a, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_append_array(godot_pool_vector2_array *p_pv2a, const godot_pool_vector2_array *p_array);

int GDAPI godot_pool_vector2_array_insert(godot_pool_vector2_array *p_pv2a, const godot_int p_idx, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_invert(godot_pool_vector2_array *p_pv2a);

void GDAPI godot_pool_vector2_array_push_back(godot_pool_vector2_array *p_pv2a, const godot_vector2 *p_data);

void GDAPI godot_pool_vector2_array_remove(godot_pool_vector2_array *p_pv2a, const godot_int p_idx);

void GDAPI godot_pool_vector2_array_resize(godot_pool_vector2_array *p_pv2a, const godot_int p_size);

void GDAPI godot_pool_vector2_array_set(godot_pool_vector2_array *p_pv2a, const godot_int p_idx, const godot_vector2 *p_data);
godot_vector2 GDAPI godot_pool_vector2_array_get(godot_pool_vector2_array *p_pv2a, const godot_int p_idx);

godot_int GDAPI godot_pool_vector2_array_size(godot_pool_vector2_array *p_pv2a);

void GDAPI godot_pool_vector2_array_destroy(godot_pool_vector2_array *p_pv2a);

// vector3

void GDAPI godot_pool_vector3_array_new(godot_pool_vector3_array *p_pv3a);
void GDAPI godot_pool_vector3_array_new_with_array(godot_pool_vector3_array *p_pv3a, const godot_array *p_a);

void GDAPI godot_pool_vector3_array_append(godot_pool_vector3_array *p_pv3a, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_append_array(godot_pool_vector3_array *p_pv3a, const godot_pool_vector3_array *p_array);

int GDAPI godot_pool_vector3_array_insert(godot_pool_vector3_array *p_pv3a, const godot_int p_idx, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_invert(godot_pool_vector3_array *p_pv3a);

void GDAPI godot_pool_vector3_array_push_back(godot_pool_vector3_array *p_pv3a, const godot_vector3 *p_data);

void GDAPI godot_pool_vector3_array_remove(godot_pool_vector3_array *p_pv3a, const godot_int p_idx);

void GDAPI godot_pool_vector3_array_resize(godot_pool_vector3_array *p_pv3a, const godot_int p_size);

void GDAPI godot_pool_vector3_array_set(godot_pool_vector3_array *p_pv3a, const godot_int p_idx, const godot_vector3 *p_data);
godot_vector3 GDAPI godot_pool_vector3_array_get(godot_pool_vector3_array *p_pv3a, const godot_int p_idx);

godot_int GDAPI godot_pool_vector3_array_size(godot_pool_vector3_array *p_pv3a);

void GDAPI godot_pool_vector3_array_destroy(godot_pool_vector3_array *p_pv3a);

// color

void GDAPI godot_pool_color_array_new(godot_pool_color_array *p_pca);
void GDAPI godot_pool_color_array_new_with_array(godot_pool_color_array *p_pca, const godot_array *p_a);

void GDAPI godot_pool_color_array_append(godot_pool_color_array *p_pca, const godot_color *p_data);

void GDAPI godot_pool_color_array_append_array(godot_pool_color_array *p_pca, const godot_pool_color_array *p_array);

int GDAPI godot_pool_color_array_insert(godot_pool_color_array *p_pca, const godot_int p_idx, const godot_color *p_data);

void GDAPI godot_pool_color_array_invert(godot_pool_color_array *p_pca);

void GDAPI godot_pool_color_array_push_back(godot_pool_color_array *p_pca, const godot_color *p_data);

void GDAPI godot_pool_color_array_remove(godot_pool_color_array *p_pca, const godot_int p_idx);

void GDAPI godot_pool_color_array_resize(godot_pool_color_array *p_pca, const godot_int p_size);

void GDAPI godot_pool_color_array_set(godot_pool_color_array *p_pca, const godot_int p_idx, const godot_color *p_data);
godot_color GDAPI godot_pool_color_array_get(godot_pool_color_array *p_pca, const godot_int p_idx);

godot_int GDAPI godot_pool_color_array_size(godot_pool_color_array *p_pca);

void GDAPI godot_pool_color_array_destroy(godot_pool_color_array *p_pca);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_POOL_ARRAYS_H
