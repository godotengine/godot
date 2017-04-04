#ifndef GODOT_VARIANT_H
#define GODOT_VARIANT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_VARIANT_TYPE_DEFINED
typedef struct godot_variant {
	uint8_t _dont_touch_that[24];
} godot_variant;
#endif

struct godot_transform2d;
typedef struct godot_transform2d godot_transform2d;

#include "godot_array.h"
#include "godot_dictionary.h"
#include "godot_input_event.h"
#include "godot_node_path.h"
#include "godot_rid.h"
#include "godot_transform2d.h"

typedef enum godot_variant_type {
	GODOT_VARIANT_TYPE_NIL,

	// atomic types
	GODOT_VARIANT_TYPE_BOOL,
	GODOT_VARIANT_TYPE_INT,
	GODOT_VARIANT_TYPE_REAL,
	GODOT_VARIANT_TYPE_STRING,

	// math types

	GODOT_VARIANT_TYPE_VECTOR2, // 5
	GODOT_VARIANT_TYPE_RECT2,
	GODOT_VARIANT_TYPE_VECTOR3,
	GODOT_VARIANT_TYPE_TRANSFORM2D,
	GODOT_VARIANT_TYPE_PLANE,
	GODOT_VARIANT_TYPE_QUAT, // 10
	GODOT_VARIANT_TYPE_RECT3, //sorry naming convention fail :( not like it's used often
	GODOT_VARIANT_TYPE_BASIS,
	GODOT_VARIANT_TYPE_TRANSFORM,

	// misc types
	GODOT_VARIANT_TYPE_COLOR,
	GODOT_VARIANT_TYPE_IMAGE, // 15
	GODOT_VARIANT_TYPE_NODE_PATH,
	GODOT_VARIANT_TYPE_RID,
	GODOT_VARIANT_TYPE_OBJECT,
	GODOT_VARIANT_TYPE_INPUT_EVENT,
	GODOT_VARIANT_TYPE_DICTIONARY, // 20
	GODOT_VARIANT_TYPE_ARRAY,

	// arrays
	GODOT_VARIANT_TYPE_POOL_BYTE_ARRAY,
	GODOT_VARIANT_TYPE_POOL_INT_ARRAY,
	GODOT_VARIANT_TYPE_POOL_REAL_ARRAY,
	GODOT_VARIANT_TYPE_POOL_STRING_ARRAY, // 25
	GODOT_VARIANT_TYPE_POOL_VECTOR2_ARRAY,
	GODOT_VARIANT_TYPE_POOL_VECTOR3_ARRAY,
	GODOT_VARIANT_TYPE_POOL_COLOR_ARRAY,
} godot_variant_type;

godot_variant_type GDAPI godot_variant_get_type(const godot_variant *p_v);

void GDAPI godot_variant_copy(godot_variant *p_dest, const godot_variant *p_src);

void GDAPI godot_variant_new_nil(godot_variant *p_v);

void GDAPI godot_variant_new_bool(godot_variant *p_v, const godot_bool p_b);
void GDAPI godot_variant_new_uint(godot_variant *p_v, const uint64_t p_i);
void GDAPI godot_variant_new_int(godot_variant *p_v, const int64_t p_i);
void GDAPI godot_variant_new_real(godot_variant *p_v, const double p_r);
void GDAPI godot_variant_new_string(godot_variant *p_v, const godot_string *p_s);
void GDAPI godot_variant_new_vector2(godot_variant *p_v, const godot_vector2 *p_v2);
void GDAPI godot_variant_new_rect2(godot_variant *p_v, const godot_rect2 *p_rect2);
void GDAPI godot_variant_new_vector3(godot_variant *p_v, const godot_vector3 *p_v3);
void GDAPI godot_variant_new_transform2d(godot_variant *p_v, const godot_transform2d *p_t2d);
void GDAPI godot_variant_new_plane(godot_variant *p_v, const godot_plane *p_plane);
void GDAPI godot_variant_new_quat(godot_variant *p_v, const godot_quat *p_quat);
void GDAPI godot_variant_new_rect3(godot_variant *p_v, const godot_rect3 *p_rect3);
void GDAPI godot_variant_new_basis(godot_variant *p_v, const godot_basis *p_basis);
void GDAPI godot_variant_new_transform(godot_variant *p_v, const godot_transform *p_trans);
void GDAPI godot_variant_new_color(godot_variant *p_v, const godot_color *p_color);
void GDAPI godot_variant_new_image(godot_variant *p_v, const godot_image *p_img);
void GDAPI godot_variant_new_node_path(godot_variant *p_v, const godot_node_path *p_np);
void GDAPI godot_variant_new_rid(godot_variant *p_v, const godot_rid *p_rid);
void GDAPI godot_variant_new_object(godot_variant *p_v, const godot_object *p_obj);
void GDAPI godot_variant_new_input_event(godot_variant *p_v, const godot_input_event *p_event);
void GDAPI godot_variant_new_dictionary(godot_variant *p_v, const godot_dictionary *p_dict);
void GDAPI godot_variant_new_array(godot_variant *p_v, const godot_array *p_arr);
void GDAPI godot_variant_new_pool_byte_array(godot_variant *p_v, const godot_pool_byte_array *p_pba);
void GDAPI godot_variant_new_pool_int_array(godot_variant *p_v, const godot_pool_int_array *p_pia);
void GDAPI godot_variant_new_pool_real_array(godot_variant *p_v, const godot_pool_real_array *p_pra);
void GDAPI godot_variant_new_pool_string_array(godot_variant *p_v, const godot_pool_string_array *p_psa);
void GDAPI godot_variant_new_pool_vector2_array(godot_variant *p_v, const godot_pool_vector2_array *p_pv2a);
void GDAPI godot_variant_new_pool_vector3_array(godot_variant *p_v, const godot_pool_vector3_array *p_pv3a);
void GDAPI godot_variant_new_pool_color_array(godot_variant *p_v, const godot_pool_color_array *p_pca);

godot_bool GDAPI godot_variant_as_bool(const godot_variant *p_v);
uint64_t GDAPI godot_variant_as_uint(const godot_variant *p_v);
int64_t GDAPI godot_variant_as_int(const godot_variant *p_v);
double GDAPI godot_variant_as_real(const godot_variant *p_v);
godot_string GDAPI godot_variant_as_string(const godot_variant *p_v);
godot_vector2 GDAPI godot_variant_as_vector2(const godot_variant *p_v);
godot_rect2 GDAPI godot_variant_as_rect2(const godot_variant *p_v);
godot_vector3 GDAPI godot_variant_as_vector3(const godot_variant *p_v);
godot_transform2d GDAPI godot_variant_as_transform2d(const godot_variant *p_v);
godot_plane GDAPI godot_variant_as_plane(const godot_variant *p_v);
godot_quat GDAPI godot_variant_as_quat(const godot_variant *p_v);
godot_rect3 GDAPI godot_variant_as_rect3(const godot_variant *p_v);
godot_basis GDAPI godot_variant_as_basis(const godot_variant *p_v);
godot_transform GDAPI godot_variant_as_transform(const godot_variant *p_v);
godot_color GDAPI godot_variant_as_color(const godot_variant *p_v);
godot_image GDAPI godot_variant_as_image(const godot_variant *p_v);
godot_node_path GDAPI godot_variant_as_node_path(const godot_variant *p_v);
godot_rid GDAPI godot_variant_as_rid(const godot_variant *p_v);
godot_object GDAPI *godot_variant_as_object(const godot_variant *p_v);
godot_input_event GDAPI godot_variant_as_input_event(const godot_variant *p_v);
godot_dictionary GDAPI godot_variant_as_dictionary(const godot_variant *p_v);
godot_array GDAPI godot_variant_as_array(const godot_variant *p_v);
godot_pool_byte_array GDAPI godot_variant_as_pool_byte_array(const godot_variant *p_v);
godot_pool_int_array GDAPI godot_variant_as_pool_int_array(const godot_variant *p_v);
godot_pool_real_array GDAPI godot_variant_as_pool_real_array(const godot_variant *p_v);
godot_pool_string_array GDAPI godot_variant_as_pool_string_array(const godot_variant *p_v);
godot_pool_vector2_array GDAPI godot_variant_as_pool_vector2_array(const godot_variant *p_v);
godot_pool_vector3_array GDAPI godot_variant_as_pool_vector3_array(const godot_variant *p_v);
godot_pool_color_array GDAPI godot_variant_as_pool_color_array(const godot_variant *p_v);

godot_variant GDAPI godot_variant_call(godot_variant *p_v, const godot_string *p_method, const godot_variant **p_args, const godot_int p_argcount /*, godot_variant_call_error *r_error */);

godot_bool GDAPI godot_variant_has_method(godot_variant *p_v, const godot_string *p_method);

godot_bool GDAPI godot_variant_operator_equal(const godot_variant *p_a, const godot_variant *p_b);
godot_bool GDAPI godot_variant_operator_less(const godot_variant *p_a, const godot_variant *p_b);

godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_a, const godot_variant *p_b);

godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_v, godot_bool *p_valid);

void GDAPI godot_variant_destroy(godot_variant *p_v);

#ifdef __cplusplus
}
#endif

#endif
