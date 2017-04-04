#include "godot_variant.h"

#include "../godot.h"

#include "variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void _variant_api_anchor() {
}

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

godot_variant_type GDAPI godot_variant_get_type(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	return (godot_variant_type)v->get_type();
}

void GDAPI godot_variant_copy(godot_variant *p_dest, const godot_variant *p_src) {
	Variant *dest = (Variant *)p_dest;
	Variant *src = (Variant *)p_src;
	*dest = *src;
}

void GDAPI godot_variant_new_nil(godot_variant *p_v) {
	Variant *v = (Variant *)p_v;
	memnew_placement(v, Variant);
}

void GDAPI godot_variant_new_bool(godot_variant *p_v, const godot_bool p_b) {
	Variant *v = (Variant *)p_v;
	memnew_placement_custom(v, Variant, Variant(p_b));
}

void GDAPI godot_variant_new_uint(godot_variant *p_v, const uint64_t p_i) {
	Variant *v = (Variant *)p_v;
	memnew_placement_custom(v, Variant, Variant(p_i));
}

void GDAPI godot_variant_new_int(godot_variant *p_v, const int64_t p_i) {
	Variant *v = (Variant *)p_v;
	memnew_placement_custom(v, Variant, Variant(p_i));
}

void GDAPI godot_variant_new_real(godot_variant *p_v, const double p_r) {
	Variant *v = (Variant *)p_v;
	memnew_placement_custom(v, Variant, Variant(p_r));
}

void GDAPI godot_variant_new_string(godot_variant *p_v, const godot_string *p_s) {
	Variant *v = (Variant *)p_v;
	String *s = (String *)p_s;
	memnew_placement_custom(v, Variant, Variant(*s));
}

void GDAPI godot_variant_new_vector2(godot_variant *p_v, const godot_vector2 *p_v2) {
	Variant *v = (Variant *)p_v;
	Vector2 *v2 = (Vector2 *)p_v2;
	memnew_placement_custom(v, Variant, Variant(*v2));
}

void GDAPI godot_variant_new_rect2(godot_variant *p_v, const godot_rect2 *p_rect2) {
	Variant *v = (Variant *)p_v;
	Rect2 *rect2 = (Rect2 *)p_rect2;
	memnew_placement_custom(v, Variant, Variant(*rect2));
}

void GDAPI godot_variant_new_vector3(godot_variant *p_v, const godot_vector3 *p_v3) {
	Variant *v = (Variant *)p_v;
	Vector3 *v3 = (Vector3 *)p_v3;
	memnew_placement_custom(v, Variant, Variant(*v3));
}

void GDAPI godot_variant_new_transform2d(godot_variant *p_v, const godot_transform2d *p_t2d) {
	Variant *v = (Variant *)p_v;
	Transform2D *t2d = (Transform2D *)p_t2d;
	memnew_placement_custom(v, Variant, Variant(*t2d));
}

void GDAPI godot_variant_new_plane(godot_variant *p_v, const godot_plane *p_plane) {
	Variant *v = (Variant *)p_v;
	Plane *plane = (Plane *)p_plane;
	memnew_placement_custom(v, Variant, Variant(*plane));
}

void GDAPI godot_variant_new_quat(godot_variant *p_v, const godot_quat *p_quat) {
	Variant *v = (Variant *)p_v;
	Quat *quat = (Quat *)p_quat;
	memnew_placement_custom(v, Variant, Variant(*quat));
}

void GDAPI godot_variant_new_rect3(godot_variant *p_v, const godot_rect3 *p_rect3) {
	Variant *v = (Variant *)p_v;
	Rect3 *rect3 = (Rect3 *)p_rect3;
	memnew_placement_custom(v, Variant, Variant(*rect3));
}

void GDAPI godot_variant_new_basis(godot_variant *p_v, const godot_basis *p_basis) {
	Variant *v = (Variant *)p_v;
	Basis *basis = (Basis *)p_basis;
	memnew_placement_custom(v, Variant, Variant(*basis));
}

void GDAPI godot_variant_new_transform(godot_variant *p_v, const godot_transform *p_trans) {
	Variant *v = (Variant *)p_v;
	Transform *trans = (Transform *)p_trans;
	memnew_placement_custom(v, Variant, Variant(*trans));
}

void GDAPI godot_variant_new_color(godot_variant *p_v, const godot_color *p_color) {
	Variant *v = (Variant *)p_v;
	Color *color = (Color *)p_color;
	memnew_placement_custom(v, Variant, Variant(*color));
}

void GDAPI godot_variant_new_image(godot_variant *p_v, const godot_image *p_img) {
	Variant *v = (Variant *)p_v;
	Image *img = (Image *)p_img;
	memnew_placement_custom(v, Variant, Variant(*img));
}

void GDAPI godot_variant_new_node_path(godot_variant *p_v, const godot_node_path *p_np) {
	Variant *v = (Variant *)p_v;
	NodePath *np = (NodePath *)p_np;
	memnew_placement_custom(v, Variant, Variant(*np));
}

void GDAPI godot_variant_new_rid(godot_variant *p_v, const godot_rid *p_rid) {
	Variant *v = (Variant *)p_v;
	RID *rid = (RID *)p_rid;
	memnew_placement_custom(v, Variant, Variant(*rid));
}

void GDAPI godot_variant_new_object(godot_variant *p_v, const godot_object *p_obj) {
	Variant *v = (Variant *)p_v;
	Object *obj = (Object *)p_obj;
	memnew_placement_custom(v, Variant, Variant(obj));
}

void GDAPI godot_variant_new_input_event(godot_variant *p_v, const godot_input_event *p_event) {
	Variant *v = (Variant *)p_v;
	InputEvent *event = (InputEvent *)p_event;
	memnew_placement_custom(v, Variant, Variant(*event));
}

void GDAPI godot_variant_new_dictionary(godot_variant *p_v, const godot_dictionary *p_dict) {
	Variant *v = (Variant *)p_v;
	Dictionary *dict = (Dictionary *)p_dict;
	memnew_placement_custom(v, Variant, Variant(*dict));
}

void GDAPI godot_variant_new_array(godot_variant *p_v, const godot_array *p_arr) {
	Variant *v = (Variant *)p_v;
	Array *arr = (Array *)p_arr;
	memnew_placement_custom(v, Variant, Variant(*arr));
}

void GDAPI godot_variant_new_pool_byte_array(godot_variant *p_v, const godot_pool_byte_array *p_pba) {
	Variant *v = (Variant *)p_v;
	PoolByteArray *pba = (PoolByteArray *)p_pba;
	memnew_placement_custom(v, Variant, Variant(*pba));
}

void GDAPI godot_variant_new_pool_int_array(godot_variant *p_v, const godot_pool_int_array *p_pia) {
	Variant *v = (Variant *)p_v;
	PoolIntArray *pia = (PoolIntArray *)p_pia;
	memnew_placement_custom(v, Variant, Variant(*pia));
}

void GDAPI godot_variant_new_pool_real_array(godot_variant *p_v, const godot_pool_real_array *p_pra) {
	Variant *v = (Variant *)p_v;
	PoolRealArray *pra = (PoolRealArray *)p_pra;
	memnew_placement_custom(v, Variant, Variant(*pra));
}

void GDAPI godot_variant_new_pool_string_array(godot_variant *p_v, const godot_pool_string_array *p_psa) {
	Variant *v = (Variant *)p_v;
	PoolStringArray *psa = (PoolStringArray *)p_psa;
	memnew_placement_custom(v, Variant, Variant(*psa));
}

void GDAPI godot_variant_new_pool_vector2_array(godot_variant *p_v, const godot_pool_vector2_array *p_pv2a) {
	Variant *v = (Variant *)p_v;
	PoolVector2Array *pv2a = (PoolVector2Array *)p_pv2a;
	memnew_placement_custom(v, Variant, Variant(*pv2a));
}

void GDAPI godot_variant_new_pool_vector3_array(godot_variant *p_v, const godot_pool_vector3_array *p_pv3a) {
	Variant *v = (Variant *)p_v;
	PoolVector3Array *pv3a = (PoolVector3Array *)p_pv3a;
	memnew_placement_custom(v, Variant, Variant(*pv3a));
}

void GDAPI godot_variant_new_pool_color_array(godot_variant *p_v, const godot_pool_color_array *p_pca) {
	Variant *v = (Variant *)p_v;
	PoolColorArray *pca = (PoolColorArray *)p_pca;
	memnew_placement_custom(v, Variant, Variant(*pca));
}

godot_bool GDAPI godot_variant_as_bool(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	return v->operator bool();
}

uint64_t GDAPI godot_variant_as_uint(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	return v->operator uint64_t();
}

int64_t GDAPI godot_variant_as_int(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	return v->operator int64_t();
}

double GDAPI godot_variant_as_real(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	return v->operator double();
}

godot_string GDAPI godot_variant_as_string(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_string s;
	godot_string_new(&s);
	String *str = (String *)&s;
	*str = v->operator String();
	return s;
}

godot_vector2 GDAPI godot_variant_as_vector2(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_vector2 v2;
	Vector2 *vec2 = (Vector2 *)&v2;
	*vec2 = *v;
	return v2;
}

godot_rect2 GDAPI godot_variant_as_rect2(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_rect2 r2;
	Rect2 *rect2 = (Rect2 *)&r2;
	*rect2 = *v;
	return r2;
}

godot_vector3 GDAPI godot_variant_as_vector3(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_vector3 v3;
	Vector3 *vec3 = (Vector3 *)&v3;
	*vec3 = *v;
	return v3;
}

godot_transform2d GDAPI godot_variant_as_transform2d(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_transform2d t2;
	Transform2D *t = (Transform2D *)&t2;
	*t = *v;
	return t2;
}

godot_plane GDAPI godot_variant_as_plane(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_plane p;
	Plane *pl = (Plane *)&p;
	*pl = *v;
	return p;
}

godot_quat GDAPI godot_variant_as_quat(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_quat q;
	Quat *qt = (Quat *)&q;
	*qt = *v;
	return q;
}

godot_rect3 GDAPI godot_variant_as_rect3(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_rect3 r;
	Rect3 *r3 = (Rect3 *)&r;
	*r3 = *v;
	return r;
}

godot_basis GDAPI godot_variant_as_basis(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_basis b;
	Basis *bs = (Basis *)&b;
	*bs = *v;
	return b;
}

godot_transform GDAPI godot_variant_as_transform(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_transform t;
	Transform *tr = (Transform *)&t;
	*tr = *v;
	return t;
}

godot_color GDAPI godot_variant_as_color(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_color c;
	Color *col = (Color *)&c;
	*col = *v;
	return c;
}

godot_image GDAPI godot_variant_as_image(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_image img;
	godot_image_new(&img);
	Image *i = (Image *)&img;
	*i = *v;
	return img;
}

godot_node_path GDAPI godot_variant_as_node_path(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_node_path np;
	memnew_placement_custom((NodePath *)&np, NodePath, NodePath((String)*v));
	return np;
}

godot_rid GDAPI godot_variant_as_rid(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_rid rid;
	memnew_placement_custom((RID *)&rid, RID, RID(*v));
	return rid;
}

godot_object GDAPI *godot_variant_as_object(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_object *p = NULL;
	Object **op = (Object **)&p;
	*op = *v;
	return p;
}

godot_input_event GDAPI godot_variant_as_input_event(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_input_event ev;
	InputEvent *event = (InputEvent *)&ev;
	*event = *v;
	return ev;
}

godot_dictionary GDAPI godot_variant_as_dictionary(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_dictionary dict;
	godot_dictionary_new(&dict);
	Dictionary *d = (Dictionary *)&dict;
	*d = *v;
	return dict;
}

godot_array GDAPI godot_variant_as_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_array array;
	godot_array_new(&array);
	Array *a = (Array *)&array;
	*a = *v;
	return array;
}

godot_pool_byte_array GDAPI godot_variant_as_pool_byte_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_byte_array pba;
	godot_pool_byte_array_new(&pba);
	PoolByteArray *p = (PoolByteArray *)&pba;
	*p = *v;
	return pba;
}

godot_pool_int_array GDAPI godot_variant_as_pool_int_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_int_array pba;
	godot_pool_int_array_new(&pba);
	PoolIntArray *p = (PoolIntArray *)&pba;
	*p = *v;
	return pba;
}

godot_pool_real_array GDAPI godot_variant_as_pool_real_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_real_array pba;
	godot_pool_real_array_new(&pba);
	PoolRealArray *p = (PoolRealArray *)&pba;
	*p = *v;
	return pba;
}

godot_pool_string_array GDAPI godot_variant_as_pool_string_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_string_array pba;
	godot_pool_string_array_new(&pba);
	PoolStringArray *p = (PoolStringArray *)&pba;
	*p = *v;
	return pba;
}

godot_pool_vector2_array GDAPI godot_variant_as_pool_vector2_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_vector2_array pba;
	godot_pool_vector2_array_new(&pba);
	PoolVector2Array *p = (PoolVector2Array *)&pba;
	*p = *v;
	return pba;
}

godot_pool_vector3_array GDAPI godot_variant_as_pool_vector3_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_vector3_array pba;
	godot_pool_vector3_array_new(&pba);
	PoolVector3Array *p = (PoolVector3Array *)&pba;
	*p = *v;
	return pba;
}

godot_pool_color_array GDAPI godot_variant_as_pool_color_array(const godot_variant *p_v) {
	const Variant *v = (const Variant *)p_v;
	godot_pool_color_array pba;
	godot_pool_color_array_new(&pba);
	PoolColorArray *p = (PoolColorArray *)&pba;
	*p = *v;
	return pba;
}

godot_variant GDAPI godot_variant_call(godot_variant *p_v, const godot_string *p_method, const godot_variant **p_args, const godot_int p_argcount /*, godot_variant_call_error *r_error */) {
	Variant *v = (Variant *)p_v;
	String *method = (String *)p_method;
	Variant **args = (Variant **)p_args;
	godot_variant res;
	memnew_placement_custom((Variant *)&res, Variant, Variant(v->call(*method, args, p_argcount)));
	return res;
}

godot_bool GDAPI godot_variant_has_method(godot_variant *p_v, const godot_string *p_method) {
	Variant *v = (Variant *)p_v;
	String *method = (String *)p_method;
	return v->has_method(*method);
}

godot_bool GDAPI godot_variant_operator_equal(const godot_variant *p_a, const godot_variant *p_b) {
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	return a->operator==(*b);
}

godot_bool GDAPI godot_variant_operator_less(const godot_variant *p_a, const godot_variant *p_b) {
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	return a->operator<(*b);
}

godot_bool GDAPI godot_variant_hash_compare(const godot_variant *p_a, const godot_variant *p_b) {
	const Variant *a = (const Variant *)p_a;
	const Variant *b = (const Variant *)p_b;
	return a->hash_compare(*b);
}

godot_bool GDAPI godot_variant_booleanize(const godot_variant *p_v, godot_bool *p_valid) {
	const Variant *v = (const Variant *)p_v;
	bool &valid = *p_valid;
	return v->booleanize(valid);
}

void GDAPI godot_variant_destroy(godot_variant *p_v) {
	((Variant *)p_v)->~Variant();
}

#ifdef __cplusplus
}
#endif
