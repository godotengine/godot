#include "godot_transform2d.h"

#include "../godot.h"

#include "math/math_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

void _transform2d_api_anchor() {
}

void GDAPI godot_transform2d_new_identity(godot_transform2d *p_t) {
	Transform2D *t = (Transform2D *)p_t;
	*t = Transform2D();
}

void GDAPI godot_transform2d_new_elements(godot_transform2d *p_t, const godot_vector2 *p_a, const godot_vector2 *p_b, const godot_vector2 *p_c) {
	Transform2D *t = (Transform2D *)p_t;
	Vector2 *a = (Vector2 *)p_a;
	Vector2 *b = (Vector2 *)p_b;
	Vector2 *c = (Vector2 *)p_c;
	*t = Transform2D(a->x, a->y, b->x, b->y, c->x, c->y);
}

void GDAPI godot_transform2d_new(godot_transform2d *p_t, const godot_real p_rot, const godot_vector2 *p_pos) {
	Transform2D *t = (Transform2D *)p_t;
	Vector2 *pos = (Vector2 *)p_pos;
	*t = Transform2D(p_rot, *pos);
}

godot_vector2 const GDAPI *godot_transform2d_const_index(const godot_transform2d *p_t, const godot_int p_idx) {
	const Transform2D *t = (const Transform2D *)p_t;
	const Vector2 *e = &t->operator[](p_idx);
	return (godot_vector2 const *)e;
}

godot_vector2 GDAPI *godot_transform2d_index(godot_transform2d *p_t, const godot_int p_idx) {
	Transform2D *t = (Transform2D *)p_t;
	Vector2 *e = &t->operator[](p_idx);
	return (godot_vector2 *)e;
}

godot_vector2 GDAPI godot_transform2d_get_axis(const godot_transform2d *p_t, const godot_int p_axis) {
	return *godot_transform2d_const_index(p_t, p_axis);
}

void GDAPI godot_transform2d_set_axis(godot_transform2d *p_t, const godot_int p_axis, const godot_vector2 *p_vec) {
	godot_vector2 *origin_v = godot_transform2d_index(p_t, p_axis);
	*origin_v = *p_vec;
}

// @Incomplete
// See header file

#ifdef __cplusplus
}
#endif
