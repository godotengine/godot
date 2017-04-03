#include "godot_rect3.h"

#include "math/rect3.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rect3_api_anchor() {
}

void GDAPI godot_rect3_new(godot_rect3 *p_rect) {
	Rect3 *rect = (Rect3 *)p_rect;
	*rect = Rect3();
}

void GDAPI godot_rect3_new_with_pos_and_size(godot_rect3 *p_rect, const godot_vector3 *p_pos, const godot_vector3 *p_size) {
	Rect3 *rect = (Rect3 *)p_rect;
	const Vector3 *pos = (const Vector3 *)p_pos;
	const Vector3 *size = (const Vector3 *)p_size;
	*rect = Rect3(*pos, *size);
}

godot_vector3 GDAPI *godot_rect3_get_pos(godot_rect3 *p_rect) {
	Rect3 *rect = (Rect3 *)p_rect;
	return (godot_vector3 *)&rect->pos;
}

void GDAPI godot_rect3_set_pos(godot_rect3 *p_rect, const godot_vector3 *p_pos) {
	Rect3 *rect = (Rect3 *)p_rect;
	const Vector3 *pos = (const Vector3 *)p_pos;
	rect->pos = *pos;
}

godot_vector3 GDAPI *godot_rect3_get_size(godot_rect3 *p_rect) {
	Rect3 *rect = (Rect3 *)p_rect;
	return (godot_vector3 *)&rect->size;
}

void GDAPI godot_rect3_set_size(godot_rect3 *p_rect, const godot_vector3 *p_size) {
	Rect3 *rect = (Rect3 *)p_rect;
	const Vector3 *size = (const Vector3 *)p_size;
	rect->size = *size;
}

#ifdef __cplusplus
}
#endif
