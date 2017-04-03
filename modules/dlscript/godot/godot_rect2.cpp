#include "godot_rect2.h"

#include "math/math_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rect2_api_anchor() {
}

void GDAPI godot_rect2_new(godot_rect2 *p_rect) {
	Rect2 *rect = (Rect2 *)p_rect;
	*rect = Rect2();
}

void GDAPI godot_rect2_new_with_pos_and_size(godot_rect2 *p_rect, const godot_vector2 *p_pos, const godot_vector2 *p_size) {
	Rect2 *rect = (Rect2 *)p_rect;
	const Vector2 *pos = (const Vector2 *)p_pos;
	const Vector2 *size = (const Vector2 *)p_size;
	*rect = Rect2(*pos, *size);
}

godot_vector2 GDAPI *godot_rect2_get_pos(godot_rect2 *p_rect) {
	Rect2 *rect = (Rect2 *)p_rect;
	return (godot_vector2 *)&rect->pos;
}

void GDAPI godot_rect2_set_pos(godot_rect2 *p_rect, const godot_vector2 *p_pos) {
	Rect2 *rect = (Rect2 *)p_rect;
	const Vector2 *pos = (const Vector2 *)p_pos;
	rect->pos = *pos;
}

godot_vector2 GDAPI *godot_rect2_get_size(godot_rect2 *p_rect) {
	Rect2 *rect = (Rect2 *)p_rect;
	return (godot_vector2 *)&rect->size;
}

void GDAPI godot_rect2_set_size(godot_rect2 *p_rect, const godot_vector2 *p_size) {
	Rect2 *rect = (Rect2 *)p_rect;
	const Vector2 *size = (const Vector2 *)p_size;
	rect->size = *size;
}

#ifdef __cplusplus
}
#endif
