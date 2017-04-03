#ifndef GODOT_DLSCRIPT_RECT2_H
#define GODOT_DLSCRIPT_RECT2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_RECT2_TYPE_DEFINED
typedef struct godot_rect2 {
	uint8_t _dont_touch_that[16];
} godot_rect2;
#endif

#include "../godot.h"

void GDAPI godot_rect2_new(godot_rect2 *p_rect);
void GDAPI godot_rect2_new_with_pos_and_size(godot_rect2 *p_rect, const godot_vector2 *p_pos, const godot_vector2 *p_size);

godot_vector2 GDAPI *godot_rect2_get_pos(godot_rect2 *p_rect);
void GDAPI godot_rect2_set_pos(godot_rect2 *p_rect, const godot_vector2 *p_pos);

godot_vector2 GDAPI *godot_rect2_get_size(godot_rect2 *p_rect);
void GDAPI godot_rect2_set_size(godot_rect2 *p_rect, const godot_vector2 *p_size);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_RECT3_H
