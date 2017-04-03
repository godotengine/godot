#ifndef GODOT_DLSCRIPT_COLOR_H
#define GODOT_DLSCRIPT_COLOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_COLOR_TYPE_DEFINED
typedef struct godot_color {
	uint8_t _dont_touch_that[16];
} godot_color;
#endif

#include "../godot.h"

void GDAPI godot_color_new(godot_color *p_color);
void GDAPI godot_color_new_rgba(godot_color *p_color, const godot_real r, const godot_real g, const godot_real b, const godot_real a);

uint32_t GDAPI godot_color_get_32(const godot_color *p_color);

float GDAPI *godot_color_index(godot_color *p_color, const godot_int idx);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_COLOR_H
