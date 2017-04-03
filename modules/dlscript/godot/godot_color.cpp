#include "godot_color.h"

#include "color.h"

#ifdef __cplusplus
extern "C" {
#endif

void _color_api_anchor() {
}

void GDAPI godot_color_new(godot_color *p_color) {
	Color *color = (Color *)p_color;
	*color = Color();
}

void GDAPI godot_color_new_rgba(godot_color *p_color, const godot_real r, const godot_real g, const godot_real b, const godot_real a) {
	Color *color = (Color *)p_color;
	*color = Color(r, g, b, a);
}

uint32_t GDAPI godot_color_get_32(const godot_color *p_color) {
	const Color *color = (const Color *)p_color;
	return color->to_32();
}

float GDAPI *godot_color_index(godot_color *p_color, const godot_int idx) {
	Color *color = (Color *)p_color;
	return &color->operator[](idx);
}

#ifdef __cplusplus
}
#endif
