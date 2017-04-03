#ifndef GODOT_DLSCRIPT_QUAT_H
#define GODOT_DLSCRIPT_QUAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_QUAT_TYPE_DEFINED
typedef struct godot_quat {
	uint8_t _dont_touch_that[16];
} godot_quat;
#endif

#include "../godot.h"

void GDAPI godot_quat_new(godot_quat *p_quat);
void GDAPI godot_quat_new_with_elements(godot_quat *p_quat, const godot_real x, const godot_real y, const godot_real z, const godot_real w);
void GDAPI godot_quat_new_with_rotation(godot_quat *p_quat, const godot_vector3 *p_axis, const godot_real p_angle);
void GDAPI godot_quat_new_with_shortest_arc(godot_quat *p_quat, const godot_vector3 *p_v0, const godot_vector3 *p_v1);

godot_vector3 GDAPI godot_quat_get_euler(const godot_quat *p_quat);
void GDAPI godot_quat_set_euler(godot_quat *p_quat, const godot_vector3 *p_euler);

godot_real GDAPI *godot_quat_index(godot_quat *p_quat, const godot_int p_idx);
godot_real GDAPI godot_quat_const_index(const godot_quat *p_quat, const godot_int p_idx);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_QUAT_H
