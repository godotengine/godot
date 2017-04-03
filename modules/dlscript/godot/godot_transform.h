#ifndef GODOT_DLSCRIPT_TRANSFORM_H
#define GODOT_DLSCRIPT_TRANSFORM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_TRANSFORM_TYPE_DEFINED
typedef struct godot_transform {
	uint8_t _dont_touch_that[48];
} godot_transform;
#endif

#include "../godot.h"

void GDAPI godot_transform_new(godot_transform *p_trans);
void GDAPI godot_transform_new_with_basis(godot_transform *p_trans, const godot_basis *p_basis);
void GDAPI godot_transform_new_with_basis_origin(godot_transform *p_trans, const godot_basis *p_basis, const godot_vector3 *p_origin);

godot_basis GDAPI *godot_transform_get_basis(godot_transform *p_trans);
godot_vector3 GDAPI *godot_transform_get_origin(godot_transform *p_trans);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_TRANSFORM_H
