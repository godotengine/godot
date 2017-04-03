#ifndef GODOT_DLSCRIPT_RID_H
#define GODOT_DLSCRIPT_RID_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_RID_TYPE_DEFINED
typedef struct godot_rid {
	uint8_t _dont_touch_that[8];
} godot_rid;
#endif

#include "../godot.h"

void GDAPI godot_rid_new(godot_rid *p_rid, godot_object *p_from);

uint32_t GDAPI godot_rid_get_rid(const godot_rid *p_rid);

void GDAPI godot_rid_destroy(godot_rid *p_rid);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_RID_H
