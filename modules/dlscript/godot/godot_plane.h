#ifndef GODOT_DLSCRIPT_PLANE_H
#define GODOT_DLSCRIPT_PLANE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_PLANE_TYPE_DEFINED
typedef struct godot_plane {
	uint8_t _dont_touch_that[16];
} godot_plane;
#endif

#include "godot_vector3.h"

void GDAPI godot_plane_new(godot_plane *p_pl);
void GDAPI godot_plane_new_with_normal(godot_plane *p_pl, const godot_vector3 *p_normal, const godot_real p_d);

// @Incomplete
// These are additional valid constructors
// _FORCE_INLINE_ Plane(const Vector3 &p_normal, real_t p_d);
// _FORCE_INLINE_ Plane(const Vector3 &p_point, const Vector3& p_normal);
// _FORCE_INLINE_ Plane(const Vector3 &p_point1, const Vector3 &p_point2,const Vector3 &p_point3,ClockDirection p_dir = CLOCKWISE);

void GDAPI godot_plane_set_normal(godot_plane *p_pl, const godot_vector3 *p_normal);
godot_vector3 GDAPI godot_plane_get_normal(const godot_plane *p_pl);

godot_real GDAPI godot_plane_get_d(const godot_plane *p_pl);
void GDAPI godot_plane_set_d(godot_plane *p_pl, const godot_real p_d);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_PLANE_H
