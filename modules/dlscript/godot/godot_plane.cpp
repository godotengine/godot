#include "godot_plane.h"

#include "math/plane.h"

#ifdef __cplusplus
extern "C" {
#endif

void _plane_api_anchor() {
}

void GDAPI godot_plane_new(godot_plane *p_pl) {
	Plane *pl = (Plane *)p_pl;
	*pl = Plane();
}

void GDAPI godot_plane_new_with_normal(godot_plane *p_pl, const godot_vector3 *p_normal, const godot_real p_d) {
	Plane *pl = (Plane *)p_pl;
	const Vector3 *normal = (const Vector3 *)p_normal;
	*pl = Plane(*normal, p_d);
}

void GDAPI godot_plane_set_normal(godot_plane *p_pl, const godot_vector3 *p_normal) {
	Plane *pl = (Plane *)p_pl;
	const Vector3 *normal = (const Vector3 *)p_normal;
	pl->set_normal(*normal);
}

godot_vector3 godot_plane_get_normal(const godot_plane *p_pl) {
	const Plane *pl = (const Plane *)p_pl;
	const Vector3 normal = pl->get_normal();
	godot_vector3 *v3 = (godot_vector3 *)&normal;
	return *v3;
}

void GDAPI godot_plane_set_d(godot_plane *p_pl, const godot_real p_d) {
	Plane *pl = (Plane *)p_pl;
	pl->d = p_d;
}

godot_real GDAPI godot_plane_get_d(const godot_plane *p_pl) {
	const Plane *pl = (const Plane *)p_pl;
	return pl->d;
}

#ifdef __cplusplus
}
#endif
