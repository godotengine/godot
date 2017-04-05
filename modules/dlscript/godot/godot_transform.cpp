#include "godot_transform.h"

#include "math/transform.h"

#ifdef __cplusplus
extern "C" {
#endif

void _transform_api_anchor() {
}

void GDAPI godot_transform_new(godot_transform *p_trans) {
	Transform *trans = (Transform *)p_trans;
	*trans = Transform();
}

void GDAPI godot_transform_new_with_basis(godot_transform *p_trans, const godot_basis *p_basis) {
	Transform *trans = (Transform *)p_trans;
	const Basis *basis = (const Basis *)p_basis;
	*trans = Transform(*basis);
}

void GDAPI godot_transform_new_with_basis_origin(godot_transform *p_trans, const godot_basis *p_basis, const godot_vector3 *p_origin) {
	Transform *trans = (Transform *)p_trans;
	const Basis *basis = (const Basis *)p_basis;
	const Vector3 *origin = (const Vector3 *)p_origin;
	*trans = Transform(*basis, *origin);
}

godot_basis GDAPI *godot_transform_get_basis(godot_transform *p_trans) {
	Transform *trans = (Transform *)p_trans;
	return (godot_basis *)&trans->basis;
}

godot_vector3 GDAPI *godot_transform_get_origin(godot_transform *p_trans) {
	Transform *trans = (Transform *)p_trans;
	return (godot_vector3 *)&trans->origin;
}

#ifdef __cplusplus
}
#endif
