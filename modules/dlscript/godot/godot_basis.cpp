#include "godot_basis.h"

#include "math/matrix3.h"

#ifdef __cplusplus
extern "C" {
#endif

void _basis_api_anchor() {
}

void GDAPI godot_basis_new(godot_basis *p_basis) {
	Basis *basis = (Basis *)p_basis;
	*basis = Basis();
}

void GDAPI godot_basis_new_with_euler_quat(godot_basis *p_basis, const godot_quat *p_euler) {
	Basis *basis = (Basis *)p_basis;
	Quat *euler = (Quat *)p_euler;
	*basis = Basis(*euler);
}

void GDAPI godot_basis_new_with_euler(godot_basis *p_basis, const godot_vector3 *p_euler) {
	Basis *basis = (Basis *)p_basis;
	Vector3 *euler = (Vector3 *)p_euler;
	*basis = Basis(*euler);
}

godot_quat GDAPI godot_basis_as_quat(const godot_basis *p_basis) {
	const Basis *basis = (const Basis *)p_basis;
	godot_quat quat;
	Quat *p_quat = (Quat *)&quat;
	*p_quat = basis->operator Quat();
	return quat;
}

godot_vector3 GDAPI godot_basis_get_euler(const godot_basis *p_basis) {
	const Basis *basis = (const Basis *)p_basis;
	godot_vector3 euler;
	Vector3 *p_euler = (Vector3 *)&euler;
	*p_euler = basis->get_euler();
	return euler;
}

/*
 * p_elements is a pointer to an array of 3 (!!) vector3
 */
void GDAPI godot_basis_get_elements(godot_basis *p_basis, godot_vector3 *p_elements) {
	Basis *basis = (Basis *)p_basis;
	Vector3 *elements = (Vector3 *)p_elements;
	elements[0] = basis->elements[0];
	elements[1] = basis->elements[1];
	elements[2] = basis->elements[2];
}

#ifdef __cplusplus
}
#endif
