#include "vector3i.h"

void Vector3i::set_axis(int p_axis, int32_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	coord[p_axis] = p_value;
}
int32_t Vector3i::get_axis(int p_axis) const {

	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	return operator[](p_axis);
}

int Vector3i::min_axis() const {

	return x < y ? (x < z ? 0 : 2) : (y < z ? 1 : 2);
}
int Vector3i::max_axis() const {

	return x < y ? (y < z ? 2 : 1) : (x < z ? 2 : 0);
}

Vector3i::operator String() const {

	return (itos(x) + ", " + itos(y) + ", " + itos(z));
}
