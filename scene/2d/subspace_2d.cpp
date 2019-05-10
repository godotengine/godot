#include "subspace_2d.h"

void Subspace2D::set_space_2d(const Ref<Space2D> &p_space_2d) {
	if (space_2d == p_space_2d)
		return;

	if (p_space_2d.is_valid()) {
		space_2d = p_space_2d;
	} else {
		WARN_PRINT("Invalid Space2D");
		space_2d = Ref<Space2D>(memnew(Space2D));
	}
}

Ref<Space2D> Subspace2D::get_space_2d() const {
	return space_2d;
}

void Subspace2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_space_2d", "space_2d"), &Subspace2D::set_space_2d);
	ClassDB::bind_method(D_METHOD("get_space_2d"), &Subspace2D::get_space_2d);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "space_2d", PROPERTY_HINT_RESOURCE_TYPE, "Space2D"), "set_space_2d", "get_space_2d");
}

Subspace2D::Subspace2D() {
	space_2d = Ref<Space2D>(memnew(Space2D));
}

Subspace2D::~Subspace2D() {}
