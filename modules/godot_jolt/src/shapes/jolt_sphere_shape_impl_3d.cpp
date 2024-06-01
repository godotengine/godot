#include "jolt_sphere_shape_impl_3d.hpp"

Variant JoltSphereShapeImpl3D::get_data() const {
	return radius;
}

void JoltSphereShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::FLOAT);

	const float new_radius = p_data;
	QUIET_FAIL_COND(new_radius == radius);

	radius = new_radius;

	destroy();
}

String JoltSphereShapeImpl3D::to_string() const {
	return vformat("{radius=%f}", radius);
}

JPH::ShapeRefC JoltSphereShapeImpl3D::_build() const {
	ERR_FAIL_COND_D_MSG(
		radius <= 0.0f,
		vformat(
			"Godot Jolt failed to build sphere shape with %s. "
			"Its radius must be greater than 0. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	const JPH::SphereShapeSettings shape_settings(radius);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build sphere shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return shape_result.Get();
}
