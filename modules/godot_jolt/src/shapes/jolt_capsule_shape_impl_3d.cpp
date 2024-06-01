#include "jolt_capsule_shape_impl_3d.hpp"

Variant JoltCapsuleShapeImpl3D::get_data() const {
	Dictionary data;
	data["height"] = height;
	data["radius"] = radius;
	return data;
}

void JoltCapsuleShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_height = data.get("height", {});
	ERR_FAIL_COND(maybe_height.get_type() != Variant::FLOAT);

	const Variant maybe_radius = data.get("radius", {});
	ERR_FAIL_COND(maybe_radius.get_type() != Variant::FLOAT);

	const float new_height = maybe_height;
	const float new_radius = maybe_radius;

	QUIET_FAIL_COND(new_height == height && new_radius == radius);

	height = new_height;
	radius = new_radius;

	destroy();
}

String JoltCapsuleShapeImpl3D::to_string() const {
	return vformat("{height=%f radius=%f}", height, radius);
}

JPH::ShapeRefC JoltCapsuleShapeImpl3D::_build() const {
	ERR_FAIL_COND_D_MSG(
		radius <= 0.0f,
		vformat(
			"Godot Jolt failed to build capsule shape with %s. "
			"Its radius must be greater than 0. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	ERR_FAIL_COND_D_MSG(
		height <= 0.0f,
		vformat(
			"Godot Jolt failed to build capsule shape with %s. "
			"Its height must be greater than 0. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	ERR_FAIL_COND_D_MSG(
		height < radius * 2.0f,
		vformat(
			"Godot Jolt failed to build capsule shape with %s. "
			"Its height must be at least double that of its radius. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	const float half_height = height / 2.0f;
	const float cylinder_height = half_height - radius;

	const JPH::CapsuleShapeSettings shape_settings(cylinder_height, radius);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build capsule shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return shape_result.Get();
}
