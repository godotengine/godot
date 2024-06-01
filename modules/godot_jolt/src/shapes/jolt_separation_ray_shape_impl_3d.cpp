#include "jolt_separation_ray_shape_impl_3d.hpp"

#include "shapes/jolt_custom_ray_shape.hpp"

Variant JoltSeparationRayShapeImpl3D::get_data() const {
	Dictionary data;
	data["length"] = length;
	data["slide_on_slope"] = slide_on_slope;
	return data;
}

void JoltSeparationRayShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	const Dictionary data = p_data;

	const Variant maybe_length = data.get("length", {});
	ERR_FAIL_COND(maybe_length.get_type() != Variant::FLOAT);

	const Variant maybe_slide_on_slope = data.get("slide_on_slope", {});
	ERR_FAIL_COND(maybe_slide_on_slope.get_type() != Variant::BOOL);

	const float new_length = maybe_length;
	const bool new_slide_on_slope = maybe_slide_on_slope;

	QUIET_FAIL_COND(new_length == length && new_slide_on_slope == slide_on_slope);

	length = new_length;
	slide_on_slope = new_slide_on_slope;

	destroy();
}

String JoltSeparationRayShapeImpl3D::to_string() const {
	return vformat("{length=%f slide_on_slope=%s}", length, slide_on_slope);
}

JPH::ShapeRefC JoltSeparationRayShapeImpl3D::_build() const {
	ERR_FAIL_COND_D_MSG(
		length <= 0.0f,
		vformat(
			"Godot Jolt failed to build separation ray shape with %s. "
			"Its length must be greater than 0. "
			"This shape belongs to %s.",
			to_string(),
			_owners_to_string()
		)
	);

	const JoltCustomRayShapeSettings shape_settings(length, slide_on_slope);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build separation ray shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return shape_result.Get();
}
