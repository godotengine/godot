#include "jolt_box_shape_impl_3d.hpp"

#include "servers/jolt_project_settings.hpp"

namespace {

constexpr float MARGIN_FACTOR = 0.08f;

} // namespace

Variant JoltBoxShapeImpl3D::get_data() const {
	return half_extents;
}

void JoltBoxShapeImpl3D::set_data(const Variant& p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::VECTOR3);

	const Vector3 new_half_extents = p_data;
	QUIET_FAIL_COND(new_half_extents == half_extents);

	half_extents = new_half_extents;

	destroy();
}

void JoltBoxShapeImpl3D::set_margin(float p_margin) {
	QUIET_FAIL_COND(margin == p_margin);
	QUIET_FAIL_COND(!JoltProjectSettings::use_shape_margins());

	margin = p_margin;

	destroy();
}

String JoltBoxShapeImpl3D::to_string() const {
	return vformat("{half_extents=%v margin=%f}", half_extents, margin);
}

JPH::ShapeRefC JoltBoxShapeImpl3D::_build() const {
	const auto min_half_extent = (float)half_extents[half_extents.min_axis_index()];
	const float shrunk_margin = MIN(margin, min_half_extent * MARGIN_FACTOR);
	const float actual_margin = JoltProjectSettings::use_shape_margins() ? shrunk_margin : 0.0f;

	const JPH::BoxShapeSettings shape_settings(to_jolt(half_extents), actual_margin);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Godot Jolt failed to build box shape with %s. "
			"It returned the following error: '%s'. "
			"This shape belongs to %s.",
			to_string(),
			to_godot(shape_result.GetError()),
			_owners_to_string()
		)
	);

	return shape_result.Get();
}
