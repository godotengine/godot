#include "jolt_custom_double_sided_shape.hpp"

#include "servers/jolt_project_settings.hpp"

namespace {

JPH::Shape* construct_double_sided() {
	return new JoltCustomDoubleSidedShape();
}

void collide_double_sided_vs_shape(
	const JPH::Shape* p_shape1,
	const JPH::Shape* p_shape2,
	JPH::Vec3Arg p_scale1,
	JPH::Vec3Arg p_scale2,
	JPH::Mat44Arg p_center_of_mass_transform1,
	JPH::Mat44Arg p_center_of_mass_transform2,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator1,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator2,
	const JPH::CollideShapeSettings& p_collide_shape_settings,
	JPH::CollideShapeCollector& p_collector,
	const JPH::ShapeFilter& p_shape_filter
) {
	ERR_FAIL_COND(p_shape1->GetSubType() != JoltCustomShapeSubType::DOUBLE_SIDED);

	const auto* shape1 = static_cast<const JoltCustomDoubleSidedShape*>(p_shape1);

	JPH::CollideShapeSettings new_collide_shape_settings = p_collide_shape_settings;
	new_collide_shape_settings.mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;

	JPH::CollisionDispatch::sCollideShapeVsShape(
		shape1->GetInnerShape(),
		p_shape2,
		p_scale1,
		p_scale2,
		p_center_of_mass_transform1,
		p_center_of_mass_transform2,
		p_sub_shape_id_creator1,
		p_sub_shape_id_creator2,
		new_collide_shape_settings,
		p_collector,
		p_shape_filter
	);
}

void collide_shape_vs_double_sided(
	const JPH::Shape* p_shape1,
	const JPH::Shape* p_shape2,
	JPH::Vec3Arg p_scale1,
	JPH::Vec3Arg p_scale2,
	JPH::Mat44Arg p_center_of_mass_transform1,
	JPH::Mat44Arg p_center_of_mass_transform2,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator1,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator2,
	const JPH::CollideShapeSettings& p_collide_shape_settings,
	JPH::CollideShapeCollector& p_collector,
	const JPH::ShapeFilter& p_shape_filter
) {
	ERR_FAIL_COND(p_shape2->GetSubType() != JoltCustomShapeSubType::DOUBLE_SIDED);

	const auto* shape2 = static_cast<const JoltCustomDoubleSidedShape*>(p_shape2);

	JPH::CollideShapeSettings new_collide_shape_settings = p_collide_shape_settings;
	new_collide_shape_settings.mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;

	JPH::CollisionDispatch::sCollideShapeVsShape(
		p_shape1,
		shape2->GetInnerShape(),
		p_scale1,
		p_scale2,
		p_center_of_mass_transform1,
		p_center_of_mass_transform2,
		p_sub_shape_id_creator1,
		p_sub_shape_id_creator2,
		new_collide_shape_settings,
		p_collector,
		p_shape_filter
	);
}

void cast_shape_vs_double_sided(
	const JPH::ShapeCast& p_shape_cast,
	const JPH::ShapeCastSettings& p_shape_cast_settings,
	const JPH::Shape* p_shape,
	JPH::Vec3Arg p_scale,
	const JPH::ShapeFilter& p_shape_filter,
	JPH::Mat44Arg p_center_of_mass_transform2,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator1,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator2,
	JPH::CastShapeCollector& p_collector
) {
	ERR_FAIL_COND(p_shape->GetSubType() != JoltCustomShapeSubType::DOUBLE_SIDED);

	JPH::ShapeCastSettings new_shape_cast_settings = p_shape_cast_settings;
	new_shape_cast_settings.mBackFaceModeTriangles = JPH::EBackFaceMode::CollideWithBackFaces;

	const auto* shape = static_cast<const JoltCustomDoubleSidedShape*>(p_shape);

	JPH::CollisionDispatch::sCastShapeVsShapeLocalSpace(
		p_shape_cast,
		new_shape_cast_settings,
		shape->GetInnerShape(),
		p_scale,
		p_shape_filter,
		p_center_of_mass_transform2,
		p_sub_shape_id_creator1,
		p_sub_shape_id_creator2,
		p_collector
	);
}

} // namespace

JPH::ShapeSettings::ShapeResult JoltCustomDoubleSidedShapeSettings::Create() const {
	if (mCachedResult.IsEmpty()) {
		new JoltCustomDoubleSidedShape(*this, mCachedResult);
	}

	return mCachedResult;
}

void JoltCustomDoubleSidedShape::register_type() {
	JPH::ShapeFunctions& shape_functions = JPH::ShapeFunctions::sGet(
		JoltCustomShapeSubType::DOUBLE_SIDED
	);

	shape_functions.mConstruct = construct_double_sided;
	shape_functions.mColor = JPH::Color::sPurple;

	for (const JPH::EShapeSubType sub_type : JPH::sAllSubShapeTypes) {
		JPH::CollisionDispatch::sRegisterCollideShape(
			JoltCustomShapeSubType::DOUBLE_SIDED,
			sub_type,
			collide_double_sided_vs_shape
		);

		JPH::CollisionDispatch::sRegisterCollideShape(
			sub_type,
			JoltCustomShapeSubType::DOUBLE_SIDED,
			collide_shape_vs_double_sided
		);
	}

	for (const JPH::EShapeSubType sub_type : JPH::sConvexSubShapeTypes) {
		JPH::CollisionDispatch::sRegisterCastShape(
			sub_type,
			JoltCustomShapeSubType::DOUBLE_SIDED,
			cast_shape_vs_double_sided
		);
	}
}

void JoltCustomDoubleSidedShape::CastRay(
	const JPH::RayCast& p_ray,
	const JPH::RayCastSettings& p_ray_cast_settings,
	const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
	JPH::CastRayCollector& p_collector,
	const JPH::ShapeFilter& p_shape_filter
) const {
	JPH::RayCastSettings new_ray_cast_settings = p_ray_cast_settings;

	if (!back_face_collision && !JoltProjectSettings::use_legacy_ray_casting()) {
		new_ray_cast_settings.SetBackFaceMode(JPH::EBackFaceMode::IgnoreBackFaces);
	}

	return mInnerShape->CastRay(
		p_ray,
		new_ray_cast_settings,
		p_sub_shape_id_creator,
		p_collector,
		p_shape_filter
	);
}
