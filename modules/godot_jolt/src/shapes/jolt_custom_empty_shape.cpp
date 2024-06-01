#include "jolt_custom_empty_shape.hpp"

namespace {

JPH::Shape* construct_empty() {
	return new JoltCustomEmptyShape();
}

void collide_noop(
	[[maybe_unused]] const JPH::Shape* p_shape1,
	[[maybe_unused]] const JPH::Shape* p_shape2,
	[[maybe_unused]] JPH::Vec3Arg p_scale1,
	[[maybe_unused]] JPH::Vec3Arg p_scale2,
	[[maybe_unused]] JPH::Mat44Arg p_center_of_mass_transform1,
	[[maybe_unused]] JPH::Mat44Arg p_center_of_mass_transform2,
	[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator1,
	[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator2,
	[[maybe_unused]] const JPH::CollideShapeSettings& p_collide_shape_settings,
	[[maybe_unused]] JPH::CollideShapeCollector& p_collector,
	[[maybe_unused]] const JPH::ShapeFilter& p_shape_filter
) { }

void cast_noop(
	[[maybe_unused]] const JPH::ShapeCast& p_shape_cast,
	[[maybe_unused]] const JPH::ShapeCastSettings& p_shape_cast_settings,
	[[maybe_unused]] const JPH::Shape* p_shape,
	[[maybe_unused]] JPH::Vec3Arg p_scale,
	[[maybe_unused]] const JPH::ShapeFilter& p_shape_filter,
	[[maybe_unused]] JPH::Mat44Arg p_center_of_mass_transform2,
	[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator1,
	[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator2,
	[[maybe_unused]] JPH::CastShapeCollector& p_collector
) { }

} // namespace

JPH::ShapeSettings::ShapeResult JoltCustomEmptyShapeSettings::Create() const {
	if (mCachedResult.IsEmpty()) {
		new JoltCustomEmptyShape(*this, mCachedResult);
	}

	return mCachedResult;
}

void JoltCustomEmptyShape::register_type() {
	JPH::ShapeFunctions& shape_functions = JPH::ShapeFunctions::sGet(JoltCustomShapeSubType::EMPTY);

	shape_functions.mConstruct = construct_empty;
	shape_functions.mColor = JPH::Color::sBlack;

	for (const JPH::EShapeSubType sub_type : JPH::sAllSubShapeTypes) {
		JPH::CollisionDispatch::sRegisterCollideShape(
			JoltCustomShapeSubType::EMPTY,
			sub_type,
			collide_noop
		);

		JPH::CollisionDispatch::sRegisterCollideShape(
			sub_type,
			JoltCustomShapeSubType::EMPTY,
			collide_noop
		);

		JPH::CollisionDispatch::sRegisterCastShape(
			JoltCustomShapeSubType::EMPTY,
			sub_type,
			cast_noop
		);

		JPH::CollisionDispatch::sRegisterCastShape(
			sub_type,
			JoltCustomShapeSubType::EMPTY,
			cast_noop
		);
	}
}

JPH::Vec3 JoltCustomEmptyShape::GetCenterOfMass() const {
	// HACK(mihe): In Godot Physics you're able to provide a custom center-of-mass to a shapeless
	// body without affecting its inertia. We can't emulate this behavior while relying on Jolt's
	// `OffsetCenterOfMassShape` due to it translating its inner mass properties (and thus our
	// inertia) by the center of mass provided to it. So instead we have this shape provide its own.
	return center_of_mass;
}

JPH::MassProperties JoltCustomEmptyShape::GetMassProperties() const {
	JPH::MassProperties mass_properties;

	// HACK(mihe): Since this shape has no volume we can't really give it a correct set of mass
	// properties, so instead we just give it some random/arbitrary ones.
	mass_properties.mMass = 1.0f;
	mass_properties.mInertia = JPH::Mat44::sIdentity();

	return mass_properties;
}
