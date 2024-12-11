#pragma once

#include "shapes/jolt_custom_decorated_shape.hpp"
#include "shapes/jolt_custom_shape_type.hpp"

class JoltCustomUserDataShapeSettings final : public JoltCustomDecoratedShapeSettings {
public:
	using JoltCustomDecoratedShapeSettings::JoltCustomDecoratedShapeSettings;

	ShapeResult Create() const override;
};

class JoltCustomUserDataShape final : public JoltCustomDecoratedShape {
public:
	static void register_type();

	JoltCustomUserDataShape()
		: JoltCustomDecoratedShape(JoltCustomShapeSubType::OVERRIDE_USER_DATA) { }

	JoltCustomUserDataShape(
		const JoltCustomUserDataShapeSettings& p_settings,
		ShapeResult& p_result
	)
		: JoltCustomDecoratedShape(
			  JoltCustomShapeSubType::OVERRIDE_USER_DATA,
			  p_settings,
			  p_result
		  ) {
		if (!p_result.HasError()) {
			p_result.Set(this);
		}
	}

	JPH::uint64 GetSubShapeUserData([[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id
	) const override {
		return GetUserData();
	}
};
