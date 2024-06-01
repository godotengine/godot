#pragma once

#include "shapes/jolt_shape_impl_3d.hpp"

class JoltWorldBoundaryShapeImpl3D final : public JoltShapeImpl3D {
public:
	ShapeType get_type() const override { return ShapeType::SHAPE_WORLD_BOUNDARY; }

	bool is_convex() const override { return false; }

	Variant get_data() const override;

	void set_data(const Variant& p_data) override;

	float get_margin() const override { return 0.0f; }

	void set_margin([[maybe_unused]] float p_margin) override { }

private:
	JPH::ShapeRefC _build() const override;

	Plane plane;
};
