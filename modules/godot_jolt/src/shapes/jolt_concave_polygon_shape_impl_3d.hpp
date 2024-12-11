#pragma once

#include "shapes/jolt_shape_impl_3d.hpp"

class JoltConcavePolygonShapeImpl3D final : public JoltShapeImpl3D {
public:
	ShapeType get_type() const override { return ShapeType::SHAPE_CONCAVE_POLYGON; }

	bool is_convex() const override { return false; }

	Variant get_data() const override;

	void set_data(const Variant& p_data) override;

	float get_margin() const override { return 0.0f; }

	void set_margin([[maybe_unused]] float p_margin) override { }

	AABB get_aabb() const override { return aabb; }

	String to_string() const;

private:
	JPH::ShapeRefC _build() const override;

	AABB _calculate_aabb() const;

	AABB aabb;

	PackedVector3Array faces;

	bool back_face_collision = false;
};
