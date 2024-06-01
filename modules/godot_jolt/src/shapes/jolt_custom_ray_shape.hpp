#pragma once

#include "shapes/jolt_custom_shape_type.hpp"

class JoltCustomRayShapeSettings final : public JPH::ConvexShapeSettings {
public:
	JoltCustomRayShapeSettings() = default;

	JoltCustomRayShapeSettings(
		float p_length,
		bool p_slide_on_slope,
		const JPH::PhysicsMaterial* p_material = nullptr
	)
		: material(p_material)
		, length(p_length)
		, slide_on_slope(p_slide_on_slope) { }

	JPH::ShapeSettings::ShapeResult Create() const override;

	JPH::RefConst<JPH::PhysicsMaterial> material;

	float length = 1.0f;

	bool slide_on_slope = false;
};

class JoltCustomRayShape final : public JPH::ConvexShape {
public:
	static void register_type();

	JoltCustomRayShape()
		: JPH::ConvexShape(JoltCustomShapeSubType::RAY) { }

	JoltCustomRayShape(
		const JoltCustomRayShapeSettings& p_settings,
		JPH::Shape::ShapeResult& p_result
	)
		: JPH::ConvexShape(JoltCustomShapeSubType::RAY, p_settings, p_result)
		, material(p_settings.material)
		, length(p_settings.length)
		, slide_on_slope(p_settings.slide_on_slope) {
		if (!p_result.HasError()) {
			p_result.Set(this);
		}
	}

	JoltCustomRayShape(
		float p_length,
		bool p_slide_on_slope,
		const JPH::PhysicsMaterial* p_material = nullptr
	)
		: JPH::ConvexShape(JoltCustomShapeSubType::RAY)
		, material(p_material)
		, length(p_length)
		, slide_on_slope(p_slide_on_slope) { }

	JPH::AABox GetLocalBounds() const override;

	float GetInnerRadius() const override { return 0.0f; }

	JPH::MassProperties GetMassProperties() const override;

	JPH::Vec3 GetSurfaceNormal(
		[[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id,
		[[maybe_unused]] JPH::Vec3Arg p_local_surface_position
	) const override {
		return JPH::Vec3::sAxisZ();
	}

	// clang-format off

	void GetSubmergedVolume(
		[[maybe_unused]] JPH::Mat44Arg p_center_of_mass_transform,
		[[maybe_unused]] JPH::Vec3Arg p_scale,
		[[maybe_unused]] const JPH::Plane& p_surface,
		float& p_total_volume,
		float& p_submerged_volume,
		JPH::Vec3& p_center_of_buoyancy
#ifdef JPH_DEBUG_RENDERER
		, [[maybe_unused]] JPH::RVec3Arg p_base_offset
#endif // JPH_DEBUG_RENDERER
	) const override {
		p_total_volume = 0.0f;
		p_submerged_volume = 0.0f;
		p_center_of_buoyancy = JPH::Vec3::sZero();
	}

	// clang-format on

#ifdef JPH_DEBUG_RENDERER

	void Draw(
		JPH::DebugRenderer* p_renderer,
		JPH::RMat44Arg p_center_of_mass_transform,
		JPH::Vec3Arg p_scale,
		JPH::ColorArg p_color,
		bool p_use_material_colors,
		bool p_draw_wireframe
	) const override;

#endif // JPH_DEBUG_RENDERER

	bool CastRay(
		[[maybe_unused]] const JPH::RayCast& p_ray,
		[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		[[maybe_unused]] JPH::RayCastResult& p_hit
	) const override {
		return false;
	}

	void CastRay(
		[[maybe_unused]] const JPH::RayCast& p_ray,
		[[maybe_unused]] const JPH::RayCastSettings& p_ray_cast_settings,
		[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		[[maybe_unused]] JPH::CastRayCollector& p_collector,
		[[maybe_unused]] const JPH::ShapeFilter& p_shape_filter = {}
	) const override { }

	void CollidePoint(
		[[maybe_unused]] JPH::Vec3Arg p_point,
		[[maybe_unused]] const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		[[maybe_unused]] JPH::CollidePointCollector& p_collector,
		[[maybe_unused]] const JPH::ShapeFilter& p_shape_filter = {}
	) const override { }

	void CollideSoftBodyVertices(
		[[maybe_unused]] JPH::Mat44Arg p_center_of_mass_transform,
		[[maybe_unused]] JPH::Vec3Arg p_scale,
		[[maybe_unused]] JPH::SoftBodyVertex* p_vertices,
		[[maybe_unused]] JPH::uint p_num_vertices,
		[[maybe_unused]] float p_delta_time,
		[[maybe_unused]] JPH::Vec3Arg p_displacement_due_to_gravity,
		[[maybe_unused]] int p_colliding_shape_index
	) const override { }

	JPH::Shape::Stats GetStats() const override { return {sizeof(*this), 0}; }

	float GetVolume() const override { return 0.0f; }

	const JPH::ConvexShape::Support* GetSupportFunction(
		JPH::ConvexShape::ESupportMode p_mode,
		JPH::ConvexShape::SupportBuffer& p_buffer,
		JPH::Vec3Arg p_scale
	) const override;

	JPH::RefConst<JPH::PhysicsMaterial> material;

	float length = 0.0f;

	bool slide_on_slope = false;
};
