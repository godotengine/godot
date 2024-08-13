#pragma once

#include "shapes/jolt_custom_shape_type.hpp"

class JoltCustomDecoratedShapeSettings : public JPH::DecoratedShapeSettings {
public:
	using JPH::DecoratedShapeSettings::DecoratedShapeSettings;
};

class JoltCustomDecoratedShape : public JPH::DecoratedShape {
public:
	using JPH::DecoratedShape::DecoratedShape;

	JPH::AABox GetLocalBounds() const override { return mInnerShape->GetLocalBounds(); }

	JPH::AABox GetWorldSpaceBounds(JPH::Mat44Arg p_center_of_mass_transform, JPH::Vec3Arg p_scale)
		const override {
		return mInnerShape->GetWorldSpaceBounds(p_center_of_mass_transform, p_scale);
	}

	float GetInnerRadius() const override { return mInnerShape->GetInnerRadius(); }

	JPH::MassProperties GetMassProperties() const override {
		return mInnerShape->GetMassProperties();
	}

	JPH::Vec3 GetSurfaceNormal(
		const JPH::SubShapeID& p_sub_shape_id,
		JPH::Vec3Arg p_local_surface_position
	) const override {
		return mInnerShape->GetSurfaceNormal(p_sub_shape_id, p_local_surface_position);
	}

	JPH::uint64 GetSubShapeUserData(const JPH::SubShapeID& p_sub_shape_id) const override {
		return mInnerShape->GetSubShapeUserData(p_sub_shape_id);
	}

	JPH::TransformedShape GetSubShapeTransformedShape(
		const JPH::SubShapeID& p_sub_shape_id,
		JPH::Vec3Arg p_position_com,
		JPH::QuatArg p_rotation,
		JPH::Vec3Arg p_scale,
		JPH::SubShapeID& p_remainder
	) const override {
		return mInnerShape->GetSubShapeTransformedShape(
			p_sub_shape_id,
			p_position_com,
			p_rotation,
			p_scale,
			p_remainder
		);
	}

	// clang-format off

	void GetSubmergedVolume(
		JPH::Mat44Arg p_center_of_mass_transform,
		JPH::Vec3Arg p_scale,
		const JPH::Plane& p_surface,
		float& p_total_volume,
		float& p_submerged_volume,
		JPH::Vec3& p_center_of_buoyancy
		JPH_IF_DEBUG_RENDERER(, JPH::RVec3Arg p_base_offset)
	) const override {
		mInnerShape->GetSubmergedVolume(
			p_center_of_mass_transform,
			p_scale,
			p_surface,
			p_total_volume,
			p_submerged_volume,
			p_center_of_buoyancy
			JPH_IF_DEBUG_RENDERER(, p_base_offset)
		);
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
	) const override {
		mInnerShape->Draw(
			p_renderer,
			p_center_of_mass_transform,
			p_scale,
			p_color,
			p_use_material_colors,
			p_draw_wireframe
		);
	}

	void DrawGetSupportFunction(
		JPH::DebugRenderer* p_renderer,
		JPH::RMat44Arg p_center_of_mass_transform,
		JPH::Vec3Arg p_scale,
		JPH::ColorArg p_color,
		bool p_draw_support_direction
	) const override {
		mInnerShape->DrawGetSupportFunction(
			p_renderer,
			p_center_of_mass_transform,
			p_scale,
			p_color,
			p_draw_support_direction
		);
	}

	void DrawGetSupportingFace(
		JPH::DebugRenderer* p_renderer,
		JPH::RMat44Arg p_center_of_mass_transform,
		JPH::Vec3Arg p_scale
	) const override {
		mInnerShape->DrawGetSupportingFace(p_renderer, p_center_of_mass_transform, p_scale);
	}
#endif // JPH_DEBUG_RENDERER

	bool CastRay(
		const JPH::RayCast& p_ray,
		const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		JPH::RayCastResult& p_hit
	) const override {
		return mInnerShape->CastRay(p_ray, p_sub_shape_id_creator, p_hit);
	}

	void CastRay(
		const JPH::RayCast& p_ray,
		const JPH::RayCastSettings& p_ray_cast_settings,
		const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		JPH::CastRayCollector& p_collector,
		const JPH::ShapeFilter& p_shape_filter = {}
	) const override {
		return mInnerShape->CastRay(
			p_ray,
			p_ray_cast_settings,
			p_sub_shape_id_creator,
			p_collector,
			p_shape_filter
		);
	}

	void CollidePoint(
		JPH::Vec3Arg p_point,
		const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		JPH::CollidePointCollector& p_collector,
		const JPH::ShapeFilter& p_shape_filter = {}
	) const override {
		mInnerShape->CollidePoint(p_point, p_sub_shape_id_creator, p_collector, p_shape_filter);
	}

	void CollideSoftBodyVertices(
		JPH::Mat44Arg p_center_of_mass_transform,
		JPH::Vec3Arg p_scale,
		JPH::SoftBodyVertex* p_vertices,
		JPH::uint p_num_vertices,
		float p_delta_time,
		JPH::Vec3Arg p_displacement_due_to_gravity,
		int p_colliding_shape_index
	) const override {
		mInnerShape->CollideSoftBodyVertices(
			p_center_of_mass_transform,
			p_scale,
			p_vertices,
			p_num_vertices,
			p_delta_time,
			p_displacement_due_to_gravity,
			p_colliding_shape_index
		);
	}

	void CollectTransformedShapes(
		const JPH::AABox& p_box,
		JPH::Vec3Arg p_position_com,
		JPH::QuatArg p_rotation,
		JPH::Vec3Arg p_scale,
		const JPH::SubShapeIDCreator& p_sub_shape_id_creator,
		JPH::TransformedShapeCollector& p_collector,
		const JPH::ShapeFilter& p_shape_filter = {}
	) const override {
		mInnerShape->CollectTransformedShapes(
			p_box,
			p_position_com,
			p_rotation,
			p_scale,
			p_sub_shape_id_creator,
			p_collector,
			p_shape_filter
		);
	}

	void TransformShape(
		JPH::Mat44Arg p_center_of_mass_transform,
		JPH::TransformedShapeCollector& p_collector
	) const override {
		mInnerShape->TransformShape(p_center_of_mass_transform, p_collector);
	}

	void GetTrianglesStart(
		GetTrianglesContext& p_context,
		const JPH::AABox& p_box,
		JPH::Vec3Arg p_position_com,
		JPH::QuatArg p_rotation,
		JPH::Vec3Arg p_scale
	) const override {
		mInnerShape->GetTrianglesStart(p_context, p_box, p_position_com, p_rotation, p_scale);
	}

	int GetTrianglesNext(
		GetTrianglesContext& p_context,
		int p_max_triangles_requested,
		JPH::Float3* p_triangle_vertices,
		const JPH::PhysicsMaterial** p_materials = nullptr
	) const override {
		return mInnerShape->GetTrianglesNext(
			p_context,
			p_max_triangles_requested,
			p_triangle_vertices,
			p_materials
		);
	}

	Stats GetStats() const override { return {sizeof(*this), 0}; }

	float GetVolume() const override { return mInnerShape->GetVolume(); }

};
