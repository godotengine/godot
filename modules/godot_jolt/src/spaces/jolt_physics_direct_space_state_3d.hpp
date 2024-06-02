#pragma once
#include "../common.h"
#include "containers/hash_map.hpp"
#include "containers/hash_set.hpp"
#include "containers/local_vector.hpp"
#include "containers/inline_vector.hpp"

class JoltBodyImpl3D;
class JoltShapeImpl3D;
class JoltSpace3D;

class JoltPhysicsDirectSpaceState3D final : public PhysicsDirectSpaceState3D {
	GDCLASS(JoltPhysicsDirectSpaceState3D, PhysicsDirectSpaceState3D)

private:
	static void _bind_methods() { }
	thread_local static const HashSet<RID> *exclude;
	void begin_check(const HashSet<RID> *p_exclude)
	{
		exclude = p_exclude;
	}
	void end_check()
	{
		exclude = nullptr;
	}
	struct AutoCheck
	{
		AutoCheck(JoltPhysicsDirectSpaceState3D* p_owenr,const HashSet<RID> *p_exclude)
		 {
			owenr = p_owenr;
			owenr->begin_check(p_exclude); 
		}
		~AutoCheck() { owenr->end_check(); }
		JoltPhysicsDirectSpaceState3D * owenr = nullptr;
	};

public:
	JoltPhysicsDirectSpaceState3D() = default;
	bool is_body_excluded_from_query(const RID &p_body) const;

	explicit JoltPhysicsDirectSpaceState3D(JoltSpace3D* p_space);

	bool intersect_ray(const RayParameters &p_parameters,
		RayResult& p_result
	) override;

	int32_t intersect_point(const PointParameters &p_parameters,
		ShapeResult* p_results,
		int32_t p_max_results
	) override;

	int32_t intersect_shape(const ShapeParameters &p_parameters,
		ShapeResult* p_results,
		int32_t p_max_results
	) override;

	bool cast_motion(const ShapeParameters &p_parameters,
		real_t& p_closest_safe,
		real_t& p_closest_unsafe,
		ShapeRestInfo* p_info
	) override;

	bool collide_shape(const ShapeParameters &p_parameters,
		Vector3* p_results,
		int32_t p_max_results,
		int32_t& p_result_count
	) override;

	bool rest_info(const ShapeParameters &p_parameters,
		ShapeRestInfo* p_info
	) override;

	Vector3 get_closest_point_to_object_volume(RID p_object, const Vector3 p_point)
		const override;

	bool test_body_motion(
		const JoltBodyImpl3D& p_body,
		const Transform3D& p_transform,
		const Vector3& p_motion,
		float p_margin,
		int32_t p_max_collisions,
		bool p_collide_separation_ray,
		bool p_recovery_as_collision,
		PhysicsServer3DExtensionMotionResult* p_result
	) const;

	JoltSpace3D& get_space() const { return *space; }

private:
	bool cast_motion_impl(
		const JPH::Shape& p_jolt_shape,
		const Transform3D& p_transform_com,
		const Vector3& p_scale,
		const Vector3& p_motion,
		bool p_ignore_overlaps,
		const JPH::CollideShapeSettings& p_settings,
		const JPH::BroadPhaseLayerFilter& p_broad_phase_layer_filter,
		const JPH::ObjectLayerFilter& p_object_layer_filter,
		const JPH::BodyFilter& p_body_filter,
		const JPH::ShapeFilter& p_shape_filter,
		real_t& p_closest_safe,
		real_t& p_closest_unsafe
	) const;

	bool body_motion_recover(
		const JoltBodyImpl3D& p_body,
		const Transform3D& p_transform,
		float p_margin,
		Vector3& p_recovery
	) const;

	bool body_motion_cast(
		const JoltBodyImpl3D& p_body,
		const Transform3D& p_transform,
		const Vector3& p_scale,
		const Vector3& p_motion,
		bool p_collide_separation_ray,
		real_t& p_safe_fraction,
		real_t& p_unsafe_fraction
	) const;

	bool body_motion_collide(
		const JoltBodyImpl3D& p_body,
		const Transform3D& p_transform,
		const Vector3& p_motion,
		float p_margin,
		int32_t p_max_collisions,
		PhysicsServer3DExtensionMotionResult* p_result
	) const;

	void generate_manifold(
		const JPH::CollideShapeResult& p_hit,
		JPH::ContactPoints& p_contact_points1,
		JPH::ContactPoints& p_contact_points2
#ifdef JPH_DEBUG_RENDERER
		,
		JPH::RVec3Arg p_center_of_mass
#endif // JPH_DEBUG_RENDERER
	) const;

	JoltSpace3D* space = nullptr;
};
