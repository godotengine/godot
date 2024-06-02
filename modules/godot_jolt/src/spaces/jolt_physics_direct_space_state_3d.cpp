#include "jolt_physics_direct_space_state_3d.hpp"

#include "objects/jolt_area_impl_3d.hpp"
#include "objects/jolt_body_impl_3d.hpp"
#include "objects/jolt_object_impl_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"
#include "servers/jolt_project_settings.hpp"
#include "shapes/jolt_custom_motion_shape.hpp"
#include "shapes/jolt_shape_impl_3d.hpp"
#include "spaces/jolt_motion_filter_3d.hpp"
#include "spaces/jolt_query_collectors.hpp"
#include "spaces/jolt_query_filter_3d.hpp"
#include "spaces/jolt_space_3d.hpp"

bool JoltPhysicsDirectSpaceState3D::is_body_excluded_from_query(const RID &p_body) const {
	return exclude && exclude->has(p_body);
}
thread_local const HashSet<RID> *JoltPhysicsDirectSpaceState3D::exclude = nullptr;
JoltPhysicsDirectSpaceState3D::JoltPhysicsDirectSpaceState3D(JoltSpace3D* p_space)
	: space(p_space) { }

bool JoltPhysicsDirectSpaceState3D::intersect_ray(const RayParameters &p_parameters,
	RayResult& p_result
) {
	AutoCheck auto_check(this, &p_parameters.exclude);
	space->try_optimize();

	const JoltQueryFilter3D query_filter(
		*this,
		p_parameters.collision_mask,
		p_parameters.collide_with_bodies,
		p_parameters.collide_with_areas,
		p_parameters.pick_ray
	);

	const JPH::RVec3 from = to_jolt_r(p_parameters.from);
	const JPH::RVec3 to = to_jolt_r(p_parameters.to);
	const auto vector = JPH::Vec3(to - from);
	const JPH::RRayCast ray(from, vector);

	JPH::RayCastSettings settings;
	settings.mTreatConvexAsSolid = p_parameters.hit_from_inside;
	settings.mBackFaceMode = p_parameters.hit_back_faces
		? JPH::EBackFaceMode::CollideWithBackFaces
		: JPH::EBackFaceMode::IgnoreBackFaces;

	JoltQueryCollectorClosest<JPH::CastRayCollector> collector;

	space->get_narrow_phase_query()
		.CastRay(ray, settings, collector, query_filter, query_filter, query_filter);

	if (!collector.had_hit()) {
		return false;
	}

	const JPH::RayCastResult& hit = collector.get_hit();

	const JPH::BodyID& body_id = hit.mBodyID;
	const JPH::SubShapeID& sub_shape_id = hit.mSubShapeID2;

	const JoltReadableBody3D body = space->read_body(body_id);
	const JoltObjectImpl3D* object = body.as_object();
	ERR_FAIL_NULL_D(object);

	const JPH::RVec3 position = ray.GetPointOnRay(hit.mFraction);

	JPH::Vec3 normal = JPH::Vec3::sZero();

	if (!p_parameters.hit_from_inside || hit.mFraction > 0.0f) {
		normal = body->GetWorldSpaceSurfaceNormal(sub_shape_id, position);

		// HACK(mihe): If we got a back-face normal we need to flip it
		if (normal.Dot(vector) > 0) {
			normal = -normal;
		}
	}

	p_result.position = to_godot(position);
	p_result.normal = to_godot(normal);
	p_result.rid = object->get_rid();
	p_result.collider_id = object->get_instance_id();
	p_result.collider = object->get_instance_unsafe();
	p_result.shape = 0;

	if (const JoltShapedObjectImpl3D* shaped_object = object->as_shaped()) {
		const int32_t shape_index = shaped_object->find_shape_index(sub_shape_id);
		ERR_FAIL_COND_D(shape_index == -1);
		p_result.shape = shape_index;
	}

	return true;
}

int32_t JoltPhysicsDirectSpaceState3D::intersect_point(const PointParameters &p_parameters,
	ShapeResult* p_results,
	int32_t p_max_results
) {
	if (p_max_results == 0) {
		return 0;
	}
	AutoCheck auto_check(this, &p_parameters.exclude);

	space->try_optimize();

	const JoltQueryFilter3D
		query_filter(*this, p_parameters.collision_mask, p_parameters.collide_with_bodies, p_parameters.collide_with_areas);

	JoltQueryCollectorAnyMulti<JPH::CollidePointCollector, 32> collector(p_max_results);

	space->get_narrow_phase_query()
		.CollidePoint(to_jolt_r(p_parameters.position), collector, query_filter, query_filter, query_filter);

	const int32_t hit_count = collector.get_hit_count();

	for (int32_t i = 0; i < hit_count; ++i) {
		const JPH::CollidePointResult& hit = collector.get_hit(i);

		const JoltReadableBody3D body = space->read_body(hit.mBodyID);
		const JoltObjectImpl3D* object = body.as_object();
		ERR_FAIL_NULL_D(object);

		PhysicsServer3DExtensionShapeResult& result = *p_results++;

		result.rid = object->get_rid();
		result.collider_id = object->get_instance_id();
		result.collider = object->get_instance_unsafe();
		result.shape = 0;

		if (const JoltShapedObjectImpl3D* shaped_object = object->as_shaped()) {
			const int32_t shape_index = shaped_object->find_shape_index(hit.mSubShapeID2);
			ERR_FAIL_COND_D(shape_index == -1);
			result.shape = shape_index;
		}
	}
	return hit_count;
}

int32_t JoltPhysicsDirectSpaceState3D::intersect_shape(const ShapeParameters &p_parameters,
	ShapeResult* p_results,
	int32_t p_max_results
) {
	if (p_max_results == 0) {
		return 0;
	}

	AutoCheck auto_check(this, &p_parameters.exclude);
	space->try_optimize();

	auto* physics_server = static_cast<JoltPhysicsServer3D*>(PhysicsServer3D::get_singleton());

	JoltShapeImpl3D* shape = physics_server->get_shape(p_parameters.shape_rid);
	ERR_FAIL_NULL_D(shape);

	const JPH::ShapeRefC jolt_shape = shape->try_build();
	ERR_FAIL_NULL_D(jolt_shape);

	Transform3D transform = p_parameters.transform;

#ifdef TOOLS_ENABLED
	if (unlikely(transform.basis.determinant() == 0.0f)) {
		ERR_PRINT(vformat(
			"intersect_shape failed due to being passed an invalid transform. "
			"Its basis was found to be singular, which is not supported by Godot Jolt. "
			"This is likely caused by one or more axes having a scale of zero. "
			"Its basis (and thus its scale) will be treated as identity."
		));

		transform.basis = Basis();
	}
#endif // TOOLS_ENABLED

	Vector3 scale;
	decompose(transform, scale);

#ifdef TOOLS_ENABLED
	if (unlikely(!jolt_shape->IsValidScale(to_jolt(scale)))) {
		ERR_PRINT(vformat(
			"intersect_shape failed due to being passed an invalid transform. "
			"A scale of %v is not supported by Godot Jolt for this shape type. "
			"Its scale will instead be treated as (1, 1, 1).",
			scale
		));

		scale = Vector3(1, 1, 1);
	}
#endif // TOOLS_ENABLED

	const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
	const Transform3D transform_com = transform.translated_local(com_scaled);

	JPH::CollideShapeSettings settings;
	settings.mMaxSeparationDistance = (float)p_parameters.margin;

	if (JoltProjectSettings::use_enhanced_edge_removal()) {
		settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	}

	const JoltQueryFilter3D
		query_filter(*this, p_parameters.collision_mask, p_parameters.collide_with_bodies, p_parameters.collide_with_areas);

	JoltQueryCollectorAnyMultiNoEdges<32> collector(p_max_results);

	space->get_narrow_phase_query().CollideShape(
		jolt_shape,
		to_jolt(scale),
		to_jolt_r(transform_com),
		settings,
		to_jolt_r(transform_com.origin),
		collector,
		query_filter,
		query_filter,
		query_filter
	);

	collector.finish();

	const int32_t hit_count = collector.get_hit_count();

	for (int32_t i = 0; i < hit_count; ++i) {
		const JPH::CollideShapeResult& hit = collector.get_hit(i);

		const JoltReadableBody3D body = space->read_body(hit.mBodyID2);
		const JoltObjectImpl3D* object = body.as_object();
		ERR_FAIL_NULL_D(object);

		PhysicsServer3DExtensionShapeResult& result = *p_results++;

		result.rid = object->get_rid();
		result.collider_id = object->get_instance_id();
		result.collider = object->get_instance_unsafe();
		result.shape = 0;

		if (const JoltShapedObjectImpl3D* shaped_object = object->as_shaped()) {
			const int32_t shape_index = shaped_object->find_shape_index(hit.mSubShapeID2);
			ERR_FAIL_COND_D(shape_index == -1);
			result.shape = shape_index;
		}
	}
	return hit_count;
}

bool JoltPhysicsDirectSpaceState3D::cast_motion(const ShapeParameters &p_parameters,
	real_t& p_closest_safe,
	real_t& p_closest_unsafe,
	ShapeRestInfo* p_info
) {
	// HACK(mihe): This rest info parameter doesn't seem to be used anywhere within Godot, and isn't
	// exposed in the bindings, so this will be unsupported until anyone actually needs it.
	ERR_FAIL_COND_D_MSG(
		p_info != nullptr,
		"Providing rest info as part of a shape-cast is not supported by Godot Jolt."
	);

	AutoCheck auto_check(this, &p_parameters.exclude);
	space->try_optimize();

	auto* physics_server = static_cast<JoltPhysicsServer3D*>(PhysicsServer3D::get_singleton());

	JoltShapeImpl3D* shape = physics_server->get_shape(p_parameters.shape_rid);
	ERR_FAIL_NULL_D(shape);

	const JPH::ShapeRefC jolt_shape = shape->try_build();
	ERR_FAIL_NULL_D(jolt_shape);

	Transform3D transform = p_parameters.transform;

#ifdef TOOLS_ENABLED
	if (unlikely(transform.basis.determinant() == 0.0f)) {
		ERR_PRINT(vformat(
			"cast_motion failed due to being passed an invalid transform. "
			"Its basis was found to be singular, which is not supported by Godot Jolt. "
			"This is likely caused by one or more axes having a scale of zero. "
			"Its basis (and thus its scale) will be treated as identity."
		));

		transform.basis = Basis();
	}
#endif // TOOLS_ENABLED

	Vector3 scale;
	decompose(transform, scale);

#ifdef TOOLS_ENABLED
	if (unlikely(!jolt_shape->IsValidScale(to_jolt(scale)))) {
		ERR_PRINT(vformat(
			"cast_motion failed due to being passed an invalid transform. "
			"A scale of %v is not supported by Godot Jolt for this shape type. "
			"Its scale will instead be treated as (1, 1, 1).",
			scale
		));

		scale = Vector3(1, 1, 1);
	}
#endif // TOOLS_ENABLED

	const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
	Transform3D transform_com = transform.translated_local(com_scaled);

	JPH::CollideShapeSettings settings;
	settings.mMaxSeparationDistance = (float)p_parameters.margin;

	if (JoltProjectSettings::use_enhanced_edge_removal()) {
		settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	}

	const JoltQueryFilter3D
		query_filter(*this, p_parameters.collision_mask, p_parameters.collide_with_bodies, p_parameters.collide_with_areas);

	cast_motion_impl(
		*jolt_shape,
		transform_com,
		scale,
		p_parameters.motion,
		true,
		settings,
		query_filter,
		query_filter,
		query_filter,
		JPH::ShapeFilter(),
		p_closest_safe,
		p_closest_unsafe
	);

	return true;
}

bool JoltPhysicsDirectSpaceState3D::collide_shape(const ShapeParameters &p_parameters,
	Vector3* p_results,
	int32_t p_max_results,
	int32_t& p_result_count
) {
	p_result_count = 0;

	if (p_max_results == 0) {
		return false;
	}
	AutoCheck auto_check(this, &p_parameters.exclude);

	space->try_optimize();

	auto* physics_server = static_cast<JoltPhysicsServer3D*>(PhysicsServer3D::get_singleton());

	JoltShapeImpl3D* shape = physics_server->get_shape(p_parameters.shape_rid);
	ERR_FAIL_NULL_D(shape);

	const JPH::ShapeRefC jolt_shape = shape->try_build();
	ERR_FAIL_NULL_D(jolt_shape);

	Transform3D transform = p_parameters.transform;

#ifdef TOOLS_ENABLED
	if (unlikely(transform.basis.determinant() == 0.0f)) {
		ERR_PRINT(vformat(
			"collide_shape failed due to being passed an invalid transform. "
			"Its basis was found to be singular, which is not supported by Godot Jolt. "
			"This is likely caused by one or more axes having a scale of zero. "
			"Its basis (and thus its scale) will be treated as identity."
		));

		transform.basis = Basis();
	}
#endif // TOOLS_ENABLED

	Vector3 scale;
	decompose(transform, scale);

#ifdef TOOLS_ENABLED
	if (unlikely(!jolt_shape->IsValidScale(to_jolt(scale)))) {
		ERR_PRINT(vformat(
			"collide_shape failed due to being passed an invalid transform. "
			"A scale of %v is not supported by Godot Jolt for this shape type. "
			"Its scale will instead be treated as (1, 1, 1).",
			scale
		));

		scale = Vector3(1, 1, 1);
	}
#endif // TOOLS_ENABLED

	const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
	const Transform3D transform_com = transform.translated_local(com_scaled);

	JPH::CollideShapeSettings settings;
	settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	settings.mMaxSeparationDistance = (float)p_parameters.margin;

	const Vector3& base_offset = transform_com.origin;

	const JoltQueryFilter3D
		query_filter(*this, p_parameters.collision_mask, p_parameters.collide_with_bodies, p_parameters.collide_with_areas);

	JoltQueryCollectorAnyMultiNoEdges<32> collector(p_max_results);

	space->get_narrow_phase_query().CollideShape(
		jolt_shape,
		to_jolt(scale),
		to_jolt_r(transform_com),
		settings,
		to_jolt_r(base_offset),
		collector,
		query_filter,
		query_filter,
		query_filter
	);

	if (!collector.finish()) {
		return false;
	}

	auto* points = static_cast<Vector3*>(p_results);

	const int32_t max_points = p_max_results * 2;

	int32_t point_count = 0;

	for (int32_t i = 0; i < collector.get_hit_count(); ++i) {
		const JPH::CollideShapeResult& hit = collector.get_hit(i);

		const Vector3 penetration_axis = to_godot(hit.mPenetrationAxis.Normalized());
		const Vector3 margin_offset = penetration_axis * (float)p_parameters.margin;

		JPH::ContactPoints contact_points1;
		JPH::ContactPoints contact_points2;

		generate_manifold(
			hit,
			contact_points1,
			contact_points2
#ifdef JPH_DEBUG_RENDERER
			,
			to_jolt_r(base_offset)
#endif // JPH_DEBUG_RENDERER
		);

		for (JPH::uint j = 0; j < contact_points1.size(); ++j) {
			points[point_count++] = base_offset + to_godot(contact_points1[j]) + margin_offset;
			points[point_count++] = base_offset + to_godot(contact_points2[j]);

			if (point_count >= max_points) {
				break;
			}
		}

		if (point_count >= max_points) {
			break;
		}
	}

	p_result_count = point_count / 2;

	return true;
}

bool JoltPhysicsDirectSpaceState3D::rest_info(const ShapeParameters &p_parameters,
	ShapeRestInfo* p_info
) {
	AutoCheck auto_check(this, &p_parameters.exclude);
	space->try_optimize();

	auto* physics_server = static_cast<JoltPhysicsServer3D*>(PhysicsServer3D::get_singleton());

	JoltShapeImpl3D* shape = physics_server->get_shape(p_parameters.shape_rid);
	ERR_FAIL_NULL_D(shape);

	const JPH::ShapeRefC jolt_shape = shape->try_build();
	ERR_FAIL_NULL_D(jolt_shape);

	Transform3D transform = p_parameters.transform;

#ifdef TOOLS_ENABLED
	if (unlikely(transform.basis.determinant() == 0.0f)) {
		ERR_PRINT(vformat(
			"get_rest_info failed due to being passed an invalid transform. "
			"Its basis was found to be singular, which is not supported by Godot Jolt. "
			"This is likely caused by one or more axes having a scale of zero. "
			"Its basis (and thus its scale) will be treated as identity."
		));

		transform.basis = Basis();
	}
#endif // TOOLS_ENABLED

	Vector3 scale;
	decompose(transform, scale);

#ifdef TOOLS_ENABLED
	if (unlikely(!jolt_shape->IsValidScale(to_jolt(scale)))) {
		ERR_PRINT(vformat(
			"get_rest_info failed due to being passed an invalid transform. "
			"A scale of %v is not supported by Godot Jolt for this shape type. "
			"Its scale will instead be treated as (1, 1, 1).",
			scale
		));

		scale = Vector3(1, 1, 1);
	}
#endif // TOOLS_ENABLED

	const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
	const Transform3D transform_com = transform.translated_local(com_scaled);

	JPH::CollideShapeSettings settings;
	settings.mMaxSeparationDistance = (float)p_parameters.margin;

	if (JoltProjectSettings::use_enhanced_edge_removal()) {
		settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	}

	const Vector3& base_offset = transform_com.origin;

	const JoltQueryFilter3D
		query_filter(*this, p_parameters.collision_mask, p_parameters.collide_with_bodies,p_parameters.collide_with_areas);

	JoltQueryCollectorClosestNoEdges collector;

	space->get_narrow_phase_query().CollideShape(
		jolt_shape,
		to_jolt(scale),
		to_jolt_r(transform_com),
		settings,
		to_jolt_r(base_offset),
		collector,
		query_filter,
		query_filter,
		query_filter
	);

	if (!collector.finish()) {
		return false;
	}

	const JPH::CollideShapeResult& hit = collector.get_hit();

	const JoltReadableBody3D body = space->read_body(hit.mBodyID2);
	const JoltObjectImpl3D* object = body.as_object();
	ERR_FAIL_NULL_D(object);

	const Vector3 hit_point = base_offset + to_godot(hit.mContactPointOn2);

	p_info->point = hit_point;
	p_info->normal = to_godot(-hit.mPenetrationAxis.Normalized());
	p_info->rid = object->get_rid();
	p_info->collider_id = object->get_instance_id();
	p_info->shape = 0;
	p_info->linear_velocity = object->get_velocity_at_position(hit_point);

	if (const JoltShapedObjectImpl3D* shaped_object = object->as_shaped()) {
		const int32_t shape_index = shaped_object->find_shape_index(hit.mSubShapeID2);
		ERR_FAIL_COND_D(shape_index == -1);
		p_info->shape = shape_index;
	}

	return true;
}

Vector3 JoltPhysicsDirectSpaceState3D::get_closest_point_to_object_volume(
	RID p_object, const Vector3 p_point
) const {
	space->try_optimize();

	auto* physics_server = static_cast<JoltPhysicsServer3D*>(PhysicsServer3D::get_singleton());

	JoltObjectImpl3D* object = physics_server->get_area(p_object);

	if (object == nullptr) {
		object = physics_server->get_body(p_object);
	}

	ERR_FAIL_NULL_D(object);
	ERR_FAIL_COND_D(object->get_space() != space);

	const JoltReadableBody3D body = space->read_body(*object);
	const JPH::TransformedShape root_shape = body->GetTransformedShape();

	JoltQueryCollectorAll<JPH::TransformedShapeCollector, 32> collector;
	root_shape.CollectTransformedShapes(body->GetWorldSpaceBounds(), collector);

	const JPH::RVec3 point = to_jolt_r(p_point);

	float closest_distance_sq = FLT_MAX;
	JPH::RVec3 closest_point = JPH::RVec3::sZero();

	bool found_point = false;

	for (int32_t i = 0; i < collector.get_hit_count(); ++i) {
		const JPH::TransformedShape& shape_transformed = collector.get_hit(i);
		const JPH::Shape& shape = *shape_transformed.mShape;

		if (shape.GetType() != JPH::EShapeType::Convex) {
			continue;
		}

		const auto& shape_convex = static_cast<const JPH::ConvexShape&>(shape);

		JPH::GJKClosestPoint gjk;

		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
		JPH::ConvexShape::SupportBuffer shape_support_buffer;

		const JPH::ConvexShape::Support* shape_support = shape_convex.GetSupportFunction(
			JPH::ConvexShape::ESupportMode::IncludeConvexRadius,
			shape_support_buffer,
			shape_transformed.GetShapeScale()
		);

		const JPH::Quat& shape_rotation = shape_transformed.mShapeRotation;
		const JPH::RVec3& shape_pos_com = shape_transformed.mShapePositionCOM;
		const JPH::RMat44 shape_3x3 = JPH::RMat44::sRotation(shape_rotation);
		const JPH::Vec3 shape_com_local = shape.GetCenterOfMass();
		const JPH::Vec3 shape_com = shape_3x3.Multiply3x3(shape_com_local);
		const JPH::RVec3 shape_pos = shape_pos_com - JPH::RVec3(shape_com);
		const JPH::RMat44 shape_4x4 = shape_3x3.PostTranslated(shape_pos);
		const JPH::RMat44 shape_4x4_inv = shape_4x4.InversedRotationTranslation();

		JPH::PointConvexSupport point_support = {};
		point_support.mPoint = JPH::Vec3(shape_4x4_inv * point);

		JPH::Vec3 separating_axis = JPH::Vec3::sAxisX();
		JPH::Vec3 point_on_a = JPH::Vec3::sZero();
		JPH::Vec3 point_on_b = JPH::Vec3::sZero();

		const float distance_sq = gjk.GetClosestPoints(
			*shape_support,
			point_support,
			JPH::cDefaultCollisionTolerance,
			FLT_MAX,
			separating_axis,
			point_on_a,
			point_on_b
		);

		if (distance_sq == 0.0f) {
			closest_point = point;
			found_point = true;
			break;
		}

		if (distance_sq < closest_distance_sq) {
			closest_distance_sq = distance_sq;
			closest_point = shape_4x4 * point_on_a;
			found_point = true;
		}
	}

	if (found_point) {
		return to_godot(closest_point);
	} else {
		return to_godot(body->GetPosition());
	}
}

bool JoltPhysicsDirectSpaceState3D::test_body_motion(
	const JoltBodyImpl3D& p_body,
	const Transform3D& p_transform,
	const Vector3& p_motion,
	float p_margin,
	int32_t p_max_collisions,
	bool p_collide_separation_ray,
	bool p_recovery_as_collision,
	PhysicsServer3DExtensionMotionResult* p_result
) const {
	p_margin = MAX(p_margin, 0.0001f);
	p_max_collisions = MIN(p_max_collisions, 32);

	Transform3D transform = p_transform;

#ifdef TOOLS_ENABLED
	if (unlikely(transform.basis.determinant() == 0.0f)) {
		ERR_PRINT(vformat(
			"body_test_motion failed due to being passed an invalid transform. "
			"Its basis was found to be singular, which is not supported by Godot Jolt. "
			"This is likely caused by one or more axes having a scale of zero. "
			"Its basis (and thus its scale) will be treated as identity."
		));

		transform.basis = Basis();
	}
#endif // TOOLS_ENABLED

	Vector3 scale;
	decompose(transform, scale);

#ifdef TOOLS_ENABLED
	if (unlikely(!p_body.get_jolt_shape()->IsValidScale(to_jolt(scale)))) {
		ERR_PRINT(vformat(
			"body_test_motion failed due to being passed an invalid transform. "
			"A scale of %v is not supported by Godot Jolt for this shape type. "
			"Its scale will instead be treated as (1, 1, 1).",
			scale
		));

		scale = Vector3(1, 1, 1);
	}
#endif // TOOLS_ENABLED

	space->try_optimize();

	Vector3 recovery;
	const bool recovered = body_motion_recover(p_body, transform, p_margin, recovery);

	transform.origin += recovery;

	real_t safe_fraction = 1.0;
	real_t unsafe_fraction = 1.0;

	const bool hit = body_motion_cast(
		p_body,
		transform,
		scale,
		p_motion,
		p_collide_separation_ray,
		safe_fraction,
		unsafe_fraction
	);

	bool collided = false;

	if (hit || (recovered && p_recovery_as_collision)) {
		collided = body_motion_collide(
			p_body,
			transform.translated(p_motion * unsafe_fraction),
			p_motion,
			p_margin,
			p_max_collisions,
			p_result
		);
	}

	if (p_result == nullptr) {
		return collided;
	}

	if (collided) {
		const PhysicsServer3DExtensionMotionCollision& deepest = p_result->collisions[0];

		p_result->travel = recovery + p_motion * safe_fraction;
		p_result->remainder = p_motion - p_motion * safe_fraction;
		p_result->collision_depth = deepest.depth;
		p_result->collision_safe_fraction = safe_fraction;
		p_result->collision_unsafe_fraction = unsafe_fraction;
	} else {
		p_result->travel = recovery + p_motion;
		p_result->remainder = Vector3();
		p_result->collision_depth = 0.0f;
		p_result->collision_safe_fraction = 1.0f;
		p_result->collision_unsafe_fraction = 1.0f;
		p_result->collision_count = 0;
	}

	return collided;
}

bool JoltPhysicsDirectSpaceState3D::cast_motion_impl(
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
) const {
	p_closest_safe = 1.0f;
	p_closest_unsafe = 1.0f;

	ERR_FAIL_COND_D_MSG(
		p_jolt_shape.GetType() != JPH::EShapeType::Convex,
		"Shape-casting with non-convex shapes is not supported by Godot Jolt."
	);

	const auto motion_length = (float)p_motion.length();

	if (p_ignore_overlaps && motion_length == 0.0f) {
		return false;
	}

	const JPH::RMat44 transform_com = to_jolt_r(p_transform_com);
	const JPH::Vec3 scale = to_jolt(p_scale);
	const JPH::Vec3 motion = to_jolt(p_motion);
	const JPH::Vec3 motion_local = transform_com.Multiply3x3Transposed(motion);

	JPH::AABox aabb = p_jolt_shape.GetWorldSpaceBounds(transform_com, scale);
	JPH::AABox aabb_translated = aabb;
	aabb_translated.Translate(motion);
	aabb.Encapsulate(aabb_translated);

	JoltQueryCollectorAnyMulti<JPH::CollideShapeBodyCollector, 2048> aabb_collector;

	space->get_broad_phase_query()
		.CollideAABox(aabb, aabb_collector, p_broad_phase_layer_filter, p_object_layer_filter);

	if (!aabb_collector.had_hit()) {
		return false;
	}

	const JPH::RVec3 base_offset = transform_com.GetTranslation();

	JoltCustomMotionShape motion_shape(static_cast<const JPH::ConvexShape&>(p_jolt_shape));

	auto collides = [&](const JPH::Body& p_other_body, float p_fraction) {
		motion_shape.set_motion(motion_local * p_fraction);

		const JPH::TransformedShape other_shape = p_other_body.GetTransformedShape();

		JoltQueryCollectorAnyNoEdges collide_collector;

		other_shape.CollideShape(
			&motion_shape,
			scale,
			transform_com,
			p_settings,
			base_offset,
			collide_collector,
			p_shape_filter
		);

		return collide_collector.finish();
	};

	// Figure out the number of steps we need in our binary search in order to achieve millimeter
	// precision, within reason. Derived from `2^-step_count * motion_length = 0.001`.
	const int32_t step_count = CLAMP(int32_t(logf(1000.0f * motion_length) / Mathf_LN2), 4, 16);

	bool collided = false;

	for (int32_t i = 0; i < aabb_collector.get_hit_count(); ++i) {
		const JPH::BodyID other_jolt_id = aabb_collector.get_hit(i);

		if (!p_body_filter.ShouldCollide(other_jolt_id)) {
			continue;
		}

		const JoltReadableBody3D other_jolt_body = space->read_body(other_jolt_id);

		if (!p_body_filter.ShouldCollideLocked(*other_jolt_body)) {
			continue;
		}

		if (!collides(*other_jolt_body, 1.0f)) {
			continue;
		}

		if (p_ignore_overlaps && collides(*other_jolt_body, 0.0f)) {
			continue;
		}

		float lo = 0.0f;
		float hi = 1.0f;
		float coeff = 0.5f;

		for (int j = 0; j < step_count; ++j) {
			const float fraction = lo + (hi - lo) * coeff;

			if (collides(*other_jolt_body, fraction)) {
				collided = true;

				hi = fraction;

				if (j == 0 || lo > 0.0f) {
					coeff = 0.5f;
				} else {
					coeff = 0.25f;
				}
			} else {
				lo = fraction;

				if (j == 0 || hi < 1.0f) {
					coeff = 0.5f;
				} else {
					coeff = 0.75f;
				}
			}
		}

		if (lo < p_closest_safe) {
			p_closest_safe = lo;
			p_closest_unsafe = hi;
		}
	}

	return collided;
}

bool JoltPhysicsDirectSpaceState3D::body_motion_recover(
	const JoltBodyImpl3D& p_body,
	const Transform3D& p_transform,
	float p_margin,
	Vector3& p_recovery
) const {
	const int32_t recovery_iterations = JoltProjectSettings::get_kinematic_recovery_iterations();
	const float recovery_amount = JoltProjectSettings::get_kinematic_recovery_amount();

	const JPH::Shape* jolt_shape = p_body.get_jolt_shape();

	const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
	Transform3D transform_com = p_transform.translated_local(com_scaled);

	JPH::CollideShapeSettings settings;
	settings.mMaxSeparationDistance = p_margin;

	if (JoltProjectSettings::use_enhanced_edge_removal()) {
		settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	}

	const Vector3& base_offset = transform_com.origin;

	const JoltMotionFilter3D motion_filter(p_body);

	JoltQueryCollectorAnyMultiNoEdges<32> collector;

	bool recovered = false;

	for (int32_t i = 0; i < recovery_iterations; ++i) {
		collector.reset();

		space->get_narrow_phase_query().CollideShape(
			jolt_shape,
			JPH::Vec3::sReplicate(1.0f),
			to_jolt_r(transform_com),
			settings,
			to_jolt_r(base_offset),
			collector,
			motion_filter,
			motion_filter,
			motion_filter,
			motion_filter
		);

		if (!collector.finish()) {
			break;
		}

		const int32_t hit_count = collector.get_hit_count();

		float combined_priority = 0.0;

		for (int j = 0; j < hit_count; j++) {
			const JPH::CollideShapeResult& hit = collector.get_hit(j);

			const JoltReadableBody3D other_jolt_body = space->read_body(hit.mBodyID2);
			const JoltBodyImpl3D* other_body = other_jolt_body.as_body();
			ERR_CONTINUE(other_body == nullptr);

			combined_priority += other_body->get_collision_priority();
		}

		const float average_priority = MAX(
			combined_priority / (float)hit_count,
			(float)CMP_EPSILON
		);

		recovered = true;

		Vector3 recovery;

		for (int32_t j = 0; j < hit_count; ++j) {
			const JPH::CollideShapeResult& hit = collector.get_hit(j);

			const Vector3 penetration_axis = to_godot(hit.mPenetrationAxis.Normalized());
			const Vector3 margin_offset = penetration_axis * p_margin;

			const Vector3 point_on_1 = base_offset + to_godot(hit.mContactPointOn1) + margin_offset;
			const Vector3 point_on_2 = base_offset + to_godot(hit.mContactPointOn2);

			const real_t distance_to_1 = penetration_axis.dot(point_on_1 + recovery);
			const real_t distance_to_2 = penetration_axis.dot(point_on_2);

			const auto penetration_depth = float(distance_to_1 - distance_to_2);

			if (penetration_depth <= 0.0f) {
				continue;
			}

			const JoltReadableBody3D other_jolt_body = space->read_body(hit.mBodyID2);
			const JoltBodyImpl3D* other_body = other_jolt_body.as_body();
			ERR_CONTINUE(other_body == nullptr);

			const float recovery_distance = penetration_depth * recovery_amount;
			const float other_priority = other_body->get_collision_priority();
			const float other_priority_normalized = other_priority / average_priority;
			const float scaled_recovery_distance = recovery_distance * other_priority_normalized;

			recovery -= penetration_axis * scaled_recovery_distance;
		}

		if (recovery == Vector3()) {
			break;
		}

		p_recovery += recovery;
		transform_com.origin += recovery;
	}

	return recovered;
}

bool JoltPhysicsDirectSpaceState3D::body_motion_cast(
	const JoltBodyImpl3D& p_body,
	const Transform3D& p_transform,
	const Vector3& p_scale,
	const Vector3& p_motion,
	bool p_collide_separation_ray,
	real_t& p_safe_fraction,
	real_t& p_unsafe_fraction
) const {
	const Transform3D body_transform = p_transform.scaled_local(p_scale);

	JPH::CollideShapeSettings settings;

	if (JoltProjectSettings::use_enhanced_edge_removal()) {
		settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	}

	const JoltMotionFilter3D motion_filter(p_body, p_collide_separation_ray);

	bool collided = false;

	for (int32_t i = 0; i < p_body.get_shape_count(); ++i) {
		if (p_body.is_shape_disabled(i)) {
			continue;
		}

		JoltShapeImpl3D* shape = p_body.get_shape(i);

		if (!shape->is_convex()) {
			continue;
		}

		const JPH::ShapeRefC jolt_shape = shape->try_build();
		QUIET_FAIL_NULL_D(jolt_shape);

		Vector3 scale;

		const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
		const Transform3D transform_local = p_body.get_shape_transform_scaled(i);
		const Transform3D transform_com_local = transform_local.translated_local(com_scaled);
		const Transform3D transform_com = body_transform * transform_com_local;
		const Transform3D transform_com_unscaled = decomposed(transform_com, scale);

#ifdef TOOLS_ENABLED
		if (!jolt_shape->IsValidScale(to_jolt(scale))) {
			continue;
		}
#endif // TOOLS_ENABLED

		real_t shape_safe_fraction = 1.0;
		real_t shape_unsafe_fraction = 1.0;

		collided |= cast_motion_impl(
			*jolt_shape,
			transform_com_unscaled,
			scale,
			p_motion,
			false,
			settings,
			motion_filter,
			motion_filter,
			motion_filter,
			motion_filter,
			shape_safe_fraction,
			shape_unsafe_fraction
		);

		p_safe_fraction = MIN(p_safe_fraction, shape_safe_fraction);
		p_unsafe_fraction = MIN(p_unsafe_fraction, shape_unsafe_fraction);
	}

	return collided;
}

bool JoltPhysicsDirectSpaceState3D::body_motion_collide(
	const JoltBodyImpl3D& p_body,
	const Transform3D& p_transform,
	const Vector3& p_motion,
	float p_margin,
	int32_t p_max_collisions,
	PhysicsServer3DExtensionMotionResult* p_result
) const {
	if (p_max_collisions == 0) {
		return false;
	}

	const JPH::Shape* jolt_shape = p_body.get_jolt_shape();

	const Vector3 com_scaled = to_godot(jolt_shape->GetCenterOfMass());
	const Transform3D transform_com = p_transform.translated_local(com_scaled);

	JPH::CollideShapeSettings settings;
	settings.mCollectFacesMode = JPH::ECollectFacesMode::CollectFaces;
	settings.mMaxSeparationDistance = p_margin;

	const Vector3& base_offset = transform_com.origin;

	const JoltMotionFilter3D motion_filter(p_body);

	JoltQueryCollectorClosestMultiNoEdges<32> collector(p_max_collisions);

	space->get_narrow_phase_query().CollideShape(
		jolt_shape,
		JPH::Vec3::sReplicate(1.0f),
		to_jolt_r(transform_com),
		settings,
		to_jolt_r(base_offset),
		collector,
		motion_filter,
		motion_filter,
		motion_filter,
		motion_filter
	);

	const bool collided = collector.finish();

	if (!collided || p_result == nullptr) {
		return collided;
	}

	int32_t count = 0;

	for (int32_t i = 0; i < collector.get_hit_count(); ++i) {
		const JPH::CollideShapeResult& hit = collector.get_hit(i);

		const float penetration_depth = hit.mPenetrationDepth + p_margin;

		if (penetration_depth <= 0.0f) {
			continue;
		}

		const Vector3 normal = to_godot(-hit.mPenetrationAxis.Normalized());

		if (p_motion.length_squared() > 0) {
			const Vector3 direction = p_motion.normalized();

			if (direction.dot(normal) >= -CMP_EPSILON) {
				continue;
			}
		}

		JPH::ContactPoints contact_points1;
		JPH::ContactPoints contact_points2;

		if (p_max_collisions > 1) {
			generate_manifold(
				hit,
				contact_points1,
				contact_points2
#ifdef JPH_DEBUG_RENDERER
				,
				to_jolt_r(base_offset)
#endif // JPH_DEBUG_RENDERER
			);
		} else {
			contact_points2.push_back(hit.mContactPointOn2);
		}

		const JoltReadableBody3D collider_jolt_body = space->read_body(hit.mBodyID2);
		const JoltShapedObjectImpl3D* collider = collider_jolt_body.as_shaped();
		ERR_FAIL_NULL_D(collider);

		const int32_t local_shape = p_body.find_shape_index(hit.mSubShapeID1);
		ERR_FAIL_COND_D(local_shape == -1);

		const int32_t collider_shape = collider->find_shape_index(hit.mSubShapeID2);
		ERR_FAIL_COND_D(collider_shape == -1);

		for (JPH::Vec3 contact_point : contact_points2) {
			const Vector3 position = base_offset + to_godot(contact_point);

			PhysicsServer3DExtensionMotionCollision& collision = p_result->collisions[count++];

			collision.position = position;
			collision.normal = normal;
			collision.collider_velocity = collider->get_velocity_at_position(position);
			collision.collider_angular_velocity = collider->get_angular_velocity();
			collision.depth = penetration_depth;
			collision.local_shape = local_shape;
			collision.collider_id = collider->get_instance_id();
			collision.collider = collider->get_rid();
			collision.collider_shape = collider_shape;

			if (count == p_max_collisions) {
				break;
			}
		}

		if (count == p_max_collisions) {
			break;
		}
	}

	p_result->collision_count = count;

	return count > 0;
}

void JoltPhysicsDirectSpaceState3D::generate_manifold(
	const JPH::CollideShapeResult& p_hit,
	JPH::ContactPoints& p_contact_points1,
	JPH::ContactPoints& p_contact_points2
#ifdef JPH_DEBUG_RENDERER
	,
	JPH::RVec3Arg p_center_of_mass
#endif // JPH_DEBUG_RENDERER
) const {
	const JPH::PhysicsSystem& physics_system = space->get_physics_system();
	const JPH::PhysicsSettings& physics_settings = physics_system.GetPhysicsSettings();
	const JPH::Vec3 penetration_axis = p_hit.mPenetrationAxis.Normalized();

	JPH::ManifoldBetweenTwoFaces(
		p_hit.mContactPointOn1,
		p_hit.mContactPointOn2,
		penetration_axis,
		physics_settings.mManifoldToleranceSq,
		p_hit.mShape1Face,
		p_hit.mShape2Face,
		p_contact_points1,
		p_contact_points2
#ifdef JPH_DEBUG_RENDERER
		,
		p_center_of_mass
#endif // JPH_DEBUG_RENDERER
	);

	if (p_contact_points1.size() > 4) {
		JPH::PruneContactPoints(
			penetration_axis,
			p_contact_points1,
			p_contact_points2
#ifdef JPH_DEBUG_RENDERER
			,
			p_center_of_mass
#endif // JPH_DEBUG_RENDERER
		);
	}
}
