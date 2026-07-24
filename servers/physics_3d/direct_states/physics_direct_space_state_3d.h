/**************************************************************************/
/*  physics_direct_space_state_3d.h                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/variant/type_info.h"
#include "servers/physics_3d/physics_server_3d_types.h"
#include "servers/physics_3d/queries/physics_cast_motion_result_3d.h"
#include "servers/physics_3d/queries/physics_point_intersection_result_3d.h"
#include "servers/physics_3d/queries/physics_point_query_parameters_3d.h"
#include "servers/physics_3d/queries/physics_ray_intersection_result_3d.h"
#include "servers/physics_3d/queries/physics_ray_query_parameters_3d.h"
#include "servers/physics_3d/queries/physics_rest_info_result_3d.h"
#include "servers/physics_3d/queries/physics_shape_collision_result_3d.h"
#include "servers/physics_3d/queries/physics_shape_intersection_result_3d.h"
#include "servers/physics_3d/queries/physics_shape_query_parameters_3d.h"

class PhysicsDirectSpaceState3D : public Object {
	GDCLASS(PhysicsDirectSpaceState3D, Object);

private:
	Dictionary _intersect_ray(RequiredParam<PhysicsRayQueryParameters3D> rp_ray_query);
	TypedArray<Dictionary> _intersect_point(RequiredParam<PhysicsPointQueryParameters3D> rp_point_query, int p_max_results = 32);
	TypedArray<Dictionary> _intersect_shape(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query, int p_max_results = 32);
	Vector<real_t> _cast_motion(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query);
	TypedArray<Vector3> _collide_shape(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query, int p_max_results = 32);
	Dictionary _get_rest_info(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query);

	bool _intersect_ray_typed(RequiredParam<PhysicsRayQueryParameters3D> rp_ray_query, RequiredParam<PhysicsRayIntersectionResult3D> rp_result);
	bool _intersect_point_typed(RequiredParam<PhysicsPointQueryParameters3D> rp_point_query, RequiredParam<PhysicsPointIntersectionResult3D> rp_result);
	bool _intersect_shape_typed(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query, RequiredParam<PhysicsShapeIntersectionResult3D> rp_result);
	bool _cast_motion_typed(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query, RequiredParam<PhysicsCastMotionResult3D> rp_result);
	bool _collide_shape_typed(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query, RequiredParam<PhysicsShapeCollisionResult3D> rp_result);
	bool _get_rest_info_typed(RequiredParam<PhysicsShapeQueryParameters3D> rp_shape_query, RequiredParam<PhysicsRestInfoResult3D> rp_result);

protected:
	static void _bind_methods();

public:
	virtual bool intersect_ray(const PS3DT::RayParameters &p_parameters, PS3DT::RayResult &r_result) = 0;
	virtual int intersect_point(const PS3DT::PointParameters &p_parameters, PS3DT::ShapeResult *r_results, int p_result_max) = 0;
	virtual int intersect_shape(const PS3DT::ShapeParameters &p_parameters, PS3DT::ShapeResult *r_results, int p_result_max) = 0;
	virtual bool cast_motion(const PS3DT::ShapeParameters &p_parameters, real_t &p_closest_safe, real_t &p_closest_unsafe, PS3DT::ShapeRestInfo *r_info = nullptr) = 0;
	virtual bool collide_shape(const PS3DT::ShapeParameters &p_parameters, Vector3 *r_results, int p_result_max, int &r_result_count) = 0;
	virtual bool rest_info(const PS3DT::ShapeParameters &p_parameters, PS3DT::ShapeRestInfo *r_info) = 0;

	virtual Vector3 get_closest_point_to_object_volume(RID p_object, const Vector3 p_point) const = 0;

	PhysicsDirectSpaceState3D();
};
