/**************************************************************************/
/*  physics_direct_space_state2d_extension.cpp                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/physics_direct_space_state2d_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

bool PhysicsDirectSpaceState2DExtension::is_body_excluded_from_query(const RID &p_body) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectSpaceState2DExtension::get_class_static()._native_ptr(), StringName("is_body_excluded_from_query")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_body);
}

bool PhysicsDirectSpaceState2DExtension::_intersect_ray(const Vector2 &p_from, const Vector2 &p_to, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_hit_from_inside, PhysicsServer2DExtensionRayResult *p_result) {
	return false;
}

int32_t PhysicsDirectSpaceState2DExtension::_intersect_point(const Vector2 &p_position, uint64_t p_canvas_instance_id, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, PhysicsServer2DExtensionShapeResult *p_results, int32_t p_max_results) {
	return 0;
}

int32_t PhysicsDirectSpaceState2DExtension::_intersect_shape(const RID &p_shape_rid, const Transform2D &p_transform, const Vector2 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, PhysicsServer2DExtensionShapeResult *p_result, int32_t p_max_results) {
	return 0;
}

bool PhysicsDirectSpaceState2DExtension::_cast_motion(const RID &p_shape_rid, const Transform2D &p_transform, const Vector2 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, float *p_closest_safe, float *p_closest_unsafe) {
	return false;
}

bool PhysicsDirectSpaceState2DExtension::_collide_shape(const RID &p_shape_rid, const Transform2D &p_transform, const Vector2 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, void *p_results, int32_t p_max_results, int32_t *p_result_count) {
	return false;
}

bool PhysicsDirectSpaceState2DExtension::_rest_info(const RID &p_shape_rid, const Transform2D &p_transform, const Vector2 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, PhysicsServer2DExtensionShapeRestInfo *p_rest_info) {
	return false;
}

} // namespace godot
