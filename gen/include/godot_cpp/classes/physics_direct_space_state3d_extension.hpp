/**************************************************************************/
/*  physics_direct_space_state3d_extension.hpp                            */
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

#pragma once

#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/classes/physics_server3d_extension_ray_result.hpp>
#include <godot_cpp/classes/physics_server3d_extension_shape_rest_info.hpp>
#include <godot_cpp/classes/physics_server3d_extension_shape_result.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class RID;
struct Transform3D;

class PhysicsDirectSpaceState3DExtension : public PhysicsDirectSpaceState3D {
	GDEXTENSION_CLASS(PhysicsDirectSpaceState3DExtension, PhysicsDirectSpaceState3D)

public:
	bool is_body_excluded_from_query(const RID &p_body) const;
	virtual bool _intersect_ray(const Vector3 &p_from, const Vector3 &p_to, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_hit_from_inside, bool p_hit_back_faces, bool p_pick_ray, PhysicsServer3DExtensionRayResult *p_result);
	virtual int32_t _intersect_point(const Vector3 &p_position, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, PhysicsServer3DExtensionShapeResult *p_results, int32_t p_max_results);
	virtual int32_t _intersect_shape(const RID &p_shape_rid, const Transform3D &p_transform, const Vector3 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, PhysicsServer3DExtensionShapeResult *p_result_count, int32_t p_max_results);
	virtual bool _cast_motion(const RID &p_shape_rid, const Transform3D &p_transform, const Vector3 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, float *p_closest_safe, float *p_closest_unsafe, PhysicsServer3DExtensionShapeRestInfo *p_info);
	virtual bool _collide_shape(const RID &p_shape_rid, const Transform3D &p_transform, const Vector3 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, void *p_results, int32_t p_max_results, int32_t *p_result_count);
	virtual bool _rest_info(const RID &p_shape_rid, const Transform3D &p_transform, const Vector3 &p_motion, float p_margin, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas, PhysicsServer3DExtensionShapeRestInfo *p_rest_info);
	virtual Vector3 _get_closest_point_to_object_volume(const RID &p_object, const Vector3 &p_point) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PhysicsDirectSpaceState3D::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_intersect_ray), decltype(&T::_intersect_ray)>) {
			BIND_VIRTUAL_METHOD(T, _intersect_ray, 2022529123);
		}
		if constexpr (!std::is_same_v<decltype(&B::_intersect_point), decltype(&T::_intersect_point)>) {
			BIND_VIRTUAL_METHOD(T, _intersect_point, 3378904092);
		}
		if constexpr (!std::is_same_v<decltype(&B::_intersect_shape), decltype(&T::_intersect_shape)>) {
			BIND_VIRTUAL_METHOD(T, _intersect_shape, 728953575);
		}
		if constexpr (!std::is_same_v<decltype(&B::_cast_motion), decltype(&T::_cast_motion)>) {
			BIND_VIRTUAL_METHOD(T, _cast_motion, 2320624824);
		}
		if constexpr (!std::is_same_v<decltype(&B::_collide_shape), decltype(&T::_collide_shape)>) {
			BIND_VIRTUAL_METHOD(T, _collide_shape, 2320624824);
		}
		if constexpr (!std::is_same_v<decltype(&B::_rest_info), decltype(&T::_rest_info)>) {
			BIND_VIRTUAL_METHOD(T, _rest_info, 856242757);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_closest_point_to_object_volume), decltype(&T::_get_closest_point_to_object_volume)>) {
			BIND_VIRTUAL_METHOD(T, _get_closest_point_to_object_volume, 2056183332);
		}
	}

public:
};

} // namespace godot

