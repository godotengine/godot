/*************************************************************************/
/*  common_pools.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef COMMON_POOLS_H
#define COMMON_POOLS_H

#include "core/math/aabb.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/string/node_path.h"
#include "core/string/string_name.h"
#include "core/templates/fixed_pool_allocator.h"
#include "core/templates/paged_allocator.h"
#include "core/variant/array_private.h"
#include "servers/physics_2d/godot_area_pair_2d.h"
#include "servers/physics_2d/godot_body_pair_2d.h"
#include "servers/physics_3d/godot_area_pair_3d.h"
#include "servers/physics_3d/godot_body_pair_3d.h"

class CommonPools {
#ifdef DEV_ENABLED
public:
	// Only used for debugging purposes
	void debug_update();

private:
	uint64_t _next_debug_log_tick = 0;
#endif

public:
	static CommonPools &get_singleton() {
		static CommonPools s;
		return s;
	}

	CommonPools(const CommonPools &) = delete;
	CommonPools &operator=(const CommonPools &) = delete;

	// Extended types in Variant
	FixedPoolAllocator<Transform3D> pool_transform3ds;
	FixedPoolAllocator<Transform2D> pool_transform2ds;
	FixedPoolAllocator<AABB> pool_aabbs;
	FixedPoolAllocator<Basis> pool_bases;

	// Pairing constraints in physics (may possibly get away without thread protection? NYI)
	FixedPoolAllocator<GodotAreaPair3D> pool_area_pair;
	FixedPoolAllocator<GodotArea2Pair3D> pool_area2_pair;
	FixedPoolAllocator<GodotAreaSoftBodyPair3D, 512> pool_area_soft_body_pair;
	FixedPoolAllocator<GodotBodySoftBodyPair3D, 512> pool_body_soft_body_pair;
	FixedPoolAllocator<GodotBodyPair3D> pool_body_pair;

	FixedPoolAllocator<GodotArea2Pair2D> pool_area2_pair_2d;
	FixedPoolAllocator<GodotAreaPair2D> pool_area_pair_2d;
	FixedPoolAllocator<GodotBodyPair2D> pool_body_pair_2d;

	// Misc
	PagedAllocator<ArrayPrivate, true> pool_array_private;
	PagedAllocator<NodePath::Data, true> pool_nodepath_data;
	PagedAllocator<StringName::_Data, true> pool_stringname_data;

private:
	CommonPools() {}
	~CommonPools() {}
};

#endif // COMMON_POOLS_H
