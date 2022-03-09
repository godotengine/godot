/*************************************************************************/
/*  common_pools.cpp                                                     */
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

#include "common_pools.h"
#include "core/config/engine.h"

// Uncomment this to produce regular statistics logs to the output
// #define GODOT_COMMON_POOLS_LOG_STATISTICS
#define GODOT_COMMON_POOLS_LOG(A) print_line("\t" #A " allocs : " + itos(A.get_total_allocs()) + ", mem : " + itos(A.estimate_memory_use() / 1024) + " Kb")

#ifdef DEV_ENABLED
void CommonPools::debug_update() {
#ifdef GODOT_COMMON_POOLS_LOG_STATISTICS

	if (Engine::get_singleton()->get_physics_frames() < _next_debug_log_tick) {
		return;
	}
	_next_debug_log_tick = Engine::get_singleton()->get_physics_frames() + (Engine::get_singleton()->get_physics_ticks_per_second() * 10);

	print_line("CommonPool...");

	GODOT_COMMON_POOLS_LOG(pool_transform3ds);
	GODOT_COMMON_POOLS_LOG(pool_transform2ds);
	GODOT_COMMON_POOLS_LOG(pool_aabbs);
	GODOT_COMMON_POOLS_LOG(pool_bases);

	GODOT_COMMON_POOLS_LOG(pool_array_private);
	GODOT_COMMON_POOLS_LOG(pool_nodepath_data);
	GODOT_COMMON_POOLS_LOG(pool_stringname_data);

	GODOT_COMMON_POOLS_LOG(pool_area_pair);
	GODOT_COMMON_POOLS_LOG(pool_area2_pair);
	GODOT_COMMON_POOLS_LOG(pool_area_soft_body_pair);
	GODOT_COMMON_POOLS_LOG(pool_body_soft_body_pair);
	GODOT_COMMON_POOLS_LOG(pool_body_pair);

	GODOT_COMMON_POOLS_LOG(pool_area_pair_2d);
	GODOT_COMMON_POOLS_LOG(pool_area2_pair_2d);
	GODOT_COMMON_POOLS_LOG(pool_body_pair_2d);

#endif
}

#undef GODOT_COMMON_POOLS_LOG
#endif // DEV_ENABLED
