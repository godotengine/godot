/*************************************************************************/
/*  memory_tracker.h                                                     */
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

#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <stdint.h>

#ifdef ALLOCATION_TRACKING_ENABLED
// Usage:
// Use report() to get a list of all current tracked allocations.
// To compare the allocations between two points in time, use snapshots.
// Take a snapshot before using take_snapshot(). Take a snapshot later,
// then compare the two to form a difference report using compare_snapshots().

class AllocationTracking {
public:
	static void add_alloc(void *p_address, uint32_t p_size, const char *p_filename, uint32_t p_line_number);
	static void remove_alloc(void *p_address);
	static void realloc(void *p_address, uint32_t p_new_size);

	static int take_snapshot();
	static void delete_snapshot(int p_snapshot_id);

	static void report(const char *p_title);
	static void compare_snapshots(int p_snapshot_id_a, int p_snapshot_id_b, const char *p_title);

	static void frame_update();
	static void tick_update();

	static uint32_t get_allocs_per_frame();
	static uint32_t get_allocs_per_tick();
	static uint32_t get_total_alloc_size_per_frame();
	static uint32_t get_total_alloc_size_per_tick();
};

#else
// Dummy to allow easy compiling out.
class AllocationTracking {
public:
	static int take_snapshot() { return 0; }
	static void delete_snapshot(int p_snapshot_id) {}

	static void report(const char *p_title) {}
	static void compare_snapshots(int p_snapshot_id_a, int p_snapshot_id_b, const char *p_title) {}

	static void frame_update() {}
	static void tick_update() {}

	static uint32_t get_allocs_per_frame() { return 0; }
	static uint32_t get_allocs_per_tick() { return 0; }
	static uint32_t get_total_alloc_size_per_frame() { return 0; }
	static uint32_t get_total_alloc_size_per_tick() { return 0; }
};
#endif

#endif // MEMORY_TRACKER_H
