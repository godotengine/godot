/*************************************************************************/
/*  scene_diff.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

/**
	@author AndreaCatania
*/

#ifndef SCENE_DIFF_H
#define SCENE_DIFF_H

#include "core/local_vector.h"
#include "core/oa_hash_map.h"
#include "core/object.h"
#include "net_utilities.h"

class SceneSynchronizer;

struct VarDiff {
	bool is_different = false;
	Variant value;
};

/// This class is used to track the scene changes during a particular period of
/// the frame. You can use it to generate partial FrameSnapshot that contains
/// only portion of a change.
class SceneDiff : public Object {
	GDCLASS(SceneDiff, Object);

	friend class SceneSynchronizer;

	static void _bind_methods();

	uint32_t start_tracking_count = 0;
	LocalVector<LocalVector<Variant> > tracking;
	LocalVector<LocalVector<VarDiff> > diff;

public:
	SceneDiff();

	void start_tracking_scene_changes(const LocalVector<Ref<NetUtility::NodeData> > &p_nodes);
	void stop_tracking_scene_changes(const SceneSynchronizer *p_synchronizer);

	bool is_tracking_in_progress() const;
};

#endif // SCENE_DIFF_H
