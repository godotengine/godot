/**************************************************************************/
/*  lod.h                                                                 */
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

#ifndef LOD_H
#define LOD_H

#include "spatial.h"

class LOD : public Spatial {
	GDCLASS(LOD, Spatial);

	struct LODChild {
		float distance;
		int32_t child_id;
	};

	struct Data {
		LocalVector<LODChild> lod_children;
		int32_t current_lod_child = 0;
		float hysteresis = 1.0f;
		int32_t queue_id = 0;
		const Spatial *current_lod_node = nullptr;
		bool registered = false;
	} data;

	friend class LODManager;
	bool _lod_update(float p_camera_dist_squared);
	void _lod_pre_save();
	void _update_child_distances();

	void _lod_register();
	void _lod_unregister();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_hysteresis(real_t p_distance);
	real_t get_hysteresis() const { return data.hysteresis; }

	void set_lod_priority(int p_priority);
	int get_lod_priority() const { return data.queue_id; }
};

#endif // LOD_H
