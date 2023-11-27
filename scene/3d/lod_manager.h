/**************************************************************************/
/*  lod_manager.h                                                         */
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

#ifndef LOD_MANAGER_H
#define LOD_MANAGER_H

#include "core/local_vector.h"
#include "core/os/mutex.h"

class Camera;
class LOD;
struct Vector3;

class LODManager {
public:
	enum { NUM_LOD_QUEUES = 5 };

	void register_camera(Camera *p_camera);
	void remove_camera(Camera *p_camera);
	void register_lod(LOD *p_lod, uint32_t p_queue_id);
	void unregister_lod(LOD *p_lod, uint32_t p_queue_id);
	void update();

	void notify_saving(bool p_active);

	static void set_enabled(bool p_enabled) { _enabled = p_enabled; }
	static bool is_enabled() { return _enabled; }

private:
	void _update_queue(uint32_t p_queue_id, const Vector3 *p_camera_positions, uint32_t p_num_cameras);

	struct Queue {
		LocalVector<LOD *> lods;
		uint32_t lod_iterator = 0;
	};

	struct Data {
		LocalVector<Camera *> cameras;
		Queue queues[NUM_LOD_QUEUES];
		BinaryMutex mutex;
		bool saving = false;
	} data;

	static bool _enabled;
};

#endif // LOD_MANAGER_H
