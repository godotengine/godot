/**************************************************************************/
/*  lod_manager.cpp                                                       */
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

#include "lod_manager.h"

#include "scene/3d/camera.h"
#include "scene/3d/lod.h"

bool LODManager::_enabled = true;

void LODManager::register_camera(Camera *p_camera) {
	DEV_ASSERT(p_camera);
	data.cameras.push_back(p_camera);
}

void LODManager::remove_camera(Camera *p_camera) {
	data.cameras.erase(p_camera);
}

void LODManager::register_lod(LOD *p_lod, uint32_t p_queue_id) {
	ERR_FAIL_UNSIGNED_INDEX(p_queue_id, NUM_LOD_QUEUES);
	data.queues[p_queue_id].lods.push_back(p_lod);
}

void LODManager::unregister_lod(LOD *p_lod, uint32_t p_queue_id) {
	ERR_FAIL_UNSIGNED_INDEX(p_queue_id, NUM_LOD_QUEUES);
	data.queues[p_queue_id].lods.erase(p_lod);
}

void LODManager::_update_queue(uint32_t p_queue_id, const Vector3 *p_camera_positions, uint32_t p_num_cameras) {
	// Some local aliases.
	Queue &queue = data.queues[p_queue_id];
	LocalVector<LOD *> &lods = queue.lods;

	uint32_t total_lods = lods.size();

	if (!total_lods) {
		return;
	}

	// Wraparound.
	queue.lod_iterator %= total_lods;

	uint32_t first_lod = queue.lod_iterator;

	uint32_t num_lods_to_check = 1;
	uint32_t num_lods_to_change = 1;

	switch (p_queue_id) {
		case 4: {
			num_lods_to_check = 3125;
			num_lods_to_change = 32;
		} break;
		case 3: {
			num_lods_to_check = 256;
			num_lods_to_change = 8;
		} break;
		case 2: {
			num_lods_to_check = 27;
			num_lods_to_change = 4;
		} break;
		case 1: {
			num_lods_to_check = 4;
			num_lods_to_change = 1;
		} break;
		default: {
		} break;
	}

	// No point updating more lods than the total.
	num_lods_to_check = MIN(num_lods_to_check, total_lods);
	num_lods_to_change = MIN(num_lods_to_change, total_lods);

	// Find minimum distances to cameras...
	uint32_t changed = 0;

	for (uint32_t l = 0; l < num_lods_to_check; l++) {
		uint32_t lod_id = (first_lod + l) % total_lods;
		LOD *lod = lods[lod_id];
		Vector3 lod_pos = lod->get_global_translation();

		float min_dist = FLT_MAX;
		for (uint32_t c = 0; c < p_num_cameras; c++) {
			float dist = (lod_pos - p_camera_positions[c]).length_squared();
			min_dist = MIN(min_dist, dist);
		}

		if (lod->_lod_update(min_dist)) {
			changed++;
			if (changed >= num_lods_to_change) {
				// Only update the iterator to where we got to.
				num_lods_to_check = l + 1;
				break;
			}
		}
	}

	queue.lod_iterator += num_lods_to_check;
}

void LODManager::notify_saving(bool p_active) {
	// When saving in the editor, to prevent file delta due to
	// different visibilities from LOD childs, we standardize
	// to showing the first LOD.
	MutexLock lock(data.mutex);
	data.saving = p_active;

	if (p_active) {
		for (uint32_t n = 0; n < NUM_LOD_QUEUES; n++) {
			Queue &queue = data.queues[n];
			LocalVector<LOD *> &lods = queue.lods;

			for (uint32_t l = 0; l < lods.size(); l++) {
				lods[l]->_lod_pre_save();
			}
		}
	}
}

void LODManager::update() {
	if (!_enabled) {
		return;
	}

	MutexLock lock(data.mutex);

	// We don't want to change the visibilities while saving from the editor.
	if (data.saving) {
		return;
	}

	// Get all camera positions
	// Reserve enough for all cameras, some may not be used.
	Vector3 *camera_positions = (Vector3 *)alloca(data.cameras.size() * sizeof(Vector3));

	uint32_t num_cameras = 0;
	for (uint32_t c = 0; c < data.cameras.size(); c++) {
		const Camera *camera = data.cameras[c];

		// Ignore ortho cameras for LOD.
		if (!camera->get_affect_lod() || !camera->is_current() || (camera->get_projection() == Camera::PROJECTION_ORTHOGONAL)) {
			continue;
		}

		camera_positions[num_cameras++] = camera->get_global_transform().origin;
	}

	if (!num_cameras) {
		return;
	}

	for (uint32_t n = 0; n < NUM_LOD_QUEUES; n++) {
		_update_queue(n, camera_positions, num_cameras);
	}
}
