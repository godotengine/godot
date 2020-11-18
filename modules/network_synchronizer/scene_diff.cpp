/*************************************************************************/
/*  scene_diff.cpp                                                       */
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

#include "scene_diff.h"

#include "scene/main/node.h"
#include "scene_synchronizer.h"

void SceneDiff::_bind_methods() {
}

SceneDiff::SceneDiff() {
}

void SceneDiff::start_tracking_scene_changes(
		const LocalVector<NetUtility::NodeData *> &p_nodes) {

	start_tracking_count += 1;
	if (start_tracking_count > 1) {
		// Nothing to do, the tracking is already started.
		return;
	}

	tracking.resize(p_nodes.size());

	for (uint32_t i = 0; i < p_nodes.size(); i += 1) {
		if (
				p_nodes[i] == nullptr ||
				// Check if this is a controller.
				p_nodes[i]->is_controller ||
				p_nodes[i]->controlled_by != nullptr) {
			tracking[i].clear();
			continue;
		}

#ifdef DEBUG_ENABLED
		// This is never triggered because we always pass the `organized_node_data`
		// array.
		CRASH_COND(p_nodes[i]->id != i);
		// This is never triggered because when the node is invalid the node data
		// is destroyed.
		CRASH_COND(p_nodes[i]->node == nullptr);
#endif

		tracking[i].resize(p_nodes[i]->vars.size());

		for (uint32_t v = 0; v < p_nodes[i]->vars.size(); v += 1) {
			// Take the current variable value and store it.
			if (p_nodes[i]->vars[v].enabled && p_nodes[i]->vars[v].id != UINT32_MAX) {
				// Note: Taking the value using `get` so to take the most updated
				// value.
				tracking[i][v] = p_nodes[i]->node->get(p_nodes[i]->vars[v].var.name).duplicate(true);
			} else {
				tracking[i][v] = Variant();
			}
		}
	}
}

void SceneDiff::stop_tracking_scene_changes(const SceneSynchronizer *p_synchronizer) {
	ERR_FAIL_COND_MSG(
			start_tracking_count == 0,
			"The tracking is not yet started on this SceneDiff, so can't be end.");

	start_tracking_count -= 1;
	if (start_tracking_count > 0) {
		// Nothing to do, the tracking is still ongoing.
		return;
	}

	if (p_synchronizer->get_biggest_node_id() == UINT32_MAX) {
		// No nodes to track.
		tracking.clear();
		return;
	}

	if (tracking.size() > (p_synchronizer->get_biggest_node_id() + 1)) {
		NET_DEBUG_ERR("[BUG] The tracked nodes are exceeding the sync nodes. Probably the sync is different or it has reset?");
		tracking.clear();
		return;
	}

	if (diff.size() < tracking.size()) {
		// Make sure the diff has room to store the needed info.
		diff.resize(tracking.size());
	}

	for (NetNodeId i = 0; i < tracking.size(); i += 1) {
		const NetUtility::NodeData *nd = p_synchronizer->get_node_data(i);
		if (nd == nullptr) {
			continue;
		}

#ifdef DEBUG_ENABLED
		// This is never triggered because we always pass the `organized_node_data`
		// array.
		CRASH_COND(nd->id != i);
		// This is never triggered because when the node is invalid the node data
		// is destroyed.
		CRASH_COND(nd->node == nullptr);
#endif

		if (nd->vars.size() != tracking[i].size()) {
			// These two arrays are different because the node was null
			// during the start. So we can assume we are not tracking it.
			continue;
		}

		if (diff[i].size() < tracking[i].size()) {
			// Make sure the diff has room to store the variable info.
			diff[i].resize(tracking[i].size());
		}

		for (uint32_t v = 0; v < tracking[i].size(); v += 1) {
			if (nd->vars[v].id == UINT32_MAX || nd->vars[v].enabled == false) {
				continue;
			}

			// Take the current variable value.
			const Variant current_value =
					nd->node->get(nd->vars[v].var.name);

			// Compare the current value with the one taken during the start.
			if (p_synchronizer->synchronizer_variant_evaluation(
						tracking[i][v],
						current_value) == false) {

				diff[i][v].is_different = true;
				diff[i][v].value = current_value;
			}
		}
	}

	tracking.clear();
}

bool SceneDiff::is_tracking_in_progress() const {
	return start_tracking_count > 0;
}
