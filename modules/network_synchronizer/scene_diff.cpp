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
		const LocalVector<Ref<NetUtility::NodeData> > &p_nodes) {

	start_tracking_count += 1;
	if (start_tracking_count > 1) {
		// Nothing to do, the tracking is already started.
		return;
	}

	tracking.resize(p_nodes.size());

	for (uint32_t i = 0; i < p_nodes.size(); i += 1) {
		if (p_nodes[i]->node && p_nodes[i]->id > 0 && p_nodes[i]->valid) {
			tracking[i].node_data = p_nodes[i];
			tracking[i].variables.resize(p_nodes[i]->vars.size());
			const NetUtility::VarData *vars = p_nodes[i]->vars.ptr();

			for (int v = 0; v < p_nodes[i]->vars.size(); v += 1) {
				// Take the current variable value and store it.
				tracking[i].variables[v] = p_nodes[i]->node->get(vars[v].var.name).duplicate(true);
			}
		} else {
			tracking[i].variables.clear();
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

	for (uint32_t i = 0; i < tracking.size(); i += 1) {
		if (tracking[i].node_data->node && tracking[i].node_data->valid) {
#ifdef DEBUG_NEABLED
			// Nodes with 0 ID are skipt, so this cond is always false.
			CRASH_COND(tracking[i].node_data->id <= 0);
#endif

			if (tracking[i].variables.size() != uint32_t(tracking[i].node_data->vars.size())) {
				// These two arrays are different because the node was null
				// during the start. So we can assume we are not tracking it.
				continue;
			}

			NodeDiff *node_diff = diff.lookup_ptr(tracking[i].node_data->id);

			const NetUtility::VarData *vars = tracking[i].node_data->vars.ptr();
			for (int v = 0; v < tracking[i].node_data->vars.size(); v += 1) {
				if (vars[v].id <= 0 && vars[v].enabled == false) {
					continue;
				}

				// Take the current variable value and store it.
				const Variant current_value =
						tracking[i].node_data->node->get(vars[v].var.name);

				// Compare the current value with the one taken during the start.
				if (p_synchronizer->synchronizer_variant_evaluation(
							tracking[i].variables[v],
							current_value) == false) {

					if (node_diff == nullptr) {
						diff.insert(tracking[i].node_data->id, NodeDiff());
						node_diff = diff.lookup_ptr(tracking[i].node_data->id);
						node_diff->node_data = tracking[i].node_data;
					}

					node_diff->var_diff.set(
							vars[v].id,
							current_value);
				}
			}
		}
	}

	tracking.clear();
}

bool SceneDiff::is_tracking_in_progress() const {
	return start_tracking_count > 0;
}
