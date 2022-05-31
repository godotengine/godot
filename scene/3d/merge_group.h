/**************************************************************************/
/*  merge_group.h                                                         */
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

#ifndef MERGE_GROUP_H
#define MERGE_GROUP_H

#include "spatial.h"

class MeshInstance;
class CSGShape;
class GridMap;

class MergeGroup : public Spatial {
	GDCLASS(MergeGroup, Spatial);
	friend class MergeGroupEditorPlugin;

public:
	enum ParamEnabled {
		PARAM_ENABLED_AUTO_MERGE,
		PARAM_ENABLED_SHADOW_PROXY,
		PARAM_ENABLED_CONVERT_CSGS,
		PARAM_ENABLED_CONVERT_GRIDMAPS,
		PARAM_ENABLED_COMBINE_SURFACES,
		PARAM_ENABLED_CLEAN_MESHES,
		PARAM_ENABLED_MAX,
	};

	enum Param {
		PARAM_GROUP_SIZE,
		PARAM_SPLITS_HORIZONTAL,
		PARAM_SPLITS_VERTICAL,
		PARAM_MIN_SPLIT_POLY_COUNT,
		PARAM_MAX,
	};

	void merge_meshes();
	bool merge_meshes_in_editor();

	void set_param_enabled(ParamEnabled p_param, bool p_enabled);
	bool get_param_enabled(ParamEnabled p_param);

	void set_param(Param p_param, int p_value);
	int get_param(Param p_param);

	// These enable feedback in the Godot UI as we bake.
	typedef bool (*BakeStepFunc)(float, const String &, void *, bool); // Progress, step description, userdata, force refresh.
	typedef void (*BakeEndFunc)(uint32_t); // time_started

	static BakeStepFunc bake_step_function;
	static BakeStepFunc bake_substep_function;
	static BakeEndFunc bake_end_function;

protected:
	static void _bind_methods();
	void _notification(int p_what);

private:
	struct MeshAABB {
		MeshInstance *mi = nullptr;
		AABB aabb;
		static int _sort_axis;
		bool operator<(const MeshAABB &p_b) const {
			real_t a_min = aabb.position.coord[_sort_axis];
			real_t b_min = p_b.aabb.position.coord[_sort_axis];
			return a_min < b_min;
		}
	};

	// Main function.
	bool _merge_meshes();

	// Merging.
	void _find_mesh_instances_recursive(int p_depth, Node *p_node, LocalVector<MeshInstance *> &r_mis, bool p_shadows, bool p_flag_invalid_meshes = false);
	bool _merge_similar(LocalVector<MeshInstance *> &r_mis, bool p_shadows);
	void _merge_list(const LocalVector<MeshInstance *> &p_mis, bool p_shadows, int p_whittle_group = -1);
	void _merge_list_ex(const LocalVector<MeshAABB> &p_mesh_aabbs, bool p_shadows, int p_whittle_group = -1);
	bool _join_similar(LocalVector<MeshInstance *> &r_mis);
	void _split_mesh_by_surface(MeshInstance *p_mi, int p_num_surfaces);
	bool _split_by_locality();

	// Helper funcs.
	void _convert_source_to_spatial(Spatial *p_node);
	void _reset_mesh_instance(MeshInstance *p_mi);
	void _move_children(Node *p_from, Node *p_to, bool p_recalculate_transforms = false);
	void _recursive_tree_merge(int &r_whittle_group, LocalVector<MeshAABB> p_list);
	void _delete_node(Node *p_node);
	bool _node_ok_to_delete(Node *p_node);
	void _cleanup_source_meshes(LocalVector<MeshInstance *> &r_cleanup_list);
	void _delete_dangling_spatials(Node *p_node);

	void _node_changed(Node *p_node);
	void _node_changed_internal(Node *p_node);

	// CSG
	void _find_csg_recursive(int p_depth, Node *p_node, LocalVector<CSGShape *> &r_csgs);
	void _split_csg_by_surface(CSGShape *p_shape);

	bool _terminate_search(Node *p_node);

	// Gridmap
	void _find_gridmap_recursive(int p_depth, Node *p_node, LocalVector<GridMap *> &r_gridmaps);
	void _bake_gridmap(GridMap *p_gridmap);

	void _log(String p_string);
	void _logt(int p_tabs, String p_string);

	struct Data {
		bool params_enabled[PARAM_ENABLED_MAX];
		uint32_t params[PARAM_MAX];

		// Hidden Params.
		bool delete_sources = false;
		bool convert_sources = false;
		bool split_by_surface = false;

		// Each merge is an iteration.
		uint32_t iteration = 0;
		Node *scene_root = nullptr;

		Data();
	} data;
};

VARIANT_ENUM_CAST(MergeGroup::Param);
VARIANT_ENUM_CAST(MergeGroup::ParamEnabled);

#endif // MERGE_GROUP_H
