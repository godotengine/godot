/*************************************************************************/
/*  sorting_group_3d.h                                                   */
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

#ifndef SORTING_GROUP_3D_H
#define SORTING_GROUP_3D_H

#include "scene/3d/node_3d.h"
#include "visual_instance_3d.h"

class SortingGroup3D : public Node3D {
	GDCLASS(SortingGroup3D, Node3D);

	static bool dirty;
	static SortingGroup3D *manager_group;
	static List<SortingGroup3D *> root_groups;

	static void _add_root_group(SortingGroup3D *p_group);
	static void _remove_root_group(SortingGroup3D *p_group);

	static void _set_dirty(SortingGroup3D *p_group_changed);

	static void _update_group(SortingGroup3D *p_group, int8_t p_inherited_sort_order, uint32_t &r_index);

	bool sort_as_root = false;
	SortingGroup3D *parent_group = nullptr;
	List<SortingGroup3D *> child_groups;
	List<SortingGroup3D *>::Element *parent_entry = nullptr;

	List<VisualInstance3D *> visual_instances;

	// Start with 1, as 0 is considered outside groups
	uint32_t group_index = 1;

	int8_t sort_order = 0;
	int8_t inherited_sort_order = 0;

	void _set_group_index_and_inherited_order(uint32_t p_group_index, int8_t p_inherited_sort_order);

	List<SortingGroup3D *>::Element *_add_child_group(SortingGroup3D *p_group);
	void _remove_child_group(List<SortingGroup3D *>::Element *p_group);

	void _add_as_root_or_child();

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	static bool has_any();

	void update_all_groups();

	void add_visual_instance(VisualInstance3D *p_instance);
	void remove_visual_instance(VisualInstance3D *p_instance);

	void set_sort_as_root(bool p_sort_as_root);
	bool get_sort_as_root() const;

	void set_sort_order(int8_t p_sort_order);
	int8_t get_sort_order() const;
};

#endif // SORTING_GROUP_3D_H
