/*************************************************************************/
/*  scene_tree_glue.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef MONO_GLUE_ENABLED

#include "core/object/class_db.h"
#include "core/string/string_name.h"
#include "core/variant/array.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"

#include "../csharp_script.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../mono_gd/gd_mono_utils.h"

Array *godot_icall_SceneTree_get_nodes_in_group_Generic(SceneTree *ptr, StringName *group, MonoReflectionType *refltype) {
	List<Node *> nodes;
	Array ret;

	// Retrieve all the nodes in the group
	ptr->get_nodes_in_group(*group, &nodes);

	// No need to bother if the group is empty
	if (!nodes.is_empty()) {
		MonoType *elem_type = mono_reflection_type_get_type(refltype);
		MonoClass *mono_class = mono_class_from_mono_type(elem_type);
		GDMonoClass *klass = GDMono::get_singleton()->get_class(mono_class);

		if (klass == GDMonoUtils::get_class_native_base(klass)) {
			// If we're trying to get native objects, just check the inheritance list
			StringName native_class_name = GDMonoUtils::get_native_godot_class_name(klass);
			for (int i = 0; i < nodes.size(); ++i) {
				if (ClassDB::is_parent_class(nodes[i]->get_class(), native_class_name)) {
					ret.push_back(nodes[i]);
				}
			}
		} else {
			// If we're trying to get csharpscript instances, get the mono object and compare the classes
			for (int i = 0; i < nodes.size(); ++i) {
				CSharpInstance *si = CAST_CSHARP_INSTANCE(nodes[i]->get_script_instance());

				if (si != nullptr) {
					MonoObject *obj = si->get_mono_object();
					if (obj != nullptr && mono_object_get_class(obj) == mono_class) {
						ret.push_back(nodes[i]);
					}
				}
			}
		}
	}

	return memnew(Array(ret));
}

void godot_register_scene_tree_icalls() {
	GDMonoUtils::add_internal_call("Godot.SceneTree::godot_icall_SceneTree_get_nodes_in_group_Generic", godot_icall_SceneTree_get_nodes_in_group_Generic);
}

#endif // MONO_GLUE_ENABLED
