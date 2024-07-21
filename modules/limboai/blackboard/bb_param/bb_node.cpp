/**
 * bb_node.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bb_node.h"

#ifdef LIMBOAI_MODULE
#include "core/error/error_macros.h"
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#endif // LIMBOAI_GDEXTENSION

Variant BBNode::get_value(Node *p_scene_root, const Ref<Blackboard> &p_blackboard, const Variant &p_default) {
	ERR_FAIL_NULL_V_MSG(p_scene_root, Variant(), "BBNode: get_value() failed - scene_root is null.");
	ERR_FAIL_NULL_V_MSG(p_blackboard, Variant(), "BBNode: get_value() failed - blackboard is null.");

	Variant val;
	if (get_value_source() == SAVED_VALUE) {
		val = get_saved_value();
	} else {
		val = p_blackboard->get_var(get_variable(), p_default);
	}

	if (val.get_type() == Variant::NODE_PATH) {
		return p_scene_root->get_node_or_null(val);
	} else {
		Object *obj = val;
		if (unlikely(obj == nullptr && val.get_type() != Variant::NIL)) {
			WARN_PRINT("BBNode: Unexpected variant type of a blackboard variable.");
			return p_default;
		} else {
			return obj;
		}
	}
}
