/*************************************************************************/
/*  glue_header.h                                                        */
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

#ifdef MONO_GLUE_ENABLED

#include "../mono_gd/gd_mono_marshal.h"

void godot_register_collections_icalls();
void godot_register_gd_icalls();
void godot_register_string_name_icalls();
void godot_register_nodepath_icalls();
void godot_register_object_icalls();
void godot_register_rid_icalls();
void godot_register_string_icalls();
void godot_register_scene_tree_icalls();

/**
 * Registers internal calls that were not generated. This function is called
 * from the generated GodotSharpBindings::register_generated_icalls() function.
 */
void godot_register_glue_header_icalls() {
	godot_register_collections_icalls();
	godot_register_gd_icalls();
	godot_register_string_name_icalls();
	godot_register_nodepath_icalls();
	godot_register_object_icalls();
	godot_register_rid_icalls();
	godot_register_string_icalls();
	godot_register_scene_tree_icalls();
}

// Used by the generated glue

#include "core/array.h"
#include "core/class_db.h"
#include "core/dictionary.h"
#include "core/engine.h"
#include "core/method_bind.h"
#include "core/node_path.h"
#include "core/reference.h"
#include "core/typedefs.h"
#include "core/ustring.h"

#include "../mono_gd/gd_mono_class.h"
#include "../mono_gd/gd_mono_internals.h"
#include "../mono_gd/gd_mono_utils.h"

#define GODOTSHARP_INSTANCE_OBJECT(m_instance, m_type) \
	static ClassDB::ClassInfo *ci = nullptr;           \
	if (!ci) {                                         \
		ci = ClassDB::classes.getptr(m_type);          \
	}                                                  \
	Object *m_instance = ci->creation_func();

#include "arguments_vector.h"

#endif // MONO_GLUE_ENABLED
