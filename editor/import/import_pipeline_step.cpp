/**************************************************************************/
/*  import_pipeline_step.cpp                                              */
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

#include "import_pipeline_step.h"

#include "core/error/error_macros.h"
#include "core/io/resource_saver.h"
#include "editor/editor_node.h"
#include "scene/3d/area_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/navigation_region_3d.h"
#include "scene/3d/occluder_instance_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/vehicle_body_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/box_shape_3d.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/resource_format_text.h"
#include "scene/resources/separation_ray_shape_3d.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/world_boundary_shape_3d.h"

void ImportPipelineStep::_bind_methods() {
	GDVIRTUAL_BIND(_update);
	GDVIRTUAL_BIND(_source_changed);
	GDVIRTUAL_BIND(_get_inputs);
	GDVIRTUAL_BIND(_get_outputs);
	GDVIRTUAL_BIND(_get_tree);

	ADD_SIGNAL(MethodInfo("name_changed", PropertyInfo(Variant::STRING, "name")));
}

void ImportPipelineStep::update() {
	GDVIRTUAL_CALL(_update);
}

void ImportPipelineStep::source_changed() {
	GDVIRTUAL_CALL(_source_changed);
}

PackedStringArray ImportPipelineStep::get_inputs() {
	PackedStringArray ret;
	if (GDVIRTUAL_CALL(_get_inputs, ret)) {
		return ret;
	}
	return PackedStringArray();
}

PackedStringArray ImportPipelineStep::get_outputs() {
	PackedStringArray ret;
	if (GDVIRTUAL_CALL(_get_outputs, ret)) {
		return ret;
	}
	return PackedStringArray();
}

Node *ImportPipelineStep::get_tree() {
	Node *ret;
	if (GDVIRTUAL_CALL(_get_tree, ret)) {
		return ret;
	}
	return nullptr;
}
