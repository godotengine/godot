/**************************************************************************/
/*  utilities.cpp                                                         */
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

#include "utilities.h"

#include "light_storage.h"
#include "material_storage.h"
#include "mesh_storage.h"
#include "texture_storage.h"

using namespace RendererDummy;

Utilities *Utilities::singleton = nullptr;

RS::InstanceType Utilities::get_base_type(RID p_rid) const {
	if (RendererDummy::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		return RS::INSTANCE_MESH;
	} else if (RendererDummy::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	} else if (RendererDummy::LightStorage::get_singleton()->owns_lightmap(p_rid)) {
		return RS::INSTANCE_LIGHTMAP;
	}
	return RS::INSTANCE_NONE;
}

bool Utilities::free(RID p_rid) {
	if (RendererDummy::LightStorage::get_singleton()->free(p_rid)) {
		return true;
	} else if (RendererDummy::TextureStorage::get_singleton()->owns_texture(p_rid)) {
		RendererDummy::TextureStorage::get_singleton()->texture_free(p_rid);
		return true;
	} else if (RendererDummy::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		RendererDummy::MeshStorage::get_singleton()->mesh_free(p_rid);
		return true;
	} else if (RendererDummy::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		RendererDummy::MeshStorage::get_singleton()->multimesh_free(p_rid);
		return true;
	} else if (RendererDummy::MaterialStorage::get_singleton()->owns_shader(p_rid)) {
		RendererDummy::MaterialStorage::get_singleton()->shader_free(p_rid);
		return true;
	} else if (RendererDummy::MaterialStorage::get_singleton()->owns_material(p_rid)) {
		RendererDummy::MaterialStorage::get_singleton()->material_free(p_rid);
		return true;
	}
	return false;
}

void Utilities::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (RendererDummy::MeshStorage::get_singleton()->owns_mesh(p_base)) {
		DummyMesh *mesh = RendererDummy::MeshStorage::get_singleton()->get_mesh(p_base);
		p_instance->update_dependency(&mesh->dependency);
	}
}

Utilities::Utilities() {
	singleton = this;
}

Utilities::~Utilities() {
	singleton = nullptr;
}
