/**************************************************************************/
/*  rasterizer_dummy.cpp                                                  */
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

#include "rasterizer_dummy.h"

#include "servers/display/display_server.h"
#include "servers/rendering/dummy/environment/fog.h"
#include "servers/rendering/dummy/environment/gi.h"
#include "servers/rendering/dummy/rasterizer_canvas_dummy.h"
#include "servers/rendering/dummy/rasterizer_scene_dummy.h"
#include "servers/rendering/dummy/storage/light_storage.h"
#include "servers/rendering/dummy/storage/material_storage.h"
#include "servers/rendering/dummy/storage/mesh_storage.h"
#include "servers/rendering/dummy/storage/particles_storage.h"
#include "servers/rendering/dummy/storage/texture_storage.h"
#include "servers/rendering/dummy/storage/utilities.h"

void RasterizerDummy::end_frame(bool p_present) {
	if (p_present) {
		DisplayServer::get_singleton()->swap_buffers();
	}
}

RendererCanvasRender *RasterizerDummy::get_canvas() {
	return canvas;
}
RendererSceneRender *RasterizerDummy::get_scene() {
	return scene;
}

RendererFog *RasterizerDummy::get_fog() {
	return fog;
}
RendererGI *RasterizerDummy::get_gi() {
	return gi;
}
RendererLightStorage *RasterizerDummy::get_light_storage() {
	return light_storage;
}
RendererMaterialStorage *RasterizerDummy::get_material_storage() {
	return material_storage;
}
RendererMeshStorage *RasterizerDummy::get_mesh_storage() {
	return mesh_storage;
}
RendererParticlesStorage *RasterizerDummy::get_particles_storage() {
	return particles_storage;
}
RendererTextureStorage *RasterizerDummy::get_texture_storage() {
	return texture_storage;
}
RendererUtilities *RasterizerDummy::get_utilities() {
	return utilities;
}

RasterizerDummy::RasterizerDummy() {
	canvas = memnew(RasterizerCanvasDummy);
	scene = memnew(RasterizerSceneDummy);

	fog = memnew(RendererDummy::Fog);
	gi = memnew(RendererDummy::GI);
	light_storage = memnew(RendererDummy::LightStorage);
	material_storage = memnew(RendererDummy::MaterialStorage);
	mesh_storage = memnew(RendererDummy::MeshStorage);
	particles_storage = memnew(RendererDummy::ParticlesStorage);
	texture_storage = memnew(RendererDummy::TextureStorage);
	utilities = memnew(RendererDummy::Utilities);
}

RasterizerDummy::~RasterizerDummy() {
	memdelete(canvas);
	memdelete(scene);

	memdelete(fog);
	memdelete(gi);
	memdelete(light_storage);
	memdelete(material_storage);
	memdelete(mesh_storage);
	memdelete(particles_storage);
	memdelete(texture_storage);
	memdelete(utilities);
}
