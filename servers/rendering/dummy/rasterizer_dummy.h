/**************************************************************************/
/*  rasterizer_dummy.h                                                    */
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

#pragma once

#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/rendering_server_enums.h"

class RasterizerCanvasDummy;
class RasterizerSceneDummy;

namespace RendererDummy {
class Fog;
class GI;
class LightStorage;
class MaterialStorage;
class MeshStorage;
class ParticlesStorage;
class TextureStorage;
class Utilities;
} //namespace RendererDummy

class RasterizerDummy : public RendererCompositor {
private:
	uint64_t frame = 1;
	double delta = 0;
	double time = 0.0;

protected:
	RasterizerCanvasDummy *canvas = nullptr;
	RasterizerSceneDummy *scene = nullptr;

	RendererDummy::Fog *fog = nullptr;
	RendererDummy::GI *gi = nullptr;
	RendererDummy::LightStorage *light_storage = nullptr;
	RendererDummy::MaterialStorage *material_storage = nullptr;
	RendererDummy::MeshStorage *mesh_storage = nullptr;
	RendererDummy::ParticlesStorage *particles_storage = nullptr;
	RendererDummy::TextureStorage *texture_storage = nullptr;
	RendererDummy::Utilities *utilities = nullptr;

public:
	RendererCanvasRender *get_canvas() override;
	RendererSceneRender *get_scene() override;

	RendererFog *get_fog() override;
	RendererGI *get_gi() override;
	RendererLightStorage *get_light_storage() override;
	RendererMaterialStorage *get_material_storage() override;
	RendererMeshStorage *get_mesh_storage() override;
	RendererParticlesStorage *get_particles_storage() override;
	RendererTextureStorage *get_texture_storage() override;
	RendererUtilities *get_utilities() override;

	void set_boot_image_with_stretch(const Ref<Image> &p_image, const Color &p_color, RSE::SplashStretchMode p_stretch_mode, bool p_use_filter = true) override {}

	void initialize() override {}
	void begin_frame(double frame_step) override {
		frame++;
		delta = frame_step;
		time += frame_step;
	}

	void blit_render_targets_to_screen(int p_screen, const RenderingServerTypes::BlitToScreen *p_render_targets, int p_amount) override {}

	bool is_opengl() override { return false; }
	void gl_end_frame(bool p_swap_buffers) override {}

	void end_frame(bool p_present) override;

	void finalize() override {}

	static RendererCompositor *_create_current() {
		return memnew(RasterizerDummy);
	}

	static void make_current() {
		_create_func = _create_current;
		low_end = false;
	}

	uint64_t get_frame_number() const override { return frame; }
	double get_frame_delta_time() const override { return delta; }
	double get_total_time() const override { return time; }
	bool can_create_resources_async() const override { return false; }

	RasterizerDummy();
	~RasterizerDummy();
};
