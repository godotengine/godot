/**************************************************************************/
/*  mesh_rasterizer.h                                                     */
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

#include "servers/rendering_server.h"

class MeshRasterizer {
private:
	static MeshRasterizer *singleton;

public:
	virtual RID mesh_rasterizer_allocate() = 0;
	virtual void mesh_rasterizer_initialize(RID p_mesh_rasterizer, int p_width, int p_height, RS::RasterizedTextureFormat p_texture_format, bool p_generate_mipmaps, RD::TextureSamples p_samples) = 0;
	virtual void mesh_rasterizer_set_mesh(RID p_mesh_rasterizer, RID p_mesh, int p_surface_index) = 0;
	virtual void mesh_rasterizer_draw(RID p_mesh_rasterizer, RID p_material, const Color &p_bg_color) = 0;
	virtual RID mesh_rasterizer_get_texture(RID p_mesh_rasterizer) = 0;
	virtual bool free(RID p_mesh_rasterizer) = 0;

	static MeshRasterizer *get_singleton() { return singleton; }
	MeshRasterizer();
	virtual ~MeshRasterizer();
};
