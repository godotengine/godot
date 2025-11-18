/**************************************************************************/
/*  rendering_server_globals.h                                            */
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

#include "servers/rendering/environment/renderer_fog.h"
#include "servers/rendering/environment/renderer_gi.h"
#include "servers/rendering/renderer_canvas_cull.h"
#include "servers/rendering/renderer_canvas_render.h"
#include "servers/rendering/rendering_method.h"
#include "servers/rendering/storage/camera_attributes_storage.h"
#include "servers/rendering/storage/light_storage.h"
#include "servers/rendering/storage/material_storage.h"
#include "servers/rendering/storage/mesh_storage.h"
#include "servers/rendering/storage/particles_storage.h"
#include "servers/rendering/storage/texture_storage.h"
#include "servers/rendering/storage/utilities.h"

class RendererCanvasCull;
class RendererViewport;
class RenderingMethod;

class RenderingServerGlobals {
public:
	static inline bool threaded = false;

	static inline RendererUtilities *utilities = nullptr;
	static inline RendererLightStorage *light_storage = nullptr;
	static inline RendererMaterialStorage *material_storage = nullptr;
	static inline RendererMeshStorage *mesh_storage = nullptr;
	static inline RendererParticlesStorage *particles_storage = nullptr;
	static inline RendererTextureStorage *texture_storage = nullptr;
	static inline RendererGI *gi = nullptr;
	static inline RendererFog *fog = nullptr;
	static inline RendererCameraAttributes *camera_attributes = nullptr;
	static inline RendererCanvasRender *canvas_render = nullptr;
	static inline RendererCompositor *rasterizer = nullptr;

	static inline RendererCanvasCull *canvas = nullptr;
	static inline RendererViewport *viewport = nullptr;
	static inline RenderingMethod *scene = nullptr;
};

#define RSG RenderingServerGlobals
