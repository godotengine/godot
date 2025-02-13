/**************************************************************************/
/*  rendering_server_globals.cpp                                          */
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

#include "rendering_server_globals.h"

bool RenderingServerGlobals::threaded = false;

RendererUtilities *RenderingServerGlobals::utilities = nullptr;
RendererLightStorage *RenderingServerGlobals::light_storage = nullptr;
RendererMaterialStorage *RenderingServerGlobals::material_storage = nullptr;
RendererMeshStorage *RenderingServerGlobals::mesh_storage = nullptr;
RendererParticlesStorage *RenderingServerGlobals::particles_storage = nullptr;
RendererTextureStorage *RenderingServerGlobals::texture_storage = nullptr;
RendererGI *RenderingServerGlobals::gi = nullptr;
RendererFog *RenderingServerGlobals::fog = nullptr;
RendererCameraAttributes *RenderingServerGlobals::camera_attributes = nullptr;
RendererCanvasRender *RenderingServerGlobals::canvas_render = nullptr;
RendererCompositor *RenderingServerGlobals::rasterizer = nullptr;

RendererCanvasCull *RenderingServerGlobals::canvas = nullptr;
RendererViewport *RenderingServerGlobals::viewport = nullptr;
RenderingMethod *RenderingServerGlobals::scene = nullptr;
