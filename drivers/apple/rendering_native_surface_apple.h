/**************************************************************************/
/*  rendering_native_surface_apple.h                                      */
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

#ifndef RENDERING_NATIVE_SURFACE_APPLE_H
#define RENDERING_NATIVE_SURFACE_APPLE_H

#include "core/variant/native_ptr.h"
#include "servers/rendering/rendering_native_surface.h"

class RenderingNativeSurfaceApple : public RenderingNativeSurface {
	GDCLASS(RenderingNativeSurfaceApple, RenderingNativeSurface);

	static void _bind_methods();

	void *layer = nullptr;

public:
	// TODO: Remove workaround when SwiftGodot starts to support const void * arguments.
	static Ref<RenderingNativeSurfaceApple> create_api(/* GDExtensionConstPtr<const void> */ uint64_t p_layer);

	static Ref<RenderingNativeSurfaceApple> create(void *p_layer);

	void *get_metal_layer() const {
		return layer;
	};

	RenderingContextDriver *create_rendering_context() override;

	RenderingNativeSurfaceApple();
	~RenderingNativeSurfaceApple();
};

#endif // RENDERING_NATIVE_SURFACE_APPLE_H
