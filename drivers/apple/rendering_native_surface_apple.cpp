/**************************************************************************/
/*  rendering_native_surface_apple.cpp                                    */
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

#include "rendering_native_surface_apple.h"
#include "drivers/vulkan/rendering_context_driver_vulkan_moltenvk.h"

void RenderingNativeSurfaceApple::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceApple", D_METHOD("create", "layer"), &RenderingNativeSurfaceApple::create_api);
}

Ref<RenderingNativeSurfaceApple> RenderingNativeSurfaceApple::create_api(/* GDExtensionConstPtr<const void> */ uint64_t p_layer) {
	return RenderingNativeSurfaceApple::create((void *)p_layer /* .operator const void *() */);
}

Ref<RenderingNativeSurfaceApple> RenderingNativeSurfaceApple::create(void *p_layer) {
	Ref<RenderingNativeSurfaceApple> result = memnew(RenderingNativeSurfaceApple);
	result->layer = p_layer;
	return result;
}

RenderingContextDriver *RenderingNativeSurfaceApple::create_rendering_context() {
#ifdef __APPLE__
	return memnew(RenderingContextDriverVulkanMoltenVk);
#else
	return nullptr;
#endif
}

RenderingNativeSurfaceApple::RenderingNativeSurfaceApple() {
	// Does nothing.
}

RenderingNativeSurfaceApple::~RenderingNativeSurfaceApple() {
	// Does nothing.
}
