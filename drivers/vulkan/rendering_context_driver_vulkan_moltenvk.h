/**************************************************************************/
/*  rendering_context_driver_vulkan_moltenvk.h                            */
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

#ifndef RENDERING_CONTEXT_DRIVER_VULKAN_MOLTENVK_H
#define RENDERING_CONTEXT_DRIVER_VULKAN_MOLTENVK_H

#ifdef __APPLE__
#ifdef VULKAN_ENABLED

#include "drivers/vulkan/rendering_context_driver_vulkan.h"

#ifdef __OBJC__
@class CAMetalLayer;
#else
typedef void CAMetalLayer;
#endif

class RenderingContextDriverVulkanMoltenVk : public RenderingContextDriverVulkan {
private:
	virtual const char *_get_platform_surface_extension() const override final;

protected:
	SurfaceID surface_create(Ref<RenderingNativeSurface> p_native_surface) override final;

public:
	RenderingContextDriverVulkanMoltenVk();
	~RenderingContextDriverVulkanMoltenVk();
};

#endif // VULKAN_ENABLED
#endif // __APPLE__

#endif // RENDERING_CONTEXT_DRIVER_VULKAN_MOLTENVK_H
