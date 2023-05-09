/**************************************************************************/
/*  openxr_composition_layer_extension.h                                  */
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

#ifndef OPENXR_COMPOSITION_LAYER_EXTENSION_H
#define OPENXR_COMPOSITION_LAYER_EXTENSION_H

#include "openxr_composition_layer_provider.h"
#include "openxr_extension_wrapper.h"

#include "../openxr_api.h"

// This extension provides access to composition layers for displaying 2D content through the XR compositor.

// OpenXRCompositionLayerExtension enables the extensions related to this functionality
class OpenXRCompositionLayerExtension : public OpenXRExtensionWrapper {
public:
	enum CompositionLayerExtensions {
		COMPOSITION_LAYER_EQUIRECT_EXT,
		COMPOSITION_LAYER_EXT_MAX
	};

	static OpenXRCompositionLayerExtension *get_singleton();

	OpenXRCompositionLayerExtension();
	virtual ~OpenXRCompositionLayerExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions() override;
	bool is_available(CompositionLayerExtensions p_which);

private:
	static OpenXRCompositionLayerExtension *singleton;

	bool available[COMPOSITION_LAYER_EXT_MAX] = { false };
};

class ViewportCompositionLayerProvider : public OpenXRCompositionLayerProvider {
public:
	ViewportCompositionLayerProvider();
	virtual ~ViewportCompositionLayerProvider() override;

	bool is_supported();
	void setup_for_type(XrStructureType p_type);
	virtual OrderedCompositionLayer get_composition_layer() override;
	bool update_swapchain(uint32_t p_width, uint32_t p_height);
	void free_swapchain();
	RID get_image();

private:
	union {
		XrCompositionLayerBaseHeader composition_layer;
		XrCompositionLayerEquirect2KHR equirect_layer;
	};
	int sort_order = 1;

	OpenXRAPI *openxr_api = nullptr;
	OpenXRCompositionLayerExtension *composition_layer_extension = nullptr;

	uint32_t width = 0;
	uint32_t height = 0;
	OpenXRAPI::OpenXRSwapChainInfo swapchain_info;
};

#endif // OPENXR_COMPOSITION_LAYER_EXTENSION_H
