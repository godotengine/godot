/**************************************************************************/
/*  openxr_htc_passthrough_extension.h                                    */
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

#ifndef OPENXR_HTC_PASSTHROUGH_EXTENSION_H
#define OPENXR_HTC_PASSTHROUGH_EXTENSION_H

#include "../openxr_api.h"
#include "../util.h"
#include "openxr_composition_layer_provider.h"
#include "openxr_extension_wrapper.h"

#include <map>

class Viewport;

// Wrapper for the HTC XR passthrough extensions (extension #318).
class OpenXRHTCPassthroughExtension : public OpenXRExtensionWrapper, public OpenXRCompositionLayerProvider {
public:
	OpenXRHTCPassthroughExtension();
	~OpenXRHTCPassthroughExtension();

	virtual HashMap<String, bool *> get_requested_extensions() override;

	void on_instance_created(const XrInstance instance) override;

	void on_session_created(const XrSession session) override;

	void on_session_destroyed() override;

	void on_instance_destroyed() override;

	XrCompositionLayerBaseHeader *get_composition_layer() override;

	bool is_passthrough_supported() {
		return htc_passthrough_ext;
	}

	bool is_passthrough_enabled();

	bool start_passthrough();

	void stop_passthrough();

	// set alpha of passthrough
	void set_alpha(float alpha);

	// get alpha of passthrough
	float get_alpha() const;

	static OpenXRHTCPassthroughExtension *get_singleton();

private:
	// Create a passthrough feature
	EXT_PROTO_XRRESULT_FUNC3(xrCreatePassthroughHTC,
			(XrSession), session,
			(const XrPassthroughCreateInfoHTC *), create_info,
			(XrPassthroughHTC *), feature_out)

	// Destroy a previously created passthrough feature
	EXT_PROTO_XRRESULT_FUNC1(xrDestroyPassthroughHTC, (XrPassthroughHTC), feature)

	bool initialize_htc_passthrough_extension(const XrInstance instance);

	void cleanup();

	Viewport *get_main_viewport();

	static OpenXRHTCPassthroughExtension *singleton;

	bool htc_passthrough_ext = false; // required for any passthrough functionality

	XrPassthroughHTC passthrough_handle = XR_NULL_HANDLE; // handle for passthrough

	XrCompositionLayerPassthroughHTC composition_passthrough_layer = {
		XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_HTC, // XrStructureType
		nullptr, // next
		XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT, //  XrCompositionLayerFlags
		XR_NULL_HANDLE, // XrSpace
		XR_NULL_HANDLE, // XrPassthroughHTC
		{
				// XrPassthroughColorHTC
				XR_TYPE_PASSTHROUGH_COLOR_HTC, // XrStructureType
				nullptr, // next
				1.0f // alpha (preset to opaque)
		}
	};
};

#endif // OPENXR_HTC_PASSTHROUGH_EXTENSION_H
