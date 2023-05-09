/**************************************************************************/
/*  openxr_composition_layer.h                                            */
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

#ifndef OPENXR_COMPOSITION_LAYER_H
#define OPENXR_COMPOSITION_LAYER_H

#include "scene/main/viewport.h"

class OpenXRAPI;
class ViewportCompositionLayerProvider;

class OpenXRCompositionLayer : public SubViewport {
	GDCLASS(OpenXRCompositionLayer, SubViewport);

public:
	enum CompositionLayerTypes {
		COMPOSITION_LAYER_EQUIRECT2,
		COMPOSITION_LAYER_MAX
	};

private:
	OpenXRAPI *openxr_api = nullptr;
	ViewportCompositionLayerProvider *openxr_layer_provider = nullptr;

	CompositionLayerTypes composition_layer_type;

protected:
	static void _bind_methods();

public:
	OpenXRCompositionLayer();
	~OpenXRCompositionLayer();

	void set_composition_layer_type(const CompositionLayerTypes p_type);
	CompositionLayerTypes get_composition_layer_type() const { return composition_layer_type; };

	bool is_supported();

	void _notification(int p_what);
};

VARIANT_ENUM_CAST(OpenXRCompositionLayer::CompositionLayerTypes)

#endif // OPENXR_COMPOSITION_LAYER_H
