/*************************************************************************/
/*  resource_editor_plugin.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "resource_editor_plugin.h"

bool EditorInspectorPluginResource::can_handle(Object *p_object) {

	return Object::cast_to<Curve>(p_object) != NULL;
}

void EditorInspectorPluginResource::parse_begin(Object *p_object) {

	//Curve *curve = Object::cast_to<Curve>(p_object);
	//ERR_FAIL_COND(!curve);
	//Ref<Curve> c(curve);

	//ResourceEditor *editor = memnew(ResourceEditor);
	//editor->set_curve(curve);
	//add_custom_control(editor);
}

ResourceEditorPlugin::ResourceEditorPlugin(EditorNode *p_node) {
	Ref<EditorInspectorPluginResource> resource_plugin;
	resource_plugin.instance();
	EditorInspector::add_inspector_plugin(resource_plugin);

	get_editor_interface()->get_resource_previewer()->add_preview_generator(memnew(ResourcePreviewGenerator));
}

//-----------------------------------
// Preview generator

bool ResourcePreviewGenerator::handles(const String &p_type) const {
	return p_type == "Resource";
}

Ref<Texture> ResourcePreviewGenerator::generate(const Ref<Resource> &p_from, const Size2 p_size) const {

	Ref<Curve> curve_ref = p_from;
	ERR_FAIL_COND_V(curve_ref.is_null(), Ref<Texture>());
	Curve &curve = **curve_ref;

	// FIXME: Should be ported to use p_size as done in b2633a97
	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	Ref<Image> img_ref;
	img_ref.instance();
	Image &im = **img_ref;

	im.create(thumbnail_size, thumbnail_size / 2, 0, Image::FORMAT_RGBA8);

	im.lock();

	Color bg_color(0.1, 0.1, 0.1, 1.0);
	for (int i = 0; i < thumbnail_size; i++) {
		for (int j = 0; j < thumbnail_size / 2; j++) {
			im.set_pixel(i, j, bg_color);
		}
	}

	Color line_color(0.8, 0.8, 0.8, 1.0);
	float range_y = curve.get_max_value() - curve.get_min_value();

	int prev_y = 0;
	for (int x = 0; x < im.get_width(); ++x) {

		float t = static_cast<float>(x) / im.get_width();
		float v = (curve.interpolate_baked(t) - curve.get_min_value()) / range_y;
		int y = CLAMP(im.get_height() - v * im.get_height(), 0, im.get_height());

		// Plot point
		if (y >= 0 && y < im.get_height()) {
			im.set_pixel(x, y, line_color);
		}

		// Plot vertical line to fix discontinuity (not 100% correct but enough for a preview)
		if (x != 0 && Math::abs(y - prev_y) > 1) {
			int y0, y1;
			if (y < prev_y) {
				y0 = y;
				y1 = prev_y;
			} else {
				y0 = prev_y;
				y1 = y;
			}
			for (int ly = y0; ly < y1; ++ly) {
				im.set_pixel(x, ly, line_color);
			}
		}

		prev_y = y;
	}

	im.unlock();

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img_ref, 0);
	return ptex;
}
