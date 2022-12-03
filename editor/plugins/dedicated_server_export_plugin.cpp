/*************************************************************************/
/*  dedicated_server_export_plugin.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "dedicated_server_export_plugin.h"

#define EXPORT_OPTION_IS_DEDICATED_SERVER "dedicated_server/is_server"

void DedicatedServerExportPlugin::add_export_options(List<EditorExportPlatform::ExportOption> *r_options) {
	r_options->push_back(EditorExportPlatform::ExportOption(PropertyInfo(Variant::BOOL, EXPORT_OPTION_IS_DEDICATED_SERVER), false));
}

bool DedicatedServerExportPlugin::is_dedicated_server() const {
	Ref<EditorExportPreset> preset = get_export_preset();
	ERR_FAIL_COND_V(preset.is_null(), false);

	bool valid_prop = false;
	bool is_server = preset->get(EXPORT_OPTION_IS_DEDICATED_SERVER, &valid_prop);

	if (valid_prop) {
		return is_server;
	}

	return false;
}

PackedStringArray DedicatedServerExportPlugin::_get_export_features(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	PackedStringArray ret;
	if (is_dedicated_server()) {
		ret.append("dedicated_server");
	}
	return ret;
}

uint64_t DedicatedServerExportPlugin::_get_customization_configuration_hash() const {
	return (uint64_t)is_dedicated_server();
}

bool DedicatedServerExportPlugin::_begin_customize_resources(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) const {
	return is_dedicated_server();
}

Ref<Resource> DedicatedServerExportPlugin::_customize_resource(const Ref<Resource> &p_resource, const String &p_path) {
	if (p_resource->get_dedicated_server_export_type() == Resource::DEDICATED_SERVER_EXPORT_KEEP) {
		return Ref<Resource>();
	}

	if (const Texture2D *texture = Object::cast_to<Texture2D>(p_resource.ptr())) {
		Ref<PlaceholderTexture2D> placeholder;
		placeholder.instantiate();
		placeholder->set_size(texture->get_size());
		return placeholder;
	}

	if (const Texture2DArray *texture = Object::cast_to<Texture2DArray>(p_resource.ptr())) {
		Ref<PlaceholderTexture2DArray> placeholder;
		placeholder.instantiate();
		placeholder->set_size(Size2i(texture->get_width(), texture->get_height()));
		placeholder->set_layers(texture->get_layers());
		return placeholder;
	}

	if (const Texture3D *texture = Object::cast_to<Texture3D>(p_resource.ptr())) {
		Ref<PlaceholderTexture3D> placeholder;
		placeholder.instantiate();
		placeholder->set_size(Vector3i(texture->get_width(), texture->get_height(), texture->get_depth()));
		return placeholder;
	}

	if (const Cubemap *cubemap = Object::cast_to<Cubemap>(p_resource.ptr())) {
		Ref<PlaceholderCubemap> placeholder;
		placeholder.instantiate();
		placeholder->set_size(Size2i(cubemap->get_width(), cubemap->get_height()));
		placeholder->set_layers(cubemap->get_layers());
		return placeholder;
	}

	if (const CubemapArray *cubemap = Object::cast_to<CubemapArray>(p_resource.ptr())) {
		Ref<PlaceholderCubemapArray> placeholder;
		placeholder.instantiate();
		placeholder->set_size(Size2i(cubemap->get_width(), cubemap->get_height()));
		placeholder->set_layers(cubemap->get_layers());
		return placeholder;
	}

	if (Object::cast_to<Material>(p_resource.ptr()) != nullptr) {
		Ref<PlaceholderMaterial> placeholder;
		placeholder.instantiate();
		return placeholder;
	}

	if (const Mesh *mesh = Object::cast_to<Mesh>(p_resource.ptr())) {
		Ref<PlaceholderMesh> placeholder;
		placeholder.instantiate();
		placeholder->set_aabb(mesh->get_aabb());
		return placeholder;
	}

	return Ref<Resource>();
}
