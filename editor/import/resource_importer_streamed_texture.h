/**************************************************************************/
/*  resource_importer_streamed_texture.h                                  */
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

#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/resource_importer.h"
#include "servers/rendering/rendering_server.h"

class StreamedTexture2D;

class ResourceImporterStreamedTexture : public ResourceImporter {
	GDCLASS(ResourceImporterStreamedTexture, ResourceImporter);

	static ResourceImporterStreamedTexture *singleton;

	enum {
		MAKE_ROUGHNESS_FLAG = 1,
		MAKE_NORMAL_FLAG = 2,
	};

	Mutex mutex;
	struct MakeInfo {
		int flags = 0;
		String normal_path_for_roughness;
		RS::TextureDetectRoughnessChannel channel_for_roughness = RS::TEXTURE_DETECT_ROUGHNESS_R;
	};

	HashMap<StringName, MakeInfo> make_flags;

	static void _texture_reimport_roughness(const Ref<StreamedTexture2D> &p_tex, const String &p_normal_path, RenderingServer::TextureDetectRoughnessChannel p_channel);
	static void _texture_reimport_normal(const Ref<StreamedTexture2D> &p_tex);

public:
	static ResourceImporterStreamedTexture *get_singleton() { return singleton; }

	String get_importer_name() const override;
	String get_visible_name() const override;
	void get_recognized_extensions(List<String> *p_extensions) const override;
	String get_save_extension() const override;
	String get_resource_type() const override;
	float get_priority() const override { return 2.0; }

	void update_imports();

	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset = 0) const override;
	virtual bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override;
	virtual Error import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) override;

	ResourceImporterStreamedTexture(bool p_singleton = false);
	virtual ~ResourceImporterStreamedTexture() = default;
};
