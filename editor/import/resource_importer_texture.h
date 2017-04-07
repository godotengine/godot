/*************************************************************************/
/*  resource_importer_texture.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RESOURCEIMPORTTEXTURE_H
#define RESOURCEIMPORTTEXTURE_H

#include "io/resource_import.h"
class StreamTexture;

class ResourceImporterTexture : public ResourceImporter {
	GDCLASS(ResourceImporterTexture, ResourceImporter)

protected:
	enum {
		MAKE_3D_FLAG = 1,
		MAKE_SRGB_FLAG = 2
	};

	Mutex *mutex;
	Map<StringName, int> make_flags;

	static void _texture_reimport_srgb(const Ref<StreamTexture> &p_tex);
	static void _texture_reimport_3d(const Ref<StreamTexture> &p_tex);

	static ResourceImporterTexture *singleton;

public:
	static ResourceImporterTexture *get_singleton() { return singleton; }
	virtual String get_importer_name() const;
	virtual String get_visible_name() const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual String get_save_extension() const;
	virtual String get_resource_type() const;

	enum Preset {
		PRESET_DETECT,
		PRESET_2D,
		PRESET_2D_PIXEL,
		PRESET_3D,
	};

	enum CompressMode {
		COMPRESS_LOSSLESS,
		COMPRESS_LOSSY,
		COMPRESS_VIDEO_RAM,
		COMPRESS_UNCOMPRESSED
	};

	virtual int get_preset_count() const;
	virtual String get_preset_name(int p_idx) const;

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const;

	void _save_stex(const Image &p_image, const String &p_to_path, int p_compress_mode, float p_lossy_quality, Image::CompressMode p_vram_compression, bool p_mipmaps, int p_texture_flags, bool p_streamable, bool p_detect_3d, bool p_detect_srgb);

	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = NULL);

	void update_imports();

	ResourceImporterTexture();
	~ResourceImporterTexture();
};

#endif // RESOURCEIMPORTTEXTURE_H
