#ifndef RESOURCE_IMPORTER_LAYERED_TEXTURE_H
#define RESOURCE_IMPORTER_LAYERED_TEXTURE_H

#include "image.h"
#include "io/resource_import.h"

class StreamTexture;

class ResourceImporterLayeredTexture : public ResourceImporter {
	GDCLASS(ResourceImporterLayeredTexture, ResourceImporter)

	bool is_3d;

protected:
	static void _texture_reimport_srgb(const Ref<StreamTexture> &p_tex);
	static void _texture_reimport_3d(const Ref<StreamTexture> &p_tex);
	static void _texture_reimport_normal(const Ref<StreamTexture> &p_tex);

	static ResourceImporterLayeredTexture *singleton;

public:
	static ResourceImporterLayeredTexture *get_singleton() { return singleton; }
	virtual String get_importer_name() const;
	virtual String get_visible_name() const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual String get_save_extension() const;
	virtual String get_resource_type() const;

	enum Preset {
		PRESET_3D,
		PRESET_2D,
		PRESET_COLOR_CORRECT,
	};

	enum CompressMode {
		COMPRESS_LOSSLESS,
		COMPRESS_VIDEO_RAM,
		COMPRESS_UNCOMPRESSED
	};

	virtual int get_preset_count() const;
	virtual String get_preset_name(int p_idx) const;

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const;

	void _save_tex(const Vector<Ref<Image> > &p_images, const String &p_to_path, int p_compress_mode, Image::CompressMode p_vram_compression, bool p_mipmaps, int p_texture_flags);

	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = NULL);

	void update_imports();

	void set_3d(bool p_3d) { is_3d = p_3d; }
	ResourceImporterLayeredTexture();
	~ResourceImporterLayeredTexture();
};
#endif // RESOURCE_IMPORTER_LAYERED_TEXTURE_H
