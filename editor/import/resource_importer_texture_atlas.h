#ifndef RESOURCE_IMPORTER_TEXTURE_ATLAS_H
#define RESOURCE_IMPORTER_TEXTURE_ATLAS_H

#include "core/image.h"
#include "core/io/resource_importer.h"
class ResourceImporterTextureAtlas : public ResourceImporter {
	GDCLASS(ResourceImporterTextureAtlas, ResourceImporter)

	struct PackData {
		Rect2 region;
		bool is_mesh;
		Vector<int> chart_pieces; //one for region, many for mesh
		Vector<Vector<Vector2> > chart_vertices; //for mesh
		Ref<Image> image;
	};

public:
	enum ImportMode {
		IMPORT_MODE_REGION,
		IMPORT_MODE_2D_MESH
	};

	virtual String get_importer_name() const;
	virtual String get_visible_name() const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual String get_save_extension() const;
	virtual String get_resource_type() const;

	virtual int get_preset_count() const;
	virtual String get_preset_name(int p_idx) const;

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const;
	virtual String get_option_group_file() const;

	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = NULL, Variant *r_metadata = NULL);
	virtual Error import_group_file(const String &p_group_file, const Map<String, Map<StringName, Variant> > &p_source_file_options, const Map<String, String> &p_base_paths);

	ResourceImporterTextureAtlas();
};

#endif // RESOURCE_IMPORTER_TEXTURE_ATLAS_H
