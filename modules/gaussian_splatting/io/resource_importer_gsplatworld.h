#ifndef RESOURCE_IMPORTER_GSPLATWORLD_H
#define RESOURCE_IMPORTER_GSPLATWORLD_H

#ifdef TOOLS_ENABLED

#include "core/io/resource_importer.h"

class ResourceImporterGSplatWorld : public ResourceImporter {
	GDCLASS(ResourceImporterGSplatWorld, ResourceImporter);

protected:
	static void _bind_methods() {}

public:
	virtual String get_importer_name() const override;
	virtual String get_visible_name() const override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual String get_save_extension() const override;
	virtual String get_resource_type() const override;
	virtual float get_priority() const override { return 1.0f; }
	virtual int get_format_version() const override { return 1; }
	virtual int get_preset_count() const override;
	virtual String get_preset_name(int p_idx) const override;
	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const override;
	virtual bool get_option_visibility(const String &p_path, const String &p_option,
			const HashMap<StringName, Variant> &p_options) const override;
	virtual bool can_import_threaded() const override { return true; }

	virtual Error import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path,
			const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants,
			List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) override;
};

#endif // TOOLS_ENABLED

#endif // RESOURCE_IMPORTER_GSPLATWORLD_H
