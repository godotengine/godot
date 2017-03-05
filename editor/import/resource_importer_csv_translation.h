#ifndef RESOURCEIMPORTERCSVTRANSLATION_H
#define RESOURCEIMPORTERCSVTRANSLATION_H

#include "io/resource_import.h"


class ResourceImporterCSVTranslation : public ResourceImporter {
	GDCLASS(ResourceImporterCSVTranslation,ResourceImporter)
public:
	virtual String get_importer_name() const;
	virtual String get_visible_name() const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual String get_save_extension() const;
	virtual String get_resource_type() const;

	virtual int get_preset_count() const;
	virtual String get_preset_name(int p_idx) const;

	virtual void get_import_options(List<ImportOption> *r_options,int p_preset=0) const;
	virtual bool get_option_visibility(const String& p_option,const Map<StringName,Variant>& p_options) const;

	virtual Error import(const String& p_source_file,const String& p_save_path,const Map<StringName,Variant>& p_options,List<String>* r_platform_variants,List<String>* r_gen_files=NULL);

	ResourceImporterCSVTranslation();
};

#endif // RESOURCEIMPORTERCSVTRANSLATION_H
