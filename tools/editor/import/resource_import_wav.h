#ifndef RESOURCEIMPORTWAV_H
#define RESOURCEIMPORTWAV_H


#include "io/resource_import.h"

class ResourceImporterWAV : public ResourceImporter {
	GDCLASS(ResourceImporterWAV,ResourceImporter)
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

	void _compress_ima_adpcm(const Vector<float>& p_data,PoolVector<uint8_t>& dst_data);

	virtual Error import(const String& p_source_file,const String& p_save_path,const Map<StringName,Variant>& p_options,List<String>* r_platform_variants,List<String>* r_gen_files=NULL);

	ResourceImporterWAV();
};

#endif // RESOURCEIMPORTWAV_H
