#ifndef RESOURCE_IMPORT_H
#define RESOURCE_IMPORT_H


#include "io/resource_loader.h"
class ResourceImporter;

class ResourceFormatImporter : public ResourceFormatLoader {

	struct PathAndType {
		String path;
		String type;
	};


	Error _get_path_and_type(const String& p_path,PathAndType & r_path_and_type) const;

	Set< Ref<ResourceImporter> > importers;
public:

	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_recognized_extensions_for_type(const String& p_type,List<String> *p_extensions) const;
	bool recognize_path(const String& p_path,const String& p_for_type=String()) const;
	virtual bool handles_type(const String& p_type) const=0;
	virtual String get_resource_type(const String &p_path) const=0;
	virtual void get_dependencies(const String& p_path,List<String> *p_dependencies,bool p_add_types=false);

};


class ResourceImporter {
public:
	virtual String get_name() const=0;
	virtual String get_visible_name() const=0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const=0;
	virtual String get_resource_type() const=0;

	struct ImportOption {
		PropertyInfo option;
		Variant default_value;
	};

	virtual void get_import_options(List<ImportOption> *r_options)=0;

	virtual RES import(const String& p_path,const Map<StringName,Variant>& p_options)=0;

};

#endif // RESOURCE_IMPORT_H
