#ifndef TEXTURE_LOADER_DDS_H
#define TEXTURE_LOADER_DDS_H

#include "scene/resources/texture.h"
#include "io/resource_loader.h"

class ResourceFormatDDS : public ResourceFormatLoader{
public:

	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

	virtual ~ResourceFormatDDS() {}
};

#endif // TEXTURE_LOADER_DDS_H
