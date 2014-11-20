#ifndef RESOURCE_SAVER_PNG_H
#define RESOURCE_SAVER_PNG_H

#include "io/resource_saver.h"

class ResourceSaverPNG : public ResourceFormatSaver {
public:

	static Error save_image(const String &p_path, Image& p_img);

	virtual Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);
	virtual bool recognize(const RES& p_resource) const;
	virtual void get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const;

	ResourceSaverPNG();
};


#endif // RESOURCE_SAVER_PNG_H
