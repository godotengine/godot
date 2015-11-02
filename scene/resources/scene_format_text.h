#ifndef SCENE_FORMAT_TEXT_H
#define SCENE_FORMAT_TEXT_H

#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/resources/packed_scene.h"

class ResourceFormatSaverTextInstance  {

	String local_path;

	Ref<PackedScene> packed_scene;

	bool takeover_paths;
	bool relative_paths;
	bool bundle_resources;
	bool skip_editor;
	FileAccess *f;
	Set<RES> resource_set;
	List<RES> saved_resources;
	Map<RES,int> external_resources;
	Map<RES,int> internal_resources;

	void _find_resources(const Variant& p_variant,bool p_main=false);
	void write_property(const String& p_name,const Variant& p_property,bool *r_ok=NULL);

public:

	Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);


};

class ResourceFormatSaverText : public ResourceFormatSaver {
public:
	static ResourceFormatSaverText* singleton;
	virtual Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);
	virtual bool recognize(const RES& p_resource) const;
	virtual void get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const;

	ResourceFormatSaverText();
};


#endif // SCENE_FORMAT_TEXT_H
