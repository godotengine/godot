#ifndef EDITORPREVIEWPLUGINS_H
#define EDITORPREVIEWPLUGINS_H

#include "tools/editor/editor_resource_preview.h"

class EditorTexturePreviewPlugin : public EditorResourcePreviewGenerator {
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorTexturePreviewPlugin();
};


class EditorBitmapPreviewPlugin : public EditorResourcePreviewGenerator {
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorBitmapPreviewPlugin();
};



class EditorPackedScenePreviewPlugin : public EditorResourcePreviewGenerator {

	Ref<Texture> _gen_from_imd(Ref<ResourceImportMetadata> p_imd);
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);
	virtual Ref<Texture> generate_from_path(const String& p_path);

	EditorPackedScenePreviewPlugin();
};

class EditorMaterialPreviewPlugin : public EditorResourcePreviewGenerator {

	RID scenario;
	RID sphere;
	RID sphere_instance;
	RID viewport;
	RID light;
	RID light_instance;
	RID light2;
	RID light_instance2;
	RID camera;
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorMaterialPreviewPlugin();
	~EditorMaterialPreviewPlugin();
};

class EditorScriptPreviewPlugin : public EditorResourcePreviewGenerator {
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorScriptPreviewPlugin();
};


class EditorSamplePreviewPlugin : public EditorResourcePreviewGenerator {
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorSamplePreviewPlugin();
};


class EditorMeshPreviewPlugin : public EditorResourcePreviewGenerator {

	RID scenario;
	RID mesh_instance;
	RID viewport;
	RID light;
	RID light_instance;
	RID light2;
	RID light_instance2;
	RID camera;
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorMeshPreviewPlugin();
	~EditorMeshPreviewPlugin();
};


#endif // EDITORPREVIEWPLUGINS_H
