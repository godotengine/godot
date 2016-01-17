#ifndef EDITOR_EXPORT_SCENE_H
#define EDITOR_EXPORT_SCENE_H

#include "tools/editor/editor_import_export.h"


class EditorSceneExportPlugin : public EditorExportPlugin {
	OBJ_TYPE( EditorSceneExportPlugin, EditorExportPlugin );
public:

	virtual Vector<uint8_t> custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform);

	EditorSceneExportPlugin();
};

#endif // EDITOR_EXPORT_SCENE_H
