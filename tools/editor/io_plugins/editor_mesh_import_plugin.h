#ifndef EDITOR_MESH_IMPORT_PLUGIN_H
#define EDITOR_MESH_IMPORT_PLUGIN_H


#include "tools/editor/editor_import_export.h"
#include "scene/resources/font.h"

class EditorNode;
class EditorMeshImportDialog;

class EditorMeshImportPlugin : public EditorImportPlugin {

	OBJ_TYPE(EditorMeshImportPlugin,EditorImportPlugin);

	EditorMeshImportDialog *dialog;


public:

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);


	EditorMeshImportPlugin(EditorNode* p_editor);
};

#endif // EDITOR_MESH_IMPORT_PLUGIN_H
