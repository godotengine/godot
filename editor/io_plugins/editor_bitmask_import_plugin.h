#ifndef EDITOR_BITMASK_IMPORT_PLUGIN_H
#define EDITOR_BITMASK_IMPORT_PLUGIN_H

#include "editor/editor_import_export.h"
#include "scene/resources/font.h"

class EditorNode;
class EditorBitMaskImportDialog;

class EditorBitMaskImportPlugin : public EditorImportPlugin {

	OBJ_TYPE(EditorBitMaskImportPlugin, EditorImportPlugin);

	EditorBitMaskImportDialog *dialog;

public:
	static EditorBitMaskImportPlugin *singleton;

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String &p_from = "");
	virtual Error import(const String &p_path, const Ref<ResourceImportMetadata> &p_from);
	void import_from_drop(const Vector<String> &p_drop, const String &p_dest_path);
	virtual void reimport_multiple_files(const Vector<String> &p_list);
	virtual bool can_reimport_multiple_files() const;

	EditorBitMaskImportPlugin(EditorNode *p_editor);
};

class EditorBitMaskExportPlugin : public EditorExportPlugin {

	OBJ_TYPE(EditorBitMaskExportPlugin, EditorExportPlugin);

public:
	EditorBitMaskExportPlugin();
};

#endif // EDITOR_SAMPLE_IMPORT_PLUGIN_H
