#ifndef SHADER_GLOBALS_EDITOR_H
#define SHADER_GLOBALS_EDITOR_H

#include "core/undo_redo.h"
#include "editor/editor_autoload_settings.h"
#include "editor/editor_data.h"
#include "editor/editor_plugin_settings.h"
#include "editor/editor_sectioned_inspector.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tab_container.h"

class ShaderGlobalsEditorInterface;

class ShaderGlobalsEditor : public VBoxContainer {

	GDCLASS(ShaderGlobalsEditor, VBoxContainer)

	ShaderGlobalsEditorInterface *interface;
	EditorInspector *inspector;

	LineEdit *variable_name;
	OptionButton *variable_type;
	Button *variable_add;

	void _variable_added();
	void _variable_deleted(const String &p_variable);
	void _changed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	ShaderGlobalsEditor();
	~ShaderGlobalsEditor();
};

#endif // SHADER_GLOBALS_EDITOR_H
