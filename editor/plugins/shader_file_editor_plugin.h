#ifndef SHADER_FILE_EDITOR_PLUGIN_H
#define SHADER_FILE_EDITOR_PLUGIN_H

#include "editor/code_editor.h"
#include "editor/editor_plugin.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/main/timer.h"
#include "servers/rendering/rendering_device_binds.h"

class ShaderFileEditor : public PanelContainer {

	GDCLASS(ShaderFileEditor, PanelContainer);

	Ref<RDShaderFile> shader_file;

	HBoxContainer *stage_hb;
	ItemList *versions;
	Button *stages[RD::SHADER_STAGE_MAX];
	RichTextLabel *error_text;

	void _update_version(const StringName &p_version_txt, const RenderingDevice::ShaderStage p_stage);
	void _version_selected(int p_stage);
	void _editor_settings_changed();

	void _update_options();
	void _shader_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ShaderFileEditor *singleton;
	void edit(const Ref<RDShaderFile> &p_shader);

	ShaderFileEditor(EditorNode *p_node);
};

class ShaderFileEditorPlugin : public EditorPlugin {

	GDCLASS(ShaderFileEditorPlugin, EditorPlugin);

	ShaderFileEditor *shader_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "ShaderFile"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	ShaderFileEditor *get_shader_editor() const { return shader_editor; }

	ShaderFileEditorPlugin(EditorNode *p_node);
	~ShaderFileEditorPlugin();
};

#endif // SHADER_FILE_EDITOR_PLUGIN_H
