#ifndef TEXTURE_EDITOR_PLUGIN_H
#define TEXTURE_EDITOR_PLUGIN_H



#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/resources/texture.h"


class TextureEditor : public Control {

	GDCLASS(TextureEditor, Control);


	Ref<Texture> texture;

protected:
	void _notification(int p_what);
	void _gui_input(InputEvent p_event);
	static void _bind_methods();
public:

	void edit(Ref<Texture> p_texture);
	TextureEditor();
};


class TextureEditorPlugin : public EditorPlugin {

	GDCLASS( TextureEditorPlugin, EditorPlugin );

	TextureEditor *texture_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Texture"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	TextureEditorPlugin(EditorNode *p_node);
	~TextureEditorPlugin();

};

#endif // TEXTURE_EDITOR_PLUGIN_H
