/*
 * color_ramp_editor_plugin.h
 */

#ifndef TOOLS_EDITOR_PLUGINS_COLOR_RAMP_EDITOR_PLUGIN_H_
#define TOOLS_EDITOR_PLUGINS_COLOR_RAMP_EDITOR_PLUGIN_H_

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/gui/color_ramp_edit.h"

class ColorRampEditorPlugin : public EditorPlugin {

	OBJ_TYPE( ColorRampEditorPlugin, EditorPlugin );

	bool _2d;
	Ref<ColorRamp> color_ramp_ref;
	ColorRampEdit *ramp_editor;
	EditorNode *editor;

protected:
	static void _bind_methods();
	void _ramp_changed();
	void _undo_redo_color_ramp(const Vector<float>& offsets, const Vector<Color>& colors);

public:
	virtual String get_name() const { return "ColorRamp"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	ColorRampEditorPlugin(EditorNode *p_node, bool p_2d);
	~ColorRampEditorPlugin();

};

#endif /* TOOLS_EDITOR_PLUGINS_COLOR_RAMP_EDITOR_PLUGIN_H_ */
