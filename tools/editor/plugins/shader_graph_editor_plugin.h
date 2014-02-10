/*************************************************************************/
/*  shader_graph_editor_plugin.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef SHADER_GRAPH_EDITOR_PLUGIN_H
#define SHADER_GRAPH_EDITOR_PLUGIN_H

#if 0
#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/resources/shader.h"
#include "servers/visual/shader_graph.h"
#include "scene/gui/tree.h"
#include "scene/gui/button.h"
#include "scene/gui/popup.h"
#include "tools/editor/property_editor.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ShaderEditor : public Control {

	OBJ_TYPE(ShaderEditor, Control );

	enum MenuAction {

		GRAPH_ADD_NODE,
		GRAPH_CLEAR,
		NODE_DISCONNECT,
		NODE_ERASE,

	};

	enum ClickType {
		CLICK_NONE,
		CLICK_NODE,
		CLICK_INPUT_SLOT,
		CLICK_OUTPUT_SLOT,
		CLICK_PARAMETER
	};

	PopupMenu *node_popup;
	Popup *add_popup;
	PopupMenu *vertex_popup;
	PopupMenu *fragment_popup;
	PopupMenu *post_popup;
	Tree *add_types;
	Button *add_confirm;
	HScrollBar *h_scroll;
	VScrollBar *v_scroll;

	Ref<Shader> shader;
	List<int> order;
	Set<int> active_nodes;
	ShaderGraph shader_graph;
	int last_x,last_y;
	uint32_t last_id;

	CustomPropertyEditor *property_editor;

	Point2 offset;
	ClickType click_type;
	Point2 click_pos;
	int click_node;
	int click_slot;
	Point2 click_motion;
	ClickType rclick_type;
	int rclick_node;
	int rclick_slot;

	Size2 _get_maximum_size();
	Size2 get_node_size(int p_node) const;
	void _draw_node(int p_node);

	void _add_node_from_text(const String& p_text);
	void _update_scrollbars();
	void _scroll_moved();
	void _node_param_changed();
	void _node_add_callback();
	void _node_add(VisualServer::ShaderNodeType p_type);
	void _node_edit_property(int p_node);
	void _node_menu_item(int p_item);
	void _vertex_item(int p_item);
	void _fragment_item(int p_item);
	void _post_item(int p_item);

	ClickType _locate_click(const Point2& p_click,int *p_node_id,int *p_slot_index) const;
	Point2 _get_slot_pos(int p_node_id,bool p_input,int p_slot);

	Error validate_graph();

	void _read_shader_graph();
	void _write_shader_graph();

	virtual bool has_point(const Point2& p_point) const;
protected:
	void _notification(int p_what);
	void _input_event(InputEvent p_event);
	static void _bind_methods();
public:

	void edit(Ref<Shader> p_shader);
	ShaderEditor();
};

class ShaderEditorPlugin : public EditorPlugin {

	OBJ_TYPE( ShaderEditorPlugin, EditorPlugin );

	ShaderEditor *shader_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Shader"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	ShaderEditorPlugin(EditorNode *p_node);
	~ShaderEditorPlugin();

};
#endif
#endif // SHADER_GRAPH_EDITOR_PLUGIN_H
