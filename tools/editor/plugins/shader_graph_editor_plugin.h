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


#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/resources/shader.h"
#include "servers/visual/shader_graph.h"
#include "scene/gui/tree.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "tools/editor/property_editor.h"
#include "scene/resources/shader_graph.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class GraphColorRampEdit : public Control {

	OBJ_TYPE(GraphColorRampEdit,Control);


	struct Point {

		float offset;
		Color color;
		bool operator<(const Point& p_ponit) const {
			return offset<p_ponit.offset;
		}
	};

	PopupPanel *popup;
	ColorPicker *picker;


	bool grabbing;
	int grabbed;
	float grabbed_at;
	Vector<Point> points;

	void _color_changed(const Color& p_color);

protected:
	void _input_event(const InputEvent& p_event);
	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_ramp(const Vector<float>& p_offsets,const Vector<Color>& p_colors);
	Vector<float> get_offsets() const;
	Vector<Color> get_colors() const;
	virtual Size2 get_minimum_size() const;
	GraphColorRampEdit();
};


class GraphCurveMapEdit : public Control {

	OBJ_TYPE(GraphCurveMapEdit,Control);


	struct Point {

		float offset;
		float height;
		bool operator<(const Point& p_ponit) const {
			return offset<p_ponit.offset;
		}
	};


	bool grabbing;
	int grabbed;
	Vector<Point> points;

	void _plot_curve(const Vector2& p_a,const Vector2& p_b,const Vector2& p_c,const Vector2& p_d);
protected:
	void _input_event(const InputEvent& p_event);
	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_points(const Vector<Vector2>& p_points);
	Vector<Vector2> get_points() const;
	virtual Size2 get_minimum_size() const;
	GraphCurveMapEdit();
};

class ShaderGraphView : public Node {

	OBJ_TYPE(ShaderGraphView,Node);



	CustomPropertyEditor *ped_popup;
	bool block_update;

	Label *status;
	GraphEdit *graph_edit;
	Ref<ShaderGraph> graph;
	int edited_id;

	ShaderGraph::ShaderType type;

	void _update_graph();
	void _create_node(int p_id);



	void _connection_request(const String& p_from, int p_from_slot,const String& p_to,int p_to_slot);
	void _disconnection_request(const String& p_from, int p_from_slot,const String& p_to,int p_to_slot);

	void _node_removed(int p_id);
	void _node_moved(const Vector2& p_from, const Vector2& p_to,int p_id);
	void _move_node(int p_id,const Vector2& p_to);

	void _scalar_const_changed(double p_value,int p_id);
	void _vec_const_changed(double p_value, int p_id, Array p_arr);
	void _rgb_const_changed(const Color& p_color, int p_id);
	void _xform_const_changed(int p_id,Node* p_button);
	void _scalar_op_changed(int p_op, int p_id);
	void _vec_op_changed(int p_op, int p_id);
	void _vec_scalar_op_changed(int p_op, int p_id);
	void _rgb_op_changed(int p_op, int p_id);
	void _xform_inv_rev_changed(bool p_enabled, int p_id);
	void _scalar_func_changed(int p_func, int p_id);
	void _vec_func_changed(int p_func, int p_id);
	void _scalar_input_changed(double p_value,int p_id);
	void _vec_input_changed(double p_value, int p_id, Array p_arr);
	void _xform_input_changed(int p_id,Node* p_button);
	void _rgb_input_changed(const Color& p_color, int p_id);
	void _tex_input_change(int p_id,Node* p_button);
	void _cube_input_change(int p_id);
	void _input_name_changed(const String& p_name,int p_id,Node* p_line_edit);
	void _tex_edited(int p_id,Node* p_button);
	void _cube_edited(int p_id,Node* p_button);
	void _variant_edited();
	void _comment_edited(int p_id,Node* p_button);
	void _color_ramp_changed(int p_id,Node* p_ramp);
	void _curve_changed(int p_id,Node* p_curve);
	void _sg_updated();
	Map<int,GraphNode*> node_map;
protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	void add_node(int p_type);
	GraphEdit *get_graph_edit() { return graph_edit; }
	void set_graph(Ref<ShaderGraph> p_graph);

	ShaderGraphView(ShaderGraph::ShaderType p_type=ShaderGraph::SHADER_TYPE_FRAGMENT);
};

class ShaderGraphEditor : public VBoxContainer {

	OBJ_TYPE(ShaderGraphEditor,VBoxContainer);

	MenuButton *menu;
	TabContainer *tabs;
	ShaderGraphView *graph_edits[ShaderGraph::SHADER_TYPE_MAX];
	static const char* node_names[ShaderGraph::NODE_TYPE_MAX];

	bool _2d;
	void _add_node(int p_type);
protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	void edit(Ref<ShaderGraph> p_shader);
	ShaderGraphEditor(bool p_2d);
};

class ShaderGraphEditorPlugin : public EditorPlugin {

	OBJ_TYPE( ShaderGraphEditorPlugin, EditorPlugin );

	bool _2d;
	ShaderGraphEditor *shader_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "ShaderGraph"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	ShaderGraphEditorPlugin(EditorNode *p_node,bool p_2d);
	~ShaderGraphEditorPlugin();

};
#endif

