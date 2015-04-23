/*************************************************************************/
/*  script_editor_debugger.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SCRIPT_EDITOR_DEBUGGER_H
#define SCRIPT_EDITOR_DEBUGGER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "core/io/tcp_server.h"
#include "core/io/packet_peer.h"

class Tree;
class PropertyEditor;
class EditorNode;
class ScriptEditorDebuggerVariables;
class LineEdit;
class TabContainer;
class RichTextLabel;
class TextureButton;
class AcceptDialog;
class TreeItem;
class HSplitContainer;

class ScriptEditorDebugger : public Control {

	OBJ_TYPE( ScriptEditorDebugger, Control );

	AcceptDialog *msgdialog;



	LineEdit *clicked_ctrl;
	LineEdit *clicked_ctrl_type;
	Tree *scene_tree;
	HSplitContainer *info;
	Button *scene_tree_refresh;

	TextureButton *tb;


	TabContainer *tabs;

	Label *reason;
	bool log_forced_visible;
	ScriptEditorDebuggerVariables *variables;

	Button *step;
	Button *next;
	Button *back;
	Button *forward;
	Button *dobreak;
	Button *docontinue;

	List<Vector<float> > perf_history;
	Vector<float> perf_max;
	Vector<TreeItem*> perf_items;

	Tree *perf_monitors;
	Control *perf_draw;

	Tree *stack_dump;
	PropertyEditor *inspector;

	Ref<TCP_Server> server;
	Ref<StreamPeerTCP> connection;
	Ref<PacketPeerStream> ppeer;

	String message_type;
	Array message;
	int pending_in_queue;


	EditorNode *editor;

	bool breaked;

	void _performance_draw();
	void _performance_select(Object *, int, bool);
	void _stack_dump_frame_selected();
	void _output_clear();
	void _hide_request();

	void _scene_tree_request();
	void _parse_message(const String& p_msg,const Array& p_data);

protected:

	void _notification(int p_what);
	static void _bind_methods();

public:

	void start();
	void pause();
	void unpause();
	void stop();

	void debug_next();
	void debug_step();
	void debug_break();
	void debug_continue();

	String get_var_value(const String& p_var) const;

	virtual Size2 get_minimum_size() const;
	ScriptEditorDebugger(EditorNode *p_editor=NULL);
	~ScriptEditorDebugger();
};

#endif // SCRIPT_EDITOR_DEBUGGER_H
