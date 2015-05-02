/*************************************************************************/
/*  console.h                                                            */
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
#ifndef CONSOLE_H
#define CONSOLE_H

#include "scene/gui/popup.h"
#include "scene/gui/button.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"
#include "output_strings.h"
#include "property_editor.h"
#include "scene_tree_editor.h"
#include "editor_data.h"

class Console : public Popup {

	OBJ_TYPE( Console, Popup );

	TabContainer *tabs;
	OutputStrings *output;
	OutputStrings *errors;
	Control *status;
	Control *inspect;
	Control *globals;
	Button *close;
	int height;

	EditorHistory inspect_history;
	SceneTreeEditor *inspect_tree_editor;
	PropertyEditor *inspect_property_editor;
	PropertyEditor *globals_property_editor;

	Tree *stats_tree;

	struct StatsItems {

		TreeItem *render_objects_in_frame;
		TreeItem *material_changes_in_frame;

		TreeItem *usage_video_mem_total;
		TreeItem *usage_video_mem_used;
		TreeItem *usage_texture_mem_used;
		TreeItem *usage_vertex_mem_used;
		
		TreeItem *usage_static_memory_total;
		TreeItem *usage_static_memory;
		TreeItem *usage_dynamic_memory_total;
		TreeItem *usage_dynamic_memory;
		TreeItem *usage_objects_instanced;

	} stats;

	struct OutputQueue {

		OutputStrings::LineType type;
		Variant meta;
		String text;
	};

	Mutex *output_queue_mutex;
	List<OutputQueue> output_queue;


	ErrorHandlerList err_handler;
	PrintHandlerList print_handler;

	void _inspector_node_selected();

	static void _error_handle(void *p_this,const char*p_function,const char* p_file,int p_line,const char *p_error, const char *p_explanation,ErrorHandlerType p_type);
	static void _print_handle(void *p_this,const String& p_string);

protected:

	virtual void _window_input_event(InputEvent p_event);
	virtual void _window_resize_event();

	void _stats_update_timer_callback();
	void _resized();
	void _close_pressed();

	void _notification(int p_what);

	static void _bind_methods();
public:
	Console();
	~Console();
};

#endif // CONSOLE_H
