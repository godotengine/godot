/*************************************************************************/
/*  console.cpp                                                          */
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
#include "console.h"
#include "os/os.h"
#include "os/keyboard.h"
 
#include "editor_icons.h"
#include "scene/gui/label.h"
#include "globals.h"


void Console::_stats_update_timer_callback() {

	if (!status->is_visible())
		return;

	VisualServer *vs = VisualServer::get_singleton();

	stats.render_objects_in_frame->set_text(1,String::num(vs->get_render_info( VisualServer::INFO_OBJECTS_IN_FRAME ) ) );
	stats.material_changes_in_frame->set_text(1,String::num(vs->get_render_info( VisualServer::INFO_MATERIAL_CHANGES_IN_FRAME ) ) );

	int64_t total_vmem = vs->get_render_info( VisualServer::INFO_USAGE_VIDEO_MEM_TOTAL );
	if (total_vmem<0)
		stats.usage_video_mem_total->set_text(1, "Unknown");
	else
		stats.usage_video_mem_total->set_text(1,String::humanize_size( total_vmem ) );

	stats.usage_video_mem_used->set_text(1,String::humanize_size( vs->get_render_info( VisualServer::INFO_VIDEO_MEM_USED ) ) );
	stats.usage_texture_mem_used->set_text(1,String::humanize_size( vs->get_render_info( VisualServer::INFO_TEXTURE_MEM_USED ) ) );
	stats.usage_vertex_mem_used->set_text(1,String::humanize_size( vs->get_render_info( VisualServer::INFO_VERTEX_MEM_USED ) ) );


	stats.usage_static_memory_total->set_text(1,String::humanize_size( Memory::get_static_mem_available() ) );
	stats.usage_static_memory->set_text(1,String::humanize_size( Memory::get_static_mem_usage() ) );
	stats.usage_dynamic_memory_total->set_text(1,String::humanize_size( Memory::get_dynamic_mem_available() ) );
	stats.usage_dynamic_memory->set_text(1,String::humanize_size( Memory::get_dynamic_mem_usage() ) );
	stats.usage_objects_instanced->set_text(1,String::num( ObjectDB::get_object_count()) );


}

void Console::_print_handle(void *p_this,const String& p_string) {


	return;
	Console *self = (Console*)p_this;

	OutputQueue oq;
	oq.text=p_string;
	oq.type=OutputStrings::LINE_NORMAL;


	if (self->output_queue_mutex)
		self->output_queue_mutex->lock();

	self->output_queue.push_back(oq);

	if (self->output_queue_mutex)
		self->output_queue_mutex->unlock();

}
void Console::_error_handle(void *p_this,const char*p_function,const char* p_file,int p_line,const char *p_error, const char *p_explanation,ErrorHandlerType p_type) {


	Console *self = (Console*)p_this;

	OutputQueue oq;
	oq.text="ERROR: "+String(p_file)+":"+itos(p_line)+", in function: "+String(p_function);
	oq.text+="\n         "+String(p_error)+".";
	if (p_explanation && p_explanation[0])
		oq.text+="\n            Reason: "+String(p_explanation);
	oq.text+="\n";
	oq.type=OutputStrings::LINE_ERROR;


	if (self->output_queue_mutex)
		self->output_queue_mutex->lock();

	self->output_queue.push_back(oq);

	if (self->output_queue_mutex)
		self->output_queue_mutex->unlock();


}

void Console::_window_input_event(InputEvent p_event) {

	Control::_window_input_event(p_event);

	if (p_event.type==InputEvent::KEY && p_event.key.pressed) {

		if (p_event.key.scancode==KEY_QUOTELEFT && p_event.key.mod.control) {

			if (is_visible())
				hide();
			else {
				globals_property_editor->edit( NULL );
				globals_property_editor->edit( Globals::get_singleton() );
				show();
			};
		}

		if (p_event.key.scancode==KEY_ESCAPE && !window_has_modal_stack() && is_visible()) {
			hide();
			get_tree()->call_group(0,"windows","_cancel_input_ID",p_event.ID);
		}


	}
}

void Console::_window_resize_event() {

//	Control::_window_resize_event();
	_resized();
}


void Console::_resized() {

	set_pos( Point2( 0, OS::get_singleton()->get_video_mode().height-height) );
	set_size( Size2( OS::get_singleton()->get_video_mode().width, height) );
}

void Console::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			_resized();
			show();
			globals_property_editor->edit( Globals::get_singleton() );

		} break;

		case NOTIFICATION_PROCESS: {
		//pop messies

			if (output_queue_mutex)
				output_queue_mutex->lock();

			while(output_queue.size()) {

				OutputQueue q = output_queue.front()->get();
				if (q.type==OutputStrings::LINE_ERROR || q.type==OutputStrings::LINE_WARNING)
					errors->add_line(q.text,q.meta,q.type);
				output->add_line(q.text,q.meta,q.type);
				output_queue.pop_front();
			}

			if (output_queue_mutex)
				output_queue_mutex->unlock();

		} break;
		case NOTIFICATION_DRAW: {

			RID ci = get_canvas_item();
			get_stylebox("panel","Panel")->draw(ci,Rect2(Point2(),get_size()));

		} break;
	}
}


void Console::_close_pressed() {

	hide();
}

void Console::_inspector_node_selected() {


	Node *node = inspect_tree_editor->get_selected();

	if (!node)
		inspect_property_editor->edit(NULL);
	else {

		inspect_history.add_object(node->get_instance_ID());

		inspect_property_editor->edit(node);
	}

}

void Console::_bind_methods() {

	ObjectTypeDB::bind_method("_stats_update_timer_callback",&Console::_stats_update_timer_callback);
	ObjectTypeDB::bind_method("_close_pressed",&Console::_close_pressed);
	ObjectTypeDB::bind_method("_inspector_node_selected",&Console::_inspector_node_selected);
}


Console::Console() {

	Ref<Theme> theme( memnew( Theme ) );
	set_theme( theme );
	editor_register_icons(theme);

	height=300;
	tabs = memnew( TabContainer );
	tabs->set_tab_align(TabContainer::ALIGN_LEFT);
	add_child(tabs);
	tabs->set_area_as_parent_rect();

	output = memnew( OutputStrings );
	output->set_name("Output");
	tabs->add_child(output);
	errors = memnew( OutputStrings );
	errors->set_name("Errors");
	tabs->add_child(errors);
	status = memnew( Control );
	status->set_name("Stats");
	tabs->add_child(status);
	inspect  = memnew( Control );
	inspect->set_name("Inspect");
	tabs->add_child(inspect);
	globals = memnew( Control );
	globals->set_name("Globals");
	tabs->add_child(globals);

	// stats

	stats_tree = memnew( Tree );
	stats_tree->set_hide_root(true);
	stats_tree->set_columns(2);
	status->add_child(stats_tree);
	stats_tree->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	stats_tree->set_anchor( MARGIN_RIGHT, ANCHOR_RATIO );
	stats_tree->set_margin( MARGIN_RIGHT, 0.5 );
	stats_tree->set_begin( Point2( 20,25 ) );
	stats_tree->set_end( Point2( 0.5,5 ) );

	Label *stats_label = memnew( Label );
	stats_label->set_text("Engine Statistics:");
	stats_label->set_pos( Point2( 5,5 ) );
	status->add_child(stats_label);

	TreeItem *stats_tree_root = stats_tree->create_item(NULL);

	{
		//system items
		TreeItem *system_item = stats_tree->create_item(stats_tree_root);
		system_item->set_text(0,"System");

		stats.usage_static_memory_total = stats_tree->create_item(system_item);
		stats.usage_static_memory_total->set_text(0,"Total Static Mem");;
		stats.usage_static_memory = stats_tree->create_item(system_item);
		stats.usage_static_memory->set_text(0,"Static Mem Usage");;
		stats.usage_dynamic_memory_total = stats_tree->create_item(system_item);
		stats.usage_dynamic_memory_total->set_text(0,"Total Dynamic Mem");;
		stats.usage_dynamic_memory = stats_tree->create_item(system_item);
		stats.usage_dynamic_memory->set_text(0,"Dynamic Mem Usage");
		stats.usage_objects_instanced = stats_tree->create_item(system_item);
		stats.usage_objects_instanced->set_text(0,"Instanced Objects");

		//render items
		TreeItem *render_item = stats_tree->create_item(stats_tree_root);
		render_item->set_text(0,"Render");
		stats.render_objects_in_frame = stats_tree->create_item(render_item);
		stats.render_objects_in_frame->set_text(0,"Visible Objects");
		stats.material_changes_in_frame = stats_tree->create_item(render_item);
		stats.material_changes_in_frame->set_text(0,"Material Changes");
		stats.usage_video_mem_total = stats_tree->create_item(render_item);
		stats.usage_video_mem_total->set_text(0,"Total Video Mem");
		stats.usage_texture_mem_used = stats_tree->create_item(render_item);
		stats.usage_texture_mem_used->set_text(0,"Texture Mem Usage");
		stats.usage_vertex_mem_used = stats_tree->create_item(render_item);
		stats.usage_vertex_mem_used->set_text(0,"Vertex Mem Usage");
		stats.usage_video_mem_used = stats_tree->create_item(render_item);
		stats.usage_video_mem_used->set_text(0,"Combined Mem Usage");
	}

	{

		inspect_tree_editor = memnew( SceneTreeEditor );
		inspect_tree_editor->set_anchor( MARGIN_RIGHT, ANCHOR_RATIO );
		inspect_tree_editor->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
		inspect_tree_editor->set_begin( Point2( 20, 5 ) );
		inspect_tree_editor->set_end( Point2( 0.49, 5 ) );
		inspect->add_child(inspect_tree_editor);

		inspect_property_editor = memnew( PropertyEditor );
		inspect_property_editor->set_anchor( MARGIN_LEFT, ANCHOR_RATIO );
		inspect_property_editor->set_anchor( MARGIN_RIGHT, ANCHOR_END );
		inspect_property_editor->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
		inspect_property_editor->set_begin( Point2( 0.51, 5 ) );
		inspect_property_editor->set_end( Point2( 5, 5 ) );
		inspect->add_child(inspect_property_editor);
	}


	{ //globals

		globals_property_editor = memnew( PropertyEditor );
		globals_property_editor->set_anchor( MARGIN_RIGHT, ANCHOR_END );
		globals_property_editor->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
		globals_property_editor->set_begin( Point2( 15, 5 ) );
		globals_property_editor->set_end( Point2( 5, 5 ) );
		globals_property_editor->get_top_label()->set_text("Globals Editor:");
		globals->add_child(globals_property_editor);

	}


#ifndef NO_THREADS
	output_queue_mutex = Mutex::create();
#else
	output_queue_mutex = NULL;
#endif


	hide();
	set_process(true);

	close = memnew( Button );
	add_child(close);
	close->set_anchor( MARGIN_LEFT, ANCHOR_END);
	close->set_anchor( MARGIN_RIGHT, ANCHOR_END);
	close->set_begin( Point2( 25, 3 ) );
	close->set_end( Point2( 5, 3 ) );
	close->set_flat(true);
	close->connect("pressed", this,"_close_pressed");


	close->set_icon( get_icon("close","Icons") );
//	force_top_viewport(true);


	err_handler.userdata=this;
	err_handler.errfunc=_error_handle;
	add_error_handler(&err_handler);

	print_handler.userdata=this;
	print_handler.printfunc=_print_handle;
	add_print_handler(&print_handler);

	Timer *timer = memnew( Timer );
	add_child(timer);
	timer->set_wait_time(1);
	timer->start();
	timer->connect("timeout", this,"_stats_update_timer_callback");
	inspect_tree_editor->connect("node_selected", this,"_inspector_node_selected");



}


Console::~Console() {

	if (output_queue_mutex)
		memdelete(output_queue_mutex);

	remove_error_handler(&err_handler);
	remove_print_handler(&print_handler);

}
