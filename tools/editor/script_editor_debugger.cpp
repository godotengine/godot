/*************************************************************************/
/*  script_editor_debugger.cpp                                           */
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
#include "script_editor_debugger.h"
#include "scene/gui/separator.h"
#include "scene/gui/label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"
#include "scene/gui/texture_button.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/margin_container.h"
#include "property_editor.h"
#include "globals.h"
#include "editor_node.h"
#include "main/performance.h"

class ScriptEditorDebuggerVariables : public Object {

	OBJ_TYPE( ScriptEditorDebuggerVariables, Object );

	List<PropertyInfo> props;
	Map<StringName,Variant> values;
protected:

	bool _set(const StringName& p_name, const Variant& p_value) {

		return false;
	}

	bool _get(const StringName& p_name,Variant &r_ret) const {

		if (!values.has(p_name))
			return false;
		r_ret=values[p_name];
		return true;
	}
	void _get_property_list( List<PropertyInfo> *p_list) const {

		for(const List<PropertyInfo>::Element *E=props.front();E;E=E->next() )
			p_list->push_back(E->get());
	}


public:


	void clear() {

		props.clear();
		values.clear();
	}

	String get_var_value(const String& p_var) const {

		for(Map<StringName,Variant>::Element *E=values.front();E;E=E->next()) {
			String v = E->key().operator String().get_slice("/",1);
			if (v==p_var)
				return E->get();
		}

		return "";
	}

	void add_property(const String &p_name, const Variant& p_value) {

		PropertyInfo pinfo;
		pinfo.name=p_name;
		pinfo.type=p_value.get_type();
		props.push_back(pinfo);
		values[p_name]=p_value;

	}

	void update() {
		_change_notify();
	}


	ScriptEditorDebuggerVariables() {

	}
};

void ScriptEditorDebugger::debug_next() {

	ERR_FAIL_COND(!breaked);
	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());
	Array msg;
	msg.push_back("next");
	ppeer->put_var(msg);
	stack_dump->clear();
	inspector->edit(NULL);

}
void ScriptEditorDebugger::debug_step() {

	ERR_FAIL_COND(!breaked);
	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());

	Array msg;
	msg.push_back("step");
	ppeer->put_var(msg);
	stack_dump->clear();
	inspector->edit(NULL);
}

void ScriptEditorDebugger::debug_break() {

	ERR_FAIL_COND(breaked);
	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());

	Array msg;
	msg.push_back("break");
	ppeer->put_var(msg);

}

void ScriptEditorDebugger::debug_continue() {

	ERR_FAIL_COND(!breaked);
	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());

	Array msg;
	msg.push_back("continue");
	ppeer->put_var(msg);

}

void ScriptEditorDebugger::_scene_tree_request() {

	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());

	Array msg;
	msg.push_back("request_scene_tree");
	ppeer->put_var(msg);

}

void ScriptEditorDebugger::_video_mem_request() {

	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());

	Array msg;
	msg.push_back("request_video_mem");
	ppeer->put_var(msg);

}

Size2 ScriptEditorDebugger::get_minimum_size() const {

	Size2 ms = Control::get_minimum_size();
	ms.y = MAX(ms.y , 250 );
	return ms;

}
void ScriptEditorDebugger::_parse_message(const String& p_msg,const Array& p_data) {



	if (p_msg=="debug_enter") {

		Array msg;
		msg.push_back("get_stack_dump");
		ppeer->put_var(msg);
		ERR_FAIL_COND(p_data.size()!=2);
		bool can_continue=p_data[0];
		String error = p_data[1];
		step->set_disabled(!can_continue);
		next->set_disabled(!can_continue);
		reason->set_text(error);
		reason->set_tooltip(error);
		breaked=true;
		dobreak->set_disabled(true);
		docontinue->set_disabled(false);
		emit_signal("breaked",true,can_continue);
		OS::get_singleton()->move_window_to_foreground();
		tabs->set_current_tab(0);

	} else if (p_msg=="debug_exit") {

		breaked=false;
		step->set_disabled(true);
		next->set_disabled(true);
		reason->set_text("");
		reason->set_tooltip("");
		back->set_disabled(true);
		forward->set_disabled(true);
		dobreak->set_disabled(false);
		docontinue->set_disabled(true);
		emit_signal("breaked",false,false);
		//tabs->set_current_tab(0);

	} else if (p_msg=="message:click_ctrl") {

		clicked_ctrl->set_text(p_data[0]);
		clicked_ctrl_type->set_text(p_data[1]);

	} else if (p_msg=="message:scene_tree") {

		scene_tree->clear();
		Map<int,TreeItem*> lv;

		for(int i=0;i<p_data.size();i+=3) {

			TreeItem *p;
			int level = p_data[i];
			if (level==0) {
				p = NULL;
			} else {
				ERR_CONTINUE(!lv.has(level-1));
				p=lv[level-1];
			}

			TreeItem *it = scene_tree->create_item(p);
			it->set_text(0,p_data[i+1]);
			if (has_icon(p_data[i+2],"EditorIcons"))
				it->set_icon(0,get_icon(p_data[i+2],"EditorIcons"));
			lv[level]=it;
		}

		le_clear->set_disabled(false);
		le_set->set_disabled(false);

	} else if (p_msg=="message:video_mem") {

		vmem_tree->clear();
		TreeItem* root=vmem_tree->create_item();

		int total=0;

		for(int i=0;i<p_data.size();i+=4) {

			TreeItem *it = vmem_tree->create_item(root);
			String type=p_data[i+1];
			int bytes=p_data[i+3].operator int();
			it->set_text(0,p_data[i+0]); //path
			it->set_text(1,type); //type
			it->set_text(2,p_data[i+2]); //type
			it->set_text(3,String::humanize_size(bytes)); //type
			total+=bytes;

			if (has_icon(type,"EditorIcons"))
				it->set_icon(0,get_icon(type,"EditorIcons"));
		}

		vmem_total->set_tooltip("Bytes: "+itos(total));
		vmem_total->set_text(String::humanize_size(total));

	} else if (p_msg=="stack_dump") {

		stack_dump->clear();
		TreeItem *r = stack_dump->create_item();

		for(int i=0;i<p_data.size();i++) {

			Dictionary d = p_data[i];
			ERR_CONTINUE(!d.has("function"));
			ERR_CONTINUE(!d.has("file"));
			ERR_CONTINUE(!d.has("line"));
			ERR_CONTINUE(!d.has("id"));
			TreeItem *s = stack_dump->create_item(r);
			d["frame"]=i;
			s->set_metadata(0,d);

//			String line = itos(i)+" - "+String(d["file"])+":"+itos(d["line"])+" - at func: "+d["function"];
			String line = itos(i)+" - "+String(d["file"])+":"+itos(d["line"]);
			s->set_text(0,line);

			if (i==0)
				s->select(0);
		}
	} else if (p_msg=="stack_frame_vars") {


		variables->clear();



		int ofs =0;
		int mcount = p_data[ofs];

		ofs++;
		for(int i=0;i<mcount;i++) {

			String n = p_data[ofs+i*2+0];
			Variant v = p_data[ofs+i*2+1];

			if (n.begins_with("*")) {

				n=n.substr(1,n.length());
			}

			variables->add_property("members/"+n,v);
		}
		ofs+=mcount*2;

		mcount = p_data[ofs];

		ofs++;
		for(int i=0;i<mcount;i++) {

			String n = p_data[ofs+i*2+0];
			Variant v = p_data[ofs+i*2+1];

			if (n.begins_with("*")) {

				n=n.substr(1,n.length());
			}


			variables->add_property("locals/"+n,v);
		}

		variables->update();
		inspector->edit(variables);

	} else if (p_msg=="output") {

		//OUT
		for(int i=0;i<p_data.size();i++) {

			String t = p_data[i];
			//LOG

			if (EditorNode::get_log()->is_hidden()) {
				log_forced_visible=true;
				EditorNode::get_log()->show();
			}
			EditorNode::get_log()->add_message(t);

		}

	} else if (p_msg=="performance") {
		Array arr = p_data[0];
		Vector<float> p;
		p.resize(arr.size());
		for(int i=0;i<arr.size();i++) {
			p[i]=arr[i];
			if (i<perf_items.size()) {
				perf_items[i]->set_text(1,rtos(p[i]));
				if (p[i]>perf_max[i])
					perf_max[i]=p[i];
			}

		}
		perf_history.push_front(p);
		perf_draw->update();

	} else if (p_msg=="error") {

		Array err = p_data[0];

		Array vals;
		vals.push_back(err[0]);
		vals.push_back(err[1]);
		vals.push_back(err[2]);
		vals.push_back(err[3]);

		bool warning = err[9];
		bool e;
		String time = String("%d:%02d:%02d:%04d").sprintf(vals,&e);
		String txt=time+" - "+String(err[8]);

		String tooltip="Type:"+String(warning?"Warning":"Error");
		tooltip+="\nDescription: "+String(err[8]);
		tooltip+="\nTime: "+time;
		tooltip+="\nC Error: "+String(err[7]);
		tooltip+="\nC Source: "+String(err[5])+":"+String(err[6]);
		tooltip+="\nC Function: "+String(err[4]);



		error_list->add_item(txt,EditorNode::get_singleton()->get_gui_base()->get_icon(warning?"Warning":"Error","EditorIcons"));
		error_list->set_item_tooltip( error_list->get_item_count() -1,tooltip );

		int scc = p_data[1];

		Array stack;
		stack.resize(scc);
		for(int i=0;i<scc;i++) {
			stack[i]=p_data[2+i];
		}

		error_list->set_item_metadata( error_list->get_item_count() -1,stack );

		error_count++;
		/*
		int count = p_data[1];

		Array cstack;

		OutputError oe = errors.front()->get();

		packet_peer_stream->put_var(oe.hr);
		packet_peer_stream->put_var(oe.min);
		packet_peer_stream->put_var(oe.sec);
		packet_peer_stream->put_var(oe.msec);
		packet_peer_stream->put_var(oe.source_func);
		packet_peer_stream->put_var(oe.source_file);
		packet_peer_stream->put_var(oe.source_line);
		packet_peer_stream->put_var(oe.error);
		packet_peer_stream->put_var(oe.error_descr);
		packet_peer_stream->put_var(oe.warning);
		packet_peer_stream->put_var(oe.callstack);
		*/
	} else if (p_msg=="kill_me") {

		editor->call_deferred("stop_child_process");
	}

}


void ScriptEditorDebugger::_performance_select(Object*,int,bool) {

	perf_draw->update();

}

void ScriptEditorDebugger::_performance_draw() {


	Vector<int> which;
	for(int i=0;i<perf_items.size();i++) {


		if (perf_items[i]->is_selected(0))
			which.push_back(i);
	}


	if(which.empty())
		return;

	Color graph_color=get_color("font_color","TextEdit");
	Ref<StyleBox> graph_sb = get_stylebox("normal","TextEdit");
	Ref<Font> graph_font = get_font("font","TextEdit");

	int cols = Math::ceil(Math::sqrt(which.size()));
	int rows = (which.size()+1)/cols;
	if (which.size()==1)
		rows=1;


	int margin =3;
	int point_sep=5;
	Size2i s = Size2i(perf_draw->get_size())/Size2i(cols,rows);
	for(int i=0;i<which.size();i++) {

		Point2i p(i%cols,i/cols);
		Rect2i r(p*s,s);
		r.pos+=Point2(margin,margin);
		r.size-=Point2(margin,margin)*2.0;
		perf_draw->draw_style_box(graph_sb,r);
		r.pos+=graph_sb->get_offset();
		r.size-=graph_sb->get_minimum_size();
		int pi=which[i];
		Color c = Color(0.7,0.9,0.5);
		c.set_hsv(Math::fmod(c.get_h()+pi*0.7654,1),c.get_s(),c.get_v());

		c.a=0.8;
		perf_draw->draw_string(graph_font,r.pos+Point2(0,graph_font->get_ascent()),perf_items[pi]->get_text(0),c,r.size.x);
		c.a=0.6;
		perf_draw->draw_string(graph_font,r.pos+Point2(graph_font->get_char_size('X').width,graph_font->get_ascent()+graph_font->get_height()),perf_items[pi]->get_text(1),c,r.size.y);

		float spacing=point_sep/float(cols);
		float from = r.size.width;

		List<Vector<float> >::Element *E=perf_history.front();
		float prev=-1;
		while(from>=0 && E) {

			float m = perf_max[pi];
			if (m==0)
				m=0.00001;
			float h = E->get()[pi]/m;
			h=(1.0-h)*r.size.y;

			c.a=0.7;
			if (E!=perf_history.front())
				perf_draw->draw_line(r.pos+Point2(from,h),r.pos+Point2(from+spacing,prev),c,2.0);
			prev=h;
			E=E->next();
			from-=spacing;
		}

	}

}

void ScriptEditorDebugger::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			step->set_icon( get_icon("DebugStep","EditorIcons"));
			next->set_icon( get_icon("DebugNext","EditorIcons"));
			back->set_icon( get_icon("Back","EditorIcons"));
			forward->set_icon( get_icon("Forward","EditorIcons"));
			dobreak->set_icon( get_icon("Pause","EditorIcons"));
			docontinue->set_icon( get_icon("DebugContinue","EditorIcons"));
			tb->set_normal_texture( get_icon("Close","EditorIcons"));
			tb->set_hover_texture( get_icon("CloseHover","EditorIcons"));
			tb->set_pressed_texture( get_icon("Close","EditorIcons"));
			scene_tree_refresh->set_icon( get_icon("Reload","EditorIcons"));
			le_set->connect("pressed",this,"_live_edit_set");
			le_clear->connect("pressed",this,"_live_edit_clear");
			error_list->connect("item_selected",this,"_error_selected");
			error_stack->connect("item_selected",this,"_error_stack_selected");
			vmem_refresh->set_icon( get_icon("Reload","EditorIcons"));

		} break;
		case NOTIFICATION_PROCESS: {

			if (error_count!=last_error_count) {

				if (error_count==0) {
					error_split->set_name("Errors");
				} else {
					error_split->set_name("Errors ("+itos(error_count)+")");
				}
				last_error_count=error_count;
			}
			if (connection.is_null()) {

				if (server->is_connection_available()) {

					connection = server->take_connection();
					if (connection.is_null())
						break;

					EditorNode::get_log()->add_message("** Debug Process Started **");
					log_forced_visible=false;

					ppeer->set_stream_peer(connection);


					show();
					dobreak->set_disabled(false);
					tabs->set_current_tab(0);

					emit_signal("show_debugger",true);
					reason->set_text("Child Process Connected");
					reason->set_tooltip("Child Process Connected");
					scene_tree->clear();
					le_set->set_disabled(true);
					le_clear->set_disabled(false);
					error_list->clear();
					error_stack->clear();
					error_count=0;
					//live_edit_root->set_text("/root");

					update_live_edit_root();

				} else {

					break;
				}
			};

			if (!connection->is_connected()) {
				stop();
				editor->notify_child_process_exited(); //somehow, exited
				break;
			};

			if (ppeer->get_available_packet_count() <= 0) {
				break;
			};

			while(ppeer->get_available_packet_count() > 0) {

				if (pending_in_queue) {

					int todo = MIN( ppeer->get_available_packet_count(), pending_in_queue );

					for(int i=0;i<todo;i++) {

						Variant cmd;
						Error ret = ppeer->get_var(cmd);
						if (ret!=OK) {
							stop();
							ERR_FAIL_COND(ret!=OK);
						}

						message.push_back(cmd);
						pending_in_queue--;
					}


					if (pending_in_queue==0) {
						_parse_message(message_type,message);
						message.clear();

					}


				} else {

					if (ppeer->get_available_packet_count()>=2) {


						Variant cmd;
						Error ret = ppeer->get_var(cmd);
						if (ret!=OK) {
							stop();
							ERR_FAIL_COND(ret!=OK);
						}
						if (cmd.get_type()!=Variant::STRING) {
							stop();
							ERR_FAIL_COND(cmd.get_type()!=Variant::STRING);
						}

						message_type=cmd;

						ret = ppeer->get_var(cmd);
						if (ret!=OK) {
							stop();
							ERR_FAIL_COND(ret!=OK);
						}
						if (cmd.get_type()!=Variant::INT) {
							stop();
							ERR_FAIL_COND(cmd.get_type()!=Variant::INT);
						}

						pending_in_queue=cmd;

						if (pending_in_queue==0) {
							_parse_message(message_type,Array());
							message.clear();
						}

					} else {


						break;
					}

				}
			}



		} break;
	}

}


void ScriptEditorDebugger::start() {

	stop();


	uint16_t port = GLOBAL_DEF("debug/remote_port",6007);
	perf_history.clear();
	for(int i=0;i<Performance::MONITOR_MAX;i++) {

		perf_max[i]=0;
	}

	server->listen(port);
	set_process(true);

}

void ScriptEditorDebugger::pause(){


}

void ScriptEditorDebugger::unpause(){


}

void ScriptEditorDebugger::stop(){


	set_process(false);

	server->stop();

	ppeer->set_stream_peer(Ref<StreamPeer>());

	if (connection.is_valid()) {
		EditorNode::get_log()->add_message("** Debug Process Stopped **");
		connection.unref();
	}

	pending_in_queue=0;
	message.clear();

	if (log_forced_visible) {
		EditorNode::get_log()->hide();
		log_forced_visible=false;
	}

	node_path_cache.clear();
	res_path_cache.clear();
	le_clear->set_disabled(false);
	le_set->set_disabled(true);


	hide();
	emit_signal("show_debugger",false);

}


void ScriptEditorDebugger::_stack_dump_frame_selected() {

	TreeItem *ti = stack_dump->get_selected();
	if (!ti)
		return;


	Dictionary d = ti->get_metadata(0);

	Ref<Script> s = ResourceLoader::load(d["file"]);
	emit_signal("goto_script_line",s,int(d["line"])-1);

	ERR_FAIL_COND(connection.is_null());
	ERR_FAIL_COND(!connection->is_connected());
	///

	Array msg;
	msg.push_back("get_stack_frame_vars");
	msg.push_back(d["frame"]);
	ppeer->put_var(msg);

}

void ScriptEditorDebugger::_hide_request() {

	hide();
	emit_signal("show_debugger",false);

}

void ScriptEditorDebugger::_output_clear() {

	//output->clear();
	//output->push_color(Color(0,0,0));

}

String ScriptEditorDebugger::get_var_value(const String& p_var) const {
	if (!breaked)
		return String();
	return variables->get_var_value(p_var);
}

int ScriptEditorDebugger::_get_node_path_cache(const NodePath& p_path) {

	const int *r = node_path_cache.getptr(p_path);
	if (r)
		return *r;

	last_path_id++;

	node_path_cache[p_path]=last_path_id;
	Array msg;
	msg.push_back("live_node_path");
	msg.push_back(p_path);
	msg.push_back(last_path_id);
	ppeer->put_var(msg);


	return last_path_id;
}

int ScriptEditorDebugger::_get_res_path_cache(const String& p_path) {

	Map<String,int>::Element *E=res_path_cache.find(p_path);

	if (E)
		return E->get();

	last_path_id++;

	res_path_cache[p_path]=last_path_id;
	Array msg;
	msg.push_back("live_res_path");
	msg.push_back(p_path);
	msg.push_back(last_path_id);
	ppeer->put_var(msg);


	return last_path_id;
}

void ScriptEditorDebugger::_method_changed(Object*p_base,const StringName& p_name,VARIANT_ARG_DECLARE) {

	if (!p_base || !live_debug || !connection.is_valid() || !editor->get_edited_scene())
		return;

	Node *node = p_base->cast_to<Node>();

	VARIANT_ARGPTRS

	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		//no pointers, sorry
		if (argptr[i] && (argptr[i]->get_type()==Variant::OBJECT || argptr[i]->get_type()==Variant::_RID))
			return;
	}

	if (node) {

		NodePath path = editor->get_edited_scene()->get_path_to(node);
		int pathid = _get_node_path_cache(path);



		Array msg;
		msg.push_back("live_node_call");
		msg.push_back(pathid);
		msg.push_back(p_name);
		for(int i=0;i<VARIANT_ARG_MAX;i++) {
			//no pointers, sorry
			msg.push_back(*argptr[i]);
		}
		ppeer->put_var(msg);

		return;

	}

	Resource *res = p_base->cast_to<Resource>();

	if (res && res->get_path()!=String()) {

		String respath = res->get_path();
		int pathid = _get_res_path_cache(respath);

		Array msg;
		msg.push_back("live_res_call");
		msg.push_back(pathid);
		msg.push_back(p_name);
		for(int i=0;i<VARIANT_ARG_MAX;i++) {
			//no pointers, sorry
			msg.push_back(*argptr[i]);
		}
		ppeer->put_var(msg);

		return;
	}

	//print_line("method");
}

void ScriptEditorDebugger::_property_changed(Object*p_base,const StringName& p_property,const Variant& p_value){

	if (!p_base || !live_debug || !connection.is_valid() || !editor->get_edited_scene())
		return;

	Node *node = p_base->cast_to<Node>();

	if (node) {

		NodePath path = editor->get_edited_scene()->get_path_to(node);
		int pathid = _get_node_path_cache(path);


		if (p_value.is_ref()) {
			Ref<Resource> res = p_value;
			if (res.is_valid() && res->get_path()!=String()) {

				Array msg;
				msg.push_back("live_node_prop_res");
				msg.push_back(pathid);
				msg.push_back(p_property);
				msg.push_back(res->get_path());
				ppeer->put_var(msg);
			}
		} else {

			Array msg;
			msg.push_back("live_node_prop");
			msg.push_back(pathid);
			msg.push_back(p_property);
			msg.push_back(p_value);
			ppeer->put_var(msg);
		}


		return;

	}

	Resource *res = p_base->cast_to<Resource>();

	if (res && res->get_path()!=String()) {

		String respath = res->get_path();
		int pathid = _get_res_path_cache(respath);


		if (p_value.is_ref()) {
			Ref<Resource> res = p_value;
			if (res.is_valid() && res->get_path()!=String()) {

				Array msg;
				msg.push_back("live_res_prop_res");
				msg.push_back(pathid);
				msg.push_back(p_property);
				msg.push_back(res->get_path());
				ppeer->put_var(msg);
			}
		} else {

			Array msg;
			msg.push_back("live_res_prop");
			msg.push_back(pathid);
			msg.push_back(p_property);
			msg.push_back(p_value);
			ppeer->put_var(msg);
		}


		return;
	}


	//print_line("prop");
}

void ScriptEditorDebugger::_method_changeds(void *p_ud,Object*p_base,const StringName& p_name,VARIANT_ARG_DECLARE) {

	ScriptEditorDebugger *sed = (ScriptEditorDebugger*)p_ud;
	sed->_method_changed(p_base,p_name,VARIANT_ARG_PASS);


}

void ScriptEditorDebugger::_property_changeds(void *p_ud,Object*p_base,const StringName& p_property,const Variant& p_value){

	ScriptEditorDebugger *sed = (ScriptEditorDebugger*)p_ud;
	sed->_property_changed(p_base,p_property,p_value);

}

void ScriptEditorDebugger::set_live_debugging(bool p_enable) {

	live_debug=p_enable;
}

void ScriptEditorDebugger::_live_edit_set() {

	if (!connection.is_valid())
		return;

	TreeItem* ti = scene_tree->get_selected();
	if (!ti)
		return;
	String path;

	while(ti) {
		String lp=ti->get_text(0);
		path="/"+lp+path;
		ti=ti->get_parent();

	}

	NodePath np = path;

	editor->get_editor_data().set_edited_scene_live_edit_root(np);

	update_live_edit_root();


}

void ScriptEditorDebugger::_live_edit_clear() {

	NodePath np = NodePath("/root");
	editor->get_editor_data().set_edited_scene_live_edit_root(np);

	update_live_edit_root();

}

void ScriptEditorDebugger::update_live_edit_root() {

	NodePath np = editor->get_editor_data().get_edited_scene_live_edit_root();

	if (connection.is_valid()) {
		Array msg;
		msg.push_back("live_set_root");
		msg.push_back(np);
		if (editor->get_edited_scene())
			msg.push_back(editor->get_edited_scene()->get_filename());
		else
			msg.push_back("");
		ppeer->put_var(msg);
	}
	live_edit_root->set_text(np);

}

void ScriptEditorDebugger::live_debug_create_node(const NodePath& p_parent,const String& p_type,const String& p_name) {

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_create_node");
		msg.push_back(p_parent);
		msg.push_back(p_type);
		msg.push_back(p_name);
		ppeer->put_var(msg);
	}
}

void ScriptEditorDebugger::live_debug_instance_node(const NodePath& p_parent,const String& p_path,const String& p_name){

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_instance_node");
		msg.push_back(p_parent);
		msg.push_back(p_path);
		msg.push_back(p_name);
		ppeer->put_var(msg);
	}

}
void ScriptEditorDebugger::live_debug_remove_node(const NodePath& p_at){

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_remove_node");
		msg.push_back(p_at);
		ppeer->put_var(msg);
	}

}
void ScriptEditorDebugger::live_debug_remove_and_keep_node(const NodePath& p_at,ObjectID p_keep_id) {

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_remove_and_keep_node");
		msg.push_back(p_at);
		msg.push_back(p_keep_id);
		ppeer->put_var(msg);
	}

}
void ScriptEditorDebugger::live_debug_restore_node(ObjectID p_id, const NodePath& p_at, int p_at_pos){

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_restore_node");
		msg.push_back(p_id);
		msg.push_back(p_at);
		msg.push_back(p_at_pos);
		ppeer->put_var(msg);
	}

}
void ScriptEditorDebugger::live_debug_duplicate_node(const NodePath& p_at,const String& p_new_name){

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_duplicate_node");
		msg.push_back(p_at);
		msg.push_back(p_new_name);
		ppeer->put_var(msg);
	}

}
void ScriptEditorDebugger::live_debug_reparent_node(const NodePath& p_at, const NodePath& p_new_place, const String &p_new_name, int p_at_pos){

	if (live_debug && connection.is_valid()) {
		Array msg;
		msg.push_back("live_reparent_node");
		msg.push_back(p_at);
		msg.push_back(p_new_place);
		msg.push_back(p_new_name);
		msg.push_back(p_at_pos);
		ppeer->put_var(msg);
	}

}

void ScriptEditorDebugger::set_breakpoint(const String& p_path,int p_line,bool p_enabled) {

	if (connection.is_valid()) {
	       Array msg;
	       msg.push_back("breakpoint");
	       msg.push_back(p_path);
	       msg.push_back(p_line);
	       msg.push_back(p_enabled);
	       ppeer->put_var(msg);
       }
}


void ScriptEditorDebugger::_error_selected(int p_idx) {

	error_stack->clear();
	Array st=error_list->get_item_metadata(p_idx);
	for(int i=0;i<st.size();i+=2) {

		String script=st[i];
		int line=st[i+1];
		Array md;
		md.push_back(st[i]);
		md.push_back(st[i+1]);

		String str = script.get_file()+":"+itos(line);

		error_stack->add_item(str);
		error_stack->set_item_metadata(error_stack->get_item_count()-1,md);
		error_stack->set_item_tooltip(error_stack->get_item_count()-1,"File: "+String(st[i])+"\nLine: "+itos(line));
	}
}

void ScriptEditorDebugger:: _error_stack_selected(int p_idx){

	Array arr = error_stack->get_item_metadata(p_idx);
	if (arr.size()!=2)
		return;


	Ref<Script> s = ResourceLoader::load(arr[0]);
	emit_signal("goto_script_line",s,int(arr[1])-1);

}


void ScriptEditorDebugger::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_stack_dump_frame_selected"),&ScriptEditorDebugger::_stack_dump_frame_selected);
	ObjectTypeDB::bind_method(_MD("debug_next"),&ScriptEditorDebugger::debug_next);
	ObjectTypeDB::bind_method(_MD("debug_step"),&ScriptEditorDebugger::debug_step);
	ObjectTypeDB::bind_method(_MD("debug_break"),&ScriptEditorDebugger::debug_break);
	ObjectTypeDB::bind_method(_MD("debug_continue"),&ScriptEditorDebugger::debug_continue);
	ObjectTypeDB::bind_method(_MD("_output_clear"),&ScriptEditorDebugger::_output_clear);
	ObjectTypeDB::bind_method(_MD("_hide_request"),&ScriptEditorDebugger::_hide_request);
	ObjectTypeDB::bind_method(_MD("_performance_draw"),&ScriptEditorDebugger::_performance_draw);
	ObjectTypeDB::bind_method(_MD("_performance_select"),&ScriptEditorDebugger::_performance_select);
	ObjectTypeDB::bind_method(_MD("_scene_tree_request"),&ScriptEditorDebugger::_scene_tree_request);
	ObjectTypeDB::bind_method(_MD("_video_mem_request"),&ScriptEditorDebugger::_video_mem_request);
	ObjectTypeDB::bind_method(_MD("_live_edit_set"),&ScriptEditorDebugger::_live_edit_set);
	ObjectTypeDB::bind_method(_MD("_live_edit_clear"),&ScriptEditorDebugger::_live_edit_clear);

	ObjectTypeDB::bind_method(_MD("_error_selected"),&ScriptEditorDebugger::_error_selected);
	ObjectTypeDB::bind_method(_MD("_error_stack_selected"),&ScriptEditorDebugger::_error_stack_selected);

	ObjectTypeDB::bind_method(_MD("live_debug_create_node"),&ScriptEditorDebugger::live_debug_create_node);
	ObjectTypeDB::bind_method(_MD("live_debug_instance_node"),&ScriptEditorDebugger::live_debug_instance_node);
	ObjectTypeDB::bind_method(_MD("live_debug_remove_node"),&ScriptEditorDebugger::live_debug_remove_node);
	ObjectTypeDB::bind_method(_MD("live_debug_remove_and_keep_node"),&ScriptEditorDebugger::live_debug_remove_and_keep_node);
	ObjectTypeDB::bind_method(_MD("live_debug_restore_node"),&ScriptEditorDebugger::live_debug_restore_node);
	ObjectTypeDB::bind_method(_MD("live_debug_duplicate_node"),&ScriptEditorDebugger::live_debug_duplicate_node);
	ObjectTypeDB::bind_method(_MD("live_debug_reparent_node"),&ScriptEditorDebugger::live_debug_reparent_node);

	ADD_SIGNAL(MethodInfo("goto_script_line"));
	ADD_SIGNAL(MethodInfo("breaked",PropertyInfo(Variant::BOOL,"reallydid")));
	ADD_SIGNAL(MethodInfo("show_debugger",PropertyInfo(Variant::BOOL,"reallydid")));
}

ScriptEditorDebugger::ScriptEditorDebugger(EditorNode *p_editor){



	ppeer = Ref<PacketPeerStream>( memnew( PacketPeerStream ) );
	editor=p_editor;

	tabs = memnew( TabContainer );
	tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	tabs->set_area_as_parent_rect();
	add_child(tabs);

	tb = memnew( TextureButton );
	tb->connect("pressed",this,"_hide_request");
	tb->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,20);
	tb->set_margin(MARGIN_TOP,2);
	add_child(tb);



	VBoxContainer *vbc = memnew( VBoxContainer );
	vbc->set_name("Debugger");
	//tabs->add_child(vbc);
	Control *dbg=vbc;

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_child(hbc);


	reason = memnew( Label );
	reason->set_text("");
	hbc->add_child(reason);
	reason->add_color_override("font_color",Color(1,0.4,0.0,0.8));
	reason->set_h_size_flags(SIZE_EXPAND_FILL);
	reason->set_clip_text(true);

	hbc->add_child( memnew( VSeparator) );

	step = memnew( Button );
	hbc->add_child(step);
	step->set_tooltip("Step Into");
	step->connect("pressed",this,"debug_step");

	next = memnew( Button );
	hbc->add_child(next);
	next->set_tooltip("Step Over");
	next->connect("pressed",this,"debug_next");

	hbc->add_child( memnew( VSeparator) );

	dobreak = memnew( Button );
	hbc->add_child(dobreak);
	dobreak->set_tooltip("Break");
	dobreak->connect("pressed",this,"debug_break");

	docontinue = memnew( Button );
	hbc->add_child(docontinue);
	docontinue->set_tooltip("Continue");
	docontinue->connect("pressed",this,"debug_continue");

	hbc->add_child( memnew( VSeparator) );

	back = memnew( Button );
	hbc->add_child(back);
	back->set_tooltip("Inspect Previous Instance");

	forward = memnew( Button );
	hbc->add_child(forward);
	back->set_tooltip("Inspect Next Instance");


	HSplitContainer *sc = memnew( HSplitContainer );
	vbc->add_child(sc);
	sc->set_v_size_flags(SIZE_EXPAND_FILL);

	stack_dump = memnew( Tree );
	stack_dump->set_columns(1);
	stack_dump->set_column_titles_visible(true);
	stack_dump->set_column_title(0,"Stack Frames");
	stack_dump->set_h_size_flags(SIZE_EXPAND_FILL);
	stack_dump->set_hide_root(true);
	stack_dump->connect("cell_selected",this,"_stack_dump_frame_selected");
	sc->add_child(stack_dump);

	inspector = memnew( PropertyEditor );
	inspector->set_h_size_flags(SIZE_EXPAND_FILL);
	inspector->hide_top_label();
	inspector->get_scene_tree()->set_column_title(0,"Variable");
	inspector->set_capitalize_paths(false);
	inspector->set_read_only(true);
	sc->add_child(inspector);

	server = TCP_Server::create_ref();

	pending_in_queue=0;

	variables = memnew( ScriptEditorDebuggerVariables );
	inspector->edit(variables);
	breaked=false;

	tabs->add_child(dbg);
	//tabs->move_child(vbc,0);

	hbc = memnew( HBoxContainer );
	vbc->add_child(hbc);


	error_split = memnew( HSplitContainer );
	VBoxContainer *errvb = memnew( VBoxContainer );
	errvb->set_h_size_flags(SIZE_EXPAND_FILL);
	error_list = memnew( ItemList );
	errvb->add_margin_child("Errors:",error_list,true);
	error_split->add_child(errvb);

	errvb = memnew( VBoxContainer );
	errvb->set_h_size_flags(SIZE_EXPAND_FILL);
	error_stack = memnew( ItemList );
	errvb->add_margin_child("Stack Trace (if applies):",error_stack,true);
	error_split->add_child(errvb);

	error_split->set_name("Errors");
	tabs->add_child(error_split);


	HSplitContainer *hsp = memnew( HSplitContainer );

	perf_monitors = memnew(Tree);
	perf_monitors->set_columns(2);
	perf_monitors->set_column_title(0,"Monitor");
	perf_monitors->set_column_title(1,"Value");
	perf_monitors->set_column_titles_visible(true);
	hsp->add_child(perf_monitors);
	perf_monitors->set_select_mode(Tree::SELECT_MULTI);
	perf_monitors->connect("multi_selected",this,"_performance_select");
	perf_draw = memnew( Control );
	perf_draw->connect("draw",this,"_performance_draw");
	hsp->add_child(perf_draw);
	hsp->set_name("Performance");
	hsp->set_split_offset(300);
	tabs->add_child(hsp);
	perf_max.resize(Performance::MONITOR_MAX);

	Map<String,TreeItem*> bases;
	TreeItem *root=perf_monitors->create_item();
	perf_monitors->set_hide_root(true);
	for(int i=0;i<Performance::MONITOR_MAX;i++) {

		String n = Performance::get_singleton()->get_monitor_name(Performance::Monitor(i));
		String base = n.get_slice("/",0);
		String name = n.get_slice("/",1);
		if (!bases.has(base)) {
			TreeItem *b = perf_monitors->create_item(root);
			b->set_text(0,base.capitalize());
			b->set_editable(0,false);
			b->set_selectable(0,false);
			bases[base]=b;
		}

		TreeItem *it = perf_monitors->create_item(bases[base]);
		it->set_editable(0,false);
		it->set_selectable(0,true);
		it->set_text(0,name.capitalize());
		perf_items.push_back(it);
		perf_max[i]=0;

	}

	VBoxContainer *vmem_vb = memnew( VBoxContainer );
	HBoxContainer *vmem_hb = memnew( HBoxContainer );
	Label *vmlb = memnew(Label("List of Video Memory Usage by Resource: ") );
	vmlb->set_h_size_flags(SIZE_EXPAND_FILL);
	vmem_hb->add_child( vmlb );
	vmem_hb->add_child( memnew(Label("Total: ")) );
	vmem_total = memnew( LineEdit );
	vmem_total->set_editable(false);
	vmem_total->set_custom_minimum_size(Size2(100,1));
	vmem_hb->add_child(vmem_total);
	vmem_refresh = memnew( Button );
	vmem_hb->add_child(vmem_refresh);
	vmem_vb->add_child(vmem_hb);
	vmem_refresh->connect("pressed",this,"_video_mem_request");

	MarginContainer *vmmc = memnew( MarginContainer );
	vmmc = memnew( MarginContainer );
	vmem_tree = memnew( Tree );
	vmem_tree->set_v_size_flags(SIZE_EXPAND_FILL);
	vmem_tree->set_h_size_flags(SIZE_EXPAND_FILL);
	vmmc->add_child(vmem_tree);
	vmmc->set_v_size_flags(SIZE_EXPAND_FILL);
	vmem_vb->add_child(vmmc);

	vmem_vb->set_name("Video Mem");
	vmem_tree->set_columns(4);
	vmem_tree->set_column_titles_visible(true);
	vmem_tree->set_column_title(0,"Resource Path");
	vmem_tree->set_column_expand(0,true);
	vmem_tree->set_column_expand(1,false);
	vmem_tree->set_column_title(1,"Type");
	vmem_tree->set_column_min_width(1,100);
	vmem_tree->set_column_expand(2,false);
	vmem_tree->set_column_title(2,"Format");
	vmem_tree->set_column_min_width(2,150);
	vmem_tree->set_column_expand(3,false);
	vmem_tree->set_column_title(3,"Usage");
	vmem_tree->set_column_min_width(3,80);
	vmem_tree->set_hide_root(true);

	tabs->add_child(vmem_vb);

	info = memnew( HSplitContainer );
	info->set_name("Info");
	tabs->add_child(info);

	VBoxContainer *info_left = memnew( VBoxContainer );
	info_left->set_h_size_flags(SIZE_EXPAND_FILL);
	info->add_child(info_left);
	clicked_ctrl = memnew( LineEdit );
	info_left->add_margin_child("Clicked Control:",clicked_ctrl);
	clicked_ctrl_type = memnew( LineEdit );
	info_left->add_margin_child("Clicked Control Type:",clicked_ctrl_type);

	live_edit_root = memnew( LineEdit );

	{
		HBoxContainer *lehb = memnew( HBoxContainer );
		Label *l = memnew( Label("Live Edit Root:") );
		lehb->add_child(l);
		l->set_h_size_flags(SIZE_EXPAND_FILL);
		le_set = memnew( Button("Set From Tree") );
		lehb->add_child(le_set);
		le_clear = memnew( Button("Clear") );
		lehb->add_child(le_clear);
		info_left->add_child(lehb);
		MarginContainer *mc = memnew( MarginContainer );
		mc->add_child(live_edit_root);
		info_left->add_child(mc);
		le_set->set_disabled(true);
		le_clear->set_disabled(true);
	}

	VBoxContainer *info_right = memnew(VBoxContainer);
	info_right->set_h_size_flags(SIZE_EXPAND_FILL);
	info->add_child(info_right);
	HBoxContainer *inforhb = memnew( HBoxContainer );
	info_right->add_child(inforhb);
	Label *l2 = memnew( Label("Scene Tree:" ) );
	l2->set_h_size_flags(SIZE_EXPAND_FILL);
	inforhb->add_child( l2 );
	Button *refresh = memnew( Button );
	inforhb->add_child(refresh);
	refresh->connect("pressed",this,"_scene_tree_request");
	scene_tree_refresh=refresh;
	MarginContainer *infomc = memnew( MarginContainer );
	info_right->add_child(infomc);
	infomc->set_v_size_flags(SIZE_EXPAND_FILL);
	scene_tree = memnew( Tree );
	infomc->add_child(scene_tree);


	msgdialog = memnew( AcceptDialog );
	add_child(msgdialog);

	hide();
	log_forced_visible=false;

	p_editor->get_undo_redo()->set_method_notify_callback(_method_changeds,this);
	p_editor->get_undo_redo()->set_property_notify_callback(_property_changeds,this);
	live_debug=false;
	last_path_id=false;
	error_count=0;
	last_error_count=0;


}

ScriptEditorDebugger::~ScriptEditorDebugger() {

//	inspector->edit(NULL);
	memdelete(variables);

	ppeer->set_stream_peer(Ref<StreamPeer>());

	server->stop();

}
