/*************************************************************************/
/*  script_editor_plugin.cpp                                             */
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
#include "script_editor_plugin.h"
#include "tools/editor/editor_settings.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "tools/editor/editor_node.h"
#include "tools/editor/script_editor_debugger.h"
#include "globals.h"
#include "os/file_access.h"
#include "scene/main/viewport.h"
#include "os/keyboard.h"
#include "os/input.h"

/*** SCRIPT EDITOR ****/


class EditorScriptCodeCompletionCache : public ScriptCodeCompletionCache {


	struct Cache {
		uint64_t time_loaded;
		RES cache;
	};

	Map<String,Cache> cached;


public:

	uint64_t max_time_cache;
	int max_cache_size;

	void cleanup() {

		List< Map<String,Cache>::Element * > to_clean;


		Map<String,Cache>::Element *I=cached.front();
		while(I) {
			if ((OS::get_singleton()->get_ticks_msec()-I->get().time_loaded)>max_time_cache) {
				to_clean.push_back(I);
			}
			I=I->next();
		}

		while(to_clean.front()) {
			cached.erase(to_clean.front()->get());
			to_clean.pop_front();
		}
	}

	RES get_cached_resource(const String& p_path) {

		Map<String,Cache>::Element *E=cached.find(p_path);
		if (!E) {

			Cache c;
			c.cache=ResourceLoader::load(p_path);
			E=cached.insert(p_path,c);
		}

		E->get().time_loaded=OS::get_singleton()->get_ticks_msec();

		if (cached.size()>max_cache_size) {
			uint64_t older;
			Map<String,Cache>::Element *O=cached.front();
			older=O->get().time_loaded;
			Map<String,Cache>::Element *I=O;
			while(I) {
				if (I->get().time_loaded<older) {
					older = I->get().time_loaded;
					O=I;
				}
				I=I->next();
			}

			if (O!=E) {//should never heppane..
				cached.erase(O);
			}
		}

		return E->get().cache;
	}


	EditorScriptCodeCompletionCache() {

		max_cache_size=128;
		max_time_cache=5*60*1000; //minutes, five
	}

};

#define SORT_SCRIPT_LIST

void ScriptEditorQuickOpen::popup(const Vector<String>& p_functions, bool p_dontclear) {

	popup_centered_ratio(0.6);
	if (p_dontclear)
		search_box->select_all();
	else
		search_box->clear();
	search_box->grab_focus();
	functions=p_functions;
	_update_search();


}


void ScriptEditorQuickOpen::_text_changed(const String& p_newtext) {

	_update_search();
}

void ScriptEditorQuickOpen::_sbox_input(const InputEvent& p_ie) {

	if (p_ie.type==InputEvent::KEY && (
		p_ie.key.scancode == KEY_UP ||
		p_ie.key.scancode == KEY_DOWN ||
		p_ie.key.scancode == KEY_PAGEUP ||
		p_ie.key.scancode == KEY_PAGEDOWN ) ) {

		search_options->call("_input_event",p_ie);
		search_box->accept_event();
	}

}



void ScriptEditorQuickOpen::_update_search() {


	search_options->clear();
	TreeItem *root = search_options->create_item();

	for(int i=0;i<functions.size();i++) {

		String file = functions[i];
		if ((search_box->get_text()=="" || file.findn(search_box->get_text())!=-1)) {

			TreeItem *ti = search_options->create_item(root);
			ti->set_text(0,file);
			if (root->get_children()==ti)
				ti->select(0);

		}
	}

	get_ok()->set_disabled(root->get_children()==NULL);

}

void ScriptEditorQuickOpen::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;
	int line = ti->get_text(0).get_slice(":",1).to_int();

	emit_signal("goto_line",line-1);
	hide();
}

void ScriptEditorQuickOpen::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		connect("confirmed",this,"_confirmed");


	}
}




void ScriptEditorQuickOpen::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_text_changed"),&ScriptEditorQuickOpen::_text_changed);
	ObjectTypeDB::bind_method(_MD("_confirmed"),&ScriptEditorQuickOpen::_confirmed);
	ObjectTypeDB::bind_method(_MD("_sbox_input"),&ScriptEditorQuickOpen::_sbox_input);

	ADD_SIGNAL(MethodInfo("goto_line",PropertyInfo(Variant::INT,"line")));

}


ScriptEditorQuickOpen::ScriptEditorQuickOpen() {


	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);
	search_box = memnew( LineEdit );
	vbc->add_margin_child("Search:",search_box);
	search_box->connect("text_changed",this,"_text_changed");
	search_box->connect("input_event",this,"_sbox_input");
	search_options = memnew( Tree );
	vbc->add_margin_child("Matches:",search_options,true);
	get_ok()->set_text("Open");
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated",this,"_confirmed");
	search_options->set_hide_root(true);
}


/////////////////////////////////

ScriptEditor *ScriptEditor::script_editor=NULL;

Vector<String> ScriptTextEditor::get_functions()  {


	String errortxt;
	int line=-1,col;
	TextEdit *te=get_text_edit();
	String text = te->get_text();
	List<String> fnc;

	if (script->get_language()->validate(text,line,col,errortxt,script->get_path(),&fnc)) {

		//if valid rewrite functions to latest
		functions.clear();
		for (List<String>::Element *E=fnc.front();E;E=E->next()) {

			functions.push_back(E->get());
		}


	}

	return functions;
}

void ScriptTextEditor::apply_code() {

	if (script.is_null())
		return;
//	print_line("applying code");
	script->set_source_code(get_text_edit()->get_text());
	script->update_exports();
}

Ref<Script> ScriptTextEditor::get_edited_script() const {

	return script;
}

void ScriptTextEditor::_load_theme_settings() {

	get_text_edit()->clear_colors();

	/* keyword color */


	get_text_edit()->set_custom_bg_color(EDITOR_DEF("text_editor/background_color",Color(0,0,0,0)));
	get_text_edit()->add_color_override("font_color",EDITOR_DEF("text_editor/text_color",Color(0,0,0)));
	get_text_edit()->add_color_override("font_selected_color",EDITOR_DEF("text_editor/text_selected_color",Color(1,1,1)));
	get_text_edit()->add_color_override("selection_color",EDITOR_DEF("text_editor/selection_color",Color(0.2,0.2,1)));
	get_text_edit()->add_color_override("brace_mismatch_color",EDITOR_DEF("text_editor/brace_mismatch_color",Color(1,0.2,0.2)));
	get_text_edit()->add_color_override("current_line_color",EDITOR_DEF("text_editor/current_line_color",Color(0.3,0.5,0.8,0.15)));

	Color keyword_color= EDITOR_DEF("text_editor/keyword_color",Color(0.5,0.0,0.2));

	get_text_edit()->set_syntax_coloring(true);
	List<String> keywords;
	script->get_language()->get_reserved_words(&keywords);
	for(List<String>::Element *E=keywords.front();E;E=E->next()) {

		get_text_edit()->add_keyword_color(E->get(),keyword_color);
	}

	//colorize core types
	Color basetype_color= EDITOR_DEF("text_editor/base_type_color",Color(0.3,0.3,0.0));

	get_text_edit()->add_keyword_color("Vector2",basetype_color);
	get_text_edit()->add_keyword_color("Vector3",basetype_color);
	get_text_edit()->add_keyword_color("Plane",basetype_color);
	get_text_edit()->add_keyword_color("Quat",basetype_color);
	get_text_edit()->add_keyword_color("AABB",basetype_color);
	get_text_edit()->add_keyword_color("Matrix3",basetype_color);
	get_text_edit()->add_keyword_color("Transform",basetype_color);
	get_text_edit()->add_keyword_color("Color",basetype_color);
	get_text_edit()->add_keyword_color("Image",basetype_color);
	get_text_edit()->add_keyword_color("InputEvent",basetype_color);

	//colorize engine types
	Color type_color= EDITOR_DEF("text_editor/engine_type_color",Color(0.0,0.2,0.4));

	List<StringName> types;
	ObjectTypeDB::get_type_list(&types);

	for(List<StringName>::Element *E=types.front();E;E=E->next()) {

		get_text_edit()->add_keyword_color(E->get(),type_color);
	}

	//colorize comments
	Color comment_color = EDITOR_DEF("text_editor/comment_color",Color::hex(0x797e7eff));
	List<String> comments;
	script->get_language()->get_comment_delimiters(&comments);

	for(List<String>::Element *E=comments.front();E;E=E->next()) {

		String comment = E->get();
		String beg = comment.get_slice(" ",0);
		String end = comment.get_slice_count(" ")>1?comment.get_slice(" ",1):String();

		get_text_edit()->add_color_region(beg,end,comment_color,end=="");
	}

	//colorize strings
	Color string_color = EDITOR_DEF("text_editor/string_color",Color::hex(0x6b6f00ff));
	List<String> strings;
	script->get_language()->get_string_delimiters(&strings);

	for (List<String>::Element *E=strings.front();E;E=E->next()) {

		String string = E->get();
		String beg = string.get_slice(" ",0);
		String end = string.get_slice_count(" ")>1?string.get_slice(" ",1):String();
		get_text_edit()->add_color_region(beg,end,string_color,end=="");
	}

	//colorize symbols
	Color symbol_color= EDITOR_DEF("text_editor/symbol_color",Color::hex(0x005291ff));
	get_text_edit()->set_symbol_color(symbol_color);

}


void ScriptTextEditor::reload_text() {

	ERR_FAIL_COND(script.is_null())	;

	TextEdit *te = get_text_edit();
	int column = te->cursor_get_column();
	int row = te->cursor_get_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(script->get_source_code());
	te->clear_undo_history();
	te->cursor_set_line(row);
	te->cursor_set_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	_line_col_changed();

}

void ScriptTextEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_READY) {

		//emit_signal("name_changed");
	}
}


bool ScriptTextEditor::is_unsaved()  {

	return get_text_edit()->get_version()!=get_text_edit()->get_saved_version();
}

String ScriptTextEditor::get_name()  {
	String name;

	if (script->get_path().find("local://")==-1 && script->get_path().find("::")==-1) {
		name=script->get_path().get_file();
		if (get_text_edit()->get_version()!=get_text_edit()->get_saved_version()) {
			name+="(*)";
		}
	} else if (script->get_name()!="")
		name=script->get_name();
	else
		name=script->get_type()+"("+itos(script->get_instance_ID())+")";

	return name;

}

Ref<Texture> ScriptTextEditor::get_icon() {

	if (get_parent_control() && get_parent_control()->has_icon(script->get_type(),"EditorIcons")) {
		return get_parent_control()->get_icon(script->get_type(),"EditorIcons");
	}

	return Ref<Texture>();
}



void ScriptTextEditor::set_edited_script(const Ref<Script>& p_script) {

	ERR_FAIL_COND(!script.is_null());

	script=p_script;


	_load_theme_settings();

	get_text_edit()->set_text(script->get_source_code());
	get_text_edit()->clear_undo_history();
	get_text_edit()->tag_saved_version();


	emit_signal("name_changed");
	_line_col_changed();
}


void ScriptTextEditor::_validate_script() {

	String errortxt;
	int line=-1,col;
	TextEdit *te=get_text_edit();

	String text = te->get_text();
	List<String> fnc;

	if (!script->get_language()->validate(text,line,col,errortxt,script->get_path(),&fnc)) {
		String error_text="error("+itos(line)+","+itos(col)+"): "+errortxt;
		set_error(error_text);
	} else {
		set_error("");
		line=-1;
		if (!script->is_tool()) {
			script->set_source_code(text);
			script->update_exports();
			//script->reload(); //will update all the variables in property editors
		}

		functions.clear();
		for (List<String>::Element *E=fnc.front();E;E=E->next()) {

			functions.push_back(E->get());
		}

	}

	line--;
	for(int i=0;i<te->get_line_count();i++) {
		te->set_line_as_marked(i,line==i);
	}

	emit_signal("name_changed");
}


static Node* _find_node_for_script(Node* p_base, Node*p_current, const Ref<Script>& p_script) {

	if (p_current->get_owner()!=p_base && p_base!=p_current)
		return NULL;
	Ref<Script> c = p_current->get_script();
	if (c==p_script)
		return p_current;
	for(int i=0;i<p_current->get_child_count();i++) {
		Node *found = _find_node_for_script(p_base,p_current->get_child(i),p_script);
		if (found)
			return found;
	}

	return NULL;
}

static void _find_changed_scripts_for_external_editor(Node* p_base, Node*p_current, Set<Ref<Script> > &r_scripts) {

	if (p_current->get_owner()!=p_base && p_base!=p_current)
		return;
	Ref<Script> c = p_current->get_script();

	if (c.is_valid())
		r_scripts.insert(c);

	for(int i=0;i<p_current->get_child_count();i++) {
		_find_changed_scripts_for_external_editor(p_base,p_current->get_child(i),r_scripts);
	}

}

void ScriptEditor::_update_modified_scripts_for_external_editor() {

	if (!bool(EditorSettings::get_singleton()->get("external_editor/use_external_editor")))
		return;

	Set<Ref<Script> > scripts;

	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		_find_changed_scripts_for_external_editor(base,base,scripts);
	}

	for (Set<Ref<Script> >::Element *E=scripts.front();E;E=E->next()) {

		Ref<Script> script = E->get();

		if (script->get_path()=="" || script->get_path().find("local://")!=-1 || script->get_path().find("::")!=-1) {

			continue; //internal script, who cares, though weird
		}

		uint64_t last_date = script->get_last_modified_time();
		uint64_t date = FileAccess::get_modified_time(script->get_path());

		if (last_date!=date) {

			Ref<Script> rel_script = ResourceLoader::load(script->get_path(),script->get_type(),true);
			ERR_CONTINUE(!rel_script.is_valid());
			script->set_source_code( rel_script->get_source_code() );
			script->set_last_modified_time( rel_script->get_last_modified_time() );
			script->update_exports();
		}

	}
}



void ScriptTextEditor::_code_complete_script(const String& p_code, List<String>* r_options) {

	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		base = _find_node_for_script(base,base,script);
	}
	String hint;
	Error err = script->get_language()->complete_code(p_code,script->get_path().get_base_dir(),base,r_options,hint);
	if (hint!="") {
		get_text_edit()->set_code_hint(hint);
		print_line("hint: "+hint.replace(String::chr(0xFFFF),"|"));
	}

}
void ScriptTextEditor::_bind_methods() {

	ADD_SIGNAL(MethodInfo("name_changed"));
}

ScriptTextEditor::ScriptTextEditor() {

	get_text_edit()->set_draw_tabs(true);
}

/*** SCRIPT EDITOR ******/

String ScriptEditor::_get_debug_tooltip(const String&p_text,Node *_ste) {

	ScriptTextEditor *ste=_ste->cast_to<ScriptTextEditor>();

	String val = debugger->get_var_value(p_text);
	if (val!=String()) {
		return p_text+": "+val;
	} else {

		return String();
	}
}

void ScriptEditor::_breaked(bool p_breaked,bool p_can_debug) {

	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_NEXT), !(p_breaked && p_can_debug));
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_STEP), !(p_breaked && p_can_debug) );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_BREAK), p_breaked );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), !p_breaked );

}

void ScriptEditor::_show_debugger(bool p_show) {

	debug_menu->get_popup()->set_item_checked( debug_menu->get_popup()->get_item_index(DEBUG_SHOW), p_show);
}

void ScriptEditor::_script_created(Ref<Script> p_script) {
	editor->push_item(p_script.operator->());
}

void ScriptEditor::_goto_script_line2(int p_line) {

	int selected = tab_container->get_current_tab();
	if (selected<0 || selected>=tab_container->get_child_count())
		return;

	ScriptTextEditor *current = tab_container->get_child(selected)->cast_to<ScriptTextEditor>();
	if (!current)
		return;

	current->get_text_edit()->cursor_set_line(p_line);

}

void ScriptEditor::_goto_script_line(REF p_script,int p_line) {


	editor->push_item(p_script.ptr());
	_goto_script_line2(p_line);

}


void ScriptEditor::_update_history_arrows() {

	script_back->set_disabled( history_pos<=0 );
	script_forward->set_disabled( history_pos>=history.size()-1 );
}


void ScriptEditor::_go_to_tab(int p_idx) {

	Node *cn = tab_container->get_child(p_idx);
	if (!cn)
		return;
	Control *c = cn->cast_to<Control>();
	if (!c)
		return;

	if (history_pos>=0 && history_pos<history.size() && history[history_pos].control==tab_container->get_current_tab_control()) {

		Node *n = tab_container->get_current_tab_control();

		if (n->cast_to<ScriptTextEditor>()) {

			history[history_pos].scroll_pos=n->cast_to<ScriptTextEditor>()->get_text_edit()->get_v_scroll();
			history[history_pos].cursor_column=n->cast_to<ScriptTextEditor>()->get_text_edit()->cursor_get_column();
			history[history_pos].cursor_row=n->cast_to<ScriptTextEditor>()->get_text_edit()->cursor_get_line();
		}
		if (n->cast_to<EditorHelp>()) {

			history[history_pos].scroll_pos=n->cast_to<EditorHelp>()->get_scroll();
		}
	}

	history.resize(history_pos+1);
	ScriptHistory sh;
	sh.control=c;
	sh.scroll_pos=0;

	history.push_back(sh);
	history_pos++;


	tab_container->set_current_tab(p_idx);

	c = tab_container->get_current_tab_control();

	if (c->cast_to<ScriptTextEditor>()) {

		script_name_label->set_text(c->cast_to<ScriptTextEditor>()->get_name());
		script_icon->set_texture(c->cast_to<ScriptTextEditor>()->get_icon());
		if (is_visible())
			c->cast_to<ScriptTextEditor>()->get_text_edit()->grab_focus();
	}
	if (c->cast_to<EditorHelp>()) {

		script_name_label->set_text(c->cast_to<EditorHelp>()->get_class_name());
		script_icon->set_texture(get_icon("Help","EditorIcons"));
		if (is_visible())
			c->cast_to<EditorHelp>()->set_focused();
	}



	c->set_meta("__editor_pass",++edit_pass);
	_update_history_arrows();
	_update_script_colors();
}

void ScriptEditor::_close_current_tab() {

	int selected = tab_container->get_current_tab();
	if (selected<0 || selected>=tab_container->get_child_count())
		return;
	
	Node *tselected = tab_container->get_child(selected);
	ScriptTextEditor *current = tab_container->get_child(selected)->cast_to<ScriptTextEditor>();
	if (current) {
		apply_scripts();
	}

	//remove from history
	history.resize(history_pos+1);

	for(int i=0;i<history.size();i++) {
		if (history[i].control==tselected) {
			history.remove(i);
			i--;
			history_pos--;
		}
	}

	if (history_pos>=history.size()) {
		history_pos=history.size()-1;
	}

	int idx = tab_container->get_current_tab();
	memdelete(tselected);
	if (idx>=tab_container->get_child_count())
		idx=tab_container->get_child_count()-1;
	if (idx>=0) {

		if (history_pos>=0) {
			idx = history[history_pos].control->get_index();
		}
		tab_container->set_current_tab(idx);

		//script_list->select(idx);
	}


	_update_history_arrows();



	_update_script_names();
	EditorNode::get_singleton()->save_layout();

}


void ScriptEditor::_resave_scripts(const String& p_str) {

	apply_scripts();

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;


		Ref<Script> script = ste->get_edited_script();

		if (script->get_path()=="" || script->get_path().find("local://")!=-1 || script->get_path().find("::")!=-1)
			continue; //internal script, who cares


		editor->save_resource(script);
		ste->get_text_edit()->tag_saved_version();
	}

	disk_changed->hide();

}

void ScriptEditor::_reload_scripts(){



	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste) {

			continue;
		}


		Ref<Script> script = ste->get_edited_script();

		if (script->get_path()=="" || script->get_path().find("local://")!=-1 || script->get_path().find("::")!=-1) {

			continue; //internal script, who cares
		}


		Ref<Script> rel_script = ResourceLoader::load(script->get_path(),script->get_type(),true);
		ERR_CONTINUE(!rel_script.is_valid());
		script->set_source_code( rel_script->get_source_code() );
		script->set_last_modified_time( rel_script->get_last_modified_time() );
		script->reload();
		ste->reload_text();


	}

	disk_changed->hide();
	_update_script_names();

}



void ScriptEditor::_res_saved_callback(const Ref<Resource>& p_res) {



	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste) {

			continue;
		}


		Ref<Script> script = ste->get_edited_script();

		if (script->get_path()=="" || script->get_path().find("local://")!=-1 || script->get_path().find("::")!=-1) {
			continue; //internal script, who cares
		}

		if (script==p_res) {

			ste->get_text_edit()->tag_saved_version();
		}

	}

	_update_script_names();

}

bool ScriptEditor::_test_script_times_on_disk() {


	disk_changed_list->clear();
	TreeItem *r = disk_changed_list->create_item();
	disk_changed_list->set_hide_root(true);

	bool need_ask=false;
	bool need_reload=false;
	bool use_autoreload=bool(EDITOR_DEF("text_editor/auto_reload_scripts_on_external_change",false));



	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (ste) {

			Ref<Script> script = ste->get_edited_script();

			if (script->get_path()=="" || script->get_path().find("local://")!=-1 || script->get_path().find("::")!=-1)
				continue; //internal script, who cares


			uint64_t last_date = script->get_last_modified_time();
			uint64_t date = FileAccess::get_modified_time(script->get_path());

			//printf("last date: %lli vs date: %lli\n",last_date,date);
			if (last_date!=date) {

				TreeItem *ti = disk_changed_list->create_item(r);
				ti->set_text(0,script->get_path().get_file());

				if (!use_autoreload || ste->is_unsaved()) {
					need_ask=true;
				}
				need_reload=true;
				//r->set_metadata(0,);
			}
		}
	}



	if (need_reload) {
		if (!need_ask) {
			script_editor->_reload_scripts();
			need_reload=false;
		} else {
			disk_changed->call_deferred("popup_centered_ratio",0.5);
		}
	}

	return need_reload;
}

void ScriptEditor::swap_lines(TextEdit *tx, int line1, int line2)
{
    String tmp = tx->get_line(line1);
    String tmp2 = tx->get_line(line2);
    tx->set_line(line2, tmp);
    tx->set_line(line1, tmp2);

    tx->cursor_set_line(line2);
}

void ScriptEditor::_menu_option(int p_option) {


	switch(p_option) {
		case FILE_NEW: {
			script_create_dialog->config("Node", ".gd");
			script_create_dialog->popup_centered(Size2(300, 300));
		} break;
		case FILE_OPEN: {

			editor->open_resource("Script");
			return;
		} break;
		case FILE_SAVE_ALL: {

			if (!_test_script_times_on_disk())
				return;

			for(int i=0;i<tab_container->get_child_count();i++) {

				ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
				if (!ste)
					continue;


				Ref<Script> script = ste->get_edited_script();

				if (script->get_path()=="" || script->get_path().find("local://")!=-1 || script->get_path().find("::")!=-1)
					continue; //internal script, who cares


				editor->save_resource( script );
			}


		} break;
		case SEARCH_HELP: {

			help_search_dialog->popup();
		} break;
		case SEARCH_CLASSES: {

			String current;

			if (tab_container->get_tab_count()>0) {
				EditorHelp *eh = tab_container->get_child( tab_container->get_current_tab() )->cast_to<EditorHelp>();
				if (eh) {
					current=eh->get_class_name();
				}
			}

			help_index->popup_centered_ratio(0.6);

			if (current!="") {
				help_index->call_deferred("select_class",current);
			}
		} break;
		case SEARCH_WEBSITE: {

			OS::get_singleton()->shell_open("http://www.godotengine.org/projects/godot-engine/wiki/Documentation#Tutorials");
		} break;

		case WINDOW_NEXT: {

			_history_forward();
		} break;
		case WINDOW_PREV: {
			_history_back();
		} break;
		case DEBUG_SHOW: {
			if (debugger) {
				bool visible = debug_menu->get_popup()->is_item_checked( debug_menu->get_popup()->get_item_index(DEBUG_SHOW) );
				debug_menu->get_popup()->set_item_checked( debug_menu->get_popup()->get_item_index(DEBUG_SHOW), !visible);
				if (visible)
					debugger->hide();
				else
					debugger->show();
			}
		} break;
		case DEBUG_SHOW_KEEP_OPEN: {
			bool visible = debug_menu->get_popup()->is_item_checked( debug_menu->get_popup()->get_item_index(DEBUG_SHOW_KEEP_OPEN) );
			if (debugger)
				debugger->set_hide_on_stop(visible);
			debug_menu->get_popup()->set_item_checked( debug_menu->get_popup()->get_item_index(DEBUG_SHOW_KEEP_OPEN), !visible);
		} break;
	}


	int selected = tab_container->get_current_tab();
	if (selected<0 || selected>=tab_container->get_child_count())
		return;

	ScriptTextEditor *current = tab_container->get_child(selected)->cast_to<ScriptTextEditor>();
	if (current) {

		switch(p_option) {
			case FILE_NEW: {
				script_create_dialog->config("Node", ".gd");
				script_create_dialog->popup_centered(Size2(300, 300));
			} break;
			case FILE_SAVE: {
				if (!_test_script_times_on_disk())
					return;
				editor->save_resource( current->get_edited_script() );

			} break;
			case FILE_SAVE_AS: {

				editor->save_resource_as( current->get_edited_script() );

			} break;
			case EDIT_UNDO: {
				current->get_text_edit()->undo();
				current->get_text_edit()->call_deferred("grab_focus");
			} break;
			case EDIT_REDO: {
				current->get_text_edit()->redo();
				current->get_text_edit()->call_deferred("grab_focus");
			} break;
			case EDIT_CUT: {

				current->get_text_edit()->cut();
				current->get_text_edit()->call_deferred("grab_focus");
			} break;
			case EDIT_COPY: {
				current->get_text_edit()->copy();
				current->get_text_edit()->call_deferred("grab_focus");

			} break;
			case EDIT_PASTE: {
				current->get_text_edit()->paste();
				current->get_text_edit()->call_deferred("grab_focus");

			} break;
			case EDIT_SELECT_ALL: {

				current->get_text_edit()->select_all();
				current->get_text_edit()->call_deferred("grab_focus");

			} break;
			case EDIT_MOVE_LINE_UP: {

				TextEdit *tx = current->get_text_edit();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;

				if (tx->is_selection_active())
				{
					int from_line = tx->get_selection_from_line();
					int from_col  = tx->get_selection_from_column();
					int to_line   = tx->get_selection_to_line();
					int to_column = tx->get_selection_to_column();

					for (int i = from_line; i <= to_line; i++)
					{
						int line_id = i;
						int next_id = i - 1;

						if (line_id == 0 || next_id < 0)
							return;

						swap_lines(tx, line_id, next_id);
					}
					int from_line_up = from_line > 0 ? from_line-1 : from_line;
					int to_line_up   = to_line   > 0 ? to_line-1   : to_line;
					tx->select(from_line_up, from_col, to_line_up, to_column);
				}
				else
				{
					int line_id = tx->cursor_get_line();
					int next_id = line_id - 1;

					if (line_id == 0 || next_id < 0)
						return;

					swap_lines(tx, line_id, next_id);
				}
				tx->update();

			} break;
			case EDIT_MOVE_LINE_DOWN: {

				TextEdit *tx = current->get_text_edit();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;

				if (tx->is_selection_active())
				{
					int from_line = tx->get_selection_from_line();
					int from_col  = tx->get_selection_from_column();
					int to_line   = tx->get_selection_to_line();
					int to_column = tx->get_selection_to_column();

					for (int i = to_line; i >= from_line; i--)
					{
						int line_id = i;
						int next_id = i + 1;

						if (line_id == tx->get_line_count()-1 || next_id > tx->get_line_count())
							return;

						swap_lines(tx, line_id, next_id);
					}
					int from_line_down = from_line < tx->get_line_count() ? from_line+1 : from_line;
					int to_line_down   = to_line   < tx->get_line_count() ? to_line+1   : to_line;
					tx->select(from_line_down, from_col, to_line_down, to_column);
				}
				else
				{
					int line_id = tx->cursor_get_line();
					int next_id = line_id + 1;

					if (line_id == tx->get_line_count()-1 || next_id > tx->get_line_count())
						return;

					swap_lines(tx, line_id, next_id);
				}
				tx->update();

			} break;
			case EDIT_INDENT_LEFT: {

				TextEdit *tx = current->get_text_edit();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;


				if (tx->is_selection_active())
				{
					int begin = tx->get_selection_from_line();
					int end = tx->get_selection_to_line();
					for (int i = begin; i <= end; i++)
					{
						String line_text = tx->get_line(i);
						// begins with tab
						if (line_text.begins_with("\t"))
						{
							line_text = line_text.substr(1, line_text.length());
							tx->set_line(i, line_text);
						}
						// begins with 4 spaces
						else if (line_text.begins_with("    "))
						{
							line_text = line_text.substr(4, line_text.length());
							tx->set_line(i, line_text);
						}
					}
				}
				else
				{
					int begin = tx->cursor_get_line();
					String line_text = tx->get_line(begin);
					// begins with tab
					if (line_text.begins_with("\t"))
					{
						line_text = line_text.substr(1, line_text.length());
						tx->set_line(begin, line_text);
					}
					// begins with 4 spaces
					else if (line_text.begins_with("    "))
					{
						line_text = line_text.substr(4, line_text.length());
						tx->set_line(begin, line_text);
					}
				}
				tx->update();
				//tx->deselect();

			} break;
			case EDIT_INDENT_RIGHT: {

				TextEdit *tx = current->get_text_edit();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;

				if (tx->is_selection_active())
				{
					int begin = tx->get_selection_from_line();
					int end = tx->get_selection_to_line();
					for (int i = begin; i <= end; i++)
					{
						String line_text = tx->get_line(i);
						line_text = '\t' + line_text;
						tx->set_line(i, line_text);
					}
				}
				else
				{
					int begin = tx->cursor_get_line();
					String line_text = tx->get_line(begin);
					line_text = '\t' + line_text;
					tx->set_line(begin, line_text);
				}
				tx->update();
				//tx->deselect();

			} break;
			case EDIT_CLONE_DOWN: {

				TextEdit *tx = current->get_text_edit();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;
				int line = tx->cursor_get_line();
				int next_line = line + 1;
				int column = tx->cursor_get_column();

				if (line >= tx->get_line_count() - 1)
					tx->set_line(line, tx->get_line(line) + "\n");

				String line_clone = tx->get_line(line);
				tx->insert_at(line_clone, next_line);
				tx->cursor_set_column(column);
				tx->update();

			} break;
			case EDIT_TOGGLE_COMMENT: {

				TextEdit *tx = current->get_text_edit();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;



				if (tx->is_selection_active())
				{
					int begin = tx->get_selection_from_line();
					int end = tx->get_selection_to_line();
					for (int i = begin; i <= end; i++)
					{
						String line_text = tx->get_line(i);

						if (line_text.begins_with("#"))
							line_text = line_text.substr(1, line_text.length());
						else
							line_text = "#" + line_text;
						tx->set_line(i, line_text);
					}
				}
				else
				{
					int begin = tx->cursor_get_line();
					String line_text = tx->get_line(begin);

					if (line_text.begins_with("#"))
						line_text = line_text.substr(1, line_text.length());
					else
						line_text = "#" + line_text;
					tx->set_line(begin, line_text);
				}
				tx->update();
				//tx->deselect();

			} break;
			case EDIT_COMPLETE: {

				current->get_text_edit()->query_code_comple();

			} break;
			case EDIT_AUTO_INDENT: {

				TextEdit *te = current->get_text_edit();
				String text = te->get_text();
				Ref<Script> scr = current->get_edited_script();
				if (scr.is_null())
					return;
				int begin,end;
				if (te->is_selection_active()) {
					begin=te->get_selection_from_line();
					end=te->get_selection_to_line();
				} else {
					begin=0;
					end=te->get_line_count()-1;
				}
				scr->get_language()->auto_indent_code(text,begin,end);
				te->set_text(text);


			} break;
			case SEARCH_FIND: {

				find_replace_dialog->set_text_edit(current->get_text_edit());
				find_replace_dialog->popup_search();
			} break;
			case SEARCH_FIND_NEXT: {

				find_replace_dialog->set_text_edit(current->get_text_edit());
				find_replace_dialog->search_next();
			} break;
			case SEARCH_REPLACE: {

				find_replace_dialog->set_text_edit(current->get_text_edit());
				find_replace_dialog->popup_replace();
			} break;
			case SEARCH_LOCATE_FUNCTION: {

				if (!current)
					return;
				quick_open->popup(current->get_functions());
			} break;
			case SEARCH_GOTO_LINE: {

				goto_line_dialog->popup_find_line(current->get_text_edit());
			} break;
			case DEBUG_TOGGLE_BREAKPOINT: {
				int line=current->get_text_edit()->cursor_get_line();
				bool dobreak = !current->get_text_edit()->is_line_set_as_breakpoint(line);
				current->get_text_edit()->set_line_as_breakpoint(line,dobreak);
				get_debugger()->set_breakpoint(current->get_edited_script()->get_path(),line+1,dobreak);
			} break;
			case DEBUG_NEXT: {

				if (debugger)
					debugger->debug_next();
			} break;
			case DEBUG_STEP: {

				if (debugger)
					debugger->debug_step();

			} break;
			case DEBUG_BREAK: {

				if (debugger)
					debugger->debug_break();

			} break;
			case DEBUG_CONTINUE: {

				if (debugger)
					debugger->debug_continue();

			} break;
			case HELP_CONTEXTUAL: {
				String text = current->get_text_edit()->get_selection_text();
				if (text == "")
					text = current->get_text_edit()->get_word_under_cursor();
				if (text != "")
					help_search_dialog->popup(text);
			} break;
			case FILE_CLOSE: {
				if (current->get_text_edit()->get_version()!=current->get_text_edit()->get_saved_version()) {
					erase_tab_confirm->set_text("Close and save changes?\n\""+current->get_name()+"\"");
					erase_tab_confirm->popup_centered_minsize();
				} else {
					_close_current_tab();
				}
			} break;
			case WINDOW_MOVE_LEFT: {

				if (tab_container->get_current_tab()>0) {
					tab_container->call_deferred("set_current_tab",tab_container->get_current_tab()-1);
					script_list->call_deferred("select",tab_container->get_current_tab()-1);
					tab_container->move_child(current,tab_container->get_current_tab()-1);
					_update_script_names();
				}
			} break;
			case WINDOW_MOVE_RIGHT: {

				if (tab_container->get_current_tab()<tab_container->get_child_count()-1) {
					tab_container->call_deferred("set_current_tab",tab_container->get_current_tab()+1);
					script_list->call_deferred("select",tab_container->get_current_tab()+1);
					tab_container->move_child(current,tab_container->get_current_tab()+1);
					_update_script_names();
				}


			} break;

			default: {

				if (p_option>=WINDOW_SELECT_BASE) {

					tab_container->set_current_tab(p_option-WINDOW_SELECT_BASE);
					script_list->select(p_option-WINDOW_SELECT_BASE);

				}
			}
		}
	}

	EditorHelp *help = tab_container->get_child(selected)->cast_to<EditorHelp>();
	if (help) {

		switch(p_option) {

			case SEARCH_FIND: {
				help->popup_search();
			} break;
			case SEARCH_FIND_NEXT: {
				help->search_again();
			} break;
			case FILE_CLOSE: {
				_close_current_tab();
			} break;


		}
	}


}

void ScriptEditor::_tab_changed(int p_which) {

	ensure_select_current();
}

void ScriptEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		editor->connect("play_pressed",this,"_editor_play");
		editor->connect("pause_pressed",this,"_editor_pause");
		editor->connect("stop_pressed",this,"_editor_stop");
		editor->connect("script_add_function_request",this,"_add_callback");
		editor->connect("resource_saved",this,"_res_saved_callback");
		script_list->connect("item_selected",this,"_script_selected");
		script_split->connect("dragged",this,"_script_split_dragged");
		autosave_timer->connect("timeout",this,"_autosave_scripts");
		{
			float autosave_time = EditorSettings::get_singleton()->get("text_editor/autosave_interval_secs");
			if (autosave_time>0) {
				autosave_timer->set_wait_time(autosave_time);
				autosave_timer->start();
			} else {
				autosave_timer->stop();
			}
		}

		EditorSettings::get_singleton()->connect("settings_changed",this,"_editor_settings_changed");
		help_search->set_icon(get_icon("Help","EditorIcons"));
		site_search->set_icon(get_icon("Godot","EditorIcons"));
		class_search->set_icon(get_icon("ClassList","EditorIcons"));

		script_forward->set_icon(get_icon("Forward","EditorIcons"));
		script_back->set_icon(get_icon("Back","EditorIcons"));




	}

	if (p_what==NOTIFICATION_READY) {

		get_tree()->connect("tree_changed",this,"_tree_changed");
		editor->connect("request_help",this,"_request_help");
	}

	if (p_what==NOTIFICATION_EXIT_TREE) {

		editor->disconnect("play_pressed",this,"_editor_play");
		editor->disconnect("pause_pressed",this,"_editor_pause");
		editor->disconnect("stop_pressed",this,"_editor_stop");

	}

	if (p_what==MainLoop::NOTIFICATION_WM_FOCUS_IN) {

		_test_script_times_on_disk();
		_update_modified_scripts_for_external_editor();
	}

	if (p_what==NOTIFICATION_PROCESS) {

	}

}

void ScriptEditor::edited_scene_changed() {

	_update_modified_scripts_for_external_editor();

}

static const Node * _find_node_with_script(const Node* p_node, const RefPtr & p_script)  {

	if (p_node->get_script()==p_script)
		return p_node;

	for(int i=0;i<p_node->get_child_count();i++) {

		const Node *result = _find_node_with_script(p_node->get_child(i),p_script);
		if (result)
			return result;
	}

	return NULL;
}

Dictionary ScriptEditor::get_state() const {


//	apply_scripts();

	Dictionary state;
#if 0
	Array paths;
	int open=-1;

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;


		Ref<Script> script = ste->get_edited_script();
		if (script->get_path()!="" && script->get_path().find("local://")==-1 && script->get_path().find("::")==-1) {

			paths.push_back(script->get_path());
		} else {


			const Node *owner = _find_node_with_script(get_tree()->get_root(),script.get_ref_ptr());
			if (owner)
				paths.push_back(owner->get_path());

		}

		if (i==tab_container->get_current_tab())
			open=i;
	}

	if (paths.size())
		state["sources"]=paths;
	if (open!=-1)
		state["current"]=open;

#endif
	return state;
}
void ScriptEditor::set_state(const Dictionary& p_state) {

#if 0
	print_line("attempt set state: "+String(Variant(p_state)));

	if (!p_state.has("sources"))
		return; //bleh

	Array sources = p_state["sources"];
	for(int i=0;i<sources.size();i++) {

		Variant source=sources[i];

		Ref<Script> script;

		if (source.get_type()==Variant::NODE_PATH) {


			Node *owner=get_tree()->get_root()->get_node(source);
			if (!owner)
				continue;

			script = owner->get_script();
		} else if (source.get_type()==Variant::STRING) {


			script = ResourceLoader::load(source,"Script");
		}


		if (script.is_null()) //ah well..
			continue;

		editor->call("_resource_selected",script);
	}

	if (p_state.has("current")) {
		tab_container->set_current_tab(p_state["current"]);
	}
#endif

}
void ScriptEditor::clear() {
#if 0
	List<ScriptTextEditor*> stes;
	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;
		stes.push_back(ste);

	}

	while(stes.size()) {

		memdelete(stes.front()->get());
		stes.pop_front();
	}

	int idx = tab_container->get_current_tab();
	if (idx>=tab_container->get_child_count())
		idx=tab_container->get_child_count()-1;
	if (idx>=0) {
		tab_container->set_current_tab(idx);
		script_list->select( script_list->find_metadata(idx) );
	}

#endif


}


void ScriptEditor::get_breakpoints(List<String> *p_breakpoints) {

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;

		List<int> bpoints;
		ste->get_text_edit()->get_breakpoints(&bpoints);

		Ref<Script> script = ste->get_edited_script();
		String base = script->get_path();
		ERR_CONTINUE( base.begins_with("local://") || base=="" );

		for(List<int>::Element *E=bpoints.front();E;E=E->next()) {

			p_breakpoints->push_back(base+":"+itos(E->get()+1));
		}
	}

}





void ScriptEditor::ensure_focus_current() {

	if (!is_inside_tree())
		return;

	int cidx = tab_container->get_current_tab();
	if (cidx<0 || cidx>=tab_container->get_tab_count());
	Control *c = tab_container->get_child(cidx)->cast_to<Control>();
	if (!c)
		return;
	ScriptTextEditor *ste = c->cast_to<ScriptTextEditor>();
	if (!ste)
		return;
	ste->get_text_edit()->grab_focus();
}

void ScriptEditor::_script_selected(int p_idx) {

	grab_focus_block = !Input::get_singleton()->is_mouse_button_pressed(1); //amazing hack, simply amazing

	_go_to_tab(script_list->get_item_metadata(p_idx));
	grab_focus_block=false;
}

void ScriptEditor::ensure_select_current() {


	if (tab_container->get_child_count() && tab_container->get_current_tab()>=0) {

		Node *current = tab_container->get_child(tab_container->get_current_tab());


		ScriptTextEditor *ste = current->cast_to<ScriptTextEditor>();
		if (ste) {

			Ref<Script> script = ste->get_edited_script();

			if (!grab_focus_block && is_inside_tree())
				ste->get_text_edit()->grab_focus();

			edit_menu->show();
			search_menu->show();
			script_search_menu->hide();


		}

		EditorHelp *eh = current->cast_to<EditorHelp>();

		if (eh) {
			edit_menu->hide();
			search_menu->hide();
			script_search_menu->show();

		}
	}





}

void ScriptEditor::_find_scripts(Node* p_base, Node* p_current, Set<Ref<Script> > &used) {
	if (p_current!=p_base && p_current->get_owner()!=p_base)
		return;

	if (p_current->get_script_instance()) {
		Ref<Script> scr = p_current->get_script();
		if (scr.is_valid())
			used.insert(scr);
	}

	for(int i=0;i<p_current->get_child_count();i++) {
		_find_scripts(p_base,p_current->get_child(i),used);
	}

}

struct _ScriptEditorItemData {

	String name;
	Ref<Texture> icon;
	int index;
	String tooltip;
	bool used;
	int category;


	bool operator<(const _ScriptEditorItemData& id) const {

		return category==id.category?name.nocasecmp_to(id.name)<0:category<id.category;
	}

};


void ScriptEditor::_update_script_colors() {

	bool enabled = EditorSettings::get_singleton()->get("text_editor/script_temperature_enabled");
	if (!enabled)
		return;

	int hist_size = EditorSettings::get_singleton()->get("text_editor/script_temperature_history_size");
	Color hot_color=EditorSettings::get_singleton()->get("text_editor/script_temperature_hot_color");
	Color cold_color=EditorSettings::get_singleton()->get("text_editor/script_temperature_cold_color");

	for(int i=0;i<script_list->get_item_count();i++) {

		int c = script_list->get_item_metadata(i);
		Node *n = tab_container->get_child(c);
		if (!n)
			continue;

		script_list->set_item_custom_bg_color(i,Color(0,0,0,0));
		if (!n->has_meta("__editor_pass")) {
			continue;
		}

		int pass=n->get_meta("__editor_pass");
		int h = edit_pass - pass;
		if (h>hist_size) {
			continue;
		}
		float v = Math::ease((edit_pass-pass)/float_t(hist_size),0.4);


		script_list->set_item_custom_bg_color(i,hot_color.linear_interpolate(cold_color,v));
	}
}

void ScriptEditor::_update_script_names() {

	waiting_update_names=false;
	Set<Ref<Script> > used;
	Node* edited = EditorNode::get_singleton()->get_edited_scene();
	if (edited) {
		_find_scripts(edited,edited,used);
	}

	script_list->clear();
	bool split_script_help = EditorSettings::get_singleton()->get("text_editor/group_help_pages");

	Vector<_ScriptEditorItemData> sedata;

	for(int i=0;i<tab_container->get_child_count();i++) {


		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (ste) {

			String name = ste->get_name();
			Ref<Texture> icon = ste->get_icon();
			String tooltip = ste->get_edited_script()->get_path();

			_ScriptEditorItemData sd;
			sd.icon=icon;
			sd.name=name;
			sd.tooltip=tooltip;
			sd.index=i;
			sd.used=used.has(ste->get_edited_script());
			sd.category=0;

			sedata.push_back(sd);
		}

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();
		if (eh) {

			String name = eh->get_class_name();
			Ref<Texture> icon = get_icon("Help","EditorIcons");
			String tooltip = name+" Class Reference";

			_ScriptEditorItemData sd;
			sd.icon=icon;
			sd.name=name;
			sd.tooltip=tooltip;
			sd.index=i;
			sd.used=false;
			sd.category=split_script_help?1:0;
			sedata.push_back(sd);

		}

	}

	sedata.sort();

	for(int i=0;i<sedata.size();i++) {

		script_list->add_item(sedata[i].name,sedata[i].icon);
		int index = script_list->get_item_count()-1;
		script_list->set_item_tooltip(index,sedata[i].tooltip);
		script_list->set_item_metadata(index,sedata[i].index);
		if (sedata[i].used) {

			script_list->set_item_custom_bg_color(index,Color(88/255.0,88/255.0,60/255.0));
		}
		if (tab_container->get_current_tab()==sedata[i].index) {
			script_list->select(index);
			script_name_label->set_text(sedata[i].name);
			script_icon->set_texture(sedata[i].icon);

		}
	}

	_update_script_colors();




}

void ScriptEditor::edit(const Ref<Script>& p_script) {

	if (p_script.is_null())
		return;

	// see if already has it

	bool open_dominant = EditorSettings::get_singleton()->get("text_editor/open_dominant_script_on_scene_change");

	if (p_script->get_path().is_resource_file() && bool(EditorSettings::get_singleton()->get("external_editor/use_external_editor"))) {

		String path = EditorSettings::get_singleton()->get("external_editor/exec_path");
		String flags = EditorSettings::get_singleton()->get("external_editor/exec_flags");
		List<String> args;
		flags=flags.strip_edges();
		if (flags!=String()) {
			Vector<String> flagss = flags.split(" ",false);
			for(int i=0;i<flagss.size();i++)
				args.push_back(flagss[i]);
		}

		args.push_back(Globals::get_singleton()->globalize_path(p_script->get_path()));
		Error err = OS::get_singleton()->execute(path,args,false);
		if (err==OK)
			return;
		WARN_PRINT("Couldn't open external text editor, using internal");
	}


	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;

		if (ste->get_edited_script()==p_script) {

			if (open_dominant || !EditorNode::get_singleton()->is_changing_scene()) {
				if (tab_container->get_current_tab()!=i) {
					_go_to_tab(i);
					script_list->select( script_list->find_metadata(i) );
				}
				if (is_visible())
					ste->get_text_edit()->grab_focus();
			}
			return;
		}
	}

	// doesn't have it, make a new one

	ScriptTextEditor *ste = memnew( ScriptTextEditor );
	ste->set_edited_script(p_script);
	ste->get_text_edit()->set_tooltip_request_func(this,"_get_debug_tooltip",ste);
	ste->get_text_edit()->set_auto_brace_completion(EditorSettings::get_singleton()->get("text_editor/auto_brace_complete"));
	tab_container->add_child(ste);
	_go_to_tab(tab_container->get_tab_count()-1);




	_update_script_names();
	ste->connect("name_changed",this,"_update_script_names");
	if (!restoring_layout) {
		EditorNode::get_singleton()->save_layout();
	}
}

void ScriptEditor::save_external_data() {

	apply_scripts();


	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;

		Ref<Script> script = ste->get_edited_script();
		if (script->get_path()!="" && script->get_path().find("local://")==-1 &&script->get_path().find("::")==-1) {
			//external script, save it
			editor->save_resource(script);
			//ResourceSaver::save(script->get_path(),script);
		}
	}

}

void ScriptEditor::apply_scripts() const {

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;
		ste->apply_code();
	}

}

void ScriptEditor::_editor_play() {

	debugger->start();
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_STEP), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_BREAK), false );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true );

    //debugger_gui->start_listening(Globals::get_singleton()->get("debug/debug_port"));
}

void ScriptEditor::_editor_pause() {


}
void ScriptEditor::_editor_stop() {

	debugger->stop();
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_STEP), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_BREAK), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true );
}


void ScriptEditor::_add_callback(Object *p_obj, const String& p_function, const StringArray& p_args) {

	print_line("add callback! hohoho");
	ERR_FAIL_COND(!p_obj);
	Ref<Script> script = p_obj->get_script();
	ERR_FAIL_COND( !script.is_valid() );

	editor->push_item(script.ptr());

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (!ste)
			continue;
		if (ste->get_edited_script()!=script)
			continue;

		String code = ste->get_text_edit()->get_text();
		int pos = script->get_language()->find_function(p_function,code);
		if (pos==-1) {
			//does not exist
			ste->get_text_edit()->deselect();
			pos=ste->get_text_edit()->get_line_count()+2;
			String func = script->get_language()->make_function("",p_function,p_args);
			//code=code+func;
			ste->get_text_edit()->cursor_set_line(pos+1);
			ste->get_text_edit()->cursor_set_column(1000000); //none shall be that big
			ste->get_text_edit()->insert_text_at_cursor("\n\n"+func);
		}

		_go_to_tab(i);
		ste->get_text_edit()->cursor_set_line(pos);
		ste->get_text_edit()->cursor_set_column(1);

		script_list->select( script_list->find_metadata(i) );

		break;

	}

}

void ScriptEditor::_editor_settings_changed() {

	print_line("settings changed");
	float autosave_time = EditorSettings::get_singleton()->get("text_editor/autosave_interval_secs");
	if (autosave_time>0) {
		autosave_timer->set_wait_time(autosave_time);
		autosave_timer->start();
	} else {
		autosave_timer->stop();
	}

}

void ScriptEditor::_autosave_scripts() {

	print_line("autosaving");
	save_external_data();
}

void ScriptEditor::_tree_changed() {

	if (waiting_update_names)
		return;

	waiting_update_names=true;
	call_deferred("_update_script_names");
}

void ScriptEditor::_script_split_dragged(float) {

	EditorNode::get_singleton()->save_layout();
}

void ScriptEditor::set_window_layout(Ref<ConfigFile> p_layout) {

	if (!bool(EDITOR_DEF("text_editor/restore_scripts_on_load",true))) {
		return;
	}

	if (!p_layout->has_section_key("ScriptEditor","open_scripts") && !p_layout->has_section_key("ScriptEditor","open_help"))
		return;

	Array scripts = p_layout->get_value("ScriptEditor","open_scripts");
	Array helps;
	if (p_layout->has_section_key("ScriptEditor","open_help"))
		helps=p_layout->get_value("ScriptEditor","open_help");

	restoring_layout=true;

	for(int i=0;i<scripts.size();i++) {

		String path = scripts[i];
		Ref<Script> scr = ResourceLoader::load(path);
		if (scr.is_valid()) {
			edit(scr);
		}
	}


	for(int i=0;i<helps.size();i++) {

		String path = helps[i];
		_help_class_open(path);
	}

	for(int i=0;i<tab_container->get_child_count();i++) {
		tab_container->get_child(i)->set_meta("__editor_pass",Variant());
	}


	if (p_layout->has_section_key("ScriptEditor","split_offset")) {
		script_split->set_split_offset(p_layout->get_value("ScriptEditor","split_offset"));
	}


	restoring_layout=false;

}

void ScriptEditor::get_window_layout(Ref<ConfigFile> p_layout) {

	Array scripts;
	Array helps;

	for(int i=0;i<tab_container->get_child_count();i++) {

		ScriptTextEditor *ste = tab_container->get_child(i)->cast_to<ScriptTextEditor>();
		if (ste) {

			String path = ste->get_edited_script()->get_path();
			if (!path.is_resource_file())
				continue;

			scripts.push_back(path);
		}

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (eh) {

			helps.push_back(eh->get_class_name());
		}


	}

	p_layout->set_value("ScriptEditor","open_scripts",scripts);
	p_layout->set_value("ScriptEditor","open_help",helps);
	p_layout->set_value("ScriptEditor","split_offset",script_split->get_split_offset());

}


void ScriptEditor::_help_class_open(const String& p_class) {


	for(int i=0;i<tab_container->get_child_count();i++) {

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (eh && eh->get_class_name()==p_class) {

			_go_to_tab(i);
			_update_script_names();
			return;
		}
	}

	EditorHelp * eh = memnew( EditorHelp );


	eh->set_name(p_class);
	tab_container->add_child(eh);
	_go_to_tab(tab_container->get_tab_count()-1);
	eh->go_to_class(p_class,0);
	eh->connect("go_to_help",this,"_help_class_goto");
	_update_script_names();

}

void ScriptEditor::_help_class_goto(const String& p_desc) {


	String cname=p_desc.get_slice(":",1);

	for(int i=0;i<tab_container->get_child_count();i++) {

		EditorHelp *eh = tab_container->get_child(i)->cast_to<EditorHelp>();

		if (eh && eh->get_class_name()==cname) {

			_go_to_tab(i);
			eh->go_to_help(p_desc);
			_update_script_names();
			return;
		}
	}

	EditorHelp * eh = memnew( EditorHelp );

	eh->set_name(cname);
	tab_container->add_child(eh);
	_go_to_tab(tab_container->get_tab_count()-1);
	eh->go_to_help(p_desc);
	eh->connect("go_to_help",this,"_help_class_goto");
	_update_script_names();

}

void ScriptEditor::_update_history_pos(int p_new_pos) {

	Node *n = tab_container->get_current_tab_control();

	if (n->cast_to<ScriptTextEditor>()) {

		history[history_pos].scroll_pos=n->cast_to<ScriptTextEditor>()->get_text_edit()->get_v_scroll();
		history[history_pos].cursor_column=n->cast_to<ScriptTextEditor>()->get_text_edit()->cursor_get_column();
		history[history_pos].cursor_row=n->cast_to<ScriptTextEditor>()->get_text_edit()->cursor_get_line();
	}
	if (n->cast_to<EditorHelp>()) {

		history[history_pos].scroll_pos=n->cast_to<EditorHelp>()->get_scroll();
	}

	history_pos=p_new_pos;
	tab_container->set_current_tab(history[history_pos].control->get_index());

	n = history[history_pos].control;

	if (n->cast_to<ScriptTextEditor>()) {

		n->cast_to<ScriptTextEditor>()->get_text_edit()->set_v_scroll(history[history_pos].scroll_pos);
		n->cast_to<ScriptTextEditor>()->get_text_edit()->cursor_set_column( history[history_pos].cursor_column );
		n->cast_to<ScriptTextEditor>()->get_text_edit()->cursor_set_line( history[history_pos].cursor_row );
		n->cast_to<ScriptTextEditor>()->get_text_edit()->grab_focus();
	}

	if (n->cast_to<EditorHelp>()) {

		n->cast_to<EditorHelp>()->set_scroll(history[history_pos].scroll_pos);
		n->cast_to<EditorHelp>()->set_focused();
	}

	n->set_meta("__editor_pass",++edit_pass);
	_update_script_names();
	_update_history_arrows();

}

void ScriptEditor::_history_forward() {

	if (history_pos<history.size()-1) {
		_update_history_pos(history_pos+1);
	}
}

void ScriptEditor::_history_back(){

	if (history_pos>0) {
		_update_history_pos(history_pos-1);
	}

}
void ScriptEditor::set_scene_root_script( Ref<Script> p_script ) {

	bool open_dominant = EditorSettings::get_singleton()->get("text_editor/open_dominant_script_on_scene_change");
	if (bool(EditorSettings::get_singleton()->get("external_editor/use_external_editor")))
		return;

	if (open_dominant && p_script.is_valid()) {
		edit(p_script);
	}
}

void ScriptEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_tab_changed",&ScriptEditor::_tab_changed);
	ObjectTypeDB::bind_method("_menu_option",&ScriptEditor::_menu_option);
	ObjectTypeDB::bind_method("_close_current_tab",&ScriptEditor::_close_current_tab);
	ObjectTypeDB::bind_method("_editor_play",&ScriptEditor::_editor_play);
	ObjectTypeDB::bind_method("_editor_pause",&ScriptEditor::_editor_pause);
	ObjectTypeDB::bind_method("_editor_stop",&ScriptEditor::_editor_stop);
	ObjectTypeDB::bind_method("_add_callback",&ScriptEditor::_add_callback);
	ObjectTypeDB::bind_method("_reload_scripts",&ScriptEditor::_reload_scripts);
	ObjectTypeDB::bind_method("_resave_scripts",&ScriptEditor::_resave_scripts);
	ObjectTypeDB::bind_method("_res_saved_callback",&ScriptEditor::_res_saved_callback);
	ObjectTypeDB::bind_method("_goto_script_line",&ScriptEditor::_goto_script_line);
	ObjectTypeDB::bind_method("_goto_script_line2",&ScriptEditor::_goto_script_line2);
	ObjectTypeDB::bind_method("_breaked",&ScriptEditor::_breaked);
	ObjectTypeDB::bind_method("_show_debugger",&ScriptEditor::_show_debugger);
	ObjectTypeDB::bind_method("_get_debug_tooltip",&ScriptEditor::_get_debug_tooltip);
	ObjectTypeDB::bind_method("_autosave_scripts",&ScriptEditor::_autosave_scripts);
	ObjectTypeDB::bind_method("_editor_settings_changed",&ScriptEditor::_editor_settings_changed);
	ObjectTypeDB::bind_method("_update_script_names",&ScriptEditor::_update_script_names);
	ObjectTypeDB::bind_method("_tree_changed",&ScriptEditor::_tree_changed);
	ObjectTypeDB::bind_method("_script_selected",&ScriptEditor::_script_selected);
	ObjectTypeDB::bind_method("_script_created",&ScriptEditor::_script_created);
	ObjectTypeDB::bind_method("_script_split_dragged",&ScriptEditor::_script_split_dragged);
	ObjectTypeDB::bind_method("_help_class_open",&ScriptEditor::_help_class_open);
	ObjectTypeDB::bind_method("_help_class_goto",&ScriptEditor::_help_class_goto);
	ObjectTypeDB::bind_method("_request_help",&ScriptEditor::_help_class_open);
	ObjectTypeDB::bind_method("_history_forward",&ScriptEditor::_history_forward);
	ObjectTypeDB::bind_method("_history_back",&ScriptEditor::_history_back);
}

ScriptEditor::ScriptEditor(EditorNode *p_editor) {

	completion_cache = memnew( EditorScriptCodeCompletionCache );
	restoring_layout=false;
	waiting_update_names=false;
	editor=p_editor;

	menu_hb = memnew( HBoxContainer );
	add_child(menu_hb);

	v_split = memnew( VSplitContainer );
	add_child(v_split);
	v_split->set_v_size_flags(SIZE_EXPAND_FILL);

	script_split = memnew( HSplitContainer );
	v_split->add_child(script_split);
	script_split->set_v_size_flags(SIZE_EXPAND_FILL);

	script_list = memnew( ItemList );
	script_split->add_child(script_list);
	script_list->set_custom_minimum_size(Size2(70,0));
	script_split->set_split_offset(70);

	tab_container = memnew( TabContainer );
	tab_container->set_tabs_visible(false);
	script_split->add_child(tab_container);


	tab_container->set_h_size_flags(SIZE_EXPAND_FILL);

	file_menu = memnew( MenuButton );
	menu_hb->add_child(file_menu);
	file_menu->set_text("File");
	file_menu->get_popup()->add_item("New",FILE_NEW);
	file_menu->get_popup()->add_item("Open",FILE_OPEN);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item("Save",FILE_SAVE,KEY_MASK_ALT|KEY_MASK_CMD|KEY_S);
	file_menu->get_popup()->add_item("Save As..",FILE_SAVE_AS);
	file_menu->get_popup()->add_item("Save All",FILE_SAVE_ALL,KEY_MASK_CMD|KEY_MASK_SHIFT|KEY_S);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item("History Prev",WINDOW_PREV,KEY_MASK_CTRL|KEY_MASK_ALT|KEY_LEFT);
	file_menu->get_popup()->add_item("History Next",WINDOW_NEXT,KEY_MASK_CTRL|KEY_MASK_ALT|KEY_RIGHT);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item("Close",FILE_CLOSE,KEY_MASK_CMD|KEY_W);
	file_menu->get_popup()->connect("item_pressed", this,"_menu_option");

	edit_menu = memnew( MenuButton );
	menu_hb->add_child(edit_menu);
	edit_menu->set_text("Edit");
	edit_menu->get_popup()->add_item("Undo",EDIT_UNDO,KEY_MASK_CMD|KEY_Z);
	edit_menu->get_popup()->add_item("Redo",EDIT_REDO,KEY_MASK_CMD|KEY_Y);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_item("Cut",EDIT_CUT,KEY_MASK_CMD|KEY_X);
	edit_menu->get_popup()->add_item("Copy",EDIT_COPY,KEY_MASK_CMD|KEY_C);
	edit_menu->get_popup()->add_item("Paste",EDIT_PASTE,KEY_MASK_CMD|KEY_V);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_item("Select All",EDIT_SELECT_ALL,KEY_MASK_CMD|KEY_A);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_item("Move Up",EDIT_MOVE_LINE_UP,KEY_MASK_ALT|KEY_UP);
	edit_menu->get_popup()->add_item("Move Down",EDIT_MOVE_LINE_DOWN,KEY_MASK_ALT|KEY_DOWN);
	edit_menu->get_popup()->add_item("Indent Left",EDIT_INDENT_LEFT,KEY_MASK_ALT|KEY_LEFT);
	edit_menu->get_popup()->add_item("Indent Right",EDIT_INDENT_RIGHT,KEY_MASK_ALT|KEY_RIGHT);
	edit_menu->get_popup()->add_item("Toggle Comment",EDIT_TOGGLE_COMMENT,KEY_MASK_CMD|KEY_K);
	edit_menu->get_popup()->add_item("Clone Down",EDIT_CLONE_DOWN,KEY_MASK_CMD|KEY_B);
	edit_menu->get_popup()->add_separator();
#ifdef OSX_ENABLED
	edit_menu->get_popup()->add_item("Complete Symbol",EDIT_COMPLETE,KEY_MASK_CTRL|KEY_SPACE);
#else
	edit_menu->get_popup()->add_item("Complete Symbol",EDIT_COMPLETE,KEY_MASK_CMD|KEY_SPACE);
#endif
	edit_menu->get_popup()->add_item("Auto Indent",EDIT_AUTO_INDENT,KEY_MASK_CMD|KEY_I);
	edit_menu->get_popup()->connect("item_pressed", this,"_menu_option");


	search_menu = memnew( MenuButton );
	menu_hb->add_child(search_menu);
	search_menu->set_text("Search");
	search_menu->get_popup()->add_item("Find..",SEARCH_FIND,KEY_MASK_CMD|KEY_F);
	search_menu->get_popup()->add_item("Find Next",SEARCH_FIND_NEXT,KEY_F3);
	search_menu->get_popup()->add_item("Replace..",SEARCH_REPLACE,KEY_MASK_CMD|KEY_R);
	search_menu->get_popup()->add_separator();
	search_menu->get_popup()->add_item("Goto Function..",SEARCH_LOCATE_FUNCTION,KEY_MASK_SHIFT|KEY_MASK_CMD|KEY_F);
	search_menu->get_popup()->add_item("Goto Line..",SEARCH_GOTO_LINE,KEY_MASK_CMD|KEY_L);
	search_menu->get_popup()->connect("item_pressed", this,"_menu_option");

	script_search_menu = memnew( MenuButton );
	menu_hb->add_child(script_search_menu);
	script_search_menu->set_text("Search");
	script_search_menu->get_popup()->add_item("Find..",SEARCH_FIND,KEY_MASK_CMD|KEY_F);
	script_search_menu->get_popup()->add_item("Find Next",SEARCH_FIND_NEXT,KEY_F3);
	script_search_menu->get_popup()->connect("item_pressed", this,"_menu_option");
	script_search_menu->hide();


	debug_menu = memnew( MenuButton );
	menu_hb->add_child(debug_menu);
	debug_menu->set_text("Debug");
	debug_menu->get_popup()->add_item("Toggle Breakpoint",DEBUG_TOGGLE_BREAKPOINT,KEY_F9);
	debug_menu->get_popup()->add_separator();
	debug_menu->get_popup()->add_item("Step Over",DEBUG_NEXT,KEY_F10);
	debug_menu->get_popup()->add_item("Step Into",DEBUG_STEP,KEY_F11);
	debug_menu->get_popup()->add_separator();
	debug_menu->get_popup()->add_item("Break",DEBUG_BREAK);
	debug_menu->get_popup()->add_item("Continue",DEBUG_CONTINUE);
	debug_menu->get_popup()->add_separator();
	debug_menu->get_popup()->add_check_item("Show Debugger",DEBUG_SHOW);
	debug_menu->get_popup()->add_check_item("Keep Debuger Open",DEBUG_SHOW_KEEP_OPEN);
	debug_menu->get_popup()->connect("item_pressed", this,"_menu_option");

	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_NEXT), true);
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_STEP), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_BREAK), true );
	debug_menu->get_popup()->set_item_disabled( debug_menu->get_popup()->get_item_index(DEBUG_CONTINUE), true );


#if 0
	window_menu = memnew( MenuButton );
	menu_hb->add_child(window_menu);
	window_menu->set_text("Window");
	window_menu->get_popup()->add_item("Close",WINDOW_CLOSE,KEY_MASK_CMD|KEY_W);
	window_menu->get_popup()->add_separator();
	window_menu->get_popup()->add_item("Move Left",WINDOW_MOVE_LEFT,KEY_MASK_CMD|KEY_LEFT);
	window_menu->get_popup()->add_item("Move Right",WINDOW_MOVE_RIGHT,KEY_MASK_CMD|KEY_RIGHT);
	window_menu->get_popup()->add_separator();
	window_menu->get_popup()->connect("item_pressed", this,"_menu_option");

#endif

	help_menu = memnew( MenuButton );
	menu_hb->add_child(help_menu);
	help_menu->set_text("Help");
	help_menu->get_popup()->add_item("Contextual", HELP_CONTEXTUAL, KEY_MASK_SHIFT|KEY_F1);
	help_menu->get_popup()->connect("item_pressed", this,"_menu_option");

	menu_hb->add_spacer();


	script_icon = memnew( TextureFrame );
	menu_hb->add_child(script_icon);
	script_name_label = memnew( Label );
	menu_hb->add_child(script_name_label);

	script_icon->hide();
	script_name_label->hide();

	menu_hb->add_spacer();

	site_search = memnew( ToolButton );
	site_search->set_text("Tutorials");
	site_search->connect("pressed",this,"_menu_option",varray(SEARCH_WEBSITE));
	menu_hb->add_child(site_search);
	site_search->set_tooltip("Open http://www.godotengine.org at tutorials section.");

	class_search = memnew( ToolButton );
	class_search->set_text("Classes");
	class_search->connect("pressed",this,"_menu_option",varray(SEARCH_CLASSES));
	menu_hb->add_child(class_search);
	class_search->set_tooltip("Search the class hierarchy.");

	help_search = memnew( ToolButton );
	help_search->set_text("Search Help");
	help_search->connect("pressed",this,"_menu_option",varray(SEARCH_HELP));
	menu_hb->add_child(help_search);
	help_search->set_tooltip("Search the reference documentation.");

	menu_hb->add_child( memnew( VSeparator) );

	script_back = memnew( ToolButton );
	script_back->connect("pressed",this,"_history_back");
	menu_hb->add_child(script_back);
	script_back->set_disabled(true);
	help_search->set_tooltip("Go to previous edited document.");

	script_forward = memnew( ToolButton );
	script_forward->connect("pressed",this,"_history_forward");
	menu_hb->add_child(script_forward);
	script_forward->set_disabled(true);
	help_search->set_tooltip("Go to next edited document.");



	tab_container->connect("tab_changed", this,"_tab_changed");

	find_replace_dialog = memnew(FindReplaceDialog);
	add_child(find_replace_dialog);

	erase_tab_confirm = memnew( ConfirmationDialog );
	add_child(erase_tab_confirm);
	erase_tab_confirm->connect("confirmed", this,"_close_current_tab");

	script_create_dialog = memnew(ScriptCreateDialog);
	script_create_dialog->set_title("Create Script");
	add_child(script_create_dialog);
	script_create_dialog->connect("script_created", this, "_script_created");

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	debugger = memnew( ScriptEditorDebugger(editor) );
	debugger->connect("goto_script_line",this,"_goto_script_line");
	debugger->connect("show_debugger",this,"_show_debugger");

	disk_changed = memnew( ConfirmationDialog );
	{
		VBoxContainer *vbc = memnew( VBoxContainer );
		disk_changed->add_child(vbc);
		disk_changed->set_child_rect(vbc);

		Label *dl = memnew( Label );
		dl->set_text("The following files are newer on disk.\nWhat action should be taken?:");
		vbc->add_child(dl);

		disk_changed_list = memnew( Tree );
		vbc->add_child(disk_changed_list);
		disk_changed_list->set_v_size_flags(SIZE_EXPAND_FILL);

		disk_changed->connect("confirmed",this,"_reload_scripts");
		disk_changed->get_ok()->set_text("Reload");

		disk_changed->add_button("Resave",!OS::get_singleton()->get_swap_ok_cancel(),"resave");
		disk_changed->connect("custom_action",this,"_resave_scripts");


	}

	add_child(disk_changed);

	script_editor=this;

	quick_open = memnew( ScriptEditorQuickOpen );
	add_child(quick_open);

	quick_open->connect("goto_line",this,"_goto_script_line2");

	v_split->add_child(debugger);
	debugger->connect("breaked",this,"_breaked");

	autosave_timer = memnew( Timer );
	autosave_timer->set_one_shot(false);
	add_child(autosave_timer);

	grab_focus_block=false;

	help_search_dialog = memnew( EditorHelpSearch );
	add_child(help_search_dialog);
	help_search_dialog->connect("go_to_help",this,"_help_class_goto");


	help_index = memnew( EditorHelpIndex );
	add_child(help_index);
	help_index->connect("open_class",this,"_help_class_open");

	history_pos=-1;
//	debugger_gui->hide();

	edit_pass=0;

}


ScriptEditor::~ScriptEditor() {

	memdelete(completion_cache);
}

void ScriptEditorPlugin::edit(Object *p_object) {

	if (!p_object->cast_to<Script>())
		return;

	script_editor->edit(p_object->cast_to<Script>());

}

bool ScriptEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("Script");
}

void ScriptEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		script_editor->show();
		script_editor->set_process(true);
		script_editor->ensure_select_current();
	} else {

		script_editor->hide();
		script_editor->set_process(false);
	}

}

void ScriptEditorPlugin::selected_notify() {

	script_editor->ensure_select_current();
}

Dictionary ScriptEditorPlugin::get_state() const {

	return script_editor->get_state();
}

void ScriptEditorPlugin::set_state(const Dictionary& p_state) {

	script_editor->set_state(p_state);
}
void ScriptEditorPlugin::clear() {

	script_editor->clear();
}

void ScriptEditorPlugin::save_external_data() {

	script_editor->save_external_data();
}

void ScriptEditorPlugin::apply_changes() {

	script_editor->apply_scripts();
}

void ScriptEditorPlugin::restore_global_state() {


}

void ScriptEditorPlugin::save_global_state() {

}

void ScriptEditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {

	script_editor->set_window_layout(p_layout);
}

void ScriptEditorPlugin::get_window_layout(Ref<ConfigFile> p_layout){

	script_editor->get_window_layout(p_layout);
}


void ScriptEditorPlugin::get_breakpoints(List<String> *p_breakpoints) {


	return script_editor->get_breakpoints(p_breakpoints);
}

void ScriptEditorPlugin::edited_scene_changed() {

	script_editor->edited_scene_changed();
}

ScriptEditorPlugin::ScriptEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	script_editor = memnew( ScriptEditor(p_node) );
	editor->get_viewport()->add_child(script_editor);
	script_editor->set_area_as_parent_rect();

	script_editor->hide();

	EDITOR_DEF("text_editor/auto_reload_scripts_on_external_change",true);
	EDITOR_DEF("text_editor/open_dominant_script_on_scene_change",true);
	EDITOR_DEF("external_editor/use_external_editor",false);
	EDITOR_DEF("external_editor/exec_path","");
	EDITOR_DEF("text_editor/script_temperature_enabled",true);
	EDITOR_DEF("text_editor/script_temperature_history_size",15);
	EDITOR_DEF("text_editor/script_temperature_hot_color",Color(1,0,0,0.3));
	EDITOR_DEF("text_editor/script_temperature_cold_color",Color(0,0,1,0.3));
	EDITOR_DEF("text_editor/group_help_pages",false);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"external_editor/exec_path",PROPERTY_HINT_GLOBAL_FILE));
	EDITOR_DEF("external_editor/exec_flags","");

}


ScriptEditorPlugin::~ScriptEditorPlugin()
{
}

