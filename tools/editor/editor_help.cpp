/*************************************************************************/
/*  editor_help.cpp                                                      */
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
#include "editor_help.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "os/keyboard.h"
#include "doc_data_compressed.h"



#include "os/keyboard.h"


void EditorHelpSearch::popup(const String& p_term) {

	popup_centered_ratio(0.6);
	if (p_term!="") {
		search_box->set_text(p_term);
		search_box->select_all();
		_update_search();
	} else
		search_box->clear();
	search_box->grab_focus();
}


void EditorHelpSearch::_text_changed(const String& p_newtext) {

	_update_search();
}

void EditorHelpSearch::_sbox_input(const InputEvent& p_ie) {

	if (p_ie.type==InputEvent::KEY && (
		p_ie.key.scancode == KEY_UP ||
		p_ie.key.scancode == KEY_DOWN ||
		p_ie.key.scancode == KEY_PAGEUP ||
		p_ie.key.scancode == KEY_PAGEDOWN ) ) {

		search_options->call("_input_event",p_ie);
		search_box->accept_event();
	}

}

void EditorHelpSearch::_update_search() {

	search_options->clear();
	search_options->set_hide_root(true);

	/*
	TreeItem *root = search_options->create_item();
	_parse_fs(EditorFileSystem::get_singleton()->get_filesystem());
*/

	List<StringName> type_list;
	ObjectTypeDB::get_type_list(&type_list);

	DocData *doc=EditorHelp::get_doc_data();
	String term = search_box->get_text();
	if (term.length()<3)
		return;

	TreeItem *root = search_options->create_item();



	Ref<Texture> def_icon = get_icon("Node","EditorIcons");
	//classes first
	for (Map<String,DocData::ClassDoc>::Element *E=doc->class_list.front();E;E=E->next()) {

		if (E->key().findn(term)!=-1) {

			TreeItem *item = search_options->create_item(root);
			item->set_metadata(0,"class_name:"+E->key());
			item->set_text(0,E->key()+" (Class)");
			if (has_icon(E->key(),"EditorIcons"))
				item->set_icon(0,get_icon(E->key(),"EditorIcons"));
			else
				item->set_icon(0,def_icon);


		}

	}

	//class methods, etc second
	for (Map<String,DocData::ClassDoc>::Element *E=doc->class_list.front();E;E=E->next()) {


		DocData::ClassDoc & c = E->get();

		Ref<Texture> cicon;
		if (has_icon(E->key(),"EditorIcons"))
			cicon=get_icon(E->key(),"EditorIcons");
		else
			cicon=def_icon;

		for(int i=0;i<c.methods.size();i++) {
			if( (term.begins_with(".") && c.methods[i].name.begins_with(term.right(1)))
				|| (term.ends_with("(") && c.methods[i].name.ends_with(term.left(term.length()-1).strip_edges()))
				|| (term.begins_with(".") && term.ends_with("(") && c.methods[i].name==term.substr(1,term.length()-2).strip_edges())
				|| c.methods[i].name.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_method:"+E->key()+":"+c.methods[i].name);
				item->set_text(0,E->key()+"."+c.methods[i].name+" (Method)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.signals.size();i++) {

			if (c.signals[i].name.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_signal:"+E->key()+":"+c.signals[i].name);
				item->set_text(0,E->key()+"."+c.signals[i].name+" (Signal)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.constants.size();i++) {

			if (c.constants[i].name.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_constant:"+E->key()+":"+c.constants[i].name);
				item->set_text(0,E->key()+"."+c.constants[i].name+" (Constant)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.properties.size();i++) {

			if (c.properties[i].name.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_property:"+E->key()+":"+c.properties[i].name);
				item->set_text(0,E->key()+"."+c.properties[i].name+" (Property)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.theme_properties.size();i++) {

			if (c.theme_properties[i].name.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_theme_item:"+E->key()+":"+c.theme_properties[i].name);
				item->set_text(0,E->key()+"."+c.theme_properties[i].name+" (Theme Item)");
				item->set_icon(0,cicon);
			}
		}


	}

	//same but descriptions

	for (Map<String,DocData::ClassDoc>::Element *E=doc->class_list.front();E;E=E->next()) {


		DocData::ClassDoc & c = E->get();

		Ref<Texture> cicon;
		if (has_icon(E->key(),"EditorIcons"))
			cicon=get_icon(E->key(),"EditorIcons");
		else
			cicon=def_icon;

		if (c.description.findn(term)!=-1) {


			TreeItem *item = search_options->create_item(root);
			item->set_metadata(0,"class_desc:"+E->key());
			item->set_text(0,E->key()+" (Class Description)");
			item->set_icon(0,cicon);

		}

		for(int i=0;i<c.methods.size();i++) {

			if (c.methods[i].description.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_method_desc:"+E->key()+":"+c.methods[i].name);
				item->set_text(0,E->key()+"."+c.methods[i].name+" (Method Description)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.signals.size();i++) {

			if (c.signals[i].description.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_signal:"+E->key()+":"+c.signals[i].name);
				item->set_text(0,E->key()+"."+c.signals[i].name+" (Signal Description)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.constants.size();i++) {

			if (c.constants[i].description.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_constant:"+E->key()+":"+c.constants[i].name);
				item->set_text(0,E->key()+"."+c.constants[i].name+" (Constant Description)");
				item->set_icon(0,cicon);
			}
		}

		for(int i=0;i<c.properties.size();i++) {

			if (c.properties[i].description.findn(term)!=-1) {

				TreeItem *item = search_options->create_item(root);
				item->set_metadata(0,"class_property_desc:"+E->key()+":"+c.properties[i].name);
				item->set_text(0,E->key()+"."+c.properties[i].name+" (Property Description)");
				item->set_icon(0,cicon);
			}
		}

	}

	get_ok()->set_disabled(root->get_children()==NULL);

}

void EditorHelpSearch::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;

	String mdata=ti->get_metadata(0);
	emit_signal("go_to_help",mdata);
	editor->call("_editor_select",3); // in case EditorHelpSearch beeen invoked on top of other editor window
	// go to that
	hide();
}

void EditorHelpSearch::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		connect("confirmed",this,"_confirmed");
		_update_search();
	}

	if (p_what==NOTIFICATION_VISIBILITY_CHANGED) {

		if (is_visible()) {

			search_box->call_deferred("grab_focus"); // still not visible
			search_box->select_all();
		}
	}

}


void EditorHelpSearch::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_text_changed"),&EditorHelpSearch::_text_changed);
	ObjectTypeDB::bind_method(_MD("_confirmed"),&EditorHelpSearch::_confirmed);
	ObjectTypeDB::bind_method(_MD("_sbox_input"),&EditorHelpSearch::_sbox_input);
	ObjectTypeDB::bind_method(_MD("_update_search"),&EditorHelpSearch::_update_search);

	ADD_SIGNAL(MethodInfo("go_to_help"));

}


EditorHelpSearch::EditorHelpSearch(EditorNode *p_editor) {

	editor=p_editor;
	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);
	HBoxContainer *sb_hb = memnew( HBoxContainer);
	search_box = memnew( LineEdit );
	sb_hb->add_child(search_box);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	Button *sb = memnew( Button("Search"));
	sb->connect("pressed",this,"_update_search");
	sb_hb->add_child(sb);
	vbc->add_margin_child("Search:",sb_hb);
	search_box->connect("text_changed",this,"_text_changed");
	search_box->connect("input_event",this,"_sbox_input");
	search_options = memnew( Tree );
	vbc->add_margin_child("Matches:",search_options,true);
	get_ok()->set_text("View");
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated",this,"_confirmed");
	set_title("Search Classes");
//	search_options->set_hide_root(true);

}


DocData *EditorHelp::doc=NULL;

void EditorHelp::_unhandled_key_input(const InputEvent& p_ev) {

	if (!is_visible())
		return;
	if ( p_ev.key.mod.control && p_ev.key.scancode==KEY_F) {

		search->grab_focus();
		search->select_all();
	} else if (p_ev.key.mod.shift && p_ev.key.scancode==KEY_F1) {
		class_search->popup();
	}
}

void EditorHelp::_search(const String&) {

	if (search->get_text()=="")
		return;


	String stext=search->get_text();
	bool keep = prev_search==stext && class_list->get_selected() && prev_search_page==class_list->get_selected()->get_text(0);

	class_desc->search(stext, keep);

	prev_search=stext;
	if (class_list->get_selected())
		prev_search_page=class_list->get_selected()->get_text(0);


}

void EditorHelp::_button_pressed(int p_idx) {

	if (p_idx==PAGE_CLASS_LIST) {

	//	edited_class->set_pressed(false);
	//	class_list_button->set_pressed(true);
	//	tabs->set_current_tab(PAGE_CLASS_LIST);

	} else if (p_idx==PAGE_CLASS_DESC) {

	//	edited_class->set_pressed(true);
	//	class_list_button->set_pressed(false);
	//	tabs->set_current_tab(PAGE_CLASS_DESC);

	} else if (p_idx==PAGE_CLASS_PREV) {

		if (history_pos<2)
			return;
		history_pos--;
		ERR_FAIL_INDEX(history_pos-1,history.size());
		_goto_desc(history[history_pos-1].c,false,history[history_pos-1].scroll);
		_update_history_buttons();


	} else if (p_idx==PAGE_CLASS_NEXT) {

		if (history_pos>=history.size())
			return;

		history_pos++;
		ERR_FAIL_INDEX(history_pos-1,history.size());
		_goto_desc(history[history_pos-1].c,false,history[history_pos-1].scroll);
		_update_history_buttons();

	} else if (p_idx==PAGE_SEARCH) {

		_search("");
	} else if (p_idx==CLASS_SEARCH) {

		class_search->popup();
	}


}




void EditorHelp::_class_list_select(const String& p_select) {

	_goto_desc(p_select);
}

void EditorHelp::_class_desc_select(const String& p_select) {

	if (p_select.begins_with("#")) {
		_goto_desc(p_select.substr(1,p_select.length()));
		return;
	} else if (p_select.begins_with("@")) {

		String m = p_select.substr(1,p_select.length());
		if (!method_line.has(m))
			return;
		class_desc->scroll_to_line(method_line[m]);
		return;
	}


}

void EditorHelp::_add_type(const String& p_type) {

	String t = p_type;
	if (t=="")
		t="void";
	bool can_ref = (t!="int" && t!="real" && t!="bool" && t!="void");

	class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/base_type_color"));
	if (can_ref)
		class_desc->push_meta("#"+t); //class
	class_desc->add_text(t);
	if (can_ref)
		class_desc->pop();
	class_desc->pop();

}

void EditorHelp::_update_history_buttons() {

	back->set_disabled(history_pos<2);
	forward->set_disabled(history_pos>=history.size());

}


void EditorHelp::_scroll_changed(double p_scroll) {

	if (scroll_locked)
		return;

	int p = history_pos -1;
	if (p<0 || p>=history.size())
		return;

	if (class_desc->get_v_scroll()->is_hidden())
		p_scroll=0;

	history[p].scroll=p_scroll;
}

Error EditorHelp::_goto_desc(const String& p_class,bool p_update_history,int p_vscr) {

	//ERR_FAIL_COND(!doc->class_list.has(p_class));
	if (!doc->class_list.has(p_class))
		return ERR_DOES_NOT_EXIST;


	if (tree_item_map.has(p_class)) {
		select_locked = true;
		tree_item_map[p_class]->select(0);
		class_list->ensure_cursor_is_visible();
	}

	class_desc->show();
	//tabs->set_current_tab(PAGE_CLASS_DESC);
	edited_class->set_pressed(true);
	class_list_button->set_pressed(false);
	description_line=0;

	if (p_class==edited_class->get_text())
		return OK; //already there

	scroll_locked=true;

	if (p_update_history) {

		history.resize(history_pos);
		history_pos++;
		History h;
		h.c=p_class;
		h.scroll=0;
		history.push_back(h);
		_update_history_buttons();
		class_desc->get_v_scroll()->set_val(0);
	}

	class_desc->clear();
	method_line.clear();
	edited_class->set_text(p_class);
	//edited_class->show();


	DocData::ClassDoc &cd=doc->class_list[p_class];

	Color h_color;

	Ref<Font> doc_font = get_font("normal","Fonts");
	Ref<Font> doc_code_font = get_font("source","Fonts");
	Ref<Font> doc_title_font = get_font("large","Fonts");


	h_color=Color(1,1,1,1);

	class_desc->push_font(doc_title_font);
	class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
	class_desc->add_text("Class: ");
	class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/base_type_color"));
	class_desc->add_text(p_class);
	class_desc->pop();
	class_desc->pop();
	class_desc->pop();
	class_desc->add_newline();

	if (cd.inherits!="") {

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Inherits: ");
		class_desc->pop();
		class_desc->pop();
		class_desc->push_font(doc_font);
		_add_type(cd.inherits);
		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();

	}

	if (cd.brief_description!="") {

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Brief Description:");
		class_desc->pop();
		class_desc->pop();

		//class_desc->add_newline();
		class_desc->add_newline();
		_add_text(cd.brief_description);
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	bool method_descr=false;

	if (cd.methods.size()) {

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Public Methods:");
		class_desc->pop();
		class_desc->pop();

		//class_desc->add_newline();
		class_desc->add_newline();

		class_desc->push_indent(1);

		for(int i=0;i<cd.methods.size();i++) {

			method_line[cd.methods[i].name]=class_desc->get_line_count()-2;	//gets overriden if description
			class_desc->push_font(doc_code_font);
			_add_type(cd.methods[i].return_type);
			class_desc->add_text(" ");
			if (cd.methods[i].description!="") {
				method_descr=true;
				class_desc->push_meta("@"+cd.methods[i].name);
			}
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
			class_desc->add_text(cd.methods[i].name);
			class_desc->pop();
			if (cd.methods[i].description!="")
				class_desc->pop();
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(cd.methods[i].arguments.size()?"( ":"(");
			class_desc->pop();
			for(int j=0;j<cd.methods[i].arguments.size();j++) {
				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
				if (j>0)
					class_desc->add_text(", ");
				_add_type(cd.methods[i].arguments[j].type);
				class_desc->add_text(" "+cd.methods[i].arguments[j].name);
				if (cd.methods[i].arguments[j].default_value!="") {

					class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
					class_desc->add_text("=");
					class_desc->pop();
					class_desc->add_text(cd.methods[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(cd.methods[i].arguments.size()?" )":")");
			class_desc->pop();
			if (cd.methods[i].qualifiers!="") {

				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
				class_desc->add_text(" "+cd.methods[i].qualifiers);
				class_desc->pop();

			}
			class_desc->pop();//monofont
			class_desc->add_newline();

		}

		class_desc->pop();
		class_desc->add_newline();

	}

	if (cd.properties.size()) {

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Members:");
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();

		class_desc->push_indent(1);

		//class_desc->add_newline();

		for(int i=0;i<cd.properties.size();i++) {

			property_line[cd.properties[i].name]=class_desc->get_line_count()-2;	//gets overriden if description
			class_desc->push_font(doc_code_font);
			_add_type(cd.properties[i].type);
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
			class_desc->add_text(" "+cd.properties[i].name);
			class_desc->pop();
			class_desc->pop();

			if (cd.properties[i].description!="") {
				class_desc->push_font(doc_font);
				class_desc->add_text("  ");
				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/comment_color"));
				class_desc->add_text(cd.properties[i].description);
				class_desc->pop();
				class_desc->pop();

			}

			class_desc->add_newline();
		}

		class_desc->add_newline();
		class_desc->pop();


	}

	if (cd.theme_properties.size()) {


		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("GUI Theme Items:");
		class_desc->pop();
		class_desc->pop();
		class_desc->add_newline();

		class_desc->push_indent(1);

		//class_desc->add_newline();

		for(int i=0;i<cd.theme_properties.size();i++) {

			theme_property_line[cd.theme_properties[i].name]=class_desc->get_line_count()-2;	//gets overriden if description
			class_desc->push_font(doc_code_font);
			_add_type(cd.theme_properties[i].type);
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
			class_desc->add_text(" "+cd.theme_properties[i].name);
			class_desc->pop();
			class_desc->pop();

			if (cd.theme_properties[i].description!="") {
				class_desc->push_font(doc_font);
				class_desc->add_text("  ");
				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/comment_color"));
				class_desc->add_text(cd.theme_properties[i].description);
				class_desc->pop();
				class_desc->pop();

			}

			class_desc->add_newline();
		}

		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();

	}
	if (cd.signals.size()) {

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Signals:");
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		//class_desc->add_newline();

		class_desc->push_indent(1);

		for(int i=0;i<cd.signals.size();i++) {

			signal_line[cd.signals[i].name]=class_desc->get_line_count()-2;	//gets overriden if description
			class_desc->push_font(doc_code_font);
			//_add_type("void");
			//class_desc->add_text(" ");
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
			class_desc->add_text(cd.signals[i].name);
			class_desc->pop();
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(cd.signals[i].arguments.size()?"( ":"(");
			class_desc->pop();
			for(int j=0;j<cd.signals[i].arguments.size();j++) {
				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
				if (j>0)
					class_desc->add_text(", ");
				_add_type(cd.signals[i].arguments[j].type);
				class_desc->add_text(" "+cd.signals[i].arguments[j].name);
				if (cd.signals[i].arguments[j].default_value!="") {

					class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
					class_desc->add_text("=");
					class_desc->pop();
					class_desc->add_text(cd.signals[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(cd.signals[i].arguments.size()?" )":")");
			class_desc->pop();
			if (cd.signals[i].description!="") {

				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/comment_color"));
				class_desc->add_text(" "+cd.signals[i].description);
				class_desc->pop();

			}
			class_desc->pop();//monofont
			class_desc->add_newline();

		}

		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();

	}

	if (cd.constants.size()) {


		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Constants:");
		class_desc->pop();
		class_desc->pop();
		class_desc->push_indent(1);

		class_desc->add_newline();
		//class_desc->add_newline();

		for(int i=0;i<cd.constants.size();i++) {

			constant_line[cd.constants[i].name]=class_desc->get_line_count()-2;
			class_desc->push_font(doc_code_font);
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/base_type_color"));
			class_desc->add_text(cd.constants[i].name);
			class_desc->pop();
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(" = ");
			class_desc->pop();
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
			class_desc->add_text(cd.constants[i].value);
			class_desc->pop();
			class_desc->pop();
			if (cd.constants[i].description!="") {
				class_desc->push_font(doc_font);
				class_desc->add_text("  ");
				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/comment_color"));
				class_desc->add_text(cd.constants[i].description);
				class_desc->pop();
				class_desc->pop();
			}

			class_desc->add_newline();
		}

		class_desc->pop();
		class_desc->add_newline();
		class_desc->add_newline();


	}

	if (cd.description!="") {

		description_line=class_desc->get_line_count()-2;

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Description:");
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		_add_text(cd.description);
		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->add_newline();
	}

	if (method_descr) {

		class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
		class_desc->push_font(doc_title_font);
		class_desc->add_text("Method Description:");
		class_desc->pop();
		class_desc->pop();

		class_desc->add_newline();
		class_desc->add_newline();
		class_desc->push_indent(1);


		for(int i=0;i<cd.methods.size();i++) {

			method_line[cd.methods[i].name]=class_desc->get_line_count()-2;

			if( cd.methods[i].description != "") {
				class_desc->add_newline();
			}
			class_desc->push_font(doc_code_font);
			_add_type(cd.methods[i].return_type);

			class_desc->add_text(" ");
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
			class_desc->add_text(cd.methods[i].name);
			class_desc->pop();
			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(cd.methods[i].arguments.size()?"( ":"(");
			class_desc->pop();
			for(int j=0;j<cd.methods[i].arguments.size();j++) {
				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
				if (j>0)
					class_desc->add_text(", ");
				_add_type(cd.methods[i].arguments[j].type);
				class_desc->add_text(" "+cd.methods[i].arguments[j].name);
				if (cd.methods[i].arguments[j].default_value!="") {

					class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
					class_desc->add_text("=");
					class_desc->pop();
					class_desc->add_text(cd.methods[i].arguments[j].default_value);
				}

				class_desc->pop();
			}

			class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/symbol_color"));
			class_desc->add_text(cd.methods[i].arguments.size()?" )":")");
			class_desc->pop();
			if (cd.methods[i].qualifiers!="") {

				class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/keyword_color"));
				class_desc->add_text(" "+cd.methods[i].qualifiers);
				class_desc->pop();

			}

			class_desc->pop();

			if( cd.methods[i].description != "") {
				class_desc->add_text("  ");
				_add_text(cd.methods[i].description);
				class_desc->add_newline();
				class_desc->add_newline();
			}
			class_desc->add_newline();
			class_desc->add_newline();

		}





	}

	if (!p_update_history) {

		class_desc->get_v_scroll()->set_val(history[history_pos-1].scroll);
	}

	scroll_locked=false;

	return OK;
}

void EditorHelp::_request_help(const String& p_string) {
	Error err = _goto_desc(p_string);
	if (err==OK) {
		editor->call("_editor_select",3);
	} else {
		class_search->popup(p_string);
	}
	//100 palabras
}


void EditorHelp::_help_callback(const String& p_topic) {

	String what = p_topic.get_slice(":",0);
	String clss = p_topic.get_slice(":",1);
	String name;
	if (p_topic.get_slice_count(":")==3)
		name=p_topic.get_slice(":",2);

	_request_help(clss); //first go to class

	int line=0;

	if (what=="class_desc") {
		line=description_line;
	} else if (what=="class_signal") {
		if (signal_line.has(name))
			line=signal_line[name];
	} else if (what=="class_method" || what=="class_method_desc") {
		if (method_line.has(name))
			line=method_line[name];
	} else if (what=="class_property") {

		if (property_line.has(name))
			line=property_line[name];
	} else if (what=="class_theme_item") {

		if (theme_property_line.has(name))
			line=theme_property_line[name];
	} else if (what=="class_constant") {

		if (constant_line.has(name))
			line=constant_line[name];
	}

	class_desc->scroll_to_line(line);

}

void EditorHelp::_add_text(const String& p_bbcode) {


	class_desc->push_color(EditorSettings::get_singleton()->get("text_editor/text_color"));
	class_desc->push_font( get_font("normal","Fonts") );
	class_desc->push_indent(1);
	int pos = 0;

	List<String> tag_stack;

	while(pos < p_bbcode.length()) {


		int brk_pos = p_bbcode.find("[",pos);

		if (brk_pos<0)
			brk_pos=p_bbcode.length();

		if (brk_pos > pos) {
			class_desc->add_text(p_bbcode.substr(pos,brk_pos-pos));

		}

		if (brk_pos==p_bbcode.length())
			break; //nothing else o add

		int brk_end = p_bbcode.find("]",brk_pos+1);

		if (brk_end==-1) {
			//no close, add the rest
			class_desc->add_text(p_bbcode.substr(brk_pos,p_bbcode.length()-brk_pos));

			break;
		}


		String tag = p_bbcode.substr(brk_pos+1,brk_end-brk_pos-1);


		if (tag.begins_with("/")) {
			bool tag_ok = tag_stack.size() && tag_stack.front()->get()==tag.substr(1,tag.length());
			if (tag_stack.size()) {



			}
			if (!tag_ok) {

				class_desc->add_text("[");
				pos++;
				continue;
			}

			tag_stack.pop_front();
			pos=brk_end+1;
			if (tag!="/img")
				class_desc->pop();

		} else if (tag.begins_with("method ")) {

			String m = tag.substr(7,tag.length());
			class_desc->push_meta("@"+m);
			class_desc->add_text(m+"()");
			class_desc->pop();
			pos=brk_end+1;

		} else if (doc->class_list.has(tag)) {


			class_desc->push_meta("#"+tag);
			class_desc->add_text(tag);
			class_desc->pop();
			pos=brk_end+1;

		} else if (tag=="b") {

			//use bold font
			class_desc->push_font(get_font("source","Fonts"));
			pos=brk_end+1;
			tag_stack.push_front(tag);
		} else if (tag=="i") {

			//use italics font
			//class_desc->push_font(get_font("italic","Fonts"));
			pos=brk_end+1;
			tag_stack.push_front(tag);
		} else if (tag=="code") {

			//use monospace font
			class_desc->push_font(get_font("source","EditorFonts"));
			pos=brk_end+1;
			tag_stack.push_front(tag);
		} else if (tag=="center") {

			//use monospace font
			class_desc->push_align(RichTextLabel::ALIGN_CENTER);
			pos=brk_end+1;
			tag_stack.push_front(tag);
		} else if (tag=="br") {

			//use monospace font
			class_desc->add_newline();
			pos=brk_end+1;
		} else if (tag=="u") {

			//use underline
			class_desc->push_underline();
			pos=brk_end+1;
			tag_stack.push_front(tag);
		} else if (tag=="s") {

			//use strikethrough (not supported underline instead)
			class_desc->push_underline();
			pos=brk_end+1;
			tag_stack.push_front(tag);

		} else if (tag=="url") {

			//use strikethrough (not supported underline instead)
			int end=p_bbcode.find("[",brk_end);
			if (end==-1)
				end=p_bbcode.length();
			String url = p_bbcode.substr(brk_end+1,end-brk_end-1);
			class_desc->push_meta(url);

			pos=brk_end+1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("url=")) {

			String url = tag.substr(4,tag.length());
			class_desc->push_meta(url);
			pos=brk_end+1;
			tag_stack.push_front("url");
		} else if (tag=="img") {

			//use strikethrough (not supported underline instead)
			int end=p_bbcode.find("[",brk_end);
			if (end==-1)
				end=p_bbcode.length();
			String image = p_bbcode.substr(brk_end+1,end-brk_end-1);

			Ref<Texture> texture = ResourceLoader::load(base_path+"/"+image,"Texture");
			if (texture.is_valid())
				class_desc->add_image(texture);

			pos=end;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("color=")) {

			String col = tag.substr(6,tag.length());
			Color color;

			if (col.begins_with("#"))
				color=Color::html(col);
			else if (col=="aqua")
				color=Color::html("#00FFFF");
			else if (col=="black")
				color=Color::html("#000000");
			else if (col=="blue")
				color=Color::html("#0000FF");
			else if (col=="fuchsia")
				color=Color::html("#FF00FF");
			else if (col=="gray" || col=="grey")
				color=Color::html("#808080");
			else if (col=="green")
				color=Color::html("#008000");
			else if (col=="lime")
				color=Color::html("#00FF00");
			else if (col=="maroon")
				color=Color::html("#800000");
			else if (col=="navy")
				color=Color::html("#000080");
			else if (col=="olive")
				color=Color::html("#808000");
			else if (col=="purple")
				color=Color::html("#800080");
			else if (col=="red")
				color=Color::html("#FF0000");
			else if (col=="silver")
				color=Color::html("#C0C0C0");
			else if (col=="teal")
				color=Color::html("#008008");
			else if (col=="white")
				color=Color::html("#FFFFFF");
			else if (col=="yellow")
				color=Color::html("#FFFF00");
			else
				color=Color(0,0,0,1); //base_color;



			class_desc->push_color(color);
			pos=brk_end+1;
			tag_stack.push_front("color");

		} else if (tag.begins_with("font=")) {

			String fnt = tag.substr(5,tag.length());


			Ref<Font> font = ResourceLoader::load(base_path+"/"+fnt,"Font");
			if (font.is_valid())
				class_desc->push_font(font);
			else {
				class_desc->push_font(get_font("source","rFonts"));
			}

			pos=brk_end+1;
			tag_stack.push_front("font");


		} else {

			class_desc->add_text("["); //ignore
			pos=brk_pos+1;

		}
	}

	class_desc->pop();
	class_desc->pop();

}


void EditorHelp::add_type(const String& p_type,HashMap<String,TreeItem*>& p_types,TreeItem *p_root) {

	if (p_types.has(p_type))
		return;
//	if (!ObjectTypeDB::is_type(p_type,base) || p_type==base)
//		return;

	String inherits=doc->class_list[p_type].inherits;

	TreeItem *parent=p_root;


	if (inherits.length()) {

		if (!p_types.has(inherits)) {

			add_type(inherits,p_types,p_root);
		}

		if (p_types.has(inherits) )
			parent=p_types[inherits];
	}

	TreeItem *item = class_list->create_item(parent);
	item->set_metadata(0,p_type);
	item->set_tooltip(0,doc->class_list[p_type].brief_description);
	item->set_text(0,p_type);


	if (has_icon(p_type,"EditorIcons")) {

		item->set_icon(0, get_icon(p_type,"EditorIcons"));
	}

	p_types[p_type]=item;
}



void EditorHelp::_update_doc() {


	class_list->clear();

	List<StringName> type_list;

	tree_item_map.clear();

	TreeItem *root = class_list->create_item();
	class_list->set_hide_root(true);
	List<StringName>::Element *I=type_list.front();

	for(Map<String,DocData::ClassDoc>::Element *E=doc->class_list.front();E;E=E->next()) {


		add_type(E->key(),tree_item_map,root);
	}

}


void EditorHelp::generate_doc() {

	doc = memnew( DocData );
	doc->generate(true);
	DocData compdoc;
	compdoc.load_compressed(_doc_data_compressed,_doc_data_compressed_size,_doc_data_uncompressed_size);
	doc->merge_from(compdoc); //ensure all is up to date


}

void EditorHelp::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_READY: {


			forward->set_icon(get_icon("Forward","EditorIcons"));
			back->set_icon(get_icon("Back","EditorIcons"));
			_update_doc();
			editor->connect("request_help",this,"_request_help");

		} break;
	}
}

void EditorHelp::_tree_item_selected() {

	if (select_locked) {
		select_locked = false;
		return;
	}
	TreeItem *s=class_list->get_selected();
	if (!s)
		return;
	select_locked=true;
	_goto_desc(s->get_text(0));
	select_locked=false;
}

void EditorHelp::_bind_methods() {

	ObjectTypeDB::bind_method("_class_list_select",&EditorHelp::_class_list_select);
	ObjectTypeDB::bind_method("_class_desc_select",&EditorHelp::_class_desc_select);
	ObjectTypeDB::bind_method("_button_pressed",&EditorHelp::_button_pressed);
	ObjectTypeDB::bind_method("_scroll_changed",&EditorHelp::_scroll_changed);
	ObjectTypeDB::bind_method("_request_help",&EditorHelp::_request_help);
	ObjectTypeDB::bind_method("_unhandled_key_input",&EditorHelp::_unhandled_key_input);
	ObjectTypeDB::bind_method("_search",&EditorHelp::_search);
	ObjectTypeDB::bind_method("_tree_item_selected",&EditorHelp::_tree_item_selected);
	ObjectTypeDB::bind_method("_help_callback",&EditorHelp::_help_callback);

}

EditorHelp::EditorHelp(EditorNode *p_editor) {

	editor=p_editor;

	VBoxContainer *vbc = this;

	HBoxContainer *panel_hb = memnew( HBoxContainer );

	Button *b = memnew( Button );
	b->set_text("Class List");
	panel_hb->add_child(b);
	vbc->add_child(panel_hb);
	b->set_toggle_mode(true);
	b->set_pressed(true);
	b->connect("pressed",this,"_button_pressed",make_binds(PAGE_CLASS_LIST));
	class_list_button=b;
	class_list_button->hide();

	b = memnew( Button );
	b->set_text("Class");
	panel_hb->add_child(b);
	edited_class=b;
	edited_class->hide();
	b->set_toggle_mode(true);
	b->connect("pressed",this,"_button_pressed",make_binds(PAGE_CLASS_DESC));

	b = memnew( Button );
	b->set_text("Search in Classes");
	panel_hb->add_child(b);
	b->connect("pressed",this,"_button_pressed",make_binds(CLASS_SEARCH));

	Control *expand = memnew( Control );
	expand->set_h_size_flags(SIZE_EXPAND_FILL);
	panel_hb->add_child(expand);

	b = memnew( Button );
	panel_hb->add_child(b);
	back=b;
	b->connect("pressed",this,"_button_pressed",make_binds(PAGE_CLASS_PREV));

	b = memnew( Button );
	panel_hb->add_child(b);
	forward=b;
	b->connect("pressed",this,"_button_pressed",make_binds(PAGE_CLASS_NEXT));

	Separator *hs = memnew( VSeparator );
	panel_hb->add_child(hs);
	Control *ec = memnew( Control );
	ec->set_custom_minimum_size(Size2(200,1));
	panel_hb->add_child(ec);
	search = memnew( LineEdit );
	ec->add_child(search);
	search->set_area_as_parent_rect();
	search->connect("text_entered",this,"_search");

	b = memnew( Button );
	b->set_text("Find");
	panel_hb->add_child(b);
	b->connect("pressed",this,"_button_pressed",make_binds(PAGE_SEARCH));

	hs = memnew( VSeparator );
	panel_hb->add_child(hs);

	h_split = memnew( HSplitContainer );
	h_split->set_v_size_flags(SIZE_EXPAND_FILL);


	vbc->add_child(h_split);

	class_list = memnew( Tree );
	h_split->add_child(class_list);
	//class_list->connect("meta_clicked",this,"_class_list_select");
	//class_list->set_selection_enabled(true);

	{
		PanelContainer *pc = memnew( PanelContainer );
		Ref<StyleBoxFlat> style( memnew( StyleBoxFlat ) );
		style->set_bg_color( EditorSettings::get_singleton()->get("text_editor/background_color") );	
		style->set_default_margin(MARGIN_LEFT,20);
		style->set_default_margin(MARGIN_TOP,20);
		pc->add_style_override("panel", style); //get_stylebox("normal","TextEdit"));
		h_split->add_child(pc);
		class_desc = memnew( RichTextLabel );
		pc->add_child(class_desc);
		class_desc->connect("meta_clicked",this,"_class_desc_select");
	}

	class_desc->get_v_scroll()->connect("value_changed",this,"_scroll_changed");
	class_desc->set_selection_enabled(true);
	editor=p_editor;
	history_pos=0;
	scroll_locked=false;
	select_locked=false;
	set_process_unhandled_key_input(true);
	h_split->set_split_offset(200);
	class_list->connect("cell_selected",this,"_tree_item_selected");
	class_desc->hide();

	class_search = memnew( EditorHelpSearch(editor) );
	editor->get_gui_base()->add_child(class_search);
	class_search->connect("go_to_help",this,"_help_callback");
//	prev_search_page=-1;
}

EditorHelp::~EditorHelp() {
	if (doc)
		memdelete(doc);
}


void EditorHelpPlugin::edit(Object *p_object) {

	if (!p_object->cast_to<Script>())
		return;

	//editor_help->edit(p_object->cast_to<Script>());
}

bool EditorHelpPlugin::handles(Object *p_object) const {

	return false;
}

void EditorHelpPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		editor_help->show();
	} else {

		editor_help->hide();
	}

}

void EditorHelpPlugin::selected_notify() {

	//editor_help->ensure_select_current();
}

Dictionary EditorHelpPlugin::get_state() const {

	return Dictionary();
}

void EditorHelpPlugin::set_state(const Dictionary& p_state) {

	//editor_help->set_state(p_state);
}
void EditorHelpPlugin::clear() {

	//editor_help->clear();
}

void EditorHelpPlugin::save_external_data() {

	//editor_help->save_external_data();
}

void EditorHelpPlugin::apply_changes() {

	//editor_help->apply_helps();
}

void EditorHelpPlugin::restore_global_state() {

	//if (bool(EDITOR_DEF("text_editor/restore_helps_on_load",true))) {
//		editor_help->_load_files_state();
	//}

}

void EditorHelpPlugin::save_global_state() {

	//if (bool(EDITOR_DEF("text_editor/restore_helps_on_load",true))) {
//		editor_help->_save_files_state();
//	}

}


EditorHelpPlugin::EditorHelpPlugin(EditorNode *p_node) {

	editor=p_node;
	editor_help = memnew( EditorHelp(p_node) );
	editor->get_viewport()->add_child(editor_help);
	editor_help->set_area_as_parent_rect();
	editor_help->hide();


}


EditorHelpPlugin::~EditorHelpPlugin()
{
}
