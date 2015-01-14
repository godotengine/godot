/*************************************************************************/
/*  code_editor.cpp                                                      */
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
#include "code_editor.h"
#include "editor_settings.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/separator.h"

void GotoLineDialog::popup_find_line(TextEdit *p_edit) {

	text_editor=p_edit;

	line->set_text(itos(text_editor->cursor_get_line()));
	line->select_all();
	popup_centered(Size2(180,80));
	line->grab_focus();
}


int GotoLineDialog::get_line() const {

	return line->get_text().to_int();
}


void GotoLineDialog::ok_pressed() {

	if (get_line()<1 || get_line()>text_editor->get_line_count())
		return;
	text_editor->cursor_set_line(get_line()-1);
	hide();
}

GotoLineDialog::GotoLineDialog() {

	set_title("Go to Line");
	Label *l = memnew(Label);
	l->set_text("Line Number:");
	l->set_pos(Point2(5,5));
	add_child(l);

	line = memnew( LineEdit );
	line->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	line->set_begin( Point2(15,22) );
	line->set_end( Point2(15,35) );
	add_child(line);
	register_text_enter(line);
	text_editor=NULL;

	set_hide_on_ok(false);
}


void FindReplaceDialog::popup_search() {

	set_title("Search");
	replace_mc->hide();
	replace_label->hide();
	replace_vb->hide();
	skip->hide();
	popup_centered(Point2(300,190));
	get_ok()->set_text("Find");
	search_text->grab_focus();	
	if (text_edit->is_selection_active() && ( text_edit->get_selection_from_line() == text_edit->get_selection_to_line())) {

		search_text->set_text( text_edit->get_selection_text() );
	}
	search_text->select_all();

	error_label->set_text("");

}

void FindReplaceDialog::popup_replace() {

	set_title("Replace");
	bool do_selection=(text_edit->is_selection_active() && text_edit->get_selection_from_line() < text_edit->get_selection_to_line());
	set_replace_selection_only(do_selection);

	if (!do_selection && text_edit->is_selection_active()) {
		search_text->set_text(text_edit->get_selection_text());
	}

	replace_mc->show();
	replace_label->show();
	replace_vb->show();
	popup_centered(Point2(300,300));
	if (search_text->get_text()!="" && replace_text->get_text()=="") {
		search_text->select(0,0);
		replace_text->grab_focus();
	} else {
		search_text->grab_focus();
		search_text->select_all();
	}
	error_label->set_text("");

	if (prompt->is_pressed()) {
		skip->show();
		get_ok()->set_text("Next");
		selection_only->set_disabled(true);

	} else {
		skip->hide();
		get_ok()->set_text("Replace");
		selection_only->set_disabled(false);
	}

}

void FindReplaceDialog::_search_callback() {

	if (is_replace_mode())
		_replace();
	else
		_search();

}

void FindReplaceDialog::_replace_skip_callback() {

	_search();
}

void FindReplaceDialog::_replace() {

	if (is_replace_all_mode()) {

		//line as x so it gets priority in comparison, column as y
		Point2i orig_cursor(text_edit->cursor_get_line(),text_edit->cursor_get_column());
		Point2i prev_match=Point2(-1,-1);


		bool selection_enabled = text_edit->is_selection_active();
		Point2i selection_begin,selection_end;
		if (selection_enabled) {
			selection_begin=Point2i(text_edit->get_selection_from_line(),text_edit->get_selection_from_column());
			selection_end=Point2i(text_edit->get_selection_to_line(),text_edit->get_selection_to_column());
		}
		int vsval = text_edit->get_v_scroll();
		//int hsval = text_edit->get_h_scroll();

		text_edit->cursor_set_line(0);
		text_edit->cursor_set_column(0);

		int rc=0;

		while(_search()) {

			if (!text_edit->is_selection_active()) {
				//search selects
				break;
			}

			//replace area
			Point2i match_from(text_edit->get_selection_from_line(),text_edit->get_selection_from_column());
			Point2i match_to(text_edit->get_selection_to_line(),text_edit->get_selection_to_column());

			if (match_from < prev_match)
				break; //done

			prev_match=match_to;

			if (selection_enabled && is_replace_selection_only()) {

				if (match_from<selection_begin || match_to>selection_end)
					continue;

				//replace but adjust selection bounds

				text_edit->insert_text_at_cursor(get_replace_text());
				if (match_to.x==selection_end.x)
					selection_end.y+=get_replace_text().length() - get_search_text().length();
			} else {
				//just replace
				text_edit->insert_text_at_cursor(get_replace_text());
			}
			rc++;

		}
		//restore editor state (selection, cursor, scroll)
		text_edit->cursor_set_line(orig_cursor.x);
		text_edit->cursor_set_column(orig_cursor.y);

		if (selection_enabled && is_replace_selection_only()) {
			//reselect
			text_edit->select(selection_begin.x,selection_begin.y,selection_end.x,selection_end.y);
		} else {
			text_edit->deselect();
		}

		text_edit->set_v_scroll(vsval);
//		text_edit->set_h_scroll(hsval);
		error_label->set_text("Replaced "+itos(rc)+" ocurrence(s).");


		//hide();
	} else {

		if (text_edit->get_selection_text()==get_search_text()) {

			text_edit->insert_text_at_cursor(get_replace_text());
		}

		_search();
	}

}



bool FindReplaceDialog::_search() {


	String text=get_search_text();
	uint32_t flags=0;

	if (is_whole_words())
		flags|=TextEdit::SEARCH_WHOLE_WORDS;
	if (is_case_sensitive())
		flags|=TextEdit::SEARCH_MATCH_CASE;
	if (is_backwards())
		flags|=TextEdit::SEARCH_BACKWARDS;

	int line=text_edit->cursor_get_line(),col=text_edit->cursor_get_column();

	if (is_backwards()) {
		col-=1;
		if (col<0) {
			line-=1;
			if (line<0) {
				line=text_edit->get_line_count()-1;
			}
			col=text_edit->get_line(line).length();
		}
	}
	bool found = text_edit->search(text,flags,line,col,line,col);


	if (found) {
		// print_line("found");
		text_edit->cursor_set_line(line);
		if (is_backwards())
			text_edit->cursor_set_column(col);
		else
			text_edit->cursor_set_column(col+text.length());
		text_edit->select(line,col,line,col+text.length());
		set_error("");
		return true;
	} else {

		set_error("Not Found!");
		return false;
	}

}

void FindReplaceDialog::_prompt_changed() {

	if (prompt->is_pressed()) {
		skip->show();
		get_ok()->set_text("Next");
		selection_only->set_disabled(true);

	} else {
		skip->hide();
		get_ok()->set_text("Replace");
		selection_only->set_disabled(false);
	}
}


void FindReplaceDialog::_skip_pressed() {

	_replace_skip_callback();
}

bool FindReplaceDialog::is_replace_mode() const {

	return replace_text->is_visible();
}

bool FindReplaceDialog::is_replace_all_mode() const {

	return !prompt->is_pressed();
}

bool FindReplaceDialog::is_replace_selection_only() const {

	return 	selection_only->is_pressed();
}
void FindReplaceDialog::set_replace_selection_only(bool p_enable){

	selection_only->set_pressed(p_enable);
}


void FindReplaceDialog::ok_pressed() {

	_search_callback();
}

void FindReplaceDialog::_search_text_entered(const String& p_text) {

	if (replace_text->is_visible())
		return;
	emit_signal("search");
	_search();

}

void FindReplaceDialog::_replace_text_entered(const String& p_text) {

	if (!replace_text->is_visible())
		return;

	emit_signal("search");
	_replace();

}


String FindReplaceDialog::get_search_text() const {

	return search_text->get_text();
}
String FindReplaceDialog::get_replace_text() const {

	return replace_text->get_text();
}
bool FindReplaceDialog::is_whole_words() const {

	return whole_words->is_pressed();
}
bool FindReplaceDialog::is_case_sensitive() const {

	return case_sensitive->is_pressed();

}
bool FindReplaceDialog::is_backwards() const {

	return backwards->is_pressed();

}

void FindReplaceDialog::set_error(const String& p_error) {

	error_label->set_text(p_error);
}

void FindReplaceDialog::set_text_edit(TextEdit *p_text_edit) {

	text_edit=p_text_edit;
}

void FindReplaceDialog::search_next() {
	_search();
}


void FindReplaceDialog::_bind_methods() {

	ObjectTypeDB::bind_method("_search_text_entered",&FindReplaceDialog::_search_text_entered);
	ObjectTypeDB::bind_method("_replace_text_entered",&FindReplaceDialog::_replace_text_entered);
	ObjectTypeDB::bind_method("_prompt_changed",&FindReplaceDialog::_prompt_changed);
	ObjectTypeDB::bind_method("_skip_pressed",&FindReplaceDialog::_skip_pressed);
	ADD_SIGNAL(MethodInfo("search"));
	ADD_SIGNAL(MethodInfo("skip"));

}

FindReplaceDialog::FindReplaceDialog() {

	set_self_opacity(0.8);

	VBoxContainer *vb = memnew( VBoxContainer );
	add_child(vb);
	set_child_rect(vb);


	search_text = memnew( LineEdit );
	vb->add_margin_child("Search",search_text);
	search_text->connect("text_entered", this,"_search_text_entered");
	//search_text->set_self_opacity(0.7);



	replace_label = memnew( Label);
	replace_label->set_text("Replace By");
	vb->add_child(replace_label);
	replace_mc= memnew( MarginContainer);
	vb->add_child(replace_mc);

	replace_text = memnew( LineEdit );
	replace_text->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	replace_text->set_begin( Point2(15,132) );
	replace_text->set_end( Point2(15,135) );
	//replace_text->set_self_opacity(0.7);
	replace_mc->add_child(replace_text);


	replace_text->connect("text_entered", this,"_replace_text_entered");



	MarginContainer *opt_mg = memnew( MarginContainer );
	vb->add_child(opt_mg);
	VBoxContainer *svb = memnew( VBoxContainer);
	opt_mg->add_child(svb);

	svb ->add_child(memnew(Label));

	whole_words = memnew( CheckButton );
	whole_words->set_text("Whole Words");
	svb->add_child(whole_words);

	case_sensitive = memnew( CheckButton );
	case_sensitive->set_text("Case Sensitive");
	svb->add_child(case_sensitive);

	backwards = memnew( CheckButton );
	backwards->set_text("Backwards");
	svb->add_child(backwards);

	opt_mg = memnew( MarginContainer );
	vb->add_child(opt_mg);
	VBoxContainer *rvb = memnew( VBoxContainer);
	opt_mg->add_child(rvb);
	replace_vb=rvb;
//	rvb ->add_child(memnew(HSeparator));
	rvb ->add_child(memnew(Label));

	prompt = memnew( CheckButton );
	prompt->set_text("Prompt On Replace");
	rvb->add_child(prompt);
	prompt->connect("pressed", this,"_prompt_changed");

	selection_only = memnew( CheckButton );
	selection_only->set_text("Selection Only");
	rvb->add_child(selection_only);


	int margin = get_constant("margin","Dialogs");
	int button_margin = get_constant("button_margin","Dialogs");

	skip = memnew( Button );
	skip->set_anchor( MARGIN_LEFT, ANCHOR_END );
	skip->set_anchor( MARGIN_TOP, ANCHOR_END );
	skip->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	skip->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	skip->set_begin( Point2( 70, button_margin ) );
	skip->set_end( Point2(  10, margin ) );
	skip->set_text("Skip");
	add_child(skip);
	skip->connect("pressed", this,"_skip_pressed");


	error_label = memnew( Label );
	error_label->set_align(Label::ALIGN_CENTER);
	error_label->add_color_override("font_color",Color(1,0.4,0.3));
	error_label->add_color_override("font_color_shadow",Color(0,0,0,0.2));
	error_label->add_constant_override("shadow_as_outline",1);

	vb->add_child(error_label);


	set_hide_on_ok(false);

}


/*** CODE EDITOR ****/

void CodeTextEditor::_line_col_changed() {

	String text = String()+"Line: "+itos(text_editor->cursor_get_line()+1)+", Col: "+itos(text_editor->cursor_get_column());
	line_col->set_text(text);
}

void CodeTextEditor::_text_changed() {

	code_complete_timer->start();
	idle->start();
}

void CodeTextEditor::_code_complete_timer_timeout() {
	if (!is_visible())
		return;
	if (enable_complete_timer)
		text_editor->query_code_comple();
}

void CodeTextEditor::_complete_request() {

	List<String> entries;
	_code_complete_script(text_editor->get_text_for_completion(),&entries);
	// print_line("COMPLETE: "+p_request);
	if (entries.size()==0)
		return;
	Vector<String> strs;
	strs.resize(entries.size());
	int i=0;
	for(List<String>::Element *E=entries.front();E;E=E->next()) {

		strs[i++]=E->get();
	}

	text_editor->code_complete(strs);
}

void CodeTextEditor::set_error(const String& p_error) {

	if (p_error!="") {
		error->set_text(p_error);
		error->show();
	} else {
		error->hide();
	}

}

void CodeTextEditor::_on_settings_change() {
	
	// FONTS
	String editor_font = EDITOR_DEF("text_editor/font", "");
	bool font_overrode = false;
	if (editor_font!="") {
		Ref<Font> fnt = ResourceLoader::load(editor_font);
		if (fnt.is_valid()) {
			text_editor->add_font_override("font",fnt);
			font_overrode = true;
		}
	}
	if(!font_overrode)
		text_editor->add_font_override("font",get_font("source","Fonts"));
	
	// AUTO BRACE COMPLETION 
	text_editor->set_auto_brace_completion(
		EDITOR_DEF("text_editor/auto_brace_complete", true)
	);

	code_complete_timer->set_wait_time(
		EDITOR_DEF("text_editor/code_complete_delay",.3f)
	);

	enable_complete_timer = EDITOR_DEF("text_editor/enable_code_completion_delay",true);
}

void CodeTextEditor::_text_changed_idle_timeout() {


	_validate_script();
}

void CodeTextEditor::_notification(int p_what) {


	if (p_what==EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED)
		_load_theme_settings();
}

void CodeTextEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_line_col_changed",&CodeTextEditor::_line_col_changed);
	ObjectTypeDB::bind_method("_text_changed",&CodeTextEditor::_text_changed);
	ObjectTypeDB::bind_method("_on_settings_change",&CodeTextEditor::_on_settings_change);
	ObjectTypeDB::bind_method("_text_changed_idle_timeout",&CodeTextEditor::_text_changed_idle_timeout);
	ObjectTypeDB::bind_method("_code_complete_timer_timeout",&CodeTextEditor::_code_complete_timer_timeout);
	ObjectTypeDB::bind_method("_complete_request",&CodeTextEditor::_complete_request);
}

CodeTextEditor::CodeTextEditor() {

	text_editor = memnew( TextEdit );
	add_child(text_editor);
	text_editor->set_area_as_parent_rect();
	text_editor->set_margin(MARGIN_BOTTOM,20);

	String editor_font = EDITOR_DEF("text_editor/font", "");
	bool font_overrode = false;
	if (editor_font!="") {
		Ref<Font> fnt = ResourceLoader::load(editor_font);
		if (fnt.is_valid()) {
			text_editor->add_font_override("font",fnt);
			font_overrode = true;
		}
	}

	if (!font_overrode)
		text_editor->add_font_override("font",get_font("source","Fonts"));
	text_editor->set_show_line_numbers(true);
	text_editor->set_brace_matching(true);

	line_col = memnew( Label );
	add_child(line_col);
	line_col->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,135);
	line_col->set_anchor_and_margin(MARGIN_TOP,ANCHOR_END,20);
	line_col->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,1);
	line_col->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
	//line_col->set_align(Label::ALIGN_RIGHT);
	idle = memnew( Timer );
	add_child(idle);
	idle->set_one_shot(true);
	idle->set_wait_time(EDITOR_DEF("text_editor/idle_parse_delay",2));

	code_complete_timer = memnew(Timer);
	add_child(code_complete_timer);
	code_complete_timer->set_one_shot(true);
	enable_complete_timer = EDITOR_DEF("text_editor/enable_code_completion_delay",true);

	code_complete_timer->set_wait_time(EDITOR_DEF("text_editor/code_complete_delay",.3f));

	error = memnew( Label );
	add_child(error);
	error->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,5);
	error->set_anchor_and_margin(MARGIN_TOP,ANCHOR_END,20);
	error->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,1);
	error->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,130);
	error->hide();
	error->add_color_override("font_color",Color(1,0.7,0.6,0.9));



	text_editor->connect("cursor_changed", this,"_line_col_changed");
	text_editor->connect("text_changed", this,"_text_changed");
	text_editor->connect("request_completion", this,"_complete_request");
	Vector<String> cs;
	cs.push_back(".");
	cs.push_back(",");
	cs.push_back("(");
	text_editor->set_completion(true,cs);
	idle->connect("timeout", this,"_text_changed_idle_timeout");

	code_complete_timer->connect("timeout", this,"_code_complete_timer_timeout");

	EditorSettings::get_singleton()->connect("settings_changed",this,"_on_settings_change");
}
