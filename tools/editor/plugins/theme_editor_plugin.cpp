/*************************************************************************/
/*  theme_editor_plugin.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "version.h"
#include "theme_editor_plugin.h"
#include "os/file_access.h"

void ThemeEditor::edit(const Ref<Theme>& p_theme) {

	theme=p_theme;
	main_vb->set_theme(p_theme);

}


void ThemeEditor::_propagate_redraw(Control *p_at) {

	p_at->notification(NOTIFICATION_THEME_CHANGED);
	p_at->minimum_size_changed();
	p_at->update();
	for(int i=0;i<p_at->get_child_count();i++) {
		Control *a = p_at->get_child(i)->cast_to<Control>();
		if (a)
			_propagate_redraw(a);

	}
}

void ThemeEditor::_refresh_interval() {

	_propagate_redraw(main_vb);

}

void ThemeEditor::_type_menu_cbk(int p_option) {


	type_edit->set_text( type_menu->get_popup()->get_item_text(p_option) );
}

void ThemeEditor::_name_menu_about_to_show() {

	String fromtype=type_edit->get_text();
	List<StringName> names;

	if (popup_mode==POPUP_ADD) {

		switch(type_select->get_selected()) {

			case 0: Theme::get_default()->get_icon_list(fromtype,&names); break;
			case 1: Theme::get_default()->get_stylebox_list(fromtype,&names); break;
			case 2: Theme::get_default()->get_font_list(fromtype,&names); break;
			case 3: Theme::get_default()->get_color_list(fromtype,&names); break;
			case 4: Theme::get_default()->get_constant_list(fromtype,&names); break;
		}
	} else if (popup_mode==POPUP_REMOVE) {

		theme->get_icon_list(fromtype,&names);
		theme->get_stylebox_list(fromtype,&names);
		theme->get_font_list(fromtype,&names);
		theme->get_color_list(fromtype,&names);
		theme->get_constant_list(fromtype,&names);
	}


	name_menu->get_popup()->clear();

	for(List<StringName>::Element *E=names.front();E;E=E->next()) {

		name_menu->get_popup()->add_item(E->get());
	}
}

void ThemeEditor::_name_menu_cbk(int p_option) {

	name_edit->set_text( name_menu->get_popup()->get_item_text(p_option) );
}

struct _TECategory {

	template<class T>
	struct RefItem {

		Ref<T> item;
		StringName name;
		bool operator<(const RefItem<T>& p) const { return item->get_instance_ID() < p.item->get_instance_ID(); }
	};

	template<class T>
	struct Item {

		T item;
		String name;
		bool operator<(const Item<T>& p) const { return name < p.name; }
	};


	Set<RefItem<StyleBox> > stylebox_items;
	Set<RefItem<Font> > font_items;
	Set<RefItem<Texture> > icon_items;

	Set<Item<Color> > color_items;
	Set<Item<int> > constant_items;

};


void ThemeEditor::_save_template_cbk(String fname) {

	String filename = file_dialog->get_current_path();

	Map<String,_TECategory> categories;

	//fill types
	List<StringName> type_list;
	Theme::get_default()->get_type_list(&type_list);
	for (List<StringName>::Element *E=type_list.front();E;E=E->next()) {
		categories.insert(E->get(),_TECategory());
	}

	//fill default theme
	for(Map<String,_TECategory>::Element *E=categories.front();E;E=E->next() ) {

		_TECategory &tc = E->get();

		List<StringName> stylebox_list;
		Theme::get_default()->get_stylebox_list(E->key(),&stylebox_list);
		for (List<StringName>::Element *F=stylebox_list.front();F;F=F->next()) {
			_TECategory::RefItem<StyleBox> it;
			it.name=F->get();
			it.item=Theme::get_default()->get_stylebox(F->get(),E->key());
			tc.stylebox_items.insert(it);
		}

		List<StringName> font_list;
		Theme::get_default()->get_font_list(E->key(),&font_list);
		for (List<StringName>::Element *F=font_list.front();F;F=F->next()) {
			_TECategory::RefItem<Font> it;
			it.name=F->get();
			it.item=Theme::get_default()->get_font(F->get(),E->key());
			tc.font_items.insert(it);
		}

		List<StringName> icon_list;
		Theme::get_default()->get_icon_list(E->key(),&icon_list);
		for (List<StringName>::Element *F=icon_list.front();F;F=F->next()) {
			_TECategory::RefItem<Texture> it;
			it.name=F->get();
			it.item=Theme::get_default()->get_icon(F->get(),E->key());
			tc.icon_items.insert(it);
		}

		List<StringName> color_list;
		Theme::get_default()->get_color_list(E->key(),&color_list);
		for (List<StringName>::Element *F=color_list.front();F;F=F->next()) {
			_TECategory::Item<Color> it;
			it.name=F->get();
			it.item=Theme::get_default()->get_color(F->get(),E->key());
			tc.color_items.insert(it);
		}

		List<StringName> constant_list;
		Theme::get_default()->get_constant_list(E->key(),&constant_list);
		for (List<StringName>::Element *F=constant_list.front();F;F=F->next()) {
			_TECategory::Item<int> it;
			it.name=F->get();
			it.item=Theme::get_default()->get_constant(F->get(),E->key());
			tc.constant_items.insert(it);
		}

	}

	FileAccess *file = FileAccess::open(filename,FileAccess::WRITE);
	if (!file) {


		ERR_EXPLAIN("Can't save theme to file: "+filename);
		return;
	}
	file->store_line("; ******************* ");
	file->store_line("; Template Theme File ");
	file->store_line("; ******************* ");
	file->store_line("; ");
	file->store_line("; Theme Syntax: ");
	file->store_line("; ------------- ");
	file->store_line("; ");
	file->store_line("; Must be placed in section [theme]");
	file->store_line("; ");
	file->store_line("; Type.item = [value] ");
	file->store_line("; ");
	file->store_line("; [value] examples:");
	file->store_line("; ");
	file->store_line("; Type.item = 6 ; numeric constant. ");
	file->store_line("; Type.item = #FF00FF ; HTML color ");
	file->store_line("; Type.item = #55FF00FF ; HTML color with alpha 55.");
	file->store_line("; Type.item = icon(image.png) ; icon in a png file (relative to theme file).");
	file->store_line("; Type.item = font(font.xres) ; font in a resource (relative to theme file).");
	file->store_line("; Type.item = sbox(stylebox.xres) ; stylebox in a resource (relative to theme file).");
	file->store_line("; Type.item = sboxf(2,#FF00FF) ; flat stylebox with margin 2.");
	file->store_line("; Type.item = sboxf(2,#FF00FF,#FFFFFF) ; flat stylebox with margin 2 and border.");
	file->store_line("; Type.item = sboxf(2,#FF00FF,#FFFFFF,#000000) ; flat stylebox with margin 2, light & dark borders.");
	file->store_line("; Type.item = sboxt(base.png,2,2,2,2) ; textured stylebox with 3x3 stretch and stretch margins.");
	file->store_line(";   -Additionally, 4 extra integers can be added to sboxf and sboxt to specify custom padding of contents:");
	file->store_line("; Type.item = sboxt(base.png,2,2,2,2,5,4,2,4) ;");
	file->store_line(";   -Order for all is always left, top, right, bottom.");
	file->store_line("; ");
	file->store_line("; Special values:");
	file->store_line("; Type.item = default ; use the value in the default theme (must exist there).");
	file->store_line("; Type.item = @somebutton_color ; reference to a library value previously defined.");
	file->store_line("; ");
	file->store_line("; Library Syntax: ");
	file->store_line("; --------------- ");
	file->store_line("; ");
	file->store_line("; Must be placed in section [library], but usage is optional.");
	file->store_line("; ");
	file->store_line("; item = [value] ; same as Theme, but assign to library.");
	file->store_line("; ");
	file->store_line("; examples:");
	file->store_line("; ");
	file->store_line("; [library]");
	file->store_line("; ");
	file->store_line("; default_button_color = #FF00FF");
	file->store_line("; ");
	file->store_line("; [theme]");
	file->store_line("; ");
	file->store_line("; Button.color = @default_button_color ; used reference.");
	file->store_line("; ");
	file->store_line("; ******************* ");
	file->store_line("; ");
	file->store_line("; Template Generated Using: "+String(VERSION_MKSTRING));
	file->store_line(";    ");
	file->store_line("; ");
	file->store_line("");
	file->store_line("[library]");
	file->store_line("");
	file->store_line("; place library stuff here");
	file->store_line("");
	file->store_line("[theme]");
	file->store_line("");
	file->store_line("");

	//write default theme
	for(Map<String,_TECategory>::Element *E=categories.front();E;E=E->next() ) {

		_TECategory &tc = E->get();

		String underline="; ";
		for(int i=0;i<E->key().length();i++)
			underline+="*";

		file->store_line("");
		file->store_line(underline);
		file->store_line("; "+E->key());
		file->store_line(underline);

		if (tc.stylebox_items.size())
			file->store_line("\n; StyleBox Items:\n");

		for (Set<_TECategory::RefItem<StyleBox> >::Element *F=tc.stylebox_items.front();F;F=F->next()) {

			file->store_line(E->key()+"."+F->get().name+" = default");
		}

		if (tc.font_items.size())
			file->store_line("\n; Font Items:\n");

		for (Set<_TECategory::RefItem<Font> >::Element *F=tc.font_items.front();F;F=F->next()) {

			file->store_line(E->key()+"."+F->get().name+" = default");
		}

		if (tc.icon_items.size())
			file->store_line("\n; Icon Items:\n");

		for (Set<_TECategory::RefItem<Texture> >::Element *F=tc.icon_items.front();F;F=F->next()) {

			file->store_line(E->key()+"."+F->get().name+" = default");
		}

		if (tc.color_items.size())
			file->store_line("\n; Color Items:\n");

		for (Set<_TECategory::Item<Color> >::Element *F=tc.color_items.front();F;F=F->next()) {

			file->store_line(E->key()+"."+F->get().name+" = default");
		}

		if (tc.constant_items.size())
			file->store_line("\n; Constant Items:\n");

		for (Set<_TECategory::Item<int> >::Element *F=tc.constant_items.front();F;F=F->next()) {

			file->store_line(E->key()+"."+F->get().name+" = default");
		}

	}

	file->close();
	memdelete(file);
}

void ThemeEditor::_dialog_cbk() {

	switch(popup_mode) {
		case POPUP_ADD: {

			switch(type_select->get_selected()) {

				case 0: theme->set_icon(name_edit->get_text(),type_edit->get_text(),Ref<Texture>()); break;
				case 1: theme->set_stylebox(name_edit->get_text(),type_edit->get_text(),Ref<StyleBox>()); break;
				case 2: theme->set_font(name_edit->get_text(),type_edit->get_text(),Ref<Font>()); break;
				case 3: theme->set_color(name_edit->get_text(),type_edit->get_text(),Color()); break;
				case 4: theme->set_constant(name_edit->get_text(),type_edit->get_text(),0); break;
			}

		} break;
		case POPUP_CLASS_ADD: {

			StringName fromtype = type_edit->get_text();
			List<StringName> names;

			{
				names.clear();
				Theme::get_default()->get_icon_list(fromtype,&names);
				for(List<StringName>::Element *E=names.front();E;E=E->next()) {
					theme->set_icon(E->get(),fromtype,Theme::get_default()->get_icon(E->get(),fromtype));

				}

			}
			{
				names.clear();
				Theme::get_default()->get_stylebox_list(fromtype,&names);
				for(List<StringName>::Element *E=names.front();E;E=E->next()) {
					theme->set_stylebox(E->get(),fromtype,Theme::get_default()->get_stylebox(E->get(),fromtype));

				}

			}
			{
				names.clear();
				Theme::get_default()->get_font_list(fromtype,&names);
				for(List<StringName>::Element *E=names.front();E;E=E->next()) {
					theme->set_font(E->get(),fromtype,Theme::get_default()->get_font(E->get(),fromtype));

				}
			}
			{
				names.clear();
				Theme::get_default()->get_color_list(fromtype,&names);
				for(List<StringName>::Element *E=names.front();E;E=E->next()) {
					theme->set_color(E->get(),fromtype,Theme::get_default()->get_color(E->get(),fromtype));

				}
			}
			{
				names.clear();
				Theme::get_default()->get_constant_list(fromtype,&names);
				for(List<StringName>::Element *E=names.front();E;E=E->next()) {
					theme->set_constant(E->get(),fromtype,Theme::get_default()->get_constant(E->get(),fromtype));

				}
			}
		} break;
		case POPUP_REMOVE: {
			switch(type_select->get_selected()) {

				case 0: theme->clear_icon(name_edit->get_text(),type_edit->get_text()); break;
				case 1: theme->clear_stylebox(name_edit->get_text(),type_edit->get_text()); break;
				case 2: theme->clear_font(name_edit->get_text(),type_edit->get_text()); break;
				case 3: theme->clear_color(name_edit->get_text(),type_edit->get_text()); break;
				case 4: theme->clear_constant(name_edit->get_text(),type_edit->get_text()); break;
			}


		} break;
	}

}

void ThemeEditor::_theme_menu_cbk(int p_option) {


	if (p_option==POPUP_CREATE_TEMPLATE) {

		file_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
		file_dialog->set_current_path("custom.theme");
		file_dialog->popup_centered_ratio();
		return;
	}

	Ref<Theme> base_theme;

	type_select->show();
	type_select_label->show();
	name_select_label->show();
	name_edit->show();
	name_menu->show();


	if (p_option==POPUP_ADD) {//add

		add_del_dialog->set_title("Add Item");
		add_del_dialog->get_ok()->set_text("Add");
		add_del_dialog->popup_centered(Size2(490,85));

		base_theme=Theme::get_default();

	} else if (p_option==POPUP_CLASS_ADD) {//add

		add_del_dialog->set_title("Add All Items");
		add_del_dialog->get_ok()->set_text("Add All");
		add_del_dialog->popup_centered(Size2(240,85));

		base_theme=Theme::get_default();

		type_select->hide();
		name_select_label->hide();
		type_select_label->hide();
		name_edit->hide();
		name_menu->hide();

	} else if (p_option==POPUP_REMOVE) {

		add_del_dialog->set_title("Remove Item");
		add_del_dialog->get_ok()->set_text("Remove");
		add_del_dialog->popup_centered(Size2(490,85));

		base_theme=theme;

	}
	popup_mode=p_option;

	ERR_FAIL_COND( theme.is_null() );

	List<StringName> types;
	base_theme->get_type_list(&types);
	type_menu->get_popup()->clear();;

	if (p_option==0 || p_option==1) {//add

		List<StringName> new_types;
		theme->get_type_list(&new_types);
		//uh kind of sucks
		for(List<StringName>::Element *F=new_types.front();F;F=F->next()) {

			bool found=false;
			for(List<StringName>::Element *E=types.front();E;E=E->next()) {

				if (E->get()==F->get())	{
					found=true;
					break;
				}
			}

			if (!found)
				types.push_back(F->get());
		}
	}

	types.sort();

	for(List<StringName>::Element *E=types.front();E;E=E->next()) {

		type_menu->get_popup()->add_item( E->get() );
	}

}

void ThemeEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_PROCESS) {

		time_left-=get_process_delta_time();
		if (time_left<0) {
			time_left=1.5;
			_refresh_interval();
		}
	}
}

void ThemeEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_type_menu_cbk",&ThemeEditor::_type_menu_cbk);
	ObjectTypeDB::bind_method("_name_menu_about_to_show",&ThemeEditor::_name_menu_about_to_show);
	ObjectTypeDB::bind_method("_name_menu_cbk",&ThemeEditor::_name_menu_cbk);
	ObjectTypeDB::bind_method("_theme_menu_cbk",&ThemeEditor::_theme_menu_cbk);
	ObjectTypeDB::bind_method("_dialog_cbk",&ThemeEditor::_dialog_cbk);
	ObjectTypeDB::bind_method("_save_template_cbk",&ThemeEditor::_save_template_cbk);
}

ThemeEditor::ThemeEditor() {

	time_left=0;

	Panel * panel = memnew( Panel );
	add_child(panel);
	panel->set_area_as_parent_rect(0);
	panel->set_margin(MARGIN_TOP,25);

	main_vb= memnew( VBoxContainer );
	panel->add_child(main_vb);
	main_vb->set_area_as_parent_rect(4);


	HBoxContainer *hb_menu = memnew(HBoxContainer);
	main_vb->add_child(hb_menu);



	theme_menu = memnew( MenuButton );
	theme_menu->set_text("Theme");
	theme_menu->get_popup()->add_item("Add Item",POPUP_ADD);
	theme_menu->get_popup()->add_item("Add Class Items",POPUP_CLASS_ADD);
	theme_menu->get_popup()->add_item("Remove Item",POPUP_REMOVE);
	theme_menu->get_popup()->add_separator();
	theme_menu->get_popup()->add_item("Create Template",POPUP_CREATE_TEMPLATE);
	hb_menu->add_child(theme_menu);
	theme_menu->get_popup()->connect("item_pressed", this,"_theme_menu_cbk");


	HBoxContainer *main_hb = memnew( HBoxContainer );
	main_vb->add_child(main_hb);



	VBoxContainer *first_vb = memnew( VBoxContainer);
	first_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	main_hb->add_child(first_vb);



//	main_panel->add_child(panel);
//	panel->set_area_as_parent_rect();
//	panel->set_margin( MARGIN_TOP,20 );

	first_vb->add_child(memnew( Label("Label") ));

	first_vb->add_child(memnew( Button("Button")) );
	ToolButton *tb = memnew( ToolButton );
	tb->set_text("ToolButton");
	first_vb->add_child(tb );
	CheckButton *cb = memnew( CheckButton );
	cb->set_text("CheckButton");
	first_vb->add_child(cb );
	CheckBox *cbx = memnew( CheckBox );
	cbx->set_text("CheckBox");
	first_vb->add_child(cbx );


	ButtonGroup *bg = memnew( ButtonGroup );
	bg->set_v_size_flags(SIZE_EXPAND_FILL);
	VBoxContainer *gbvb = memnew( VBoxContainer );
	gbvb->set_v_size_flags(SIZE_EXPAND_FILL);
	CheckBox *rbx1 = memnew( CheckBox );
	rbx1->set_text("CheckBox Radio1");
	rbx1->set_pressed(true);
	gbvb->add_child(rbx1);
	CheckBox *rbx2 = memnew( CheckBox );
	rbx2->set_text("CheckBox Radio2");
	gbvb->add_child(rbx2);
	bg->add_child(gbvb);
	first_vb->add_child(bg);

	MenuButton* test_menu_button = memnew( MenuButton );
	test_menu_button->set_text("MenuButton");
	test_menu_button->get_popup()->add_item("Item");
	test_menu_button->get_popup()->add_separator();
	test_menu_button->get_popup()->add_check_item("Check Item");
	test_menu_button->get_popup()->add_check_item("Checked Item");
	test_menu_button->get_popup()->set_item_checked(2,true);
	first_vb->add_child(test_menu_button);

	OptionButton *test_option_button = memnew( OptionButton );
	test_option_button->add_item("OptionButton");
	test_option_button->add_separator();
	test_option_button->add_item("Has");
	test_option_button->add_item("Many");
	test_option_button->add_item("Options");
	first_vb->add_child(test_option_button);

	ColorPickerButton *cpb = memnew( ColorPickerButton );
	first_vb->add_child(cpb );

	first_vb->add_child( memnew( HSeparator ));
	first_vb->add_child( memnew( HSlider ));
	first_vb->add_child( memnew( HScrollBar ));
	first_vb->add_child( memnew( SpinBox ));
	ProgressBar *pb=memnew( ProgressBar );
	pb->set_val(50);
	first_vb->add_child( pb);
	Panel *pn=memnew( Panel );
	pn->set_custom_minimum_size(Size2(40,40));
	first_vb->add_child( pn);
	first_vb->add_constant_override("separation",10);

	VBoxContainer *second_vb = memnew( VBoxContainer );
	second_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	main_hb->add_child(second_vb);
	second_vb->add_constant_override("separation",10);
	LineEdit *le =  memnew( LineEdit );
	le->set_text("LineEdit");
	second_vb->add_child(le);
	TextEdit *te =  memnew( TextEdit );
	te->set_text("TextEdit");
	//te->set_v_size_flags(SIZE_EXPAND_FILL);
	te->set_custom_minimum_size(Size2(0,160));
	second_vb->add_child(te);

	Tree *test_tree = memnew(Tree);
	second_vb->add_child(test_tree);
	test_tree->set_custom_minimum_size(Size2(0,160));


	TreeItem *item = test_tree->create_item();
	item->set_editable(0,true);
	item->set_text(0,"Tree");
	item = test_tree->create_item( test_tree->get_root() );
	item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	item->set_editable(0,true);
	item->set_text(0,"check");
	item = test_tree->create_item( test_tree->get_root() );
	item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
	item->set_editable(0,true);
	item->set_range_config(0,0,20,0.1);
	item->set_range(0,2);
	item = test_tree->create_item( test_tree->get_root() );
	item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
	item->set_editable(0,true);
	item->set_text(0,"Have,Many,Several,Options!");
	item->set_range(0,2);

	VBoxContainer *third_vb = memnew( VBoxContainer );
	third_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	third_vb->add_constant_override("separation",10);

	main_hb->add_child(third_vb);

	HBoxContainer *vhb = memnew( HBoxContainer );
	vhb->set_custom_minimum_size(Size2(0,160));
	vhb->add_child(memnew(VSeparator));
	vhb->add_child(memnew(VSlider));
	vhb->add_child(memnew(VScrollBar));
	third_vb->add_child(vhb);

	TabContainer *tc = memnew( TabContainer );
	third_vb->add_child(tc);
	tc->set_custom_minimum_size(Size2(0,160));
	Control *tcc = memnew( Control );
	tcc->set_name("Tab 1");
	tc->add_child(tcc);
	tcc = memnew( Control );
	tcc->set_name("Tab 2");
	tc->add_child(tcc);
	tcc = memnew( Control );
	tcc->set_name("Tab 3");
	tc->add_child(tcc);

	main_hb->add_constant_override("separation",20);




/*
	test_h_scroll = memnew( HScrollBar );
	test_h_scroll->set_pos( Point2( 25, 225 ) );
	test_h_scroll->set_size( Point2( 150, 5 ) );
	panel->add_child(test_h_scroll);

	line_edit = memnew( LineEdit );
	line_edit->set_pos( Point2( 25, 275 ) );
	line_edit->set_size( Point2( 150, 5 ) );
	line_edit->set_text("Line Edit");
	panel->add_child(line_edit);

	test_v_scroll = memnew( VScrollBar );
	test_v_scroll->set_pos( Point2( 200, 25 ) );
	test_v_scroll->set_size( Point2( 5, 150 ) );
	panel->add_child(test_v_scroll);

	test_tree = memnew(Tree);
	test_tree->set_pos( Point2( 300, 25 ) );
	test_tree->set_size( Point2( 200, 200 ) );
	panel->add_child(test_tree);


	TreeItem *item = test_tree->create_item();
	item->set_editable(0,true);
	item->set_text(0,"root");
	item = test_tree->create_item( test_tree->get_root() );
	item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	item->set_editable(0,true);
	item->set_text(0,"check");
	item = test_tree->create_item( test_tree->get_root() );
	item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
	item->set_editable(0,true);
	item->set_range_config(0,0,20,0.1);
	item->set_range(0,2);
	item = test_tree->create_item( test_tree->get_root() );
	item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
	item->set_editable(0,true);
	item->set_text(0,"Have,Many,Several,Options!");
	item->set_range(0,2);

	Button *fd_button= memnew( Button );
	fd_button->set_pos(Point2(300,275));
	fd_button->set_text("Open File Dialog");
	panel->add_child(fd_button);

	test_file_dialog = memnew( EditorFileDialog );
	panel->add_child(test_file_dialog);

	fd_button->connect("pressed", this,"_open_file_dialog");
*/

	add_del_dialog = memnew(ConfirmationDialog);
	add_del_dialog->hide();
	add_child(add_del_dialog);


	Label *l = memnew( Label );
	l->set_pos( Point2(5,5) );
	l->set_text("Type:");
	add_del_dialog->add_child(l);
	dtype_select_label=l;


	type_edit = memnew( LineEdit );
	type_edit->set_pos(Point2(5,25));
	type_edit->set_size(Point2(150,5));
	add_del_dialog->add_child(type_edit);
	type_menu = memnew( MenuButton );
	type_menu->set_pos(Point2(160,25));
	type_menu->set_size(Point2(30,5));
	type_menu->set_text("..");
	add_del_dialog->add_child(type_menu);

	type_menu->get_popup()->connect("item_pressed", this,"_type_menu_cbk");

	l = memnew( Label );
	l->set_pos( Point2(200,5) );
	l->set_text("Name:");
	add_del_dialog->add_child(l);
	name_select_label=l;

	name_edit = memnew( LineEdit );
	name_edit->set_pos(Point2(200,25));
	name_edit->set_size(Point2(150,5));
	add_del_dialog->add_child(name_edit);
	name_menu = memnew( MenuButton );
	name_menu->set_pos(Point2(360,25));
	name_menu->set_size(Point2(30,5));
	name_menu->set_text("..");

	add_del_dialog->add_child(name_menu);

	name_menu->get_popup()->connect("about_to_show", this,"_name_menu_about_to_show");
	name_menu->get_popup()->connect("item_pressed", this,"_name_menu_cbk");

	type_select_label= memnew( Label );
	type_select_label->set_pos( Point2(400,5) );
	type_select_label->set_text("Data Type:");
	add_del_dialog->add_child(type_select_label);

	type_select = memnew( OptionButton );
	type_select->add_item("Icon");
	type_select->add_item("Style");
	type_select->add_item("Font");
	type_select->add_item("Color");
	type_select->add_item("Constant");
	type_select->set_pos( Point2( 400,25 ) );
	type_select->set_size( Point2( 80,5 ) );


	add_del_dialog->add_child(type_select);

	add_del_dialog->get_ok()->connect("pressed", this,"_dialog_cbk");


	file_dialog = memnew( EditorFileDialog );
	file_dialog->add_filter("*.theme ; Theme File");
	add_child(file_dialog);
	file_dialog->connect("file_selected",this,"_save_template_cbk");

	//MenuButton *name_menu;
	//LineEdit *name_edit;

}

void ThemeEditorPlugin::edit(Object *p_node) {

	if (p_node && p_node->cast_to<Theme>()) {
		theme_editor->show();
		theme_editor->edit( p_node->cast_to<Theme>() );
	} else {
		theme_editor->edit( Ref<Theme>() );
		theme_editor->hide();
	}
}

bool ThemeEditorPlugin::handles(Object *p_node) const{

	return p_node->is_type("Theme");
}

void ThemeEditorPlugin::make_visible(bool p_visible){

	if (p_visible) {
		theme_editor->show();
		theme_editor->set_process(true);
	} else {
		theme_editor->hide();
		theme_editor->set_process(false);
	}
}

ThemeEditorPlugin::ThemeEditorPlugin(EditorNode *p_node) {

	theme_editor = memnew( ThemeEditor );

	p_node->get_viewport()->add_child(theme_editor);
	theme_editor->set_area_as_parent_rect();
	theme_editor->hide();

}

