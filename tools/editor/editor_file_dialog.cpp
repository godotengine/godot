#include "editor_file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/center_container.h"
#include "print_string.h"
#include "os/keyboard.h"
#include "editor_resource_preview.h"


EditorFileDialog::GetIconFunc EditorFileDialog::get_icon_func=NULL;
EditorFileDialog::GetIconFunc EditorFileDialog::get_large_icon_func=NULL;

EditorFileDialog::RegisterFunc EditorFileDialog::register_func=NULL;
EditorFileDialog::RegisterFunc EditorFileDialog::unregister_func=NULL;


VBoxContainer *EditorFileDialog::get_vbox() {
	return vbox;

}

void EditorFileDialog::_notification(int p_what) {
	if (p_what==NOTIFICATION_PROCESS) {

		if (preview_waiting) {
			preview_wheel_timeout-=get_process_delta_time();
			if (preview_wheel_timeout<=0) {
				preview_wheel_index++;
				if (preview_wheel_index>=8)
					preview_wheel_index=0;
				Ref<Texture> frame = get_icon("WaitPreview"+itos(preview_wheel_index+1),"EditorIcons");
				preview->set_texture(frame);
				preview_wheel_timeout=0.1;
			}
		}
	}

	if (p_what==NOTIFICATION_DRAW) {

		//RID ci = get_canvas_item();
		//get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	}
}

void EditorFileDialog::set_enable_multiple_selection(bool p_enable) {

	tree->set_select_mode(p_enable?Tree::SELECT_MULTI : Tree::SELECT_SINGLE);
};

Vector<String> EditorFileDialog::get_selected_files() const {

	Vector<String> list;

	TreeItem* item = tree->get_root();
	while ( (item = tree->get_next_selected(item)) ) {

		list.push_back(dir_access->get_current_dir().plus_file(item->get_text(0)));
	};

	return list;
};

void EditorFileDialog::update_dir() {

	dir->set_text(dir_access->get_current_dir());
}

void EditorFileDialog::_dir_entered(String p_dir) {


	dir_access->change_dir(p_dir);
	file->set_text("");
	invalidate();
	update_dir();
}

void EditorFileDialog::_file_entered(const String& p_file) {

	_action_pressed();
}

void EditorFileDialog::_save_confirm_pressed() {
	String f=dir_access->get_current_dir().plus_file(file->get_text());
	emit_signal("file_selected",f);
	hide();
}

void EditorFileDialog::_post_popup() {

	ConfirmationDialog::_post_popup();
	if (invalidated) {
		update_file_list();
		invalidated=false;
	}
	if (mode==MODE_SAVE_FILE)
		file->grab_focus();
	else
		tree->grab_focus();

	if (is_visible() && get_current_file()!="")
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));

}

void EditorFileDialog::_thumbnail_done(const String& p_path,const Ref<Texture>& p_preview, const Variant& p_udata) {

	set_process(false);
	preview_waiting=false;

	if (p_preview.is_valid() && get_current_path()==p_path) {

		preview->set_texture(p_preview);
		preview_vb->show();

	} else {
		preview_vb->hide();
		preview->set_texture(Ref<Texture>());

	}

}

void EditorFileDialog::_request_single_thumbnail(const String& p_path) {

	EditorResourcePreview::get_singleton()->queue_resource_preview(p_path,this,"_thumbnail_done",p_path);
	print_line("want file "+p_path);
	set_process(true);
	preview_waiting=true;
	preview_wheel_timeout=0;

}

void EditorFileDialog::_action_pressed() {

	if (mode==MODE_OPEN_FILES) {

		TreeItem *ti=tree->get_next_selected(NULL);
		String fbase=dir_access->get_current_dir();

		DVector<String> files;
		while(ti) {

			files.push_back( fbase.plus_file(ti->get_text(0)) );
			ti=tree->get_next_selected(ti);
		}

		if (files.size()) {
			emit_signal("files_selected",files);
			hide();
		}

		return;
	}

	String f=dir_access->get_current_dir().plus_file(file->get_text());

	if (mode==MODE_OPEN_FILE && dir_access->file_exists(f)) {
		emit_signal("file_selected",f);
		hide();
	}

	if (mode==MODE_OPEN_DIR) {


		String path=dir_access->get_current_dir();
		/*if (tree->get_selected()) {
			Dictionary d = tree->get_selected()->get_metadata(0);
			if (d["dir"]) {
				path=path+"/"+String(d["name"]);
			}
		}*/
		path=path.replace("\\","/");
		emit_signal("dir_selected",path);
		hide();
	}

	if (mode==MODE_SAVE_FILE) {

		bool valid=false;

		if (filter->get_selected()==filter->get_item_count()-1) {
			valid=true; //match none
		} else if (filters.size()>1 && filter->get_selected()==0) {
			// match all filters
			for (int i=0;i<filters.size();i++) {

				String flt=filters[i].get_slice(";",0);
				for (int j=0;j<flt.get_slice_count(",");j++) {

					String str = flt.get_slice(",",j).strip_edges();
					if (f.match(str)) {
						valid=true;
						break;
					}
				}
				if (valid)
					break;
			}
		} else {
			int idx=filter->get_selected();
			if (filters.size()>1)
				idx--;
			if (idx>=0 && idx<filters.size()) {

				String flt=filters[idx].get_slice(";",0);
				int filterSliceCount=flt.get_slice_count(",");
				for (int j=0;j<filterSliceCount;j++) {

					String str = (flt.get_slice(",",j).strip_edges());
					if (f.match(str)) {
						valid=true;
						break;
					}
				}

				if (!valid && filterSliceCount>0) {
					String str = (flt.get_slice(",",0).strip_edges());
					f+=str.substr(1, str.length()-1);
					_request_single_thumbnail(get_current_dir().plus_file(f.get_file()));
					file->set_text(f.get_file());
					valid=true;
				}
			} else {
				valid=true;
			}
		}


		if (!valid) {

			exterr->popup_centered_minsize(Size2(250,80));
			return;

		}

		if (dir_access->file_exists(f)) {
			confirm_save->set_text("File Exists, Overwrite?");
			confirm_save->popup_centered(Size2(200,80));
		} else {


			emit_signal("file_selected",f);
			hide();
		}
	}
}

void EditorFileDialog::_cancel_pressed() {

	file->set_text("");
	invalidate();
	hide();
}

void EditorFileDialog::_tree_selected() {

	TreeItem *ti=tree->get_selected();
	if (!ti)
		return;
	Dictionary d=ti->get_metadata(0);

	if (!d["dir"]) {

		file->set_text(d["name"]);
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
	}
}

void EditorFileDialog::_tree_dc_selected() {


	TreeItem *ti=tree->get_selected();
	if (!ti)
		return;

	Dictionary d=ti->get_metadata(0);

	if (d["dir"]) {

		dir_access->change_dir(d["name"]);
		if (mode==MODE_OPEN_FILE || mode==MODE_OPEN_FILES || mode==MODE_OPEN_DIR)
			file->set_text("");
		call_deferred("_update_file_list");
		call_deferred("_update_dir");
	} else {

		_action_pressed();
	}
}

void EditorFileDialog::update_file_list() {

	tree->clear();
	dir_access->list_dir_begin();

	TreeItem *root = tree->create_item();
	Ref<Texture> folder = get_icon("folder","FileDialog");
	List<String> files;
	List<String> dirs;

	bool isdir;
	bool ishidden;
	bool show_hidden = show_hidden_files;
	String item;

	while ((item=dir_access->get_next(&isdir))!="") {

		ishidden = dir_access->current_is_hidden();

		if (show_hidden || !ishidden) {
			if (!isdir)
				files.push_back(item);
			else
				dirs.push_back(item);
		}
	}

	dirs.sort_custom<NoCaseComparator>();
	files.sort_custom<NoCaseComparator>();

	while(!dirs.empty()) {

		if (dirs.front()->get()!=".") {
			TreeItem *ti=tree->create_item(root);
			ti->set_text(0,dirs.front()->get()+"/");
			ti->set_icon(0,folder);
			Dictionary d;
			d["name"]=dirs.front()->get();
			d["dir"]=true;
			ti->set_metadata(0,d);
		}
		dirs.pop_front();

	}

	dirs.clear();

	List<String> patterns;
	// build filter
	if (filter->get_selected()==filter->get_item_count()-1) {

		// match all
	} else if (filters.size()>1 && filter->get_selected()==0) {
		// match all filters
		for (int i=0;i<filters.size();i++) {

			String f=filters[i].get_slice(";",0);
			for (int j=0;j<f.get_slice_count(",");j++) {

				patterns.push_back(f.get_slice(",",j).strip_edges());
			}
		}
	} else {
		int idx=filter->get_selected();
		if (filters.size()>1)
			idx--;

		if (idx>=0 && idx<filters.size()) {

			String f=filters[idx].get_slice(";",0);
			for (int j=0;j<f.get_slice_count(",");j++) {

				patterns.push_back(f.get_slice(",",j).strip_edges());
			}
		}
	}


	String base_dir = dir_access->get_current_dir();


	while(!files.empty()) {

		bool match=patterns.empty();

		for(List<String>::Element *E=patterns.front();E;E=E->next()) {

			if (files.front()->get().matchn(E->get())) {

				match=true;
				break;
			}
		}

		if (match) {
			TreeItem *ti=tree->create_item(root);
			ti->set_text(0,files.front()->get());

			if (get_icon_func) {

				Ref<Texture> icon = get_icon_func(base_dir.plus_file(files.front()->get()));
				ti->set_icon(0,icon);
			}

			if (mode==MODE_OPEN_DIR) {
				ti->set_custom_color(0,get_color("files_disabled"));
				ti->set_selectable(0,false);
			}
			Dictionary d;
			d["name"]=files.front()->get();
			d["dir"]=false;
			ti->set_metadata(0,d);

			if (file->get_text()==files.front()->get())
				ti->select(0);
		}

		files.pop_front();
	}

	if (tree->get_root() && tree->get_root()->get_children())
		tree->get_root()->get_children()->select(0);

	files.clear();

}

void EditorFileDialog::_filter_selected(int) {

	update_file_list();
}

void EditorFileDialog::update_filters() {

	filter->clear();

	if (filters.size()>1) {
		String all_filters;

		const int max_filters=5;

		for(int i=0;i<MIN( max_filters, filters.size()) ;i++) {
			String flt=filters[i].get_slice(";",0);
			if (i>0)
				all_filters+=",";
			all_filters+=flt;
		}

		if (max_filters<filters.size())
			all_filters+=", ...";

		filter->add_item("All Recognized ( "+all_filters+" )");
	}
	for(int i=0;i<filters.size();i++) {

		String flt=filters[i].get_slice(";",0).strip_edges();
		String desc=filters[i].get_slice(";",1).strip_edges();
		if (desc.length())
			filter->add_item(desc+" ( "+flt+" )");
		else
			filter->add_item("( "+flt+" )");
	}

	filter->add_item("All Files (*)");

}

void EditorFileDialog::clear_filters() {

	filters.clear();
	update_filters();
	invalidate();
}
void EditorFileDialog::add_filter(const String& p_filter) {

	filters.push_back(p_filter);
	update_filters();
	invalidate();

}

String EditorFileDialog::get_current_dir() const {

	return dir->get_text();
}
String EditorFileDialog::get_current_file() const {

	return file->get_text();
}
String EditorFileDialog::get_current_path() const {

	return dir->get_text().plus_file(file->get_text());
}
void EditorFileDialog::set_current_dir(const String& p_dir) {

	dir_access->change_dir(p_dir);
	update_dir();
	invalidate();

}
void EditorFileDialog::set_current_file(const String& p_file) {

	file->set_text(p_file);
	update_dir();
	invalidate();
	int lp = p_file.find_last(".");
	if (lp!=-1) {
		file->select(0,lp);
		file->grab_focus();
	}

	if (is_visible())
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));


}
void EditorFileDialog::set_current_path(const String& p_path) {

	if (!p_path.size())
		return;
	int pos=MAX( p_path.find_last("/"), p_path.find_last("\\") );
	if (pos==-1) {

		set_current_file(p_path);
	} else {

		String dir=p_path.substr(0,pos);
		String file=p_path.substr(pos+1,p_path.length());
		set_current_dir(dir);
		set_current_file(file);
	}
}


void EditorFileDialog::set_mode(Mode p_mode) {

	mode=p_mode;
	switch(mode) {

		case MODE_OPEN_FILE: get_ok()->set_text("Open"); set_title("Open a File"); makedir->hide(); break;
		case MODE_OPEN_FILES: get_ok()->set_text("Open"); set_title("Open File(s)"); makedir->hide(); break;
		case MODE_SAVE_FILE: get_ok()->set_text("Save"); set_title("Save a File"); makedir->show(); break;
		case MODE_OPEN_DIR: get_ok()->set_text("Open"); set_title("Open a Directory"); makedir->show(); break;
	}

	if (mode==MODE_OPEN_FILES) {
		tree->set_select_mode(Tree::SELECT_MULTI);
	} else {
		tree->set_select_mode(Tree::SELECT_SINGLE);

	}
}

EditorFileDialog::Mode EditorFileDialog::get_mode() const {

	return mode;
}

void EditorFileDialog::set_access(Access p_access) {

	ERR_FAIL_INDEX(p_access,3);
	if (access==p_access)
		return;
	memdelete( dir_access );
	switch(p_access) {
		case ACCESS_FILESYSTEM: {

			dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		} break;
		case ACCESS_RESOURCES: {

			dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		} break;
		case ACCESS_USERDATA: {

			dir_access = DirAccess::create(DirAccess::ACCESS_USERDATA);
		} break;
	}
	access=p_access;
	_update_drives();
	invalidate();
	update_filters();
	update_dir();
}

void EditorFileDialog::invalidate() {

	if (is_visible()) {
		update_file_list();
		invalidated=false;
	} else {
		invalidated=true;
	}

}

EditorFileDialog::Access EditorFileDialog::get_access() const{

	return access;
}

void EditorFileDialog::_make_dir_confirm() {


	Error err = dir_access->make_dir( makedirname->get_text() );
	if (err==OK) {
		dir_access->change_dir(makedirname->get_text());
		invalidate();
		update_filters();
		update_dir();
	} else {
		mkdirerr->popup_centered_minsize(Size2(250,50));
	}
}


void EditorFileDialog::_make_dir() {

	makedialog->popup_centered_minsize(Size2(250,80));
	makedirname->grab_focus();

}

void EditorFileDialog::_select_drive(int p_idx) {

	String d = drives->get_item_text(p_idx);
	dir_access->change_dir(d);
	file->set_text("");
	invalidate();
	update_dir();

}

void EditorFileDialog::_update_drives() {


	int dc = dir_access->get_drive_count();
	if (dc==0 || access!=ACCESS_FILESYSTEM) {
		drives->hide();
	} else {
		drives->clear();
		drives->show();

		for(int i=0;i<dir_access->get_drive_count();i++) {
			String d = dir_access->get_drive(i);
			drives->add_item(dir_access->get_drive(i));
		}

		drives->select(dir_access->get_current_drive());

	}
}

bool EditorFileDialog::default_show_hidden_files=true;


void EditorFileDialog::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_tree_selected"),&EditorFileDialog::_tree_selected);
	ObjectTypeDB::bind_method(_MD("_tree_db_selected"),&EditorFileDialog::_tree_dc_selected);
	ObjectTypeDB::bind_method(_MD("_dir_entered"),&EditorFileDialog::_dir_entered);
	ObjectTypeDB::bind_method(_MD("_file_entered"),&EditorFileDialog::_file_entered);
	ObjectTypeDB::bind_method(_MD("_action_pressed"),&EditorFileDialog::_action_pressed);
	ObjectTypeDB::bind_method(_MD("_cancel_pressed"),&EditorFileDialog::_cancel_pressed);
	ObjectTypeDB::bind_method(_MD("_filter_selected"),&EditorFileDialog::_filter_selected);
	ObjectTypeDB::bind_method(_MD("_save_confirm_pressed"),&EditorFileDialog::_save_confirm_pressed);

	ObjectTypeDB::bind_method(_MD("clear_filters"),&EditorFileDialog::clear_filters);
	ObjectTypeDB::bind_method(_MD("add_filter","filter"),&EditorFileDialog::add_filter);
	ObjectTypeDB::bind_method(_MD("get_current_dir"),&EditorFileDialog::get_current_dir);
	ObjectTypeDB::bind_method(_MD("get_current_file"),&EditorFileDialog::get_current_file);
	ObjectTypeDB::bind_method(_MD("get_current_path"),&EditorFileDialog::get_current_path);
	ObjectTypeDB::bind_method(_MD("set_current_dir","dir"),&EditorFileDialog::set_current_dir);
	ObjectTypeDB::bind_method(_MD("set_current_file","file"),&EditorFileDialog::set_current_file);
	ObjectTypeDB::bind_method(_MD("set_current_path","path"),&EditorFileDialog::set_current_path);
	ObjectTypeDB::bind_method(_MD("set_mode","mode"),&EditorFileDialog::set_mode);
	ObjectTypeDB::bind_method(_MD("get_mode"),&EditorFileDialog::get_mode);
	ObjectTypeDB::bind_method(_MD("get_vbox:VBoxContainer"),&EditorFileDialog::get_vbox);
	ObjectTypeDB::bind_method(_MD("set_access","access"),&EditorFileDialog::set_access);
	ObjectTypeDB::bind_method(_MD("get_access"),&EditorFileDialog::get_access);
	ObjectTypeDB::bind_method(_MD("set_show_hidden_files"),&EditorFileDialog::set_show_hidden_files);
	ObjectTypeDB::bind_method(_MD("is_showing_hidden_files"),&EditorFileDialog::is_showing_hidden_files);
	ObjectTypeDB::bind_method(_MD("_select_drive"),&EditorFileDialog::_select_drive);
	ObjectTypeDB::bind_method(_MD("_make_dir"),&EditorFileDialog::_make_dir);
	ObjectTypeDB::bind_method(_MD("_make_dir_confirm"),&EditorFileDialog::_make_dir_confirm);
	ObjectTypeDB::bind_method(_MD("_update_file_list"),&EditorFileDialog::update_file_list);
	ObjectTypeDB::bind_method(_MD("_update_dir"),&EditorFileDialog::update_dir);
	ObjectTypeDB::bind_method(_MD("_thumbnail_done"),&EditorFileDialog::_thumbnail_done);

	ObjectTypeDB::bind_method(_MD("invalidate"),&EditorFileDialog::invalidate);

	ADD_SIGNAL(MethodInfo("file_selected",PropertyInfo( Variant::STRING,"path")));
	ADD_SIGNAL(MethodInfo("files_selected",PropertyInfo( Variant::STRING_ARRAY,"paths")));
	ADD_SIGNAL(MethodInfo("dir_selected",PropertyInfo( Variant::STRING,"dir")));

	BIND_CONSTANT( MODE_OPEN_FILE );
	BIND_CONSTANT( MODE_OPEN_FILES );
	BIND_CONSTANT( MODE_OPEN_DIR );
	BIND_CONSTANT( MODE_SAVE_FILE );

	BIND_CONSTANT( ACCESS_RESOURCES );
	BIND_CONSTANT( ACCESS_USERDATA );
	BIND_CONSTANT( ACCESS_FILESYSTEM );

}


void EditorFileDialog::set_show_hidden_files(bool p_show) {
	show_hidden_files=p_show;
	invalidate();
}

bool EditorFileDialog::is_showing_hidden_files() const {
	return show_hidden_files;
}

void EditorFileDialog::set_default_show_hidden_files(bool p_show) {
	default_show_hidden_files=p_show;
}

EditorFileDialog::EditorFileDialog() {

	show_hidden_files=true;

	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);

	mode=MODE_SAVE_FILE;
	set_title("Save a File");

	dir = memnew(LineEdit);
	HBoxContainer *pathhb = memnew( HBoxContainer );
	pathhb->add_child(dir);
	dir->set_h_size_flags(SIZE_EXPAND_FILL);

	drives = memnew( OptionButton );
	pathhb->add_child(drives);
	drives->connect("item_selected",this,"_select_drive");

	makedir = memnew( Button );
	makedir->set_text("Create Folder");
	makedir->connect("pressed",this,"_make_dir");
	pathhb->add_child(makedir);

	vbc->add_margin_child("Path:",pathhb);

	list_hb = memnew( HBoxContainer );
	vbc->add_margin_child("Directories & Files:",list_hb,true);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_h_size_flags(SIZE_EXPAND_FILL);
	list_hb->add_child(tree);

	HBoxContainer* filter_hb = memnew( HBoxContainer );
	vbc->add_child(filter_hb);

	VBoxContainer *filter_vb = memnew( VBoxContainer );
	filter_hb->add_child(filter_vb);
	filter_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	preview_vb = memnew( VBoxContainer );
	filter_hb->add_child(preview_vb);
	CenterContainer *prev_cc = memnew( CenterContainer );
	preview_vb->add_margin_child("Preview:",prev_cc);
	preview = memnew( TextureFrame );
	prev_cc->add_child(preview);
	preview_vb->hide();


	file = memnew(LineEdit);
	//add_child(file);
	filter_vb->add_margin_child("File:",file);


	filter = memnew( OptionButton );
	//add_child(filter);
	filter_vb->add_margin_child("Filter:",filter);
	filter->set_clip_text(true);//too many extensions overflow it

	dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	access=ACCESS_RESOURCES;
	_update_drives();


	connect("confirmed", this,"_action_pressed");
	//cancel->connect("pressed", this,"_cancel_pressed");
	tree->connect("cell_selected", this,"_tree_selected",varray(),CONNECT_DEFERRED);
	tree->connect("item_activated", this,"_tree_db_selected",varray());
	dir->connect("text_entered", this,"_dir_entered");
	file->connect("text_entered", this,"_file_entered");
	filter->connect("item_selected", this,"_filter_selected");


	confirm_save = memnew( ConfirmationDialog );
	confirm_save->set_as_toplevel(true);
	add_child(confirm_save);


	confirm_save->connect("confirmed", this,"_save_confirm_pressed");

	makedialog = memnew( ConfirmationDialog );
	makedialog->set_title("Create Folder");
	VBoxContainer *makevb= memnew( VBoxContainer );
	makedialog->add_child(makevb);
	makedialog->set_child_rect(makevb);
	makedirname = memnew( LineEdit );
	makevb->add_margin_child("Name:",makedirname);
	add_child(makedialog);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed",this,"_make_dir_confirm");
	mkdirerr = memnew( AcceptDialog );
	mkdirerr->set_text("Could not create folder.");
	add_child(mkdirerr);

	exterr = memnew( AcceptDialog );
	exterr->set_text("Must use a valid extension.");
	add_child(exterr);


	//update_file_list();
	update_filters();
	update_dir();

	set_hide_on_ok(false);
	vbox=vbc;


	invalidated=true;
	if (register_func)
		register_func(this);

	preview_wheel_timeout=0;
	preview_wheel_index=0;
	preview_waiting=false;

}


EditorFileDialog::~EditorFileDialog() {

	if (unregister_func)
		unregister_func(this);
	memdelete(dir_access);
}


void EditorLineEditFileChooser::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_browse"),&EditorLineEditFileChooser::_browse);
	ObjectTypeDB::bind_method(_MD("_chosen"),&EditorLineEditFileChooser::_chosen);
	ObjectTypeDB::bind_method(_MD("get_button:Button"),&EditorLineEditFileChooser::get_button);
	ObjectTypeDB::bind_method(_MD("get_line_edit:LineEdit"),&EditorLineEditFileChooser::get_line_edit);
	ObjectTypeDB::bind_method(_MD("get_file_dialog:EditorFileDialog"),&EditorLineEditFileChooser::get_file_dialog);

}

void EditorLineEditFileChooser::_chosen(const String& p_text){

	line_edit->set_text(p_text);
	line_edit->emit_signal("text_entered",p_text);
}

void EditorLineEditFileChooser::_browse() {

	dialog->popup_centered_ratio();
}

EditorLineEditFileChooser::EditorLineEditFileChooser() {

	line_edit = memnew( LineEdit );
	add_child(line_edit);
	line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	button = memnew( Button );
	button->set_text(" .. ");
	add_child(button);
	button->connect("pressed",this,"_browse");
	dialog = memnew( EditorFileDialog);
	add_child(dialog);
	dialog->connect("file_selected",this,"_chosen");
	dialog->connect("dir_selected",this,"_chosen");
	dialog->connect("files_selected",this,"_chosen");

}
