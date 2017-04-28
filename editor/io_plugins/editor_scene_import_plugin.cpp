/*************************************************************************/
/*  editor_scene_import_plugin.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_scene_import_plugin.h"
#if 0
#include "editor/create_dialog.h"
#include "editor/editor_node.h"
#include "global_config.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "os/os.h"
#include "scene/3d/body_shape.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/navigation.h"
#include "scene/3d/path.h"
#include "scene/3d/physics_body.h"
#include "scene/3d/portal.h"
#include "scene/3d/room_instance.h"
#include "scene/3d/vehicle_body.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/sphere_shape.h"
#include <scene/resources/box_shape.h>
#include <scene/resources/plane_shape.h>
#include <scene/resources/ray_shape.h>




EditorSceneImporter::EditorSceneImporter() {


}

void EditorScenePostImport::_bind_methods() {

	BIND_VMETHOD( MethodInfo("post_import",PropertyInfo(Variant::OBJECT,"scene")) );

}

Node *EditorScenePostImport::post_import(Node* p_scene) {

	if (get_script_instance())
		return get_script_instance()->call("post_import",p_scene);

	return p_scene;
}

EditorScenePostImport::EditorScenePostImport() {


}


/////////////////////////////


class EditorImportAnimationOptions : public VBoxContainer {

	GDCLASS( EditorImportAnimationOptions, VBoxContainer );



	TreeItem *fps;
	TreeItem* optimize_linear_error;
	TreeItem* optimize_angular_error;
	TreeItem* optimize_max_angle;

	TreeItem *clips_base;

	TextEdit *filters;
	Vector<TreeItem*> clips;

	Tree *flags;
	Tree *clips_tree;
	Tree *optimization_tree;
	Vector<TreeItem*> items;


	bool updating;
	bool validating;



	void _changed();
	void _item_edited();
	void _button_action(Object *p_obj,int p_col,int p_id);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:

	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;

	void set_fps(int p_fps);
	int get_fps() const;

	void set_optimize_linear_error(float p_error);
	float get_optimize_linear_error() const;

	void set_optimize_angular_error(float p_error);
	float get_optimize_angular_error() const;

	void set_optimize_max_angle(float p_error);
	float get_optimize_max_angle() const;

	void setup_clips(const Array& p_clips);
	Array get_clips() const;

	void set_filter(const String& p_filter);
	String get_filter() const;

	EditorImportAnimationOptions();


};

////////////////////////////

class EditorSceneImportDialog : public ConfirmationDialog  {

	GDCLASS(EditorSceneImportDialog,ConfirmationDialog);


	struct FlagInfo {
		int value;
		const char *category;
		const char *text;
		bool defval;
	};

	static const FlagInfo scene_flag_names[];

	EditorImportTextureOptions *texture_options;
	EditorImportAnimationOptions *animation_options;

	EditorSceneImportPlugin *plugin;

	EditorNode *editor;

	LineEdit *import_path;
	LineEdit *save_path;
	LineEdit *script_path;
	Tree *import_options;
	EditorFileDialog *file_select;
	EditorFileDialog *script_select;
	EditorDirDialog *save_select;
	OptionButton *texture_action;
	CreateDialog *root_type_choose;
	LineEdit *root_node_name;

	ConfirmationDialog *confirm_open;

	ConfirmationDialog *confirm_import;
	RichTextLabel *missing_files;

	Vector<TreeItem*> scene_flags;

	Map<Ref<Mesh>,Ref<Shape> > collision_map;
	ConfirmationDialog *error_dialog;

	Button *root_type;
	CheckBox *root_default;


	void _root_default_pressed();
	void _root_type_pressed();
	void _set_root_type();

	void _choose_file(const String& p_path);
	void _choose_save_file(const String& p_path);
	void _choose_script(const String& p_path);
	void _browse();
	void _browse_target();
	void _browse_script();
	void _import(bool p_and_open=false);
	void _import_confirm();

	Ref<ResourceImportMetadata> wip_rimd;
	Node *wip_import;
	String wip_save_file;
	bool wip_blocked;
	bool wip_open;

	void _dialog_hid();
	void _open_and_import();


protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void setup_popup(const String& p_from,const String& p_to_path) {
		_choose_file(p_from);
		_choose_save_file(p_to_path);
	}

	Error import(const String& p_from, const String& p_to, const String& p_preset);
	void popup_import(const String& p_from);
	EditorSceneImportDialog(EditorNode *p_editor,EditorSceneImportPlugin *p_plugin);
};

///////////////////////////////////


static const char *anim_flag_names[]={
	"Detect Loop (-loop,-cycle)",
	"Keep Value Tracks",
	"Optimize",
	"Force All Tracks in All Clips",
	NULL
};

static const char *anim_flag_descript[]={
	"Set loop flag for animation names that\ncontain 'cycle' or 'loop' in the name.",
	"When merging an existing aimation,\nkeep the user-created value-tracks.",
	"Remove redundant keyframes in\n transform tacks.",
	"Some exporters will rely on default pose for some bones.\nThis forces those bones to have at least one animation key.",
	NULL
};



void EditorImportAnimationOptions::set_flags(uint32_t p_flags){

	updating=true;
	for(int i=0;i<items.size();i++) {

		items[i]->set_checked(0,p_flags&(1<<i));
	}
	updating=false;

}
uint32_t EditorImportAnimationOptions::get_flags() const{

	uint32_t f=0;
	for(int i=0;i<items.size();i++) {

		if (items[i]->is_checked(0))
			f|=(1<<i);
	}

	return f;
}


void EditorImportAnimationOptions::_changed() {

	if (updating)
		return;
	emit_signal("changed");
}


void EditorImportAnimationOptions::_button_action(Object *p_obj,int p_col,int p_id) {

	memdelete(p_obj);

}


void EditorImportAnimationOptions::_item_edited() {

	if (validating)
		return;

	if (clips.size()==0)
		return;
	validating=true;
	print_line("edited");
	TreeItem *item = clips_tree->get_edited();
	if (item==clips[clips.size()-1]) {
		//add new
		print_line("islast");
		if (item->get_text(0).find("<")!=-1 || item->get_text(0).find(">")!=-1) {
			validating=false;
			return; //fuckit
		}

		item->set_editable(1,true);
		item->set_editable(2,true);
		item->add_button(0,EditorNode::get_singleton()->get_gui_base()->get_icon("Del","EditorIcons"));
		item->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
		item->set_range_config(1,0,3600,0.01);
		item->set_range(1,0);
		item->set_editable(1,true);
		item->set_cell_mode(2,TreeItem::CELL_MODE_RANGE);
		item->set_range_config(2,0,3600,0.01);
		item->set_range(2,0);
		item->set_cell_mode(3,TreeItem::CELL_MODE_CHECK);
		item->set_editable(3,true);

		TreeItem *newclip = clips_tree->create_item(clips_base);
		newclip->set_text(0,"<new clip>");
		newclip->set_editable(0,true);
		newclip->set_editable(1,false);
		newclip->set_editable(2,false);
		clips.push_back(newclip);



	}


	//make name unique JUST IN CASE
	String name = item->get_text(0);
	name=name.replace("/","_").replace(":","_").strip_edges();
	if (name=="")
		name=TTR("New Clip");

	if (clips.size()>2) {
		int index=1;
		while(true) {
			bool valid = true;
			String try_name=name;
			if (index>1)
				try_name+=" "+itos(index);

			for(int i=0;i<clips.size()-1;i++) {

				if (clips[i]==item)
					continue;
				if (clips[i]->get_text(0)==try_name) {
					index++;
					valid=false;
					break;
				}
			}

			if (valid) {
				name=try_name;
				break;
			}

		}
	}

	if (item->get_text(0)!=name)
		item->set_text(0,name);

	validating=false;

}

void EditorImportAnimationOptions::_bind_methods() {

	ClassDB::bind_method("_changed",&EditorImportAnimationOptions::_changed);
	ClassDB::bind_method("_item_edited",&EditorImportAnimationOptions::_item_edited);
	ClassDB::bind_method("_button_action",&EditorImportAnimationOptions::_button_action);
	//ClassDB::bind_method("_changedp",&EditorImportAnimationOptions::_changedp);

	ADD_SIGNAL(MethodInfo("changed"));
}


void EditorImportAnimationOptions::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		flags->connect("item_edited",this,"_changed");
		clips_tree->connect("item_edited",this,"_item_edited");
		clips_tree->connect("button_pressed",this,"_button_action",varray(),CONNECT_DEFERRED);
		//format->connect("item_selected",this,"_changedp");
	}
}


Array EditorImportAnimationOptions::get_clips()  const {

	Array arr;
	for(int i=0;i<clips.size()-1;i++) {

		arr.push_back(clips[i]->get_text(0));
		arr.push_back(clips[i]->get_range(1));
		arr.push_back(clips[i]->get_range(2));
		arr.push_back(clips[i]->is_checked(3));
	}

	return arr;
}


void EditorImportAnimationOptions::setup_clips(const Array& p_clips) {

	ERR_FAIL_COND(p_clips.size()%4!=0);
	for(int i=0;i<clips.size();i++) {

		memdelete(clips[i]);
	}


	clips.clear();

	for(int i=0;i<p_clips.size();i+=4) {

		TreeItem *clip = clips_tree->create_item(clips_base);
		clip->set_text(0,p_clips[i]);
		clip->add_button(0,EditorNode::get_singleton()->get_gui_base()->get_icon("Del","EditorIcons"));
		clip->set_editable(0,true);
		clip->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
		clip->set_range_config(1,0,3600,0.01);
		clip->set_range(1,p_clips[i+1]);
		clip->set_editable(1,true);
		clip->set_cell_mode(2,TreeItem::CELL_MODE_RANGE);
		clip->set_range_config(2,0,3600,0.01);
		clip->set_range(2,p_clips[i+2]);
		clip->set_editable(2,true);
		clip->set_cell_mode(3,TreeItem::CELL_MODE_CHECK);
		clip->set_editable(3,true);
		clip->set_checked(3,p_clips[i+3]);
		clips.push_back(clip);

	}

	TreeItem *newclip = clips_tree->create_item(clips_base);
	newclip->set_text(0,"<new clip>");
	newclip->set_editable(0,true);
	newclip->set_editable(1,false);
	newclip->set_editable(2,false);
	newclip->set_editable(3,false);
	clips.push_back(newclip);

}


EditorImportAnimationOptions::EditorImportAnimationOptions() {


	updating=false;
	validating=false;

	TabContainer *tab= memnew(TabContainer);
	add_margin_child(TTR("Animation Options"),tab,true);

	flags = memnew( Tree );
	flags->set_hide_root(true);
	tab->add_child(flags);
	flags->set_name(TTR("Flags"));
	TreeItem *root = flags->create_item();

	const char ** fname=anim_flag_names;
	const char ** fdescr=anim_flag_descript;

	while( *fname ) {

		TreeItem*ti = flags->create_item(root);
		ti->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		ti->set_text(0,*fname);
		ti->set_editable(0,true);
		ti->set_tooltip(0,*fdescr);
		items.push_back(ti);
		fname++;
		fdescr++;
	}


	TreeItem *fps_base = flags->create_item(root);
	fps_base->set_text(0,TTR("Bake FPS:"));
	fps_base->set_editable(0,false);
	fps = flags->create_item(fps_base);
	fps->set_cell_mode(0,TreeItem::CELL_MODE_RANGE);
	fps->set_editable(0,true);
	fps->set_range_config(0,1,120,1);
	fps->set_range(0,15);

	optimization_tree = memnew( Tree );
	optimization_tree->set_columns(2);
	tab->add_child(optimization_tree);
	optimization_tree->set_name(TTR("Optimizer"));
	optimization_tree->set_column_expand(0,true);
	optimization_tree->set_column_expand(1,false);
	optimization_tree->set_column_min_width(1,80);
	optimization_tree->set_hide_root(true);


	TreeItem *optimize_root = optimization_tree->create_item();

	optimize_linear_error = optimization_tree->create_item(optimize_root);
	optimize_linear_error->set_text(0,TTR("Max Linear Error"));
	optimize_linear_error->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
	optimize_linear_error->set_editable(1,true);
	optimize_linear_error->set_range_config(1,0,1,0.001);
	optimize_linear_error->set_range(1,0.05);

	optimize_angular_error = optimization_tree->create_item(optimize_root);
	optimize_angular_error->set_text(0,TTR("Max Angular Error"));
	optimize_angular_error->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
	optimize_angular_error->set_editable(1,true);
	optimize_angular_error->set_range_config(1,0,1,0.001);
	optimize_angular_error->set_range(1,0.01);

	optimize_max_angle = optimization_tree->create_item(optimize_root);
	optimize_max_angle->set_text(0,TTR("Max Angle"));
	optimize_max_angle->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
	optimize_max_angle->set_editable(1,true);
	optimize_max_angle->set_range_config(1,0,360,0.001);
	optimize_max_angle->set_range(1,int(180*0.125));

	clips_tree = memnew( Tree );
	clips_tree->set_hide_root(true);
	tab->add_child(clips_tree);
	clips_tree->set_name(TTR("Clips"));

	clips_tree->set_columns(4);
	clips_tree->set_column_expand(0,1);
	clips_tree->set_column_expand(1,0);
	clips_tree->set_column_expand(2,0);
	clips_tree->set_column_expand(3,0);
	clips_tree->set_column_min_width(1,60);
	clips_tree->set_column_min_width(2,60);
	clips_tree->set_column_min_width(3,40);
	clips_tree->set_column_titles_visible(true);
	clips_tree->set_column_title(0,TTR("Name"));
	clips_tree->set_column_title(1,TTR("Start(s)"));
	clips_tree->set_column_title(2,TTR("End(s)"));
	clips_tree->set_column_title(3,TTR("Loop"));
	clips_base =clips_tree->create_item(0);


	setup_clips(Array());


	filters = memnew( TextEdit );
	tab->add_child(filters);
	filters->set_name(TTR("Filters"));
}



void EditorImportAnimationOptions::set_fps(int p_fps) {

	fps->set_range(0,p_fps);
}

int EditorImportAnimationOptions::get_fps() const {

	return fps->get_range(0);
}


void EditorImportAnimationOptions::set_optimize_linear_error(float p_optimize_linear_error) {

	optimize_linear_error->set_range(1,p_optimize_linear_error);
}

float EditorImportAnimationOptions::get_optimize_linear_error() const {

	return optimize_linear_error->get_range(1);
}

void EditorImportAnimationOptions::set_optimize_angular_error(float p_optimize_angular_error) {

	optimize_angular_error->set_range(1,p_optimize_angular_error);
}

float EditorImportAnimationOptions::get_optimize_angular_error() const {

	return optimize_angular_error->get_range(1);
}

void EditorImportAnimationOptions::set_optimize_max_angle(float p_optimize_max_angle) {

	optimize_max_angle->set_range(1,p_optimize_max_angle);
}

float EditorImportAnimationOptions::get_optimize_max_angle() const {

	return optimize_max_angle->get_range(1);
}


void EditorImportAnimationOptions::set_filter(const String& p_filter) {

	filters->set_text(p_filter);
}

String EditorImportAnimationOptions::get_filter() const {

	return filters->get_text();
}





////////////////////////////////////////////////////////



void EditorSceneImportDialog::_choose_file(const String& p_path) {
#if 0
	StringName sn = EditorImportDB::get_singleton()->find_source_path(p_path);
	if (sn!=StringName()) {

		EditorImportDB::ImportScene isc;
		if (EditorImportDB::get_singleton()->get_scene(sn,isc)==OK) {

			save_path->set_text(String(sn).get_base_dir());
			texture_options->set_flags( isc.image_flags );
			texture_options->set_quality( isc.image_quality );
			texture_options->set_format( isc.image_format );
			animation_options->set_flags( isc.anim_flags );
			script_path->set_text( isc.import_script );
			uint32_t f = isc.flags;
			for(int i=0;i<scene_flags.size();i++) {

				scene_flags[i]->set_checked(0,f&(1<<i));
			}
		}
	} else {
#endif
		save_path->set_text("");
		root_node_name->set_text("");
		//save_path->set_text(p_path.get_file().basename()+".scn");
#if 0
	}
#endif

	if (p_path!=String()) {

		String from_path = EditorFileSystem::get_singleton()->find_resource_from_source(EditorImportPlugin::validate_source_path(p_path));
		print_line("from path.."+from_path);
		if (from_path!=String()) {
			popup_import(from_path);

		}
	}


	import_path->set_text(p_path);
	if (root_node_name->get_text().size()==0){
		root_node_name->set_text(import_path->get_text().get_file().get_basename());
	}

}
void EditorSceneImportDialog::_choose_save_file(const String& p_path) {

	save_path->set_text(p_path);
}

void EditorSceneImportDialog::_choose_script(const String& p_path) {

	String p = GlobalConfig::get_singleton()->localize_path(p_path);
	if (!p.is_resource_file())
		p=GlobalConfig::get_singleton()->get_resource_path().path_to(p_path.get_base_dir())+p_path.get_file();
	script_path->set_text(p);

}


void EditorSceneImportDialog::_open_and_import() {

	bool unsaved=EditorNode::has_unsaved_changes();

	if (unsaved) {

		confirm_open->popup_centered_minsize(Size2(300,80)*EDSCALE);
	} else {
		_import(true);
	}
}

void EditorSceneImportDialog::_import(bool p_and_open) {

	wip_open=p_and_open;
//'	ImportMonitorBlock imb;


	if (import_path->get_text().strip_edges()=="") {
		error_dialog->set_text(TTR("Source path is empty."));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (save_path->get_text().strip_edges()=="") {
		error_dialog->set_text(TTR("Target path is empty."));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (!save_path->get_text().begins_with("res://")) {
		error_dialog->set_text(TTR("Target path must be a complete resource path."));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (!DirAccess::exists(save_path->get_text())) {
		error_dialog->set_text(TTR("Target path must exist."));
		error_dialog->popup_centered_minsize();
		return;
	}

	String dst_path;

	if (texture_action->get_selected()==0)
		dst_path=save_path->get_text();//.get_base_dir();
	else
		dst_path=GlobalConfig::get_singleton()->get("import/shared_textures");

	uint32_t flags=0;

	for(int i=0;i<scene_flags.size();i++) {

		if (scene_flags[i]->is_checked(0)) {
			int md = scene_flags[i]->get_metadata(0);
			flags|=md;
		}
	}





	if (script_path->get_text()!="") {
		Ref<Script> scr = ResourceLoader::load(script_path->get_text());
		if (!scr.is_valid()) {
			error_dialog->set_text(TTR("Couldn't load post-import script."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}

		Ref<EditorScenePostImport> pi = Ref<EditorScenePostImport>( memnew( EditorScenePostImport ) );
		pi->set_script(scr.get_ref_ptr());
		if (!pi->get_script_instance()) {

			error_dialog->set_text(TTR("Invalid/broken script for post-import."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}

	}


	// Scenes should always be imported as binary format since vertex data is large and would take
	// up a lot of space and time to load if imported as text format (GH-5778)
	String save_file = save_path->get_text().plus_file(import_path->get_text().get_file().get_basename()+".scn");
	print_line("Saving to: "+save_file);





	Node *scene=NULL;


	Ref<ResourceImportMetadata> rim = memnew( ResourceImportMetadata );

	rim->add_source(EditorImportPlugin::validate_source_path(import_path->get_text()));
	rim->set_option("flags",flags);
	print_line("GET FLAGS: "+itos(texture_options->get_flags()));
	rim->set_option("texture_flags",texture_options->get_flags());
	rim->set_option("texture_format",texture_options->get_format());
	rim->set_option("texture_quality",texture_options->get_quality());
	rim->set_option("animation_flags",animation_options->get_flags());
	rim->set_option("animation_bake_fps",animation_options->get_fps());
	rim->set_option("animation_optimizer_linear_error",animation_options->get_optimize_linear_error());
	rim->set_option("animation_optimizer_angular_error",animation_options->get_optimize_angular_error());
	rim->set_option("animation_optimizer_max_angle",animation_options->get_optimize_max_angle());
	rim->set_option("animation_filters",animation_options->get_filter());
	rim->set_option("animation_clips",animation_options->get_clips());
	rim->set_option("post_import_script",script_path->get_text());
	rim->set_option("reimport",true);
	if (!root_default->is_pressed()) {
		rim->set_option("root_type",root_type->get_text());
	}
	if (root_node_name->get_text().size()==0) {
		root_node_name->set_text(import_path->get_text().get_file().get_basename());
	}
	rim->set_option("root_name",root_node_name->get_text());

	List<String> missing;
	Error err = plugin->import1(rim,&scene,&missing);

	if (err || !scene) {

		error_dialog->set_text(TTR("Error importing scene."));
		error_dialog->popup_centered(Size2(200,100)*EDSCALE);
		return;
	}

	if (missing.size()) {

		missing_files->clear();
		for(List<String>::Element *E=missing.front();E;E=E->next()) {

			missing_files->add_text(E->get());
			missing_files->add_newline();
		}
		wip_import=scene;
		wip_rimd=rim;
		wip_save_file=save_file;
		confirm_import->popup_centered_ratio();
		return;

	} else {

		err = plugin->import2(scene,save_file,rim);

		if (err) {

			error_dialog->set_text(TTR("Error importing scene."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}
		if (wip_open)
			EditorNode::get_singleton()->load_scene(save_file,false,false,false);

	}

	hide();

	/*
	editor->clear_scene();

	Error err = EditorImport::import_scene(import_path->get_text(),save_file,dst_path,flags,texture_options->get_format(),compression,texture_options->get_flags(),texture_options->get_quality(),animation_options->get_flags(), &scene,pi);

	if (err) {

		error_dialog->set_text("Error importing scene.");
		error_dialog->popup_centered(Size2(200,100));
		return;
	}

	editor->save_import_export();
	if (scene)
		editor->set_edited_scene(scene);

	hide();
	*/
};


void EditorSceneImportDialog::_import_confirm() {

	wip_blocked=true;
	print_line("import confirm!");
	Error err = plugin->import2(wip_import,wip_save_file,wip_rimd);
	wip_blocked=false;
	wip_import=NULL;
	wip_rimd=Ref<ResourceImportMetadata>();
	confirm_import->hide();
	if (err) {

		wip_save_file="";
		error_dialog->set_text(TTR("Error importing scene."));
		error_dialog->popup_centered(Size2(200,100)*EDSCALE);
		return;
	}

	if (wip_open)
		EditorNode::get_singleton()->load_scene(wip_save_file,false,false,false);
	wip_open=false;
	wip_save_file="";

	hide();

}


void EditorSceneImportDialog::_browse() {

	file_select->popup_centered_ratio();
}

void EditorSceneImportDialog::_browse_target() {

	save_select->popup_centered_ratio();
	if (save_path->get_text()!="")
		save_select->set_current_path(save_path->get_text());

}

void EditorSceneImportDialog::_browse_script() {

	script_select->popup_centered_ratio();

}

void EditorSceneImportDialog::popup_import(const String &p_from) {

	popup_centered(Size2(750,550)*EDSCALE);
	if (p_from!="") {
		Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_from);
		if (rimd.is_null())
			return;

		int flags = rimd->get_option("flags");

		for(int i=0;i<scene_flags.size();i++) {

			int md = scene_flags[i]->get_metadata(0);
			scene_flags[i]->set_checked(0,flags&md);
		}

		texture_options->set_flags(rimd->get_option("texture_flags"));
		texture_options->set_format(EditorTextureImportPlugin::ImageFormat(int(rimd->get_option("texture_format"))));
		texture_options->set_quality(rimd->get_option("texture_quality"));
		animation_options->set_flags(rimd->get_option("animation_flags"));
		if (rimd->has_option("animation_clips"))
			animation_options->setup_clips(rimd->get_option("animation_clips"));
		if (rimd->has_option("animation_filters"))
			animation_options->set_filter(rimd->get_option("animation_filters"));
		if (rimd->has_option("animation_bake_fps"))
			animation_options->set_fps(rimd->get_option("animation_bake_fps"));
		if (rimd->has_option("animation_optimizer_linear_error"))
			animation_options->set_optimize_linear_error(rimd->get_option("animation_optimizer_linear_error"));
		if (rimd->has_option("animation_optimizer_angular_error"))
			animation_options->set_optimize_angular_error(rimd->get_option("animation_optimizer_angular_error"));
		if (rimd->has_option("animation_optimizer_max_angle"))
			animation_options->set_optimize_max_angle(rimd->get_option("animation_optimizer_max_angle"));

		if (rimd->has_option("root_type")) {
			root_default->set_pressed(false);
			String type = rimd->get_option("root_type");
			root_type->set_text(type);
			root_type->set_disabled(false);

			if (has_icon(type,"EditorIcons")) {
				root_type->set_icon(get_icon(type,"EditorIcons"));
			} else {
				root_type->set_icon(get_icon("Object","EditorIcons"));
			}

		} else {
			root_default->set_pressed(true);
			root_type->set_disabled(true);
		}
		if (rimd->has_option("root_name")) {
			root_node_name->set_text(rimd->get_option("root_name"));
		} else {
			root_node_name->set_text(root_type->get_text()); // backward compatibility for 2.1 or before
		}
		script_path->set_text(rimd->get_option("post_import_script"));

		save_path->set_text(p_from.get_base_dir());
		import_path->set_text(EditorImportPlugin::expand_source_path(rimd->get_source_path(0)));

	}
}


void EditorSceneImportDialog::_notification(int p_what) {


	if (p_what==NOTIFICATION_ENTER_TREE) {


		List<String> extensions;
		file_select->clear_filters();
		root_type->set_icon(get_icon("Spatial","EditorIcons"));
		root_type->set_text("Spatial");
		root_type->set_disabled(true);

		for(int i=0;i<plugin->get_importers().size();i++) {
			plugin->get_importers()[i]->get_extensions(&extensions);
		}


		for(int i=0;i<extensions.size();i++) {

			file_select->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
		}

		extensions.clear();

		//EditorImport::get_import_extensions(&extensions)
	/*	ResourceLoader::get_recognized_extensions_for_type("PackedScene",&extensions);
		save_select->clear_filters();
		for(int i=0;i<extensions.size();i++) {

			save_select->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
		}*/


	}
}

Error EditorSceneImportDialog::import(const String& p_from, const String& p_to, const String& p_preset) {

	import_path->set_text(p_from);
	save_path->set_text(p_to);
	script_path->set_text(p_preset);

	_import();



	return OK;
}

void EditorSceneImportDialog::_dialog_hid() {

	if (wip_blocked)
		return;
	print_line("DIALOGHID!");
	if (wip_import) {
		memdelete(wip_import);
		wip_import=NULL;
		wip_save_file="";
		wip_rimd=Ref<ResourceImportMetadata>();
	}
}
void EditorSceneImportDialog::_root_default_pressed() {

	root_type->set_disabled(root_default->is_pressed());
}

void EditorSceneImportDialog::_root_type_pressed() {


	root_type_choose->popup(false);
}


void EditorSceneImportDialog::_set_root_type() {

	String type = root_type_choose->get_selected_type();
	if (type==String())
		return;
	root_type->set_text(type);
	if (has_icon(type,"EditorIcons")) {
		root_type->set_icon(get_icon(type,"EditorIcons"));
	} else {
		root_type->set_icon(get_icon("Object","EditorIcons"));
	}
}

void EditorSceneImportDialog::_bind_methods() {


	ClassDB::bind_method("_choose_file",&EditorSceneImportDialog::_choose_file);
	ClassDB::bind_method("_choose_save_file",&EditorSceneImportDialog::_choose_save_file);
	ClassDB::bind_method("_choose_script",&EditorSceneImportDialog::_choose_script);
	ClassDB::bind_method("_import",&EditorSceneImportDialog::_import,DEFVAL(false));
	ClassDB::bind_method("_browse",&EditorSceneImportDialog::_browse);
	ClassDB::bind_method("_browse_target",&EditorSceneImportDialog::_browse_target);
	ClassDB::bind_method("_browse_script",&EditorSceneImportDialog::_browse_script);
	ClassDB::bind_method("_dialog_hid",&EditorSceneImportDialog::_dialog_hid);
	ClassDB::bind_method("_import_confirm",&EditorSceneImportDialog::_import_confirm);
	ClassDB::bind_method("_open_and_import",&EditorSceneImportDialog::_open_and_import);
	ClassDB::bind_method("_root_default_pressed",&EditorSceneImportDialog::_root_default_pressed);
	ClassDB::bind_method("_root_type_pressed",&EditorSceneImportDialog::_root_type_pressed);
	ClassDB::bind_method("_set_root_type",&EditorSceneImportDialog::_set_root_type);


	ADD_SIGNAL( MethodInfo("imported",PropertyInfo(Variant::OBJECT,"scene")) );
}



const EditorSceneImportDialog::FlagInfo EditorSceneImportDialog::scene_flag_names[]={

	{EditorSceneImportPlugin::SCENE_FLAG_REMOVE_NOIMP,("Actions"),"Remove Nodes (-noimp)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_IMPORT_ANIMATIONS,("Actions"),"Import Animations",true},
	{EditorSceneImportPlugin::SCENE_FLAG_COMPRESS_GEOMETRY,("Actions"),"Compress Geometry",false},
	{EditorSceneImportPlugin::SCENE_FLAG_GENERATE_TANGENT_ARRAYS,("Actions"),"Force Generation of Tangent Arrays",false},
	{EditorSceneImportPlugin::SCENE_FLAG_LINEARIZE_DIFFUSE_TEXTURES,("Actions"),"SRGB->Linear Of Diffuse Textures",false},
	{EditorSceneImportPlugin::SCENE_FLAG_CONVERT_NORMALMAPS_TO_XY,("Actions"),"Convert Normal Maps to XY",true},
	{EditorSceneImportPlugin::SCENE_FLAG_SET_LIGHTMAP_TO_UV2_IF_EXISTS,("Actions"),"Set Material Lightmap to UV2 if Tex2Array Exists",true},
	{EditorSceneImportPlugin::SCENE_FLAG_MERGE_KEEP_MATERIALS,("Merge"),"Keep Materials after first import (delete them for re-import).",true},
	{EditorSceneImportPlugin::SCENE_FLAG_MERGE_KEEP_EXTRA_ANIM_TRACKS,("Merge"),"Keep user-added Animation tracks.",true},
	{EditorSceneImportPlugin::SCENE_FLAG_DETECT_ALPHA,("Materials"),"Set Alpha in Materials (-alpha)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_DETECT_VCOLOR,("Materials"),"Set Vert. Color in Materials (-vcol)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_COLLISIONS,("Create"),"Create Collisions and/or Rigid Bodies (-col,-colonly,-rigid)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_PORTALS,("Create"),"Create Portals (-portal)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_ROOMS,("Create"),"Create Rooms (-room)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_SIMPLIFY_ROOMS,("Create"),"Simplify Rooms",false},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_BILLBOARDS,("Create"),"Create Billboards (-bb)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_IMPOSTORS,("Create"),"Create Impostors (-imp:dist)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_LODS,("Create"),"Create LODs (-lod:dist)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_CARS,("Create"),"Create Vehicles (-vehicle)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_WHEELS,("Create"),"Create Vehicle Wheels (-wheel)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_CREATE_NAVMESH,("Create"),"Create Navigation Meshes (-navmesh)",true},
	{EditorSceneImportPlugin::SCENE_FLAG_DETECT_LIGHTMAP_LAYER,("Create"),"Detect LightMap Layer (-lm:<int>).",true},
	{-1,NULL,NULL,false}
};


EditorSceneImportDialog::EditorSceneImportDialog(EditorNode *p_editor, EditorSceneImportPlugin *p_plugin) {


	editor=p_editor;
	plugin=p_plugin;

	set_title(TTR("Import 3D Scene"));
	HBoxContainer *import_hb = memnew( HBoxContainer );
	add_child(import_hb);
	//set_child_rect(import_hb);

	VBoxContainer *vbc = memnew( VBoxContainer );
	import_hb->add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_margin_child(TTR("Source Scene:"),hbc);

	import_path = memnew( LineEdit );
	import_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(import_path);

	Button * import_choose = memnew( Button );
	import_choose->set_text(" .. ");
	hbc->add_child(import_choose);

	import_choose->connect("pressed", this,"_browse");

	hbc = memnew( HBoxContainer );
	vbc->add_margin_child(TTR("Target Path:"),hbc);

	save_path = memnew( LineEdit );
	save_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(save_path);

	Button * save_choose = memnew( Button );
	save_choose->set_text(" .. ");
	hbc->add_child(save_choose);

	save_choose->connect("pressed", this,"_browse_target");

	texture_action = memnew( OptionButton );
	texture_action->add_item(TTR("Same as Target Scene"));
	texture_action->add_item(TTR("Shared"));
	texture_action->select(0);
	vbc->add_margin_child(TTR("Target Texture Folder:"),texture_action);

	import_options = memnew( Tree );
	vbc->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_margin_child(TTR("Options:"),import_options,true);

	file_select = memnew(EditorFileDialog);
	file_select->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	add_child(file_select);


	file_select->set_mode(EditorFileDialog::MODE_OPEN_FILE);

	file_select->connect("file_selected", this,"_choose_file");

	save_select = memnew(EditorDirDialog);
	add_child(save_select);

	//save_select->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	save_select->connect("dir_selected", this,"_choose_save_file");

	get_ok()->connect("pressed", this,"_import");
	get_ok()->set_text(TTR("Import"));

	TreeItem *root = import_options->create_item(NULL);
	import_options->set_hide_root(true);

	const FlagInfo* fn=scene_flag_names;

	Map<String,TreeItem*> categories;

	while(fn->text) {

		String cat = fn->category;
		TreeItem *parent;
		if (!categories.has(cat)) {
			parent = import_options->create_item(root);
			parent->set_text(0,cat);
			categories[cat]=parent;
		} else {
			parent=categories[cat];
		}

		TreeItem *opt = import_options->create_item(parent);
		opt->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		opt->set_checked(0,fn->defval);
		opt->set_editable(0,true);
		opt->set_text(0,fn->text);
		opt->set_metadata(0,fn->value);

		scene_flags.push_back(opt);
		fn++;
	}

	hbc = memnew( HBoxContainer );
	vbc->add_margin_child(TTR("Post-Process Script:"),hbc);

	script_path = memnew( LineEdit );
	script_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(script_path);

	Button * script_choose = memnew( Button );
	script_choose->set_text(" .. ");
	hbc->add_child(script_choose);

	script_choose->connect("pressed", this,"_browse_script");

	script_select = memnew(EditorFileDialog);
	add_child(script_select);
	for(int i=0;i<ScriptServer::get_language_count();i++) {

		ScriptLanguage *sl=ScriptServer::get_language(i);
		String ext = sl->get_extension();
		if (ext=="")
			continue;
		script_select->add_filter("*."+ext+" ; "+sl->get_name());
	}


	script_select->set_mode(EditorFileDialog::MODE_OPEN_FILE);

	script_select->connect("file_selected", this,"_choose_script");

	error_dialog = memnew ( ConfirmationDialog );
	add_child(error_dialog);
	error_dialog->get_ok()->set_text(TTR("Accept"));
	//error_dialog->get_cancel()->hide();


	HBoxContainer *custom_root_hb = memnew( HBoxContainer );
	vbc->add_margin_child(TTR("Custom Root Node Type:"),custom_root_hb);
	root_type = memnew(Button);
	root_type->set_h_size_flags(SIZE_EXPAND_FILL);
	root_type->set_text_align(Button::ALIGN_LEFT);
	root_type->connect("pressed",this,"_root_type_pressed");
	custom_root_hb->add_child(root_type);

	root_default = memnew(CheckBox);
	root_default->set_text(TTR("Auto"));
	root_default->set_pressed(true);
	root_default->connect("pressed",this,"_root_default_pressed");
	custom_root_hb->add_child(root_default);

	root_node_name = memnew( LineEdit );
	root_node_name->set_h_size_flags(SIZE_EXPAND_FILL);
	vbc->add_margin_child(TTR("Root Node Name:"),root_node_name);
	/*
	this_import = memnew( OptionButton );
	this_import->add_item("Overwrite Existing Scene");
	this_import->add_item("Overwrite Existing, Keep Materials");
	this_import->add_item("Keep Existing, Merge with New");
	this_import->add_item("Keep Existing, Ignore New");
	vbc->add_margin_child("This Time:",this_import);

	next_import = memnew( OptionButton );
	next_import->add_item("Overwrite Existing Scene");
	next_import->add_item("Overwrite Existing, Keep Materials");
	next_import->add_item("Keep Existing, Merge with New");
	next_import->add_item("Keep Existing, Ignore New");
	vbc->add_margin_child("Next Time:",next_import);
*/
	set_hide_on_ok(false);

	GLOBAL_DEF("import/shared_textures","res://");
	GlobalConfig::get_singleton()->set_custom_property_info("import/shared_textures",PropertyInfo(Variant::STRING,"import/shared_textures",PROPERTY_HINT_DIR));

	import_hb->add_constant_override("separation",30);


	VBoxContainer *ovb = memnew( VBoxContainer);
	ovb->set_h_size_flags(SIZE_EXPAND_FILL);
	import_hb->add_child(ovb);

	texture_options = memnew( EditorImportTextureOptions );
	ovb->add_child(texture_options);
	texture_options->set_v_size_flags(SIZE_EXPAND_FILL);
	//animation_options->set_flags(EditorImport::
	texture_options->set_format(EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM);
	texture_options->set_flags( EditorTextureImportPlugin::IMAGE_FLAG_FIX_BORDER_ALPHA | EditorTextureImportPlugin::IMAGE_FLAG_REPEAT  | EditorTextureImportPlugin::IMAGE_FLAG_FILTER );


	animation_options = memnew( EditorImportAnimationOptions );
	ovb->add_child(animation_options);
	animation_options->set_v_size_flags(SIZE_EXPAND_FILL);
	animation_options->set_flags(EditorSceneAnimationImportPlugin::ANIMATION_DETECT_LOOP|EditorSceneAnimationImportPlugin::ANIMATION_KEEP_VALUE_TRACKS|EditorSceneAnimationImportPlugin::ANIMATION_OPTIMIZE|EditorSceneAnimationImportPlugin::ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS);


	confirm_import = memnew( ConfirmationDialog );
	add_child(confirm_import);
	VBoxContainer *cvb = memnew( VBoxContainer );
	confirm_import->add_child(cvb);
	//confirm_import->set_child_rect(cvb);

	PanelContainer *pc = memnew( PanelContainer );
	pc->add_style_override("panel",get_stylebox("normal","TextEdit"));
	//ec->add_child(pc);
	missing_files = memnew( RichTextLabel );
	cvb->add_margin_child(TTR("The Following Files are Missing:"),pc,true);
	pc->add_child(missing_files);
	confirm_import->get_ok()->set_text(TTR("Import Anyway"));
	confirm_import->get_cancel()->set_text(TTR("Cancel"));
	confirm_import->connect("popup_hide",this,"_dialog_hid");
	confirm_import->connect("confirmed",this,"_import_confirm");
	confirm_import->set_hide_on_ok(false);

	add_button(TTR("Import & Open"),!OS::get_singleton()->get_swap_ok_cancel())->connect("pressed",this,"_open_and_import");

	confirm_open = memnew( ConfirmationDialog );
	add_child(confirm_open);
	confirm_open->set_text(TTR("Edited scene has not been saved, open imported scene anyway?"));
	confirm_open->connect("confirmed",this,"_import",varray(true));


	wip_import=NULL;
	wip_blocked=false;
	wip_open=false;
	//texture_options->set_format(EditorImport::IMAGE_FORMAT_C);

	root_type_choose = memnew( CreateDialog );
	add_child(root_type_choose);
	root_type_choose->set_base_type("Node");
	root_type_choose->connect("create",this,"_set_root_type");
}



////////////////////////////////



String EditorSceneImportPlugin::get_name() const {

	return "scene_3d";
}

String EditorSceneImportPlugin::get_visible_name() const{

	return TTR("Scene");
}

void EditorSceneImportPlugin::import_dialog(const String& p_from){

	dialog->popup_import(p_from);
}


//////////////////////////


static bool _teststr(const String& p_what,const String& p_str) {

	if (p_what.findn("$"+p_str)!=-1) //blender and other stuff
		return true;
	if (p_what.to_lower().ends_with("-"+p_str)) //collada only supports "_" and "-" besides letters
		return true;
	if (p_what.to_lower().ends_with("_"+p_str)) //collada only supports "_" and "-" besides letters
		return true;
	return false;
}

static String _fixstr(const String& p_what,const String& p_str) {

	if (p_what.findn("$"+p_str)!=-1) //blender and other stuff
		return p_what.replace("$"+p_str,"");
	if (p_what.to_lower().ends_with("-"+p_str)) //collada only supports "_" and "-" besides letters
		return p_what.substr(0,p_what.length()-(p_str.length()+1));
	if (p_what.to_lower().ends_with("_"+p_str)) //collada only supports "_" and "-" besides letters
		return p_what.substr(0,p_what.length()-(p_str.length()+1));
	return p_what;
}



void EditorSceneImportPlugin::_find_resources(const Variant& p_var, Map<Ref<ImageTexture>, TextureRole> &image_map,int p_flags) {


	switch(p_var.get_type()) {

		case Variant::OBJECT: {

			Ref<Resource> res = p_var;
			if (res.is_valid()) {

				if (res->is_class("Texture") && !image_map.has(res)) {

					image_map.insert(res,TEXTURE_ROLE_DEFAULT);


				} else {


					List<PropertyInfo> pl;
					res->get_property_list(&pl);
					for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

						if (E->get().type==Variant::OBJECT || E->get().type==Variant::ARRAY || E->get().type==Variant::DICTIONARY) {
							if (E->get().type==Variant::OBJECT && res->cast_to<SpatialMaterial>() && (E->get().name=="textures/diffuse" || E->get().name=="textures/detail" || E->get().name=="textures/emission")) {

								Ref<ImageTexture> tex =res->get(E->get().name);
								if (tex.is_valid()) {

									image_map.insert(tex,TEXTURE_ROLE_DIFFUSE);
								}

							} else if (E->get().type==Variant::OBJECT && res->cast_to<SpatialMaterial>() && (E->get().name=="textures/normal")) {

								Ref<ImageTexture> tex =res->get(E->get().name);
								if (tex.is_valid()) {

									image_map.insert(tex,TEXTURE_ROLE_NORMALMAP);
									/*
									if (p_flags&SCENE_FLAG_CONVERT_NORMALMAPS_TO_XY)
										res->cast_to<SpatialMaterial>()->set_fixed_flag(SpatialMaterial::FLAG_USE_XY_NORMALMAP,true);
									*/
								}


							} else {
								_find_resources(res->get(E->get().name),image_map,p_flags);
							}
						}
					}

				}
			}

		} break;
		case Variant::DICTIONARY: {

			Dictionary d= p_var;

			List<Variant> keys;
			d.get_key_list(&keys);

			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {


				_find_resources(E->get(),image_map,p_flags);
				_find_resources(d[E->get()],image_map,p_flags);

			}


		} break;
		case Variant::ARRAY: {

			Array a = p_var;
			for(int i=0;i<a.size();i++) {

				_find_resources(a[i],image_map,p_flags);
			}

		} break;
		default: {}

	}

}


Node* EditorSceneImportPlugin::_fix_node(Node *p_node,Node *p_root,Map<Ref<Mesh>,Ref<Shape> > &collision_map,uint32_t p_flags,Map<Ref<ImageTexture>,TextureRole >& image_map) {

	// children first..
	for(int i=0;i<p_node->get_child_count();i++) {


		Node *r = _fix_node(p_node->get_child(i),p_root,collision_map,p_flags,image_map);
		if (!r) {
			print_line("was erased..");
			i--; //was erased
		}
	}

	String name = p_node->get_name();

	bool isroot = p_node==p_root;


	if (!isroot && p_flags&SCENE_FLAG_REMOVE_NOIMP && _teststr(name,"noimp")) {

		memdelete(p_node);
		return NULL;
	}

	{

		List<PropertyInfo> pl;
		p_node->get_property_list(&pl);
		for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

			if (E->get().type==Variant::OBJECT || E->get().type==Variant::ARRAY || E->get().type==Variant::DICTIONARY) {
				_find_resources(p_node->get(E->get().name),image_map,p_flags);
			}
		}

	}




	if (p_flags&SCENE_FLAG_CREATE_BILLBOARDS && p_node->cast_to<MeshInstance>()) {

		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		bool bb=false;

		if ((_teststr(name,"bb"))) {
			bb=true;
		} else if (mi->get_mesh().is_valid() && (_teststr(mi->get_mesh()->get_name(),"bb"))) {
			bb=true;

		}

		if (bb) {
			mi->set_flag(GeometryInstance::FLAG_BILLBOARD,true);
			if (mi->get_mesh().is_valid()) {

				Ref<Mesh> m = mi->get_mesh();
				for(int i=0;i<m->get_surface_count();i++) {

					Ref<SpatialMaterial> fm = m->surface_get_material(i);
					if (fm.is_valid()) {
						//fm->set_flag(Material::FLAG_UNSHADED,true);
						//fm->set_flag(Material::FLAG_DOUBLE_SIDED,true);
						//fm->set_depth_draw_mode(Material::DEPTH_DRAW_NEVER);
						//fm->set_fixed_flag(SpatialMaterial::FLAG_USE_ALPHA,true);
					}
				}
			}
		}
	}


	if (p_flags&(SCENE_FLAG_DETECT_ALPHA|SCENE_FLAG_DETECT_VCOLOR|SCENE_FLAG_SET_LIGHTMAP_TO_UV2_IF_EXISTS) && p_node->cast_to<MeshInstance>()) {

		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		Ref<Mesh> m = mi->get_mesh();

		if (m.is_valid()) {

			for(int i=0;i<m->get_surface_count();i++) {

				Ref<SpatialMaterial> mat = m->surface_get_material(i);
				if (!mat.is_valid())
					continue;

				if (p_flags&SCENE_FLAG_DETECT_ALPHA && _teststr(mat->get_name(),"alpha")) {

					//mat->set_fixed_flag(SpatialMaterial::FLAG_USE_ALPHA,true);
					//mat->set_name(_fixstr(mat->get_name(),"alpha"));
				}
				if (p_flags&SCENE_FLAG_DETECT_VCOLOR && _teststr(mat->get_name(),"vcol")) {

					//mat->set_fixed_flag(SpatialMaterial::FLAG_USE_COLOR_ARRAY,true);
					//mat->set_name(_fixstr(mat->get_name(),"vcol"));
				}

				if (p_flags&SCENE_FLAG_SET_LIGHTMAP_TO_UV2_IF_EXISTS && m->surface_get_format(i)&Mesh::ARRAY_FORMAT_TEX_UV2) {
					//mat->set_flag(Material::FLAG_LIGHTMAP_ON_UV2,true);
				}

			}
		}
	}

	if (p_flags&SCENE_FLAG_REMOVE_NOIMP && p_node->cast_to<AnimationPlayer>()) {
		//remove animations referencing non-importable nodes
		AnimationPlayer *ap = p_node->cast_to<AnimationPlayer>();

		List<StringName> anims;
		ap->get_animation_list(&anims);
		for(List<StringName>::Element *E=anims.front();E;E=E->next()) {

			Ref<Animation> anim=ap->get_animation(E->get());
			ERR_CONTINUE(anim.is_null());
			for(int i=0;i<anim->get_track_count();i++) {
				NodePath path = anim->track_get_path(i);

				for(int j=0;j<path.get_name_count();j++) {
					String node = path.get_name(j);
					if (_teststr(node,"noimp")) {
						anim->remove_track(i);
						i--;
						break;
					}
				}
			}

		}
	}


	if (p_flags&SCENE_FLAG_CREATE_IMPOSTORS && p_node->cast_to<MeshInstance>()) {

		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		String str;

		if ((_teststr(name,"imp"))) {
			str=name;
		} else if (mi->get_mesh().is_valid() && (_teststr(mi->get_mesh()->get_name(),"imp"))) {
			str=mi->get_mesh()->get_name();

		}


		if (p_node->get_parent() && p_node->get_parent()->cast_to<MeshInstance>()) {
			MeshInstance *mi = p_node->cast_to<MeshInstance>();
			MeshInstance *mip = p_node->get_parent()->cast_to<MeshInstance>();
			String d=str.substr(str.find("imp")+3,str.length());
			if (d!="") {
				if ((d[0]<'0' || d[0]>'9'))
					d=d.substr(1,d.length());
				if (d.length() && d[0]>='0' && d[0]<='9') {
					float dist = d.to_double();
					mi->set_flag(GeometryInstance::FLAG_BILLBOARD,true);
					mi->set_flag(GeometryInstance::FLAG_BILLBOARD_FIX_Y,true);
					//mi->set_draw_range_begin(dist);
					//mi->set_draw_range_end(100000);

					//mip->set_draw_range_begin(0);
					//mip->set_draw_range_end(dist);

					if (mi->get_mesh().is_valid()) {

						Ref<Mesh> m = mi->get_mesh();
						for(int i=0;i<m->get_surface_count();i++) {

							Ref<SpatialMaterial> fm = m->surface_get_material(i);
							if (fm.is_valid()) {
								//fm->set_flag(Material::FLAG_UNSHADED,true);
								//fm->set_flag(Material::FLAG_DOUBLE_SIDED,true);
								//fm->set_depth_draw_mode(Material::DEPTH_DRAW_NEVER);
								//fm->set_fixed_flag(SpatialMaterial::FLAG_USE_ALPHA,true);
							}
						}
					}
				}
			}
		}
	}

    if (p_flags&SCENE_FLAG_CREATE_LODS && p_node->cast_to<MeshInstance>()) {

	MeshInstance *mi = p_node->cast_to<MeshInstance>();

	String str;

	if ((_teststr(name,"lod"))) {
	    str=name;
	} else if (mi->get_mesh().is_valid() && (_teststr(mi->get_mesh()->get_name(),"lod"))) {
	    str=mi->get_mesh()->get_name();

	}


	if (p_node->get_parent() && p_node->get_parent()->cast_to<MeshInstance>()) {
	    MeshInstance *mi = p_node->cast_to<MeshInstance>();
	    MeshInstance *mip = p_node->get_parent()->cast_to<MeshInstance>();
	    String d=str.substr(str.find("lod")+3,str.length());
	    if (d!="") {
		if ((d[0]<'0' || d[0]>'9'))
		    d=d.substr(1,d.length());
		if (d.length() && d[0]>='0' && d[0]<='9') {
		    float dist = d.to_double();
		  ///  mi->set_draw_range_begin(dist);
		  //  mi->set_draw_range_end(100000);

		  //  mip->set_draw_range_begin(0);
		  //  mip->set_draw_range_end(dist);

		    /*if (mi->get_mesh().is_valid()) {

			Ref<Mesh> m = mi->get_mesh();
			for(int i=0;i<m->get_surface_count();i++) {

			    Ref<SpatialMaterial> fm = m->surface_get_material(i);
			    if (fm.is_valid()) {
				fm->set_flag(Material::FLAG_UNSHADED,true);
				fm->set_flag(Material::FLAG_DOUBLE_SIDED,true);
				fm->set_hint(Material::HINT_NO_DEPTH_DRAW,true);
				fm->set_fixed_flag(SpatialMaterial::FLAG_USE_ALPHA,true);
			    }
			}
		    }*/
		}
	    }
	}
    }


	if (p_flags&SCENE_FLAG_DETECT_LIGHTMAP_LAYER && _teststr(name,"lm") && p_node->cast_to<MeshInstance>()) {

		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		String str=name;
		int layer = str.substr(str.find("lm")+3,str.length()).to_int();
		//mi->set_baked_light_texture_id(layer);
	}

	if (p_flags&SCENE_FLAG_CREATE_COLLISIONS && _teststr(name,"colonly")) {

		if (isroot)
			return p_node;
		
		if (p_node->cast_to<MeshInstance>()) {
			MeshInstance *mi = p_node->cast_to<MeshInstance>();
			Node * col = mi->create_trimesh_collision_node();
			ERR_FAIL_COND_V(!col,NULL);

			col->set_name(_fixstr(name,"colonly"));
			col->cast_to<Spatial>()->set_transform(mi->get_transform());
			p_node->replace_by(col);
			memdelete(p_node);
			p_node=col;

			StaticBody *sb = col->cast_to<StaticBody>();
			CollisionShape *colshape = memnew( CollisionShape);
			colshape->set_shape(sb->get_shape(0));
			colshape->set_name("shape");
			sb->add_child(colshape);
			colshape->set_owner(p_node->get_owner());
		} else if (p_node->has_meta("empty_draw_type")) {
			String empty_draw_type = String(p_node->get_meta("empty_draw_type"));
			print_line(empty_draw_type);
			StaticBody *sb = memnew( StaticBody);
			sb->set_name(_fixstr(name,"colonly"));
			sb->cast_to<Spatial>()->set_transform(p_node->cast_to<Spatial>()->get_transform());
			p_node->replace_by(sb);
			memdelete(p_node);
			CollisionShape *colshape = memnew( CollisionShape);
			if (empty_draw_type == "CUBE") {
				BoxShape *boxShape = memnew( BoxShape);
				boxShape->set_extents(Vector3(1, 1, 1));
				colshape->set_shape(boxShape);
				colshape->set_name("BoxShape");
			} else if (empty_draw_type == "SINGLE_ARROW") {
				RayShape *rayShape = memnew( RayShape);
				rayShape->set_length(1);
				colshape->set_shape(rayShape);
				colshape->set_name("RayShape");
				sb->cast_to<Spatial>()->rotate_x(Math_PI / 2);
			} else if (empty_draw_type == "IMAGE") {
				PlaneShape *planeShape = memnew( PlaneShape);
				colshape->set_shape(planeShape);
				colshape->set_name("PlaneShape");
			} else {
				SphereShape *sphereShape = memnew( SphereShape);
				sphereShape->set_radius(1);
				colshape->set_shape(sphereShape);
				colshape->set_name("SphereShape");
			}
			sb->add_child(colshape);
			colshape->set_owner(sb->get_owner());
		}

	} else if (p_flags&SCENE_FLAG_CREATE_COLLISIONS && _teststr(name,"rigid") && p_node->cast_to<MeshInstance>()) {

		if (isroot)
			return p_node;

		// get mesh instance and bounding box
		MeshInstance *mi = p_node->cast_to<MeshInstance>();
		Rect3 aabb = mi->get_aabb();

		// create a new rigid body collision node
		RigidBody * rigid_body = memnew( RigidBody );
		Node * col = rigid_body;
		ERR_FAIL_COND_V(!col,NULL);

		// remove node name postfix
		col->set_name(_fixstr(name,"rigid"));
		// get mesh instance xform matrix to the rigid body collision node
		col->cast_to<Spatial>()->set_transform(mi->get_transform());
		// save original node by duplicating it into a new instance and correcting the name
		Node * mesh = p_node->duplicate();
		mesh->set_name(_fixstr(name,"rigid"));
		// reset the xform matrix of the duplicated node so it can inherit parent node xform
		mesh->cast_to<Spatial>()->set_transform(Transform(Basis()));
		// reparent the new mesh node to the rigid body collision node
		p_node->add_child(mesh);
		mesh->set_owner(p_node->get_owner());
		// replace the original node with the rigid body collision node
		p_node->replace_by(col);
		memdelete(p_node);
		p_node=col;

		// create an alias for the rigid body collision node
		RigidBody *rb = col->cast_to<RigidBody>();
		// create a new Box collision shape and set the right extents
		Ref<BoxShape> shape = memnew( BoxShape );
		shape->set_extents(aabb.get_size() * 0.5);
		CollisionShape *colshape = memnew( CollisionShape);
		colshape->set_name("shape");
		colshape->set_shape(shape);
		// reparent the new collision shape to the rigid body collision node
		rb->add_child(colshape);
		colshape->set_owner(p_node->get_owner());

	} else if (p_flags&SCENE_FLAG_CREATE_COLLISIONS &&_teststr(name,"col") && p_node->cast_to<MeshInstance>()) {


		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		mi->set_name(_fixstr(name,"col"));
		Node *col= mi->create_trimesh_collision_node();
		ERR_FAIL_COND_V(!col,NULL);

		col->set_name("col");
		p_node->add_child(col);

		StaticBody *sb=col->cast_to<StaticBody>();
		CollisionShape *colshape = memnew( CollisionShape);
		colshape->set_shape(sb->get_shape(0));
		colshape->set_name("shape");
		col->add_child(colshape);
		colshape->set_owner(p_node->get_owner());
		sb->set_owner(p_node->get_owner());

	} else if (p_flags&SCENE_FLAG_CREATE_NAVMESH &&_teststr(name,"navmesh") && p_node->cast_to<MeshInstance>()) {

		if (isroot)
			return p_node;

		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		Ref<Mesh> mesh=mi->get_mesh();
		ERR_FAIL_COND_V(mesh.is_null(),NULL);
		NavigationMeshInstance *nmi = memnew(  NavigationMeshInstance );


		nmi->set_name(_fixstr(name,"navmesh"));
		Ref<NavigationMesh> nmesh = memnew( NavigationMesh);
		nmesh->create_from_mesh(mesh);
		nmi->set_navigation_mesh(nmesh);
		nmi->cast_to<Spatial>()->set_transform(mi->get_transform());
		p_node->replace_by(nmi);
		memdelete(p_node);
		p_node=nmi;
	} else if (p_flags&SCENE_FLAG_CREATE_CARS &&_teststr(name,"vehicle")) {

		if (isroot)
			return p_node;

		Node *owner = p_node->get_owner();
		Spatial *s = p_node->cast_to<Spatial>();
		VehicleBody *bv = memnew( VehicleBody );
		String n = _fixstr(p_node->get_name(),"vehicle");
		bv->set_name(n);
		p_node->replace_by(bv);
		p_node->set_name(n);
		bv->add_child(p_node);
		bv->set_owner(owner);
		p_node->set_owner(owner);
		bv->set_transform(s->get_transform());
		s->set_transform(Transform());

		p_node=bv;


	} else if (p_flags&SCENE_FLAG_CREATE_CARS &&_teststr(name,"wheel")) {

		if (isroot)
			return p_node;

		Node *owner = p_node->get_owner();
		Spatial *s = p_node->cast_to<Spatial>();
		VehicleWheel *bv = memnew( VehicleWheel );
		String n = _fixstr(p_node->get_name(),"wheel");
		bv->set_name(n);
		p_node->replace_by(bv);
		p_node->set_name(n);
		bv->add_child(p_node);
		bv->set_owner(owner);
		p_node->set_owner(owner);
		bv->set_transform(s->get_transform());
		s->set_transform(Transform());

		p_node=bv;

	} else if (p_flags&SCENE_FLAG_CREATE_ROOMS && _teststr(name,"room") && p_node->cast_to<MeshInstance>()) {


		if (isroot)
			return p_node;

		MeshInstance *mi = p_node->cast_to<MeshInstance>();
		PoolVector<Face3> faces = mi->get_faces(VisualInstance::FACES_SOLID);


		BSP_Tree bsptree(faces);

		Ref<RoomBounds> area = memnew( RoomBounds );
		//area->set_bounds(faces);
		//area->set_geometry_hint(faces);


		Room * room = memnew( Room );
		room->set_name(_fixstr(name,"room"));
		room->set_transform(mi->get_transform());
		room->set_room(area);

		p_node->replace_by(room);
		memdelete(p_node);
		p_node=room;

	} else if (p_flags&SCENE_FLAG_CREATE_ROOMS &&_teststr(name,"room")) {

		if (isroot)
			return p_node;

		Spatial *dummy = p_node->cast_to<Spatial>();
		ERR_FAIL_COND_V(!dummy,NULL);

		Room * room = memnew( Room );
		room->set_name(_fixstr(name,"room"));
		room->set_transform(dummy->get_transform());

		p_node->replace_by(room);
		memdelete(p_node);
		p_node=room;

		//room->compute_room_from_subtree();

	} else if (p_flags&SCENE_FLAG_CREATE_PORTALS &&_teststr(name,"portal") && p_node->cast_to<MeshInstance>()) {

		if (isroot)
			return p_node;

		MeshInstance *mi = p_node->cast_to<MeshInstance>();
		PoolVector<Face3> faces = mi->get_faces(VisualInstance::FACES_SOLID);

		ERR_FAIL_COND_V(faces.size()==0,NULL);
		//step 1 compute the plane
		Set<Vector3> points;
		Plane plane;

		Vector3 center;

		for(int i=0;i<faces.size();i++) {

			Face3 f = faces.get(i);
			Plane p = f.get_plane();
			plane.normal+=p.normal;
			plane.d+=p.d;

			for(int i=0;i<3;i++) {

				Vector3 v = f.vertex[i].snapped(0.01);
				if (!points.has(v)) {
					points.insert(v);
					center+=v;
				}
			}
		}

		plane.normal.normalize();
		plane.d/=faces.size();
		center/=points.size();

		//step 2, create points

		Transform t;
		t.basis.from_z(plane.normal);
		t.basis.transpose();
		t.origin=center;

		Vector<Point2> portal_points;

		for(Set<Vector3>::Element *E=points.front();E;E=E->next()) {

			Vector3 local = t.xform_inv(E->get());
			portal_points.push_back(Point2(local.x,local.y));
		}
		// step 3 bubbly sort points

		int swaps=0;

		do {
			swaps=0;

			for(int i=0;i<portal_points.size()-1;i++) {

				float a = portal_points[i].angle();
				float b = portal_points[i+1].angle();

				if (a>b) {
					SWAP( portal_points[i], portal_points[i+1] );
					swaps++;
				}

			}

		} while(swaps);


		Portal *portal = memnew( Portal );

		portal->set_shape(portal_points);
		portal->set_transform( mi->get_transform() * t);

		p_node->replace_by(portal);
		memdelete(p_node);
		p_node=portal;

	} else if (p_node->cast_to<MeshInstance>()) {

		//last attempt, maybe collision insde the mesh data

		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		Ref<Mesh> mesh = mi->get_mesh();
		if (!mesh.is_null()) {

			if (p_flags&SCENE_FLAG_CREATE_COLLISIONS && _teststr(mesh->get_name(),"col")) {

				mesh->set_name( _fixstr(mesh->get_name(),"col") );
				Ref<Shape> shape;

				if (collision_map.has(mesh)) {
					shape = collision_map[mesh];

				} else {

					shape = mesh->create_trimesh_shape();
					if (!shape.is_null())
						collision_map[mesh]=shape;


				}

				if (!shape.is_null()) {
#if 0
					StaticBody* static_body = memnew( StaticBody );
					ERR_FAIL_COND_V(!static_body,NULL);
					static_body->set_name( String(mesh->get_name()) + "_col" );
					shape->set_name(static_body->get_name());
					static_body->add_shape(shape);

					mi->add_child(static_body);
					if (mi->get_owner())
						static_body->set_owner( mi->get_owner() );
#endif
				}

			}

			for(int i=0;i<mesh->get_surface_count();i++) {

				Ref<SpatialMaterial> fm = mesh->surface_get_material(i);
				if (fm.is_valid()) {
					String name = fm->get_name();
				/*	if (_teststr(name,"alpha")) {
						fm->set_fixed_flag(SpatialMaterial::FLAG_USE_ALPHA,true);
						name=_fixstr(name,"alpha");
					}

					if (_teststr(name,"vcol")) {
						fm->set_fixed_flag(SpatialMaterial::FLAG_USE_COLOR_ARRAY,true);
						name=_fixstr(name,"vcol");
					}*/
					fm->set_name(name);
				}
			}

		}

	}


	return p_node;
}

#if 0

Error EditorImport::import_scene(const String& p_path,const String& p_dest_path,const String& p_resource_path,uint32_t p_flags,ImageFormat p_image_format,ImageCompression p_image_compression,uint32_t p_image_flags,float p_quality,uint32_t animation_flags,Node **r_scene,Ref<EditorPostImport> p_post_import) {


}
#endif

void EditorSceneImportPlugin::_tag_import_paths(Node *p_scene,Node *p_node) {

	if (p_scene!=p_node && p_node->get_owner()!=p_scene)
		return;

	NodePath path = p_scene->get_path_to(p_node);
	p_node->set_import_path( path );

	Spatial *snode=p_node->cast_to<Spatial>();

	if (snode) {

		snode->set_import_transform(snode->get_transform());
	}

	for(int i=0;i<p_node->get_child_count();i++) {
		_tag_import_paths(p_scene,p_node->get_child(i));
	}

}

Error EditorSceneImportPlugin::import1(const Ref<ResourceImportMetadata>& p_from,Node**r_node,List<String> *r_missing) {

	Ref<ResourceImportMetadata> from=p_from;

	ERR_FAIL_COND_V(from->get_source_count()!=1,ERR_INVALID_PARAMETER);

	String src_path=EditorImportPlugin::expand_source_path(from->get_source_path(0));

	Ref<EditorSceneImporter> importer;
	String ext=src_path.get_extension().to_lower();


	EditorProgress progress("import",TTR("Import Scene"),104);
	progress.step(TTR("Importing Scene.."),0);

	for(int i=0;i<importers.size();i++) {

		List<String> extensions;
		importers[i]->get_extensions(&extensions);

		for(List<String>::Element *E=extensions.front();E;E=E->next()) {

			if (E->get().to_lower()==ext) {

				importer = importers[i];
				break;
			}
		}

		if (importer.is_valid())
			break;
	}

	ERR_FAIL_COND_V(!importer.is_valid(),ERR_FILE_UNRECOGNIZED);

	int animation_flags=p_from->get_option("animation_flags");
	int scene_flags = from->get_option("flags");
	int fps = 24;
	if (from->has_option("animation_bake_fps"))
		fps=from->get_option("animation_bake_fps");


	Array clips;
	if (from->has_option("animation_clips"))
		clips=from->get_option("animation_clips");

	uint32_t import_flags=0;
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_DETECT_LOOP)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION_DETECT_LOOP;
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_KEEP_VALUE_TRACKS)
		import_flags |= EditorSceneImporter::IMPORT_ANIMATION_KEEP_VALUE_TRACKS;
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_OPTIMIZE)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION_OPTIMIZE;
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS;
	if (scene_flags&SCENE_FLAG_IMPORT_ANIMATIONS)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION;
	/*
	if (scene_flags&SCENE_FLAG_FAIL_ON_MISSING_IMAGES)
		import_flags|=EditorSceneImporter::IMPORT_FAIL_ON_MISSING_DEPENDENCIES;
	*/
	if (scene_flags&SCENE_FLAG_GENERATE_TANGENT_ARRAYS)
		import_flags|=EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS;





	Error err=OK;
	Node *scene = importer->import_scene(src_path,import_flags,fps,r_missing,&err);
	if (!scene || err!=OK) {
		return err;
	}

	if (from->has_option("root_type")) {
		String type = from->get_option("root_type");
		Object *base = ClassDB::instance(type);
		Node *base_node = NULL;
		if (base)
			base_node=base->cast_to<Node>();

		if (base_node) {

			scene->replace_by(base_node);
			memdelete(scene);
			scene=base_node;
		}
	}

	scene->set_name(from->get_option("root_name"));
	_tag_import_paths(scene,scene);

	*r_node=scene;
	return OK;
}


void EditorSceneImportPlugin::_create_clips(Node *scene, const Array& p_clips,bool p_bake_all) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;

	Node* n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = n->cast_to<AnimationPlayer>();
	ERR_FAIL_COND(!anim);

	if (!anim->has_animation("default"))
		return;


	Ref<Animation> default_anim = anim->get_animation("default");

	for(int i=0;i<p_clips.size();i+=4) {

		String name = p_clips[i];
		float from=p_clips[i+1];
		float to=p_clips[i+2];
		bool loop=p_clips[i+3];
		if (from>=to)
			continue;

		Ref<Animation> new_anim = memnew( Animation );

		for(int j=0;j<default_anim->get_track_count();j++) {


			List<float> keys;
			int kc = default_anim->track_get_key_count(j);
			int dtrack=-1;
			for(int k=0;k<kc;k++) {

				float kt = default_anim->track_get_key_time(j,k);
				if (kt>=from && kt<to) {

					//found a key within range, so create track
					if (dtrack==-1) {
						new_anim->add_track(default_anim->track_get_type(j));
						dtrack = new_anim->get_track_count()-1;
						new_anim->track_set_path(dtrack,default_anim->track_get_path(j));

						if (kt>(from+0.01) && k>0) {

							if (default_anim->track_get_type(j)==Animation::TYPE_TRANSFORM) {
								Quat q;
								Vector3 p;
								Vector3 s;
								default_anim->transform_track_interpolate(j,from,&p,&q,&s);
								new_anim->transform_track_insert_key(dtrack,0,p,q,s);
							}
						}

					}

					if (default_anim->track_get_type(j)==Animation::TYPE_TRANSFORM) {
						Quat q;
						Vector3 p;
						Vector3 s;
						default_anim->transform_track_get_key(j,k,&p,&q,&s);
						new_anim->transform_track_insert_key(dtrack,kt-from,p,q,s);
					}

				}

				if (dtrack!=-1 && kt>=to) {

					if (default_anim->track_get_type(j)==Animation::TYPE_TRANSFORM) {
						Quat q;
						Vector3 p;
						Vector3 s;
						default_anim->transform_track_interpolate(j,to,&p,&q,&s);
						new_anim->transform_track_insert_key(dtrack,to-from,p,q,s);
					}
				}

			}

			if (dtrack==-1 && p_bake_all) {
				new_anim->add_track(default_anim->track_get_type(j));
				dtrack = new_anim->get_track_count()-1;
				new_anim->track_set_path(dtrack,default_anim->track_get_path(j));
				if (default_anim->track_get_type(j)==Animation::TYPE_TRANSFORM) {


					Quat q;
					Vector3 p;
					Vector3 s;
					default_anim->transform_track_interpolate(j,from,&p,&q,&s);
					new_anim->transform_track_insert_key(dtrack,0,p,q,s);
					default_anim->transform_track_interpolate(j,to,&p,&q,&s);
					new_anim->transform_track_insert_key(dtrack,to-from,p,q,s);
				}

			}
		}


		new_anim->set_loop(loop);
		new_anim->set_length(to-from);
		anim->add_animation(name,new_anim);
	}

	anim->remove_animation("default"); //remove default (no longer needed)
}

void EditorSceneImportPlugin::_filter_anim_tracks(Ref<Animation> anim,Set<String> &keep) {

	Ref<Animation> a = anim;
	ERR_FAIL_COND(!a.is_valid());

	print_line("From Anim "+anim->get_name()+":");

	for(int j=0;j<a->get_track_count();j++) {

		String path = a->track_get_path(j);

		if (!keep.has(path)) {

			print_line("Remove: "+path);
			a->remove_track(j);
			j--;
		}

	}
}


void EditorSceneImportPlugin::_filter_tracks(Node *scene, const String& p_text) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;
	Node* n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = n->cast_to<AnimationPlayer>();
	ERR_FAIL_COND(!anim);

	Vector<String> strings = p_text.split("\n");
	for(int i=0;i<strings.size();i++) {

		strings[i]=strings[i].strip_edges();
	}

	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for(List<StringName>::Element *E=anim_names.front();E;E=E->next()) {

		String name = E->get();
		bool valid_for_this=false;
		bool valid=false;

		Set<String> keep;
		Set<String> keep_local;


		for(int i=0;i<strings.size();i++) {


			if (strings[i].begins_with("@")) {

				valid_for_this=false;
				for(Set<String>::Element *F=keep_local.front();F;F=F->next()) {
					keep.insert(F->get());
				}
				keep_local.clear();

				Vector<String> filters=strings[i].substr(1,strings[i].length()).split(",");
				for(int j=0;j<filters.size();j++) {

					String fname = filters[j].strip_edges();
					if (fname=="")
						continue;
					int fc = fname[0];
					bool plus;
					if (fc=='+')
						plus=true;
					else if (fc=='-')
						 plus=false;
					else
						continue;

					String filter=fname.substr(1,fname.length()).strip_edges();

					if (!name.matchn(filter))
						continue;
					valid_for_this=plus;
				}

				if (valid_for_this)
					valid=true;

			} else if (valid_for_this) {

				Ref<Animation> a = anim->get_animation(name);
				if (!a.is_valid())
					continue;

				for(int j=0;j<a->get_track_count();j++) {

					String path = a->track_get_path(j);

					String tname = strings[i];
					if (tname=="")
						continue;
					int fc = tname[0];
					bool plus;
					if (fc=='+')
						plus=true;
					else if (fc=='-')
						 plus=false;
					else
						continue;

					String filter=tname.substr(1,tname.length()).strip_edges();

					if (!path.matchn(filter))
						continue;

					if (plus)
						keep_local.insert(path);
					else if (!keep.has(path)) {
						keep_local.erase(path);
					}
				}

			}

		}

		if (valid) {
			for(Set<String>::Element *F=keep_local.front();F;F=F->next()) {
				keep.insert(F->get());
			}
			print_line("FILTERING ANIM: "+String(E->get()));
			_filter_anim_tracks(anim->get_animation(name),keep);
		} else {
			print_line("NOT FILTERING ANIM: "+String(E->get()));

		}

	}



}

void EditorSceneImportPlugin::_optimize_animations(Node *scene, float p_max_lin_error,float p_max_ang_error,float p_max_angle) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;
	Node* n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = n->cast_to<AnimationPlayer>();
	ERR_FAIL_COND(!anim);


	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for(List<StringName>::Element *E=anim_names.front();E;E=E->next()) {

		Ref<Animation> a = anim->get_animation(E->get());
		a->optimize(p_max_lin_error,p_max_ang_error,Math::deg2rad(p_max_angle));
	}
}


void EditorSceneImportPlugin::_find_resources_to_merge(Node *scene, Node *node, bool p_merge_material, Map<String, Ref<Material> > &materials, bool p_merge_anims, Map<String,Ref<Animation> >& merged_anims,Set<Ref<Mesh> > &tested_meshes) {

	if (node!=scene && node->get_owner()!=scene)
		return;

	String path = scene->get_path_to(node);

	if (p_merge_anims && node->cast_to<AnimationPlayer>()) {

		AnimationPlayer *ap = node->cast_to<AnimationPlayer>();
		List<StringName> anims;
		ap->get_animation_list(&anims);
		for (List<StringName>::Element *E=anims.front();E;E=E->next()) {
			Ref<Animation> anim = ap->get_animation(E->get());
			Ref<Animation> clone;

			bool has_user_tracks=false;

			for(int i=0;i<anim->get_track_count();i++) {

				if (!anim->track_is_imported(i)) {
					has_user_tracks=true;
					break;
				}
			}

			if (has_user_tracks) {

				clone = anim->duplicate();
				for(int i=0;i<clone->get_track_count();i++) {
					if (clone->track_is_imported(i)) {
						clone->remove_track(i);
						i--;
					}
				}

				merged_anims[path+"::"+String(E->get())]=clone;
			}
		}
	}



	if (p_merge_material && node->cast_to<MeshInstance>()) {
		MeshInstance *mi=node->cast_to<MeshInstance>();
		Ref<Mesh> mesh = mi->get_mesh();
		if (mesh.is_valid() && mesh->get_name()!=String() && !tested_meshes.has(mesh)) {

			for(int i=0;i<mesh->get_surface_count();i++) {
				Ref<Material> material = mesh->surface_get_material(i);

				if (material.is_valid()) {

					String sname = mesh->surface_get_name(i);
					if (sname=="")
						sname="surf_"+itos(i);

					sname=mesh->get_name()+":surf:"+sname;
					materials[sname]=material;
				}
			}

			tested_meshes.insert(mesh);
		}

		if (mesh.is_valid()) {

			for(int i=0;i<mesh->get_surface_count();i++) {
				Ref<Material> material = mi->get_surface_material(i);
				if (material.is_valid()) {
					String sname = mesh->surface_get_name(i);
					if (sname=="")
						sname="surf_"+itos(i);

					sname=path+":inst_surf:"+sname;
					materials[sname]=material;
				}
			}

		}

		Ref<Material> override = mi->get_material_override();

		if (override.is_valid()) {

			materials[path+":override"]=override;
		}
	}



	for(int i=0;i<node->get_child_count();i++) {
		_find_resources_to_merge(scene,node->get_child(i),p_merge_material,materials,p_merge_anims,merged_anims,tested_meshes);
	}

}


void EditorSceneImportPlugin::_merge_found_resources(Node *scene, Node *node, bool p_merge_material, const Map<String, Ref<Material> > &materials, bool p_merge_anims, const Map<String,Ref<Animation> >& merged_anims, Set<Ref<Mesh> > &tested_meshes) {

	if (node!=scene && node->get_owner()!=scene)
		return;

	String path = scene->get_path_to(node);

	print_line("at path: "+path);

	if (node->cast_to<AnimationPlayer>()) {

		AnimationPlayer *ap = node->cast_to<AnimationPlayer>();
		List<StringName> anims;
		ap->get_animation_list(&anims);
		for (List<StringName>::Element *E=anims.front();E;E=E->next()) {
			Ref<Animation> anim = ap->get_animation(E->get());

			String anim_path = path+"::"+String(E->get());

			if (merged_anims.has(anim_path)) {

				Ref<Animation> user_tracks = merged_anims[anim_path];
				for(int i=0;i<user_tracks->get_track_count();i++) {

					int idx = anim->get_track_count();
					anim->add_track(user_tracks->track_get_type(i));
					anim->track_set_path(idx,user_tracks->track_get_path(i));
					anim->track_set_interpolation_type(idx,user_tracks->track_get_interpolation_type(i));
					for(int j=0;j<user_tracks->track_get_key_count(i);j++) {

						float ofs = user_tracks->track_get_key_time(i,j);
						float trans = user_tracks->track_get_key_transition(i,j);
						Variant value = user_tracks->track_get_key_value(i,j);

						anim->track_insert_key(idx,ofs,value,trans);
					}
				}
			}
		}
	}



	if (node->cast_to<MeshInstance>()) {
		MeshInstance *mi=node->cast_to<MeshInstance>();
		Ref<Mesh> mesh = mi->get_mesh();
		if (mesh.is_valid() && mesh->get_name()!=String() && !tested_meshes.has(mesh)) {

			for(int i=0;i<mesh->get_surface_count();i++) {

				String sname = mesh->surface_get_name(i);
				if (sname=="")
					sname="surf_"+itos(i);

				sname=mesh->get_name()+":surf:"+sname;


				if (materials.has(sname)) {

					mesh->surface_set_material(i,materials[sname]);
				}
			}

			tested_meshes.insert(mesh);
		}

		if (mesh.is_valid()) {

			for(int i=0;i<mesh->get_surface_count();i++) {

				String sname = mesh->surface_get_name(i);
				if (sname=="")
					sname="surf_"+itos(i);

				sname=path+":inst_surf:"+sname;


				if (materials.has(sname)) {

					mi->set_surface_material(i,materials[sname]);
				}
			}

		}


		String opath = path+":override";
		if (materials.has(opath)) {
			mi->set_material_override(materials[opath]);
		}

	}



	for(int i=0;i<node->get_child_count();i++) {
		_merge_found_resources(scene,node->get_child(i),p_merge_material,materials,p_merge_anims,merged_anims,tested_meshes);
	}

}

Error EditorSceneImportPlugin::import2(Node *scene, const String& p_dest_path, const Ref<ResourceImportMetadata>& p_from) {

	Error err=OK;
	Ref<ResourceImportMetadata> from=p_from;
	String src_path=EditorImportPlugin::expand_source_path(from->get_source_path(0));
	int animation_flags=p_from->get_option("animation_flags");
	Array animation_clips = p_from->get_option("animation_clips");
	String animation_filter = p_from->get_option("animation_filters");
	int scene_flags = from->get_option("flags");
	float anim_optimizer_linerr=0.05;
	float anim_optimizer_angerr=0.01;
	float anim_optimizer_maxang=22;

	if (from->has_option("animation_optimizer_linear_error"))
		anim_optimizer_linerr=from->get_option("animation_optimizer_linear_error");
	if (from->has_option("animation_optimizer_angular_error"))
		anim_optimizer_angerr=from->get_option("animation_optimizer_angular_error");
	if (from->has_option("animation_optimizer_max_angle"))
		anim_optimizer_maxang=from->get_option("animation_optimizer_max_angle");

	EditorProgress progress("import",TTR("Import Scene"),104);
	progress.step(TTR("Importing Scene.."),2);


	from->set_source_md5(0,FileAccess::get_md5(src_path));
	from->set_editor(get_name());

	from->set_option("reimport",false);
	String target_res_path=p_dest_path.get_base_dir();

	Map<Ref<Mesh>,Ref<Shape> > collision_map;

	Map< Ref<ImageTexture>,TextureRole > imagemap;

	scene=_fix_node(scene,scene,collision_map,scene_flags,imagemap);
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_OPTIMIZE)
		_optimize_animations(scene,anim_optimizer_linerr,anim_optimizer_angerr,anim_optimizer_maxang);
	if (animation_clips.size())
		_create_clips(scene,animation_clips,animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS);

	_filter_tracks(scene,animation_filter);


	if (scene_flags&(SCENE_FLAG_MERGE_KEEP_MATERIALS|SCENE_FLAG_MERGE_KEEP_EXTRA_ANIM_TRACKS) && FileAccess::exists(p_dest_path)) {
		//must merge!

		print_line("MUST MERGE");
		Ref<PackedScene> pscene = ResourceLoader::load(p_dest_path,"PackedScene",true);
		if (pscene.is_valid()) {

			Node *instance = pscene->instance();
			if (instance) {
				Map<String,Ref<Animation> > merged_anims;
				Map<String,Ref<Material> > merged_materials;
				Set<Ref<Mesh> > tested_meshes;

				_find_resources_to_merge(instance,instance,scene_flags&SCENE_FLAG_MERGE_KEEP_MATERIALS,merged_materials,scene_flags&SCENE_FLAG_MERGE_KEEP_EXTRA_ANIM_TRACKS,merged_anims,tested_meshes);

				tested_meshes.clear();
				_merge_found_resources(scene,scene,scene_flags&SCENE_FLAG_MERGE_KEEP_MATERIALS,merged_materials,scene_flags&SCENE_FLAG_MERGE_KEEP_EXTRA_ANIM_TRACKS,merged_anims,tested_meshes);

				memdelete(instance);
			}

		}

	}

	/// BEFORE ANYTHING, RUN SCRIPT

	progress.step(TTR("Running Custom Script.."),2);

	String post_import_script_path = from->get_option("post_import_script");
	Ref<EditorScenePostImport>  post_import_script;

	if (post_import_script_path!="") {
		post_import_script_path = post_import_script_path;
		Ref<Script> scr = ResourceLoader::load(post_import_script_path);
		if (!scr.is_valid()) {
			EditorNode::add_io_error(TTR("Couldn't load post-import script:")+" "+post_import_script_path);
		} else {

			post_import_script = Ref<EditorScenePostImport>( memnew( EditorScenePostImport ) );
			post_import_script->set_script(scr.get_ref_ptr());
			if (!post_import_script->get_script_instance()) {
				EditorNode::add_io_error(TTR("Invalid/broken script for post-import (check console):")+" "+post_import_script_path);
				post_import_script.unref();
				return ERR_CANT_CREATE;
			}
		}
	}


	if (post_import_script.is_valid()) {
		scene = post_import_script->post_import(scene);
		if (!scene) {
			EditorNode::add_io_error(TTR("Error running post-import script:")+" "+post_import_script_path);
			return err;
		}


	}


	/// IMPORT IMAGES


	int idx=0;

	int image_format = from->get_option("texture_format");
	int image_flags =  from->get_option("texture_flags");
	float image_quality = from->get_option("texture_quality");

	for (Map< Ref<ImageTexture>,TextureRole >::Element *E=imagemap.front();E;E=E->next()) {

		//texture could be converted to something more useful for 3D, that could load individual mipmaps and stuff
		//but not yet..

		Ref<ImageTexture> texture = E->key();

		ERR_CONTINUE(!texture.is_valid());

		String path = texture->get_path();
		String fname= path.get_file();
		String target_path = GlobalConfig::get_singleton()->localize_path(target_res_path.plus_file(fname));
		progress.step(TTR("Import Image:")+" "+fname,3+(idx)*100/imagemap.size());

		idx++;

		if (path==target_path) {

			EditorNode::add_io_error(TTR("Can't import a file over itself:")+" "+target_path);
			continue;
		}

		if (!target_path.begins_with("res://")) {
			EditorNode::add_io_error(vformat(TTR("Couldn't localize path: %s (already local)"),target_path));
			continue;
		}


		{


			target_path=target_path.get_basename()+".tex";

			Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );

			uint32_t flags = image_flags;
			if (E->get()==TEXTURE_ROLE_DIFFUSE && scene_flags&SCENE_FLAG_LINEARIZE_DIFFUSE_TEXTURES)
				flags|=EditorTextureImportPlugin::IMAGE_FLAG_CONVERT_TO_LINEAR;

			if (E->get()==TEXTURE_ROLE_NORMALMAP && scene_flags&SCENE_FLAG_CONVERT_NORMALMAPS_TO_XY)
				flags|=EditorTextureImportPlugin::IMAGE_FLAG_CONVERT_NORMAL_TO_XY;

			imd->set_option("flags",flags);
			imd->set_option("format",image_format);
			imd->set_option("quality",image_quality);
			imd->set_option("atlas",false);
			imd->add_source(EditorImportPlugin::validate_source_path(path));


			if (FileAccess::exists(target_path)) {

				 Ref<ResourceImportMetadata> rimdex = ResourceLoader::load_import_metadata(target_path);
				 if (rimdex.is_valid()) {
					//make sure the options are the same, otherwise re-import
					List<String> opts;
					imd->get_options(&opts);
					bool differ=false;
					for (List<String>::Element *E=opts.front();E;E=E->next()) {
						if (!(rimdex->get_option(E->get())==imd->get_option(E->get()))) {
							differ=true;
							break;
						}
					}

					if (!differ) {
						texture->set_path(target_path);
						continue; //already imported
					}
				}
			}

			EditorTextureImportPlugin::get_singleton()->import(target_path,imd);

		}
	}



	progress.step(TTR("Saving.."),104);

	Ref<PackedScene> packer = memnew( PackedScene );
	packer->pack(scene);
	//packer->set_path(p_dest_path); do not take over, let the changed files reload themselves
	packer->set_import_metadata(from);

	print_line("SAVING TO: "+p_dest_path);
	err = ResourceSaver::save(p_dest_path,packer); //do not take over, let the changed files reload themselves

	//EditorFileSystem::get_singleton()->update_resource(packer);

	memdelete(scene);

	/*
	scene->set_filename(p_dest_path);
	if (r_scene) {
		*r_scene=scene;
	} else {
		memdelete(scene);
	}

	String sp;
	if (p_post_import.is_valid() && !p_post_import->get_script().is_null()) {
		Ref<Script> scr = p_post_import->get_script();
		if (scr.is_valid())
			sp=scr->get_path();
	}

	String op=_getrelpath(p_path,p_dest_path);

	*/

	EditorNode::get_singleton()->reload_scene(p_dest_path);

	return err;

}


Error EditorSceneImportPlugin::import(const String& p_dest_path, const Ref<ResourceImportMetadata>& p_from){


	Node *n=NULL;
	Error err = import1(p_from,&n);
	if (err!=OK) {
		if (n) {
			memdelete(n);
		}
		return err;
	}
	return import2(n,p_dest_path,p_from);
}

void EditorSceneImportPlugin::add_importer(const Ref<EditorSceneImporter>& p_importer) {

	importers.push_back(p_importer);
}

void EditorSceneImportPlugin::import_from_drop(const Vector<String>& p_drop,const String& p_dest_path) {

	List<String> extensions;
	for(int i=0;i<importers.size();i++) {
		importers[i]->get_extensions(&extensions);
	}
	//bool warn_compatible=false;
	for(int i=0;i<p_drop.size();i++) {

		String extension = p_drop[i].get_extension().to_lower();

		for(List<String>::Element *E=extensions.front();E;E=E->next()) {

			if (E->get()==extension) {

				dialog->popup_import(String());
				dialog->setup_popup(p_drop[i],p_dest_path);
				return;
			}
		}
	}

}


EditorSceneImportPlugin::EditorSceneImportPlugin(EditorNode* p_editor) {

	dialog = memnew( EditorSceneImportDialog(p_editor,this) );
	p_editor->get_gui_base()->add_child(dialog);
}


///////////////////////////////


String EditorSceneAnimationImportPlugin::get_name() const {

	return "anim_3d";
}
String EditorSceneAnimationImportPlugin::get_visible_name() const{


	return TTR("3D Scene Animation");
}
void EditorSceneAnimationImportPlugin::import_dialog(const String& p_from){


}
Error EditorSceneAnimationImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){

	return OK;
}

EditorSceneAnimationImportPlugin::EditorSceneAnimationImportPlugin(EditorNode* p_editor) {


}
#endif
