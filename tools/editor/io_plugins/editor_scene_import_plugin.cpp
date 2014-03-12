/*************************************************************************/
/*  editor_scene_import_plugin.cpp                                       */
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
#include "editor_scene_import_plugin.h"
#include "globals.h"
#include "tools/editor/editor_node.h"
#include "scene/resources/packed_scene.h"
#include "os/file_access.h"
#include "scene/3d/path.h"
#include "scene/animation/animation_player.h"
#include "io/resource_saver.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/room_instance.h"
#include "scene/3d/portal.h"
#include "core/translation.h"




EditorSceneImporter::EditorSceneImporter() {


}

void EditorScenePostImport::_bind_methods() {

	BIND_VMETHOD( MethodInfo("post_import",PropertyInfo(Variant::OBJECT,"scene")) );

}

Error EditorScenePostImport::post_import(Node* p_scene) {

	if (get_script_instance())
		return Error(int(get_script_instance()->call("post_import",p_scene)));
	return OK;
}

EditorScenePostImport::EditorScenePostImport() {


}


/////////////////////////////


class EditorImportAnimationOptions : public VBoxContainer {

	OBJ_TYPE( EditorImportAnimationOptions, VBoxContainer );


	Tree *flags;
	Vector<TreeItem*> items;

	bool updating;

	void _changed();
protected:
	static void _bind_methods();
	void _notification(int p_what);

public:

	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;


	EditorImportAnimationOptions();


};

////////////////////////////

class EditorSceneImportDialog : public ConfirmationDialog  {

	OBJ_TYPE(EditorSceneImportDialog,ConfirmationDialog);


	EditorImportTextureOptions *texture_options;
	EditorImportAnimationOptions *animation_options;

	EditorSceneImportPlugin *plugin;

	EditorNode *editor;

	LineEdit *import_path;
	LineEdit *save_path;
	LineEdit *script_path;
	Tree *import_options;
	FileDialog *file_select;
	FileDialog *script_select;
	EditorDirDialog *save_select;
	OptionButton *texture_action;

	Vector<TreeItem*> scene_flags;

	Map<Ref<Mesh>,Ref<Shape> > collision_map;
	ConfirmationDialog *error_dialog;

	void _choose_file(const String& p_path);
	void _choose_save_file(const String& p_path);
	void _choose_script(const String& p_path);
	void _browse();
	void _browse_target();
	void _browse_script();
	void _import();


protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	Error import(const String& p_from, const String& p_to, const String& p_preset);
	void popup_import(const String& p_from);
	EditorSceneImportDialog(EditorNode *p_editor,EditorSceneImportPlugin *p_plugin);
};

///////////////////////////////////


static const char *anim_flag_names[]={
	"Detect Loop",
	"Keep Value Tracks",
	"Optimize",
	NULL
};

static const char *anim_flag_descript[]={
	"Set loop flag for animation names that\ncontain 'cycle' or 'loop' in the name.",
	"When merging an existing aimation,\nkeep the user-created value-tracks.",
	"Remove redundant keyframes in\n transform tacks.",
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


void EditorImportAnimationOptions::_bind_methods() {

	ObjectTypeDB::bind_method("_changed",&EditorImportAnimationOptions::_changed);
//	ObjectTypeDB::bind_method("_changedp",&EditorImportAnimationOptions::_changedp);

	ADD_SIGNAL(MethodInfo("changed"));
}


void EditorImportAnimationOptions::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_SCENE) {

		flags->connect("item_edited",this,"_changed");
//		format->connect("item_selected",this,"_changedp");
	}
}

EditorImportAnimationOptions::EditorImportAnimationOptions() {


	updating=false;

	flags = memnew( Tree );
	flags->set_hide_root(true);
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


	add_margin_child(_TR("Animation Options"),flags,true);

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
		//save_path->set_text(p_path.get_file().basename()+".scn");
#if 0
	}
#endif

	import_path->set_text(p_path);

}
void EditorSceneImportDialog::_choose_save_file(const String& p_path) {

	save_path->set_text(p_path);
}

void EditorSceneImportDialog::_choose_script(const String& p_path) {

	String p = Globals::get_singleton()->localize_path(p_path);
	if (!p.is_resource_file())
		p=Globals::get_singleton()->get_resource_path().path_to(p_path.get_base_dir())+p_path.get_file();
	script_path->set_text(p);

}

void EditorSceneImportDialog::_import() {

//'	ImportMonitorBlock imb;

	if (import_path->get_text()=="") {
		error_dialog->set_text(_TR("Source path is empty."));
		error_dialog->popup_centered(Size2(200,100));
		return;
	}

	if (save_path->get_text()=="") {
		error_dialog->set_text(_TR("Target path is empty."));
		error_dialog->popup_centered(Size2(200,100));
		return;
	}

	String dst_path;

	if (texture_action->get_selected()==0)
		dst_path=save_path->get_text();//.get_base_dir();
	else
		dst_path=Globals::get_singleton()->get("import/shared_textures");

	uint32_t flags=0;

	for(int i=0;i<scene_flags.size();i++) {

		if (scene_flags[i]->is_checked(0))
			flags|=(1<<i);
	}


	Ref<EditorScenePostImport> pi;

	if (script_path->get_text()!="") {
		Ref<Script> scr = ResourceLoader::load(script_path->get_text());
		if (!scr.is_valid()) {
			error_dialog->set_text(_TR("Couldn't load Post-Import Script."));
			error_dialog->popup_centered(Size2(200,100));
			return;
		}

		pi = Ref<EditorScenePostImport>( memnew( EditorScenePostImport ) );
		pi->set_script(scr.get_ref_ptr());
		if (!pi->get_script_instance()) {

			error_dialog->set_text(_TR("Invalid/Broken Script for Post-Import."));
			error_dialog->popup_centered(Size2(200,100));
			return;
		}
	}


	String save_file = save_path->get_text().plus_file(import_path->get_text().get_file().basename()+".scn");
	print_line("Saving to: "+save_file);



	Node *scene=NULL;


	Ref<ResourceImportMetadata> rim = memnew( ResourceImportMetadata );

	rim->add_source(EditorImportPlugin::validate_source_path(import_path->get_text()));
	rim->set_option("flags",flags);
	rim->set_option("texture_flags",texture_options->get_flags());
	rim->set_option("texture_format",texture_options->get_format());
	rim->set_option("texture_quality",texture_options->get_quality());
	rim->set_option("animation_flags",animation_options->get_flags());
	rim->set_option("post_import_script",script_path->get_text()!=String()?EditorImportPlugin::validate_source_path(script_path->get_text()):String());
	rim->set_option("reimport",true);

	Error err = plugin->import(save_file,rim);

	if (err) {

		error_dialog->set_text(_TR("Error importing scene."));
		error_dialog->popup_centered(Size2(200,100));
		return;
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

void EditorSceneImportDialog::_browse() {

	file_select->popup_centered_ratio();
}

void EditorSceneImportDialog::_browse_target() {

	if (save_path->get_text()!="")
		save_select->set_current_path(save_path->get_text());
	save_select->popup_centered_ratio();

}

void EditorSceneImportDialog::_browse_script() {

	script_select->popup_centered_ratio();

}

void EditorSceneImportDialog::popup_import(const String &p_from) {

	popup_centered(Size2(700,500));
	if (p_from!="") {
		Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_from);
		if (rimd.is_null())
			return;

		int flags = rimd->get_option("flags");

		for(int i=0;i<scene_flags.size();i++) {

			scene_flags[i]->set_checked(0,flags&(1<<i));
		}

		texture_options->set_flags(rimd->get_option("texture_flags"));
		texture_options->set_format(EditorTextureImportPlugin::ImageFormat(int(rimd->get_option("texture_format"))));
		texture_options->set_quality(rimd->get_option("texture_quality"));
		animation_options->set_flags(rimd->get_option("animation_flags"));
		script_path->set_text(rimd->get_option("post_import_script"));

		save_path->set_text(p_from.get_base_dir());
		import_path->set_text(EditorImportPlugin::expand_source_path(rimd->get_source_path(0)));

	}
}


void EditorSceneImportDialog::_notification(int p_what) {


	if (p_what==NOTIFICATION_ENTER_SCENE) {


		List<String> extensions;
		file_select->clear_filters();

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

void EditorSceneImportDialog::_bind_methods() {


	ObjectTypeDB::bind_method("_choose_file",&EditorSceneImportDialog::_choose_file);
	ObjectTypeDB::bind_method("_choose_save_file",&EditorSceneImportDialog::_choose_save_file);
	ObjectTypeDB::bind_method("_choose_script",&EditorSceneImportDialog::_choose_script);
	ObjectTypeDB::bind_method("_import",&EditorSceneImportDialog::_import);
	ObjectTypeDB::bind_method("_browse",&EditorSceneImportDialog::_browse);
	ObjectTypeDB::bind_method("_browse_target",&EditorSceneImportDialog::_browse_target);
	ObjectTypeDB::bind_method("_browse_script",&EditorSceneImportDialog::_browse_script);
	ADD_SIGNAL( MethodInfo("imported",PropertyInfo(Variant::OBJECT,"scene")) );
}



static const char *scene_flag_names[]={
	"Create Collisions (-col,-colonly)",
	"Create Portals (-portal)",
	"Create Rooms (-room)",
	"Simplify Rooms",
	"Create Billboards (-bb)",
	"Create Impostors (-imp:dist)",
	"Create LODs (-lod:dist)",
	"Remove Nodes (-noimp)",
	"Import Animations",
	"Compress Geometry",
	"Fail on Missing Images",
	"Force Generation of Tangent Arrays",
	NULL
};


EditorSceneImportDialog::EditorSceneImportDialog(EditorNode *p_editor, EditorSceneImportPlugin *p_plugin) {


	editor=p_editor;
	plugin=p_plugin;

	set_title(_TR("Import 3D Scene"));
	HBoxContainer *import_hb = memnew( HBoxContainer );
	add_child(import_hb);
	set_child_rect(import_hb);

	VBoxContainer *vbc = memnew( VBoxContainer );
	import_hb->add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_margin_child(_TR("Source Scene:"),hbc);

	import_path = memnew( LineEdit );
	import_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(import_path);

	Button * import_choose = memnew( Button );
	import_choose->set_text(" .. ");
	hbc->add_child(import_choose);

	import_choose->connect("pressed", this,"_browse");

	hbc = memnew( HBoxContainer );
	vbc->add_margin_child(_TR("Target Scene:"),hbc);

	save_path = memnew( LineEdit );
	save_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(save_path);

	Button * save_choose = memnew( Button );
	save_choose->set_text(" .. ");
	hbc->add_child(save_choose);

	save_choose->connect("pressed", this,"_browse_target");

	texture_action = memnew( OptionButton );
	texture_action->add_item(_TR("Same as Target Scene"));
	texture_action->add_item(_TR("Shared"));
	texture_action->select(0);
	vbc->add_margin_child(_TR("Target Texture Folder:"),texture_action);

	import_options = memnew( Tree );
	vbc->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_margin_child(_TR("Options:"),import_options,true);

	file_select = memnew(FileDialog);
	file_select->set_access(FileDialog::ACCESS_FILESYSTEM);
	add_child(file_select);


	file_select->set_mode(FileDialog::MODE_OPEN_FILE);

	file_select->connect("file_selected", this,"_choose_file");

	save_select = memnew(EditorDirDialog);
	add_child(save_select);

	//save_select->set_mode(FileDialog::MODE_SAVE_FILE);
	save_select->connect("dir_selected", this,"_choose_save_file");

	get_ok()->connect("pressed", this,"_import");
	get_ok()->set_text(_TR("Import"));

	TreeItem *root = import_options->create_item(NULL);
	import_options->set_hide_root(true);




	TreeItem *importopts = import_options->create_item(root);
	importopts->set_text(0,_TR("Import:"));

	const char ** fn=scene_flag_names;

	while(*fn) {

		TreeItem *opt = import_options->create_item(importopts);
		opt->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		opt->set_checked(0,true);
		opt->set_editable(0,true);
		opt->set_text(0,*fn);
		scene_flags.push_back(opt);
		fn++;
	}

	hbc = memnew( HBoxContainer );
	vbc->add_margin_child(_TR("Post-Process Script:"),hbc);

	script_path = memnew( LineEdit );
	script_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(script_path);

	Button * script_choose = memnew( Button );
	script_choose->set_text(" .. ");
	hbc->add_child(script_choose);

	script_choose->connect("pressed", this,"_browse_script");

	script_select = memnew(FileDialog);
	add_child(script_select);
	for(int i=0;i<ScriptServer::get_language_count();i++) {

		ScriptLanguage *sl=ScriptServer::get_language(i);
		String ext = sl->get_extension();
		if (ext=="")
			continue;
		script_select->add_filter("*."+ext+" ; "+sl->get_name());
	}


	script_select->set_mode(FileDialog::MODE_OPEN_FILE);

	script_select->connect("file_selected", this,"_choose_script");

	error_dialog = memnew ( ConfirmationDialog );
	add_child(error_dialog);
	error_dialog->get_ok()->set_text(_TR("Accept"));
//	error_dialog->get_cancel()->hide();

	set_hide_on_ok(false);

	GLOBAL_DEF("import/shared_textures","res://");
	Globals::get_singleton()->set_custom_property_info("import/shared_textures",PropertyInfo(Variant::STRING,"import/shared_textures",PROPERTY_HINT_DIR));

	import_hb->add_constant_override("separation",30);

	VBoxContainer *ovb = memnew( VBoxContainer);
	ovb->set_h_size_flags(SIZE_EXPAND_FILL);
	import_hb->add_child(ovb);

	texture_options = memnew( EditorImportTextureOptions );
	ovb->add_child(texture_options);
	texture_options->set_v_size_flags(SIZE_EXPAND_FILL);
	//animation_options->set_flags(EditorImport::
	texture_options->set_format(EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM);

	animation_options = memnew( EditorImportAnimationOptions );
	ovb->add_child(animation_options);
	animation_options->set_v_size_flags(SIZE_EXPAND_FILL);
	animation_options->set_flags(EditorSceneAnimationImportPlugin::ANIMATION_DETECT_LOOP|EditorSceneAnimationImportPlugin::ANIMATION_KEEP_VALUE_TRACKS|EditorSceneAnimationImportPlugin::ANIMATION_OPTIMIZE);

	//texture_options->set_format(EditorImport::IMAGE_FORMAT_C);

}



////////////////////////////////



String EditorSceneImportPlugin::get_name() const {

	return "scene_3d";
}

String EditorSceneImportPlugin::get_visible_name() const{

	return _TR("3D Scene");
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
	return false;
}

static String _fixstr(const String& p_what,const String& p_str) {

	if (p_what.findn("$"+p_str)!=-1) //blender and other stuff
		return p_what.replace("$"+p_str,"");
	if (p_what.to_lower().ends_with("-"+p_str)) //collada only supports "_" and "-" besides letters
		return p_what.substr(0,p_what.length()-(p_str.length()+1));
	return p_what;
}



void EditorSceneImportPlugin::_find_resources(const Variant& p_var,Set<Ref<ImageTexture> >& image_map) {


	switch(p_var.get_type()) {

		case Variant::OBJECT: {

			Ref<Resource> res = p_var;
			if (res.is_valid()) {

				if (res->is_type("Texture")) {

					image_map.insert(res);


				} else {


					List<PropertyInfo> pl;
					res->get_property_list(&pl);
					for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

						if (E->get().type==Variant::OBJECT || E->get().type==Variant::ARRAY || E->get().type==Variant::DICTIONARY) {
							_find_resources(res->get(E->get().name),image_map);
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


				_find_resources(E->get(),image_map);
				_find_resources(d[E->get()],image_map);

			}


		} break;
		case Variant::ARRAY: {

			Array a = p_var;
			for(int i=0;i<a.size();i++) {

				_find_resources(a[i],image_map);
			}

		} break;

	}

}


Node* EditorSceneImportPlugin::_fix_node(Node *p_node,Node *p_root,Map<Ref<Mesh>,Ref<Shape> > &collision_map,uint32_t p_flags,Set<Ref<ImageTexture> >& image_map) {

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
				_find_resources(p_node->get(E->get().name),image_map);
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

					Ref<FixedMaterial> fm = m->surface_get_material(i);
					if (fm.is_valid()) {
						fm->set_flag(Material::FLAG_UNSHADED,true);
						fm->set_flag(Material::FLAG_DOUBLE_SIDED,true);
						fm->set_hint(Material::HINT_NO_DEPTH_DRAW,true);
						fm->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA,true);
					}
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
					mi->set_draw_range_begin(dist);
					mi->set_draw_range_end(100000);

					mip->set_draw_range_begin(0);
					mip->set_draw_range_end(dist);

					if (mi->get_mesh().is_valid()) {

						Ref<Mesh> m = mi->get_mesh();
						for(int i=0;i<m->get_surface_count();i++) {

							Ref<FixedMaterial> fm = m->surface_get_material(i);
							if (fm.is_valid()) {
								fm->set_flag(Material::FLAG_UNSHADED,true);
								fm->set_flag(Material::FLAG_DOUBLE_SIDED,true);
								fm->set_hint(Material::HINT_NO_DEPTH_DRAW,true);
								fm->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA,true);
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
		    mi->set_draw_range_begin(dist);
		    mi->set_draw_range_end(100000);

		    mip->set_draw_range_begin(0);
		    mip->set_draw_range_end(dist);

		    /*if (mi->get_mesh().is_valid()) {

			Ref<Mesh> m = mi->get_mesh();
			for(int i=0;i<m->get_surface_count();i++) {

			    Ref<FixedMaterial> fm = m->surface_get_material(i);
			    if (fm.is_valid()) {
				fm->set_flag(Material::FLAG_UNSHADED,true);
				fm->set_flag(Material::FLAG_DOUBLE_SIDED,true);
				fm->set_hint(Material::HINT_NO_DEPTH_DRAW,true);
				fm->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA,true);
			    }
			}
		    }*/
		}
	    }
	}
    }
	if (p_flags&SCENE_FLAG_CREATE_COLLISIONS && _teststr(name,"colonly") && p_node->cast_to<MeshInstance>()) {

		if (isroot)
			return p_node;

		MeshInstance *mi = p_node->cast_to<MeshInstance>();
		Node * col = mi->create_trimesh_collision_node();

		ERR_FAIL_COND_V(!col,NULL);
		col->set_name(_fixstr(name,"colonly"));
		col->cast_to<Spatial>()->set_transform(mi->get_transform());
		p_node->replace_by(col);
		memdelete(p_node);
		p_node=col;

	} else if (p_flags&SCENE_FLAG_CREATE_COLLISIONS &&_teststr(name,"col") && p_node->cast_to<MeshInstance>()) {


		MeshInstance *mi = p_node->cast_to<MeshInstance>();

		mi->set_name(_fixstr(name,"col"));
		mi->create_trimesh_collision();
	} else if (p_flags&SCENE_FLAG_CREATE_ROOMS && _teststr(name,"room") && p_node->cast_to<MeshInstance>()) {


		if (isroot)
			return p_node;

		MeshInstance *mi = p_node->cast_to<MeshInstance>();
		DVector<Face3> faces = mi->get_faces(VisualInstance::FACES_SOLID);


		BSP_Tree bsptree(faces);

		Ref<RoomBounds> area = memnew( RoomBounds );
		area->set_bounds(faces);
		area->set_geometry_hint(faces);


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

		room->compute_room_from_subtree();

	} else if (p_flags&SCENE_FLAG_CREATE_PORTALS &&_teststr(name,"portal") && p_node->cast_to<MeshInstance>()) {

		if (isroot)
			return p_node;

		MeshInstance *mi = p_node->cast_to<MeshInstance>();
		DVector<Face3> faces = mi->get_faces(VisualInstance::FACES_SOLID);

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

				float a = portal_points[i].atan2();
				float b = portal_points[i+1].atan2();

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

		}

	}




	return p_node;
}


void EditorSceneImportPlugin::_merge_node(Node *p_node,Node*p_root,Node *p_existing,Set<Ref<Resource> >& checked_resources) {


	NodePath path = p_root->get_path_to(p_node);

	bool valid=false;

	if (p_existing->has_node(path)) {

		Node *existing = p_existing->get_node(path);

		if (existing->get_type()==p_node->get_type()) {
			//same thing, check what it is

			if (existing->get_type()=="MeshInstance") {

				//merge mesh instance, this is a special case!
				MeshInstance *mi_existing=existing->cast_to<MeshInstance>();
				MeshInstance *mi_node=p_node->cast_to<MeshInstance>();

				Ref<Mesh> mesh_existing = mi_existing->get_mesh();
				Ref<Mesh> mesh_node = mi_node->get_mesh();

				if (mesh_existing.is_null() || checked_resources.has(mesh_node)) {

					if (mesh_node.is_valid())
						mi_existing->set_mesh(mesh_node);
				} else if (mesh_node.is_valid()) {

					//mesh will always be overwritten, so check materials from original

					for(int i=0;i<mesh_node->get_surface_count();i++) {

						String name = mesh_node->surface_get_name(i);

						if (name!="") {

							for(int j=0;j<mesh_existing->get_surface_count();j++) {

								Ref<Material> keep;

								if (name==mesh_existing->surface_get_name(j)) {

									Ref<Material> mat = mesh_existing->surface_get_material(j);

									if (mat.is_valid()) {
										if (mat->get_path()!="" && mat->get_path().begins_with("res://") && mat->get_path().find("::")==-1) {
											keep=mat; //mat was loaded from file
										} else if (mat->is_edited()) {
											keep=mat; //mat was edited
										}
									}
									break;
								}
								if (keep.is_valid())
									mesh_node->surface_set_material(i,keep); //kept
							}
						}
					}

					mi_existing->set_mesh(mesh_node); //always overwrite mesh
					checked_resources.insert(mesh_node);

				}
			} else if (existing->get_type()=="Path") {

				Path *path_existing =existing->cast_to<Path>();
				Path *path_node =p_node->cast_to<Path>();

				if (path_node->get_curve().is_valid()) {

					if (!path_existing->get_curve().is_valid() || !path_existing->get_curve()->is_edited()) {
						path_existing->set_curve(path_node->get_curve());
					}
				}
			}
		}

		valid=true;
	} else {

		if (p_node!=p_root && p_existing->has_node(p_root->get_path_to(p_node->get_parent()))) {

			Node *parent = p_existing->get_node(p_root->get_path_to(p_node->get_parent()));
			NodePath path = p_root->get_path_to(p_node);

			//add it.. because not existing in existing scene
			Object *o = ObjectTypeDB::instance(p_existing->get_type());
			Node *n=NULL;
			if (o)
				n=o->cast_to<Node>();

			if (n) {

				List<PropertyInfo> pl;
				p_existing->get_property_list(&pl);
				for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {
					if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
						continue;
					n->set( E->get().name, p_existing->get(E->get().name) );
				}

				parent->add_child(n);

				valid=true;
			}
		}

	}


	if (valid) {

		for(int i=0;i<p_node->get_child_count();i++) {
			_merge_node(p_node->get_child(i),p_root,p_existing,checked_resources);
		}
	}

}


void EditorSceneImportPlugin::_merge_scenes(Node *p_existing,Node *p_new) {

	Set<Ref<Resource> > checked_resources;
	_merge_node(p_new,p_new,p_existing,checked_resources);

}

#if 0

Error EditorImport::import_scene(const String& p_path,const String& p_dest_path,const String& p_resource_path,uint32_t p_flags,ImageFormat p_image_format,ImageCompression p_image_compression,uint32_t p_image_flags,float p_quality,uint32_t animation_flags,Node **r_scene,Ref<EditorPostImport> p_post_import) {


}
#endif

Error EditorSceneImportPlugin::import(const String& p_dest_path, const Ref<ResourceImportMetadata>& p_from){

	Ref<ResourceImportMetadata> from=p_from;

	ERR_FAIL_COND_V(from->get_source_count()!=1,ERR_INVALID_PARAMETER);

	String src_path=EditorImportPlugin::expand_source_path(from->get_source_path(0));

	Ref<EditorSceneImporter> importer;
	String ext=src_path.extension().to_lower();


	EditorNode::progress_add_task("import",_TR("Import Scene"),104);
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

	if (!importer.is_valid()) {
		EditorNode::progress_end_task("import");
	}
	ERR_FAIL_COND_V(!importer.is_valid(),ERR_FILE_UNRECOGNIZED);

	int animation_flags=p_from->get_option("animation_flags");
	int scene_flags = from->get_option("flags");

	uint32_t import_flags=0;
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_DETECT_LOOP)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION_DETECT_LOOP;
	if (animation_flags&EditorSceneAnimationImportPlugin::ANIMATION_OPTIMIZE)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION_OPTIMIZE;		
	if (scene_flags&SCENE_FLAG_IMPORT_ANIMATIONS)
		import_flags|=EditorSceneImporter::IMPORT_ANIMATION;
	if (scene_flags&SCENE_FLAG_FAIL_ON_MISSING_IMAGES)
		import_flags|=EditorSceneImporter::IMPORT_FAIL_ON_MISSING_DEPENDENCIES;
	if (scene_flags&SCENE_FLAG_GENERATE_TANGENT_ARRAYS)
		import_flags|=EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS;



	EditorNode::progress_task_step("import",_TR("Importing Scene.."),0);


	Error err=OK;
	Node *scene = importer->import_scene(src_path,import_flags,&err);
	if (!scene || err!=OK) {
		EditorNode::progress_end_task("import");
		return err;
	}


	bool merge = !bool(from->get_option("reimport"));
	from->set_source_md5(0,FileAccess::get_md5(src_path));
	from->set_editor(get_name());

	from->set_option("reimport",false);
	String target_res_path=p_dest_path.get_base_dir();

	Map<Ref<Mesh>,Ref<Shape> > collision_map;

	Ref<ResourceImportMetadata> imd = memnew(ResourceImportMetadata);

	Set< Ref<ImageTexture> > imagemap;
	EditorNode::progress_task_step("import",_TR("Post-Processing Scene.."),1);



	scene=_fix_node(scene,scene,collision_map,scene_flags,imagemap);


	/// BEFORE ANYTHING, RUN SCRIPT

	EditorNode::progress_task_step("import",_TR("Running Custom Script.."),2);

	String post_import_script_path = from->get_option("post_import_script");
	Ref<EditorScenePostImport>  post_import_script;

	if (post_import_script_path!="") {
		post_import_script_path = EditorImportPlugin::expand_source_path(post_import_script_path);
		Ref<Script> scr = ResourceLoader::load(post_import_script_path);
		if (!scr.is_valid()) {
			EditorNode::add_io_error(_TR("Couldn't load post-import script: '")+post_import_script_path);
		} else {

			post_import_script = Ref<EditorScenePostImport>( memnew( EditorScenePostImport ) );
			post_import_script->set_script(scr.get_ref_ptr());
			if (!post_import_script->get_script_instance()) {
				EditorNode::add_io_error(_TR("Invalid/Broken Script for Post-Import: '")+post_import_script_path);
				post_import_script.unref();
			}
		}
	}


	if (post_import_script.is_valid()) {
		err = post_import_script->post_import(scene);
		if (err) {
			EditorNode::add_io_error(_TR("Error running Post-Import script: '")+post_import_script_path);
			EditorNode::progress_end_task("import");
			return err;
		}
	}

	/// IMPORT IMAGES


	int idx=0;

	int image_format = from->get_option("texture_format");
	int image_flags =  from->get_option("texture_flags");
	float image_quality = from->get_option("texture_quality");

	for (Set< Ref<ImageTexture> >::Element *E=imagemap.front();E;E=E->next()) {

		//texture could be converted to something more useful for 3D, that could load individual mipmaps and stuff
		//but not yet..

		Ref<ImageTexture> texture = E->get();

		ERR_CONTINUE(!texture.is_valid());

		String path = texture->get_path();
		String fname= path.get_file();
		String target_path = Globals::get_singleton()->localize_path(target_res_path.plus_file(fname));
		EditorNode::progress_task_step("import",_TR("Import Img: ")+fname,3+(idx)*100/imagemap.size());

		idx++;

		if (path==target_path) {

			EditorNode::add_io_error(_TR("Can't import a file over itself: '")+target_path);
			continue;
		}

		if (!target_path.begins_with("res://")) {
			EditorNode::add_io_error(_TR("Couldn't localize path: '")+target_path+"' (already local)");
			continue;
		}


		{


			target_path=target_path.basename()+".tex";

			if (FileAccess::exists(target_path)) {
				texture->set_path(target_path);
				continue; //already imported
			}
			Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
			imd->set_option("flags",image_flags);
			imd->set_option("format",image_format);
			imd->set_option("quality",image_quality);
			imd->set_option("atlas",false);
			imd->add_source(EditorImportPlugin::validate_source_path(path));

			Error err = EditorTextureImportPlugin::get_singleton(EditorTextureImportPlugin::MODE_TEXTURE_3D)->import(target_path,imd);

		}
	}


	/// BEFORE SAVING - MERGE


	if (merge) {

		EditorNode::progress_task_step("import",_TR("Merging.."),103);

		FileAccess *fa = FileAccess::create(FileAccess::ACCESS_FILESYSTEM);
		if (fa->file_exists(p_dest_path)) {

			//try to merge

			Ref<PackedScene> s = ResourceLoader::load(p_dest_path);
			if (s.is_valid()) {

				Node *existing = s->instance(true);

				if (existing) {

					_merge_scenes(scene,existing);

					memdelete(scene);
					scene=existing;
				}
			}

		}

		memdelete(fa);
	}


	EditorNode::progress_task_step("import",_TR("Saving.."),104);

	Ref<PackedScene> packer = memnew( PackedScene );
	packer->pack(scene);
	packer->set_path(p_dest_path);
	packer->set_import_metadata(from);

	print_line("SAVING TO: "+p_dest_path);
	err = ResourceSaver::save(p_dest_path,packer);

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
	EditorNode::progress_end_task("import");

	return err;

}

void EditorSceneImportPlugin::add_importer(const Ref<EditorSceneImporter>& p_importer) {

	importers.push_back(p_importer);
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

	return _TR("3D Scene Animation");
}
void EditorSceneAnimationImportPlugin::import_dialog(const String& p_from){


}
Error EditorSceneAnimationImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){

	return OK;
}

EditorSceneAnimationImportPlugin::EditorSceneAnimationImportPlugin(EditorNode* p_editor) {


}

