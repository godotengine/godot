/*************************************************************************/
/*  editor_settings.cpp                                                  */
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
#include "editor_settings.h"
#include "os/os.h"
#include "os/dir_access.h"
#include "os/file_access.h"

#include "version.h"
#include "scene/main/scene_main_loop.h"
#include "os/os.h"
#include "scene/main/node.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "scene/main/viewport.h"
#include "io/config_file.h"
#include "editor_node.h"
#include "globals.h"

Ref<EditorSettings> EditorSettings::singleton=NULL;

EditorSettings *EditorSettings::get_singleton() {

	return singleton.ptr();
}


bool EditorSettings::_set(const StringName& p_name, const Variant& p_value) {

	_THREAD_SAFE_METHOD_
	if (p_value.get_type()==Variant::NIL)
		props.erase(p_name);
	else {

		if (props.has(p_name))
			props[p_name].variant=p_value;
		else
			props[p_name]=VariantContainer(p_value,last_order++);
	}

	emit_signal("settings_changed");
	return true;
}
bool EditorSettings::_get(const StringName& p_name,Variant &r_ret) const {

	_THREAD_SAFE_METHOD_

	const VariantContainer *v=props.getptr(p_name);
	if (!v)
		return false;
	r_ret = v->variant;
	return true;
}

struct _EVCSort {

	String name;
	Variant::Type type;
	int order;

	bool operator<(const _EVCSort& p_vcs) const{ return order< p_vcs.order; }
};

void EditorSettings::_get_property_list(List<PropertyInfo> *p_list) const {

	_THREAD_SAFE_METHOD_

	const String *k=NULL;
	Set<_EVCSort> vclist;

	while ((k=props.next(k))) {

		const VariantContainer *v=props.getptr(*k);

		if (v->hide_from_editor)
			continue;

		_EVCSort vc;
		vc.name=*k;
		vc.order=v->order;
		vc.type=v->variant.get_type();

		vclist.insert(vc);
	}

	for(Set<_EVCSort>::Element *E=vclist.front();E;E=E->next()) {

		int pinfo = PROPERTY_USAGE_STORAGE;
		if (!E->get().name.begins_with("_"))
			pinfo|=PROPERTY_USAGE_EDITOR;

		PropertyInfo pi(E->get().type, E->get().name);
		pi.usage=pinfo;
		if (hints.has(E->get().name))
			pi=hints[E->get().name];


		p_list->push_back( pi );
	}
}

bool EditorSettings::has(String p_var) const {

	_THREAD_SAFE_METHOD_

	return props.has(p_var);
}

void EditorSettings::erase(String p_var) {

	_THREAD_SAFE_METHOD_

	props.erase(p_var);
}

void EditorSettings::raise_order(const String& p_name) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!props.has(p_name));
	props[p_name].order=++last_order;


}


Variant _EDITOR_DEF( const String& p_var, const Variant& p_default) {

	if (EditorSettings::get_singleton()->has(p_var))
		return EditorSettings::get_singleton()->get(p_var);
	EditorSettings::get_singleton()->set(p_var,p_default);
	return p_default;

}


void EditorSettings::create() {


	if (singleton.ptr())
		return; //pointless

	DirAccess *dir=NULL;
	Object *object;
	Variant meta;

	String config_path;
	String config_dir;
	String config_file="editor_settings.xml";

	if (OS::get_singleton()->has_environment("APPDATA")) {
		// Most likely under windows, save here
		config_path=OS::get_singleton()->get_environment("APPDATA");
		config_dir=String(_MKSTR(VERSION_SHORT_NAME)).capitalize();
	} else if (OS::get_singleton()->has_environment("HOME")) {

		config_path=OS::get_singleton()->get_environment("HOME");
		config_dir="."+String(_MKSTR(VERSION_SHORT_NAME)).to_lower();
	}

	ObjectTypeDB::register_type<EditorSettings>(); //otherwise it can't be unserialized
	String config_file_path;

	if (config_path!=""){

		dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (dir->change_dir(config_path)!=OK) {
			ERR_PRINT("Cannot find path for config directory!");
			memdelete(dir);
			goto fail;
		}

		if (dir->change_dir(config_dir)!=OK) {
			dir->make_dir(config_dir);
			if (dir->change_dir(config_dir)!=OK) {
				ERR_PRINT("Cannot create config directory!");
				memdelete(dir);
				goto fail;
			}
		}

		if (dir->change_dir("templates")!=OK) {
			dir->make_dir("templates");
		} else {

			dir->change_dir("..");
		}

		if (dir->change_dir("tmp")!=OK) {
			dir->make_dir("tmp");
		} else {

			dir->change_dir("..");
		}

		if (dir->change_dir("plugins")!=OK) {
			dir->make_dir("plugins");
		} else {

			dir->change_dir("..");
		}

		if (dir->change_dir("config")!=OK) {
			dir->make_dir("config");
		} else {

			dir->change_dir("..");
		}

		dir->change_dir("config");

		String pcp=Globals::get_singleton()->get_resource_path();
		if (pcp.ends_with("/"))
			pcp=config_path.substr(0,pcp.size()-1);
		pcp=pcp.get_file()+"-"+pcp.md5_text();

		if (dir->change_dir(pcp)) {
			dir->make_dir(pcp);
		} else {
			dir->change_dir("..");
		}

		dir->change_dir("..");

		// path at least is validated, so validate config file


		config_file_path = config_path+"/"+config_dir+"/"+config_file;

		if (!dir->file_exists(config_file)) {
			memdelete(dir);
			WARN_PRINT("Config file does not exist, creating.")
			goto fail;
		}

		memdelete(dir);

		singleton = ResourceLoader::load(config_file_path,"EditorSettings");
		if (singleton.is_null()) {
			WARN_PRINT("Could not open config file.");
			goto fail;
		}

		singleton->config_file_path=config_file_path;
		singleton->project_config_path=pcp;
		singleton->settings_path=config_path+"/"+config_dir;

		if (OS::get_singleton()->is_stdout_verbose()) {

			print_line("EditorSettings: Load OK!");
		}

		singleton->setup_network();
		singleton->load_favorites();
		singleton->scan_plugins();

		return;

	}



	fail:

	singleton = Ref<EditorSettings>( memnew( EditorSettings ) );
	singleton->config_file_path=config_file_path;
	singleton->settings_path=config_path+"/"+config_dir;
	singleton->_load_defaults();
	singleton->setup_network();
	singleton->scan_plugins();


}

String EditorSettings::get_settings_path() const {

	return settings_path;
}


Error EditorSettings::_load_plugin(const String& p_path, Plugin &plugin) {

	if (!FileAccess::exists(p_path))
		return ERR_FILE_NOT_FOUND;

	Ref<ConfigFile> cf = memnew(ConfigFile);
	Error err = cf->load(p_path);
	ERR_EXPLAIN("Error loading plugin description for: "+p_path);
	ERR_FAIL_COND_V(err!=OK,ERR_CANT_OPEN);

	plugin.instance=NULL;
	ERR_FAIL_COND_V(!cf->has_section_key("plugin","name"),ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!cf->has_section_key("plugin","installs"),ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!cf->has_section_key("plugin","author"),ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!cf->has_section_key("plugin","version"),ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!cf->has_section_key("plugin","script"),ERR_INVALID_DATA);
	plugin.name=cf->get_value("plugin","name");
	plugin.author=cf->get_value("plugin","author");
	plugin.version=cf->get_value("plugin","version");
	plugin.script=cf->get_value("plugin","script");

	if (cf->has_section_key("plugin","description"))
		plugin.description=cf->get_value("plugin","description");
	plugin.installs=cf->get_value("plugin","installs");
	if (cf->has_section_key("plugin","install_files"))
		plugin.install_files=cf->get_value("plugin","install_files");

	return OK;
}

void EditorSettings::setup_network() {

	List<IP_Address> local_ip;
	IP::get_singleton()->get_local_addresses(&local_ip);
	String lip;
	String hint;
	String current=get("network/debug_host");

	for(List<IP_Address>::Element *E=local_ip.front();E;E=E->next()) {

		String ip = E->get();
		if (ip=="127.0.0.1")
			continue;

		if (lip!="")
			lip=ip;
		if (ip==current)
			lip=current; //so it saves
		if (hint!="")
			hint+=",";
		hint+=ip;

	}

	set("network/debug_host",lip);
	add_property_hint(PropertyInfo(Variant::STRING,"network/debug_host",PROPERTY_HINT_ENUM,hint));

}

void EditorSettings::scan_plugins() {

	Map<String,Plugin> new_plugins;

	new_plugins.clear();
	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->change_dir(get_settings_path().plus_file("plugins"));
	if (err!=OK) {
		memdelete(d);
		ERR_EXPLAIN("Plugin dir does not exist!")
		ERR_FAIL_COND(err!=OK);
	}
	d->list_dir_begin();

	String base = d->get_current_dir();
	//print_line("list diring on: "+base);
	while(true) {
		String p = d->get_next();
		if (p=="")
			break;
		if (!d->current_is_dir() || p.begins_with("."))
			continue;

		String cfpath=d->get_current_dir().plus_file(p+"/plugin.cfg");

		Plugin plugin;
		Error err = _load_plugin(cfpath,plugin);
		ERR_CONTINUE(err!=OK);

		if (plugins.has(p))
			plugin.instance=plugins[p].instance;

		new_plugins[p]=plugin;
	}


	plugins=new_plugins;

	memdelete(d);
}

void EditorSettings::save() {

	//_THREAD_SAFE_METHOD_

	if (!singleton.ptr())
		return;

	if (singleton->config_file_path=="") {
		ERR_PRINT("Cannot save EditorSettings config, no valid path");
		return;
	}

	Error err = ResourceSaver::save(singleton->config_file_path,singleton);

	if (err!=OK) {
		ERR_PRINT("Can't Save!");
		return;
	}

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("EditorSettings Save OK!");
	}

}

void EditorSettings::destroy() {

	if (!singleton.ptr())
		return;
	save();
	singleton=Ref<EditorSettings>();
}

void EditorSettings::_load_defaults() {

	_THREAD_SAFE_METHOD_

	set("global/font","");
	hints["global/font"]=PropertyInfo(Variant::STRING,"global/font",PROPERTY_HINT_GLOBAL_FILE,"*.fnt");
	set("global/autoscan_project_path","");
	hints["global/autoscan_project_path"]=PropertyInfo(Variant::STRING,"global/autoscan_project_path",PROPERTY_HINT_GLOBAL_DIR);
	set("global/default_project_path","");
	hints["global/default_project_path"]=PropertyInfo(Variant::STRING,"global/default_project_path",PROPERTY_HINT_GLOBAL_DIR);
	set("global/default_project_export_path","");
	hints["global/default_project_export_path"]=PropertyInfo(Variant::STRING,"global/default_project_export_path",PROPERTY_HINT_GLOBAL_DIR);

	set("text_editor/background_color",Color::html("3b000000"));
	set("text_editor/text_color",Color::html("aaaaaa"));
	set("text_editor/text_selected_color",Color::html("000000"));
	set("text_editor/keyword_color",Color::html("ffffb3"));
	set("text_editor/base_type_color",Color::html("a4ffd4"));
	set("text_editor/engine_type_color",Color::html("83d3ff"));
	set("text_editor/comment_color",Color::html("983d1b"));
	set("text_editor/string_color",Color::html("ef6ebe"));
	set("text_editor/symbol_color",Color::html("badfff"));
	set("text_editor/selection_color",Color::html("7b5dbe"));
	set("text_editor/brace_mismatch_color",Color(1,0.2,0.2));
	set("text_editor/current_line_color",Color(0.3,0.5,0.8,0.15));

	set("text_editor/idle_parse_delay",2);
	set("text_editor/create_signal_callbacks",true);
	set("text_editor/autosave_interval_secs",0);

	set("text_editor/font","");
	hints["text_editor/font"]=PropertyInfo(Variant::STRING,"text_editor/font",PROPERTY_HINT_GLOBAL_FILE,"*.fnt");
	set("text_editor/auto_brace_complete", false);
	set("text_editor/restore_scripts_on_load",true);


	set("scenetree_editor/duplicate_node_name_num_separator",0);
	hints["scenetree_editor/duplicate_node_name_num_separator"]=PropertyInfo(Variant::INT,"scenetree_editor/duplicate_node_name_num_separator",PROPERTY_HINT_ENUM, "None,Space,Underscore,Dash");

	set("gridmap_editor/pick_distance", 5000.0);

	set("3d_editor/default_fov",45.0);
	set("3d_editor/default_z_near",0.1);
	set("3d_editor/default_z_far",500.0);

	set("3d_editor/navigation_scheme",0);
	hints["3d_editor/navigation_scheme"]=PropertyInfo(Variant::INT,"3d_editor/navigation_scheme",PROPERTY_HINT_ENUM,"Godot,Maya,Modo");
	set("3d_editor/zoom_style",0);
	hints["3d_editor/zoom_style"]=PropertyInfo(Variant::INT,"3d_editor/zoom_style",PROPERTY_HINT_ENUM,"Vertical, Horizontal");
	set("3d_editor/orbit_modifier",0);
	hints["3d_editor/orbit_modifier"]=PropertyInfo(Variant::INT,"3d_editor/orbit_modifier",PROPERTY_HINT_ENUM,"None,Shift,Alt,Meta,Ctrl");
	set("3d_editor/pan_modifier",1);
	hints["3d_editor/pan_modifier"]=PropertyInfo(Variant::INT,"3d_editor/pan_modifier",PROPERTY_HINT_ENUM,"None,Shift,Alt,Meta,Ctrl");
	set("3d_editor/zoom_modifier",4);
	hints["3d_editor/zoom_modifier"]=PropertyInfo(Variant::INT,"3d_editor/zoom_modifier",PROPERTY_HINT_ENUM,"None,Shift,Alt,Meta,Ctrl");

	set("2d_editor/bone_width",5);
	set("2d_editor/bone_color1",Color(1.0,1.0,1.0,0.9));
	set("2d_editor/bone_color2",Color(0.75,0.75,0.75,0.9));
	set("2d_editor/bone_selected_color",Color(0.9,0.45,0.45,0.9));
	set("2d_editor/bone_ik_color",Color(0.9,0.9,0.45,0.9));

	set("game_window_placement/rect",0);
	hints["game_window_placement/rect"]=PropertyInfo(Variant::INT,"game_window_placement/rect",PROPERTY_HINT_ENUM,"Default,Centered,Custom Position,Force Maximized,Force Full Screen");
	String screen_hints="Default (Same as Editor)";
	for(int i=0;i<OS::get_singleton()->get_screen_count();i++) {
		screen_hints+=",Monitor "+itos(i+1);
	}
	set("game_window_placement/rect_custom_position",Vector2());
	set("game_window_placement/screen",0);
	hints["game_window_placement/screen"]=PropertyInfo(Variant::INT,"game_window_placement/screen",PROPERTY_HINT_ENUM,screen_hints);

	set("on_save/compress_binary_resources",true);
	set("on_save/save_modified_external_resources",true);
	set("on_save/save_paths_as_relative",false);
	set("on_save/save_paths_without_extension",false);

	set("text_editor/create_signal_callbacks",true);

	set("file_dialog/show_hidden_files", false);
	set("file_dialog/thumbnail_size", 64);
	hints["file_dialog/thumbnail_size"]=PropertyInfo(Variant::INT,"file_dialog/thumbnail_size",PROPERTY_HINT_RANGE,"32,128,16");

	set("animation/autorename_animation_tracks",true);
	set("animation/confirm_insert_track",true);

	set("property_editor/texture_preview_width",48);
	set("property_editor/auto_refresh_interval",0.3);
	set("help/doc_path","");

	set("import/ask_save_before_reimport",false);

	set("import/pvrtc_texture_tool","");
#ifdef WINDOWS_ENABLED
	hints["import/pvrtc_texture_tool"]=PropertyInfo(Variant::STRING,"import/pvrtc_texture_tool",PROPERTY_HINT_GLOBAL_FILE,"*.exe");
#else
	hints["import/pvrtc_texture_tool"]=PropertyInfo(Variant::STRING,"import/pvrtc_texture_tool",PROPERTY_HINT_GLOBAL_FILE,"");
#endif
	set("PVRTC/fast_conversion",false);


	set("run/auto_save_before_running",true);
	set("resources/save_compressed_resources",true);
	set("resources/auto_reload_modified_images",true);
}

void EditorSettings::notify_changes() {

	_THREAD_SAFE_METHOD_

	SceneTree *sml=NULL;

	if (OS::get_singleton()->get_main_loop())
		sml = OS::get_singleton()->get_main_loop()->cast_to<SceneTree>();

	if (!sml) {
		print_line("not SML");
		return;
	}

	Node *root = sml->get_root()->get_child(0);

	if (!root) {
		return;
	}
	root->propagate_notification(NOTIFICATION_EDITOR_SETTINGS_CHANGED);

}

void EditorSettings::add_property_hint(const PropertyInfo& p_hint) {

	_THREAD_SAFE_METHOD_

	hints[p_hint.name]=p_hint;
}


bool EditorSettings::is_plugin_enabled(const String& p_plugin) {

	if (!has("_plugins/enabled"))
		return false;

	StringArray sa=get("_plugins/enabled");

	for(int i=0;i<sa.size();i++) {

		String plugin = sa[i];
		if (!plugins.has(plugin))
			continue;
		if (plugin==p_plugin)
			return true;
	}

	return false;

}

void EditorSettings::enable_plugins() {

	// editor plugins
	if (has("_plugins/enabled")) {
	StringArray sa=get("_plugins/enabled");

		for(int i=0;i<sa.size();i++) {

			String plugin = sa[i];
			if (!plugins.has(plugin))
				continue;
			if (plugins[plugin].installs)
				continue; //not configured here
			set_plugin_enabled(plugin,true);
		}
	}

	// installed plugins
	List<PropertyInfo> pi;
	Globals::get_singleton()->get_property_list(&pi);
	for (List<PropertyInfo>::Element *E=pi.front();E;E=E->next()) {

		String p = E->get().name;

		if (p.begins_with("plugins/")) {
			load_installed_plugin(p.replace_first("plugins/",""));
		}
	}

}

void EditorSettings::load_installed_plugin(const String& p_plugin) {

	ERR_FAIL_COND( !Globals::get_singleton()->has("plugins/"+p_plugin) );
	String path = Globals::get_singleton()->get("plugins/"+p_plugin);

	Plugin plugin;
	Error err = _load_plugin(path.plus_file("plugin.cfg"),plugin);

	if (err)
		return;

	print_line("installing plugin...");
	EditorPlugin *ep=_load_plugin_editor(path.plus_file(plugin.script));
	ERR_FAIL_COND(!ep);
	print_line("load!");
	EditorNode::add_editor_plugin(ep);

}


EditorPlugin *EditorSettings::_load_plugin_editor(const String& p_path) {

	Ref<Script> script = ResourceLoader::load(p_path);
	ERR_EXPLAIN("Invalid Script for plugin: "+p_path);
	ERR_FAIL_COND_V(script.is_null(),NULL);
	ERR_EXPLAIN("Script has errors: "+p_path);
	ERR_FAIL_COND_V(!script->can_instance(),NULL);
	ERR_EXPLAIN("Script does not inherit EditorPlugin: "+p_path);
	ERR_FAIL_COND_V(script->get_instance_base_type().operator String()!="EditorPlugin",NULL);

	EditorPlugin *ep = memnew( EditorPlugin );
	ep->set_script(script.get_ref_ptr());
	if (!ep->get_script_instance()) {
		memdelete(ep);
		ERR_EXPLAIN("Script could't load: "+p_path);
		ERR_FAIL_V(NULL);
	}


	return ep;
}

void EditorSettings::set_plugin_enabled(const String& p_plugin, bool p_enabled) {

	ERR_FAIL_COND(!plugins.has(p_plugin));
	if (p_enabled == (plugins[p_plugin].instance!=NULL)) //already enabled or disabled
		return;

	print_line("REQUEST "+p_plugin+" to "+itos(p_enabled));
	StringArray sa;
	if (has("_plugins/enabled"))
		sa=get("_plugins/enabled");

	int idx=-1;
	for(int i=0;i<sa.size();i++) {

		if (sa[i]==p_plugin) {
			idx=i;
			break;
		}
	}

	if (p_enabled) {


		String ppath = get_settings_path().plus_file("plugins/"+p_plugin+"/"+plugins[p_plugin].script);
		EditorPlugin *ep=_load_plugin_editor(ppath);
		if (!ep)
			return;
		plugins[p_plugin].instance=ep;
		EditorNode::add_editor_plugin(ep);

		if (idx==-1)
			sa.push_back(p_plugin);
	} else {

		print_line("DISABLING");
		EditorNode::remove_editor_plugin(plugins[p_plugin].instance);
		memdelete(plugins[p_plugin].instance);
		plugins[p_plugin].instance=NULL;
		if (idx!=-1)
			sa.remove(idx);

	}

	if (sa.size()==0)
		set("_plugins/enabled",Variant());
	else
		set("_plugins/enabled",sa);

}

void EditorSettings::set_favorite_dirs(const Vector<String>& p_favorites) {

	favorite_dirs=p_favorites;
	FileAccess *f = FileAccess::open(get_project_settings_path().plus_file("favorite_dirs"),FileAccess::WRITE);
	if (f) {
		for(int i=0;i<favorite_dirs.size();i++)
			f->store_line(favorite_dirs[i]);
		memdelete(f);
	}

}

Vector<String> EditorSettings::get_favorite_dirs() const {

	return favorite_dirs;
}


void EditorSettings::set_recent_dirs(const Vector<String>& p_recent) {

	recent_dirs=p_recent;
	FileAccess *f = FileAccess::open(get_project_settings_path().plus_file("recent_dirs"),FileAccess::WRITE);
	if (f) {
		for(int i=0;i<recent_dirs.size();i++)
			f->store_line(recent_dirs[i]);
		memdelete(f);
	}
}

Vector<String> EditorSettings::get_recent_dirs() const {

	return recent_dirs;
}

String EditorSettings::get_project_settings_path() const {


	return get_settings_path().plus_file("config").plus_file(project_config_path);
}


void EditorSettings::load_favorites() {

	FileAccess *f = FileAccess::open(get_project_settings_path().plus_file("favorite_dirs"),FileAccess::READ);
	if (f) {
		String line = f->get_line().strip_edges();
		while(line!="") {
			favorite_dirs.push_back(line);
			line = f->get_line().strip_edges();
		}
		memdelete(f);
	}

	f = FileAccess::open(get_project_settings_path().plus_file("recent_dirs"),FileAccess::READ);
	if (f) {
		String line = f->get_line().strip_edges();
		while(line!="") {
			recent_dirs.push_back(line);
			line = f->get_line().strip_edges();
		}
		memdelete(f);
	}

}


void EditorSettings::_bind_methods() {

	ADD_SIGNAL(MethodInfo("settings_changed"));

}

EditorSettings::EditorSettings() {


	//singleton=this;
	last_order=0;
	_load_defaults();
}


EditorSettings::~EditorSettings() {

//	singleton=NULL;
}


