/*************************************************************************/
/*  globals.cpp                                                          */
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
#include "global_config.h"

#include "bind/core_bind.h"
#include "io/file_access_network.h"
#include "io/file_access_pack.h"
#include "io/marshalls.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "variant_parser.h"

#define FORMAT_VERSION 3

GlobalConfig *GlobalConfig::singleton = NULL;

GlobalConfig *GlobalConfig::get_singleton() {

	return singleton;
}

String GlobalConfig::get_resource_path() const {

	return resource_path;
};

String GlobalConfig::localize_path(const String &p_path) const {

	if (resource_path == "")
		return p_path; //not initialied yet

	if (p_path.begins_with("res://") || p_path.begins_with("user://") ||
			(p_path.is_abs_path() && !p_path.begins_with(resource_path)))
		return p_path.simplify_path();

	DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	String path = p_path.replace("\\", "/").simplify_path();

	if (dir->change_dir(path) == OK) {

		String cwd = dir->get_current_dir();
		cwd = cwd.replace("\\", "/");

		memdelete(dir);

		if (!cwd.begins_with(resource_path)) {
			return p_path;
		};

		return cwd.replace_first(resource_path, "res:/");
	} else {

		memdelete(dir);

		int sep = path.find_last("/");
		if (sep == -1) {
			return "res://" + path;
		};

		String parent = path.substr(0, sep);

		String plocal = localize_path(parent);
		if (plocal == "") {
			return "";
		};
		return plocal + path.substr(sep, path.size() - sep);
	};
}

void GlobalConfig::set_initial_value(const String &p_name, const Variant &p_value) {

	ERR_FAIL_COND(!props.has(p_name));
	props[p_name].initial = p_value;
}

String GlobalConfig::globalize_path(const String &p_path) const {

	if (p_path.begins_with("res://")) {

		if (resource_path != "") {

			return p_path.replace("res:/", resource_path);
		};
		return p_path.replace("res://", "");
	};

	return p_path;
}

bool GlobalConfig::_set(const StringName &p_name, const Variant &p_value) {

	_THREAD_SAFE_METHOD_

	if (p_value.get_type() == Variant::NIL)
		props.erase(p_name);
	else {
		if (props.has(p_name)) {
			if (!props[p_name].overrided)
				props[p_name].variant = p_value;

			if (props[p_name].order >= NO_ORDER_BASE && registering_order) {
				props[p_name].order = last_order++;
			}
		} else {
			props[p_name] = VariantContainer(p_value, last_order++ + (registering_order ? 0 : NO_ORDER_BASE));
		}
	}

	if (!disable_platform_override) {

		String s = String(p_name);
		int sl = s.find("/");
		int p = s.find(".");
		if (p != -1 && sl != -1 && p < sl) {

			Vector<String> ps = s.substr(0, sl).split(".");
			String prop = s.substr(sl, s.length() - sl);
			for (int i = 1; i < ps.size(); i++) {

				if (ps[i] == OS::get_singleton()->get_name()) {

					String fullprop = ps[0] + prop;

					set(fullprop, p_value);
					props[fullprop].overrided = true;
				}
			}
		}
	}

	return true;
}
bool GlobalConfig::_get(const StringName &p_name, Variant &r_ret) const {

	_THREAD_SAFE_METHOD_

	if (!props.has(p_name)) {
		print_line("WARNING: not found: " + String(p_name));
		return false;
	}
	r_ret = props[p_name].variant;
	return true;
}

struct _VCSort {

	String name;
	Variant::Type type;
	int order;
	int flags;

	bool operator<(const _VCSort &p_vcs) const { return order == p_vcs.order ? name < p_vcs.name : order < p_vcs.order; }
};

void GlobalConfig::_get_property_list(List<PropertyInfo> *p_list) const {

	_THREAD_SAFE_METHOD_

	Set<_VCSort> vclist;

	for (Map<StringName, VariantContainer>::Element *E = props.front(); E; E = E->next()) {

		const VariantContainer *v = &E->get();

		if (v->hide_from_editor)
			continue;

		_VCSort vc;
		vc.name = E->key();
		vc.order = v->order;
		vc.type = v->variant.get_type();
		if (vc.name.begins_with("input/") || vc.name.begins_with("import/") || vc.name.begins_with("export/") || vc.name.begins_with("/remap") || vc.name.begins_with("/locale") || vc.name.begins_with("/autoload"))
			vc.flags = PROPERTY_USAGE_STORAGE;
		else
			vc.flags = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE;

		vclist.insert(vc);
	}

	for (Set<_VCSort>::Element *E = vclist.front(); E; E = E->next()) {

		if (custom_prop_info.has(E->get().name)) {
			PropertyInfo pi = custom_prop_info[E->get().name];
			pi.name = E->get().name;
			pi.usage = E->get().flags;
			p_list->push_back(pi);
		} else
			p_list->push_back(PropertyInfo(E->get().type, E->get().name, PROPERTY_HINT_NONE, "", E->get().flags));
	}
}

bool GlobalConfig::_load_resource_pack(const String &p_pack) {

	if (PackedData::get_singleton()->is_disabled())
		return false;

	bool ok = PackedData::get_singleton()->add_pack(p_pack) == OK;

	if (!ok)
		return false;

	//if data.pck is found, all directory access will be from here
	DirAccess::make_default<DirAccessPack>(DirAccess::ACCESS_RESOURCES);
	using_datapack = true;

	return true;
}

Error GlobalConfig::setup(const String &p_path, const String &p_main_pack) {

	//If looking for files in network, just use network!

	if (FileAccessNetworkClient::get_singleton()) {

		if (_load_settings("res://godot.cfg") == OK || _load_settings_binary("res://godot.cfb") == OK) {

			_load_settings("res://override.cfg");
		}

		return OK;
	}

	String exec_path = OS::get_singleton()->get_executable_path();

	//Attempt with a passed main pack first

	if (p_main_pack != "") {

		bool ok = _load_resource_pack(p_main_pack);
		ERR_FAIL_COND_V(!ok, ERR_CANT_OPEN);

		if (_load_settings("res://godot.cfg") == OK || _load_settings_binary("res://godot.cfb") == OK) {
			//load override from location of the main pack
			_load_settings(p_main_pack.get_base_dir().plus_file("override.cfg"));
		}

		return OK;
	}

	//Attempt with execname.pck
	if (exec_path != "") {

		if (_load_resource_pack(exec_path.get_basename() + ".pck")) {

			if (_load_settings("res://godot.cfg") == OK || _load_settings_binary("res://godot.cfb") == OK) {
				//load override from location of executable
				_load_settings(exec_path.get_base_dir().plus_file("override.cfg"));
			}

			return OK;
		}
	}

	//Try to use the filesystem for files, according to OS. (only Android -when reading from pck- and iOS use this)
	if (OS::get_singleton()->get_resource_dir() != "") {
		//OS will call Globals->get_resource_path which will be empty if not overridden!
		//if the OS would rather use somewhere else, then it will not be empty.

		resource_path = OS::get_singleton()->get_resource_dir().replace("\\", "/");
		if (resource_path.length() && resource_path[resource_path.length() - 1] == '/')
			resource_path = resource_path.substr(0, resource_path.length() - 1); // chop end

		// data.pck and data.zip are deprecated and no longer supported, apologies.
		// make sure this is loaded from the resource path

		if (_load_settings("res://godot.cfg") == OK || _load_settings_binary("res://godot.cfb") == OK) {
			_load_settings("res://override.cfg");
		}

		return OK;
	}

	//Nothing was found, try to find a godot.cfg somewhere!

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!d, ERR_CANT_CREATE);

	d->change_dir(p_path);

	String candidate = d->get_current_dir();
	String current_dir = d->get_current_dir();
	bool found = false;

	while (true) {
		//try to load settings in ascending through dirs shape!

		if (_load_settings(current_dir + "/godot.cfg") == OK || _load_settings_binary(current_dir + "/godot.cfb") == OK) {

			_load_settings(current_dir + "/override.cfg");
			candidate = current_dir;
			found = true;
			break;
		}

		d->change_dir("..");
		if (d->get_current_dir() == current_dir)
			break; //not doing anything useful
		current_dir = d->get_current_dir();
	}

	resource_path = candidate;
	resource_path = resource_path.replace("\\", "/"); // windows path to unix path just in case
	memdelete(d);

	if (!found)
		return ERR_FILE_NOT_FOUND;

	if (resource_path.length() && resource_path[resource_path.length() - 1] == '/')
		resource_path = resource_path.substr(0, resource_path.length() - 1); // chop end

	return OK;
}

bool GlobalConfig::has(String p_var) const {

	_THREAD_SAFE_METHOD_

	return props.has(p_var);
}

void GlobalConfig::set_registering_order(bool p_enable) {

	registering_order = p_enable;
}

Error GlobalConfig::_load_settings_binary(const String p_path) {

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err != OK) {
		return err;
	}

	uint8_t hdr[4];
	f->get_buffer(hdr, 4);
	if (hdr[0] != 'E' || hdr[1] != 'C' || hdr[2] != 'F' || hdr[3] != 'G') {

		memdelete(f);
		ERR_EXPLAIN("Corrupted header in binary godot.cfb (not ECFG)");
		ERR_FAIL_V(ERR_FILE_CORRUPT;)
	}

	set_registering_order(false);

	uint32_t count = f->get_32();

	for (uint32_t i = 0; i < count; i++) {

		uint32_t slen = f->get_32();
		CharString cs;
		cs.resize(slen + 1);
		cs[slen] = 0;
		f->get_buffer((uint8_t *)cs.ptr(), slen);
		String key;
		key.parse_utf8(cs.ptr());

		uint32_t vlen = f->get_32();
		Vector<uint8_t> d;
		d.resize(vlen);
		f->get_buffer(d.ptr(), vlen);
		Variant value;
		Error err = decode_variant(value, d.ptr(), d.size());
		ERR_EXPLAIN("Error decoding property: " + key);
		ERR_CONTINUE(err != OK);
		set(key, value);
	}

	set_registering_order(true);

	return OK;
}
Error GlobalConfig::_load_settings(const String p_path) {

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);

	if (!f)
		return ERR_CANT_OPEN;

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines = 0;
	String error_text;

	String section;

	while (true) {

		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, NULL, true);
		if (err == ERR_FILE_EOF) {
			memdelete(f);
			return OK;
		} else if (err != OK) {
			ERR_PRINTS("GlobalConfig::load - " + p_path + ":" + itos(lines) + " error: " + error_text);
			memdelete(f);
			return err;
		}

		if (assign != String()) {
			if (section == String() && assign == "config_version") {
				int config_version = value;
				if (config_version > FORMAT_VERSION) {
					memdelete(f);
					ERR_FAIL_COND_V(config_version > FORMAT_VERSION, ERR_FILE_CANT_OPEN);
				}
			}
			set(section + "/" + assign, value);
		} else if (next_tag.name != String()) {
			section = next_tag.name;
		}
	}

	memdelete(f);

	return OK;
}

int GlobalConfig::get_order(const String &p_name) const {

	ERR_FAIL_COND_V(!props.has(p_name), -1);
	return props[p_name].order;
}

void GlobalConfig::set_order(const String &p_name, int p_order) {

	ERR_FAIL_COND(!props.has(p_name));
	props[p_name].order = p_order;
}

void GlobalConfig::clear(const String &p_name) {

	ERR_FAIL_COND(!props.has(p_name));
	props.erase(p_name);
}

Error GlobalConfig::save() {

	return save_custom(get_resource_path() + "/godot.cfg");
}

Error GlobalConfig::_save_settings_binary(const String &p_file, const Map<String, List<String> > &props, const CustomMap &p_custom) {

	Error err;
	FileAccess *file = FileAccess::open(p_file, FileAccess::WRITE, &err);
	if (err != OK) {

		ERR_EXPLAIN("Coudln't save godot.cfb at " + p_file);
		ERR_FAIL_COND_V(err, err)
	}

	uint8_t hdr[4] = { 'E', 'C', 'F', 'G' };
	file->store_buffer(hdr, 4);

	int count = 0;

	for (Map<String, List<String> >::Element *E = props.front(); E; E = E->next()) {

		for (List<String>::Element *F = E->get().front(); F; F = F->next()) {

			count++;
		}
	}

	file->store_32(count); //store how many properties are saved

	for (Map<String, List<String> >::Element *E = props.front(); E; E = E->next()) {

		for (List<String>::Element *F = E->get().front(); F; F = F->next()) {

			String key = F->get();
			if (E->key() != "")
				key = E->key() + "/" + key;
			Variant value;
			if (p_custom.has(key))
				value = p_custom[key];
			else
				value = get(key);

			file->store_32(key.length());
			file->store_string(key);

			int len;
			Error err = encode_variant(value, NULL, len);
			if (err != OK)
				memdelete(file);
			ERR_FAIL_COND_V(err != OK, ERR_INVALID_DATA);

			Vector<uint8_t> buff;
			buff.resize(len);

			err = encode_variant(value, &buff[0], len);
			if (err != OK)
				memdelete(file);
			ERR_FAIL_COND_V(err != OK, ERR_INVALID_DATA);
			file->store_32(len);
			file->store_buffer(buff.ptr(), buff.size());
		}
	}

	file->close();
	memdelete(file);

	return OK;
}

Error GlobalConfig::_save_settings_text(const String &p_file, const Map<String, List<String> > &props, const CustomMap &p_custom) {

	Error err;
	FileAccess *file = FileAccess::open(p_file, FileAccess::WRITE, &err);

	if (err) {
		ERR_EXPLAIN("Coudln't save godot.cfg - " + p_file);
		ERR_FAIL_COND_V(err, err)
	}

	file->store_string("config_version=" + itos(FORMAT_VERSION) + "\n");

	for (Map<String, List<String> >::Element *E = props.front(); E; E = E->next()) {

		if (E != props.front())
			file->store_string("\n");

		if (E->key() != "")
			file->store_string("[" + E->key() + "]\n\n");
		for (List<String>::Element *F = E->get().front(); F; F = F->next()) {

			String key = F->get();
			if (E->key() != "")
				key = E->key() + "/" + key;
			Variant value;
			if (p_custom.has(key))
				value = p_custom[key];
			else
				value = get(key);

			String vstr;
			VariantWriter::write_to_string(value, vstr);
			file->store_string(F->get() + "=" + vstr + "\n");
		}
	}

	file->close();
	memdelete(file);

	return OK;
}

Error GlobalConfig::_save_custom_bnd(const String &p_file) { // add other params as dictionary and array?

	return save_custom(p_file);
};

Error GlobalConfig::save_custom(const String &p_path, const CustomMap &p_custom, const Set<String> &p_ignore_masks) {

	ERR_FAIL_COND_V(p_path == "", ERR_INVALID_PARAMETER);

	Set<_VCSort> vclist;

	for (Map<StringName, VariantContainer>::Element *G = props.front(); G; G = G->next()) {

		const VariantContainer *v = &G->get();

		if (v->hide_from_editor)
			continue;

		if (p_custom.has(G->key()))
			continue;

		bool discard = false;

		for (const Set<String>::Element *E = p_ignore_masks.front(); E; E = E->next()) {

			if (String(G->key()).match(E->get())) {
				discard = true;
				break;
			}
		}

		if (discard)
			continue;

		_VCSort vc;
		vc.name = G->key(); //*k;
		vc.order = v->order;
		vc.type = v->variant.get_type();
		vc.flags = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE;
		if (v->variant == v->initial)
			continue;

		vclist.insert(vc);
	}

	for (const Map<String, Variant>::Element *E = p_custom.front(); E; E = E->next()) {

		_VCSort vc;
		vc.name = E->key();
		vc.order = 0xFFFFFFF;
		vc.type = E->get().get_type();
		vc.flags = PROPERTY_USAGE_STORAGE;
		vclist.insert(vc);
	}

	Map<String, List<String> > props;

	for (Set<_VCSort>::Element *E = vclist.front(); E; E = E->next()) {

		String category = E->get().name;
		String name = E->get().name;

		int div = category.find("/");

		if (div < 0)
			category = "";
		else {

			category = category.substr(0, div);
			name = name.substr(div + 1, name.size());
		}
		props[category].push_back(name);
	}

	if (p_path.ends_with(".cfg"))
		return _save_settings_text(p_path, props, p_custom);
	else if (p_path.ends_with(".cfb"))
		return _save_settings_binary(p_path, props, p_custom);
	else {

		ERR_EXPLAIN("Unknown config file format: " + p_path);
		ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
	}

	return OK;

#if 0
	Error err = file->open(dst_file,FileAccess::WRITE);
	if (err) {
		memdelete(file);
		ERR_EXPLAIN("Coudln't save godot.cfg");
		ERR_FAIL_COND_V(err,err)
	}


	for(Map<String,List<String> >::Element *E=props.front();E;E=E->next()) {

		if (E!=props.front())
			file->store_string("\n");

		if (E->key()!="")
			file->store_string("["+E->key()+"]\n\n");
		for(List<String>::Element *F=E->get().front();F;F=F->next()) {

			String key = F->get();
			if (E->key()!="")
				key=E->key()+"/"+key;
			Variant value;

			if (p_custom.has(key))
				value=p_custom[key];
			else
				value = get(key);

			file->store_string(F->get()+"="+_encode_variant(value)+"\n");

		}
	}

	file->close();
	memdelete(file);


	return OK;
#endif
}

Variant _GLOBAL_DEF(const String &p_var, const Variant &p_default) {

	if (GlobalConfig::get_singleton()->has(p_var)) {
		GlobalConfig::get_singleton()->set_initial_value(p_var, p_default);
		return GlobalConfig::get_singleton()->get(p_var);
	}
	GlobalConfig::get_singleton()->set(p_var, p_default);
	GlobalConfig::get_singleton()->set_initial_value(p_var, p_default);
	return p_default;
}

void GlobalConfig::add_singleton(const Singleton &p_singleton) {

	singletons.push_back(p_singleton);
	singleton_ptrs[p_singleton.name] = p_singleton.ptr;
}

Object *GlobalConfig::get_singleton_object(const String &p_name) const {

	const Map<StringName, Object *>::Element *E = singleton_ptrs.find(p_name);
	if (!E)
		return NULL;
	else
		return E->get();
};

bool GlobalConfig::has_singleton(const String &p_name) const {

	return get_singleton_object(p_name) != NULL;
};

void GlobalConfig::get_singletons(List<Singleton> *p_singletons) {

	for (List<Singleton>::Element *E = singletons.front(); E; E = E->next())
		p_singletons->push_back(E->get());
}

Vector<String> GlobalConfig::get_optimizer_presets() const {

	List<PropertyInfo> pi;
	GlobalConfig::get_singleton()->get_property_list(&pi);
	Vector<String> names;

	for (List<PropertyInfo>::Element *E = pi.front(); E; E = E->next()) {

		if (!E->get().name.begins_with("optimizer_presets/"))
			continue;
		names.push_back(E->get().name.get_slicec('/', 1));
	}

	names.sort();

	return names;
}

void GlobalConfig::_add_property_info_bind(const Dictionary &p_info) {

	ERR_FAIL_COND(!p_info.has("name"));
	ERR_FAIL_COND(!p_info.has("type"));

	PropertyInfo pinfo;
	pinfo.name = p_info["name"];
	ERR_FAIL_COND(!props.has(pinfo.name));
	pinfo.type = Variant::Type(p_info["type"].operator int());
	ERR_FAIL_INDEX(pinfo.type, Variant::VARIANT_MAX);

	if (p_info.has("hint"))
		pinfo.hint = PropertyHint(p_info["hint"].operator int());
	if (p_info.has("hint_string"))
		pinfo.hint_string = p_info["hint_string"];

	set_custom_property_info(pinfo.name, pinfo);
}

void GlobalConfig::set_custom_property_info(const String &p_prop, const PropertyInfo &p_info) {

	ERR_FAIL_COND(!props.has(p_prop));
	custom_prop_info[p_prop] = p_info;
	custom_prop_info[p_prop].name = p_prop;
}

void GlobalConfig::set_disable_platform_override(bool p_disable) {

	disable_platform_override = p_disable;
}

bool GlobalConfig::is_using_datapack() const {

	return using_datapack;
}

bool GlobalConfig::property_can_revert(const String &p_name) {

	if (!props.has(p_name))
		return false;

	return props[p_name].initial != props[p_name].variant;
}

Variant GlobalConfig::property_get_revert(const String &p_name) {

	if (!props.has(p_name))
		return Variant();

	return props[p_name].initial;
}

void GlobalConfig::_bind_methods() {

	ClassDB::bind_method(D_METHOD("has", "name"), &GlobalConfig::has);
	ClassDB::bind_method(D_METHOD("set_order", "name", "pos"), &GlobalConfig::set_order);
	ClassDB::bind_method(D_METHOD("get_order", "name"), &GlobalConfig::get_order);
	ClassDB::bind_method(D_METHOD("set_initial_value", "name", "value"), &GlobalConfig::set_initial_value);
	ClassDB::bind_method(D_METHOD("add_property_info", "hint"), &GlobalConfig::_add_property_info_bind);
	ClassDB::bind_method(D_METHOD("clear", "name"), &GlobalConfig::clear);
	ClassDB::bind_method(D_METHOD("localize_path", "path"), &GlobalConfig::localize_path);
	ClassDB::bind_method(D_METHOD("globalize_path", "path"), &GlobalConfig::globalize_path);
	ClassDB::bind_method(D_METHOD("save"), &GlobalConfig::save);
	ClassDB::bind_method(D_METHOD("has_singleton", "name"), &GlobalConfig::has_singleton);
	ClassDB::bind_method(D_METHOD("get_singleton", "name"), &GlobalConfig::get_singleton_object);
	ClassDB::bind_method(D_METHOD("load_resource_pack", "pack"), &GlobalConfig::_load_resource_pack);
	ClassDB::bind_method(D_METHOD("property_can_revert", "name"), &GlobalConfig::property_can_revert);
	ClassDB::bind_method(D_METHOD("property_get_revert", "name"), &GlobalConfig::property_get_revert);

	ClassDB::bind_method(D_METHOD("save_custom", "file"), &GlobalConfig::_save_custom_bnd);
}

GlobalConfig::GlobalConfig() {

	singleton = this;
	last_order = 0;
	disable_platform_override = false;
	registering_order = true;

	Array va;
	InputEvent key;
	key.type = InputEvent::KEY;
	InputEvent joyb;
	joyb.type = InputEvent::JOYPAD_BUTTON;

	GLOBAL_DEF("application/name", "");
	GLOBAL_DEF("application/main_scene", "");
	custom_prop_info["application/main_scene"] = PropertyInfo(Variant::STRING, "application/main_scene", PROPERTY_HINT_FILE, "tscn,scn,xscn,xml,res");
	GLOBAL_DEF("application/disable_stdout", false);
	GLOBAL_DEF("application/disable_stderr", false);
	GLOBAL_DEF("application/use_shared_user_dir", true);

	key.key.scancode = KEY_RETURN;
	va.push_back(key);
	key.key.scancode = KEY_ENTER;
	va.push_back(key);
	key.key.scancode = KEY_SPACE;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_BUTTON_0;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_accept", va);
	input_presets.push_back("input/ui_accept");

	va = Array();
	key.key.scancode = KEY_SPACE;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_BUTTON_3;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_select", va);
	input_presets.push_back("input/ui_select");

	va = Array();
	key.key.scancode = KEY_ESCAPE;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_BUTTON_1;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_cancel", va);
	input_presets.push_back("input/ui_cancel");

	va = Array();
	key.key.scancode = KEY_TAB;
	va.push_back(key);
	GLOBAL_DEF("input/ui_focus_next", va);
	input_presets.push_back("input/ui_focus_next");

	va = Array();
	key.key.scancode = KEY_TAB;
	key.key.mod.shift = true;
	va.push_back(key);
	GLOBAL_DEF("input/ui_focus_prev", va);
	input_presets.push_back("input/ui_focus_prev");
	key.key.mod.shift = false;

	va = Array();
	key.key.scancode = KEY_LEFT;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_DPAD_LEFT;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_left", va);
	input_presets.push_back("input/ui_left");

	va = Array();
	key.key.scancode = KEY_RIGHT;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_DPAD_RIGHT;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_right", va);
	input_presets.push_back("input/ui_right");

	va = Array();
	key.key.scancode = KEY_UP;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_DPAD_UP;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_up", va);
	input_presets.push_back("input/ui_up");

	va = Array();
	key.key.scancode = KEY_DOWN;
	va.push_back(key);
	joyb.joy_button.button_index = JOY_DPAD_DOWN;
	va.push_back(joyb);
	GLOBAL_DEF("input/ui_down", va);
	input_presets.push_back("input/ui_down");

	va = Array();
	key.key.scancode = KEY_PAGEUP;
	va.push_back(key);
	GLOBAL_DEF("input/ui_page_up", va);
	input_presets.push_back("input/ui_page_up");

	va = Array();
	key.key.scancode = KEY_PAGEDOWN;
	va.push_back(key);
	GLOBAL_DEF("input/ui_page_down", va);
	input_presets.push_back("input/ui_page_down");

	//GLOBAL_DEF("display/handheld/orientation", "landscape");

	custom_prop_info["display/handheld/orientation"] = PropertyInfo(Variant::STRING, "display/handheld/orientation", PROPERTY_HINT_ENUM, "landscape,portrait,reverse_landscape,reverse_portrait,sensor_landscape,sensor_portrait,sensor");
	custom_prop_info["rendering/threads/thread_model"] = PropertyInfo(Variant::INT, "rendering/threads/thread_model", PROPERTY_HINT_ENUM, "Single-Unsafe,Single-Safe,Multi-Threaded");
	custom_prop_info["physics/2d/thread_model"] = PropertyInfo(Variant::INT, "physics/2d/thread_model", PROPERTY_HINT_ENUM, "Single-Unsafe,Single-Safe,Multi-Threaded");

	GLOBAL_DEF("debug/profiler/max_functions", 16384);
	using_datapack = false;
}

GlobalConfig::~GlobalConfig() {

	singleton = NULL;
}
