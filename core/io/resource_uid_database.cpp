/**************************************************************************/
/*  resource_uid_database.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "resource_uid_database.h"

#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/time.h"

UIDDB *UIDDB::singleton = nullptr;

UIDDB::UIDDB() {
	Ref<ConfigFile> cfg;
	cfg.instantiate();
	if (cfg->load(db_path) == OK) {
		const String section = "UIDs";
		if (cfg->has_section(section)) {
			PackedStringArray keys = cfg->get_section_keys(section);
			for (const String &k : keys) {
				if (k.is_empty()) {
					continue;
				}
				entries.insert(k, cfg->get_value(section, k));
			}
		}
	}
}

UIDDB::~UIDDB() {}

UIDDB *UIDDB::get_singleton() {
	if (!singleton) {
		singleton = memnew(UIDDB);
	}
	return singleton;
}

String UIDDB::_utc_time_from_ts(int64_t p_ts) {
	Time *t = Time::get_singleton();
	if (!t) {
		return String("Unknown");
	}
	Dictionary dt = t->get_datetime_dict_from_unix_time(p_ts);
	return vformat("%04d-%02d-%02d (%02d:%02d:%02d) UTC",
			(int)dt["year"], (int)dt["month"], (int)dt["day"],
			(int)dt["hour"], (int)dt["minute"], (int)dt["second"]);
}

String UIDDB::_ext_to_type(const String &p_ext, const String &p_path) const {
	String e = p_ext.to_lower();

	// Scenes
	static const char *scene_exts[] = { "tscn", "scn", nullptr };
	for (int i = 0; scene_exts[i] != nullptr; i++) {
		if (e == scene_exts[i]) {
			return "scene";
		}
	}

	// Scripts
	static const char *script_exts[] = { "gd", "cs", "cpp", "c", "h", "hpp", "rs", "nim", "lua", "js", "kt", "java", "swift", nullptr };
	for (int i = 0; script_exts[i] != nullptr; i++) {
		if (e == script_exts[i]) {
			return "script";
		}
	}

	// Resources
	static const char *resource_exts[] = { "res", "tres", "anim", "shape", "multimesh", "occ", nullptr };
	for (int i = 0; resource_exts[i] != nullptr; i++) {
		if (e == resource_exts[i]) {
			return "resource";
		}
	}

	// Empty path = UNKNOWN
	if (p_path.is_empty()) {
		return "unknown";
	}

	return "asset";
}

void UIDDB::ensure_db_dir() {
	String dir = db_path.get_base_dir();
	Ref<DirAccess> da = DirAccess::open(dir);
	if (da.is_valid()) {
		da->make_dir_recursive(dir);
	}

	Ref<ConfigFile> cfg;
	cfg.instantiate();
	if (cfg->load(db_path) != OK) {
		return;
	}
	const String section = "UIDs";
	if (!cfg->has_section(section)) {
		return;
	}
	PackedStringArray keys = cfg->get_section_keys(section);
	for (const String &k : keys) {
		if (k.is_empty()) {
			continue;
		}
		entries.insert(k, cfg->get_value(section, k));
	}
}

void UIDDB::save_db() {
	MutexLock ml(mutex);

	// sort by created_on_ts oldest → newest
	Vector<Pair<int64_t, String>> order;
	order.reserve(entries.size());

	for (const KeyValue<String, Variant> &e : entries) {
		if (e.key.is_empty()) {
			continue;
		}

		int64_t ts = 0;
		if (e.value.get_type() == Variant::DICTIONARY) {
			ts = ((Dictionary)e.value).get("created_on_ts", (int64_t)0);
		}
		order.append({ ts, e.key });
	}
	order.sort();

	Ref<FileAccess> f = FileAccess::open(db_path, FileAccess::WRITE);
	if (f.is_valid()) {
		f->store_line(";This file is automatically generated by the engine. It stores all created UIDs of Scripts, Scenes, Resources, and other Assets ever generated by engine.");
		f->store_line("");
		f->store_line("[UIDs]");
		f->store_line("");

		for (int i = 0; i < order.size(); ++i) {
			const String &key = order[i].second;
			Variant val = entries[key];

			Dictionary dict;
			if (val.get_type() == Variant::DICTIONARY) {
				Dictionary src = val;
				dict["type"] = src.get("type", String());
				dict["filename"] = src.get("filename", String());
				dict["path"] = src.get("path", String());
				dict["created_on"] = src.get("created_on", String());
				dict["created_on_ts"] = src.get("created_on_ts", (int64_t)0);
				dict["uid_u64"] = src.get("uid_u64", (uint64_t)0);

				for (const String sk : src.keys()) {
					if (!dict.has(sk)) {
						dict[sk] = src[sk];
					}
				}
			} else {
				dict = val;
			}

			f->store_line("\"" + key + "\" = {");

			// Write fields
			Array dkeys = dict.keys();
			for (int j = 0; j < dkeys.size(); ++j) {
				String k = dkeys[j];
				Variant v = dict[k];

				String line = "\t\"" + k + "\": ";
				if (v.get_type() == Variant::STRING) {
					line += "\"" + String(v).c_escape() + "\"";
				} else {
					line += String(v);
				}
				if (j < dkeys.size() - 1) {
					line += ",";
				}
				f->store_line(line);
			}

			if (i < order.size() - 1) {
				f->store_line("},");
				f->store_line("");
			} else {
				f->store_line("}");
			}
		}

		if (f->get_error() == OK) {
			f->close();
			return;
		}
		f->close();
	}

	// Fallback: if save failed, use ConfigFile (less pretty formatting).
	Ref<ConfigFile> cfg;
	cfg.instantiate();
	const String section = "UIDs";

	for (const Pair<int64_t, String> &p : order) {
		const String &key = p.second;
		Variant v = entries[key];

		if (v.get_type() == Variant::DICTIONARY) {
			Dictionary src = v;
			Dictionary d;
			d["type"] = src.get("type", String());
			d["filename"] = src.get("filename", String());
			d["path"] = src.get("path", String());
			d["created_on"] = src.get("created_on", String());
			d["created_on_ts"] = src.get("created_on_ts", (int64_t)0);
			d["uid_u64"] = src.get("uid_u64", (uint64_t)0);

			for (const String sk : src.keys()) {
				if (!d.has(sk)) {
					d[sk] = src[sk];
				}
			}

			cfg->set_value(section, key, d);
		} else {
			cfg->set_value(section, key, v);
		}
	}

	cfg->save(db_path);
}

void UIDDB::record_uid(uint64_t p_uid, const String &p_path) {
	MutexLock ml(mutex);

	ensure_db_dir();

	String key = ResourceUID::get_singleton()->id_to_text((ResourceUID::ID)p_uid);
	if (key.is_empty()) {
		return;
	}
	if (entries.has(key)) {
		return;
	}

	Dictionary ent;
	ent["type"] = _ext_to_type(p_path.get_extension(), p_path);
	ent["filename"] = p_path.is_empty() ? String() : p_path.get_file();
	ent["path"] = p_path;

	int64_t ts = Time::get_singleton()->get_unix_time_from_system();
	ent["created_on"] = _utc_time_from_ts(ts);
	ent["created_on_ts"] = ts;
	ent["uid_u64"] = (ResourceUID::ID)p_uid;

	entries.insert(key, ent);

	save_db();
}
