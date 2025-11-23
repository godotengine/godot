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

	Vector<Pair<int64_t, String>> order;
	order.reserve(entries.size());

	for (const KeyValue<String, Variant> &e : entries) {
		if (e.key.is_empty()) {
			continue;
		}

		int64_t ts = 0;
		Variant v = e.value;
		if (v.get_type() == Variant::DICTIONARY) {
			ts = ((Dictionary)v).get("created_on_ts", (int64_t)0);
		} else if (v.get_type() == Variant::ARRAY) {
			Array arr = v;
			for (Variant ai : arr) {
				if (ai.get_type() == Variant::DICTIONARY) {
					int64_t ats = ((Dictionary)ai).get("created_on_ts", (int64_t)0);
					if (ts == 0 || ats < ts) {
						ts = ats;
					}
				}
			}
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
			Variant v = entries[key];

			Array elements;
			if (v.get_type() == Variant::ARRAY) {
				elements = v;
			} else if (v.get_type() == Variant::DICTIONARY) {
				elements.append(v);
			} else {
				Dictionary wrap;
				wrap["value"] = v;
				elements.append(wrap);
			}

			// Build a sorted Vector<Variant> by inserting each element into its correct place.
			Vector<Variant> sorted;
			for (Variant curr : elements) {
				int64_t curr_ts = 0;
				if (curr.get_type() == Variant::DICTIONARY) {
					curr_ts = ((Dictionary)curr).get("created_on_ts", (int64_t)0);
				}

				// find insertion index (scan from end for likely append)
				int insert_at = sorted.size();
				for (int s = sorted.size() - 1; s >= 0; --s) {
					int64_t s_ts = 0;
					if (sorted[s].get_type() == Variant::DICTIONARY) {
						s_ts = ((Dictionary)sorted[s]).get("created_on_ts", (int64_t)0);
					}
					if (s_ts <= curr_ts) {
						insert_at = s + 1;
						break;
					}
					if (s == 0) {
						insert_at = 0;
					}
				}
				sorted.insert(insert_at, curr);
			}

			f->store_line("\"" + key + "\" = [");

			for (int ai = 0; ai < sorted.size(); ++ai) {
				Dictionary src;
				if (sorted[ai].get_type() == Variant::DICTIONARY) {
					src = sorted[ai];
				} else {
					src = Dictionary();
					src["value"] = sorted[ai];
				}

				Dictionary dict = build_normalized_dict(src);

				f->store_line("\t{");

				Array dkeys = dict.keys();
				for (int j = 0; j < dkeys.size(); ++j) {
					String k = dkeys[j];
					Variant valfield = dict[k];

					String line = "\t\t\"" + k + "\": ";
					if (valfield.get_type() == Variant::STRING) {
						line += "\"" + String(valfield).c_escape() + "\"";
					} else {
						line += String(valfield);
					}
					if (j < dkeys.size() - 1) {
						line += ",";
					}
					f->store_line(line);
				}

				if (ai < sorted.size() - 1) {
					f->store_line("\t},");
					f->store_line("");
				} else {
					f->store_line("\t}");
				}
			}

			if (i < order.size() - 1) {
				f->store_line("],");
				f->store_line("");
			} else {
				f->store_line("]");
			}
		}

		if (f->get_error() == OK) {
			f->close();
			return;
		}
		f->close();
	}

	// Fallback: ConfigFile (normalized: arrays of dicts or dict wrapped into array), also sorted
	Ref<ConfigFile> cfg;
	cfg.instantiate();
	const String section = "UIDs";

	for (const Pair<int64_t, String> &p : order) {
		const String &key = p.second;
		Variant v = entries[key];

		Array out_arr;
		if (v.get_type() == Variant::ARRAY) {
			Array src_arr = v;
			for (int ai = 0; ai < src_arr.size(); ++ai) {
				if (src_arr[ai].get_type() == Variant::DICTIONARY) {
					out_arr.append(build_normalized_dict(src_arr[ai]));
				} else {
					Dictionary wrap;
					wrap["value"] = ai;
					out_arr.append(wrap);
				}
			}
		} else if (v.get_type() == Variant::DICTIONARY) {
			out_arr.append(build_normalized_dict(v));
		} else {
			Dictionary wrap;
			wrap["value"] = v;
			out_arr.append(wrap);
		}

		// sort out_arr the same way
		Vector<Variant> sorted2;
		for (Variant curr : out_arr) {
			int64_t curr_ts = 0;
			if (curr.get_type() == Variant::DICTIONARY) {
				curr_ts = ((Dictionary)curr).get("created_on_ts", (int64_t)0);
			}

			int insert_at = sorted2.size();
			for (int s = sorted2.size() - 1; s >= 0; --s) {
				int64_t s_ts = 0;
				if (sorted2[s].get_type() == Variant::DICTIONARY) {
					s_ts = ((Dictionary)sorted2[s]).get("created_on_ts", (int64_t)0);
				}
				if (s_ts <= curr_ts) {
					insert_at = s + 1;
					break;
				}
				if (s == 0) {
					insert_at = 0;
				}
			}
			sorted2.insert(insert_at, curr);
		}

		out_arr.clear();
		for (const Variant &E : sorted2) {
			out_arr.append(E);
		}

		cfg->set_value(section, key, out_arr);
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

	Dictionary ent;
	ent["type"] = _ext_to_type(p_path.get_extension(), p_path);
	ent["filename"] = p_path.is_empty() ? String() : p_path.get_file();
	ent["path"] = p_path;

	int64_t ts = Time::get_singleton()->get_unix_time_from_system();
	ent["created_on"] = _utc_time_from_ts(ts);
	ent["created_on_ts"] = ts;
	ent["uid_u64"] = (ResourceUID::ID)p_uid;

	if (entries.has(key)) {
		Variant existing = entries[key];

		Array arr;
		if (existing.get_type() == Variant::ARRAY) {
			arr = existing;
		} else if (existing.get_type() == Variant::DICTIONARY) {
			arr.append(existing);
		} else {
			Dictionary wrap;
			wrap["value"] = existing;
			arr.append(wrap);
		}

		bool duplicate = false;
		for (Variant i : arr) {
			if (i.get_type() == Variant::DICTIONARY) {
				Dictionary d = i;
				uint64_t existing_uid = (uint64_t)d.get("uid_u64", (uint64_t)0);
				String existing_path = d.get("path", String());
				if (existing_uid == (uint64_t)p_uid && existing_path == p_path) {
					duplicate = true;
					break;
				}
			}
		}

		if (!duplicate) {
			arr.append(ent);
			entries[key] = arr;
			save_db();
		}
	} else {
		Array arr;
		arr.append(ent);
		entries.insert(key, arr);
		save_db();
	}
}

Dictionary UIDDB::build_normalized_dict(const Dictionary &src) {
	Dictionary d;
	d["type"] = src.get("type", String());
	d["filename"] = src.get("filename", String());
	d["path"] = src.get("path", String());
	d["created_on"] = src.get("created_on", String());
	d["created_on_ts"] = src.get("created_on_ts", (int64_t)0);
	d["uid_u64"] = src.get("uid_u64", (uint64_t)0);

	Array keys = src.keys();
	for (String k : keys) {
		if (!d.has(k)) {
			d[k] = src[k];
		}
	}

	return d;
}
