/*************************************************************************/
/*  editor_file_system.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "editor_file_system.h"

#include "editor_node.h"
#include "editor_resource_preview.h"
#include "editor_settings.h"
#include "io/resource_import.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "os/os.h"
#include "project_settings.h"
#include "variant_parser.h"

EditorFileSystem *EditorFileSystem::singleton = NULL;

void EditorFileSystemDirectory::sort_files() {

	files.sort_custom<FileInfoSort>();
}

int EditorFileSystemDirectory::find_file_index(const String &p_file) const {

	for (int i = 0; i < files.size(); i++) {
		if (files[i]->file == p_file)
			return i;
	}
	return -1;
}
int EditorFileSystemDirectory::find_dir_index(const String &p_dir) const {

	for (int i = 0; i < subdirs.size(); i++) {
		if (subdirs[i]->name == p_dir)
			return i;
	}

	return -1;
}

int EditorFileSystemDirectory::get_subdir_count() const {

	return subdirs.size();
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_subdir(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, subdirs.size(), NULL);
	return subdirs[p_idx];
}

int EditorFileSystemDirectory::get_file_count() const {

	return files.size();
}

String EditorFileSystemDirectory::get_file(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, files.size(), "");

	return files[p_idx]->file;
}

String EditorFileSystemDirectory::get_path() const {

	String p;
	const EditorFileSystemDirectory *d = this;
	while (d->parent) {
		p = d->name + "/" + p;
		d = d->parent;
	}

	return "res://" + p;
}

String EditorFileSystemDirectory::get_file_path(int p_idx) const {

	String file = get_file(p_idx);
	const EditorFileSystemDirectory *d = this;
	while (d->parent) {
		file = d->name + "/" + file;
		d = d->parent;
	}

	return "res://" + file;
}

Vector<String> EditorFileSystemDirectory::get_file_deps(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, files.size(), Vector<String>());
	return files[p_idx]->deps;
}

bool EditorFileSystemDirectory::get_file_import_is_valid(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, files.size(), false);
	return files[p_idx]->import_valid;
}

StringName EditorFileSystemDirectory::get_file_type(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, files.size(), "");
	return files[p_idx]->type;
}

String EditorFileSystemDirectory::get_name() {

	return name;
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_parent() {

	return parent;
}

void EditorFileSystemDirectory::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_subdir_count"), &EditorFileSystemDirectory::get_subdir_count);
	ClassDB::bind_method(D_METHOD("get_subdir", "idx"), &EditorFileSystemDirectory::get_subdir);
	ClassDB::bind_method(D_METHOD("get_file_count"), &EditorFileSystemDirectory::get_file_count);
	ClassDB::bind_method(D_METHOD("get_file", "idx"), &EditorFileSystemDirectory::get_file);
	ClassDB::bind_method(D_METHOD("get_file_path", "idx"), &EditorFileSystemDirectory::get_file_path);
	ClassDB::bind_method(D_METHOD("get_file_type", "idx"), &EditorFileSystemDirectory::get_file_type);
	ClassDB::bind_method(D_METHOD("get_file_import_is_valid", "idx"), &EditorFileSystemDirectory::get_file_import_is_valid);
	ClassDB::bind_method(D_METHOD("get_name"), &EditorFileSystemDirectory::get_name);
	ClassDB::bind_method(D_METHOD("get_path"), &EditorFileSystemDirectory::get_path);
	ClassDB::bind_method(D_METHOD("get_parent"), &EditorFileSystemDirectory::get_parent);
	ClassDB::bind_method(D_METHOD("find_file_index", "name"), &EditorFileSystemDirectory::find_file_index);
	ClassDB::bind_method(D_METHOD("find_dir_index", "name"), &EditorFileSystemDirectory::find_dir_index);
}

EditorFileSystemDirectory::EditorFileSystemDirectory() {

	modified_time = 0;
	parent = NULL;
	verified = false;
}

EditorFileSystemDirectory::~EditorFileSystemDirectory() {

	for (int i = 0; i < files.size(); i++) {

		memdelete(files[i]);
	}

	for (int i = 0; i < subdirs.size(); i++) {

		memdelete(subdirs[i]);
	}
}

void EditorFileSystem::_scan_filesystem() {

	ERR_FAIL_COND(!scanning || new_filesystem);

	//read .fscache
	String cpath;

	sources_changed.clear();
	file_cache.clear();

	String project = ProjectSettings::get_singleton()->get_resource_path();

	String fscache = EditorSettings::get_singleton()->get_project_settings_path().plus_file("filesystem_cache3");
	FileAccess *f = FileAccess::open(fscache, FileAccess::READ);

	if (f) {
		//read the disk cache
		while (!f->eof_reached()) {

			String l = f->get_line().strip_edges();
			if (l == String())
				continue;

			if (l.begins_with("::")) {
				Vector<String> split = l.split("::");
				ERR_CONTINUE(split.size() != 3);
				String name = split[1];

				cpath = name;

			} else {
				Vector<String> split = l.split("::");
				ERR_CONTINUE(split.size() != 6);
				String name = split[0];
				String file;

				file = name;
				name = cpath.plus_file(name);

				FileCache fc;
				fc.type = split[1];
				fc.modification_time = split[2].to_int64();
				fc.import_modification_time = split[3].to_int64();
				fc.import_valid = split[4].to_int64() != 0;

				String deps = split[5].strip_edges();
				if (deps.length()) {
					Vector<String> dp = deps.split("<>");
					for (int i = 0; i < dp.size(); i++) {
						String path = dp[i];
						fc.deps.push_back(path);
					}
				}

				file_cache[name] = fc;
			}
		}

		f->close();
		memdelete(f);
	}

	String update_cache = EditorSettings::get_singleton()->get_project_settings_path().plus_file("filesystem_update3");

	if (FileAccess::exists(update_cache)) {
		{
			FileAccessRef f = FileAccess::open(update_cache, FileAccess::READ);
			String l = f->get_line().strip_edges();
			while (l != String()) {

				file_cache.erase(l); //erase cache for this, so it gets updated
				l = f->get_line().strip_edges();
			}
		}

		DirAccessRef d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		d->remove(update_cache); //bye bye update cache
	}

	EditorProgressBG scan_progress("efs", "ScanFS", 1000);

	ScanProgress sp;
	sp.low = 0;
	sp.hi = 1;
	sp.progress = &scan_progress;

	new_filesystem = memnew(EditorFileSystemDirectory);
	new_filesystem->parent = NULL;

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	d->change_dir("res://");
	_scan_new_dir(new_filesystem, d, sp);

	file_cache.clear(); //clear caches, no longer needed

	memdelete(d);

	f = FileAccess::open(fscache, FileAccess::WRITE);
	_save_filesystem_cache(new_filesystem, f);
	f->close();
	memdelete(f);

	scanning = false;
}

void EditorFileSystem::_save_filesystem_cache() {
	String fscache = EditorSettings::get_singleton()->get_project_settings_path().plus_file("filesystem_cache3");

	FileAccess *f = FileAccess::open(fscache, FileAccess::WRITE);
	_save_filesystem_cache(filesystem, f);
	f->close();
	memdelete(f);
}

void EditorFileSystem::_thread_func(void *_userdata) {

	EditorFileSystem *sd = (EditorFileSystem *)_userdata;
	sd->_scan_filesystem();
}

bool EditorFileSystem::_update_scan_actions() {

	sources_changed.clear();

	bool fs_changed = false;

	Vector<String> reimports;

	for (List<ItemAction>::Element *E = scan_actions.front(); E; E = E->next()) {

		ItemAction &ia = E->get();

		switch (ia.action) {
			case ItemAction::ACTION_NONE: {

			} break;
			case ItemAction::ACTION_DIR_ADD: {

				int idx = 0;
				for (int i = 0; i < ia.dir->subdirs.size(); i++) {

					if (ia.new_dir->name < ia.dir->subdirs[i]->name)
						break;
					idx++;
				}
				if (idx == ia.dir->subdirs.size()) {
					ia.dir->subdirs.push_back(ia.new_dir);
				} else {
					ia.dir->subdirs.insert(idx, ia.new_dir);
				}

				fs_changed = true;
			} break;
			case ItemAction::ACTION_DIR_REMOVE: {

				ERR_CONTINUE(!ia.dir->parent);
				ia.dir->parent->subdirs.erase(ia.dir);
				memdelete(ia.dir);
				fs_changed = true;
			} break;
			case ItemAction::ACTION_FILE_ADD: {

				int idx = 0;
				for (int i = 0; i < ia.dir->files.size(); i++) {

					if (ia.new_file->file < ia.dir->files[i]->file)
						break;
					idx++;
				}
				if (idx == ia.dir->files.size()) {
					ia.dir->files.push_back(ia.new_file);
				} else {
					ia.dir->files.insert(idx, ia.new_file);
				}

				fs_changed = true;

			} break;
			case ItemAction::ACTION_FILE_REMOVE: {

				int idx = ia.dir->find_file_index(ia.file);
				ERR_CONTINUE(idx == -1);
				_delete_internal_files(ia.dir->files[idx]->file);
				memdelete(ia.dir->files[idx]);
				ia.dir->files.remove(idx);

				fs_changed = true;

			} break;
			case ItemAction::ACTION_FILE_REIMPORT: {

				int idx = ia.dir->find_file_index(ia.file);
				ERR_CONTINUE(idx == -1);
				String full_path = ia.dir->get_file_path(idx);
				reimports.push_back(full_path);

				fs_changed = true;
			} break;
		}
	}

	if (reimports.size()) {
		reimport_files(reimports);
	}
	scan_actions.clear();

	return fs_changed;
}

void EditorFileSystem::scan() {

	if (false /*&& bool(Globals::get_singleton()->get("debug/disable_scan"))*/)
		return;

	if (scanning || scanning_changes || thread)
		return;

	_update_extensions();

	abort_scan = false;
	if (!use_threads) {
		scanning = true;
		scan_total = 0;
		_scan_filesystem();
		if (filesystem)
			memdelete(filesystem);
		//file_type_cache.clear();
		filesystem = new_filesystem;
		new_filesystem = NULL;
		_update_scan_actions();
		scanning = false;
		emit_signal("filesystem_changed");
		emit_signal("sources_changed", sources_changed.size() > 0);

	} else {

		ERR_FAIL_COND(thread);
		set_process(true);
		Thread::Settings s;
		scanning = true;
		scan_total = 0;
		s.priority = Thread::PRIORITY_LOW;
		thread = Thread::create(_thread_func, this, s);
		//tree->hide();
		//progress->show();
	}
}

void EditorFileSystem::ScanProgress::update(int p_current, int p_total) const {

	float ratio = low + ((hi - low) / p_total) * p_current;
	progress->step(ratio * 1000);
	EditorFileSystem::singleton->scan_total = ratio;
}

EditorFileSystem::ScanProgress EditorFileSystem::ScanProgress::get_sub(int p_current, int p_total) const {

	ScanProgress sp = *this;
	float slice = (sp.hi - sp.low) / p_total;
	sp.low += slice * p_current;
	sp.hi = slice;
	return sp;
}

bool EditorFileSystem::_check_missing_imported_files(const String &p_path) {

	if (!reimport_on_missing_imported_files)
		return true;

	Error err;
	FileAccess *f = FileAccess::open(p_path + ".import", FileAccess::READ, &err);

	if (!f) {
		print_line("could not open import for " + p_path);
		return false;
	}

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines = 0;
	String error_text;

	List<String> to_check;

	while (true) {

		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, NULL, true);
		if (err == ERR_FILE_EOF) {
			memdelete(f);
			return OK;
		} else if (err != OK) {
			ERR_PRINTS("ResourceFormatImporter::load - " + p_path + ".import:" + itos(lines) + " error: " + error_text);
			memdelete(f);
			return false;
		}

		if (assign != String()) {
			if (assign.begins_with("path")) {
				to_check.push_back(value);
			} else if (assign == "files") {
				Array fa = value;
				for (int i = 0; i < fa.size(); i++) {
					to_check.push_back(fa[i]);
				}
			}

		} else if (next_tag.name != "remap" && next_tag.name != "deps") {
			break;
		}
	}

	memdelete(f);

	for (List<String>::Element *E = to_check.front(); E; E = E->next()) {
		if (!FileAccess::exists(E->get())) {
			return false;
		}
	}
	return true;
}

void EditorFileSystem::_scan_new_dir(EditorFileSystemDirectory *p_dir, DirAccess *da, const ScanProgress &p_progress) {

	List<String> dirs;
	List<String> files;

	String cd = da->get_current_dir();

	p_dir->modified_time = FileAccess::get_modified_time(cd);

	da->list_dir_begin();
	while (true) {

		bool isdir;
		String f = da->get_next(&isdir);
		if (f == "")
			break;

		if (isdir) {

			if (f.begins_with(".")) //ignore hidden and . / ..
				continue;

			if (FileAccess::exists(cd.plus_file(f).plus_file("project.godot"))) // skip if another project inside this
				continue;
			if (FileAccess::exists(cd.plus_file(f).plus_file(".gdignore"))) // skip if another project inside this
				continue;

			dirs.push_back(f);

		} else {

			files.push_back(f);
		}
	}

	da->list_dir_end();

	dirs.sort_custom<NaturalNoCaseComparator>();
	files.sort_custom<NaturalNoCaseComparator>();

	int total = dirs.size() + files.size();
	int idx = 0;

	for (List<String>::Element *E = dirs.front(); E; E = E->next(), idx++) {

		if (da->change_dir(E->get()) == OK) {

			String d = da->get_current_dir();

			if (d == cd || !d.begins_with(cd)) {
				da->change_dir(cd); //avoid recursion
			} else {

				EditorFileSystemDirectory *efd = memnew(EditorFileSystemDirectory);

				efd->parent = p_dir;
				efd->name = E->get();

				_scan_new_dir(efd, da, p_progress.get_sub(idx, total));

				int idx = 0;
				for (int i = 0; i < p_dir->subdirs.size(); i++) {

					if (efd->name < p_dir->subdirs[i]->name)
						break;
					idx++;
				}
				if (idx == p_dir->subdirs.size()) {
					p_dir->subdirs.push_back(efd);
				} else {
					p_dir->subdirs.insert(idx, efd);
				}

				da->change_dir("..");
			}
		} else {
			ERR_PRINTS("Cannot go into subdir: " + E->get());
		}

		p_progress.update(idx, total);
	}

	for (List<String>::Element *E = files.front(); E; E = E->next(), idx++) {

		String ext = E->get().get_extension().to_lower();
		if (!valid_extensions.has(ext)) {
			continue; //invalid
		}

		EditorFileSystemDirectory::FileInfo *fi = memnew(EditorFileSystemDirectory::FileInfo);
		fi->file = E->get();

		String path = cd.plus_file(fi->file);

		FileCache *fc = file_cache.getptr(path);
		uint64_t mt = FileAccess::get_modified_time(path);

		if (import_extensions.has(ext)) {

			//is imported
			uint64_t import_mt = 0;
			if (FileAccess::exists(path + ".import")) {
				import_mt = FileAccess::get_modified_time(path + ".import");
			}

			if (fc && fc->modification_time == mt && fc->import_modification_time == import_mt && _check_missing_imported_files(path)) {

				fi->type = fc->type;
				fi->deps = fc->deps;
				fi->modified_time = fc->modification_time;
				fi->import_modified_time = fc->import_modification_time;
				fi->import_valid = fc->import_valid;
				if (fc->type == String()) {
					fi->type = ResourceLoader::get_resource_type(path);
					//there is also the chance that file type changed due to reimport, must probably check this somehow here (or kind of note it for next time in another file?)
					//note: I think this should not happen any longer..
				}

			} else {

				fi->type = ResourceFormatImporter::get_singleton()->get_resource_type(path);
				fi->modified_time = 0;
				fi->import_modified_time = 0;
				fi->import_valid = ResourceLoader::is_import_valid(path);

				ItemAction ia;
				ia.action = ItemAction::ACTION_FILE_REIMPORT;
				ia.dir = p_dir;
				ia.file = E->get();
				scan_actions.push_back(ia);
			}
		} else {

			if (fc && fc->modification_time == mt) {
				//not imported, so just update type if changed
				fi->type = fc->type;
				fi->modified_time = fc->modification_time;
				fi->deps = fc->deps;
				fi->import_modified_time = 0;
				fi->import_valid = true;
			} else {
				//new or modified time
				fi->type = ResourceLoader::get_resource_type(path);
				fi->deps = _get_dependencies(path);
				fi->modified_time = mt;
				fi->import_modified_time = 0;
				fi->import_valid = true;
			}
		}

		p_dir->files.push_back(fi);
		p_progress.update(idx, total);
	}
}

void EditorFileSystem::_scan_fs_changes(EditorFileSystemDirectory *p_dir, const ScanProgress &p_progress) {

	uint64_t current_mtime = FileAccess::get_modified_time(p_dir->get_path());

	bool updated_dir = false;
	String cd = p_dir->get_path();

	if (current_mtime != p_dir->modified_time) {

		updated_dir = true;
		p_dir->modified_time = current_mtime;
		//ooooops, dir changed, see what's going on

		//first mark everything as veryfied

		for (int i = 0; i < p_dir->files.size(); i++) {

			p_dir->files[i]->verified = false;
		}

		for (int i = 0; i < p_dir->subdirs.size(); i++) {

			p_dir->get_subdir(i)->verified = false;
		}

		//then scan files and directories and check what's different

		DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);

		da->change_dir(cd);
		da->list_dir_begin();
		while (true) {

			bool isdir;
			String f = da->get_next(&isdir);
			if (f == "")
				break;

			if (isdir) {

				if (f.begins_with(".")) //ignore hidden and . / ..
					continue;

				int idx = p_dir->find_dir_index(f);
				if (idx == -1) {

					if (FileAccess::exists(cd.plus_file(f).plus_file("project.godot"))) // skip if another project inside this
						continue;
					if (FileAccess::exists(cd.plus_file(f).plus_file(".gdignore"))) // skip if another project inside this
						continue;

					EditorFileSystemDirectory *efd = memnew(EditorFileSystemDirectory);

					efd->parent = p_dir;
					efd->name = f;
					DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
					d->change_dir(cd.plus_file(f));
					_scan_new_dir(efd, d, p_progress.get_sub(1, 1));
					memdelete(d);

					ItemAction ia;
					ia.action = ItemAction::ACTION_DIR_ADD;
					ia.dir = p_dir;
					ia.file = f;
					ia.new_dir = efd;
					scan_actions.push_back(ia);
				} else {
					p_dir->subdirs[idx]->verified = true;
				}

			} else {
				String ext = f.get_extension().to_lower();
				if (!valid_extensions.has(ext))
					continue; //invalid

				int idx = p_dir->find_file_index(f);

				if (idx == -1) {
					//never seen this file, add actition to add it
					EditorFileSystemDirectory::FileInfo *fi = memnew(EditorFileSystemDirectory::FileInfo);
					fi->file = f;

					String path = cd.plus_file(fi->file);
					fi->modified_time = FileAccess::get_modified_time(path);
					fi->import_modified_time = 0;
					fi->type = ResourceLoader::get_resource_type(path);
					fi->import_valid = ResourceLoader::is_import_valid(path);

					{
						ItemAction ia;
						ia.action = ItemAction::ACTION_FILE_ADD;
						ia.dir = p_dir;
						ia.file = f;
						ia.new_file = fi;
						scan_actions.push_back(ia);
					}

					if (import_extensions.has(ext)) {
						//if it can be imported, and it was added, it needs to be reimported
						ItemAction ia;
						ia.action = ItemAction::ACTION_FILE_REIMPORT;
						ia.dir = p_dir;
						ia.file = f;
						scan_actions.push_back(ia);
					}

				} else {
					p_dir->files[idx]->verified = true;
				}
			}
		}

		da->list_dir_end();
		memdelete(da);
	}

	for (int i = 0; i < p_dir->files.size(); i++) {

		if (updated_dir && !p_dir->files[i]->verified) {
			//this file was removed, add action to remove it
			ItemAction ia;
			ia.action = ItemAction::ACTION_FILE_REMOVE;
			ia.dir = p_dir;
			ia.file = p_dir->files[i]->file;
			scan_actions.push_back(ia);
			continue;
		}

		if (import_extensions.has(p_dir->files[i]->file.get_extension().to_lower())) {
			//check here if file must be imported or not

			String path = cd.plus_file(p_dir->files[i]->file);

			uint64_t mt = FileAccess::get_modified_time(path);

			bool reimport = false;

			if (mt != p_dir->files[i]->modified_time) {
				reimport = true; //it was modified, must be reimported.
			} else if (!FileAccess::exists(path + ".import")) {
				reimport = true; //no .import file, obviously reimport
			} else {

				uint64_t import_mt = FileAccess::get_modified_time(path + ".import");
				if (import_mt != p_dir->files[i]->import_modified_time) {
					reimport = true;
				} else if (!_check_missing_imported_files(path)) {
					reimport = true;
				}
			}

			if (reimport) {

				ItemAction ia;
				ia.action = ItemAction::ACTION_FILE_REIMPORT;
				ia.dir = p_dir;
				ia.file = p_dir->files[i]->file;
				scan_actions.push_back(ia);
			}
		}

		EditorResourcePreview::get_singleton()->check_for_invalidation(p_dir->get_file_path(i));
	}

	for (int i = 0; i < p_dir->subdirs.size(); i++) {

		if (updated_dir && !p_dir->subdirs[i]->verified) {
			//this directory was removed, add action to remove it
			ItemAction ia;
			ia.action = ItemAction::ACTION_DIR_REMOVE;
			ia.dir = p_dir->subdirs[i];
			scan_actions.push_back(ia);
			continue;
		}
		_scan_fs_changes(p_dir->get_subdir(i), p_progress);
	}
}

void EditorFileSystem::_delete_internal_files(String p_file) {
	if (FileAccess::exists(p_file + ".import")) {
		List<String> paths;
		ResourceFormatImporter::get_singleton()->get_internal_resource_path_list(p_file, &paths);
		DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		for (List<String>::Element *E = paths.front(); E; E = E->next()) {
			da->remove(E->get());
		}
		da->remove(p_file + ".import");
		memdelete(da);
	}
}

void EditorFileSystem::_thread_func_sources(void *_userdata) {

	EditorFileSystem *efs = (EditorFileSystem *)_userdata;
	if (efs->filesystem) {
		EditorProgressBG pr("sources", TTR("ScanSources"), 1000);
		ScanProgress sp;
		sp.progress = &pr;
		sp.hi = 1;
		sp.low = 0;
		efs->_scan_fs_changes(efs->filesystem, sp);
	}
	efs->scanning_changes_done = true;
}

void EditorFileSystem::get_changed_sources(List<String> *r_changed) {

	*r_changed = sources_changed;
}

void EditorFileSystem::scan_changes() {

	if (scanning || scanning_changes || thread)
		return;

	_update_extensions();
	sources_changed.clear();
	scanning_changes = true;
	scanning_changes_done = false;

	abort_scan = false;

	if (!use_threads) {
		if (filesystem) {
			EditorProgressBG pr("sources", TTR("ScanSources"), 1000);
			ScanProgress sp;
			sp.progress = &pr;
			sp.hi = 1;
			sp.low = 0;
			scan_total = 0;
			_scan_fs_changes(filesystem, sp);
			if (_update_scan_actions())
				emit_signal("filesystem_changed");
		}
		scanning_changes = false;
		scanning_changes_done = true;
		emit_signal("sources_changed", sources_changed.size() > 0);
	} else {

		ERR_FAIL_COND(thread_sources);
		set_process(true);
		scan_total = 0;
		Thread::Settings s;
		s.priority = Thread::PRIORITY_LOW;
		thread_sources = Thread::create(_thread_func_sources, this, s);
	}
}

void EditorFileSystem::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			scan();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (use_threads && thread) {
				//abort thread if in progress
				abort_scan = true;
				while (scanning) {
					OS::get_singleton()->delay_usec(1000);
				}
				Thread::wait_to_finish(thread);
				memdelete(thread);
				thread = NULL;
				WARN_PRINTS("Scan thread aborted...");
				set_process(false);
			}

			if (filesystem)
				memdelete(filesystem);
			if (new_filesystem)
				memdelete(new_filesystem);
			filesystem = NULL;
			new_filesystem = NULL;

		} break;
		case NOTIFICATION_PROCESS: {

			if (use_threads) {

				if (scanning_changes) {

					if (scanning_changes_done) {

						scanning_changes = false;

						set_process(false);

						Thread::wait_to_finish(thread_sources);
						memdelete(thread_sources);
						thread_sources = NULL;
						if (_update_scan_actions())
							emit_signal("filesystem_changed");
						emit_signal("sources_changed", sources_changed.size() > 0);
					}
				} else if (!scanning) {

					set_process(false);

					if (filesystem)
						memdelete(filesystem);
					filesystem = new_filesystem;
					new_filesystem = NULL;
					Thread::wait_to_finish(thread);
					memdelete(thread);
					thread = NULL;
					_update_scan_actions();
					emit_signal("filesystem_changed");
					emit_signal("sources_changed", sources_changed.size() > 0);
				}
			}
		} break;
	}
}

bool EditorFileSystem::is_scanning() const {

	return scanning || scanning_changes;
}
float EditorFileSystem::get_scanning_progress() const {

	return scan_total;
}

EditorFileSystemDirectory *EditorFileSystem::get_filesystem() {

	return filesystem;
}

void EditorFileSystem::_save_filesystem_cache(EditorFileSystemDirectory *p_dir, FileAccess *p_file) {

	if (!p_dir)
		return; //none
	p_file->store_line("::" + p_dir->get_path() + "::" + String::num(p_dir->modified_time));

	for (int i = 0; i < p_dir->files.size(); i++) {

		String s = p_dir->files[i]->file + "::" + p_dir->files[i]->type + "::" + itos(p_dir->files[i]->modified_time) + "::" + itos(p_dir->files[i]->import_modified_time) + "::" + itos(p_dir->files[i]->import_valid);
		s += "::";
		for (int j = 0; j < p_dir->files[i]->deps.size(); j++) {

			if (j > 0)
				s += "<>";
			s += p_dir->files[i]->deps[j];
		}

		p_file->store_line(s);
	}

	for (int i = 0; i < p_dir->subdirs.size(); i++) {

		_save_filesystem_cache(p_dir->subdirs[i], p_file);
	}
}

bool EditorFileSystem::_find_file(const String &p_file, EditorFileSystemDirectory **r_d, int &r_file_pos) const {
	//todo make faster

	if (!filesystem || scanning)
		return false;

	String f = ProjectSettings::get_singleton()->localize_path(p_file);

	if (!f.begins_with("res://"))
		return false;
	f = f.substr(6, f.length());
	f = f.replace("\\", "/");

	Vector<String> path = f.split("/");

	if (path.size() == 0)
		return false;
	String file = path[path.size() - 1];
	path.resize(path.size() - 1);

	EditorFileSystemDirectory *fs = filesystem;

	for (int i = 0; i < path.size(); i++) {

		if (path[i].begins_with("."))
			return false;

		int idx = -1;
		for (int j = 0; j < fs->get_subdir_count(); j++) {

			if (fs->get_subdir(j)->get_name() == path[i]) {
				idx = j;
				break;
			}
		}

		if (idx == -1) {
			//does not exist, create i guess?
			EditorFileSystemDirectory *efsd = memnew(EditorFileSystemDirectory);

			efsd->name = path[i];
			efsd->parent = fs;

			int idx2 = 0;
			for (int j = 0; j < fs->get_subdir_count(); j++) {

				if (efsd->name < fs->get_subdir(j)->get_name())
					break;
				idx2++;
			}

			if (idx2 == fs->get_subdir_count())
				fs->subdirs.push_back(efsd);
			else
				fs->subdirs.insert(idx2, efsd);
			fs = efsd;
		} else {

			fs = fs->get_subdir(idx);
		}
	}

	int cpos = -1;
	for (int i = 0; i < fs->files.size(); i++) {

		if (fs->files[i]->file == file) {
			cpos = i;
			break;
		}
	}

	r_file_pos = cpos;
	*r_d = fs;

	if (cpos != -1) {

		return true;
	} else {

		return false;
	}
}

String EditorFileSystem::get_file_type(const String &p_file) const {

	EditorFileSystemDirectory *fs = NULL;
	int cpos = -1;

	if (!_find_file(p_file, &fs, cpos)) {

		return "";
	}

	return fs->files[cpos]->type;
}

EditorFileSystemDirectory *EditorFileSystem::find_file(const String &p_file, int *r_index) const {

	if (!filesystem || scanning)
		return NULL;

	EditorFileSystemDirectory *fs = NULL;
	int cpos = -1;
	if (!_find_file(p_file, &fs, cpos)) {

		return NULL;
	}

	if (r_index)
		*r_index = cpos;

	return fs;
}

EditorFileSystemDirectory *EditorFileSystem::get_filesystem_path(const String &p_path) {

	if (!filesystem || scanning)
		return NULL;

	String f = ProjectSettings::get_singleton()->localize_path(p_path);

	if (!f.begins_with("res://"))
		return NULL;

	f = f.substr(6, f.length());
	f = f.replace("\\", "/");
	if (f == String())
		return filesystem;

	if (f.ends_with("/"))
		f = f.substr(0, f.length() - 1);

	Vector<String> path = f.split("/");

	if (path.size() == 0)
		return NULL;

	EditorFileSystemDirectory *fs = filesystem;

	for (int i = 0; i < path.size(); i++) {

		int idx = -1;
		for (int j = 0; j < fs->get_subdir_count(); j++) {

			if (fs->get_subdir(j)->get_name() == path[i]) {
				idx = j;
				break;
			}
		}

		if (idx == -1) {
			return NULL;
		} else {

			fs = fs->get_subdir(idx);
		}
	}

	return fs;
}

void EditorFileSystem::_save_late_updated_files() {
	//files that already existed, and were modified, need re-scanning for dependencies upon project restart. This is done via saving this special file
	String fscache = EditorSettings::get_singleton()->get_project_settings_path().plus_file("filesystem_update3");
	FileAccessRef f = FileAccess::open(fscache, FileAccess::WRITE);
	for (Set<String>::Element *E = late_update_files.front(); E; E = E->next()) {
		f->store_line(E->get());
	}
}

void EditorFileSystem::_resource_saved(const String &p_path) {

	EditorFileSystem::get_singleton()->update_file(p_path);
}

Vector<String> EditorFileSystem::_get_dependencies(const String &p_path) {

	List<String> deps;
	ResourceLoader::get_dependencies(p_path, &deps);

	Vector<String> ret;
	for (List<String>::Element *E = deps.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}

	return ret;
}

void EditorFileSystem::update_file(const String &p_file) {

	EditorFileSystemDirectory *fs = NULL;
	int cpos = -1;

	if (!_find_file(p_file, &fs, cpos)) {

		if (!fs)
			return;
	}

	if (!FileAccess::exists(p_file)) {
		//was removed
		_delete_internal_files(p_file);
		memdelete(fs->files[cpos]);
		fs->files.remove(cpos);
		call_deferred("emit_signal", "filesystem_changed"); //update later
		return;
	}

	String type = ResourceLoader::get_resource_type(p_file);

	if (cpos == -1) {

		//the file did not exist, it was added

		late_added_files.insert(p_file); //remember that it was added. This mean it will be scanned and imported on editor restart
		int idx = 0;

		for (int i = 0; i < fs->files.size(); i++) {
			if (p_file < fs->files[i]->file)
				break;
			idx++;
		}

		EditorFileSystemDirectory::FileInfo *fi = memnew(EditorFileSystemDirectory::FileInfo);
		fi->file = p_file.get_file();
		fi->import_modified_time = 0;
		fi->import_valid = ResourceLoader::is_import_valid(p_file);

		if (idx == fs->files.size()) {
			fs->files.push_back(fi);
		} else {

			fs->files.insert(idx, fi);
		}
		cpos = idx;
	} else {

		//the file exists and it was updated, and was not added in this step.
		//this means we must force upon next restart to scan it again, to get proper type and dependencies
		late_update_files.insert(p_file);
		_save_late_updated_files(); //files need to be updated in the re-scan
	}

	fs->files[cpos]->type = type;
	fs->files[cpos]->modified_time = FileAccess::get_modified_time(p_file);
	fs->files[cpos]->deps = _get_dependencies(p_file);
	fs->files[cpos]->import_valid = ResourceLoader::is_import_valid(p_file);

	EditorResourcePreview::get_singleton()->call_deferred("check_for_invalidation", p_file);
	call_deferred("emit_signal", "filesystem_changed"); //update later
}

void EditorFileSystem::_reimport_file(const String &p_file) {

	EditorFileSystemDirectory *fs = NULL;
	int cpos = -1;
	bool found = _find_file(p_file, &fs, cpos);
	ERR_FAIL_COND(!found);

	//try to obtain existing params

	Map<StringName, Variant> params;
	String importer_name;

	if (FileAccess::exists(p_file + ".import")) {
		//use existing
		Ref<ConfigFile> cf;
		cf.instance();
		Error err = cf->load(p_file + ".import");
		if (err == OK) {
			List<String> sk;
			cf->get_section_keys("params", &sk);
			for (List<String>::Element *E = sk.front(); E; E = E->next()) {
				params[E->get()] = cf->get_value("params", E->get());
			}
			importer_name = cf->get_value("remap", "importer");
		}

	} else {
		late_added_files.insert(p_file); //imported files do not call update_file(), but just in case..
	}

	Ref<ResourceImporter> importer;
	bool load_default = false;
	//find the importer
	if (importer_name != "") {
		importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(importer_name);
	}

	if (importer.is_null()) {
		//not found by name, find by extension
		importer = ResourceFormatImporter::get_singleton()->get_importer_by_extension(p_file.get_extension());
		load_default = true;
		if (importer.is_null()) {
			ERR_PRINT("BUG: File queued for import, but can't be imported!");
			ERR_FAIL();
		}
	}

	//mix with default params, in case a parameter is missing

	List<ResourceImporter::ImportOption> opts;
	importer->get_import_options(&opts);
	for (List<ResourceImporter::ImportOption>::Element *E = opts.front(); E; E = E->next()) {
		if (!params.has(E->get().option.name)) { //this one is not present
			params[E->get().option.name] = E->get().default_value;
		}
	}

	if (load_default && ProjectSettings::get_singleton()->has_setting("importer_defaults/" + importer->get_importer_name())) {
		//use defaults if exist
		Dictionary d = ProjectSettings::get_singleton()->get("importer_defaults/" + importer->get_importer_name());
		List<Variant> v;
		d.get_key_list(&v);

		for (List<Variant>::Element *E = v.front(); E; E = E->next()) {
			params[E->get()] = d[E->get()];
		}
	}

	//finally, perform import!!
	String base_path = ResourceFormatImporter::get_singleton()->get_import_base_path(p_file);

	List<String> import_variants;
	List<String> gen_files;

	Error err = importer->import(p_file, base_path, params, &import_variants, &gen_files);

	if (err != OK) {
		ERR_PRINTS("Error importing: " + p_file);
	}

	//as import is complete, save the .import file

	FileAccess *f = FileAccess::open(p_file + ".import", FileAccess::WRITE);
	ERR_FAIL_COND(!f);

	//write manually, as order matters ([remap] has to go first for performance).
	f->store_line("[remap]");
	f->store_line("");
	f->store_line("importer=\"" + importer->get_importer_name() + "\"");
	if (importer->get_resource_type() != "") {
		f->store_line("type=\"" + importer->get_resource_type() + "\"");
	}

	if (err == OK) {

		if (importer->get_save_extension() == "") {
			//no path
		} else if (import_variants.size()) {
			//import with variants
			for (List<String>::Element *E = import_variants.front(); E; E = E->next()) {

				String path = base_path.c_escape() + "." + E->get() + "." + importer->get_save_extension();

				f->store_line("path." + E->get() + "=\"" + path + "\"");
			}
		} else {

			f->store_line("path=\"" + base_path + "." + importer->get_save_extension() + "\"");
		}

	} else {

		f->store_line("valid=false");
	}

	f->store_line("");

	if (gen_files.size()) {
		f->store_line("[gen]");
		Array genf;
		for (List<String>::Element *E = gen_files.front(); E; E = E->next()) {
			genf.push_back(E->get());
		}

		String value;
		VariantWriter::write_to_string(genf, value);
		f->store_line("files=" + value);
		f->store_line("");
	}

	f->store_line("[params]");
	f->store_line("");

	//store options in provided order, to avoid file changing. Order is also important because first match is accepted first.

	for (List<ResourceImporter::ImportOption>::Element *E = opts.front(); E; E = E->next()) {

		String base = E->get().option.name;
		String value;
		VariantWriter::write_to_string(params[base], value);
		f->store_line(base + "=" + value);
	}

	f->close();
	memdelete(f);

	//update modified times, to avoid reimport
	fs->files[cpos]->modified_time = FileAccess::get_modified_time(p_file);
	fs->files[cpos]->import_modified_time = FileAccess::get_modified_time(p_file + ".import");
	fs->files[cpos]->deps = _get_dependencies(p_file);
	fs->files[cpos]->type = importer->get_resource_type();
	fs->files[cpos]->import_valid = ResourceLoader::is_import_valid(p_file);

	//if file is currently up, maybe the source it was loaded from changed, so import math must be updated for it
	//to reload properly
	if (ResourceCache::has(p_file)) {

		Resource *r = ResourceCache::get(p_file);

		if (r->get_import_path() != String()) {

			String dst_path = ResourceFormatImporter::get_singleton()->get_internal_resource_path(p_file);
			r->set_import_path(dst_path);
			r->set_import_last_modified_time(0);
		}
	}
}

void EditorFileSystem::reimport_files(const Vector<String> &p_files) {

	importing = true;
	EditorProgress pr("reimport", TTR("(Re)Importing Assets"), p_files.size());

	Vector<ImportFile> files;

	for (int i = 0; i < p_files.size(); i++) {
		ImportFile ifile;
		ifile.path = p_files[i];
		ifile.order = ResourceFormatImporter::get_singleton()->get_import_order(p_files[i]);
		files.push_back(ifile);
	}

	files.sort();

	for (int i = 0; i < files.size(); i++) {
		pr.step(files[i].path.get_file(), i);

		_reimport_file(files[i].path);
	}

	_save_filesystem_cache();
	importing = false;
	if (!is_scanning()) {
		emit_signal("filesystem_changed");
	}

	emit_signal("resources_reimported", p_files);
}

void EditorFileSystem::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_filesystem"), &EditorFileSystem::get_filesystem);
	ClassDB::bind_method(D_METHOD("is_scanning"), &EditorFileSystem::is_scanning);
	ClassDB::bind_method(D_METHOD("get_scanning_progress"), &EditorFileSystem::get_scanning_progress);
	ClassDB::bind_method(D_METHOD("scan"), &EditorFileSystem::scan);
	ClassDB::bind_method(D_METHOD("scan_sources"), &EditorFileSystem::scan_changes);
	ClassDB::bind_method(D_METHOD("update_file", "path"), &EditorFileSystem::update_file);
	ClassDB::bind_method(D_METHOD("get_filesystem_path", "path"), &EditorFileSystem::get_filesystem_path);
	ClassDB::bind_method(D_METHOD("get_file_type", "path"), &EditorFileSystem::get_file_type);

	ADD_SIGNAL(MethodInfo("filesystem_changed"));
	ADD_SIGNAL(MethodInfo("sources_changed", PropertyInfo(Variant::BOOL, "exist")));
	ADD_SIGNAL(MethodInfo("resources_reimported", PropertyInfo(Variant::POOL_STRING_ARRAY, "resources")));
}

void EditorFileSystem::_update_extensions() {

	valid_extensions.clear();
	import_extensions.clear();

	List<String> extensionsl;
	ResourceLoader::get_recognized_extensions_for_type("", &extensionsl);
	for (List<String>::Element *E = extensionsl.front(); E; E = E->next()) {

		valid_extensions.insert(E->get());
	}

	extensionsl.clear();
	ResourceFormatImporter::get_singleton()->get_recognized_extensions(&extensionsl);
	for (List<String>::Element *E = extensionsl.front(); E; E = E->next()) {

		import_extensions.insert(E->get());
	}
}

EditorFileSystem::EditorFileSystem() {

	reimport_on_missing_imported_files = GLOBAL_DEF("editor/reimport_missing_imported_files", true);

	singleton = this;
	filesystem = memnew(EditorFileSystemDirectory); //like, empty
	filesystem->parent = NULL;

	thread = NULL;
	scanning = false;
	importing = false;
	use_threads = true;
	thread_sources = NULL;
	new_filesystem = NULL;

	abort_scan = false;
	scanning_changes = false;
	scanning_changes_done = false;
	ResourceSaver::set_save_callback(_resource_saved);

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (da->change_dir("res://.import") != OK) {
		da->make_dir("res://.import");
	}
	memdelete(da);

	scan_total = 0;
}

EditorFileSystem::~EditorFileSystem() {
}
