/*************************************************************************/
/*  editor_file_system.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "globals.h"
#include "io/resource_loader.h"
#include "os/os.h"
#include "os/file_access.h"
#include "editor_node.h"
#include "io/resource_saver.h"
#include "editor_settings.h"
#include "editor_resource_preview.h"

EditorFileSystem *EditorFileSystem::singleton=NULL;

void EditorFileSystemDirectory::sort_files() {

	files.sort_custom<FileInfoSort>();
}

int EditorFileSystemDirectory::find_file_index(const String& p_file) const {

	for(int i=0;i<files.size();i++)	{
		if (files[i]->file==p_file)
			return i;
	}
	return -1;

}
int EditorFileSystemDirectory::find_dir_index(const String& p_dir) const{


	for(int i=0;i<subdirs.size();i++) {
		if (subdirs[i]->name==p_dir)
			return i;
	}

	return -1;
}


int EditorFileSystemDirectory::get_subdir_count() const {

	return subdirs.size();
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_subdir(int p_idx){

	ERR_FAIL_INDEX_V(p_idx,subdirs.size(),NULL);
	return subdirs[p_idx];

}

int EditorFileSystemDirectory::get_file_count() const{

	return files.size();
}

String EditorFileSystemDirectory::get_file(int p_idx) const{

	ERR_FAIL_INDEX_V(p_idx,files.size(),"");

	return files[p_idx]->file;
}

String EditorFileSystemDirectory::get_path() const {

	String p;
	const EditorFileSystemDirectory *d=this;
	while(d->parent) {
		p=d->name+"/"+p;
		d=d->parent;
	}

	return "res://"+p;

}


String EditorFileSystemDirectory::get_file_path(int p_idx) const {

	String file = get_file(p_idx);
	const EditorFileSystemDirectory *d=this;
	while(d->parent) {
		file=d->name+"/"+file;
		d=d->parent;
	}

	return "res://"+file;
}

bool EditorFileSystemDirectory::get_file_meta(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),"");
	return files[p_idx]->meta.enabled;
}

Vector<String> EditorFileSystemDirectory::get_file_deps(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),Vector<String>());
	return files[p_idx]->meta.deps;

}
Vector<String> EditorFileSystemDirectory::get_missing_sources(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),Vector<String>());
	Vector<String> missing;
	for(int i=0;i<files[p_idx]->meta.sources.size();i++) {
		if (files[p_idx]->meta.sources[i].missing)
			missing.push_back(files[p_idx]->meta.sources[i].path);
	}

	return missing;


}
bool EditorFileSystemDirectory::is_missing_sources(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),false);
	for(int i=0;i<files[p_idx]->meta.sources.size();i++) {
		if (files[p_idx]->meta.sources[i].missing)
			return true;
	}

	return false;
}

bool EditorFileSystemDirectory::have_sources_changed(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),false);
	return files[p_idx]->meta.sources_changed;

}

int EditorFileSystemDirectory::get_source_count(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),0);
	if (!files[p_idx]->meta.enabled)
		return 0;
	return files[p_idx]->meta.sources.size();
}
String EditorFileSystemDirectory::get_source_file(int p_idx,int p_source) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),String());
	ERR_FAIL_INDEX_V(p_source,files[p_idx]->meta.sources.size(),String());
	if (!files[p_idx]->meta.enabled)
		return String();

	return files[p_idx]->meta.sources[p_source].path;

}
bool EditorFileSystemDirectory::is_source_file_missing(int p_idx,int p_source) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),false);
	ERR_FAIL_INDEX_V(p_source,files[p_idx]->meta.sources.size(),false);
	if (!files[p_idx]->meta.enabled)
		return false;

	return files[p_idx]->meta.sources[p_source].missing;
}


StringName EditorFileSystemDirectory::get_file_type(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),"");
	return files[p_idx]->type;
}

String EditorFileSystemDirectory::get_name() {

	return name;
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_parent() {

	return parent;
}

void EditorFileSystemDirectory::_bind_methods() {

	ClassDB::bind_method(_MD("get_subdir_count"),&EditorFileSystemDirectory::get_subdir_count);
	ClassDB::bind_method(_MD("get_subdir","idx"),&EditorFileSystemDirectory::get_subdir);
	ClassDB::bind_method(_MD("get_file_count"),&EditorFileSystemDirectory::get_file_count);
	ClassDB::bind_method(_MD("get_file","idx"),&EditorFileSystemDirectory::get_file);
	ClassDB::bind_method(_MD("get_file_path","idx"),&EditorFileSystemDirectory::get_file_path);
	ClassDB::bind_method(_MD("get_file_type","idx"),&EditorFileSystemDirectory::get_file_type);
	ClassDB::bind_method(_MD("is_missing_sources","idx"),&EditorFileSystemDirectory::is_missing_sources);
	ClassDB::bind_method(_MD("get_name"),&EditorFileSystemDirectory::get_name);
	ClassDB::bind_method(_MD("get_path"),&EditorFileSystemDirectory::get_path);
	ClassDB::bind_method(_MD("get_parent:EditorFileSystemDirectory"),&EditorFileSystemDirectory::get_parent);
	ClassDB::bind_method(_MD("find_file_index","name"),&EditorFileSystemDirectory::find_file_index);
	ClassDB::bind_method(_MD("find_dir_index","name"),&EditorFileSystemDirectory::find_dir_index);


}


EditorFileSystemDirectory::EditorFileSystemDirectory() {

	modified_time=0;
	parent=NULL;
}

EditorFileSystemDirectory::~EditorFileSystemDirectory() {

	for(int i=0;i<files.size();i++) {

		memdelete(files[i]);
	}

	for(int i=0;i<subdirs.size();i++) {

		memdelete(subdirs[i]);
	}
}






EditorFileSystemDirectory::ImportMeta EditorFileSystem::_get_meta(const String& p_path) {

	Ref<ResourceImportMetadata> imd = ResourceLoader::load_import_metadata(p_path);
	EditorFileSystemDirectory::ImportMeta m;
	if (imd.is_null()) {
		m.enabled=false;
		m.sources_changed=false;
	} else {
		m.enabled=true;
		m.sources_changed=false;

		for(int i=0;i<imd->get_source_count();i++) {
			EditorFileSystemDirectory::ImportMeta::Source s;
			s.path=imd->get_source_path(i);
			s.md5=imd->get_source_md5(i);
			s.modified_time=0;
			s.missing=false;
			m.sources.push_back(s);
		}
		m.import_editor=imd->get_editor();
	}

	List<String> deps;
	ResourceLoader::get_dependencies(p_path,&deps);
	for(List<String>::Element *E=deps.front();E;E=E->next()) {
		m.deps.push_back(E->get());
	}

	return m;
}


void EditorFileSystem::_scan_filesystem() {

	ERR_FAIL_COND(!scanning || new_filesystem);

	//read .fscache
	String cpath;

	sources_changed.clear();
	file_cache.clear();

	String project=GlobalConfig::get_singleton()->get_resource_path();

	String fscache = EditorSettings::get_singleton()->get_project_settings_path().plus_file("filesystem_cache");
	FileAccess *f =FileAccess::open(fscache,FileAccess::READ);

	if (f) {
		//read the disk cache
		while(!f->eof_reached()) {

			String l = f->get_line().strip_edges();
			if (l==String())
				continue;

			if (l.begins_with("::")) {
				Vector<String> split = l.split("::");
				ERR_CONTINUE( split.size() != 3);
				String name = split[1];

				cpath=name;

			} else {
				Vector<String> split = l.split("::");
				ERR_CONTINUE( split.size() != 5);
				String name = split[0];
				String file;

				file=name;
				name=cpath.plus_file(name);

				FileCache fc;
				fc.type=split[1];
				fc.modification_time=split[2].to_int64();
				String meta = split[3].strip_edges();
				fc.meta.enabled=false;
				if (meta.find("<>")!=-1){
					Vector<String> spl = meta.split("<>");
					int sc = spl.size()-1;
					if (sc%3==0){
						fc.meta.enabled=true;
						fc.meta.import_editor=spl[0];
						fc.meta.sources.resize(sc/3);
						for(int i=0;i<fc.meta.sources.size();i++) {
							fc.meta.sources[i].path=spl[1+i*3+0];
							fc.meta.sources[i].md5=spl[1+i*3+1];
							fc.meta.sources[i].modified_time=spl[1+i*3+2].to_int64();
						}

					}

				}
				String deps = split[4].strip_edges();
				if (deps.length()) {
					Vector<String> dp = deps.split("<>");
					for(int i=0;i<dp.size();i++) {
						String path=dp[i];
						fc.meta.deps.push_back(path);
					}
				}

				file_cache[name]=fc;

			}

		}

		f->close();
		memdelete(f);
	}



	EditorProgressBG scan_progress("efs","ScanFS",1000);

	ScanProgress sp;
	sp.low=0;
	sp.hi=1;
	sp.progress=&scan_progress;


	new_filesystem = memnew( EditorFileSystemDirectory );
	new_filesystem->parent=NULL;

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	d->change_dir("res://");
	_scan_new_dir(new_filesystem,d,sp);

	file_cache.clear(); //clear caches, no longer needed

	memdelete(d);


	//save back the findings
//	String fscache = EditorSettings::get_singleton()->get_project_settings_path().plus_file("file_cache");

	f=FileAccess::open(fscache,FileAccess::WRITE);
	_save_filesystem_cache(new_filesystem,f);
	f->close();
	memdelete(f);

	scanning=false;

}



void EditorFileSystem::_thread_func(void *_userdata) {

	EditorFileSystem *sd = (EditorFileSystem*)_userdata;
	sd->_scan_filesystem();

}

bool EditorFileSystem::_update_scan_actions() {

	sources_changed.clear();

	bool fs_changed=false;

	for (List<ItemAction>::Element *E=scan_actions.front();E;E=E->next()) {

		ItemAction&ia = E->get();

		switch(ia.action) {
			case ItemAction::ACTION_NONE: {

			} break;
			case ItemAction::ACTION_DIR_ADD: {

				//print_line("*ACTION ADD DIR: "+ia.new_dir->get_name());
				int idx=0;
				for(int i=0;i<ia.dir->subdirs.size();i++) {

					if (ia.new_dir->name<ia.dir->subdirs[i]->name)
						break;
					idx++;
				}
				if (idx==ia.dir->subdirs.size()) {
					ia.dir->subdirs.push_back(ia.new_dir);
				} else {
					ia.dir->subdirs.insert(idx,ia.new_dir);
				}

				fs_changed=true;
			} break;
			case ItemAction::ACTION_DIR_REMOVE: {

				ERR_CONTINUE(!ia.dir->parent);
				//print_line("*ACTION REMOVE DIR: "+ia.dir->get_name());
				ia.dir->parent->subdirs.erase(ia.dir);
				memdelete( ia.dir );
				fs_changed=true;
			} break;
			case ItemAction::ACTION_FILE_ADD: {

				int idx=0;
				for(int i=0;i<ia.dir->files.size();i++) {

					if (ia.new_file->file<ia.dir->files[i]->file)
						break;
					idx++;
				}
				if (idx==ia.dir->files.size()) {
					ia.dir->files.push_back(ia.new_file);
				} else {
					ia.dir->files.insert(idx,ia.new_file);
				}

				fs_changed=true;
				//print_line("*ACTION ADD FILE: "+ia.new_file->file);

			} break;
			case ItemAction::ACTION_FILE_REMOVE: {

				int idx = ia.dir->find_file_index(ia.file);
				ERR_CONTINUE(idx==-1);
				memdelete( ia.dir->files[idx] );
				ia.dir->files.remove(idx);

				fs_changed=true;
				//print_line("*ACTION REMOVE FILE: "+ia.file);

			} break;
			case ItemAction::ACTION_FILE_SOURCES_CHANGED: {

				int idx = ia.dir->find_file_index(ia.file);
				ERR_CONTINUE(idx==-1);
				String full_path = ia.dir->get_file_path(idx);
				sources_changed.push_back(full_path);

			} break;

		}
	}

	scan_actions.clear();

	return fs_changed;

}

void EditorFileSystem::scan() {

    if (false /*&& bool(Globals::get_singleton()->get("debug/disable_scan"))*/)
           return;

	if (scanning || scanning_sources|| thread)
		return;


	abort_scan=false;
	if (!use_threads) {
		scanning=true;
		scan_total=0;
		_scan_filesystem();
		if (filesystem)
			memdelete(filesystem);
//		file_type_cache.clear();
		filesystem=new_filesystem;
		new_filesystem=NULL;
		_update_scan_actions();
		scanning=false;
		emit_signal("filesystem_changed");
		emit_signal("sources_changed",sources_changed.size()>0);

	} else {

		ERR_FAIL_COND(thread);
		set_process(true);
		Thread::Settings s;
		scanning=true;
		scan_total=0;
		s.priority=Thread::PRIORITY_LOW;
		thread = Thread::create(_thread_func,this,s);
		//tree->hide();
		//progress->show();

	}



}

bool EditorFileSystem::_check_meta_sources(EditorFileSystemDirectory::ImportMeta & p_meta) {

	if (p_meta.enabled) {

		for(int j=0;j<p_meta.sources.size();j++) {


			String src = EditorImportPlugin::expand_source_path(p_meta.sources[j].path);

			if (!FileAccess::exists(src)) {
				p_meta.sources[j].missing=true;
				continue;
			}

			p_meta.sources[j].missing=false;

			uint64_t mt = FileAccess::get_modified_time(src);

			if (mt!=p_meta.sources[j].modified_time) {
				//scan
				String md5 = FileAccess::get_md5(src);
				//print_line("checking: "+src);
				//print_line("md5: "+md5);
				//print_line("vs: "+p_meta.sources[j].md5);
				if (md5!=p_meta.sources[j].md5) {
					//really changed
					return true;
				}
				p_meta.sources[j].modified_time=mt;
			}
		}
	}

	return false;
}

void EditorFileSystem::ScanProgress::update(int p_current,int p_total) const {

	float ratio = low + ((hi-low)/p_total)*p_current;
	progress->step(ratio*1000);
	EditorFileSystem::singleton->scan_total=ratio;
}

EditorFileSystem::ScanProgress EditorFileSystem::ScanProgress::get_sub(int p_current,int p_total) const{

	ScanProgress sp=*this;
	float slice = (sp.hi-sp.low)/p_total;
	sp.low+=slice*p_current;
	sp.hi=slice;
	return sp;


}


void EditorFileSystem::_scan_new_dir(EditorFileSystemDirectory *p_dir,DirAccess *da,const ScanProgress& p_progress) {

	List<String> dirs;
	List<String> files;

	String cd = da->get_current_dir();

	p_dir->modified_time = FileAccess::get_modified_time(cd);


	da->list_dir_begin();
	while (true) {

		bool isdir;
		String f = da->get_next(&isdir);
		if (f=="")
			break;

		if (isdir) {

			if (f.begins_with(".")) //ignore hidden and . / ..
				continue;

			if (FileAccess::exists(cd.plus_file(f).plus_file("engine.cfg"))) // skip if another project inside this
				continue;

			dirs.push_back(f);

		} else {

			files.push_back(f);
		}

	}

	da->list_dir_end();

	dirs.sort();
	files.sort();

	int total = dirs.size()+files.size();
	int idx=0;


	for (List<String>::Element *E=dirs.front();E;E=E->next(),idx++) {

		if (da->change_dir(E->get())==OK) {

			String d = da->get_current_dir();

			if (d==cd || !d.begins_with(cd)) {
				da->change_dir(cd); //avoid recursion
			} else {


				EditorFileSystemDirectory *efd = memnew( EditorFileSystemDirectory );

				efd->parent=p_dir;
				efd->name=E->get();

				_scan_new_dir(efd,da,p_progress.get_sub(idx,total));

				int idx=0;
				for(int i=0;i<p_dir->subdirs.size();i++) {

					if (efd->name<p_dir->subdirs[i]->name)
						break;
					idx++;
				}
				if (idx==p_dir->subdirs.size()) {
					p_dir->subdirs.push_back(efd);
				} else {
					p_dir->subdirs.insert(idx,efd);
				}

				da->change_dir("..");
			}
		} else {
			ERR_PRINTS("Cannot go into subdir: "+E->get());
		}

		p_progress.update(idx,total);

	}

	for (List<String>::Element*E=files.front();E;E=E->next(),idx++) {

		String ext = E->get().get_extension().to_lower();
		if (!valid_extensions.has(ext))
			continue; //invalid

		EditorFileSystemDirectory::FileInfo *fi = memnew( EditorFileSystemDirectory::FileInfo );
		fi->file=E->get();

		String path = cd.plus_file(fi->file);

		FileCache *fc = file_cache.getptr(path);
		uint64_t mt = FileAccess::get_modified_time(path);

		if (fc && fc->modification_time == mt) {

			fi->meta=fc->meta;
			fi->type=fc->type;
			fi->modified_time=fc->modification_time;
		} else {
			fi->meta=_get_meta(path);
			fi->type=ResourceLoader::get_resource_type(path);
			fi->modified_time=mt;

		}

		if (fi->meta.enabled) {
			if (_check_meta_sources(fi->meta)) {
				ItemAction ia;
				ia.action=ItemAction::ACTION_FILE_SOURCES_CHANGED;
				ia.dir=p_dir;
				ia.file=E->get();
				scan_actions.push_back(ia);
				fi->meta.sources_changed=true;
			} else {
				fi->meta.sources_changed=false;
			}

		} else {
			fi->meta.sources_changed=true;
		}

		p_dir->files.push_back(fi);
		p_progress.update(idx,total);
	}

}


void EditorFileSystem::_scan_fs_changes(EditorFileSystemDirectory *p_dir,const ScanProgress& p_progress) {

	uint64_t current_mtime = FileAccess::get_modified_time(p_dir->get_path());

	bool updated_dir=false;

	//print_line("dir: "+p_dir->get_path()+" MODTIME: "+itos(p_dir->modified_time)+" CTIME: "+itos(current_mtime));

	if (current_mtime!=p_dir->modified_time) {

		updated_dir=true;
		p_dir->modified_time=current_mtime;
		//ooooops, dir changed, see what's going on

		//first mark everything as veryfied

		for(int i=0;i<p_dir->files.size();i++) {

			p_dir->files[i]->verified=false;
		}

		for(int i=0;i<p_dir->subdirs.size();i++) {

			p_dir->get_subdir(i)->verified=false;
		}

		//then scan files and directories and check what's different

		DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		String cd = p_dir->get_path();
		da->change_dir(cd);
		da->list_dir_begin();
		while (true) {

			bool isdir;
			String f = da->get_next(&isdir);
			if (f=="")
				break;

			if (isdir) {

				if (f.begins_with(".")) //ignore hidden and . / ..
					continue;

				int idx = p_dir->find_dir_index(f);
				if (idx==-1) {

					if (FileAccess::exists(cd.plus_file(f).plus_file("engine.cfg"))) // skip if another project inside this
						continue;

					EditorFileSystemDirectory *efd = memnew( EditorFileSystemDirectory );

					efd->parent=p_dir;
					efd->name=f;
					DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
					d->change_dir(cd.plus_file(f));
					_scan_new_dir(efd,d,p_progress.get_sub(1,1));
					memdelete(d);


					ItemAction ia;
					ia.action=ItemAction::ACTION_DIR_ADD;
					ia.dir=p_dir;
					ia.file=f;
					ia.new_dir=efd;
					scan_actions.push_back(ia);
				} else {
					p_dir->subdirs[idx]->verified=true;
				}


			} else {
				String ext = f.get_extension().to_lower();
				if (!valid_extensions.has(ext))
					continue; //invalid

				int idx = p_dir->find_file_index(f);

				if (idx==-1) {
					//never seen this file, add actition to add it
					EditorFileSystemDirectory::FileInfo *fi = memnew( EditorFileSystemDirectory::FileInfo );
					fi->file=f;

					String path = cd.plus_file(fi->file);
					fi->modified_time=FileAccess::get_modified_time(path);
					fi->meta=_get_meta(path);
					fi->type=ResourceLoader::get_resource_type(path);

					{
						ItemAction ia;
						ia.action=ItemAction::ACTION_FILE_ADD;
						ia.dir=p_dir;
						ia.file=f;
						ia.new_file=fi;
						scan_actions.push_back(ia);
					}

					//take the chance and scan sources
					if (_check_meta_sources(fi->meta)) {

						ItemAction ia;
						ia.action=ItemAction::ACTION_FILE_SOURCES_CHANGED;
						ia.dir=p_dir;
						ia.file=f;
						scan_actions.push_back(ia);
						fi->meta.sources_changed=true;
					} else {
						fi->meta.sources_changed=false;
					}

				} else {
					p_dir->files[idx]->verified=true;
				}


			}

		}

		da->list_dir_end();
		memdelete(da);

	}

	for(int i=0;i<p_dir->files.size();i++) {

		if (updated_dir && !p_dir->files[i]->verified) {
			//this file was removed, add action to remove it
			ItemAction ia;
			ia.action=ItemAction::ACTION_FILE_REMOVE;
			ia.dir=p_dir;
			ia.file=p_dir->files[i]->file;
			scan_actions.push_back(ia);
			continue;

		}

		if (_check_meta_sources(p_dir->files[i]->meta)) {
			ItemAction ia;
			ia.action=ItemAction::ACTION_FILE_SOURCES_CHANGED;
			ia.dir=p_dir;
			ia.file=p_dir->files[i]->file;
			scan_actions.push_back(ia);
			p_dir->files[i]->meta.sources_changed=true;
		} else {
			p_dir->files[i]->meta.sources_changed=false;
		}

		EditorResourcePreview::get_singleton()->check_for_invalidation(p_dir->get_file_path(i));
	}

	for(int i=0;i<p_dir->subdirs.size();i++) {

		if (updated_dir && !p_dir->subdirs[i]->verified) {
			//this directory was removed, add action to remove it
			ItemAction ia;
			ia.action=ItemAction::ACTION_DIR_REMOVE;
			ia.dir=p_dir->subdirs[i];
			scan_actions.push_back(ia);
			continue;

		}
		_scan_fs_changes(p_dir->get_subdir(i),p_progress);
	}

}

void EditorFileSystem::_thread_func_sources(void *_userdata) {

	EditorFileSystem *efs = (EditorFileSystem*)_userdata;
	if (efs->filesystem) {
		EditorProgressBG pr("sources",TTR("ScanSources"),1000);
		ScanProgress sp;
		sp.progress=&pr;
		sp.hi=1;
		sp.low=0;
		efs->_scan_fs_changes(efs->filesystem,sp);
	}
	efs->scanning_sources_done=true;
}

void EditorFileSystem::get_changed_sources(List<String> *r_changed) {

	*r_changed=sources_changed;
}

void EditorFileSystem::scan_sources() {

	if (scanning || scanning_sources|| thread)
		return;

	sources_changed.clear();
	scanning_sources=true;
	scanning_sources_done=false;

	abort_scan=false;

	if (!use_threads) {
		if (filesystem) {
			EditorProgressBG pr("sources",TTR("ScanSources"),1000);
			ScanProgress sp;
			sp.progress=&pr;
			sp.hi=1;
			sp.low=0;
			scan_total=0;
			_scan_fs_changes(filesystem,sp);
			if (_update_scan_actions())
				emit_signal("filesystem_changed");
		}
		scanning_sources=false;
		scanning_sources_done=true;
		emit_signal("sources_changed",sources_changed.size()>0);
	} else {

		ERR_FAIL_COND(thread_sources);
		set_process(true);
		scan_total=0;
		Thread::Settings s;
		s.priority=Thread::PRIORITY_LOW;
		thread_sources = Thread::create(_thread_func_sources,this,s);
		//tree->hide();
		//print_line("SCAN BEGIN!");
		//progress->show();
	}



}

void EditorFileSystem::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {


			   scan();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (use_threads && thread) {
				//abort thread if in progress
				abort_scan=true;
				while(scanning)	{
					OS::get_singleton()->delay_usec(1000);
				}
				Thread::wait_to_finish(thread);
				memdelete(thread);
				thread=NULL;
				WARN_PRINTS("Scan thread aborted...");
				set_process(false);

			}

			if (filesystem)
				memdelete(filesystem);
			if (new_filesystem)
				memdelete(new_filesystem);
			filesystem=NULL;
			new_filesystem=NULL;

		} break;
		case NOTIFICATION_PROCESS: {

			if (use_threads) {

				if (scanning_sources) {

					if (scanning_sources_done) {

						scanning_sources=false;

						set_process(false);

						Thread::wait_to_finish(thread_sources);
						memdelete(thread_sources);
						thread_sources=NULL;
						if (_update_scan_actions())
							emit_signal("filesystem_changed");
						//print_line("sources changed: "+itos(sources_changed.size()));
						emit_signal("sources_changed",sources_changed.size()>0);
					}
				} else if (!scanning) {

					set_process(false);

					if (filesystem)
						memdelete(filesystem);
					filesystem=new_filesystem;
					new_filesystem=NULL;
					Thread::wait_to_finish(thread);
					memdelete(thread);
					thread=NULL;
					_update_scan_actions();
					emit_signal("filesystem_changed");
					emit_signal("sources_changed",sources_changed.size()>0);
					//print_line("initial sources changed: "+itos(sources_changed.size()));



				} else {
					//progress->set_text("Scanning...\n"+itos(total*100)+"%");
				}
			}
		} break;
	}

}

bool EditorFileSystem::is_scanning() const {

	return scanning;
}
float EditorFileSystem::get_scanning_progress() const {

	return scan_total;
}

EditorFileSystemDirectory *EditorFileSystem::get_filesystem() {

	return filesystem;
}

void EditorFileSystem::_save_filesystem_cache(EditorFileSystemDirectory*p_dir,FileAccess *p_file) {


	if (!p_dir)
		return; //none
	p_file->store_line("::"+p_dir->get_path()+"::"+String::num(p_dir->modified_time));

	for(int i=0;i<p_dir->files.size();i++) {

		String s=p_dir->files[i]->file+"::"+p_dir->files[i]->type+"::"+String::num(p_dir->files[i]->modified_time)+"::";
		if (p_dir->files[i]->meta.enabled) {
			s+=p_dir->files[i]->meta.import_editor;
			for(int j=0;j<p_dir->files[i]->meta.sources.size();j++){
				s+="<>"+p_dir->files[i]->meta.sources[j].path;
				s+="<>"+p_dir->files[i]->meta.sources[j].md5;
				s+="<>"+String::num(p_dir->files[i]->meta.sources[j].modified_time);

			}
		}
		s+="::";
		for(int j=0;j<p_dir->files[i]->meta.deps.size();j++) {

			if (j>0)
				s+="<>";
			s+=p_dir->files[i]->meta.deps[j];
		}

		p_file->store_line(s);
	}

	for(int i=0;i<p_dir->subdirs.size();i++) {

		_save_filesystem_cache(p_dir->subdirs[i],p_file);
	}

}






bool EditorFileSystem::_find_file(const String& p_file,EditorFileSystemDirectory ** r_d, int &r_file_pos) const {
	//todo make faster

	if (!filesystem || scanning)
		return false;


	String f = GlobalConfig::get_singleton()->localize_path(p_file);

	if (!f.begins_with("res://"))
		return false;
	f=f.substr(6,f.length());
	f=f.replace("\\","/");

	Vector<String> path = f.split("/");

	if (path.size()==0)
		return false;
	String file=path[path.size()-1];
	path.resize(path.size()-1);

	EditorFileSystemDirectory *fs=filesystem;

	for(int i=0;i<path.size();i++) {


		int idx=-1;
		for(int j=0;j<fs->get_subdir_count();j++) {

			if (fs->get_subdir(j)->get_name()==path[i]) {
				idx=j;
				break;
			}
		}

		if (idx==-1) {
			//does not exist, create i guess?
			EditorFileSystemDirectory *efsd = memnew( EditorFileSystemDirectory );
			efsd->name=path[i];
			int idx2=0;
			for(int j=0;j<fs->get_subdir_count();j++) {

				if (efsd->name<fs->get_subdir(j)->get_name())
					break;
				idx2++;
			}

			if (idx2==fs->get_subdir_count())
				fs->subdirs.push_back(efsd);
			else
				fs->subdirs.insert(idx2,efsd);
			fs=efsd;
		} else {

			fs=fs->get_subdir(idx);
		}
	}


	int cpos=-1;
	for(int i=0;i<fs->files.size();i++) {

		if (fs->files[i]->file==file) {
			cpos=i;
			break;
		}
	}

	r_file_pos=cpos;
	*r_d=fs;

	if (cpos!=-1) {

		return true;
	}  else {

		return false;
	}


}

String EditorFileSystem::get_file_type(const String& p_file) const {

    EditorFileSystemDirectory *fs=NULL;
    int cpos=-1;

    if (!_find_file(p_file,&fs,cpos)) {

	return "";
    }


    return fs->files[cpos]->type;

}

EditorFileSystemDirectory* EditorFileSystem::find_file(const String& p_file,int* r_index) const {

	if (!filesystem || scanning)
	    return NULL;

	EditorFileSystemDirectory *fs=NULL;
	int cpos=-1;
	if (!_find_file(p_file,&fs,cpos)) {

	    return NULL;
	}


	if (r_index)
		*r_index=cpos;

	return fs;
}


EditorFileSystemDirectory *EditorFileSystem::get_path(const String& p_path) {

    if (!filesystem || scanning)
    	return NULL;


    String f = GlobalConfig::get_singleton()->localize_path(p_path);

    if (!f.begins_with("res://"))
    	return NULL;


    f=f.substr(6,f.length());
    f=f.replace("\\","/");
    if (f==String())
    	return filesystem;

    if (f.ends_with("/"))
	f=f.substr(0,f.length()-1);

    Vector<String> path = f.split("/");

    if (path.size()==0)
    	return NULL;

    EditorFileSystemDirectory *fs=filesystem;

    for(int i=0;i<path.size();i++) {


	int idx=-1;
	for(int j=0;j<fs->get_subdir_count();j++) {

	    if (fs->get_subdir(j)->get_name()==path[i]) {
		idx=j;
		break;
	    }
	}

	if (idx==-1) {
		return NULL;
	} else {

	    fs=fs->get_subdir(idx);
	}
    }

    return fs;
}

void EditorFileSystem::_resource_saved(const String& p_path){


	//print_line("resource saved: "+p_path);
	EditorFileSystem::get_singleton()->update_file(p_path);

}

String EditorFileSystem::_find_first_from_source(EditorFileSystemDirectory* p_dir,const String &p_src) const {

	for(int i=0;i<p_dir->files.size();i++) {
		for(int j=0;j<p_dir->files[i]->meta.sources.size();j++) {

			if (p_dir->files[i]->meta.sources[j].path==p_src)
				return p_dir->get_file_path(i);
		}
	}

	for(int i=0;i<p_dir->subdirs.size();i++) {

		String ret = _find_first_from_source(p_dir->subdirs[i],p_src);
		if (ret.length()>0)
			return ret;
	}

	return String();
}


String EditorFileSystem::find_resource_from_source(const String& p_path) const {


	if (filesystem)
		return _find_first_from_source(filesystem,p_path);
	return String();
}

void EditorFileSystem::update_file(const String& p_file) {

    EditorFileSystemDirectory *fs=NULL;
    int cpos=-1;

    if (!_find_file(p_file,&fs,cpos)) {

	if (!fs)
		return;
    }

    if (!FileAccess::exists(p_file)) {
	    //was removed
	    memdelete( fs->files[cpos] );
	    fs->files.remove(cpos);
	    call_deferred("emit_signal","filesystem_changed"); //update later
	    return;

    }

    String type = ResourceLoader::get_resource_type(p_file);

    if (cpos==-1) {

	    int idx=0;

	    for(int i=0;i<fs->files.size();i++) {
		if (p_file<fs->files[i]->file)
		    break;
		idx++;
	    }

	    EditorFileSystemDirectory::FileInfo *fi = memnew( EditorFileSystemDirectory::FileInfo );
	    fi->file=p_file.get_file();

	    if (idx==fs->files.size()) {
		fs->files.push_back(fi);
	    } else {

		fs->files.insert(idx,fi);
	    }
	    cpos=idx;


    }

	//print_line("UPDATING: "+p_file);
	fs->files[cpos]->type=type;
	fs->files[cpos]->modified_time=FileAccess::get_modified_time(p_file);
	fs->files[cpos]->meta=_get_meta(p_file);

	EditorResourcePreview::get_singleton()->call_deferred("check_for_invalidation",p_file);
	call_deferred("emit_signal","filesystem_changed"); //update later

}



void EditorFileSystem::_bind_methods() {


	ClassDB::bind_method(_MD("get_filesystem:EditorFileSystemDirectory"),&EditorFileSystem::get_filesystem);
	ClassDB::bind_method(_MD("is_scanning"),&EditorFileSystem::is_scanning);
	ClassDB::bind_method(_MD("get_scanning_progress"),&EditorFileSystem::get_scanning_progress);
	ClassDB::bind_method(_MD("scan"),&EditorFileSystem::scan);
	ClassDB::bind_method(_MD("scan_sources"),&EditorFileSystem::scan_sources);
	ClassDB::bind_method(_MD("update_file","path"),&EditorFileSystem::update_file);
	ClassDB::bind_method(_MD("get_path:EditorFileSystemDirectory","path"),&EditorFileSystem::get_path);
	ClassDB::bind_method(_MD("get_file_type","path"),&EditorFileSystem::get_file_type);

	ADD_SIGNAL( MethodInfo("filesystem_changed") );
	ADD_SIGNAL( MethodInfo("sources_changed",PropertyInfo(Variant::BOOL,"exist")) );

}



EditorFileSystem::EditorFileSystem() {


	singleton=this;
	filesystem=memnew( EditorFileSystemDirectory ); //like, empty

	thread = NULL;
	scanning=false;
	use_threads=true;
	thread_sources=NULL;
	new_filesystem=NULL;

	scanning_sources=false;
	ResourceSaver::set_save_callback(_resource_saved);

	List<String> extensionsl;
	ResourceLoader::get_recognized_extensions_for_type("",&extensionsl);
	for(List<String>::Element *E = extensionsl.front();E;E=E->next()) {

		valid_extensions.insert(E->get());
	}


	scan_total=0;
}

EditorFileSystem::~EditorFileSystem() {


}
