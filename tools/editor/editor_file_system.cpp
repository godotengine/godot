/*************************************************************************/
/*  editor_file_system.cpp                                               */
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
#include "editor_file_system.h"
#include "globals.h"
#include "io/resource_loader.h"
#include "os/os.h"
#include "os/file_access.h"
#include "editor_node.h"
#include "io/resource_saver.h"


EditorFileSystem *EditorFileSystem::singleton=NULL;


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

	return files[p_idx].file;
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
	return files[p_idx].meta.enabled;
}

Vector<String> EditorFileSystemDirectory::get_missing_sources(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),Vector<String>());
	Vector<String> missing;
	for(int i=0;i<files[p_idx].meta.sources.size();i++) {
		if (files[p_idx].meta.sources[i].missing)
			missing.push_back(files[p_idx].meta.sources[i].path);
	}

	return missing;


}
bool EditorFileSystemDirectory::is_missing_sources(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),false);
	for(int i=0;i<files[p_idx].meta.sources.size();i++) {
		if (files[p_idx].meta.sources[i].missing)
			return true;
	}

	return false;
}

String EditorFileSystemDirectory::get_file_type(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx,files.size(),"");
	return files[p_idx].type;
}

String EditorFileSystemDirectory::get_name() {

	return name;
}

EditorFileSystemDirectory *EditorFileSystemDirectory::get_parent() {

	return parent;
}

void EditorFileSystemDirectory::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_subdir_count"),&EditorFileSystemDirectory::get_subdir_count);
	ObjectTypeDB::bind_method(_MD("get_subdir","idx"),&EditorFileSystemDirectory::get_subdir);
	ObjectTypeDB::bind_method(_MD("get_file_count"),&EditorFileSystemDirectory::get_file_count);
	ObjectTypeDB::bind_method(_MD("get_file","idx"),&EditorFileSystemDirectory::get_file);
	ObjectTypeDB::bind_method(_MD("get_file_path","idx"),&EditorFileSystemDirectory::get_file_path);
	ObjectTypeDB::bind_method(_MD("get_file_types","idx"),&EditorFileSystemDirectory::get_file_type);
	ObjectTypeDB::bind_method(_MD("is_missing_sources","idx"),&EditorFileSystemDirectory::is_missing_sources);
	ObjectTypeDB::bind_method(_MD("get_name"),&EditorFileSystemDirectory::get_name);
	ObjectTypeDB::bind_method(_MD("get_parent"),&EditorFileSystemDirectory::get_parent);

}


EditorFileSystemDirectory::EditorFileSystemDirectory() {

	parent=NULL;
}

EditorFileSystemDirectory::~EditorFileSystemDirectory() {

	for(int i=0;i<subdirs.size();i++) {

		memdelete(subdirs[i]);
	}
}








EditorFileSystem::DirItem::~DirItem() {

	for(int i=0;i<dirs.size();i++) {
		memdelete(dirs[i]);
	}

	for(int i=0;i<files.size();i++) {
		memdelete(files[i]);
	}
}

EditorFileSystemDirectory::ImportMeta EditorFileSystem::_get_meta(const String& p_path) {

	Ref<ResourceImportMetadata> imd = ResourceLoader::load_import_metadata(p_path);
	EditorFileSystemDirectory::ImportMeta m;
	if (imd.is_null()) {
		m.enabled=false;
	} else {
		m.enabled=true;
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
	return m;
}

EditorFileSystem::DirItem* EditorFileSystem::_scan_dir(DirAccess *da,Set<String> &extensions,String p_name,float p_from,float p_range,const String& p_path,HashMap<String,FileCache> &file_cache,HashMap<String,DirCache> &dir_cache,EditorProgressBG& p_prog) {

	if (abort_scan)
		return NULL;

	if (p_path!=String()) {
		if (FileAccess::exists(("res://"+p_path).plus_file("engine.cfg"))) {
			return NULL;
		}
	}

	List<String> dirs;
	List<String> files;
	Set<String> pngs;

	String path=p_path;
	if (path.ends_with("/"))
		path=path.substr(0,path.length()-1);
	String global_path = Globals::get_singleton()->get_resource_path().plus_file(path);

	path="res://"+path;
	uint64_t mtime = FileAccess::get_modified_time(global_path);

	DirCache *dc = dir_cache.getptr(path);


	if (false && dc && dc->modification_time==mtime) {
		//use the cached files, since directory did not change
		for (Set<String>::Element *E=dc->subdirs.front();E;E=E->next()) {
			dirs.push_back(E->get());
		}
		for (Set<String>::Element *E=dc->files.front();E;E=E->next()) {
			files.push_back(E->get());
		}

	} else {
		//use the filesystem, some files may have changed
		Error err = da->change_dir(global_path);
		if (err!=OK) {
			print_line("Can't change to: "+path);
			ERR_FAIL_COND_V(err!=OK,NULL);
		}


		da->list_dir_begin();
		while (true) {

			bool isdir;
			String f = da->get_next(&isdir);
			if (f=="")
				break;
			if (isdir) {
				dirs.push_back(f);
			} else {
				String ext = f.extension().to_lower();
				if (extensions.has(ext))
					files.push_back(f);

			}

		}

		da->list_dir_end();
		files.sort();
		dirs.sort();

	}



	//print_line(da->get_current_dir()+": dirs: "+itos(dirs.size())+" files:"+itos(files.size()) );

	//find subdirs
	Vector<DirItem*> subdirs;

	//String current = da->get_current_dir();
	float idx=0;
	for (List<String>::Element *E=dirs.front();E;E=E->next(),idx+=1.0) {

		String d = E->get();
		if (d.begins_with(".")) //ignore hidden and . / ..
			continue;

		//ERR_CONTINUE( da->change_dir(d)!= OK );
		DirItem *sdi = _scan_dir(da,extensions,d,p_from+(idx/dirs.size())*p_range,p_range/dirs.size(),p_path+d+"/",file_cache,dir_cache,p_prog);
		if (sdi) {
			subdirs.push_back(sdi);
		}
		//da->change_dir(current);
	}


	if (subdirs.empty() && files.empty()) {
		total=p_from+p_range;
		p_prog.step(total*100);
		return NULL; //give up, nothing to do here
	}

	DirItem *di = memnew( DirItem );
	di->path=path;
	di->name=p_name;
	di->dirs=subdirs;
	di->modified_time=mtime;

	//add files
	for (List<String>::Element *E=files.front();E;E=E->next()) {

		SceneItem * si = memnew( SceneItem );
		si->file=E->get();
		si->path="res://"+p_path+si->file;
		FileCache *fc = file_cache.getptr(si->path);
		uint64_t mt = FileAccess::get_modified_time(si->path);

		if (fc && fc->modification_time == mt) {

			si->meta=fc->meta;
			si->type=fc->type;
			si->modified_time=fc->modification_time;
		} else {
			si->meta=_get_meta(si->path);
			si->type=ResourceLoader::get_resource_type(si->path);
			si->modified_time=mt;

		}

		if (si->meta.enabled) {
			md_count++;
			if (_check_meta_sources(si->meta)) {
				sources_changed.push_back(si->path);
			}
		}
		di->files.push_back(si);
	}

	total=p_from+p_range;
	p_prog.step(total*100);

	return di;
}


void EditorFileSystem::_scan_scenes() {

	ERR_FAIL_COND(!scanning || scandir);

	//read .fscache
	HashMap<String,FileCache> file_cache;
	HashMap<String,DirCache> dir_cache;
	DirCache *dc=NULL;
	String cpath;

	sources_changed.clear();



	String project=Globals::get_singleton()->get_resource_path();
	FileAccess *f =FileAccess::open(project+"/.fscache",FileAccess::READ);

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

				dir_cache[name]=DirCache();
				dc=&dir_cache[name];
				dc->modification_time=split[2].to_int64();

				if (name!="res://") {

					cpath=name+"/";

					int sp=name.find_last("/");
					if (sp==5)
						sp=6;
					String pd = name.substr(0,sp);
					DirCache *dcp = dir_cache.getptr(pd);
					ERR_CONTINUE(!dcp);
					dcp->subdirs.insert(name.get_file());
				} else {

					cpath=name;
				}


			} else {
				Vector<String> split = l.split("::");
				ERR_CONTINUE( split.size() != 4);
				String name = split[0];
				String file;

				if (!name.begins_with("res://")) {
					file=name;
					name=cpath+name;
				} else {
					file=name.get_file();
				}

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
				file_cache[name]=fc;

				ERR_CONTINUE(!dc);
				dc->files.insert(file);
			}

		}

		f->close();
		memdelete(f);
	}






	total=0;
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	//da->change_dir( Globals::get_singleton()->get_resource_path() );


	List<String> extensionsl;
	ResourceLoader::get_recognized_extensions_for_type("",&extensionsl);
	Set<String> extensions;
	for(List<String>::Element *E = extensionsl.front();E;E=E->next()) {

		extensions.insert(E->get());
	}

	EditorProgressBG scan_progress("efs","ScanFS",100);

	md_count=0;
	scandir=_scan_dir(da,extensions,"",0,1,"",file_cache,dir_cache,scan_progress);
	memdelete(da);
	if (abort_scan && scandir) {
		memdelete(scandir);
		scandir=NULL;

	}


	//save back the findings
	f=FileAccess::open(project+"/.fscache",FileAccess::WRITE);
	_save_type_cache_fs(scandir,f);
	f->close();
	memdelete(f);

	scanning=false;

}



void EditorFileSystem::_thread_func(void *_userdata) {

	EditorFileSystem *sd = (EditorFileSystem*)_userdata;
	sd->_scan_scenes();

}

void EditorFileSystem::scan() {

    if (bool(Globals::get_singleton()->get("debug/disable_scan")))
           return;

	if (scanning || scanning_sources|| thread)
		return;


	abort_scan=false;
	if (!use_threads) {
		scanning=true;
		_scan_scenes();
		if (rootdir)
			memdelete(rootdir);
		rootdir=scandir;
		if (filesystem)
			memdelete(filesystem);
//		file_type_cache.clear();
		filesystem=_update_tree(rootdir);

		if (rootdir)
			memdelete(rootdir);
		rootdir=NULL;
		scanning=false;
		emit_signal("filesystem_changed");
		emit_signal("sources_changed",sources_changed.size()>0);

	} else {

		ERR_FAIL_COND(thread);
		set_process(true);
		Thread::Settings s;
		scanning=true;
		s.priority=Thread::PRIORITY_LOW;
		thread = Thread::create(_thread_func,this,s);
		//tree->hide();
		//progress->show();
	}



}

bool EditorFileSystem::_check_meta_sources(EditorFileSystemDirectory::ImportMeta & p_meta,EditorProgressBG *ep) {

	if (p_meta.enabled) {

		if (ep) {
			ep->step(ss_amount++);
		}

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
				print_line("checking: "+src);
				print_line("md5: "+md5);
				print_line("vs: "+p_meta.sources[j].md5);
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

void EditorFileSystem::_scan_sources(EditorFileSystemDirectory *p_dir,EditorProgressBG *ep) {

	for(int i=0;i<p_dir->files.size();i++) {

		if (_check_meta_sources(p_dir->files[i].meta,ep)) {
			sources_changed.push_back(p_dir->get_file_path(i));
		}
	}

	for(int i=0;i<p_dir->subdirs.size();i++) {

		_scan_sources(p_dir->get_subdir(i),ep);
	}

}

void EditorFileSystem::_thread_func_sources(void *_userdata) {

	EditorFileSystem *efs = (EditorFileSystem*)_userdata;
	if (efs->filesystem) {
		EditorProgressBG pr("sources","ScanSources",efs->md_count);
		efs->ss_amount=0;
		efs->_scan_sources(efs->filesystem,&pr);
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
		if (filesystem)
			_scan_sources(filesystem,NULL);
		scanning_sources=false;
		scanning_sources_done=true;
		emit_signal("sources_changed",sources_changed.size()>0);
	} else {

		ERR_FAIL_COND(thread_sources);
		set_process(true);
		Thread::Settings s;
		ss_amount=0;
		s.priority=Thread::PRIORITY_LOW;
		thread_sources = Thread::create(_thread_func_sources,this,s);
		//tree->hide();
		//print_line("SCAN BEGIN!");
		//progress->show();
	}



}

EditorFileSystemDirectory* EditorFileSystem::_update_tree(DirItem *p_item) {

	EditorFileSystemDirectory *efd = memnew( EditorFileSystemDirectory );

	if (!p_item)
		return efd; //empty likely
	efd->name=p_item->name;

	for(int i=0;i<p_item->files.size();i++) {

		String s = p_item->files[i]->type;
		//if (p_item->files[i]->meta)
		//	s="*"+s;

//		file_type_cache[p_item->files[i]->path]=s;
		if (p_item->files[i]->type=="")
			continue; //ignore because it's invalid
		EditorFileSystemDirectory::FileInfo fi;
		fi.file=p_item->files[i]->file;
		fi.type=p_item->files[i]->type;
		fi.meta=p_item->files[i]->meta;
		fi.modified_time=p_item->files[i]->modified_time;

		efd->files.push_back(fi);

	}

	for(int i=0;i<p_item->dirs.size();i++) {

		EditorFileSystemDirectory *efsd =_update_tree(p_item->dirs[i]);
		efsd->parent=efd;
		efd->subdirs.push_back( efsd );

	}


	return efd;
}

void EditorFileSystem::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			_load_type_cache();
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
				WARN_PRINT("Scan thread aborted...");
				set_process(false);

			}
			if (rootdir)
				memdelete(rootdir);
			rootdir=NULL;

			if (filesystem)
				memdelete(filesystem);
			filesystem=NULL;

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
						//print_line("sources changed: "+itos(sources_changed.size()));
						emit_signal("sources_changed",sources_changed.size()>0);
					}
				} else if (!scanning) {

					set_process(false);

					if (rootdir)
						memdelete(rootdir);
					if (filesystem)
						memdelete(filesystem);
					rootdir=scandir;
					scandir=NULL;
//					file_type_cache.clear();
					filesystem=_update_tree(rootdir);

					if (rootdir)
						memdelete(rootdir);
					rootdir=NULL;
					Thread::wait_to_finish(thread);
					memdelete(thread);
					thread=NULL;
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

	return total;
}

EditorFileSystemDirectory *EditorFileSystem::get_filesystem() {

	return filesystem;
}

void EditorFileSystem::_save_type_cache_fs(DirItem *p_dir,FileAccess *p_file) {


	if (!p_dir)
		return; //none
	p_file->store_line("::"+p_dir->path+"::"+String::num(p_dir->modified_time));

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
		p_file->store_line(s);
	}

	for(int i=0;i<p_dir->dirs.size();i++) {

		_save_type_cache_fs(p_dir->dirs[i],p_file);
	}

}



void EditorFileSystem::_load_type_cache(){

	GLOBAL_LOCK_FUNCTION


#if 0
	//this is not good, removed for now as it interferes with metadata stored in files

	String project=Globals::get_singleton()->get_resource_path();
	FileAccess *f =FileAccess::open(project+"/types.cache",FileAccess::READ);

	if (!f) {

		WARN_PRINT("Can't open types.cache.");
		return;
	}

	file_type_cache.clear();
	while(!f->eof_reached()) {

		String path=f->get_line();
		if (f->eof_reached())
			break;
		String type=f->get_line();
		file_type_cache[path]=type;
	}

	memdelete(f);
#endif
}



bool EditorFileSystem::_find_file(const String& p_file,EditorFileSystemDirectory ** r_d, int &r_file_pos) const {
	//todo make faster

    if (!filesystem || scanning)
	return false;


    String f = Globals::get_singleton()->localize_path(p_file);

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

	if (fs->files[i].file==file) {
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


    return fs->files[cpos].type;

}


EditorFileSystemDirectory *EditorFileSystem::get_path(const String& p_path) {

    if (!filesystem || scanning)
	return false;


    String f = Globals::get_singleton()->localize_path(p_path);

    if (!f.begins_with("res://"))
	return false;


    f=f.substr(6,f.length());
    f=f.replace("\\","/");
    if (f==String())
	return filesystem;

    if (f.ends_with("/"))
	f=f.substr(0,f.length()-1);

    Vector<String> path = f.split("/");

    if (path.size()==0)
	return false;

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

	EditorFileSystem::get_singleton()->update_file(p_path);
}

String EditorFileSystem::_find_first_from_source(EditorFileSystemDirectory* p_dir,const String &p_src) const {

	for(int i=0;i<p_dir->files.size();i++) {
		for(int j=0;j<p_dir->files[i].meta.sources.size();j++) {

			if (p_dir->files[i].meta.sources[j].path==p_src)
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


    String type = ResourceLoader::get_resource_type(p_file);

    if (cpos==-1) {

	    int idx=0;

	    for(int i=0;i<fs->files.size();i++) {
		if (p_file<fs->files[i].file)
		    break;
		idx++;
	    }

	    EditorFileSystemDirectory::FileInfo fi;
	    fi.file=p_file.get_file();

	    if (idx==fs->files.size()) {
		fs->files.push_back(fi);
	    } else {

		fs->files.insert(idx,fi);
	    }
	    cpos=idx;


    }

	print_line("UPDATING: "+p_file);
	fs->files[cpos].type=type;
	fs->files[cpos].modified_time=FileAccess::get_modified_time(p_file);
	fs->files[cpos].meta=_get_meta(p_file);

	call_deferred("emit_signal","filesystem_changed"); //update later

}



void EditorFileSystem::_bind_methods() {

	ADD_SIGNAL( MethodInfo("filesystem_changed") );
	ADD_SIGNAL( MethodInfo("sources_changed",PropertyInfo(Variant::BOOL,"exist")) );

}

EditorFileSystem::EditorFileSystem() {


	singleton=this;
	filesystem=memnew( EditorFileSystemDirectory ); //like, empty

	thread = NULL;
	scanning=false;
	scandir=NULL;
	rootdir=NULL;
	use_threads=true;
	thread_sources=NULL;

	scanning_sources=false;
	ResourceSaver::set_save_callback(_resource_saved);


}

EditorFileSystem::~EditorFileSystem() {


}
