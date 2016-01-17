/*************************************************/
/*  dir_access_psp.cpp                           */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "dir_access_flash.h"

#include "print_string.h"

DirAccess* DirAccessFlash::create_flash() {

	return memnew( DirAccessFlash );
};

void DirAccessFlash::make_default()  {

	instance_func=create_flash;
}


bool DirAccessFlash::list_dir_begin() {

	list_dir_end();

	dir_stream = opendir(current_dir.utf8().get_data());
	if (!dir_stream)
		return true; //error!

	return false;
};

String DirAccessFlash::get_next() {

	if (!dir_stream)
		return "";
	dirent *entry;

	entry=readdir(dir_stream);

	if (entry==NULL) {

		list_dir_end();
		return "";
	}

	//typedef struct stat Stat;
	struct stat flags;

	String fname;
	if (fname.parse_utf8(entry->d_name))
		fname=entry->d_name; //no utf8, maybe latin?

	String f=current_dir+"/"+fname;

	if (stat(f.utf8().get_data(),&flags)==0) {

		if (S_ISDIR(flags.st_mode)) {

			_cisdir=true;

		} else {

			_cisdir=false;
		}

	} else {

		_cisdir=false;

	}

	return fname;
};

bool DirAccessFlash::current_is_dir() const {

	return _cisdir;
};

void DirAccessFlash::list_dir_end() {

	if (dir_stream)
		closedir(dir_stream);
	dir_stream=0;
	_cisdir=false;
};

int DirAccessFlash::get_drive_count() {

	return 1;
};

String DirAccessFlash::get_drive(int p_drive) {

	return "host0";
};

Error DirAccessFlash::change_dir(String p_dir) {

	if (p_dir==".")
		return OK;


	if (p_dir.is_rel_path())
		current_dir+="/"+p_dir;
	else
		current_dir=fix_path(p_dir);

	current_dir=current_dir.simplify_path();
	if (current_dir.length()>1 && current_dir.ends_with("/") && !current_dir.ends_with("//"))
		current_dir=current_dir.substr(0,current_dir.length()-1);

	return OK;

};

String DirAccessFlash::get_current_dir() {

	return current_dir;
};


uint64_t DirAccessFlash::get_modified_time(String p_file) {

	return 0;
};

Error DirAccessFlash::make_dir(String p_dir) {

	return ERR_UNAVAILABLE;
};

bool DirAccessFlash::file_exists(String p_file) {

	GLOBAL_LOCK_FUNCTION


	if (p_file.is_rel_path())
		p_file=current_dir+"/"+p_file;
	else
		p_file=fix_path(p_file);

	struct stat flags;
	bool success = 	(stat(p_file.utf8().get_data(),&flags)==0);

	if (success && S_ISDIR(flags.st_mode)) {
		success=false;
	}

	return success;
};

bool DirAccessFlash::dir_exists(String p_dir) {

	GLOBAL_LOCK_FUNCTION


	if (p_dir.is_rel_path())
		p_dir=current_dir+"/"+p_dir;
	else
		p_dir=fix_path(p_dir);

	struct stat flags;
	bool success = 	(stat(p_dir.utf8().get_data(),&flags)==0);

	if (success && S_ISDIR(flags.st_mode)) {
		return true;
	}

	return false;
};

size_t DirAccessFlash::get_space_left() {

	return 0;
};

Error DirAccessFlash::rename(String p_from, String p_to) {

	return FAILED;
};

Error DirAccessFlash::remove(String p_name) {

	return FAILED;
};

extern char* psp_drive;

DirAccessFlash::DirAccessFlash() {

	dir_stream=0;
	current_dir=".";
	_cisdir=false;

	/* determine drive count */

	change_dir(current_dir);
}

DirAccessFlash::~DirAccessFlash() {

	list_dir_end();
};

