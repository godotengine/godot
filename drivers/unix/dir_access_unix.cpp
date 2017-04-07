/*************************************************************************/
/*  dir_access_unix.cpp                                                  */
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
#include "dir_access_unix.h"

#if defined(UNIX_ENABLED) || defined(LIBC_FILEIO_ENABLED)

#ifndef ANDROID_ENABLED
#include <sys/statvfs.h>
#endif

#include "os/memory.h"
#include "print_string.h"
#include <errno.h>
#include <stdio.h>

DirAccess *DirAccessUnix::create_fs() {

	return memnew(DirAccessUnix);
}

Error DirAccessUnix::list_dir_begin() {

	list_dir_end(); //close any previous dir opening!

	//char real_current_dir_name[2048]; //is this enough?!
	//getcwd(real_current_dir_name,2048);
	//chdir(curent_path.utf8().get_data());
	dir_stream = opendir(current_dir.utf8().get_data());
	//chdir(real_current_dir_name);
	if (!dir_stream)
		return ERR_CANT_OPEN; //error!

	return OK;
}

bool DirAccessUnix::file_exists(String p_file) {

	GLOBAL_LOCK_FUNCTION

	if (p_file.is_rel_path())
		p_file = current_dir.plus_file(p_file);

	p_file = fix_path(p_file);

	struct stat flags;
	bool success = (stat(p_file.utf8().get_data(), &flags) == 0);

	if (success && S_ISDIR(flags.st_mode)) {
		success = false;
	}

	return success;
}

bool DirAccessUnix::dir_exists(String p_dir) {

	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path())
		p_dir = get_current_dir().plus_file(p_dir);

	p_dir = fix_path(p_dir);

	struct stat flags;
	bool success = (stat(p_dir.utf8().get_data(), &flags) == 0);

	if (success && S_ISDIR(flags.st_mode))
		return true;

	return false;
}

uint64_t DirAccessUnix::get_modified_time(String p_file) {

	if (p_file.is_rel_path())
		p_file = current_dir.plus_file(p_file);

	p_file = fix_path(p_file);

	struct stat flags;
	bool success = (stat(p_file.utf8().get_data(), &flags) == 0);

	if (success) {
		return flags.st_mtime;
	} else {

		ERR_FAIL_V(0);
	};
	return 0;
};

String DirAccessUnix::get_next() {

	if (!dir_stream)
		return "";
	dirent *entry;

	entry = readdir(dir_stream);

	if (entry == NULL) {

		list_dir_end();
		return "";
	}

	//typedef struct stat Stat;
	struct stat flags;

	String fname = fix_unicode_name(entry->d_name);

	String f = current_dir.plus_file(fname);

	if (stat(f.utf8().get_data(), &flags) == 0) {

		if (S_ISDIR(flags.st_mode)) {

			_cisdir = true;

		} else {

			_cisdir = false;
		}

	} else {

		_cisdir = false;
	}

	_cishidden = (fname != "." && fname != ".." && fname.begins_with("."));

	return fname;
}

bool DirAccessUnix::current_is_dir() const {

	return _cisdir;
}

bool DirAccessUnix::current_is_hidden() const {

	return _cishidden;
}

void DirAccessUnix::list_dir_end() {

	if (dir_stream)
		closedir(dir_stream);
	dir_stream = 0;
	_cisdir = false;
}

int DirAccessUnix::get_drive_count() {

	return 0;
}
String DirAccessUnix::get_drive(int p_drive) {

	return "";
}

Error DirAccessUnix::make_dir(String p_dir) {

	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path())
		p_dir = get_current_dir().plus_file(p_dir);

	p_dir = fix_path(p_dir);

#if 1

	bool success = (mkdir(p_dir.utf8().get_data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0);
	int err = errno;

#else
	char real_current_dir_name[2048];
	getcwd(real_current_dir_name, 2048);
	chdir(current_dir.utf8().get_data()); //ascii since this may be unicode or wathever the host os wants

	bool success = (mkdir(p_dir.utf8().get_data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0);
	int err = errno;

	chdir(real_current_dir_name);
#endif
	if (success) {
		return OK;
	};

	if (err == EEXIST) {
		return ERR_ALREADY_EXISTS;
	};

	return ERR_CANT_CREATE;
}

Error DirAccessUnix::change_dir(String p_dir) {

	GLOBAL_LOCK_FUNCTION
	p_dir = fix_path(p_dir);

	char real_current_dir_name[2048];
	getcwd(real_current_dir_name, 2048);
	String prev_dir;
	if (prev_dir.parse_utf8(real_current_dir_name))
		prev_dir = real_current_dir_name; //no utf8, maybe latin?

	chdir(current_dir.utf8().get_data()); //ascii since this may be unicode or wathever the host os wants
	bool worked = (chdir(p_dir.utf8().get_data()) == 0); // we can only give this utf8

	String base = _get_root_path();
	if (base != "") {

		getcwd(real_current_dir_name, 2048);
		String new_dir;
		new_dir.parse_utf8(real_current_dir_name);
		if (!new_dir.begins_with(base))
			worked = false;
	}

	if (worked) {

		getcwd(real_current_dir_name, 2048);
		if (current_dir.parse_utf8(real_current_dir_name))
			current_dir = real_current_dir_name; //no utf8, maybe latin?
	}

	chdir(prev_dir.utf8().get_data());
	return worked ? OK : ERR_INVALID_PARAMETER;
}

String DirAccessUnix::get_current_dir() {

	String base = _get_root_path();
	if (base != "") {

		String bd = current_dir.replace_first(base, "");
		if (bd.begins_with("/"))
			return _get_root_string() + bd.substr(1, bd.length());
		else
			return _get_root_string() + bd;
	}
	return current_dir;
}

Error DirAccessUnix::rename(String p_path, String p_new_path) {

	if (p_path.is_rel_path())
		p_path = get_current_dir().plus_file(p_path);

	p_path = fix_path(p_path);

	if (p_new_path.is_rel_path())
		p_new_path = get_current_dir().plus_file(p_new_path);

	p_new_path = fix_path(p_new_path);

	return ::rename(p_path.utf8().get_data(), p_new_path.utf8().get_data()) == 0 ? OK : FAILED;
}
Error DirAccessUnix::remove(String p_path) {

	if (p_path.is_rel_path())
		p_path = get_current_dir().plus_file(p_path);

	p_path = fix_path(p_path);

	struct stat flags;
	if ((stat(p_path.utf8().get_data(), &flags) != 0))
		return FAILED;

	if (S_ISDIR(flags.st_mode))
		return ::rmdir(p_path.utf8().get_data()) == 0 ? OK : FAILED;
	else
		return ::unlink(p_path.utf8().get_data()) == 0 ? OK : FAILED;
}

size_t DirAccessUnix::get_space_left() {

#ifndef NO_STATVFS
	struct statvfs vfs;
	if (statvfs(current_dir.utf8().get_data(), &vfs) != 0) {

		return 0;
	};

	return vfs.f_bfree * vfs.f_bsize;
#else
#warning THIS IS BROKEN
	return 0;
#endif
};

DirAccessUnix::DirAccessUnix() {

	dir_stream = 0;
	current_dir = ".";
	_cisdir = false;

	/* determine drive count */

	change_dir(current_dir);
}

DirAccessUnix::~DirAccessUnix() {

	list_dir_end();
}

#endif //posix_enabled
