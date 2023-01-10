/**************************************************************************/
/*  dir_access_windows.cpp                                                */
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

#if defined(WINDOWS_ENABLED)

#include "dir_access_windows.h"

#include "core/os/memory.h"
#include "core/print_string.h"

#include <stdio.h>
#include <wchar.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/*

[03:57] <reduz> yessopie, so i don't havemak to rely on unicows
[03:58] <yessopie> reduz- yeah, all of the functions fail, and then you can call GetLastError () which will return 120
[03:58] <drumstick> CategoryApl, hehe, what? :)
[03:59] <CategoryApl> didn't Verona lead to some trouble
[03:59] <yessopie> 120 = ERROR_CALL_NOT_IMPLEMENTED
[03:59] <yessopie> (you can use that constant if you include winerr.h)
[03:59] <CategoryApl> well answer with winning a compo

[04:02] <yessopie> if ( SetCurrentDirectoryW ( L"." ) == FALSE && GetLastError () == ERROR_CALL_NOT_IMPLEMENTED ) { use ANSI }
*/

struct DirAccessWindowsPrivate {
	HANDLE h; //handle for findfirstfile
	WIN32_FIND_DATA f;
	WIN32_FIND_DATAW fu; //unicode version
};

// CreateFolderAsync

Error DirAccessWindows::list_dir_begin() {
	_cisdir = false;
	_cishidden = false;

	list_dir_end();
	p->h = FindFirstFileExW((current_dir + "\\*").c_str(), FindExInfoStandard, &p->fu, FindExSearchNameMatch, NULL, 0);

	return (p->h == INVALID_HANDLE_VALUE) ? ERR_CANT_OPEN : OK;
}

String DirAccessWindows::get_next() {
	if (p->h == INVALID_HANDLE_VALUE)
		return "";

	_cisdir = (p->fu.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
	_cishidden = (p->fu.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN);

	String name = p->fu.cFileName;

	if (FindNextFileW(p->h, &p->fu) == 0) {
		FindClose(p->h);
		p->h = INVALID_HANDLE_VALUE;
	}

	return name;
}

bool DirAccessWindows::current_is_dir() const {
	return _cisdir;
}

bool DirAccessWindows::current_is_hidden() const {
	return _cishidden;
}

void DirAccessWindows::list_dir_end() {
	if (p->h != INVALID_HANDLE_VALUE) {
		FindClose(p->h);
		p->h = INVALID_HANDLE_VALUE;
	}
}
int DirAccessWindows::get_drive_count() {
	return drive_count;
}
String DirAccessWindows::get_drive(int p_drive) {
	if (p_drive < 0 || p_drive >= drive_count)
		return "";

	return String::chr(drives[p_drive]) + ":";
}

Error DirAccessWindows::change_dir(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	p_dir = fix_path(p_dir);

	wchar_t real_current_dir_name[2048];
	GetCurrentDirectoryW(2048, real_current_dir_name);
	String prev_dir = real_current_dir_name;

	SetCurrentDirectoryW(current_dir.c_str());
	bool worked = (SetCurrentDirectoryW(p_dir.c_str()) != 0);

	String base = _get_root_path();
	if (base != "") {
		GetCurrentDirectoryW(2048, real_current_dir_name);
		String new_dir;
		new_dir = String(real_current_dir_name).replace("\\", "/");
		if (!new_dir.begins_with(base)) {
			worked = false;
		}
	}

	if (worked) {
		GetCurrentDirectoryW(2048, real_current_dir_name);
		current_dir = real_current_dir_name; // TODO, utf8 parser
		current_dir = current_dir.replace("\\", "/");

	} //else {

	SetCurrentDirectoryW(prev_dir.c_str());
	//}

	return worked ? OK : ERR_INVALID_PARAMETER;
}

Error DirAccessWindows::make_dir(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	p_dir = fix_path(p_dir);
	if (p_dir.is_rel_path())
		p_dir = current_dir.plus_file(p_dir);

	p_dir = p_dir.simplify_path().replace("/", "\\");

	bool success;
	int err;

	if (!p_dir.is_network_share_path()) {
		p_dir = "\\\\?\\" + p_dir;
		// Add "\\?\" to the path to extend max. path length past 248, if it's not a network share UNC path.
		// See https://msdn.microsoft.com/en-us/library/windows/desktop/aa363855(v=vs.85).aspx
	}

	success = CreateDirectoryW(p_dir.c_str(), NULL);
	err = GetLastError();

	if (success) {
		return OK;
	};

	if (err == ERROR_ALREADY_EXISTS || err == ERROR_ACCESS_DENIED) {
		return ERR_ALREADY_EXISTS;
	};

	return ERR_CANT_CREATE;
}

String DirAccessWindows::get_current_dir() {
	String base = _get_root_path();
	if (base != "") {
		String bd = current_dir.replace("\\", "/").replace_first(base, "");
		if (bd.begins_with("/"))
			return _get_root_string() + bd.substr(1, bd.length());
		else
			return _get_root_string() + bd;

	} else {
	}

	return current_dir;
}

String DirAccessWindows::get_current_dir_without_drive() {
	String dir = get_current_dir();

	if (_get_root_string() == "") {
		int p = current_dir.find(":");
		if (p != -1) {
			dir = dir.right(p + 1);
		}
	}

	return dir;
}

bool DirAccessWindows::file_exists(String p_file) {
	GLOBAL_LOCK_FUNCTION

	if (!p_file.is_abs_path())
		p_file = get_current_dir().plus_file(p_file);

	p_file = fix_path(p_file);

	//p_file.replace("/","\\");

	//WIN32_FILE_ATTRIBUTE_DATA    fileInfo;

	DWORD fileAttr;

	fileAttr = GetFileAttributesW(p_file.c_str());
	if (INVALID_FILE_ATTRIBUTES == fileAttr)
		return false;

	return !(fileAttr & FILE_ATTRIBUTE_DIRECTORY);
}

bool DirAccessWindows::dir_exists(String p_dir) {
	GLOBAL_LOCK_FUNCTION

	if (p_dir.is_rel_path())
		p_dir = get_current_dir().plus_file(p_dir);

	p_dir = fix_path(p_dir);

	//p_dir.replace("/","\\");

	//WIN32_FILE_ATTRIBUTE_DATA    fileInfo;

	DWORD fileAttr;

	fileAttr = GetFileAttributesW(p_dir.c_str());
	if (INVALID_FILE_ATTRIBUTES == fileAttr)
		return false;
	return (fileAttr & FILE_ATTRIBUTE_DIRECTORY);
}

Error DirAccessWindows::rename(String p_path, String p_new_path) {
	if (p_path.is_rel_path())
		p_path = get_current_dir().plus_file(p_path);

	p_path = fix_path(p_path);

	if (p_new_path.is_rel_path())
		p_new_path = get_current_dir().plus_file(p_new_path);

	p_new_path = fix_path(p_new_path);

	// If we're only changing file name case we need to do a little juggling
	if (p_path.to_lower() == p_new_path.to_lower()) {
		if (dir_exists(p_path)) {
			// The path is a dir; just rename
			return ::_wrename(p_path.c_str(), p_new_path.c_str()) == 0 ? OK : FAILED;
		}
		// The path is a file; juggle
		WCHAR tmpfile[MAX_PATH];

		if (!GetTempFileNameW(fix_path(get_current_dir()).c_str(), NULL, 0, tmpfile)) {
			return FAILED;
		}

		if (!::ReplaceFileW(tmpfile, p_path.c_str(), NULL, 0, NULL, NULL)) {
			DeleteFileW(tmpfile);
			return FAILED;
		}

		return ::_wrename(tmpfile, p_new_path.c_str()) == 0 ? OK : FAILED;

	} else {
		if (file_exists(p_new_path)) {
			if (remove(p_new_path) != OK) {
				return FAILED;
			}
		}

		return ::_wrename(p_path.c_str(), p_new_path.c_str()) == 0 ? OK : FAILED;
	}
}

Error DirAccessWindows::remove(String p_path) {
	if (p_path.is_rel_path())
		p_path = get_current_dir().plus_file(p_path);

	p_path = fix_path(p_path);

	DWORD fileAttr;

	fileAttr = GetFileAttributesW(p_path.c_str());
	if (INVALID_FILE_ATTRIBUTES == fileAttr)
		return FAILED;
	if ((fileAttr & FILE_ATTRIBUTE_DIRECTORY))
		return ::_wrmdir(p_path.c_str()) == 0 ? OK : FAILED;
	else
		return ::_wunlink(p_path.c_str()) == 0 ? OK : FAILED;
}
/*

FileType DirAccessWindows::get_file_type(const String& p_file) const {


	wchar_t real_current_dir_name[2048];
	GetCurrentDirectoryW(2048,real_current_dir_name);
	String prev_dir=real_current_dir_name;

	bool worked SetCurrentDirectoryW(current_dir.c_str());

	DWORD attr;
	if (worked) {

		WIN32_FILE_ATTRIBUTE_DATA    fileInfo;
		attr = GetFileAttributesExW(p_file.c_str(), GetFileExInfoStandard, &fileInfo);

	}

	SetCurrentDirectoryW(prev_dir.c_str());

	if (!worked)
		return FILE_TYPE_NONE;


	return (attr&FILE_ATTRIBUTE_DIRECTORY)?FILE_TYPE_
}
*/

uint64_t DirAccessWindows::get_space_left() {
	uint64_t bytes = 0;
	if (!GetDiskFreeSpaceEx(NULL, (PULARGE_INTEGER)&bytes, NULL, NULL))
		return 0;

	//this is either 0 or a value in bytes.
	return bytes;
}

String DirAccessWindows::get_filesystem_type() const {
	String path = fix_path(const_cast<DirAccessWindows *>(this)->get_current_dir());

	if (path.is_network_share_path()) {
		return "Network Share";
	}

	int unit_end = path.find(":");
	ERR_FAIL_COND_V(unit_end == -1, String());
	String unit = path.substr(0, unit_end + 1) + "\\";

	WCHAR szVolumeName[100];
	WCHAR szFileSystemName[10];
	DWORD dwSerialNumber = 0;
	DWORD dwMaxFileNameLength = 0;
	DWORD dwFileSystemFlags = 0;

	if (::GetVolumeInformationW(unit.c_str(),
				szVolumeName,
				sizeof(szVolumeName),
				&dwSerialNumber,
				&dwMaxFileNameLength,
				&dwFileSystemFlags,
				szFileSystemName,
				sizeof(szFileSystemName)) == TRUE) {
		return String(szFileSystemName);
	}

	ERR_FAIL_V("");
}

DirAccessWindows::DirAccessWindows() {
	p = memnew(DirAccessWindowsPrivate);
	p->h = INVALID_HANDLE_VALUE;
	current_dir = ".";

	drive_count = 0;

#ifdef UWP_ENABLED
	Windows::Storage::StorageFolder ^ install_folder = Windows::ApplicationModel::Package::Current->InstalledLocation;
	change_dir(install_folder->Path->Data());

#else

	DWORD mask = GetLogicalDrives();

	for (int i = 0; i < MAX_DRIVES; i++) {
		if (mask & (1 << i)) { //DRIVE EXISTS

			drives[drive_count] = 'A' + i;
			drive_count++;
		}
	}

	change_dir(".");
#endif
}

DirAccessWindows::~DirAccessWindows() {
	list_dir_end();

	memdelete(p);
}

#endif //windows DirAccess support
