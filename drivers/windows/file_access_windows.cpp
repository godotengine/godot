/*************************************************************************/
/*  file_access_windows.cpp                                              */
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
#ifdef WINDOWS_ENABLED

#include "file_access_windows.h"
#include "shlwapi.h"
#include <windows.h>

#include "print_string.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <tchar.h>
#include <wchar.h>

#ifdef _MSC_VER
#define S_ISREG(m) ((m)&_S_IFREG)
#endif

void FileAccessWindows::check_errors() const {

	ERR_FAIL_COND(!f);

	if (feof(f)) {

		last_error = ERR_FILE_EOF;
	}
}

Error FileAccessWindows::_open(const String &p_filename, int p_mode_flags) {

	String filename = fix_path(p_filename);
	if (f)
		close();

	const wchar_t *mode_string;

	if (p_mode_flags == READ)
		mode_string = L"rb";
	else if (p_mode_flags == WRITE)
		mode_string = L"wb";
	else if (p_mode_flags == READ_WRITE)
		mode_string = L"rb+";
	else if (p_mode_flags == WRITE_READ)
		mode_string = L"wb+";
	else
		return ERR_INVALID_PARAMETER;

	/* pretty much every implementation that uses fopen as primary
	   backend supports utf8 encoding */

	struct _stat st;
	if (_wstat(filename.c_str(), &st) == 0) {

		if (!S_ISREG(st.st_mode))
			return ERR_FILE_CANT_OPEN;
	};

	if (is_backup_save_enabled() && p_mode_flags & WRITE && !(p_mode_flags & READ)) {
		save_path = filename;
		filename = filename + ".tmp";
		//print_line("saving instead to "+path);
	}

	f = _wfopen(filename.c_str(), mode_string);

	if (f == NULL) {
		last_error = ERR_FILE_CANT_OPEN;
		return ERR_FILE_CANT_OPEN;
	} else {
		last_error = OK;
		flags = p_mode_flags;
		return OK;
	}
}
void FileAccessWindows::close() {

	if (!f)
		return;

	fclose(f);
	f = NULL;

	if (save_path != "") {

		//unlink(save_path.utf8().get_data());
		//print_line("renaming..");
		//_wunlink(save_path.c_str()); //unlink if exists
		//int rename_error = _wrename((save_path+".tmp").c_str(),save_path.c_str());

		bool rename_error;

#ifdef UWP_ENABLED
		// UWP has no PathFileExists, so we check attributes instead
		DWORD fileAttr;

		fileAttr = GetFileAttributesW(save_path.c_str());
		if (INVALID_FILE_ATTRIBUTES == fileAttr) {
#else
		if (!PathFileExistsW(save_path.c_str())) {
#endif
			//creating new file
			rename_error = _wrename((save_path + ".tmp").c_str(), save_path.c_str()) != 0;
		} else {
			//atomic replace for existing file
			rename_error = !ReplaceFileW(save_path.c_str(), (save_path + ".tmp").c_str(), NULL, 2 | 4, NULL, NULL);
		}
		if (rename_error && close_fail_notify) {
			close_fail_notify(save_path);
		}

		save_path = "";
		ERR_FAIL_COND(rename_error);
	}
}
bool FileAccessWindows::is_open() const {

	return (f != NULL);
}
void FileAccessWindows::seek(size_t p_position) {

	ERR_FAIL_COND(!f);
	last_error = OK;
	if (fseek(f, p_position, SEEK_SET))
		check_errors();
}
void FileAccessWindows::seek_end(int64_t p_position) {

	ERR_FAIL_COND(!f);
	if (fseek(f, p_position, SEEK_END))
		check_errors();
}
size_t FileAccessWindows::get_pos() const {

	size_t aux_position = 0;
	if (!(aux_position = ftell(f))) {
		check_errors();
	};
	return aux_position;
}
size_t FileAccessWindows::get_len() const {

	ERR_FAIL_COND_V(!f, 0);

	size_t pos = get_pos();
	fseek(f, 0, SEEK_END);
	int size = get_pos();
	fseek(f, pos, SEEK_SET);

	return size;
}

bool FileAccessWindows::eof_reached() const {

	check_errors();
	return last_error == ERR_FILE_EOF;
}

uint8_t FileAccessWindows::get_8() const {

	ERR_FAIL_COND_V(!f, 0);
	uint8_t b;
	if (fread(&b, 1, 1, f) == 0) {
		check_errors();
	};

	return b;
}

int FileAccessWindows::get_buffer(uint8_t *p_dst, int p_length) const {

	ERR_FAIL_COND_V(!f, -1);
	int read = fread(p_dst, 1, p_length, f);
	check_errors();
	return read;
};

Error FileAccessWindows::get_error() const {

	return last_error;
}

void FileAccessWindows::store_8(uint8_t p_dest) {

	ERR_FAIL_COND(!f);
	fwrite(&p_dest, 1, 1, f);
}

bool FileAccessWindows::file_exists(const String &p_name) {

	FILE *g;
	//printf("opening file %s\n", p_fname.c_str());
	String filename = fix_path(p_name);
	g = _wfopen(filename.c_str(), L"rb");
	if (g == NULL) {

		return false;
	} else {

		fclose(g);
		return true;
	}
}

uint64_t FileAccessWindows::_get_modified_time(const String &p_file) {

	String file = fix_path(p_file);
	if (file.ends_with("/") && file != "/")
		file = file.substr(0, file.length() - 1);

	struct _stat st;
	int rv = _wstat(file.c_str(), &st);

	if (rv == 0) {

		return st.st_mtime;
	} else {
		print_line("no access to " + file);
	}

	ERR_FAIL_V(0);
};

FileAccessWindows::FileAccessWindows() {

	f = NULL;
	flags = 0;
	last_error = OK;
}
FileAccessWindows::~FileAccessWindows() {

	close();
}

#endif
