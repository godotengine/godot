/**************************************************************************/
/*  file_access_windows.cpp                                               */
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

#ifdef WINDOWS_ENABLED

#include "file_access_windows.h"

#include "core/os/os.h"
#include "core/print_string.h"

#include <share.h> // _SH_DENYNO
#include <shlwapi.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <errno.h>
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

Error FileAccessWindows::_open(const String &p_path, int p_mode_flags) {
	path_src = p_path;
	path = fix_path(p_path);
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
	if (_wstat(path.c_str(), &st) == 0) {
		if (!S_ISREG(st.st_mode))
			return ERR_FILE_CANT_OPEN;
	};

#ifdef TOOLS_ENABLED
	// Windows is case insensitive, but all other platforms are sensitive to it
	// To ease cross-platform development, we issue a warning if users try to access
	// a file using the wrong case (which *works* on Windows, but won't on other
	// platforms).
	if (p_mode_flags == READ) {
		WIN32_FIND_DATAW d;
		HANDLE f = FindFirstFileW(path.c_str(), &d);
		if (f != INVALID_HANDLE_VALUE) {
			String fname = d.cFileName;
			if (fname != String()) {
				String base_file = path.get_file();
				if (base_file != fname && base_file.findn(fname) == 0) {
					WARN_PRINT("Case mismatch opening requested file '" + base_file + "', stored as '" + fname + "' in the filesystem. This file will not open when exported to other case-sensitive platforms.");
				}
			}
			FindClose(f);
		}
	}
#endif

	if (is_backup_save_enabled() && p_mode_flags & WRITE && !(p_mode_flags & READ)) {
		save_path = path;
		path = path + ".tmp";
	}

	f = _wfsopen((LPCWSTR)(path.c_str()), mode_string, _SH_DENYNO);

	if (f == nullptr) {
		switch (errno) {
			case ENOENT: {
				last_error = ERR_FILE_NOT_FOUND;
			} break;
			default: {
				last_error = ERR_FILE_CANT_OPEN;
			} break;
		}
		return last_error;
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
		bool rename_error = true;
		int attempts = 4;
		while (rename_error && attempts) {
			// This workaround of trying multiple times is added to deal with paranoid Windows
			// antiviruses that love reading just written files even if they are not executable, thus
			// locking the file and preventing renaming from happening.

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
			if (rename_error) {
				attempts--;
				OS::get_singleton()->delay_usec(100000); // wait 100msec and try again
			}
		}

		if (rename_error) {
			if (close_fail_notify) {
				close_fail_notify(save_path);
			}
		}

		save_path = "";

		ERR_FAIL_COND_MSG(rename_error, "Safe save failed. This may be a permissions problem, but also may happen because you are running a paranoid antivirus. If this is the case, please switch to Windows Defender or disable the 'safe save' option in editor settings. This makes it work, but increases the risk of file corruption in a crash.");
	}
}

String FileAccessWindows::get_path() const {
	return path_src;
}

String FileAccessWindows::get_path_absolute() const {
	return path;
}

bool FileAccessWindows::is_open() const {
	return (f != NULL);
}

void FileAccessWindows::seek(uint64_t p_position) {
	ERR_FAIL_COND(!f);

	last_error = OK;
	if (_fseeki64(f, p_position, SEEK_SET))
		check_errors();
	prev_op = 0;
}

void FileAccessWindows::seek_end(int64_t p_position) {
	ERR_FAIL_COND(!f);
	if (_fseeki64(f, p_position, SEEK_END))
		check_errors();
	prev_op = 0;
}

uint64_t FileAccessWindows::get_position() const {
	int64_t aux_position = _ftelli64(f);
	if (aux_position < 0) {
		check_errors();
	}
	return aux_position;
}

uint64_t FileAccessWindows::get_len() const {
	ERR_FAIL_COND_V(!f, 0);

	uint64_t pos = get_position();
	_fseeki64(f, 0, SEEK_END);
	uint64_t size = get_position();
	_fseeki64(f, pos, SEEK_SET);

	return size;
}

bool FileAccessWindows::eof_reached() const {
	check_errors();
	return last_error == ERR_FILE_EOF;
}

uint8_t FileAccessWindows::get_8() const {
	ERR_FAIL_COND_V(!f, 0);
	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == WRITE) {
			fflush(f);
		}
		prev_op = READ;
	}
	uint8_t b;
	if (fread(&b, 1, 1, f) == 0) {
		check_errors();
		b = '\0';
	};

	return b;
}

uint64_t FileAccessWindows::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);
	ERR_FAIL_COND_V(!f, -1);

	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == WRITE) {
			fflush(f);
		}
		prev_op = READ;
	}
	uint64_t read = fread(p_dst, 1, p_length, f);
	check_errors();
	return read;
};

Error FileAccessWindows::get_error() const {
	return last_error;
}

void FileAccessWindows::flush() {
	ERR_FAIL_COND(!f);
	fflush(f);
	if (prev_op == WRITE)
		prev_op = 0;
}

void FileAccessWindows::store_8(uint8_t p_dest) {
	ERR_FAIL_COND(!f);
	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == READ) {
			if (last_error != ERR_FILE_EOF) {
				fseek(f, 0, SEEK_CUR);
			}
		}
		prev_op = WRITE;
	}
	fwrite(&p_dest, 1, 1, f);
}

void FileAccessWindows::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_COND(!f);
	ERR_FAIL_COND(!p_src && p_length > 0);
	if (flags == READ_WRITE || flags == WRITE_READ) {
		if (prev_op == READ) {
			if (last_error != ERR_FILE_EOF) {
				fseek(f, 0, SEEK_CUR);
			}
		}
		prev_op = WRITE;
	}
	ERR_FAIL_COND(fwrite(p_src, 1, p_length, f) != (size_t)p_length);
}

bool FileAccessWindows::file_exists(const String &p_name) {
	String filename = fix_path(p_name);
	FILE *g = _wfsopen((LPCWSTR)(filename.c_str()), L"rb", _SH_DENYNO);
	if (g == nullptr) {
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
		print_verbose("Failed to get modified time for: " + p_file + "");
		return 0;
	}
}

uint32_t FileAccessWindows::_get_unix_permissions(const String &p_file) {
	return 0;
}

Error FileAccessWindows::_set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	return ERR_UNAVAILABLE;
}

FileAccessWindows::FileAccessWindows() :
		f(NULL),
		flags(0),
		prev_op(0),
		last_error(OK) {
}
FileAccessWindows::~FileAccessWindows() {
	close();
}

#endif
