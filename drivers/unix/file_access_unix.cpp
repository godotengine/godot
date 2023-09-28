/**************************************************************************/
/*  file_access_unix.cpp                                                  */
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

#include "file_access_unix.h"

#if defined(UNIX_ENABLED)

#include "core/os/os.h"
#include "core/string/print_string.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void FileAccessUnix::check_errors() const {
	ERR_FAIL_NULL_MSG(f, "File must be opened before use.");

	if (feof(f)) {
		last_error = ERR_FILE_EOF;
	}
}

Error FileAccessUnix::open_internal(const String &p_path, int p_mode_flags) {
	_close();

	path_src = p_path;
	path = fix_path(p_path);
	//printf("opening %s, %i\n", path.utf8().get_data(), Memory::get_static_mem_usage());

	ERR_FAIL_COND_V_MSG(f, ERR_ALREADY_IN_USE, "File is already in use.");
	const char *mode_string;

	if (p_mode_flags == READ) {
		mode_string = "rb";
	} else if (p_mode_flags == WRITE) {
		mode_string = "wb";
	} else if (p_mode_flags == READ_WRITE) {
		mode_string = "rb+";
	} else if (p_mode_flags == WRITE_READ) {
		mode_string = "wb+";
	} else {
		return ERR_INVALID_PARAMETER;
	}

	/* pretty much every implementation that uses fopen as primary
	   backend (unix-compatible mostly) supports utf8 encoding */

	//printf("opening %s as %s\n", p_path.utf8().get_data(), path.utf8().get_data());
	struct stat st = {};
	int err = stat(path.utf8().get_data(), &st);
	if (!err) {
		switch (st.st_mode & S_IFMT) {
			case S_IFLNK:
			case S_IFREG:
				break;
			default:
				return ERR_FILE_CANT_OPEN;
		}
	}

	if (is_backup_save_enabled() && (p_mode_flags == WRITE)) {
		save_path = path;
		// Create a temporary file in the same directory as the target file.
		path = path + "-XXXXXX";
		CharString cs = path.utf8();
		int fd = mkstemp(cs.ptrw());
		if (fd == -1) {
			last_error = ERR_FILE_CANT_OPEN;
			return last_error;
		}
		fchmod(fd, 0666);
		path = String::utf8(cs.ptr());

		f = fdopen(fd, mode_string);
		if (f == nullptr) {
			// Delete temp file and close descriptor if open failed.
			::unlink(cs.ptr());
			::close(fd);
			last_error = ERR_FILE_CANT_OPEN;
			return last_error;
		}
	} else {
		f = fopen(path.utf8().get_data(), mode_string);
	}

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
	}

	// Set close on exec to avoid leaking it to subprocesses.
	int fd = fileno(f);

	if (fd != -1) {
		int opts = fcntl(fd, F_GETFD);
		fcntl(fd, F_SETFD, opts | FD_CLOEXEC);
	}

	last_error = OK;
	flags = p_mode_flags;
	return OK;
}

void FileAccessUnix::_close() {
	if (!f) {
		return;
	}

	fclose(f);
	f = nullptr;

	if (close_notification_func) {
		close_notification_func(path, flags);
	}

	if (!save_path.is_empty()) {
		int rename_error = rename(path.utf8().get_data(), save_path.utf8().get_data());

		if (rename_error && close_fail_notify) {
			close_fail_notify(save_path);
		}

		save_path = "";
		ERR_FAIL_COND(rename_error != 0);
	}
}

bool FileAccessUnix::is_open() const {
	return (f != nullptr);
}

String FileAccessUnix::get_path() const {
	return path_src;
}

String FileAccessUnix::get_path_absolute() const {
	return path;
}

void FileAccessUnix::seek(uint64_t p_position) {
	ERR_FAIL_NULL_MSG(f, "File must be opened before use.");

	last_error = OK;
	if (fseeko(f, p_position, SEEK_SET)) {
		check_errors();
	}
}

void FileAccessUnix::seek_end(int64_t p_position) {
	ERR_FAIL_NULL_MSG(f, "File must be opened before use.");

	if (fseeko(f, p_position, SEEK_END)) {
		check_errors();
	}
}

uint64_t FileAccessUnix::get_position() const {
	ERR_FAIL_NULL_V_MSG(f, 0, "File must be opened before use.");

	int64_t pos = ftello(f);
	if (pos < 0) {
		check_errors();
		ERR_FAIL_V(0);
	}
	return pos;
}

uint64_t FileAccessUnix::get_length() const {
	ERR_FAIL_NULL_V_MSG(f, 0, "File must be opened before use.");

	int64_t pos = ftello(f);
	ERR_FAIL_COND_V(pos < 0, 0);
	ERR_FAIL_COND_V(fseeko(f, 0, SEEK_END), 0);
	int64_t size = ftello(f);
	ERR_FAIL_COND_V(size < 0, 0);
	ERR_FAIL_COND_V(fseeko(f, pos, SEEK_SET), 0);

	return size;
}

bool FileAccessUnix::eof_reached() const {
	return last_error == ERR_FILE_EOF;
}

uint8_t FileAccessUnix::get_8() const {
	ERR_FAIL_NULL_V_MSG(f, 0, "File must be opened before use.");
	uint8_t b;
	if (fread(&b, 1, 1, f) == 0) {
		check_errors();
		b = '\0';
	}
	return b;
}

uint64_t FileAccessUnix::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);
	ERR_FAIL_NULL_V_MSG(f, -1, "File must be opened before use.");

	uint64_t read = fread(p_dst, 1, p_length, f);
	check_errors();
	return read;
}

Error FileAccessUnix::get_error() const {
	return last_error;
}

void FileAccessUnix::flush() {
	ERR_FAIL_NULL_MSG(f, "File must be opened before use.");
	fflush(f);
}

void FileAccessUnix::store_8(uint8_t p_dest) {
	ERR_FAIL_NULL_MSG(f, "File must be opened before use.");
	ERR_FAIL_COND(fwrite(&p_dest, 1, 1, f) != 1);
}

void FileAccessUnix::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_NULL_MSG(f, "File must be opened before use.");
	ERR_FAIL_COND(!p_src && p_length > 0);
	ERR_FAIL_COND(fwrite(p_src, 1, p_length, f) != p_length);
}

bool FileAccessUnix::file_exists(const String &p_path) {
	int err;
	struct stat st = {};
	String filename = fix_path(p_path);

	// Does the name exist at all?
	err = stat(filename.utf8().get_data(), &st);
	if (err) {
		return false;
	}

	// See if we have access to the file
	if (access(filename.utf8().get_data(), F_OK)) {
		return false;
	}

	// See if this is a regular file
	switch (st.st_mode & S_IFMT) {
		case S_IFLNK:
		case S_IFREG:
			return true;
		default:
			return false;
	}
}

uint64_t FileAccessUnix::_get_modified_time(const String &p_file) {
	String file = fix_path(p_file);
	struct stat status = {};
	int err = stat(file.utf8().get_data(), &status);

	if (!err) {
		return status.st_mtime;
	} else {
		print_verbose("Failed to get modified time for: " + p_file + "");
		return 0;
	}
}

BitField<FileAccess::UnixPermissionFlags> FileAccessUnix::_get_unix_permissions(const String &p_file) {
	String file = fix_path(p_file);
	struct stat status = {};
	int err = stat(file.utf8().get_data(), &status);

	if (!err) {
		return status.st_mode & 0xFFF; //only permissions
	} else {
		ERR_FAIL_V_MSG(0, "Failed to get unix permissions for: " + p_file + ".");
	}
}

Error FileAccessUnix::_set_unix_permissions(const String &p_file, BitField<FileAccess::UnixPermissionFlags> p_permissions) {
	String file = fix_path(p_file);

	int err = chmod(file.utf8().get_data(), p_permissions);
	if (!err) {
		return OK;
	}

	return FAILED;
}

bool FileAccessUnix::_get_hidden_attribute(const String &p_file) {
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
	String file = fix_path(p_file);

	struct stat st = {};
	int err = stat(file.utf8().get_data(), &st);
	ERR_FAIL_COND_V_MSG(err, false, "Failed to get attributes for: " + p_file);

	return (st.st_flags & UF_HIDDEN);
#else
	return false;
#endif
}

Error FileAccessUnix::_set_hidden_attribute(const String &p_file, bool p_hidden) {
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
	String file = fix_path(p_file);

	struct stat st = {};
	int err = stat(file.utf8().get_data(), &st);
	ERR_FAIL_COND_V_MSG(err, FAILED, "Failed to get attributes for: " + p_file);

	if (p_hidden) {
		err = chflags(file.utf8().get_data(), st.st_flags | UF_HIDDEN);
	} else {
		err = chflags(file.utf8().get_data(), st.st_flags & ~UF_HIDDEN);
	}
	ERR_FAIL_COND_V_MSG(err, FAILED, "Failed to set attributes for: " + p_file);
	return OK;
#else
	return ERR_UNAVAILABLE;
#endif
}

bool FileAccessUnix::_get_read_only_attribute(const String &p_file) {
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
	String file = fix_path(p_file);

	struct stat st = {};
	int err = stat(file.utf8().get_data(), &st);
	ERR_FAIL_COND_V_MSG(err, false, "Failed to get attributes for: " + p_file);

	return st.st_flags & UF_IMMUTABLE;
#else
	return false;
#endif
}

Error FileAccessUnix::_set_read_only_attribute(const String &p_file, bool p_ro) {
#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
	String file = fix_path(p_file);

	struct stat st = {};
	int err = stat(file.utf8().get_data(), &st);
	ERR_FAIL_COND_V_MSG(err, FAILED, "Failed to get attributes for: " + p_file);

	if (p_ro) {
		err = chflags(file.utf8().get_data(), st.st_flags | UF_IMMUTABLE);
	} else {
		err = chflags(file.utf8().get_data(), st.st_flags & ~UF_IMMUTABLE);
	}
	ERR_FAIL_COND_V_MSG(err, FAILED, "Failed to set attributes for: " + p_file);
	return OK;
#else
	return ERR_UNAVAILABLE;
#endif
}

void FileAccessUnix::close() {
	_close();
}

CloseNotificationFunc FileAccessUnix::close_notification_func = nullptr;

FileAccessUnix::~FileAccessUnix() {
	_close();
}

#endif // UNIX_ENABLED
