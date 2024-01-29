/**************************************************************************/
/*  dir_access_extension.cpp                                              */
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

#include "dir_access_extension.h"

#include "core/io/dir_access.h"
#include "core/object/object.h"

String DirAccessExtension::fix_path(String p_path) const {
	String path = "";
	if (GDVIRTUAL_CALL(_fix_path, p_path, path)) {
		return path;
	}
	return DirAccess::fix_path(p_path);
}

Error DirAccessExtension::list_dir_begin() {
	Error err = OK;
	GDVIRTUAL_REQUIRED_CALL(_list_dir_begin, err);
	return err;
}

String DirAccessExtension::get_next() {
	String next = "";
	GDVIRTUAL_REQUIRED_CALL(_get_next, next);
	return next;
}

bool DirAccessExtension::current_is_dir() const {
	bool current_is_dir = false;
	GDVIRTUAL_REQUIRED_CALL(_current_is_dir, current_is_dir);
	return current_is_dir;
}

bool DirAccessExtension::current_is_hidden() const {
	bool current_is_hidden = false;
	GDVIRTUAL_REQUIRED_CALL(_current_is_hidden, current_is_hidden);
	return current_is_hidden;
}

void DirAccessExtension::list_dir_end() {
	GDVIRTUAL_REQUIRED_CALL(_list_dir_end);
}

int DirAccessExtension::get_drive_count() {
	int count = 0;
	GDVIRTUAL_REQUIRED_CALL(_get_drive_count, count);
	return count;
}

String DirAccessExtension::get_drive(int p_drive) {
	String drive = "";
	GDVIRTUAL_REQUIRED_CALL(_get_drive, p_drive, drive);
	return drive;
}

int DirAccessExtension::get_current_drive() {
	int drive = 0;
	if (GDVIRTUAL_CALL(_get_current_drive, drive)) {
		return drive;
	}
	return DirAccess::get_current_drive();
}

bool DirAccessExtension::drives_are_shortcuts() {
	bool drives_are_shortcuts = true;
	if (GDVIRTUAL_CALL(_drives_are_shortcuts, drives_are_shortcuts)) {
		return drives_are_shortcuts;
	}
	return DirAccess::drives_are_shortcuts();
}

Error DirAccessExtension::change_dir(String p_dir) {
	Error err = OK;
	GDVIRTUAL_REQUIRED_CALL(_change_dir, p_dir, err);
	return err;
}

String DirAccessExtension::get_current_dir(bool p_include_drive) const {
	String current_dir = "";
	GDVIRTUAL_REQUIRED_CALL(_get_current_dir, p_include_drive, current_dir);
	return current_dir;
}

Error DirAccessExtension::make_dir(String p_dir) {
	Error err = OK;
	GDVIRTUAL_REQUIRED_CALL(_make_dir, p_dir, err);
	return err;
}

Error DirAccessExtension::make_dir_recursive(String p_dir) {
	Error err = OK;
	if (GDVIRTUAL_CALL(_make_dir_recursive, p_dir, err)) {
		return err;
	}
	return DirAccess::make_dir_recursive(p_dir);
}

Error DirAccessExtension::erase_contents_recursive() {
	Error err = OK;
	if (GDVIRTUAL_CALL(_erase_contents_recursive, err)) {
		return err;
	}
	return DirAccess::erase_contents_recursive();
}

bool DirAccessExtension::file_exists(String p_file) {
	bool file_exists = false;
	GDVIRTUAL_REQUIRED_CALL(_file_exists, p_file, file_exists);
	return file_exists;
}

bool DirAccessExtension::dir_exists(String p_file) {
	bool dir_exists = false;
	GDVIRTUAL_REQUIRED_CALL(_file_exists, p_file, dir_exists);
	return dir_exists;
}

bool DirAccessExtension::is_readable(String p_dir) {
	bool is_readable = false;
	if (GDVIRTUAL_CALL(_is_readable, p_dir, is_readable)) {
		return is_readable;
	}
	return DirAccess::is_readable(p_dir);
}

bool DirAccessExtension::is_writable(String p_dir) {
	bool is_writable = false;
	if (GDVIRTUAL_CALL(_is_writable, p_dir, is_writable)) {
		return is_writable;
	}
	return DirAccess::is_writable(p_dir);
}

uint64_t DirAccessExtension::get_space_left() {
	uint64_t space_left = 0;
	GDVIRTUAL_REQUIRED_CALL(_get_space_left, space_left);
	return space_left;
}

Error DirAccessExtension::copy(String p_from, String p_to, int p_chmod_flags) {
	Error err = OK;
	if (GDVIRTUAL_CALL(_copy, p_from, p_to, p_chmod_flags, err)) {
		return err;
	}
	return DirAccess::copy(p_from, p_to, p_chmod_flags);
}

Error DirAccessExtension::rename(String p_from, String p_to) {
	Error err = OK;
	GDVIRTUAL_REQUIRED_CALL(_rename, p_from, p_to, err);
	return err;
}

Error DirAccessExtension::remove(String p_name) {
	Error err = OK;
	GDVIRTUAL_REQUIRED_CALL(_remove, p_name, err);
	return err;
}

bool DirAccessExtension::is_link(String p_file) {
	bool is_link = OK;
	GDVIRTUAL_REQUIRED_CALL(_is_link, p_file, is_link);
	return is_link;
}

String DirAccessExtension::read_link(String p_file) {
	String link = "";
	GDVIRTUAL_REQUIRED_CALL(_read_link, p_file, link);
	return link;
}

Error DirAccessExtension::create_link(String p_source, String p_target) {
	Error err = OK;
	GDVIRTUAL_REQUIRED_CALL(_create_link, p_source, p_target, err);
	return err;
}

String DirAccessExtension::get_filesystem_type() const {
	String filesystem_type = "";
	GDVIRTUAL_REQUIRED_CALL(_get_filesystem_type, filesystem_type);
	return filesystem_type;
}

bool DirAccessExtension::is_case_sensitive(const String &p_path) const {
	bool is_case_sensitive = false;
	if (GDVIRTUAL_CALL(_is_case_sensitive, p_path, is_case_sensitive)) {
		return is_case_sensitive;
	}
	return DirAccess::is_case_sensitive(p_path);
}

void DirAccessExtension::_bind_methods() {
	GDVIRTUAL_BIND(_fix_path, "path");

	GDVIRTUAL_BIND(_list_dir_begin);
	GDVIRTUAL_BIND(_get_next);
	GDVIRTUAL_BIND(_current_is_dir);
	GDVIRTUAL_BIND(_current_is_hidden);

	GDVIRTUAL_BIND(_list_dir_end);

	GDVIRTUAL_BIND(_get_drive_count);
	GDVIRTUAL_BIND(_get_drive, "drive");
	GDVIRTUAL_BIND(_get_current_drive);
	GDVIRTUAL_BIND(_drives_are_shortcuts);

	GDVIRTUAL_BIND(_change_dir, "dir");
	GDVIRTUAL_BIND(_get_current_dir, "include_drive");
	GDVIRTUAL_BIND(_make_dir, "dir");
	GDVIRTUAL_BIND(_make_dir_recursive, "dir");
	GDVIRTUAL_BIND(_erase_contents_recursive);

	GDVIRTUAL_BIND(_file_exists, "file");
	GDVIRTUAL_BIND(_dir_exists, "dir");
	GDVIRTUAL_BIND(_is_readable, "dir");
	GDVIRTUAL_BIND(_is_writable, "dir");
	GDVIRTUAL_BIND(_get_space_left);

	GDVIRTUAL_BIND(_copy, "from", "to", "chmod_flags");
	GDVIRTUAL_BIND(_rename, "from", "to");
	GDVIRTUAL_BIND(_remove, "name");

	GDVIRTUAL_BIND(_is_link, "file")
	GDVIRTUAL_BIND(_read_link, "file")
	GDVIRTUAL_BIND(_create_link, "source", "target")

	GDVIRTUAL_BIND(_get_filesystem_type)

	GDVIRTUAL_BIND(_is_case_sensitive, "path");
}

DirAccessExtension::DirAccessExtension() {
}

DirAccessExtension::~DirAccessExtension() {
}
