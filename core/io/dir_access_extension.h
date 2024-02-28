/**************************************************************************/
/*  dir_access_extension.h                                                */
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

#ifndef DIR_ACCESS_EXTENSION_H
#define DIR_ACCESS_EXTENSION_H

#include "core/io/dir_access.h"
#include "core/object/gdvirtual.gen.inc"

class DirAccessExtension : public DirAccess {
	GDCLASS(DirAccessExtension, DirAccess);

protected:
	GDVIRTUAL1RC(String, _fix_path, String);
	virtual String fix_path(const String &p_path) const override;

public:
	GDVIRTUAL0R(Error, _list_dir_begin);
	virtual Error list_dir_begin() override;
	GDVIRTUAL0R(String, _get_next);
	virtual String get_next() override;
	GDVIRTUAL0RC(bool, _current_is_dir);
	virtual bool current_is_dir() const override;
	GDVIRTUAL0RC(bool, _current_is_hidden);
	virtual bool current_is_hidden() const override;

	GDVIRTUAL0(_list_dir_end);
	virtual void list_dir_end() override;

	GDVIRTUAL0R(int, _get_drive_count);
	virtual int get_drive_count() override;
	GDVIRTUAL1R(String, _get_drive, int);
	virtual String get_drive(int p_drive) override;
	GDVIRTUAL0R(int, _get_current_drive);
	virtual int get_current_drive() override;
	GDVIRTUAL0R(bool, _drives_are_shortcuts);
	virtual bool drives_are_shortcuts() override;

	GDVIRTUAL1R(Error, _change_dir, String);
	virtual Error change_dir(String p_dir) override;
	GDVIRTUAL1RC(String, _get_current_dir, bool);
	virtual String get_current_dir(bool p_include_drive = true) const override;
	GDVIRTUAL1R(Error, _make_dir, String);
	virtual Error make_dir(String p_dir) override;
	GDVIRTUAL1R(Error, _make_dir_recursive, String);
	virtual Error make_dir_recursive(const String &p_dir) override;
	GDVIRTUAL0R(Error, _erase_contents_recursive);
	virtual Error erase_contents_recursive() override;

	GDVIRTUAL1R(bool, _file_exists, String);
	virtual bool file_exists(String p_file) override;
	GDVIRTUAL1R(bool, _dir_exists, String);
	virtual bool dir_exists(String p_dir) override;
	GDVIRTUAL1R(bool, _is_readable, String);
	virtual bool is_readable(String p_dir) override;
	GDVIRTUAL1R(bool, _is_writable, String);
	virtual bool is_writable(String p_dir) override;
	GDVIRTUAL0R(uint64_t, _get_space_left);
	virtual uint64_t get_space_left() override;

	GDVIRTUAL3R(Error, _copy, String, String, int);
	virtual Error copy(const String &p_from, const String &p_to, int p_chmod_flags = -1) override;
	GDVIRTUAL2R(Error, _rename, String, String);
	virtual Error rename(String p_from, String p_to) override;
	GDVIRTUAL1R(Error, _remove, String);
	virtual Error remove(String p_name) override;

	GDVIRTUAL1R(bool, _is_link, String);
	virtual bool is_link(String p_file) override;
	GDVIRTUAL1R(String, _read_link, String);
	virtual String read_link(String p_file) override;
	GDVIRTUAL2R(Error, _create_link, String, String);
	virtual Error create_link(String p_source, String p_target) override;

	GDVIRTUAL0RC(String, _get_filesystem_type);
	virtual String get_filesystem_type() const override;

	GDVIRTUAL1RC(bool, _is_case_sensitive, String);
	virtual bool is_case_sensitive(const String &p_path) const override;

protected:
	static void _bind_methods();

public:
	DirAccessExtension();
	~DirAccessExtension();
};

#endif // DIR_ACCESS_EXTENSION_H
