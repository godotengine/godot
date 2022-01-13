/*************************************************************************/
/*  dir_access.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DIR_ACCESS_H
#define DIR_ACCESS_H

#include "core/typedefs.h"
#include "core/ustring.h"

//@ TODO, excellent candidate for THREAD_SAFE MACRO, should go through all these and add THREAD_SAFE where it applies
class DirAccess {
public:
	enum AccessType {
		ACCESS_RESOURCES,
		ACCESS_USERDATA,
		ACCESS_FILESYSTEM,
		ACCESS_MAX
	};

	typedef DirAccess *(*CreateFunc)();

private:
	AccessType _access_type;
	static CreateFunc create_func[ACCESS_MAX]; ///< set this to instance a filesystem object

	Error _copy_dir(DirAccess *p_target_da, String p_to, int p_chmod_flags, bool p_copy_links);

protected:
	String _get_root_path() const;
	String _get_root_string() const;

	String fix_path(String p_path) const;
	bool next_is_dir;

	template <class T>
	static DirAccess *_create_builtin() {
		return memnew(T);
	}

public:
	virtual Error list_dir_begin() = 0; ///< This starts dir listing
	virtual String get_next() = 0;
	virtual bool current_is_dir() const = 0;
	virtual bool current_is_hidden() const = 0;

	virtual void list_dir_end() = 0; ///<

	virtual int get_drive_count() = 0;
	virtual String get_drive(int p_drive) = 0;
	virtual int get_current_drive();
	virtual bool drives_are_shortcuts();

	virtual Error change_dir(String p_dir) = 0; ///< can be relative or absolute, return false on success
	virtual String get_current_dir() = 0; ///< return current dir location
	virtual String get_current_dir_without_drive();
	virtual Error make_dir(String p_dir) = 0;
	virtual Error make_dir_recursive(String p_dir);
	virtual Error erase_contents_recursive(); //super dangerous, use with care!

	virtual bool file_exists(String p_file) = 0;
	virtual bool dir_exists(String p_dir) = 0;
	static bool exists(String p_dir);
	virtual uint64_t get_space_left() = 0;

	Error copy_dir(String p_from, String p_to, int p_chmod_flags = -1, bool p_copy_links = false);
	virtual Error copy(String p_from, String p_to, int p_chmod_flags = -1);
	virtual Error rename(String p_from, String p_to) = 0;
	virtual Error remove(String p_name) = 0;

	virtual bool is_link(String p_file) = 0;
	virtual String read_link(String p_file) = 0;
	virtual Error create_link(String p_source, String p_target) = 0;

	// Meant for editor code when we want to quickly remove a file without custom
	// handling (e.g. removing a cache file).
	static void remove_file_or_error(String p_path) {
		DirAccess *da = create(ACCESS_FILESYSTEM);
		if (da->file_exists(p_path)) {
			if (da->remove(p_path) != OK) {
				ERR_FAIL_MSG("Cannot remove file or directory: " + p_path);
			}
		}
		memdelete(da);
	}

	virtual String get_filesystem_type() const = 0;
	static String get_full_path(const String &p_path, AccessType p_access);
	static DirAccess *create_for_path(const String &p_path);

	/*
	enum DirType {

		FILE_TYPE_INVALID,
		FILE_TYPE_FILE,
		FILE_TYPE_DIR,
	};

	//virtual DirType get_file_type() const=0;
*/
	static DirAccess *create(AccessType p_access);

	template <class T>
	static void make_default(AccessType p_access) {
		create_func[p_access] = _create_builtin<T>;
	}

	static DirAccess *open(const String &p_path, Error *r_error = nullptr);

	DirAccess();
	virtual ~DirAccess();
};

struct DirAccessRef {
	_FORCE_INLINE_ DirAccess *operator->() {
		return f;
	}

	operator bool() const { return f != nullptr; }
	DirAccess *f;
	DirAccessRef(DirAccess *fa) { f = fa; }
	~DirAccessRef() {
		if (f) {
			memdelete(f);
		}
	}
};

#endif
