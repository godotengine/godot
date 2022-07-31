/*************************************************************************/
/*  file_access.h                                                        */
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

#ifndef FILE_ACCESS_H
#define FILE_ACCESS_H

#include "core/math/math_defs.h"
#include "core/os/memory.h"
#include "core/typedefs.h"
#include "core/ustring.h"

/**
 * Multi-Platform abstraction for accessing to files.
 */

class FileAccess {
public:
	enum AccessType {
		ACCESS_RESOURCES,
		ACCESS_USERDATA,
		ACCESS_FILESYSTEM,
		ACCESS_MAX
	};

	typedef void (*FileCloseFailNotify)(const String &);

	typedef FileAccess *(*CreateFunc)();
	bool endian_swap;
	bool real_is_double;

	virtual uint32_t _get_unix_permissions(const String &p_file) = 0;
	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions) = 0;

protected:
	String fix_path(const String &p_path) const;
	virtual Error _open(const String &p_path, int p_mode_flags) = 0; ///< open a file
	virtual uint64_t _get_modified_time(const String &p_file) = 0;

	static FileCloseFailNotify close_fail_notify;

private:
	static bool backup_save;

	AccessType _access_type;
	static CreateFunc create_func[ACCESS_MAX]; /** default file access creation function for a platform */
	template <class T>
	static FileAccess *_create_builtin() {
		return memnew(T);
	}

public:
	static void set_file_close_fail_notify_callback(FileCloseFailNotify p_cbk) { close_fail_notify = p_cbk; }

	virtual void _set_access_type(AccessType p_access);

	enum ModeFlags {

		READ = 1,
		WRITE = 2,
		READ_WRITE = 3,
		WRITE_READ = 7,
	};

	virtual void close() = 0; ///< close a file
	virtual bool is_open() const = 0; ///< true when file is open

	virtual String get_path() const { return ""; } /// returns the path for the current open file
	virtual String get_path_absolute() const { return ""; } /// returns the absolute path for the current open file

	virtual void seek(uint64_t p_position) = 0; ///< seek to a given position
	virtual void seek_end(int64_t p_position = 0) = 0; ///< seek from the end of file with negative offset
	virtual uint64_t get_position() const = 0; ///< get position in the file
	virtual uint64_t get_len() const = 0; ///< get size of the file

	virtual bool eof_reached() const = 0; ///< reading passed EOF

	virtual uint8_t get_8() const = 0; ///< get a byte
	virtual uint16_t get_16() const; ///< get 16 bits uint
	virtual uint32_t get_32() const; ///< get 32 bits uint
	virtual uint64_t get_64() const; ///< get 64 bits uint

	virtual float get_float() const;
	virtual double get_double() const;
	virtual real_t get_real() const;

	virtual uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const; ///< get an array of bytes
	virtual String get_line() const;
	virtual String get_token() const;
	virtual Vector<String> get_csv_line(const String &p_delim = ",") const;
	virtual String get_as_utf8_string(bool p_skip_cr = true) const; // Skip CR by default for compat.

	/**< use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
	 * It's not about the current CPU type but file formats.
	 * this flags get reset to false (little endian) on each open
	 */

	virtual void set_endian_swap(bool p_swap) { endian_swap = p_swap; }
	inline bool get_endian_swap() const { return endian_swap; }

	virtual Error get_error() const = 0; ///< get last error

	virtual void flush() = 0;
	virtual void store_8(uint8_t p_dest) = 0; ///< store a byte
	virtual void store_16(uint16_t p_dest); ///< store 16 bits uint
	virtual void store_32(uint32_t p_dest); ///< store 32 bits uint
	virtual void store_64(uint64_t p_dest); ///< store 64 bits uint

	virtual void store_float(float p_dest);
	virtual void store_double(double p_dest);
	virtual void store_real(real_t p_real);

	virtual void store_string(const String &p_string);
	virtual void store_line(const String &p_line);
	virtual void store_csv_line(const Vector<String> &p_values, const String &p_delim = ",");

	virtual void store_pascal_string(const String &p_string);
	virtual String get_pascal_string();

	virtual void store_buffer(const uint8_t *p_src, uint64_t p_length); ///< store an array of bytes

	virtual bool file_exists(const String &p_name) = 0; ///< return true if a file exists

	virtual Error reopen(const String &p_path, int p_mode_flags); ///< does not change the AccessType

	static FileAccess *create(AccessType p_access); /// Create a file access (for the current platform) this is the only portable way of accessing files.
	static FileAccess *create_for_path(const String &p_path);
	static FileAccess *open(const String &p_path, int p_mode_flags, Error *r_error = nullptr); /// Create a file access (for the current platform) this is the only portable way of accessing files.
	static CreateFunc get_create_func(AccessType p_access);
	static bool exists(const String &p_name); ///< return true if a file exists
	static uint64_t get_modified_time(const String &p_file);
	static uint32_t get_unix_permissions(const String &p_file);
	static Error set_unix_permissions(const String &p_file, uint32_t p_permissions);

	static void set_backup_save(bool p_enable) { backup_save = p_enable; };
	static bool is_backup_save_enabled() { return backup_save; };

	static String get_md5(const String &p_file);
	static String get_sha256(const String &p_file);
	static String get_multiple_md5(const Vector<String> &p_file);

	static Vector<uint8_t> get_file_as_array(const String &p_path, Error *r_error = nullptr);
	static String get_file_as_string(const String &p_path, Error *r_error = nullptr);

	template <class T>
	static void make_default(AccessType p_access) {
		create_func[p_access] = _create_builtin<T>;
	}

	FileAccess();
	virtual ~FileAccess() {}
};

struct FileAccessRef {
	_FORCE_INLINE_ FileAccess *operator->() {
		return f;
	}

	operator bool() const { return f != nullptr; }
	FileAccess *f;
	operator FileAccess *() { return f; }
	FileAccessRef(FileAccess *fa) { f = fa; }
	FileAccessRef(FileAccessRef &&other) {
		f = other.f;
		other.f = nullptr;
	}
	~FileAccessRef() {
		if (f) {
			memdelete(f);
		}
	}
};

#endif // FILE_ACCESS_H
