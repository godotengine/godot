/**************************************************************************/
/*  godot_sqlite.h                                                        */
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

#ifndef GODOT_SQLITE_H
#define GODOT_SQLITE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "spmemvfs/spmemvfs.h"
#include "sqlite/sqlite3.h"

class SQLite;

class SQLiteQuery : public RefCounted {
	GDCLASS(SQLiteQuery, RefCounted);

	SQLite *db = nullptr;
	sqlite3_stmt *stmt = nullptr;
	String query;

protected:
	static void _bind_methods();

public:
	SQLiteQuery();
	~SQLiteQuery();
	void init(SQLite *p_db, const String &p_query);
	bool is_ready() const;
	String get_last_error_message() const;
	Array get_columns();
	void finalize();
	Variant execute(const Array p_args);
	Variant batch_execute(Array p_rows);

private:
	bool prepare();
};

class SQLite : public RefCounted {
	GDCLASS(SQLite, RefCounted);

	friend SQLiteQuery;

private:
	sqlite3 *db = nullptr;
	spmemvfs_db_t spmemvfs_db{};
	bool memory_read = false;

	::LocalVector<WeakRef *, uint32_t, true> queries;

	sqlite3_stmt *prepare(const char *statement);
	Array fetch_rows(const String &query, const Array &args, int result_type = RESULT_BOTH);
	sqlite3 *get_handler() const { return memory_read ? spmemvfs_db.handle : db; }
	Dictionary parse_row(sqlite3_stmt *stmt, int result_type);

public:
	static bool bind_args(sqlite3_stmt *stmt, const Array &args);

protected:
	static void _bind_methods();

public:
	enum { RESULT_BOTH = 0,
		RESULT_NUM,
		RESULT_ASSOC };

	SQLite();
	~SQLite();

	bool open(const String &path);
	bool open_in_memory();
	bool open_buffered(const String &name, const PackedByteArray &buffers, int64_t size);
	void close();

	Ref<SQLiteQuery> create_query(String p_query);

	String get_last_error_message() const;
};
#endif // GODOT_SQLITE_H
