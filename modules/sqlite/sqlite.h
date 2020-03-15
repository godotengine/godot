/*************************************************************************/
/*  sqlite.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

// Original author: @Khairul Hidayat
// Copyright (c) 2017-2019

#ifndef SQLITE_H
#define SQLITE_H

#include "core/object/reference.h"

// SQLite3
#include "thirdparty/sqlite/spmemvfs.h"
#include "thirdparty/sqlite/sqlite3.h"

class SQLite : public Reference {
	GDCLASS(SQLite, Reference);

private:
	// sqlite handler
	sqlite3 *db;

	// vfs
	spmemvfs_db_t p_db;
	bool memory_read;

	sqlite3_stmt *prepare(const char *statement);
	Array fetch_rows(String query, Array args, int result_type = RESULT_BOTH);
	Dictionary parse_row(sqlite3_stmt *stmt, int result_type);
	sqlite3 *get_handler() { return (memory_read ? p_db.handle : db); }
	bool bind_args(sqlite3_stmt *stmt, Array args);

protected:
	static void _bind_methods();

public:
	enum {
		RESULT_BOTH = 0,
		RESULT_NUM,
		RESULT_ASSOC
	};

	// constructor
	SQLite();
	~SQLite();
	void _init() {}

	// methods
	bool open(String path);
	bool open_buffered(String name, PackedByteArray buffers, int64_t size);
	void close();

	bool query(String statement);
	bool query_with_args(String statement, Array args);
	Array fetch_array(String statement);
	Array fetch_array_with_args(String statement, Array args);
	Array fetch_assoc(String statement);
	Array fetch_assoc_with_args(String statement, Array args);
};

#endif // SQLITE_H
