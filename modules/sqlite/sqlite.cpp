/*************************************************************************/
/*  sqlite.cpp                                                           */
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

#include "sqlite.h"
#include "core/core_bind.h"
#include "core/os/os.h"
#include "editor/project_settings_editor.h"

SQLite::SQLite() {
	db = nullptr;
	memory_read = false;
}
/*
	Open a database file.
	If this is running outside of the editor, databases under res:// are assumed to be packed.
	@param path The database resource path.
	@return status
*/
bool SQLite::open(String path) {
	if (!path.strip_edges().length())
		return false;

	if (!Engine::get_singleton()->is_editor_hint() && path.begins_with("res://")) {
		Ref<_File> dbfile;
		dbfile.instance();
		if (dbfile->open(path, _File::READ) != Error::OK) {
			print_error("Cannot open packed database!");
			return false;
		}
		int64_t size = dbfile->get_len();
		PackedByteArray buffer = dbfile->get_buffer(size);
		return open_buffered(path, buffer, size);
	}

	String real_path = ProjectSettings::get_singleton()->globalize_path(path.strip_edges());

	int result = sqlite3_open(real_path.utf8().get_data(), &db);

	if (result != SQLITE_OK) {
		print_error("Cannot open database!");
		return false;
	}

	return true;
}
/*
  Open the database and initialize memory buffer.
  @param name Name of the database.
  @param buffers The database buffer.
  @param size Size of the database;
  @return status
*/
bool SQLite::open_buffered(String name, PackedByteArray buffers, int64_t size) {
	if (!name.strip_edges().length()) {
		return false;
	}

	if (!buffers.size() || !size) {
		return false;
	}

	spmembuffer_t *p_mem = (spmembuffer_t *)calloc(1, sizeof(spmembuffer_t));
	p_mem->total = p_mem->used = size;
	p_mem->data = (char *)malloc(size + 1);
	memcpy(p_mem->data, buffers.ptr(), size);
	p_mem->data[size] = '\0';

	//
	spmemvfs_env_init();
	int err = spmemvfs_open_db(&p_db, name.utf8().get_data(), p_mem);

	if (err != SQLITE_OK || p_db.mem != p_mem) {
		print_error("Cannot open buffered database!");
		return false;
	}

	memory_read = true;
	return true;
}

void SQLite::close() {
	if (db) {
		// Cannot close database!
		if (sqlite3_close_v2(db) != SQLITE_OK) {
			print_error("Cannot close database!");
		} else {
			db = nullptr;
		}
	}

	if (memory_read) {
		// Close virtual filesystem database
		spmemvfs_close_db(&p_db);
		spmemvfs_env_fini();
		memory_read = false;
	}
}

sqlite3_stmt *SQLite::prepare(const char *query) {
	// Get database pointer
	sqlite3 *dbs = get_handler();

	if (!dbs) {
		print_error("Cannot prepare query! Database is not opened.");
		return nullptr;
	}

	// Prepare the statement
	sqlite3_stmt *stmt;
	int result = sqlite3_prepare_v2(dbs, query, -1, &stmt, nullptr);

	// Cannot prepare query!
	if (result != SQLITE_OK) {
		print_error("SQL Error: " + String(sqlite3_errmsg(dbs)));
		return nullptr;
	}

	return stmt;
}

bool SQLite::bind_args(sqlite3_stmt *stmt, Array args) {
	// Check parameter count
	int param_count = sqlite3_bind_parameter_count(stmt);
	if (param_count != args.size()) {
		print_error("Query failed; expected " + itos(param_count) + " arguments, got " + itos(args.size()));
		return false;
	}

	/**
	 * SQLite data types:
	 * - NULL
	 * - INTEGER (signed, max 8 bytes)
	 * - REAL (stored as a double-precision float)
	 * - TEXT (stored in database encoding of UTF-8, UTF-16BE or UTF-16LE)
	 * - BLOB (1:1 storage)
	 */

	for (int i = 0; i < param_count; i++) {
		int retcode;
		switch (args[i].get_type()) {
			case Variant::Type::NIL:
				retcode = sqlite3_bind_null(stmt, i + 1);
				break;
			case Variant::Type::BOOL:
			case Variant::Type::INT:
				retcode = sqlite3_bind_int(stmt, i + 1, (int)args[i]);
				break;
			case Variant::Type::FLOAT:
				retcode = sqlite3_bind_double(stmt, i + 1, (double)args[i]);
				break;
			case Variant::Type::STRING:
				retcode = sqlite3_bind_text(stmt, i + 1, String(args[i]).utf8().get_data(), -1, SQLITE_TRANSIENT);
				break;
			case Variant::Type::PACKED_BYTE_ARRAY:
				retcode = sqlite3_bind_blob(stmt, i + 1, PackedByteArray(args[i]).ptr(), PackedByteArray(args[i]).size(), SQLITE_TRANSIENT);
				break;
			default:
				print_error("SQLite was passed unhandled Variant with TYPE_* enum " + itos(args[i].get_type()) + ". Please serialize your object into a String or a PoolByteArray.\n");
				return false;
		}

		if (retcode != SQLITE_OK) {
			print_error("Query failed, an error occured while binding argument" + itos(i + 1) + " of " + itos(args.size()) + " (SQLite errcode " + itos(retcode) + ")");
			return false;
		}
	}

	return true;
}

bool SQLite::query_with_args(String query, Array args) {
	sqlite3_stmt *stmt = prepare(query.utf8().get_data());

	// Failed to prepare the query
	if (!stmt) {
		return false;
	}

	// Error occurred during argument binding
	if (!bind_args(stmt, args)) {
		sqlite3_finalize(stmt);
		return false;
	}

	// Evaluate the sql query
	sqlite3_step(stmt);
	sqlite3_finalize(stmt);

	return true;
}

bool SQLite::query(String query) {
	return this->query_with_args(query, Array());
}

Array SQLite::fetch_rows(String statement, Array args, int result_type) {
	Array result;

	// Empty statement
	if (!statement.strip_edges().length()) {
		return result;
	}

	// Cannot prepare query
	sqlite3_stmt *stmt = prepare(statement.strip_edges().utf8().get_data());
	if (!stmt) {
		return result;
	}

	// Bind arguments
	if (!bind_args(stmt, args)) {
		sqlite3_finalize(stmt);
		return result;
	}

	// Fetch rows
	while (sqlite3_step(stmt) == SQLITE_ROW) {
		// Do a step
		result.append(parse_row(stmt, result_type));
	}

	// Delete prepared statement
	sqlite3_finalize(stmt);

	// Return the result
	return result;
}

Dictionary SQLite::parse_row(sqlite3_stmt *stmt, int result_type) {
	Dictionary result;

	// Get column count
	int col_count = sqlite3_column_count(stmt);

	// Fetch all column
	for (int i = 0; i < col_count; i++) {
		// Key name
		const char *col_name = sqlite3_column_name(stmt, i);
		String key = String(col_name);

		// Value
		int col_type = sqlite3_column_type(stmt, i);
		Variant value;

		// Get column value
		switch (col_type) {
			case SQLITE_INTEGER:
				value = Variant(sqlite3_column_int(stmt, i));
				break;

			case SQLITE_FLOAT:
				value = Variant(sqlite3_column_double(stmt, i));
				break;

			case SQLITE_TEXT: {
				int size = sqlite3_column_bytes(stmt, i);
				String str = String::utf8((const char *)sqlite3_column_text(stmt, i), size);
				value = Variant(str);
				break;
			}
			case SQLITE_BLOB: {
				PackedByteArray arr;
				int size = sqlite3_column_bytes(stmt, i);
				arr.resize(size);
				memcpy((void *)arr.ptr(), sqlite3_column_blob(stmt, i), size);
				value = Variant(arr);
				break;
			}

			default:
				break;
		}

		// Set dictionary value
		if (result_type == RESULT_NUM)
			result[i] = value;
		else if (result_type == RESULT_ASSOC)
			result[key] = value;
		else {
			result[i] = value;
			result[key] = value;
		}
	}

	return result;
}

Array SQLite::fetch_array(String query) {
	return fetch_rows(query, Array(), RESULT_BOTH);
}

Array SQLite::fetch_array_with_args(String query, Array args) {
	return fetch_rows(query, args, RESULT_BOTH);
}

Array SQLite::fetch_assoc(String query) {
	return fetch_rows(query, Array(), RESULT_ASSOC);
}

Array SQLite::fetch_assoc_with_args(String query, Array args) {
	return fetch_rows(query, args, RESULT_ASSOC);
}

SQLite::~SQLite() {
	// Close database
	close();
}

void SQLite::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path"), &SQLite::open);
	ClassDB::bind_method(D_METHOD("open_buffered", "path", "buffers", "size"), &SQLite::open_buffered);
	ClassDB::bind_method(D_METHOD("query", "statement"), &SQLite::query);
	ClassDB::bind_method(D_METHOD("query_with_args", "statement", "args"), &SQLite::query_with_args);
	ClassDB::bind_method(D_METHOD("close"), &SQLite::close);
	ClassDB::bind_method(D_METHOD("fetch_array", "statement"), &SQLite::fetch_array);
	ClassDB::bind_method(D_METHOD("fetch_array_with_args", "statement", "args"), &SQLite::fetch_array_with_args);
	ClassDB::bind_method(D_METHOD("fetch_assoc", "statement"), &SQLite::fetch_assoc);
	ClassDB::bind_method(D_METHOD("fetch_assoc_with_args", "statement", "args"), &SQLite::fetch_assoc_with_args);
}
