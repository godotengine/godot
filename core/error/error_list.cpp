/*************************************************************************/
/*  error_list.cpp                                                       */
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

#include "error_list.h"

const char *error_names[] = {
	"OK",
	"FAILED",
	"ERR_UNAVAILABLE",
	"ERR_UNCONFIGURED",
	"ERR_UNAUTHORIZED",
	"ERR_PARAMETER_RANGE_ERROR",
	"ERR_OUT_OF_MEMORY",
	"ERR_FILE_NOT_FOUND",
	"ERR_FILE_BAD_DRIVE",
	"ERR_FILE_BAD_PATH",
	"ERR_FILE_NO_PERMISSION",
	"ERR_FILE_ALREADY_IN_USE",
	"ERR_FILE_CANT_OPEN",
	"ERR_FILE_CANT_WRITE",
	"ERR_FILE_CANT_READ",
	"ERR_FILE_UNRECOGNIZED",
	"ERR_FILE_CORRUPT",
	"ERR_FILE_MISSING_DEPENDENCIES",
	"ERR_FILE_EOF",
	"ERR_CANT_OPEN",
	"ERR_CANT_CREATE",
	"ERR_QUERY_FAILED",
	"ERR_ALREADY_IN_USE",
	"ERR_LOCKED",
	"ERR_TIMEOUT",
	"ERR_CANT_CONNECT",
	"ERR_CANT_RESOLVE",
	"ERR_CONNECTION_ERROR",
	"ERR_CANT_ACQUIRE_RESOURCE",
	"ERR_CANT_FORK",
	"ERR_INVALID_DATA",
	"ERR_INVALID_PARAMETER",
	"ERR_ALREADY_EXISTS",
	"ERR_DOES_NOT_EXIST",
	"ERR_DATABASE_CANT_READ",
	"ERR_DATABASE_CANT_WRITE",
	"ERR_COMPILATION_FAILED",
	"ERR_METHOD_NOT_FOUND",
	"ERR_LINK_FAILED",
	"ERR_SCRIPT_FAILED",
	"ERR_CYCLIC_LINK",
	"ERR_INVALID_DECLARATION",
	"ERR_DUPLICATE_SYMBOL",
	"ERR_PARSE_ERROR",
	"ERR_BUSY",
	"ERR_SKIP",
	"ERR_HELP",
	"ERR_BUG",
	"ERR_PRINTER_ON_FIRE",
};

const char *error_names_readable[] = {
	"OK", // OK
	"Generic error", // FAILED
	"Requested operation is unsupported/unavailable",
	"The object hasn't been set up properly",
	"Missing credentials for requested resource",
	"Parameter out of range",
	"Out of memory",
	"File not found",
	"Bad drive",
	"Bad path",
	"Permission denied",
	"Already in use",
	"Can't open file",
	"Can't write file",
	"Can't read file",
	"File unrecognized",
	"File corrupt",
	"Missing dependencies for file",
	"Unexpected eof",
	"Can't open resource/socket/file", // File too? What's the difference to ERR_FILE_CANT_OPEN
	"Can't create", // What can't be created,
	"Query failed", // What query,
	"Already in use",
	"Resource is locked",
	"Timeout",
	"Can't connect",
	"Can't resolve hostname", // I guessed it's the hostname here.
	"Connection error",
	"Can't acquire resource",
	"Can't fork",
	"Invalid data",
	"Invalid parameter",
	"Item already exists",
	"Item does not exist",
	"Can't read from database", // Comments say, it's full? Is that correct?
	"Can't write to database", // Is the database always full when this is raised?
	"Compilation failed",
	"Method not found",
	"Link failed",
	"Script failed",
	"Cyclic link detected",
	"Invalid declaration",
	"Duplicate symbol",
	"Parse error",
	"Resource is busy",
	"Skip error", // ???? What's this? String taken from the docs
	"Help error", // More specific?
	"Bug",
	"Printer on fire",
};

static_assert(sizeof(error_names) / sizeof(*error_names) == ERR_MAX);
static_assert(sizeof(error_names_readable) / sizeof(*error_names_readable) == ERR_MAX);
