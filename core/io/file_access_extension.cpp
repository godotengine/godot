/**************************************************************************/
/*  file_access_extension.cpp                                             */
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

#include "file_access_extension.h"

#include "core/object/object.h"

bool FileAccessExtension::is_open() const {
	bool is_open = false;
	GDVIRTUAL_CALL(_is_open, is_open);
	return is_open;
}

String FileAccessExtension::get_path() const {
	String path = "";
	GDVIRTUAL_CALL(_get_path, path);
	return path;
}

String FileAccessExtension::get_path_absolute() const {
	String path_absolute = "";
	GDVIRTUAL_CALL(_get_path_absolute, path_absolute);
	return path_absolute;
}

void FileAccessExtension::seek(uint64_t p_position) {
	GDVIRTUAL_CALL(_seek, p_position);
}

void FileAccessExtension::seek_end(int64_t p_position) {
	GDVIRTUAL_CALL(_seek_end, p_position);
}

uint64_t FileAccessExtension::get_position() const {
	uint64_t position = 0;
	GDVIRTUAL_CALL(_get_position, position);
	return position;
}

uint64_t FileAccessExtension::get_length() const {
	uint64_t length = 0;
	GDVIRTUAL_CALL(_get_length, length);
	return length;
}

bool FileAccessExtension::eof_reached() const {
	bool eof_reached = false;
	GDVIRTUAL_CALL(_eof_reached, eof_reached);
	return eof_reached;
}

uint8_t FileAccessExtension::get_8() const {
	uint8_t val = 0;
	GDVIRTUAL_CALL(_get_8, val);
	return val;
}

uint16_t FileAccessExtension::get_16() const {
	uint16_t val = 0;
	GDVIRTUAL_CALL(_get_16, val);
	return val;
}

uint32_t FileAccessExtension::get_32() const {
	uint32_t val = 0;
	GDVIRTUAL_CALL(_get_32, val);
	return val;
}

uint64_t FileAccessExtension::get_64() const {
	uint64_t val = 0;
	GDVIRTUAL_CALL(_get_64, val);
	return val;
}

float FileAccessExtension::get_float() const {
	float val = 0;
	GDVIRTUAL_CALL(_get_float, val);
	return val;
}

double FileAccessExtension::get_double() const {
	double val = 0;
	GDVIRTUAL_CALL(_get_double, val);
	return val;
}

real_t FileAccessExtension::get_real() const {
	real_t val = 0;
	GDVIRTUAL_CALL(_get_real, val);
	return val;
}

Variant FileAccessExtension::get_var(bool p_allow_objects) const {
	Variant var;
	GDVIRTUAL_CALL(_get_var, p_allow_objects, var);
	return var;
}

uint64_t FileAccess::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);

	Vector<uint8_t> buffer = get_buffer(p_length);
	p_dst = buffer.ptrw();
	return buffer.size();
}

Vector<uint8_t> FileAccessExtension::get_buffer(int64_t p_length) const {
	Vector<uint8_t> buffer;
	GDVIRTUAL_CALL(_get_buffer, p_length, buffer);
	return buffer;
}

String FileAccessExtension::get_line() const {
	String line;
	GDVIRTUAL_CALL(_get_line, line);
	return line;
}

String FileAccessExtension::get_token() const {
	String token;
	GDVIRTUAL_CALL(_get_token, token);
	return token;
}

Vector<String> FileAccessExtension::get_csv_line(const String &p_delim) const {
	Vector<String> csv_line;
	GDVIRTUAL_CALL(_get_csv_line, p_delim, csv_line);
	return csv_line;
}

String FileAccessExtension::get_as_text(bool p_skip_cr) const {
	String val_as_text;
	GDVIRTUAL_CALL(_get_as_text, p_skip_cr, val_as_text);
	return val_as_text;
}

String FileAccessExtension::get_as_utf8_string(bool p_skip_cr) const {
	String val_as_utf8_string;
	GDVIRTUAL_CALL(_get_as_utf8_string, p_skip_cr, val_as_utf8_string);
	return val_as_utf8_string;
}

void FileAccessExtension::_bind_methods() {
	GDVIRTUAL_BIND(_is_open);

	GDVIRTUAL_BIND(_get_path);
	GDVIRTUAL_BIND(_get_path_absolute);

	GDVIRTUAL_BIND(_seek, "position");
	GDVIRTUAL_BIND(_seek_end, "position");
	GDVIRTUAL_BIND(_get_position);
	GDVIRTUAL_BIND(_get_length);

	GDVIRTUAL_BIND(_eof_reached);

	GDVIRTUAL_BIND(_get_8);
	GDVIRTUAL_BIND(_get_16);
	GDVIRTUAL_BIND(_get_32);
	GDVIRTUAL_BIND(_get_64);

	GDVIRTUAL_BIND(_get_float);
	GDVIRTUAL_BIND(_get_double);
	GDVIRTUAL_BIND(_get_real);

	GDVIRTUAL_BIND(_get_var, "allow_objects");

	GDVIRTUAL_BIND(_get_buffer, "length");
	GDVIRTUAL_BIND(_get_line);
	GDVIRTUAL_BIND(_get_token);
	GDVIRTUAL_BIND(_get_csv_line, "delim");
	GDVIRTUAL_BIND(_get_as_text, "skip_cr");
	GDVIRTUAL_BIND(_get_as_utf8_string, "skip_cr");
}

FileAccessExtension::FileAccessExtension() {
}

FileAccessExtension::~FileAccessExtension() {
}
