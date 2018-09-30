/*************************************************************************/
/*  zip.cpp                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/io/zip_io.h"

#include "zip.h"

Error Zip::open(String path, int append) {
	f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&f);
	io.opaque = &f;
	zf = zipOpen2(path.utf8().get_data(), append, NULL, &io);
	return zf != NULL ? OK : FAILED;
}

Error Zip::close() {
	return zipClose(zf, NULL) == ZIP_OK ? OK : FAILED;
}

Error Zip::open_new_file_in_zip(String path) {
	zip_fileinfo zipfi;

	OS::Time time = OS::get_singleton()->get_time();
	OS::Date date = OS::get_singleton()->get_date();

	zipfi.tmz_date.tm_hour = time.hour;
	zipfi.tmz_date.tm_mday = date.day;
	zipfi.tmz_date.tm_min = time.min;
	zipfi.tmz_date.tm_mon = date.month - 1;
	zipfi.tmz_date.tm_sec = time.sec;
	zipfi.tmz_date.tm_year = date.year;
	zipfi.dosDate = 0;
	zipfi.external_fa = 0;
	zipfi.internal_fa = 0;

	int ret = zipOpenNewFileInZip(zf, path.utf8().get_data(), &zipfi, NULL, 0, NULL, 0, NULL, Z_DEFLATED, Z_DEFAULT_COMPRESSION);
	return ret == ZIP_OK ? OK : FAILED;
}

Error Zip::write_in_file_in_zip(Vector<uint8_t> data) {
	return zipWriteInFileInZip(zf, data.ptr(), data.size()) == ZIP_OK ? OK : FAILED;
}

Error Zip::close_file_in_zip() {
	return zipCloseFileInZip(zf) == ZIP_OK ? OK : FAILED;
}

void Zip::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path", "append"), &Zip::open);
	ClassDB::bind_method(D_METHOD("open_new_file_in_zip", "path"), &Zip::open_new_file_in_zip);
	ClassDB::bind_method(D_METHOD("write_in_file_in_zip", "data"), &Zip::write_in_file_in_zip);
	ClassDB::bind_method(D_METHOD("close_file_in_zip"), &Zip::close_file_in_zip);
	ClassDB::bind_method(D_METHOD("close"), &Zip::close);

	BIND_ENUM_CONSTANT(APPEND_CREATE);
	BIND_ENUM_CONSTANT(APPEND_CREATEAFTER);
	BIND_ENUM_CONSTANT(APPEND_ADDINZIP);
}

Zip::Zip() {
	f = NULL;
	zf = NULL;
}

Zip::~Zip() {
}
