/*************************************************************************/
/*  dir_access_android.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef ANDROID_NATIVE_ACTIVITY
#include "dir_access_android.h"
#include "file_access_android.h"

DirAccess *DirAccessAndroid::create_fs() {

	return memnew(DirAccessAndroid);
}

Error DirAccessAndroid::list_dir_begin() {

	list_dir_end();

	AAssetDir *aad = AAssetManager_openDir(FileAccessAndroid::asset_manager, current_dir.utf8().get_data());
	if (!aad)
		return ERR_CANT_OPEN; //nothing

	return OK;
}

String DirAccessAndroid::get_next() {

	const char *fn = AAssetDir_getNextFileName(aad);
	if (!fn)
		return "";
	String s;
	s.parse_utf8(fn);
	current = s;
	return s;
}

bool DirAccessAndroid::current_is_dir() const {

	String sd;
	if (current_dir == "")
		sd = current;
	else
		sd = current_dir + "/" + current;

	AAssetDir *aad2 = AAssetManager_openDir(FileAccessAndroid::asset_manager, sd.utf8().get_data());
	if (aad2) {

		AAssetDir_close(aad2);
		return true;
	}

	return false;
}

bool DirAccessAndroid::current_is_hidden() const {
	return current != "." && current != ".." && current.begins_with(".");
}

void DirAccessAndroid::list_dir_end() {

	if (aad == NULL)
		return;

	AAssetDir_close(aad);
	aad = NULL;
}

int DirAccessAndroid::get_drive_count() {

	return 0;
}

String DirAccessAndroid::get_drive(int p_drive) {

	return "";
}

Error DirAccessAndroid::change_dir(String p_dir) {

	p_dir = p_dir.simplify_path();

	if (p_dir == "" || p_dir == "." || (p_dir == ".." && current_dir == ""))
		return OK;

	String new_dir;

	if (p_dir.begins_with("/"))
		new_dir = p_dir.substr(1, p_dir.length());
	else if (p_dir.begins_with("res://"))
		new_dir = p_dir.substr(6, p_dir.length());
	else //relative
		new_dir = new_dir + "/" + p_dir;

	//test if newdir exists
	new_dir = new_dir.simplify_path();

	AAssetDir *aad = AAssetManager_openDir(FileAccessAndroid::asset_manager, new_dir.utf8().get_data());
	if (aad) {

		current_dir = new_dir;
		AAssetDir_close(aad);
		return OK;
	}

	return ERR_INVALID_PARAMETER;
}

String DirAccessAndroid::get_current_dir() {

	return "/" + current_dir;
}

bool DirAccessAndroid::file_exists(String p_file) {

	String sd;
	if (current_dir == "")
		sd = p_file;
	else
		sd = current_dir + "/" + p_file;

	AAsset *a = AAssetManager_open(FileAccessAndroid::asset_manager, sd.utf8().get_data(), AASSET_MODE_STREAMING);
	if (a) {
		AAsset_close(a);
		return true;
	}

	return false;
}

Error DirAccessAndroid::make_dir(String p_dir) {

	ERR_FAIL_V(ERR_UNAVAILABLE);
}

Error DirAccessAndroid::rename(String p_from, String p_to) {

	ERR_FAIL_V(ERR_UNAVAILABLE);
}

Error DirAccessAndroid::remove(String p_name) {

	ERR_FAIL_V(ERR_UNAVAILABLE);
}

//FileType get_file_type() const;
size_t DirAccessAndroid::get_space_left() {

	return 0;
}

void DirAccessAndroid::make_default() {

	instance_func = create_fs;
}

DirAccessAndroid::DirAccessAndroid() {

	aad = NULL;
}

DirAccessAndroid::~DirAccessAndroid() {

	list_dir_end();
}
#endif
