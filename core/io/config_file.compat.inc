/**************************************************************************/
/*  config_file.compat.inc                                                */
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

#ifndef DISABLE_DEPRECATED

Error ConfigFile::_save_compat_100447(const String &p_path) {
	return save(p_path, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

Error ConfigFile::_save_encrypted_compat_100447(const String &p_path, const Vector<uint8_t> &p_key) {
	return save_encrypted(p_path, p_key, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

Error ConfigFile::_save_encrypted_pass_compat_100447(const String &p_path, const String &p_pass) {
	return save_encrypted_pass(p_path, p_pass, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

void ConfigFile::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("save", "path"), &ConfigFile::_save_compat_100447);
	ClassDB::bind_compatibility_method(D_METHOD("save_encrypted", "path", "key"), &ConfigFile::_save_encrypted_compat_100447);
	ClassDB::bind_compatibility_method(D_METHOD("save_encrypted_pass", "path", "password"), &ConfigFile::_save_encrypted_pass_compat_100447);
}

#endif
