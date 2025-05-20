/**************************************************************************/
/*  image.compat.inc                                                      */
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

Error Image::_save_png_compat_100447(const String &p_path) const {
	return save_png(p_path, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

Error Image::_save_jpg_compat_100447(const String &p_path, float p_quality) const {
	return save_jpg(p_path, p_quality, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

Error Image::_save_dds_compat_100447(const String &p_path) const {
	return save_dds(p_path, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

Error Image::_save_exr_compat_100447(const String &p_path, bool p_grayscale) const {
	return save_exr(p_path, p_grayscale, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

Error Image::_save_webp_compat_100447(const String &p_path, const bool p_lossy, const float p_quality) const {
	return save_webp(p_path, p_lossy, p_quality, FileAccess::SAVE_INTEGRITY_DEFAULT);
}

void Image::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("save_png", "path"), &Image::_save_png_compat_100447);
	ClassDB::bind_compatibility_method(D_METHOD("save_jpg", "path", "quality"), &Image::_save_jpg_compat_100447, DEFVAL(0.75));
	ClassDB::bind_compatibility_method(D_METHOD("save_dds", "path"), &Image::_save_dds_compat_100447);
	ClassDB::bind_compatibility_method(D_METHOD("save_exr", "path", "grayscale"), &Image::_save_exr_compat_100447, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("save_webp", "path", "lossy", "quality"), &Image::_save_webp_compat_100447, DEFVAL(false), DEFVAL(0.75f));
}

#endif
