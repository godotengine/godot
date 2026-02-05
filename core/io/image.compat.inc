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

static Image::CompressProfile _astc_format_to_compress_profile(Image::ASTCFormat p_format) {
	switch (p_format) {
		case Image::ASTC_FORMAT_4x4:
			return Image::COMPRESS_PROFILE_MAX_QUALITY;
		case Image::ASTC_FORMAT_8x8:
			return Image::COMPRESS_PROFILE_MAX_COMPRESSION;
		default:
			return Image::COMPRESS_PROFILE_BALANCED;
	}
}

Error Image::_compress_bind_compat_115003(CompressMode p_mode, CompressSource p_source, ASTCFormat p_format) {
	return compress(p_mode, p_source, _astc_format_to_compress_profile(p_format));
}

Error Image::_compress_from_channels_compat_115003(CompressMode p_mode, UsedChannels p_channels, ASTCFormat p_format) {
	return compress_from_channels(p_mode, p_channels, _astc_format_to_compress_profile(p_format));
}

void Image::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("compress", "mode", "source", "astc_format"), &Image::_compress_bind_compat_115003, DEFVAL(COMPRESS_SOURCE_GENERIC), DEFVAL(ASTC_FORMAT_4x4));
	ClassDB::bind_compatibility_method(D_METHOD("compress_from_channels", "mode", "channels", "astc_format"), &Image::_compress_from_channels_compat_115003, DEFVAL(ASTC_FORMAT_4x4));
}

#endif
