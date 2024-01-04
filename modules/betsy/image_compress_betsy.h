/**************************************************************************/
/*  image_compress_betsy.h                                                */
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

#ifndef IMAGE_COMPRESS_BETSY_H
#define IMAGE_COMPRESS_BETSY_H

#include "core/io/image.h"

enum BetsyFormat {
	BETSY_FORMAT_BC1,
	BETSY_FORMAT_BC3,
	BETSY_FORMAT_BC4U,
	BETSY_FORMAT_BC5U,
	BETSY_FORMAT_BC6UF,
	BETSY_FORMAT_BC6SF,
	//BETSY_FORMAT_BC7,
	BETSY_FORMAT_ETC1,
	BETSY_FORMAT_ETC2_RGB8,
	BETSY_FORMAT_ETC2_RGBA8,
	BETSY_FORMAT_ETC2_R11,
	BETSY_FORMAT_ETC2_RG11,
	//BETSY_FORMAT_ASTC_4,
	//BETSY_FORMAT_ASTC_4_HDR,
	//BETSY_FORMAT_ASTC_8,
	//BETSY_FORMAT_ASTC_8_HDR,
};

void _compress_betsy(BetsyFormat p_format, Image *r_img);

void _betsy_compress_bptc(Image *r_img, Image::UsedChannels p_channels);
void _betsy_compress_bc(Image *r_img, Image::UsedChannels p_channels);
void _betsy_compress_etc1(Image *r_img);
void _betsy_compress_etc2(Image *r_img, Image::UsedChannels p_channels);

#endif // IMAGE_COMPRESS_BETSY_H
