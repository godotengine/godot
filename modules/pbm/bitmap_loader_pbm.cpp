/*************************************************************************/
/*  bitmap_loader_pbm.cpp                                                */
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
#include "bitmap_loader_pbm.h"
#include "os/file_access.h"
#include "scene/resources/bit_mask.h"

static bool _get_token(FileAccessRef &f, uint8_t &saved, PoolVector<uint8_t> &r_token, bool p_binary = false, bool p_single_chunk = false) {

	int token_max = r_token.size();
	PoolVector<uint8_t>::Write w;
	if (token_max)
		w = r_token.write();
	int ofs = 0;
	bool lf = false;

	while (true) {

		uint8_t b;
		if (saved) {
			b = saved;
			saved = 0;
		} else {
			b = f->get_8();
		}
		if (f->eof_reached()) {
			if (ofs) {
				w = PoolVector<uint8_t>::Write();
				r_token.resize(ofs);
				return true;
			} else {
				return false;
			}
		}

		if (!ofs && !p_binary && b == '#') {
			//skip comment
			while (b != '\n') {
				if (f->eof_reached()) {
					return false;
				}

				b = f->get_8();
			}

			lf = true;

		} else if (b <= 32 && !(p_binary && (ofs || lf))) {

			if (b == '\n') {
				lf = true;
			}

			if (ofs && !p_single_chunk) {
				w = PoolVector<uint8_t>::Write();
				r_token.resize(ofs);
				saved = b;

				return true;
			}
		} else {

			bool resized = false;
			while (ofs >= token_max) {
				if (token_max)
					token_max <<= 1;
				else
					token_max = 1;
				resized = true;
			}
			if (resized) {
				w = PoolVector<uint8_t>::Write();
				r_token.resize(token_max);
				w = r_token.write();
			}
			w[ofs++] = b;
		}
	}

	return false;
}

static int _get_number_from_token(PoolVector<uint8_t> &r_token) {

	int len = r_token.size();
	PoolVector<uint8_t>::Read r = r_token.read();
	return String::to_int((const char *)r.ptr(), len);
}

RES ResourceFormatPBM::load(const String &p_path, const String &p_original_path, Error *r_error) {

#define _RETURN(m_err)        \
	{                         \
		if (r_error)          \
			*r_error = m_err; \
		ERR_FAIL_V(RES());    \
	}

	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ);
	uint8_t saved = 0;
	if (!f)
		_RETURN(ERR_CANT_OPEN);

	PoolVector<uint8_t> token;

	if (!_get_token(f, saved, token)) {
		_RETURN(ERR_PARSE_ERROR);
	}

	if (token.size() != 2) {
		_RETURN(ERR_FILE_CORRUPT);
	}
	if (token[0] != 'P') {
		_RETURN(ERR_FILE_CORRUPT);
	}
	if (token[1] != '1' && token[1] != '4') {
		_RETURN(ERR_FILE_CORRUPT);
	}

	bool bits = token[1] == '4';

	if (!_get_token(f, saved, token)) {
		_RETURN(ERR_PARSE_ERROR);
	}

	int width = _get_number_from_token(token);
	if (width <= 0) {
		_RETURN(ERR_FILE_CORRUPT);
	}

	if (!_get_token(f, saved, token)) {
		_RETURN(ERR_PARSE_ERROR);
	}

	int height = _get_number_from_token(token);
	if (height <= 0) {
		_RETURN(ERR_FILE_CORRUPT);
	}

	Ref<BitMap> bm;
	bm.instance();
	bm->create(Size2i(width, height));

	if (!bits) {

		int required_bytes = width * height;
		if (!_get_token(f, saved, token, false, true)) {
			_RETURN(ERR_PARSE_ERROR);
		}

		if (token.size() < required_bytes) {
			_RETURN(ERR_FILE_CORRUPT);
		}

		PoolVector<uint8_t>::Read r = token.read();

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				char num = r[i * width + j];
				bm->set_bit(Point2i(j, i), num == '0');
			}
		}

	} else {
		//a single, entire token of bits!
		if (!_get_token(f, saved, token, true)) {
			_RETURN(ERR_PARSE_ERROR);
		}
		int required_bytes = Math::ceil((width * height) / 8.0);
		if (token.size() < required_bytes) {
			_RETURN(ERR_FILE_CORRUPT);
		}

		PoolVector<uint8_t>::Read r = token.read();
		int bitwidth = width;
		if (bitwidth % 8)
			bitwidth += 8 - (bitwidth % 8);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {

				int ofs = bitwidth * i + j;

				uint8_t byte = r[ofs / 8];
				bool bit = (byte >> (7 - (ofs % 8))) & 1;

				bm->set_bit(Point2i(j, i), !bit);
			}
		}
	}

	return bm;
}

void ResourceFormatPBM::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("pbm");
}
bool ResourceFormatPBM::handles_type(const String &p_type) const {
	return p_type == "BitMap";
}
String ResourceFormatPBM::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "pbm")
		return "BitMap";
	return "";
}
