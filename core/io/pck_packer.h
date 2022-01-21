/*************************************************************************/
/*  pck_packer.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PCK_PACKER_H
#define PCK_PACKER_H

#include "core/object/ref_counted.h"

class FileAccess;

class PCKPacker : public RefCounted {
	GDCLASS(PCKPacker, RefCounted);

	FileAccess *file = nullptr;
	int alignment = 0;
	uint64_t ofs = 0;

	Vector<uint8_t> key;
	bool enc_dir = false;

	static void _bind_methods();

	struct File {
		String path;
		String src_path;
		uint64_t ofs = 0;
		uint64_t size = 0;
		bool encrypted = false;
		Vector<uint8_t> md5;
	};
	Vector<File> files;

public:
	Error pck_start(const String &p_file, int p_alignment = 32, const String &p_key = "0000000000000000000000000000000000000000000000000000000000000000", bool p_encrypt_directory = false);
	Error add_file(const String &p_file, const String &p_src, bool p_encrypt = false);
	Error flush(bool p_verbose = false);

	PCKPacker() {}
	~PCKPacker();
};

#endif // PCK_PACKER_H
