/*************************************************************************/
/*  file_access_compressed.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "file_access_compressed.h"
#include "print_string.h"
void FileAccessCompressed::configure(const String &p_magic, Compression::Mode p_mode, int p_block_size) {

	magic = p_magic.ascii().get_data();
	if (magic.length() > 4)
		magic = magic.substr(0, 4);
	else {
		while (magic.length() < 4)
			magic += " ";
	}

	cmode = p_mode;
	block_size = p_block_size;
}

#define WRITE_FIT(m_bytes)                                  \
	{                                                       \
		if (write_pos + (m_bytes) > write_max) {            \
			write_max = write_pos + (m_bytes);              \
		}                                                   \
		if (write_max > write_buffer_size) {                \
			write_buffer_size = next_power_of_2(write_max); \
			buffer.resize(write_buffer_size);               \
			write_ptr = buffer.ptrw();                      \
		}                                                   \
	}

Error FileAccessCompressed::open_after_magic(FileAccess *p_base) {

	f = p_base;
	cmode = (Compression::Mode)f->get_32();
	block_size = f->get_32();
	read_total = f->get_32();
	int bc = (read_total / block_size) + 1;
	int acc_ofs = f->get_position() + bc * 4;
	int max_bs = 0;
	for (int i = 0; i < bc; i++) {

		ReadBlock rb;
		rb.offset = acc_ofs;
		rb.csize = f->get_32();
		acc_ofs += rb.csize;
		max_bs = MAX(max_bs, rb.csize);
		read_blocks.push_back(rb);
	}

	comp_buffer.resize(max_bs);
	buffer.resize(block_size);
	read_ptr = buffer.ptrw();
	f->get_buffer(comp_buffer.ptrw(), read_blocks[0].csize);
	at_end = false;
	read_eof = false;
	read_block_count = bc;
	read_block_size = read_blocks.size() == 1 ? read_total : block_size;

	Compression::decompress(buffer.ptrw(), read_block_size, comp_buffer.ptr(), read_blocks[0].csize, cmode);
	read_block = 0;
	read_pos = 0;

	return OK;
}

Error FileAccessCompressed::_open(const String &p_path, int p_mode_flags) {

	ERR_FAIL_COND_V(p_mode_flags == READ_WRITE, ERR_UNAVAILABLE);

	if (f)
		close();

	Error err;
	f = FileAccess::open(p_path, p_mode_flags, &err);
	if (err != OK) {
		//not openable

		f = NULL;
		return err;
	}

	if (p_mode_flags & WRITE) {

		buffer.clear();
		writing = true;
		write_pos = 0;
		write_buffer_size = 256;
		buffer.resize(256);
		write_max = 0;
		write_ptr = buffer.ptrw();

		//don't store anything else unless it's done saving!
	} else {

		char rmagic[5];
		f->get_buffer((uint8_t *)rmagic, 4);
		rmagic[4] = 0;
		if (magic != rmagic) {
			memdelete(f);
			f = NULL;
			return ERR_FILE_UNRECOGNIZED;
		}

		open_after_magic(f);
	}

	return OK;
}
void FileAccessCompressed::close() {

	if (!f)
		return;

	if (writing) {
		//save block table and all compressed blocks

		CharString mgc = magic.utf8();
		f->store_buffer((const uint8_t *)mgc.get_data(), mgc.length()); //write header 4
		f->store_32(cmode); //write compression mode 4
		f->store_32(block_size); //write block size 4
		f->store_32(write_max); //max amount of data written 4
		int bc = (write_max / block_size) + 1;

		for (int i = 0; i < bc; i++) {
			f->store_32(0); //compressed sizes, will update later
		}

		Vector<int> block_sizes;
		for (int i = 0; i < bc; i++) {

			int bl = i == (bc - 1) ? write_max % block_size : block_size;
			uint8_t *bp = &write_ptr[i * block_size];

			Vector<uint8_t> cblock;
			cblock.resize(Compression::get_max_compressed_buffer_size(bl, cmode));
			int s = Compression::compress(cblock.ptrw(), bp, bl, cmode);

			f->store_buffer(cblock.ptr(), s);
			block_sizes.push_back(s);
		}

		f->seek(16); //ok write block sizes
		for (int i = 0; i < bc; i++)
			f->store_32(block_sizes[i]);
		f->seek_end();
		f->store_buffer((const uint8_t *)mgc.get_data(), mgc.length()); //magic at the end too

		buffer.clear();

	} else {

		comp_buffer.clear();
		buffer.clear();
		read_blocks.clear();
	}

	memdelete(f);
	f = NULL;
}

bool FileAccessCompressed::is_open() const {

	return f != NULL;
}

void FileAccessCompressed::seek(size_t p_position) {

	ERR_FAIL_COND(!f);
	if (writing) {

		ERR_FAIL_COND(p_position > write_max);

		write_pos = p_position;

	} else {

		ERR_FAIL_COND(p_position > read_total);
		if (p_position == read_total) {
			at_end = true;
		} else {

			int block_idx = p_position / block_size;
			if (block_idx != read_block) {

				read_block = block_idx;
				f->seek(read_blocks[read_block].offset);
				f->get_buffer(comp_buffer.ptrw(), read_blocks[read_block].csize);
				Compression::decompress(buffer.ptrw(), read_blocks.size() == 1 ? read_total : block_size, comp_buffer.ptr(), read_blocks[read_block].csize, cmode);
				read_block_size = read_block == read_block_count - 1 ? read_total % block_size : block_size;
			}

			read_pos = p_position % block_size;
		}
	}
}

void FileAccessCompressed::seek_end(int64_t p_position) {

	ERR_FAIL_COND(!f);
	if (writing) {

		seek(write_max + p_position);
	} else {

		seek(read_total + p_position);
	}
}
size_t FileAccessCompressed::get_position() const {

	ERR_FAIL_COND_V(!f, 0);
	if (writing) {

		return write_pos;
	} else {

		return read_block * block_size + read_pos;
	}
}
size_t FileAccessCompressed::get_len() const {

	ERR_FAIL_COND_V(!f, 0);
	if (writing) {

		return write_max;
	} else {
		return read_total;
	}
}

bool FileAccessCompressed::eof_reached() const {

	ERR_FAIL_COND_V(!f, false);
	if (writing) {
		return false;
	} else {
		return read_eof;
	}
}

uint8_t FileAccessCompressed::get_8() const {

	ERR_FAIL_COND_V(writing, 0);
	ERR_FAIL_COND_V(!f, 0);

	if (at_end) {
		read_eof = true;
		return 0;
	}

	uint8_t ret = read_ptr[read_pos];

	read_pos++;
	if (read_pos >= read_block_size) {
		read_block++;

		if (read_block < read_block_count) {
			//read another block of compressed data
			f->get_buffer(comp_buffer.ptrw(), read_blocks[read_block].csize);
			Compression::decompress(buffer.ptrw(), read_blocks.size() == 1 ? read_total : block_size, comp_buffer.ptr(), read_blocks[read_block].csize, cmode);
			read_block_size = read_block == read_block_count - 1 ? read_total % block_size : block_size;
			read_pos = 0;

		} else {
			read_block--;
			at_end = true;
			ret = 0;
		}
	}

	return ret;
}
int FileAccessCompressed::get_buffer(uint8_t *p_dst, int p_length) const {

	ERR_FAIL_COND_V(writing, 0);
	ERR_FAIL_COND_V(!f, 0);

	if (at_end) {
		read_eof = true;
		return 0;
	}

	for (int i = 0; i < p_length; i++) {

		p_dst[i] = read_ptr[read_pos];
		read_pos++;
		if (read_pos >= read_block_size) {
			read_block++;

			if (read_block < read_block_count) {
				//read another block of compressed data
				f->get_buffer(comp_buffer.ptrw(), read_blocks[read_block].csize);
				Compression::decompress(buffer.ptrw(), read_blocks.size() == 1 ? read_total : block_size, comp_buffer.ptr(), read_blocks[read_block].csize, cmode);
				read_block_size = read_block == read_block_count - 1 ? read_total % block_size : block_size;
				read_pos = 0;

			} else {
				read_block--;
				at_end = true;
				if (i < p_length - 1)
					read_eof = true;
				return i;
			}
		}
	}

	return p_length;
}

Error FileAccessCompressed::get_error() const {

	return read_eof ? ERR_FILE_EOF : OK;
}

void FileAccessCompressed::flush() {
	ERR_FAIL_COND(!f);
	ERR_FAIL_COND(!writing);

	// compressed files keep data in memory till close()
}

void FileAccessCompressed::store_8(uint8_t p_dest) {

	ERR_FAIL_COND(!f);
	ERR_FAIL_COND(!writing);

	WRITE_FIT(1);
	write_ptr[write_pos++] = p_dest;
}

bool FileAccessCompressed::file_exists(const String &p_name) {

	FileAccess *fa = FileAccess::open(p_name, FileAccess::READ);
	if (!fa)
		return false;
	memdelete(fa);
	return true;
}

uint64_t FileAccessCompressed::_get_modified_time(const String &p_file) {

	if (f)
		return f->get_modified_time(p_file);
	else
		return 0;
}

FileAccessCompressed::FileAccessCompressed() {

	f = NULL;
	magic = "GCMP";
	cmode = Compression::MODE_ZSTD;
	writing = false;
	write_ptr = 0;
	write_buffer_size = 0;
	write_max = 0;
	block_size = 0;
	read_eof = false;
	at_end = false;
	read_total = 0;
	read_ptr = NULL;
	read_block = 0;
	read_block_count = 0;
	read_block_size = 0;
	read_pos = 0;
}

FileAccessCompressed::~FileAccessCompressed() {

	if (f)
		close();
}
