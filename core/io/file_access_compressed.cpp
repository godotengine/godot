/*************************************************************************/
/*  file_access_compressed.cpp                                           */
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

#include "file_access_compressed.h"

#include "core/string/print_string.h"

void FileAccessCompressed::configure(const String &p_magic, Compression::Mode p_mode, uint32_t p_block_size) {
	magic = p_magic.ascii().get_data();
	if (magic.length() > 4) {
		magic = magic.substr(0, 4);
	} else {
		while (magic.length() < 4) {
			magic += " ";
		}
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
	if (block_size == 0) {
		f = nullptr; // Let the caller to handle the FileAccess object if failed to open as compressed file.
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Can't open compressed file '" + p_base->get_path() + "' with block size 0, it is corrupted.");
	}
	read_total = f->get_32();
	uint32_t bc = (read_total / block_size) + 1;
	uint64_t acc_ofs = f->get_position() + bc * 4;
	uint32_t max_bs = 0;
	for (uint32_t i = 0; i < bc; i++) {
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

	if (f) {
		close();
	}

	Error err;
	f = FileAccess::open(p_path, p_mode_flags, &err);
	if (err != OK) {
		//not openable

		f = nullptr;
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
		if (magic != rmagic || open_after_magic(f) != OK) {
			memdelete(f);
			f = nullptr;
			return ERR_FILE_UNRECOGNIZED;
		}
	}

	return OK;
}

void FileAccessCompressed::close() {
	if (!f) {
		return;
	}

	if (writing) {
		//save block table and all compressed blocks

		CharString mgc = magic.utf8();
		f->store_buffer((const uint8_t *)mgc.get_data(), mgc.length()); //write header 4
		f->store_32(cmode); //write compression mode 4
		f->store_32(block_size); //write block size 4
		f->store_32(write_max); //max amount of data written 4
		uint32_t bc = (write_max / block_size) + 1;

		for (uint32_t i = 0; i < bc; i++) {
			f->store_32(0); //compressed sizes, will update later
		}

		Vector<int> block_sizes;
		for (uint32_t i = 0; i < bc; i++) {
			uint32_t bl = i == (bc - 1) ? write_max % block_size : block_size;
			uint8_t *bp = &write_ptr[i * block_size];

			Vector<uint8_t> cblock;
			cblock.resize(Compression::get_max_compressed_buffer_size(bl, cmode));
			int s = Compression::compress(cblock.ptrw(), bp, bl, cmode);

			f->store_buffer(cblock.ptr(), s);
			block_sizes.push_back(s);
		}

		f->seek(16); //ok write block sizes
		for (uint32_t i = 0; i < bc; i++) {
			f->store_32(block_sizes[i]);
		}
		f->seek_end();
		f->store_buffer((const uint8_t *)mgc.get_data(), mgc.length()); //magic at the end too

		buffer.clear();

	} else {
		comp_buffer.clear();
		buffer.clear();
		read_blocks.clear();
	}

	memdelete(f);
	f = nullptr;
}

bool FileAccessCompressed::is_open() const {
	return f != nullptr;
}

void FileAccessCompressed::seek(uint64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	if (writing) {
		ERR_FAIL_COND(p_position > write_max);

		write_pos = p_position;

	} else {
		ERR_FAIL_COND(p_position > read_total);
		if (p_position == read_total) {
			at_end = true;
		} else {
			at_end = false;
			read_eof = false;
			uint32_t block_idx = p_position / block_size;
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
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	if (writing) {
		seek(write_max + p_position);
	} else {
		seek(read_total + p_position);
	}
}

uint64_t FileAccessCompressed::get_position() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	if (writing) {
		return write_pos;
	} else {
		return (uint64_t)read_block * block_size + read_pos;
	}
}

uint64_t FileAccessCompressed::get_length() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	if (writing) {
		return write_max;
	} else {
		return read_total;
	}
}

bool FileAccessCompressed::eof_reached() const {
	ERR_FAIL_COND_V_MSG(!f, false, "File must be opened before use.");
	if (writing) {
		return false;
	} else {
		return read_eof;
	}
}

uint8_t FileAccessCompressed::get_8() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	ERR_FAIL_COND_V_MSG(writing, 0, "File has not been opened in read mode.");

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
		}
	}

	return ret;
}

uint64_t FileAccessCompressed::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);
	ERR_FAIL_COND_V_MSG(!f, -1, "File must be opened before use.");
	ERR_FAIL_COND_V_MSG(writing, -1, "File has not been opened in read mode.");

	if (at_end) {
		read_eof = true;
		return 0;
	}

	for (uint64_t i = 0; i < p_length; i++) {
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
				if (i < p_length - 1) {
					read_eof = true;
				}
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
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	ERR_FAIL_COND_MSG(!writing, "File has not been opened in write mode.");

	// compressed files keep data in memory till close()
}

void FileAccessCompressed::store_8(uint8_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	ERR_FAIL_COND_MSG(!writing, "File has not been opened in write mode.");

	WRITE_FIT(1);
	write_ptr[write_pos++] = p_dest;
}

bool FileAccessCompressed::file_exists(const String &p_name) {
	FileAccess *fa = FileAccess::open(p_name, FileAccess::READ);
	if (!fa) {
		return false;
	}
	memdelete(fa);
	return true;
}

uint64_t FileAccessCompressed::_get_modified_time(const String &p_file) {
	if (f) {
		return f->get_modified_time(p_file);
	} else {
		return 0;
	}
}

uint32_t FileAccessCompressed::_get_unix_permissions(const String &p_file) {
	if (f) {
		return f->_get_unix_permissions(p_file);
	}
	return 0;
}

Error FileAccessCompressed::_set_unix_permissions(const String &p_file, uint32_t p_permissions) {
	if (f) {
		return f->_set_unix_permissions(p_file, p_permissions);
	}
	return FAILED;
}

FileAccessCompressed::~FileAccessCompressed() {
	if (f) {
		close();
	}
}
