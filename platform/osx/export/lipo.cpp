/*************************************************************************/
/*  lipo.cpp                                                             */
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

#include "modules/modules_enabled.gen.h" // For regex.

#include "lipo.h"

#ifdef MODULE_REGEX_ENABLED

bool LipO::is_lipo(const String &p_path) {
	FileAccessRef fb = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!fb, false, vformat("LipO: Can't open file: \"%s\".", p_path));
	uint32_t magic = fb->get_32();
	return (magic == 0xbebafeca || magic == 0xcafebabe || magic == 0xbfbafeca || magic == 0xcafebabf);
}

bool LipO::create_file(const String &p_output_path, const PackedStringArray &p_files) {
	close();

	fa = FileAccess::open(p_output_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!fa, false, vformat("LipO: Can't open file: \"%s\".", p_output_path));

	uint64_t max_size = 0;
	for (int i = 0; i < p_files.size(); i++) {
		MachO mh;
		if (!mh.open_file(p_files[i])) {
			ERR_FAIL_V_MSG(false, vformat("LipO: Invalid MachO file: \"%s.\"", p_files[i]));
		}

		FatArch arch;
		arch.cputype = mh.get_cputype();
		arch.cpusubtype = mh.get_cpusubtype();
		arch.offset = 0;
		arch.size = mh.get_size();
		arch.align = mh.get_align();
		max_size += arch.size;

		archs.push_back(arch);

		FileAccessRef fb = FileAccess::open(p_files[i], FileAccess::READ);
		if (!fb) {
			close();
			ERR_FAIL_V_MSG(false, vformat("LipO: Can't open file: \"%s.\"", p_files[i]));
		}
	}

	// Write header.
	bool is_64 = (max_size >= std::numeric_limits<uint32_t>::max());
	if (is_64) {
		fa->store_32(0xbfbafeca);
	} else {
		fa->store_32(0xbebafeca);
	}
	fa->store_32(BSWAP32(archs.size()));
	uint64_t offset = archs.size() * (is_64 ? 32 : 20) + 8;
	for (int i = 0; i < archs.size(); i++) {
		archs.write[i].offset = offset + PAD(offset, uint64_t(1) << archs[i].align);
		if (is_64) {
			fa->store_32(BSWAP32(archs[i].cputype));
			fa->store_32(BSWAP32(archs[i].cpusubtype));
			fa->store_64(BSWAP64(archs[i].offset));
			fa->store_64(BSWAP64(archs[i].size));
			fa->store_32(BSWAP32(archs[i].align));
			fa->store_32(0);
		} else {
			fa->store_32(BSWAP32(archs[i].cputype));
			fa->store_32(BSWAP32(archs[i].cpusubtype));
			fa->store_32(BSWAP32(archs[i].offset));
			fa->store_32(BSWAP32(archs[i].size));
			fa->store_32(BSWAP32(archs[i].align));
		}
		offset = archs[i].offset + archs[i].size;
	}

	// Write files and padding.
	for (int i = 0; i < archs.size(); i++) {
		FileAccessRef fb = FileAccess::open(p_files[i], FileAccess::READ);
		if (!fb) {
			close();
			ERR_FAIL_V_MSG(false, vformat("LipO: Can't open file: \"%s.\"", p_files[i]));
		}
		uint64_t cur = fa->get_position();
		for (uint64_t j = cur; j < archs[i].offset; j++) {
			fa->store_8(0);
		}
		int pages = archs[i].size / 4096;
		int remain = archs[i].size % 4096;
		unsigned char step[4096];
		for (int j = 0; j < pages; j++) {
			uint64_t br = fb->get_buffer(step, 4096);
			if (br > 0) {
				fa->store_buffer(step, br);
			}
		}
		uint64_t br = fb->get_buffer(step, remain);
		if (br > 0) {
			fa->store_buffer(step, br);
		}
		fb->close();
	}
	return true;
}

bool LipO::open_file(const String &p_path) {
	close();

	fa = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!fa, false, vformat("LipO: Can't open file: \"%s\".", p_path));

	uint32_t magic = fa->get_32();
	if (magic == 0xbebafeca) {
		// 32-bit fat binary, bswap.
		uint32_t nfat_arch = BSWAP32(fa->get_32());
		for (uint32_t i = 0; i < nfat_arch; i++) {
			FatArch arch;
			arch.cputype = BSWAP32(fa->get_32());
			arch.cpusubtype = BSWAP32(fa->get_32());
			arch.offset = BSWAP32(fa->get_32());
			arch.size = BSWAP32(fa->get_32());
			arch.align = BSWAP32(fa->get_32());

			archs.push_back(arch);
		}
	} else if (magic == 0xcafebabe) {
		// 32-bit fat binary.
		uint32_t nfat_arch = fa->get_32();
		for (uint32_t i = 0; i < nfat_arch; i++) {
			FatArch arch;
			arch.cputype = fa->get_32();
			arch.cpusubtype = fa->get_32();
			arch.offset = fa->get_32();
			arch.size = fa->get_32();
			arch.align = fa->get_32();

			archs.push_back(arch);
		}
	} else if (magic == 0xbfbafeca) {
		// 64-bit fat binary, bswap.
		uint32_t nfat_arch = BSWAP32(fa->get_32());
		for (uint32_t i = 0; i < nfat_arch; i++) {
			FatArch arch;
			arch.cputype = BSWAP32(fa->get_32());
			arch.cpusubtype = BSWAP32(fa->get_32());
			arch.offset = BSWAP64(fa->get_64());
			arch.size = BSWAP64(fa->get_64());
			arch.align = BSWAP32(fa->get_32());
			fa->get_32(); // Skip, reserved.

			archs.push_back(arch);
		}
	} else if (magic == 0xcafebabf) {
		// 64-bit fat binary.
		uint32_t nfat_arch = fa->get_32();
		for (uint32_t i = 0; i < nfat_arch; i++) {
			FatArch arch;
			arch.cputype = fa->get_32();
			arch.cpusubtype = fa->get_32();
			arch.offset = fa->get_64();
			arch.size = fa->get_64();
			arch.align = fa->get_32();
			fa->get_32(); // Skip, reserved.

			archs.push_back(arch);
		}
	} else {
		close();
		ERR_FAIL_V_MSG(false, vformat("LipO: Invalid fat binary: \"%s\".", p_path));
	}
	return true;
}

int LipO::get_arch_count() const {
	ERR_FAIL_COND_V_MSG(!fa, 0, "LipO: File not opened.");
	return archs.size();
}

bool LipO::extract_arch(int p_index, const String &p_path) {
	ERR_FAIL_COND_V_MSG(!fa, false, "LipO: File not opened.");
	ERR_FAIL_INDEX_V(p_index, archs.size(), false);

	FileAccessRef fb = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!fb, false, vformat("LipO: Can't open file: \"%s\".", p_path));

	fa->seek(archs[p_index].offset);

	int pages = archs[p_index].size / 4096;
	int remain = archs[p_index].size % 4096;
	unsigned char step[4096];
	for (int i = 0; i < pages; i++) {
		uint64_t br = fa->get_buffer(step, 4096);
		if (br > 0) {
			fb->store_buffer(step, br);
		}
	}
	uint64_t br = fa->get_buffer(step, remain);
	if (br > 0) {
		fb->store_buffer(step, br);
	}
	fb->close();
	return true;
}

void LipO::close() {
	if (fa) {
		fa->close();
		memdelete(fa);
		fa = nullptr;
	}
	archs.clear();
}

LipO::~LipO() {
	close();
}

#endif // MODULE_REGEX_ENABLED
