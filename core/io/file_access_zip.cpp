/*************************************************************************/
/*  file_access_zip.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifdef MINIZIP_ENABLED

#include "file_access_zip.h"
#include "core/os/os.h"
#include "core/os/file_access.h"
#include "core/os/copymem.h"

ZipArchive* ZipArchive::instance = NULL;

extern "C" {

static void* godot_open(void* data, const char* p_fname, int mode) {

	if (mode & ZLIB_FILEFUNC_MODE_WRITE) {
		return NULL;
	};

	FileAccess* f = (FileAccess*)data;
	if(!f->is_open())
		f->open(p_fname, FileAccess::READ);

	return f->is_open()?data:NULL;

};

static uLong godot_read(void* data, void* fdata, void* buf, uLong size) {

	FileAccess* f = (FileAccess*)data;
	f->get_buffer((uint8_t*)buf, size);
	return size;
};

static uLong godot_write(voidpf opaque, voidpf stream, const void* buf, uLong size) {

	return 0;
};


static long godot_tell (voidpf opaque, voidpf stream) {

	FileAccess* f = (FileAccess*)opaque;
	return f->get_pos();
};

static long godot_seek(voidpf opaque, voidpf stream, uLong offset, int origin) {

	FileAccess* f = (FileAccess*)opaque;

	int pos = offset;
	switch (origin) {

	case ZLIB_FILEFUNC_SEEK_CUR:
		pos = f->get_pos() + offset;
		break;
	case ZLIB_FILEFUNC_SEEK_END:
		pos = f->get_len() + offset;
		break;
	default:
		break;
	};

	f->seek(pos);
	return 0;
};


static int godot_close(voidpf opaque, voidpf stream) {

	FileAccess* f = (FileAccess*)opaque;
	f->close();
	return 0;
};

static int godot_testerror(voidpf opaque, voidpf stream) {

	FileAccess* f = (FileAccess*)opaque;
	return f->get_error()!=OK?1:0;
};

static voidpf godot_alloc(voidpf opaque, uInt items, uInt size) {

	return memalloc(items * size);
};

static void godot_free(voidpf opaque, voidpf address) {

	memfree(address);
};

}; // extern "C"

void ZipArchive::close_handle(unzFile p_file) const {

	ERR_FAIL_COND(!p_file);
	FileAccess* f = (FileAccess*)unzGetOpaque(p_file);
	unzCloseCurrentFile(p_file);
	unzClose(p_file);
	memdelete(f);
};

unzFile ZipArchive::get_file_handle(String p_file) const {

	ERR_FAIL_COND_V(!file_exists(p_file), NULL);
	File file = files[p_file];

	FileAccess* f = FileAccess::open(packages[file.package].filename, FileAccess::READ);
	ERR_FAIL_COND_V(!f, NULL);

	zlib_filefunc_def io;
	zeromem(&io, sizeof(io));

	io.opaque = f;
	io.zopen_file = godot_open;
	io.zread_file = godot_read;
	io.zwrite_file = godot_write;

	io.ztell_file = godot_tell;
	io.zseek_file = godot_seek;
	io.zclose_file = godot_close;
	io.zerror_file = godot_testerror;

	io.alloc_mem = godot_alloc;
	io.free_mem = godot_free;

	unzFile pkg = unzOpen2(packages[file.package].filename.utf8().get_data(), &io);
	ERR_FAIL_COND_V(!pkg, NULL);
	int unz_err = unzGoToFilePos(pkg, &file.file_pos);
	ERR_FAIL_COND_V(unz_err != UNZ_OK, NULL);
	if (unzOpenCurrentFile(pkg) != UNZ_OK) {

		unzClose(pkg);
		ERR_FAIL_V(NULL);
	};

	return pkg;
};

bool ZipArchive::try_open_pack(const String& p_name) {

	//printf("opening zip pack %ls, %i, %i\n", p_name.c_str(), p_name.extension().nocasecmp_to("zip"), p_name.extension().nocasecmp_to("pcz"));
	if (p_name.get_extension().nocasecmp_to("zip") != 0 && p_name.get_extension().nocasecmp_to("pcz") != 0)
		return false;

	zlib_filefunc_def io;

	FileAccess* f = FileAccess::open(p_name, FileAccess::READ);
	if (!f)
		return false;
	io.opaque = f;
	io.zopen_file = godot_open;
	io.zread_file = godot_read;
	io.zwrite_file = godot_write;

	io.ztell_file = godot_tell;
	io.zseek_file = godot_seek;
	io.zclose_file = godot_close;
	io.zerror_file = godot_testerror;

	unzFile zfile = unzOpen2(p_name.utf8().get_data(), &io);
	ERR_FAIL_COND_V(!zfile, false);

	unz_global_info64 gi;
	int err = unzGetGlobalInfo64(zfile, &gi);
	ERR_FAIL_COND_V(err!=UNZ_OK, false);

	Package pkg;
	pkg.filename = p_name;
	pkg.zfile = zfile;
	packages.push_back(pkg);
	int pkg_num = packages.size()-1;

	if (OS::get_singleton()->is_stdout_verbose())
		print_line("Total files in pack: " + gi.number_entry);

	for (unsigned int i=0;i<gi.number_entry;i++) {

		char filename_inzip[256];

		unz_file_info64 file_info;
		err = unzGetCurrentFileInfo64(zfile,&file_info,filename_inzip,sizeof(filename_inzip),NULL,0,NULL,0);
		ERR_CONTINUE(err != UNZ_OK);

		File f;
		f.package = pkg_num;
		unzGetFilePos(zfile, &f.file_pos);

		String fname = String("res://") + filename_inzip;
		files[fname] = f;

		uint8_t md5[16]={0,0,0,0,0,0,0,0 , 0,0,0,0,0,0,0,0};
		PackedData::get_singleton()->add_path(p_name, fname, file_info.crc, file_info.uncompressed_size, md5, this);

		if (OS::get_singleton()->is_stdout_verbose())
			print_line(" >> " + p_name + " : " + fname);

		if ((i+1)<gi.number_entry) {
			unzGoToNextFile(zfile);
		};
	};

	return true;
};

bool ZipArchive::file_exists(String p_name) const {

	return files.has(p_name);
};

FileAccess* ZipArchive::get_file(const String& p_path, PackedData::PackedFile* p_file) {

	return memnew(FileAccessZip(p_path, *p_file));
};


ZipArchive* ZipArchive::get_singleton() {

	if (instance == NULL) {
		instance = memnew(ZipArchive);
	};

	return instance;
};

ZipArchive::ZipArchive() {

	instance = this;
	//fa_create_func = FileAccess::get_create_func();
};

ZipArchive::~ZipArchive() {

	for (int i=0; i<packages.size(); i++) {

		FileAccess* f = (FileAccess*)unzGetOpaque(packages[i].zfile);
		unzClose(packages[i].zfile);
		memdelete(f);
	};

	packages.clear();
};


Error FileAccessZip::_open(const String& p_path, int p_mode_flags) {

	close();

	ERR_FAIL_COND_V(p_mode_flags & FileAccess::WRITE, FAILED);
	const ZipArchive* arch = archive; //ZipArchive::get_singleton();
	ERR_FAIL_COND_V(!arch, FAILED);
	zfile = arch->get_file_handle(p_path);
	ERR_FAIL_COND_V(!zfile, FAILED);

	int err = unzGetCurrentFileInfo64(zfile,&file_info,NULL,0,NULL,0,NULL,0);
	ERR_FAIL_COND_V(err != UNZ_OK, FAILED);
	at_eof = false;

	return OK;
};

void FileAccessZip::close() {

	if(mem != NULL) {
		memdelete(mem);
		mem = NULL;
	}

	if (!zfile)
		return;

	const ZipArchive* arch = archive;//ZipArchive::get_singleton();
	ERR_FAIL_COND(!arch);
	arch->close_handle(zfile);
	zfile = NULL;
};

bool FileAccessZip::is_open() const {

	return zfile != NULL;
};

void FileAccessZip::seek(size_t p_position) {

	if(mem != NULL) return mem->seek(p_position);

	// load zipped file into file_access_memory
	ERR_FAIL_COND(!zfile);
	FileAccessMemory *f = memnew(FileAccessMemory);
	unzSeekCurrentFile(zfile, 0);
	data.resize(file_info.uncompressed_size);
	size_t len = get_buffer(&data[0], data.size());
	// close zipped file
	close();
	if(len != data.size()) {
		WARN_PRINT("get_buffer less data than requested");
	}
	f->open_custom(&data[0], len);
	f->seek(p_position);
	this->mem = f;
};

void FileAccessZip::seek_end(int64_t p_position) {

	if(mem) mem->seek_end(p_position);
	seek(get_len() + p_position);
};

size_t FileAccessZip::get_pos() const {

	if(mem) return mem->get_pos();
	ERR_FAIL_COND_V(!zfile, 0);
	return unztell(zfile);
};

size_t FileAccessZip::get_len() const {

	if(mem) return mem->get_len();
	ERR_FAIL_COND_V(!zfile, 0);
	return file_info.uncompressed_size;
};

bool FileAccessZip::eof_reached() const {

	if(mem) return mem->eof_reached();
	ERR_FAIL_COND_V(!zfile, true);
	return at_eof;
};

uint8_t FileAccessZip::get_8() const {

	if(mem) return mem->get_8();
	uint8_t ret = 0;
	get_buffer(&ret, 1);
	return ret;
};

int FileAccessZip::get_buffer(uint8_t *p_dst,int p_length) const {

	if(mem) return mem->get_buffer(p_dst, p_length);
	ERR_FAIL_COND_V(!zfile, -1);
	at_eof = unzeof(zfile);
	if (at_eof)
		return 0;
	int read = unzReadCurrentFile(zfile, p_dst, p_length);
	ERR_FAIL_COND_V(read < 0, read);
	if (read < p_length)
		at_eof = true;
	return read;
};

Error FileAccessZip::get_error() const {

	if(mem) return mem->get_error();
	if (!zfile) {

		return ERR_UNCONFIGURED;
	};
	if (eof_reached()) {
		return ERR_FILE_EOF;
	};

	return OK;
};

void FileAccessZip::store_8(uint8_t p_dest) {

	ERR_FAIL();
};

bool FileAccessZip::file_exists(const String& p_name) {

	return false;
};


FileAccessZip::FileAccessZip(const String& p_path, const PackedData::PackedFile& p_file) {

	zfile = NULL;
	at_eof = false;
	mem = NULL;
	archive=dynamic_cast<ZipArchive *>(p_file.src);
	ERR_FAIL_COND(archive==NULL);
	_open(p_path, FileAccess::READ);
};

FileAccessZip::~FileAccessZip() {

	close();
};

#endif
