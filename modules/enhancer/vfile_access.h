#ifndef VFILE_ACCESS
#define VFILE_ACCESS

#define USE_VECTOR_BLOCK
#ifdef USE_VECTOR_BLOCK
#include <vector>
#endif

#include "core/project_settings.h"
#include "core/os/file_access.h"
#include "core/print_string.h"
#include "core/pool_vector.h"
#include "core/ustring.h"
#include "core/hash_map.h"
// Sharable virtual file
// No COW
class VFile {
private:
	struct _Data {
#ifdef USE_VECTOR_BLOCK
		std::vector<uint8_t> block{};
#else
		PoolVector<uint8_t> block{};
#endif
		mutable SafeRefCount refcount{};
		String file_path{};
	};
	_Data *data;
	void _unref();
	void _ref(const _Data* other);
	friend class VFileAccess;
	friend class MemFS;
public:
	void reference(const VFile& other);

	VFile& operator=(const VFile& other);

	const uint8_t * read() const{
		if (!data) return nullptr;
#ifdef USE_VECTOR_BLOCK
		return data->block.data();
#else
		return data->block.read().ptr();
#endif
	}
	uint8_t* write(){
		if (!data) return nullptr;
#ifdef USE_VECTOR_BLOCK
		return data->block.data();
#else
		return data->block.write().ptr();
#endif
	}

	bool is_corrupted() const {
		if (!data) return true;
		return data->file_path.empty();
	}

	VFile();
	VFile(const VFile& other);
	~VFile();
};

class MemFS {
private:
	static HashMap<String, VFile> memfs;
public:
	static VFile get_file(const String& f_path);
	static VFile register_file(const String& f_path);
	static void virtualize_file(const String& f_path, FileAccess* fa);
	static void delete_file(const String& f_path);
	static void delete_all_file();

	static void init_memfs();
	static void free_memfs();
	static bool file_exists(const String& f_path);
	static _FORCE_INLINE_ bool is_initialized() { return true; }
};

class VFileAccess : public FileAccess {
private:
	mutable uint64_t pos;

	bool reading = false, writting = false;

	ModeFlags mode;
	VFile my_file;

	_FORCE_INLINE_ const uint8_t* ptr() const {
		// ERR_FAIL_COND_V(!is_open(), nullptr);
		return my_file.read();
	}
	_FORCE_INLINE_ uint8_t* ptrw() {
		// ERR_FAIL_COND_V(!is_open(), nullptr);
		return my_file.write();
	}
public:
	_FORCE_INLINE_ static VFileAccess* instantiate() {
		return memnew(VFileAccess());
	}
	_FORCE_INLINE_ static VFileAccess* free(const VFileAccess *vfa){
		memdelete(const_cast<VFileAccess*>(vfa));
		return nullptr;
	}
	_FORCE_INLINE_ void resize_file(const uint64_t& amount){
		ERR_FAIL_COND(!is_open());
		my_file.data->block.resize(amount);
	}
	_FORCE_INLINE_ void fill_with(const uint8_t& byte = 0){
		ERR_FAIL_COND(!is_open());
#ifdef USE_VECTOR_BLOCK
		std::fill(my_file.data->block.begin(), my_file.data->block.end(), byte);
#else
		my_file.data->block.fill(byte);
#endif
	}

	static _FORCE_INLINE_ VFileAccess* open_path(const String& p_path, int p_mode_flags){
		auto ret = instantiate();
		ret->_open(p_path, p_mode_flags);
		if (!ret->is_open()) ret = VFileAccess::free(ret);
		return ret;
	}

	virtual String get_path() const;
	virtual String get_path_absolute() const;

	virtual Error _open(const String &p_path, int p_mode_flags);
	virtual void close();
	virtual bool is_open() const;

	virtual void seek(uint64_t p_position);
	virtual void seek_end(int64_t p_position);
	virtual uint64_t get_position() const;
	virtual _FORCE_INLINE_ uint64_t get_len() const {
		return my_file.data->block.size();
	}

	virtual bool eof_reached() const;

	virtual _FORCE_INLINE_ uint8_t get_8() const {
		ERR_FAIL_COND_V(!reading, 0);
		uint8_t retval = 0;
		if (pos < get_len()){
			retval = ptr()[pos];
		}
		++pos;
		return retval;
	}

	virtual _FORCE_INLINE_ uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const{
		ERR_FAIL_COND_V(!reading, 0);
		ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);

		uint64_t left = get_len() - pos;
		uint64_t read = MIN(p_length, left);

		if (read < p_length) {
			WARN_PRINT("Reading less data than requested");
		}

		memcpy(p_dst, &(ptr()[pos]), read);
		pos += p_length;

		return read;
	}

	virtual Error get_error() const { return eof_reached() ? ERR_FILE_EOF : OK; }

	virtual _FORCE_INLINE_ void store_8(uint8_t p_byte) {
		ERR_FAIL_COND(!writting);
		if (pos < get_len()){
			ptrw()[pos++] = p_byte;
		} else {
			my_file.data->block.push_back(p_byte);
			pos = get_len();
		}
	}
	virtual _FORCE_INLINE_ void store_buffer(const uint8_t *p_src, uint64_t p_length) {
		ERR_FAIL_COND(!writting);
		ERR_FAIL_COND(!p_src && p_length > 0);
		auto left = get_len() - pos;
		auto write = MIN(p_length, left);
		auto allocate = p_length - write;
		if (allocate >= 0) {
			my_file.data->block.resize(get_len() + allocate);
		}
		for (uint64_t i = 0; i < p_length; i++){
			store_8(p_src[i]);
		}
		// memcpy(ptrw(), p_src, p_length);
		// pos += p_length;
	}

	virtual void flush() {}

	virtual _FORCE_INLINE_ bool file_exists(const String &p_name) { return MemFS::file_exists(p_name); }

	virtual uint64_t _get_modified_time(const String &p_file) { return 0; }
	virtual uint32_t _get_unix_permissions(const String &p_file) { return 0; }
	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions) { return FAILED; }

	VFileAccess();
	~VFileAccess();
};

#endif