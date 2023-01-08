#ifndef FILE_ACCESS_EPK_H
#define FILE_ACCESS_EPK_H

// #include "vfile_access.h"
// // #include "core/project_settings.h"
// #include "core/io/file_access_pack.h"

// #define EPK_HEADER_MAGIC 0x4745504bU // GEPK in ASCII
// #define EPK_FORMAT_VERSION 1
// #define EPK_COMB_LIMIT 32

// extern uint8_t script_encryption_key[32];

// class EpkArchive : public PackSource {
// private:
// 	struct EncryptedFile {
// 		int64_t package;
// 		uint32_t file_no;
// 		uint64_t offset;
// 		uint64_t size;
// 		EncryptedFile(){
// 			package = -1;
// 			offset = 0;
// 			size = 0;
// 		}
// 	};
// private:
// 	struct Package {
// 		String filename;
// 		List<List<uint32_t>> combination_list;
// 		Package() {}
// 		Package(const Package& other) {
// 			filename = other.filename;
// 			combination_list = other.combination_list;
// 		}
// 	};

// 	std::vector<Package> packages;
// 	HashMap<String, EncryptedFile> files;

// 	static Vector<double> static_offset;
// 	static Vector<uint8_t> master_key;
// 	static EpkArchive * singleton;

// 	List<uint32_t> comb_gen(const uint32_t& iter);
// 	uint32_t register_package(const String& p_path);
// public:
// 	virtual bool try_open_pack(const String &p_path, bool p_replace_files, uint64_t p_offset);
// 	FileAccess *get_file(const String &p_path, PackedData::PackedFile *p_file);

// 	static _FORCE_INLINE_ EpkArchive* get_singleton() {
// 		// if (!singleton){
// 		// 	singleton = memnew(EpkArchive);
// 		// }
// 		return singleton;
// 	}
// 	static _FORCE_INLINE_ void load_master_key(const uint8_t* key_arr) {
// 		ERR_FAIL_COND(master_key.size() != 0);
// 		master_key.resize(32);
// 		for (uint32_t i = 0; i < 32; i++){
// 			master_key[i] = key_arr[i];
// 		}
// 	}

// 	EpkArchive();
// 	~EpkArchive();
// };

// class EpkEncryptor {
// public:
// 	EPKEncryptor();
// 	~EPKEncryptor();
// };

// class FileAccessEPK : public FileAccess {
// public:

// 	FileAccessEPK(const String& p_path, PackedData::PackedFile *p_file, const Vector<uint8_t>& key);
// };

#endif