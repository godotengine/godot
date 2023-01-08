#include "file_access_epk.h"

// EpkArchive* EpkArchive::singleton = nullptr;
// Vector<uint8_t> EpkArchive::master_key;
// Vector<double> EpkArchive::static_offset;

// EpkArchive::EpkArchive(){
// 	ERR_FAIL_COND(singleton);
// 	singleton = this;
// 	static_offset.resize(32);
// 	uint8_t u8_limit = 0U;
// 	u8_limit -= 1;
// 	// Get a list of static file offset
// 	for (uint32_t i = 0; i < 32; i++){
// 		static_offset.write[i] = script_encryption_key[i] / double(u8_limit);
// 	}
// }
// EpkArchive::~EpkArchive(){

// }

// // Fragmented combination to throw people off their track
// List<uint32_t> EpkArchive::comb_gen(const uint32_t& iter){

// }

// uint32_t EpkArchive::register_package(const String& p_path){
// 	FileAccess *fa = FileAccess::open(p_path, FileAccess::READ);
// 	CRASH_COND_MSG(!fa, "Unable to open EPK file");
// 	auto total_len = fa->get_len();
// 	Package new_pck;
// 	// new_pck.combination_list.resize(EPK_COMB_LIMIT);
// 	// Craete combinations
// 	for (uint32_t i = 0; i < EPK_COMB_LIMIT; i++){
// 		// new_pck.combination_list.write[i] = comb_gen(i);
// 		new_pck.combination_list.push_back(comb_gen(i));
// 	}
// 	packages.push_back(new_pck);
// 	return packages.size() - 1;
// }

// bool EpkArchive::try_open_pack(const String &p_path, bool p_replace_files, uint64_t p_offset){
// 	FileAccess *fa = FileAccess::open(p_path, FileAccess::READ);
// 	fa->seek(0);
// 	// Check for file signature
// 	auto magic = fa->get_32();
// 	if (magic != EPK_HEADER_MAGIC){
// 		fa->close();
// 		memdelete(fa);
// 		return false;
// 	}
// 	// Check for file metadata
// 	// EOF contain offset followed by 4 empty byte
// 	fa->seek_end();
// 	fa->seek(fa->get_position() - 8 - 4);
// 	auto metadata_offset = fa->get_64();
// 	if (metadata_offset >= fa->get_len()){
// 		fa->close();
// 		memdelete(fa);
// 		return false;
// 	}
// 	fa->seek(metadata_offset);
// 	uint32_t version = f->get_32();
// 	uint32_t ver_major = f->get_32();
// 	uint32_t ver_minor = f->get_32();
// 	if (version != EPK_FORMAT_VERSION) {
// 		fa->close();
// 		memdelete(fa);
// 		ERR_FAIL_V_MSG(false, "Pack version unsupported: " + itos(version) + ".");
// 	}
// 	if (ver_major > VERSION_MAJOR || (ver_major == VERSION_MAJOR && ver_minor > VERSION_MINOR)) {
// 		fa->close();
// 		memdelete(fa);
// 		ERR_FAIL_V_MSG(false, "Pack created with a newer version of the engine: " + itos(ver_major) + "." + itos(ver_minor) + ".");
// 	}
// 	auto curr_epk_no = register_package(p_path);
// 	// Dunno why this region is reserved but whatever
// 	for (int i = 0; i < 16; i++) {
// 		//reserved
// 		f->get_32();
// 	}
// 	uint64_t file_count = fa->get_64();
// 	for (int i = 0; i < file_count; i++) {
// 		uint32_t filename_len = fa->get_32();
// 		CharString cs;
// 		cs.resize(filename_len + 1);
// 		fa->get_buffer(cs.ptrw(), filename_len);
// 		cs[filename_len] = 0;

// 		String path;
// 		path.parse_utf8(cs.ptr());
// 		uint64_t ofs = f->get_64();
// 		uint64_t size = f->get_64();
// 		uint8_t md5[16];
// 		f->get_buffer(md5, 16);

// 		PackedData::get_singleton()->add_path(p_path, path, ofs + p_offset, size, md5, this, p_replace_files);
// 	}
// 	fa->close();
// 	memdelete(fa);
// 	return true;
// }
// FileAccess *EpkArchive::get_file(const String &p_path, PackedData::PackedFile *p_file){
// 	return memnew(FileAccessEPK(p_path, p_file, master_key));
// }
// FileAccessEPK::FileAccessEPK(const String& p_path, PackedData::PackedFile *p_file, const Vector<uint8_t>& key){

// }