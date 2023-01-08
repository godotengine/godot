#include "vfile_access.h"

HashMap<String, VFile> MemFS::memfs;

void VFile::_unref(){
	if (!data) return;
	if (data->refcount.unref()){
		delete data; data = nullptr;
	}
}
void VFile::_ref(const _Data* other){
	if (other == data) return;
	_unref();
	data = nullptr;
	if (!other) return;
	if (other->refcount.ref()) data = const_cast<_Data*>(other);
}

VFile::VFile(){
	data = new _Data();
	data->refcount.init();
}

VFile::VFile(const VFile& other){
	data = nullptr;
	reference(other);
}

VFile::~VFile(){
	// _unref();
	_ref(nullptr);
}

VFile& VFile::operator=(const VFile& other){
	reference(other);
	return *this;
}

void VFile::reference(const VFile& other){
	_ref(other.data);
}

VFile MemFS::register_file(const String& f_path){
	// ERR_FAIL_COND_V_MSG(!memfs, (VFile()), "MemFS has not been initialized");
	String global_path = f_path;
	if (ProjectSettings::get_singleton()){
		global_path = ProjectSettings::get_singleton()->globalize_path(f_path);
	}
	VFile new_file{};
	new_file.data->file_path = global_path;
	(memfs)[global_path] = new_file;
	// return (*memfs)[global_path];
	return new_file;
}
void MemFS::virtualize_file(const String& f_path, FileAccess* fa){
	// ERR_FAIL_COND_MSG(!memfs, "MemFS has not been initialized");
	ERR_FAIL_COND(!fa || !fa->is_open());
	auto vfa = new VFileAccess();
	vfa->_open(f_path, FileAccess::WRITE);
	ERR_FAIL_COND(!vfa->is_open());
	fa->seek(0);
	auto total_len = fa->get_len();
	uint8_t * buffer = (uint8_t*)malloc(total_len);
	fa->get_buffer(buffer, total_len);
	vfa->seek(0);
	vfa->store_buffer(buffer, total_len);
	free(buffer);
	vfa->close();
	delete vfa;
	fa->seek(0);
}
void MemFS::delete_file(const String& f_path){
	// ERR_FAIL_COND_MSG(!memfs, "MemFS has not been initialized");
	memfs.erase(f_path);
}
void MemFS::delete_all_file(){
	// ERR_FAIL_COND_MSG(!memfs, "MemFS has not been initialized");
	memfs.clear();
}
void MemFS::init_memfs(){
	// ERR_FAIL_COND_MSG(memfs, "MemFS has already been initialized");
	// memfs = new HashMap<String, VFile>();
	ERR_FAIL();
}
void MemFS::free_memfs(){
	// ERR_FAIL_COND_MSG(!memfs, "MemFS has already been freed");
	// delete memfs;
	ERR_FAIL();
}
VFile MemFS::get_file(const String& f_path){
	// ERR_FAIL_COND_V_MSG(!memfs, (VFile()), "MemFS has not been initialized");
	String global_path = f_path;
	if (ProjectSettings::get_singleton()){
		global_path = ProjectSettings::get_singleton()->globalize_path(f_path);
	}
	return (memfs)[global_path];
}
bool MemFS::file_exists(const String& f_path){
	// ERR_FAIL_COND_V(!memfs, false);
	String global_path = f_path;
	if (ProjectSettings::get_singleton()){
		global_path = ProjectSettings::get_singleton()->globalize_path(f_path);
	}
	return memfs.has(global_path);
}

VFileAccess::VFileAccess(){
	pos = 0;
	mode = READ;
}
VFileAccess::~VFileAccess(){
	close();
}

String VFileAccess::get_path() const{
	ERR_FAIL_COND_V(!is_open(), String());
	return my_file.data->file_path;
}
String VFileAccess::get_path_absolute() const{
	ERR_FAIL_COND_V(!is_open(), String());
	return my_file.data->file_path;
}

Error VFileAccess::_open(const String &p_path, int p_mode_flags){
	ERR_FAIL_COND_V(!MemFS::is_initialized(), ERR_CANT_ACQUIRE_RESOURCE);
	ERR_FAIL_COND_V(is_open(), ERR_ALREADY_IN_USE);

	switch (p_mode_flags) {
		case READ: {
			ERR_FAIL_COND_V(!MemFS::file_exists(p_path), ERR_CANT_OPEN);
			my_file = MemFS::get_file(p_path);
			reading = true;
			break;
		}
		case READ_WRITE: reading = true;
		case WRITE: {
			writting = true;
			my_file = MemFS::register_file(p_path);
			break;
		}
		default: return ERR_UNAVAILABLE;
	}
	mode = (ModeFlags)p_mode_flags;
	return OK;
}

void VFileAccess::close(){
	flush();
	my_file = VFile();
	pos = 0;
	reading = false;
	writting = false;
}

bool VFileAccess::is_open() const {
	return !my_file.is_corrupted();
}

void VFileAccess::seek(uint64_t p_position){
	pos = p_position;
}
void VFileAccess::seek_end(int64_t p_position){
	pos = get_len() + p_position;
}
// uint64_t VFileAccess::get_len() const{
// 	return my_file.data->block.size();
// }
uint64_t VFileAccess::get_position() const{
	return pos;
}

bool VFileAccess::eof_reached() const {
	// IMPORTANT
	return pos > get_len();
}