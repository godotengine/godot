#include "file_access_encrypted.h"
#include "aes256.h"
#include "md5.h"
#include "os/copymem.h"
#define COMP_MAGIC 0x43454447


Error FileAccessEncrypted::open_and_parse(FileAccess *p_base,const Vector<uint8_t>& p_key,Mode p_mode) {

	ERR_FAIL_COND_V(file!=NULL,ERR_ALREADY_IN_USE);

	if (p_mode==MODE_WRITE_AES256) {

		ERR_FAIL_COND_V(p_key.size()!=32,ERR_INVALID_PARAMETER);
		data.clear();
		writing=true;
		file=p_base;
		mode=p_mode;
		key=p_key;

	}

	return OK;
}

Error FileAccessEncrypted::open_and_parse_password(FileAccess *p_base,const String& p_key,Mode p_mode){


	String cs = p_key.md5_text();
	ERR_FAIL_COND_V(cs.length()!=32,ERR_INVALID_PARAMETER);
	Vector<uint8_t> key;
	key.resize(32);
	for(int i=0;i<32;i++) {

		key[i]=cs[i];
	}

	return open_and_parse(p_base,key,p_mode);
}



Error FileAccessEncrypted::_open(const String& p_path, int p_mode_flags) {

	return OK;
}
void FileAccessEncrypted::close() {

	if (!file)
		return;

	if (writing) {

		Vector<uint8_t> compressed;
		size_t len = data.size();
		if (len % 16) {
			len+=16-(len % 16);
		}

		compressed.resize(len);
		zeromem( compressed.ptr(), len );
		for(int i=0;i<data.size();i++) {
			compressed[i]=data[i];
		}

		aes256_context ctx;
		aes256_init(&ctx,key.ptr());

		for(size_t i=0;i<len;i+=16) {

			aes256_encrypt_ecb(&ctx,&compressed[i]);
		}

		aes256_done(&ctx);

		file->store_32(COMP_MAGIC);
		file->store_32(mode);

		MD5_CTX md5;
		MD5Init(&md5);
		MD5Update(&md5,compressed.ptr(),compressed.size());
		MD5Final(&md5);

		file->store_buffer(md5.digest,16);
		file->store_64(data.size());

		file->store_buffer(compressed.ptr(),compressed.size());
		file->close();
		memdelete(file);
		file=NULL;

	}

}

bool FileAccessEncrypted::is_open() const{

	return file!=NULL;
}

void FileAccessEncrypted::seek(size_t p_position){

	if (writing) {
		if (p_position > (size_t)data.size())
			p_position=data.size();

		pos=p_position;
	}
}


void FileAccessEncrypted::seek_end(int64_t p_position){

	seek( data.size() + p_position );
}
size_t FileAccessEncrypted::get_pos() const{

	return pos;
	return 0;
}
size_t FileAccessEncrypted::get_len() const{

	if (writing)
		return data.size();
	return 0;
}

bool FileAccessEncrypted::eof_reached() const{

	if (!writing) {


	}

	return false;
}

uint8_t FileAccessEncrypted::get_8() const{

	return 0;
}
int FileAccessEncrypted::get_buffer(uint8_t *p_dst, int p_length) const{


	return 0;
}

Error FileAccessEncrypted::get_error() const{

	return OK;
}

void FileAccessEncrypted::store_buffer(const uint8_t *p_src,int p_length) {

	ERR_FAIL_COND(!writing);

	if (pos<data.size()) {

		for(int i=0;i<p_length;i++) {

			store_8(p_src[i]);
		}
	} else if (pos==data.size()) {

		data.resize(pos+p_length);
		for(int i=0;i<p_length;i++) {

			data[pos+i]=p_src[i];
		}
		pos+=p_length;
	}
}


void FileAccessEncrypted::store_8(uint8_t p_dest){

	ERR_FAIL_COND(!writing);

	if (pos<data.size()) {
		data[pos]=p_dest;
		pos++;
	} else if (pos==data.size()){
		data.push_back(p_dest);
		pos++;
	}
}

bool FileAccessEncrypted::file_exists(const String& p_name){

	FileAccess *fa = FileAccess::open(p_name,FileAccess::READ);
	if (!fa)
		return false;
	memdelete(fa);
	return true;
}

uint64_t FileAccessEncrypted::_get_modified_time(const String& p_file){


	return 0;
}

FileAccessEncrypted::FileAccessEncrypted() {

	file=NULL;
}


FileAccessEncrypted::~FileAccessEncrypted() {

	if (file)
		close();

	if (file) {
		memdelete(file);
	}
}
