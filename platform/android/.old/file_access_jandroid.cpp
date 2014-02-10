#include "file_access_jandroid.h"
#include "os/os.h"
#include <unistd.h>

jobject FileAccessJAndroid::io=NULL;
jclass FileAccessJAndroid::cls;
jmethodID FileAccessJAndroid::_file_open=0;
jmethodID FileAccessJAndroid::_file_get_size=0;
jmethodID FileAccessJAndroid::_file_seek=0;
jmethodID FileAccessJAndroid::_file_read=0;
jmethodID FileAccessJAndroid::_file_tell=0;
jmethodID FileAccessJAndroid::_file_eof=0;
jmethodID FileAccessJAndroid::_file_close=0;


JNIEnv * FileAccessJAndroid::env=NULL;

void FileAccessJAndroid::make_default() {

	create_func=create_jandroid;
}

FileAccess* FileAccessJAndroid::create_jandroid() {

	return memnew(FileAccessJAndroid);
}

Error FileAccessJAndroid::open(const String& p_path, int p_mode_flags) {

	if (is_open())
		close();

	String path=fix_path(p_path).simplify_path();
	if (path.begins_with("/"))
		path=path.substr(1,path.length());
	else if (path.begins_with("res://"))
		path=path.substr(6,path.length());

	//OS::get_singleton()->print("env: %p, io %p, fo: %p\n",env,io,_file_open);

	jstring js = env->NewStringUTF(path.utf8().get_data());
	int res = env->CallIntMethod(io,_file_open,js,p_mode_flags&WRITE?true:false);

	if (res<=0)
		return ERR_FILE_CANT_OPEN;
	id=res;


	return OK;
}

void FileAccessJAndroid::close() {

	if (io==0)
		return;

	env->CallVoidMethod(io,_file_close,id);
	id=0;

}
bool FileAccessJAndroid::is_open() const {

	return id!=0;
}

void FileAccessJAndroid::seek(size_t p_position) {

	ERR_FAIL_COND(!is_open());
	env->CallVoidMethod(io,_file_seek,id,p_position);
}
void FileAccessJAndroid::seek_end(int64_t p_position) {

	ERR_FAIL_COND(!is_open());

	seek(get_len());

}
size_t FileAccessJAndroid::get_pos() const {

	ERR_FAIL_COND_V(!is_open(),0);
	return env->CallIntMethod(io,_file_tell,id);

}
size_t FileAccessJAndroid::get_len() const {

	ERR_FAIL_COND_V(!is_open(),0);
	return env->CallIntMethod(io,_file_get_size,id);


}

bool FileAccessJAndroid::eof_reached() const {

	ERR_FAIL_COND_V(!is_open(),0);
	return env->CallIntMethod(io,_file_eof,id);

}

uint8_t FileAccessJAndroid::get_8() const {

	ERR_FAIL_COND_V(!is_open(),0);
	uint8_t byte;
	get_buffer(&byte,1);
	return byte;
}
int FileAccessJAndroid::get_buffer(uint8_t *p_dst, int p_length) const {

	ERR_FAIL_COND_V(!is_open(),0);
	if (p_length==0)
		return 0;

	jbyteArray jca = (jbyteArray)env->CallObjectMethod(io,_file_read,id,p_length);

	int len = env->GetArrayLength(jca);
	env->GetByteArrayRegion(jca,0,len,(jbyte*)p_dst);
	env->DeleteLocalRef((jobject)jca);

	return len;

}

Error FileAccessJAndroid::get_error() const {

	if (eof_reached())
		return ERR_FILE_EOF;
	return OK;
}

void FileAccessJAndroid::store_8(uint8_t p_dest) {


}

bool FileAccessJAndroid::file_exists(const String& p_path) {

	jstring js = env->NewStringUTF(p_path.utf8().get_data());
	int res = env->CallIntMethod(io,_file_open,js,false);
	if (res<=0)
		return false;
	env->CallVoidMethod(io,_file_close,res);
	return true;

}


void FileAccessJAndroid::setup( JNIEnv *p_env, jobject p_io) {

	io=p_io;
	env=p_env;

	__android_log_print(ANDROID_LOG_INFO,"godot","STEP5");

	jclass c = env->GetObjectClass(io);
	__android_log_print(ANDROID_LOG_INFO,"godot","STEP6");
	cls=(jclass)env->NewGlobalRef(c);

	_file_open = env->GetMethodID(cls, "file_open", "(Ljava/lang/String;Z)I");	
	if(_file_open != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_open ok!!");
	}
	_file_get_size = env->GetMethodID(cls, "file_get_size", "(I)I");
	if(_file_get_size != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_get_size ok!!");
	}
	_file_tell = env->GetMethodID(cls, "file_tell", "(I)I");
	if(_file_tell != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_tell ok!!");
	}
	_file_eof = env->GetMethodID(cls, "file_eof", "(I)Z");

	if(_file_eof != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_eof ok!!");
	}
	_file_seek = env->GetMethodID(cls, "file_seek", "(II)V");
	if(_file_seek != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_seek ok!!");
	}
	_file_read = env->GetMethodID(cls, "file_read", "(II)[B");
	if(_file_read != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_read ok!!");
	}
	_file_close = env->GetMethodID(cls, "file_close", "(I)V");
	if(_file_close != 0) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*******GOT METHOD _file_close ok!!");
	}

//	(*env)->CallVoidMethod(env,obj,aMethodID, myvar);
}


FileAccessJAndroid::FileAccessJAndroid()
{

	id=0;
}

FileAccessJAndroid::~FileAccessJAndroid()
{

	if (is_open())
		close();
}


