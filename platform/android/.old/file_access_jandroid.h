#ifndef FILE_ACCESS_JANDROID_H
#define FILE_ACCESS_JANDROID_H

#include "java_glue.h"
#include "os/file_access.h"
class FileAccessJAndroid : public FileAccess {

	static jobject io;
	static jclass cls;

	static jmethodID _file_open;
	static jmethodID _file_get_size;
	static jmethodID _file_seek;
	static jmethodID _file_tell;
	static jmethodID _file_eof;
	static jmethodID _file_read;
	static jmethodID _file_close;


	static JNIEnv * env;

	int id;
	static FileAccess* create_jandroid();


public:

	virtual Error open(const String& p_path, int p_mode_flags); ///< open a file
	virtual void close(); ///< close a file
	virtual bool is_open() const; ///< true when file is open

	virtual void seek(size_t p_position); ///< seek to a given position
	virtual void seek_end(int64_t p_position=0); ///< seek from the end of file
	virtual size_t get_pos() const; ///< get position in the file
	virtual size_t get_len() const; ///< get size of the file

	virtual bool eof_reached() const; ///< reading passed EOF

	virtual uint8_t get_8() const; ///< get a byte
	virtual int get_buffer(uint8_t *p_dst, int p_length) const;

	virtual Error get_error() const; ///< get last error

	virtual void store_8(uint8_t p_dest); ///< store a byte

	virtual bool file_exists(const String& p_path); ///< return true if a file exists

	static void make_default();

	static void setup( JNIEnv *env, jobject io);
	static void update_env( JNIEnv *p_env) { env=p_env; }
	FileAccessJAndroid();
	~FileAccessJAndroid();
};

#endif // FILE_ACCESS_JANDROID_H
