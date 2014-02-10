#ifndef DIR_ACCESS_JANDROID_H
#define DIR_ACCESS_JANDROID_H

#include "java_glue.h"
#include "os/dir_access.h"
#include <stdio.h>


class DirAccessJAndroid  : public DirAccess {

	//AAssetDir* aad;

	static jobject io;
	static jclass cls;

	static jmethodID _dir_open;
	static jmethodID _dir_next;
	static jmethodID _dir_close;

	static JNIEnv * env;

	int id;

	String current_dir;
	String current;

	static DirAccess *create_fs();

public:

	virtual bool list_dir_begin(); ///< This starts dir listing
	virtual String get_next();
	virtual bool current_is_dir() const;
	virtual void list_dir_end(); ///<

	virtual int get_drive_count();
	virtual String get_drive(int p_drive);

	virtual Error change_dir(String p_dir); ///< can be relative or absolute, return false on success
	virtual String get_current_dir(); ///< return current dir location
	virtual String get_dir_separator() const ;

	virtual bool file_exists(String p_file);
	virtual uint64_t get_modified_time(String p_file);

	virtual Error make_dir(String p_dir);

	virtual Error rename(String p_from, String p_to);
	virtual Error remove(String p_name);

	//virtual FileType get_file_type() const;
	size_t get_space_left();

	static void make_default();

	static void setup( JNIEnv *env, jobject io);
	static void update_env( JNIEnv *p_env) { env=p_env; }

	DirAccessJAndroid();
	~DirAccessJAndroid();
};

#endif // DIR_ACCESS_JANDROID_H
