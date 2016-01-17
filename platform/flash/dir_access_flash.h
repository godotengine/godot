/*************************************************/
/*  dir_access_psp.h                             */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef DIR_ACCESS_FLASH_H
#define DIR_ACCESS_FLASH_H

#include "core/os/dir_access.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

class DirAccessFlash : public DirAccess {

	DIR *dir_stream;

	String current_dir;
	bool _cisdir;

	static DirAccess* create_flash();

public:

	bool list_dir_begin(); ///< This starts dir listing
	String get_next();
	bool current_is_dir() const;

	void list_dir_end(); ///<

	int get_drive_count();
	String get_drive(int p_drive);

	Error change_dir(String p_dir); ///< can be relative or absolute, return false on success
	String get_current_dir(); ///< return current dir location

	Error make_dir(String p_dir);

	bool file_exists(String p_file);
	bool dir_exists(String p_dir);

	size_t get_space_left();

	uint64_t get_modified_time(String p_file);

	Error rename(String p_from, String p_to);
	Error remove(String p_name);

	static void make_default();

	DirAccessFlash();
	~DirAccessFlash();
};

#endif // DIR_ACCESS_PSP_H

