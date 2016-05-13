/*************************************************************************/
/*  cp_file_access_wrapper.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef CP_FILE_ACCESS_WRAPPER_H
#define CP_FILE_ACCESS_WRAPPER_H

#include "cp_config.h"

class CPFileAccessWrapper {
public:

	enum ModeFlags  {

		READ=1,
		WRITE=2,
		READ_WRITE=3,
	};
	
	enum Error {

		OK,
		ERROR_FILE_NOT_FOUND,
		ERROR_FILE_BAD_DRIVE,
		ERROR_FILE_BAD_PATH,
		ERROR_FILE_NO_PERMISSION,
		ERROR_ALREADY_IN_USE,
		ERROR_INVALID_PARAMETERS,
		ERROR_OPENING_FILE,
		ERROR_READING_FILE,
		ERROR_WRITING_FILE
	};

	virtual Error open(const char *p_filename, int p_mode_flags)=0;
	virtual void close()=0;
	
	virtual void seek(uint32_t p_position)=0;
	virtual void seek_end()=0;
	virtual uint32_t get_pos()=0;

	virtual bool eof_reached()=0;

	virtual uint8_t get_byte()=0;
	virtual void get_byte_array(uint8_t *p_dest,int p_elements)=0;
	virtual void get_word_array(uint16_t *p_dest,int p_elements)=0;

	virtual uint16_t get_word()=0;
	virtual uint32_t get_dword()=0;

	// use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
	// It's not about the current CPU type but file formats.
	// this flags get reset to false (little endian) on each open
	virtual void set_endian_conversion(bool p_swap)=0;
	virtual bool is_open()=0;

	virtual Error get_error()=0;

	virtual void store_byte(uint8_t p_dest)=0;
	virtual void store_byte_array(const uint8_t *p_dest,int p_elements)=0;

	virtual void store_word(uint16_t p_dest)=0;
	virtual void store_dword(uint32_t p_dest)=0;



	virtual ~CPFileAccessWrapper(){}

};



#endif
