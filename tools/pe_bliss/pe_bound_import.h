/*************************************************************************/
/* Copyright (c) 2015 dx, http://kaimi.ru                                */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person           */
/* obtaining a copy of this software and associated documentation        */
/* files (the "Software"), to deal in the Software without               */
/* restriction, including without limitation the rights to use,          */
/* copy, modify, merge, publish, distribute, sublicense, and/or          */
/* sell copies of the Software, and to permit persons to whom the        */
/* Software is furnished to do so, subject to the following conditions:  */
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
#pragma once
#include <vector>
#include <string>
#include "pe_structures.h"
#include "pe_base.h"
#include "pe_directory.h"

namespace pe_bliss
{
//Class representing bound import reference
class bound_import_ref
{
public:
	//Default constructor
	bound_import_ref();
	//Constructor from data
	bound_import_ref(const std::string& module_name, uint32_t timestamp);

	//Returns imported module name
	const std::string& get_module_name() const;
	//Returns bound import date and time stamp
	uint32_t get_timestamp() const;

public: //Setters
	//Sets module name
	void set_module_name(const std::string& module_name);
	//Sets timestamp
	void set_timestamp(uint32_t timestamp);

private:
	std::string module_name_; //Imported module name
	uint32_t timestamp_; //Bound import timestamp
};

//Class representing image bound import information
class bound_import
{
public:
	typedef std::vector<bound_import_ref> ref_list;

public:
	//Default constructor
	bound_import();
	//Constructor from data
	bound_import(const std::string& module_name, uint32_t timestamp);

	//Returns imported module name
	const std::string& get_module_name() const;
	//Returns bound import date and time stamp
	uint32_t get_timestamp() const;

	//Returns bound references cound
	size_t get_module_ref_count() const;
	//Returns module references
	const ref_list& get_module_ref_list() const;

public: //Setters
	//Sets module name
	void set_module_name(const std::string& module_name);
	//Sets timestamp
	void set_timestamp(uint32_t timestamp);

	//Adds module reference
	void add_module_ref(const bound_import_ref& ref);
	//Clears module references list
	void clear_module_refs();
	//Returns module references
	ref_list& get_module_ref_list();

private:
	std::string module_name_; //Imported module name
	uint32_t timestamp_; //Bound import timestamp
	ref_list refs_; //Module references list
};

typedef std::vector<bound_import> bound_import_module_list;

//Returns bound import information
const bound_import_module_list get_bound_import_module_list(const pe_base& pe);//Export directory rebuilder

//imports - bound imported modules list
//imports_section - section where export directory will be placed (must be attached to PE image)
//offset_from_section_start - offset from imports_section raw data start
//save_to_pe_headers - if true, new bound import directory information will be saved to PE image headers
//auto_strip_last_section - if true and bound imports are placed in the last section, it will be automatically stripped
const image_directory rebuild_bound_imports(pe_base& pe, const bound_import_module_list& imports, section& imports_section, uint32_t offset_from_section_start = 0, bool save_to_pe_header = true, bool auto_strip_last_section = true);
}
