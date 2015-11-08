#pragma once
#include <vector>
#include <string>
#include "pe_structures.h"
#include "pe_base.h"
#include "pe_directory.h"

namespace pe_bliss
{
//Class representing exported function
class exported_function
{
public:
	//Default constructor
	exported_function();

	//Returns ordinal of function (actually, ordinal = hint + ordinal base)
	uint16_t get_ordinal() const;

	//Returns RVA of function
	uint32_t get_rva() const;

	//Returns true if function has name and name ordinal
	bool has_name() const;
	//Returns name of function
	const std::string& get_name() const;
	//Returns name ordinal of function
	uint16_t get_name_ordinal() const;

	//Returns true if function is forwarded to other library
	bool is_forwarded() const;
	//Returns the name of forwarded function
	const std::string& get_forwarded_name() const;

public: //Setters do not change everything inside image, they are used by PE class
	//You can also use them to rebuild export directory

	//Sets ordinal of function
	void set_ordinal(uint16_t ordinal);

	//Sets RVA of function
	void set_rva(uint32_t rva);

	//Sets name of function (or clears it, if empty name is passed)
	void set_name(const std::string& name);
	//Sets name ordinal
	void set_name_ordinal(uint16_t name_ordinal);

	//Sets forwarded function name (or clears it, if empty name is passed)
	void set_forwarded_name(const std::string& name);

private:
	uint16_t ordinal_; //Function ordinal
	uint32_t rva_; //Function RVA
	std::string name_; //Function name
	bool has_name_; //true == function has name
	uint16_t name_ordinal_; //Function name ordinal
	bool forward_; //true == function is forwarded
	std::string forward_name_; //Name of forwarded function
};

//Class representing export information
class export_info
{
public:
	//Default constructor
	export_info();

	//Returns characteristics
	uint32_t get_characteristics() const;
	//Returns timestamp
	uint32_t get_timestamp() const;
	//Returns major version
	uint16_t get_major_version() const;
	//Returns minor version
	uint16_t get_minor_version() const;
	//Returns DLL name
	const std::string& get_name() const;
	//Returns ordinal base
	uint32_t get_ordinal_base() const;
	//Returns number of functions
	uint32_t get_number_of_functions() const;
	//Returns number of function names
	uint32_t get_number_of_names() const;
	//Returns RVA of function address table
	uint32_t get_rva_of_functions() const;
	//Returns RVA of function name address table
	uint32_t get_rva_of_names() const;
	//Returns RVA of name ordinals table
	uint32_t get_rva_of_name_ordinals() const;

public: //Setters do not change everything inside image, they are used by PE class
	//You can also use them to rebuild export directory using rebuild_exports

	//Sets characteristics
	void set_characteristics(uint32_t characteristics);
	//Sets timestamp
	void set_timestamp(uint32_t timestamp);
	//Sets major version
	void set_major_version(uint16_t major_version);
	//Sets minor version
	void set_minor_version(uint16_t minor_version);
	//Sets DLL name
	void set_name(const std::string& name);
	//Sets ordinal base
	void set_ordinal_base(uint32_t ordinal_base);
	//Sets number of functions
	void set_number_of_functions(uint32_t number_of_functions);
	//Sets number of function names
	void set_number_of_names(uint32_t number_of_names);
	//Sets RVA of function address table
	void set_rva_of_functions(uint32_t rva_of_functions);
	//Sets RVA of function name address table
	void set_rva_of_names(uint32_t rva_of_names);
	//Sets RVA of name ordinals table
	void set_rva_of_name_ordinals(uint32_t rva_of_name_ordinals);

private:
	uint32_t characteristics_;
	uint32_t timestamp_;
	uint16_t major_version_;
	uint16_t minor_version_;
	std::string name_;
	uint32_t ordinal_base_;
	uint32_t number_of_functions_;
	uint32_t number_of_names_;
	uint32_t address_of_functions_;
	uint32_t address_of_names_;
	uint32_t address_of_name_ordinals_;
};

//Exported functions list typedef
typedef std::vector<exported_function> exported_functions_list;

//Returns array of exported functions
const exported_functions_list get_exported_functions(const pe_base& pe);
//Returns array of exported functions and information about export
const exported_functions_list get_exported_functions(const pe_base& pe, export_info& info);
	
//Helper export functions
//Returns pair: <ordinal base for supplied functions; maximum ordinal value for supplied functions>
const std::pair<uint16_t, uint16_t> get_export_ordinal_limits(const exported_functions_list& exports);

//Checks if exported function name already exists
bool exported_name_exists(const std::string& function_name, const exported_functions_list& exports);

//Checks if exported function ordinal already exists
bool exported_ordinal_exists(uint16_t ordinal, const exported_functions_list& exports);

//Export directory rebuilder
//info - export information
//exported_functions_list - list of exported functions
//exports_section - section where export directory will be placed (must be attached to PE image)
//offset_from_section_start - offset from exports_section raw data start
//save_to_pe_headers - if true, new export directory information will be saved to PE image headers
//auto_strip_last_section - if true and exports are placed in the last section, it will be automatically stripped
//number_of_functions and number_of_names parameters don't matter in "info" when rebuilding, they're calculated independently
//characteristics, major_version, minor_version, timestamp and name are the only used members of "info" structure
//Returns new export directory information
//exported_functions_list is copied intentionally to be sorted by ordinal values later
//Name ordinals in exported function don't matter, they will be recalculated
const image_directory rebuild_exports(pe_base& pe, const export_info& info, exported_functions_list exports, section& exports_section, uint32_t offset_from_section_start = 0, bool save_to_pe_header = true, bool auto_strip_last_section = true);
}
