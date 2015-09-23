#pragma once
#include <vector>
#include <string>
#include "pe_structures.h"
#include "pe_directory.h"
#include "pe_base.h"

namespace pe_bliss
{
//Class representing imported function
class imported_function
{
public:
	//Default constructor
	imported_function();

	//Returns true if imported function has name (and hint)
	bool has_name() const;
	//Returns name of function
	const std::string& get_name() const;
	//Returns hint
	uint16_t get_hint() const;
	//Returns ordinal of function
	uint16_t get_ordinal() const;

	//Returns IAT entry VA (usable if image has both IAT and original IAT and is bound)
	uint64_t get_iat_va() const;

public: //Setters do not change everything inside image, they are used by PE class
	//You also can use them to rebuild image imports
	//Sets name of function
	void set_name(const std::string& name);
	//Sets hint
	void set_hint(uint16_t hint);
	//Sets ordinal
	void set_ordinal(uint16_t ordinal);

	//Sets IAT entry VA (usable if image has both IAT and original IAT and is bound)
	void set_iat_va(uint64_t rva);

private:
	std::string name_; //Function name
	uint16_t hint_; //Hint
	uint16_t ordinal_; //Ordinal
	uint64_t iat_va_;
};

//Class representing imported library information
class import_library
{
public:
	typedef std::vector<imported_function> imported_list;

public:
	//Default constructor
	import_library();

	//Returns name of library
	const std::string& get_name() const;
	//Returns RVA to Import Address Table (IAT)
	uint32_t get_rva_to_iat() const;
	//Returns RVA to Original Import Address Table (Original IAT)
	uint32_t get_rva_to_original_iat() const;
	//Returns timestamp
	uint32_t get_timestamp() const;

	//Returns imported functions list
	const imported_list& get_imported_functions() const;

public: //Setters do not change everything inside image, they are used by PE class
	//You also can use them to rebuild image imports
	//Sets name of library
	void set_name(const std::string& name);
	//Sets RVA to Import Address Table (IAT)
	void set_rva_to_iat(uint32_t rva_to_iat);
	//Sets RVA to Original Import Address Table (Original IAT)
	void set_rva_to_original_iat(uint32_t rva_to_original_iat);
	//Sets timestamp
	void set_timestamp(uint32_t timestamp);

	//Adds imported function
	void add_import(const imported_function& func);
	//Clears imported functions list
	void clear_imports();

private:
	std::string name_; //Library name
	uint32_t rva_to_iat_; //RVA to IAT
	uint32_t rva_to_original_iat_; //RVA to original IAT
	uint32_t timestamp_; //DLL TimeStamp

	imported_list imports_;
};

//Simple import directory rebuilder
//Class representing import rebuilder advanced settings
class import_rebuilder_settings
{
public:
	//Default constructor
	//Default constructor
	//If set_to_pe_headers = true, IMAGE_DIRECTORY_ENTRY_IMPORT entry will be reset
	//to new value after import rebuilding
	//If auto_zero_directory_entry_iat = true, IMAGE_DIRECTORY_ENTRY_IAT will be set to zero
	//IMAGE_DIRECTORY_ENTRY_IAT is used by loader to temporarily make section, where IMAGE_DIRECTORY_ENTRY_IAT RVA points, writeable
	//to be able to modify IAT thunks
	explicit import_rebuilder_settings(bool set_to_pe_headers = true, bool auto_zero_directory_entry_iat = false);

	//Returns offset from section start where import directory data will be placed
	uint32_t get_offset_from_section_start() const;
	//Returns true if Original import address table (IAT) will be rebuilt
	bool build_original_iat() const;

	//Returns true if Original import address and import address tables will not be rebuilt,
	//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
	bool save_iat_and_original_iat_rvas() const;
	//Returns true if Original import address and import address tables contents will be rewritten
	//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
	//and save_iat_and_original_iat_rvas is true
	bool rewrite_iat_and_original_iat_contents() const;

	//Returns true if original missing IATs will be rebuilt
	//(only if IATs are saved)
	bool fill_missing_original_iats() const;
	//Returns true if PE headers should be updated automatically after rebuilding of imports
	bool auto_set_to_pe_headers() const;
	//Returns true if IMAGE_DIRECTORY_ENTRY_IAT must be zeroed, works only if auto_set_to_pe_headers = true
	bool zero_directory_entry_iat() const;

	//Returns true if the last section should be stripped automatically, if imports are inside it
	bool auto_strip_last_section_enabled() const;

public: //Setters
	//Sets offset from section start where import directory data will be placed
	void set_offset_from_section_start(uint32_t offset);
	//Sets if Original import address table (IAT) will be rebuilt
	void build_original_iat(bool enable);
	//Sets if Original import address and import address tables will not be rebuilt,
	//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
	//enable_rewrite_iat_and_original_iat_contents sets if Original import address and import address tables contents will be rewritten
	//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
	//and save_iat_and_original_iat_rvas is true
	void save_iat_and_original_iat_rvas(bool enable, bool enable_rewrite_iat_and_original_iat_contents = false);
	//Sets if original missing IATs will be rebuilt
	//(only if IATs are saved)
	void fill_missing_original_iats(bool enable);
	//Sets if PE headers should be updated automatically after rebuilding of imports
	void auto_set_to_pe_headers(bool enable);
	//Sets if IMAGE_DIRECTORY_ENTRY_IAT must be zeroed, works only if auto_set_to_pe_headers = true
	void zero_directory_entry_iat(bool enable);

	//Sets if the last section should be stripped automatically, if imports are inside it, default true
	void enable_auto_strip_last_section(bool enable);

private:
	uint32_t offset_from_section_start_;
	bool build_original_iat_;
	bool save_iat_and_original_iat_rvas_;
	bool fill_missing_original_iats_;
	bool set_to_pe_headers_;
	bool zero_directory_entry_iat_;
	bool rewrite_iat_and_original_iat_contents_;
	bool auto_strip_last_section_;
};

typedef std::vector<import_library> imported_functions_list;


//Returns imported functions list with related libraries info
const imported_functions_list get_imported_functions(const pe_base& pe);

template<typename PEClassType>
const imported_functions_list get_imported_functions_base(const pe_base& pe);


//You can get all image imports with get_imported_functions() function
//You can use returned value to, for example, add new imported library with some functions
//to the end of list of imported libraries
//To keep PE file working, rebuild its imports with save_iat_and_original_iat_rvas = true (default)
//Don't add new imported functions to existing imported library entries, because this can cause
//rewriting of some used memory (or other IAT/orig.IAT fields) by system loader
//The safest way is just adding import libraries with functions to the end of imported_functions_list array
const image_directory rebuild_imports(pe_base& pe, const imported_functions_list& imports, section& import_section, const import_rebuilder_settings& import_settings = import_rebuilder_settings());

template<typename PEClassType>
const image_directory rebuild_imports_base(pe_base& pe, const imported_functions_list& imports, section& import_section, const import_rebuilder_settings& import_settings = import_rebuilder_settings());
}
