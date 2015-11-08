#pragma once
#include <vector>
#include "pe_structures.h"
#include "pe_base.h"

namespace pe_bliss
{
//Class representing advanced RSDS (PDB 7.0) information
class pdb_7_0_info
{
public:
	//Default constructor
	pdb_7_0_info();
	//Constructor from data
	explicit pdb_7_0_info(const pe_win::CV_INFO_PDB70* info);

	//Returns debug PDB 7.0 structure GUID
	const pe_win::guid get_guid() const;
	//Returns age of build
	uint32_t get_age() const;
	//Returns PDB file name / path
	const std::string& get_pdb_file_name() const;

private:
	uint32_t age_;
	pe_win::guid guid_;
	std::string pdb_file_name_;
};

//Class representing advanced NB10 (PDB 2.0) information
class pdb_2_0_info
{
public:
	//Default constructor
	pdb_2_0_info();
	//Constructor from data
	explicit pdb_2_0_info(const pe_win::CV_INFO_PDB20* info);

	//Returns debug PDB 2.0 structure signature
	uint32_t get_signature() const;
	//Returns age of build
	uint32_t get_age() const;
	//Returns PDB file name / path
	const std::string& get_pdb_file_name() const;

private:
	uint32_t age_;
	uint32_t signature_;
	std::string pdb_file_name_;
};

//Class representing advanced misc (IMAGE_DEBUG_TYPE_MISC) info
class misc_debug_info
{
public:
	//Default constructor
	misc_debug_info();
	//Constructor from data
	explicit misc_debug_info(const pe_win::image_debug_misc* info);

	//Returns debug data type
	uint32_t get_data_type() const;
	//Returns true if data type is exe name
	bool is_exe_name() const;

	//Returns true if debug data is UNICODE
	bool is_unicode() const;
	//Returns debug data (ANSI or UNICODE)
	const std::string& get_data_ansi() const;
	const std::wstring& get_data_unicode() const;

private:
	uint32_t data_type_;
	bool unicode_;
	std::string debug_data_ansi_;
	std::wstring debug_data_unicode_;
};

//Class representing COFF (IMAGE_DEBUG_TYPE_COFF) debug info
class coff_debug_info
{
public:
	//Structure representing COFF symbol
	struct coff_symbol
	{
	public:
		//Default constructor
		coff_symbol();

		//Returns storage class
		uint32_t get_storage_class() const;
		//Returns symbol index
		uint32_t get_index() const;
		//Returns section number
		uint32_t get_section_number() const;
		//Returns RVA
		uint32_t get_rva() const;
		//Returns type
		uint16_t get_type() const;

		//Returns true if structure contains file name
		bool is_file() const;
		//Returns text data (symbol or file name)
		const std::string& get_symbol() const;

	public: //These functions do not change everything inside image, they are used by PE class
		//Sets storage class
		void set_storage_class(uint32_t storage_class);
		//Sets symbol index
		void set_index(uint32_t index);
		//Sets section number
		void set_section_number(uint32_t section_number);
		//Sets RVA
		void set_rva(uint32_t rva);
		//Sets type
		void set_type(uint16_t type);

		//Sets file name
		void set_file_name(const std::string& file_name);
		//Sets symbol name
		void set_symbol_name(const std::string& symbol_name);

	private:
		uint32_t storage_class_;
		uint32_t index_;
		uint32_t section_number_, rva_;
		uint16_t type_;
		bool is_filename_;
		std::string name_;
	};

public:
	typedef std::vector<coff_symbol> coff_symbols_list;

public:
	//Default constructor
	coff_debug_info();
	//Constructor from data
	explicit coff_debug_info(const pe_win::image_coff_symbols_header* info);

	//Returns number of symbols
	uint32_t get_number_of_symbols() const;
	//Returns virtual address of the first symbol
	uint32_t get_lva_to_first_symbol() const;
	//Returns number of line-number entries
	uint32_t get_number_of_line_numbers() const;
	//Returns virtual address of the first line-number entry
	uint32_t get_lva_to_first_line_number() const;
	//Returns relative virtual address of the first byte of code
	uint32_t get_rva_to_first_byte_of_code() const;
	//Returns relative virtual address of the last byte of code
	uint32_t get_rva_to_last_byte_of_code() const;
	//Returns relative virtual address of the first byte of data
	uint32_t get_rva_to_first_byte_of_data() const;
	//Returns relative virtual address of the last byte of data
	uint32_t get_rva_to_last_byte_of_data() const;

	//Returns COFF symbols list
	const coff_symbols_list& get_symbols() const;

public: //These functions do not change everything inside image, they are used by PE class
	//Adds COFF symbol
	void add_symbol(const coff_symbol& sym);

private:
	uint32_t number_of_symbols_;
	uint32_t lva_to_first_symbol_;
	uint32_t number_of_line_numbers_;
	uint32_t lva_to_first_line_number_;
	uint32_t rva_to_first_byte_of_code_;
	uint32_t rva_to_last_byte_of_code_;
	uint32_t rva_to_first_byte_of_data_;
	uint32_t rva_to_last_byte_of_data_;

private:
	coff_symbols_list symbols_;
};

//Class representing debug information
class debug_info
{
public:
	//Enumeration of debug information types
	enum debug_info_type
	{
		debug_type_unknown,
		debug_type_coff,
		debug_type_codeview,
		debug_type_fpo,
		debug_type_misc,
		debug_type_exception,
		debug_type_fixup,
		debug_type_omap_to_src,
		debug_type_omap_from_src,
		debug_type_borland,
		debug_type_reserved10,
		debug_type_clsid
	};

public:
	//Enumeration of advanced debug information types
	enum advanced_info_type
	{
		advanced_info_none, //No advanced info
		advanced_info_pdb_7_0, //PDB 7.0
		advanced_info_pdb_2_0, //PDB 2.0
		advanced_info_misc, //MISC debug info
		advanced_info_coff, //COFF debug info
		//No advanced info structures available for types below
		advanced_info_codeview_4_0, //CodeView 4.0
		advanced_info_codeview_5_0, //CodeView 5.0
		advanced_info_codeview //CodeView
	};

public:
	//Default constructor
	debug_info();
	//Constructor from data
	explicit debug_info(const pe_win::image_debug_directory& debug);
	//Copy constructor
	debug_info(const debug_info& info);
	//Copy assignment operator
	debug_info& operator=(const debug_info& info);
	//Destructor
	~debug_info();

	//Returns debug characteristics
	uint32_t get_characteristics() const;
	//Returns debug datetimestamp
	uint32_t get_time_stamp() const;
	//Returns major version
	uint32_t get_major_version() const;
	//Returns minor version
	uint32_t get_minor_version() const;
	//Returns type of debug info (unchecked)
	uint32_t get_type_raw() const;
	//Returns type of debug info from debug_info_type enumeration
	debug_info_type get_type() const;
	//Returns size of debug data (internal, .pdb or other file doesn't count)
	uint32_t get_size_of_data() const;
	//Returns RVA of debug info when mapped to memory or zero, if info is not mapped
	uint32_t get_rva_of_raw_data() const;
	//Returns raw file pointer to raw data
	uint32_t get_pointer_to_raw_data() const;

	//Returns advanced debug information type
	advanced_info_type get_advanced_info_type() const;
	//Returns advanced debug information or throws an exception,
	//if requested information type is not contained by structure
	template<typename AdvancedInfo>
	const AdvancedInfo get_advanced_debug_info() const;

public: //These functions do not change everything inside image, they are used by PE class
	//Sets advanced debug information
	void set_advanced_debug_info(const pdb_7_0_info& info);
	void set_advanced_debug_info(const pdb_2_0_info& info);
	void set_advanced_debug_info(const misc_debug_info& info);
	void set_advanced_debug_info(const coff_debug_info& info);

	//Sets advanced debug information type, if no advanced info structure available
	void set_advanced_info_type(advanced_info_type type);

private:
	uint32_t characteristics_;
	uint32_t time_stamp_;
	uint32_t major_version_, minor_version_;
	uint32_t type_;
	uint32_t size_of_data_;
	uint32_t address_of_raw_data_; //RVA when mapped or 0
	uint32_t pointer_to_raw_data_; //RAW file offset

	//Union containing advanced debug information pointer
	union advanced_info
	{
	public:
		//Default constructor
		advanced_info();

		//Returns true if advanced debug info is present
		bool is_present() const;

	public:
		pdb_7_0_info* adv_pdb_7_0_info;
		pdb_2_0_info* adv_pdb_2_0_info;
		misc_debug_info* adv_misc_info;
		coff_debug_info* adv_coff_info;
	};

	//Helper for advanced debug information copying
	void copy_advanced_info(const debug_info& info);
	//Helper for clearing any present advanced debug information
	void free_present_advanced_info();

	advanced_info advanced_debug_info_;
	//Advanced information type
	advanced_info_type advanced_info_type_;
};

typedef std::vector<debug_info> debug_info_list;

//Returns debug information list
const debug_info_list get_debug_information(const pe_base& pe);
}
