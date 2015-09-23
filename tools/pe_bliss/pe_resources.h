#pragma once
#include <vector>
#include <string>
#include <set>
#include "pe_structures.h"
#include "pe_base.h"
#include "pe_directory.h"

namespace pe_bliss
{
//Class representing resource data entry
class resource_data_entry
{
public:
	//Default constructor
	resource_data_entry();
	//Constructor from data
	resource_data_entry(const std::string& data, uint32_t codepage);

	//Returns resource data codepage
	uint32_t get_codepage() const;
	//Returns resource data
	const std::string& get_data() const;
		
public: //These functions do not change everything inside image, they are used by PE class
	//You can also use them to rebuild resource directory
		
	//Sets resource data codepage
	void set_codepage(uint32_t codepage);
	//Sets resource data
	void set_data(const std::string& data);

private:
	uint32_t codepage_; //Resource data codepage
	std::string data_; //Resource data
};

//Forward declaration
class resource_directory;

//Class representing resource directory entry
class resource_directory_entry
{
public:
	//Default constructor
	resource_directory_entry();
	//Copy constructor
	resource_directory_entry(const resource_directory_entry& other);
	//Copy assignment operator
	resource_directory_entry& operator=(const resource_directory_entry& other);

	//Returns entry ID
	uint32_t get_id() const;
	//Returns entry name
	const std::wstring& get_name() const;
	//Returns true, if entry has name
	//Returns false, if entry has ID
	bool is_named() const;

	//Returns true, if entry includes resource_data_entry
	//Returns false, if entry includes resource_directory
	bool includes_data() const;
	//Returns resource_directory if entry includes it, otherwise throws an exception
	const resource_directory& get_resource_directory() const;
	//Returns resource_data_entry if entry includes it, otherwise throws an exception
	const resource_data_entry& get_data_entry() const;

	//Destructor
	~resource_directory_entry();

public: //These functions do not change everything inside image, they are used by PE class
	//You can also use them to rebuild resource directory

	//Sets entry name
	void set_name(const std::wstring& name);
	//Sets entry ID
	void set_id(uint32_t id);
		
	//Returns resource_directory if entry includes it, otherwise throws an exception
	resource_directory& get_resource_directory();
	//Returns resource_data_entry if entry includes it, otherwise throws an exception
	resource_data_entry& get_data_entry();

	//Adds resource_data_entry
	void add_data_entry(const resource_data_entry& entry);
	//Adds resource_directory
	void add_resource_directory(const resource_directory& dir);

private:
	//Destroys included data
	void release();

private:
	uint32_t id_;
	std::wstring name_;

	union includes
	{
		//Default constructor
		includes();

		//We use pointers, we're doing manual copying here
		class resource_data_entry* data_;
		class resource_directory* dir_; //We use pointer, because structs include each other
	};

	includes ptr_;

	bool includes_data_, named_;
};

//Class representing resource directory
class resource_directory
{
public:
	typedef std::vector<resource_directory_entry> entry_list;

public:
	//Default constructor
	resource_directory();
	//Constructor from data
	explicit resource_directory(const pe_win::image_resource_directory& dir);

	//Returns characteristics of directory
	uint32_t get_characteristics() const;
	//Returns date and time stamp of directory
	uint32_t get_timestamp() const;
	//Returns number of named entries
	uint32_t get_number_of_named_entries() const;
	//Returns number of ID entries
	uint32_t get_number_of_id_entries() const;
	//Returns major version of directory
	uint16_t get_major_version() const;
	//Returns minor version of directory
	uint16_t get_minor_version() const;
	//Returns resource_directory_entry array
	const entry_list& get_entry_list() const;
	//Returns resource_directory_entry by ID. If not found - throws an exception
	const resource_directory_entry& entry_by_id(uint32_t id) const;
	//Returns resource_directory_entry by name. If not found - throws an exception
	const resource_directory_entry& entry_by_name(const std::wstring& name) const;

public: //These functions do not change everything inside image, they are used by PE class
	//You can also use them to rebuild resource directory

	//Adds resource_directory_entry
	void add_resource_directory_entry(const resource_directory_entry& entry);
	//Clears resource_directory_entry array
	void clear_resource_directory_entry_list();

	//Sets characteristics of directory
	void set_characteristics(uint32_t characteristics);
	//Sets date and time stamp of directory
	void set_timestamp(uint32_t timestamp);
	//Sets number of named entries
	void set_number_of_named_entries(uint32_t number);
	//Sets number of ID entries
	void set_number_of_id_entries(uint32_t number);
	//Sets major version of directory
	void set_major_version(uint16_t major_version);
	//Sets minor version of directory
	void get_minor_version(uint16_t minor_version);
		
	//Returns resource_directory_entry array
	entry_list& get_entry_list();

private:
	uint32_t characteristics_;
	uint32_t timestamp_;
	uint16_t major_version_, minor_version_;
	uint32_t number_of_named_entries_, number_of_id_entries_;
	entry_list entries_;

public: //Finder helpers
	//Finds resource_directory_entry by ID
	struct id_entry_finder
	{
	public:
		explicit id_entry_finder(uint32_t id);
		bool operator()(const resource_directory_entry& entry) const;

	private:
		uint32_t id_;
	};

	//Finds resource_directory_entry by name
	struct name_entry_finder
	{
	public:
		explicit name_entry_finder(const std::wstring& name);
		bool operator()(const resource_directory_entry& entry) const;

	private:
		std::wstring name_;
	};

	//Finds resource_directory_entry by name or ID (universal)
	struct entry_finder
	{
	public:
		explicit entry_finder(const std::wstring& name);
		explicit entry_finder(uint32_t id);
		bool operator()(const resource_directory_entry& entry) const;

	private:
		std::wstring name_;
		uint32_t id_;
		bool named_;
	};
};

//Returns resources (root resource_directory) from PE file
const resource_directory get_resources(const pe_base& pe);

//Resources rebuilder
//resource_directory - root resource directory
//resources_section - section where resource directory will be placed (must be attached to PE image)
//resource_directory is non-constant, because it will be sorted
//offset_from_section_start - offset from resources_section raw data start
//save_to_pe_headers - if true, new resource directory information will be saved to PE image headers
//auto_strip_last_section - if true and resources are placed in the last section, it will be automatically stripped
//number_of_id_entries and number_of_named_entries for resource directories are recalculated and not used
const image_directory rebuild_resources(pe_base& pe, resource_directory& info, section& resources_section, uint32_t offset_from_section_start = 0, bool save_to_pe_header = true, bool auto_strip_last_section = true);
}
