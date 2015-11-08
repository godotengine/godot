#pragma once
#include <string>
#include <vector>
#include "pe_structures.h"

namespace pe_bliss
{
//Enumeration of section data types, used in functions below
enum section_data_type
{
	section_data_raw,
	section_data_virtual
};

//Class representing image section
class section
{
public:
	//Default constructor
	section();

	//Sets the name of section (stripped to 8 characters)
	void set_name(const std::string& name);

	//Returns the name of section
	const std::string get_name() const;

	//Changes attributes of section
	section& readable(bool readable);
	section& writeable(bool writeable);
	section& executable(bool executable);
	section& shared(bool shared);
	section& discardable(bool discardable);

	//Returns attributes of section
	bool readable() const;
	bool writeable() const;
	bool executable() const;
	bool shared() const;
	bool discardable() const;

	//Returns true if section has no RAW data
	bool empty() const;

	//Returns raw section data from file image
	std::string& get_raw_data();
	//Returns raw section data from file image
	const std::string& get_raw_data() const;
	//Returns mapped virtual section data
	const std::string& get_virtual_data(uint32_t section_alignment) const;
	//Returns mapped virtual section data
	std::string& get_virtual_data(uint32_t section_alignment);

public: //Header getters
	//Returns section virtual size
	uint32_t get_virtual_size() const;
	//Returns section virtual address (RVA)
	uint32_t get_virtual_address() const;
	//Returns size of section raw data
	uint32_t get_size_of_raw_data() const;
	//Returns pointer to raw section data in PE file
	uint32_t get_pointer_to_raw_data() const;
	//Returns section characteristics
	uint32_t get_characteristics() const;

	//Returns raw image section header
	const pe_win::image_section_header& get_raw_header() const;

public: //Aligned sizes calculation
	//Calculates aligned virtual section size
	uint32_t get_aligned_virtual_size(uint32_t section_alignment) const;
	//Calculates aligned raw section size
	uint32_t get_aligned_raw_size(uint32_t file_alignment) const;

public: //Setters
	//Sets size of raw section data
	void set_size_of_raw_data(uint32_t size_of_raw_data);
	//Sets pointer to section raw data
	void set_pointer_to_raw_data(uint32_t pointer_to_raw_data);
	//Sets section characteristics
	void set_characteristics(uint32_t characteristics);
	//Sets raw section data from file image
	void set_raw_data(const std::string& data);

public: //Setters, be careful
	//Sets section virtual size (doesn't set internal aligned virtual size, changes only header value)
	//Better use pe_base::set_section_virtual_size
	void set_virtual_size(uint32_t virtual_size);
	//Sets section virtual address
	void set_virtual_address(uint32_t virtual_address);
	//Returns raw image section header
	pe_win::image_section_header& get_raw_header();

private:
	//Section header
	pe_win::image_section_header header_;

	//Maps virtual section data
	void map_virtual(uint32_t section_alignment) const;

	//Unmaps virtual section data
	void unmap_virtual() const;

	//Set flag (attribute) of section
	section& set_flag(uint32_t flag, bool setflag);

	//Old size of section (stored after mapping of virtual section memory)
	mutable std::size_t old_size_;

	//Section raw/virtual data
	mutable std::string raw_data_;
};

//Section by file offset finder helper (4gb max)
struct section_by_raw_offset
{
public:
	explicit section_by_raw_offset(uint32_t offset);
	bool operator()(const section& s) const;

private:
	uint32_t offset_;
};

//Helper: finder of section* in sections list
struct section_ptr_finder
{
public:
	explicit section_ptr_finder(const section& s);
	bool operator()(const section& s) const;

private:
	const section& s_;
};

typedef std::vector<section> section_list;
}
