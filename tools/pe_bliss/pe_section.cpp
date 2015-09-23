#include <string.h>
#include "utils.h"
#include "pe_section.h"

namespace pe_bliss
{
using namespace pe_win;

//Section structure default constructor
section::section()
	:old_size_(static_cast<size_t>(-1))
{
	memset(&header_, 0, sizeof(image_section_header));
}

//Sets the name of section (8 characters maximum)
void section::set_name(const std::string& name)
{
	memset(header_.Name, 0, sizeof(header_.Name));
	memcpy(header_.Name, name.c_str(), std::min<size_t>(name.length(), sizeof(header_.Name)));
}

//Returns section name
const std::string section::get_name() const
{
	char buf[9] = {0};
	memcpy(buf, header_.Name, 8);
	return std::string(buf);
}

//Set flag (attribute) of section
section& section::set_flag(uint32_t flag, bool setflag)
{
	if(setflag)
		header_.Characteristics |= flag;
	else
		header_.Characteristics &= ~flag;

	return *this;
}

//Sets "readable" attribute of section
section& section::readable(bool readable)
{
	return set_flag(image_scn_mem_read, readable);
}

//Sets "writeable" attribute of section
section& section::writeable(bool writeable)
{
	return set_flag(image_scn_mem_write, writeable);
}

//Sets "executable" attribute of section
section& section::executable(bool executable)
{
	return set_flag(image_scn_mem_execute, executable);
}

//Sets "shared" attribute of section
section& section::shared(bool shared)
{
	return set_flag(image_scn_mem_shared, shared);
}

//Sets "discardable" attribute of section
section& section::discardable(bool discardable)
{
	return set_flag(image_scn_mem_discardable, discardable);
}

//Returns true if section is readable
bool section::readable() const
{
	return (header_.Characteristics & image_scn_mem_read) != 0;
}

//Returns true if section is writeable
bool section::writeable() const
{
	return (header_.Characteristics & image_scn_mem_write) != 0;
}

//Returns true if section is executable
bool section::executable() const
{
	return (header_.Characteristics & image_scn_mem_execute) != 0;
}

bool section::shared() const
{
	return (header_.Characteristics & image_scn_mem_shared) != 0;
}

bool section::discardable() const
{
	return (header_.Characteristics & image_scn_mem_discardable) != 0;
}

//Returns true if section has no RAW data
bool section::empty() const
{
	if(old_size_ != static_cast<size_t>(-1)) //If virtual memory is mapped, check raw data length (old_size_)
		return old_size_ == 0;
	else
		return raw_data_.empty();
}

//Returns raw section data from file image
std::string& section::get_raw_data()
{
	unmap_virtual();
	return raw_data_;
}

//Sets raw section data from file image
void section::set_raw_data(const std::string& data)
{
	old_size_ = static_cast<size_t>(-1);
	raw_data_ = data;
}

//Returns raw section data from file image
const std::string& section::get_raw_data() const
{
	unmap_virtual();
	return raw_data_;
}

//Returns mapped virtual section data
const std::string& section::get_virtual_data(uint32_t section_alignment) const
{
	map_virtual(section_alignment);
	return raw_data_;
}

//Returns mapped virtual section data
std::string& section::get_virtual_data(uint32_t section_alignment)
{
	map_virtual(section_alignment);
	return raw_data_;
}

//Maps virtual section data
void section::map_virtual(uint32_t section_alignment) const
{
	uint32_t aligned_virtual_size = get_aligned_virtual_size(section_alignment);
	if(old_size_ == static_cast<size_t>(-1) && aligned_virtual_size && aligned_virtual_size > raw_data_.length())
	{
		old_size_ = raw_data_.length();
		raw_data_.resize(aligned_virtual_size, 0);
	}
}

//Unmaps virtual section data
void section::unmap_virtual() const
{
	if(old_size_ != static_cast<size_t>(-1))
	{
		raw_data_.resize(old_size_, 0);
		old_size_ = static_cast<size_t>(-1);
	}
}

//Returns section virtual size
uint32_t section::get_virtual_size() const
{
	return header_.Misc.VirtualSize;
}

//Returns section virtual address
uint32_t section::get_virtual_address() const
{
	return header_.VirtualAddress;
}

//Returns size of section raw data
uint32_t section::get_size_of_raw_data() const
{
	return header_.SizeOfRawData;
}

//Returns pointer to raw section data in PE file
uint32_t section::get_pointer_to_raw_data() const
{
	return header_.PointerToRawData;
}

//Returns section characteristics
uint32_t section::get_characteristics() const
{
	return header_.Characteristics;
}

//Returns raw image section header
const pe_win::image_section_header& section::get_raw_header() const
{
	return header_;
}

//Returns raw image section header
pe_win::image_section_header& section::get_raw_header()
{
	return header_;
}

//Calculates aligned virtual section size
uint32_t section::get_aligned_virtual_size(uint32_t section_alignment) const
{
	if(get_size_of_raw_data())
	{
		if(!get_virtual_size())
		{
			//If section virtual size is zero
			//Set aligned virtual size of section as aligned raw size
			return pe_utils::align_up(get_size_of_raw_data(), section_alignment);
		}
	}

	return pe_utils::align_up(get_virtual_size(), section_alignment);
}

//Calculates aligned raw section size
uint32_t section::get_aligned_raw_size(uint32_t file_alignment) const
{
	if(get_size_of_raw_data())
		return pe_utils::align_up(get_size_of_raw_data(), file_alignment);
	else
		return 0;
}

//Sets size of raw section data
void section::set_size_of_raw_data(uint32_t size_of_raw_data)
{
	header_.SizeOfRawData = size_of_raw_data;
}

//Sets pointer to section raw data
void section::set_pointer_to_raw_data(uint32_t pointer_to_raw_data)
{
	header_.PointerToRawData = pointer_to_raw_data;
}

//Sets section characteristics
void section::set_characteristics(uint32_t characteristics)
{
	header_.Characteristics = characteristics;
}

//Sets section virtual size
void section::set_virtual_size(uint32_t virtual_size)
{
	header_.Misc.VirtualSize = virtual_size;
}

//Sets section virtual address
void section::set_virtual_address(uint32_t virtual_address)
{
	header_.VirtualAddress = virtual_address;
}

//Section by file offset finder helper (4gb max)
section_by_raw_offset::section_by_raw_offset(uint32_t offset)
	:offset_(offset)
{}

bool section_by_raw_offset::operator()(const section& s) const
{
	return (s.get_pointer_to_raw_data() <= offset_)
		&& (s.get_pointer_to_raw_data() + s.get_size_of_raw_data() > offset_);
}

section_ptr_finder::section_ptr_finder(const section& s)
	:s_(s)
{}

bool section_ptr_finder::operator()(const section& s) const
{
	return &s == &s_;
}
}
