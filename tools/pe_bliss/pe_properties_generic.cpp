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
#include <string.h>
#include "pe_properties_generic.h"
#include "pe_exception.h"
#include "utils.h"

namespace pe_bliss
{
using namespace pe_win;
	
//Constructor
template<typename PEClassType>
std::auto_ptr<pe_properties> pe_properties_generic<PEClassType>::duplicate() const
{
	return std::auto_ptr<pe_properties>(new pe_properties_generic<PEClassType>(*this));
}

//Fills properly PE structures
template<typename PEClassType>
void pe_properties_generic<PEClassType>::create_pe(uint32_t section_alignment, uint16_t subsystem)
{
	memset(&nt_headers_, 0, sizeof(nt_headers_));
	nt_headers_.Signature = 0x4550; //"PE"
	nt_headers_.FileHeader.Machine = 0x14C; //i386
	nt_headers_.FileHeader.SizeOfOptionalHeader = sizeof(nt_headers_.OptionalHeader);
	nt_headers_.OptionalHeader.Magic = PEClassType::Id;
	nt_headers_.OptionalHeader.ImageBase = 0x400000;
	nt_headers_.OptionalHeader.SectionAlignment = section_alignment;
	nt_headers_.OptionalHeader.FileAlignment = 0x200;
	nt_headers_.OptionalHeader.SizeOfHeaders = 1024;
	nt_headers_.OptionalHeader.Subsystem = subsystem;
	nt_headers_.OptionalHeader.SizeOfHeapReserve = 0x100000;
	nt_headers_.OptionalHeader.SizeOfHeapCommit = 0x1000;
	nt_headers_.OptionalHeader.SizeOfStackReserve = 0x100000;
	nt_headers_.OptionalHeader.SizeOfStackCommit = 0x1000;
	nt_headers_.OptionalHeader.NumberOfRvaAndSizes = 0x10;
}

//Duplicate
template<typename PEClassType>
pe_properties_generic<PEClassType>::~pe_properties_generic()
{}

//Returns true if directory exists
template<typename PEClassType>
bool pe_properties_generic<PEClassType>::directory_exists(uint32_t id) const
{
	return (nt_headers_.OptionalHeader.NumberOfRvaAndSizes - 1) >= id &&
		nt_headers_.OptionalHeader.DataDirectory[id].VirtualAddress;
}

//Removes directory
template<typename PEClassType>
void pe_properties_generic<PEClassType>::remove_directory(uint32_t id)
{
	if(directory_exists(id))
	{
		nt_headers_.OptionalHeader.DataDirectory[id].VirtualAddress = 0;
		nt_headers_.OptionalHeader.DataDirectory[id].Size = 0;

		if(id == image_directory_entry_basereloc)
		{
			set_characteristics_flags(image_file_relocs_stripped);
			set_dll_characteristics(get_dll_characteristics() & ~image_dllcharacteristics_dynamic_base);
		}
		else if(id == image_directory_entry_export)
		{
			clear_characteristics_flags(image_file_dll);
		}
	}
}

//Returns directory RVA
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_directory_rva(uint32_t id) const
{
	//Check if directory exists
	if(nt_headers_.OptionalHeader.NumberOfRvaAndSizes <= id)
		throw pe_exception("Specified directory does not exist", pe_exception::directory_does_not_exist);

	return nt_headers_.OptionalHeader.DataDirectory[id].VirtualAddress;
}

//Returns directory size
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_directory_rva(uint32_t id, uint32_t va)
{
	//Check if directory exists
	if(nt_headers_.OptionalHeader.NumberOfRvaAndSizes <= id)
		throw pe_exception("Specified directory does not exist", pe_exception::directory_does_not_exist);

	nt_headers_.OptionalHeader.DataDirectory[id].VirtualAddress = va;
}

template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_directory_size(uint32_t id, uint32_t size)
{
	//Check if directory exists
	if(nt_headers_.OptionalHeader.NumberOfRvaAndSizes <= id)
		throw pe_exception("Specified directory does not exist", pe_exception::directory_does_not_exist);

	nt_headers_.OptionalHeader.DataDirectory[id].Size = size;
}

//Returns directory size
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_directory_size(uint32_t id) const
{
	//Check if directory exists
	if(nt_headers_.OptionalHeader.NumberOfRvaAndSizes <= id)
		throw pe_exception("Specified directory does not exist", pe_exception::directory_does_not_exist);

	return nt_headers_.OptionalHeader.DataDirectory[id].Size;
}

//Strips only zero DATA_DIRECTORY entries to count = min_count
//Returns resulting number of data directories
//strip_iat_directory - if true, even not empty IAT directory will be stripped
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::strip_data_directories(uint32_t min_count, bool strip_iat_directory)
{
	int i = nt_headers_.OptionalHeader.NumberOfRvaAndSizes - 1;

	//Enumerate all data directories from the end
	for(; i >= 0; i--)
	{
		//If directory exists, break
		if(nt_headers_.OptionalHeader.DataDirectory[i].VirtualAddress && (static_cast<uint32_t>(i) != image_directory_entry_iat || !strip_iat_directory))
			break;

		if(i <= static_cast<int>(min_count) - 2)
			break;
	}

	if(i == image_numberof_directory_entries - 1)
		return image_numberof_directory_entries;

	//Return new number of data directories
	return nt_headers_.OptionalHeader.NumberOfRvaAndSizes = i + 1;
}

//Returns image base for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_image_base_32() const
{
	return static_cast<uint32_t>(nt_headers_.OptionalHeader.ImageBase);
}

//Returns image base for PE32/PE64
template<typename PEClassType>
uint64_t pe_properties_generic<PEClassType>::get_image_base_64() const
{
	return static_cast<uint64_t>(nt_headers_.OptionalHeader.ImageBase);
}

//Sets new image base
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_image_base(uint32_t base)
{
	nt_headers_.OptionalHeader.ImageBase = base;
}

//Sets new image base
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_image_base_64(uint64_t base)
{
	nt_headers_.OptionalHeader.ImageBase = static_cast<typename PEClassType::BaseSize>(base);
}

//Returns image entry point
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_ep() const
{
	return nt_headers_.OptionalHeader.AddressOfEntryPoint;
}

//Sets image entry point
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_ep(uint32_t new_ep)
{
	nt_headers_.OptionalHeader.AddressOfEntryPoint = new_ep;
}

//Returns file alignment
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_file_alignment() const
{
	return nt_headers_.OptionalHeader.FileAlignment;
}

//Returns section alignment
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_section_alignment() const
{
	return nt_headers_.OptionalHeader.SectionAlignment;
}

//Sets heap size commit for PE32
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_heap_size_commit(uint32_t size)
{
	nt_headers_.OptionalHeader.SizeOfHeapCommit = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets heap size commit for PE32/PE64
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_heap_size_commit(uint64_t size)
{
	nt_headers_.OptionalHeader.SizeOfHeapCommit = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets heap size reserve for PE32
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_heap_size_reserve(uint32_t size)
{
	nt_headers_.OptionalHeader.SizeOfHeapReserve = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets heap size reserve for PE32/PE64
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_heap_size_reserve(uint64_t size)
{
	nt_headers_.OptionalHeader.SizeOfHeapReserve = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets stack size commit for PE32
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_stack_size_commit(uint32_t size)
{
	nt_headers_.OptionalHeader.SizeOfStackCommit = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets stack size commit for PE32/PE64
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_stack_size_commit(uint64_t size)
{
	nt_headers_.OptionalHeader.SizeOfStackCommit = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets stack size reserve for PE32
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_stack_size_reserve(uint32_t size)
{
	nt_headers_.OptionalHeader.SizeOfStackReserve = static_cast<typename PEClassType::BaseSize>(size);
}

//Sets stack size reserve for PE32/PE64
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_stack_size_reserve(uint64_t size)
{
	nt_headers_.OptionalHeader.SizeOfStackReserve = static_cast<typename PEClassType::BaseSize>(size);
}

//Returns heap size commit for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_heap_size_commit_32() const
{
	return static_cast<uint32_t>(nt_headers_.OptionalHeader.SizeOfHeapCommit);
}

//Returns heap size commit for PE32/PE64
template<typename PEClassType>
uint64_t pe_properties_generic<PEClassType>::get_heap_size_commit_64() const
{
	return static_cast<uint64_t>(nt_headers_.OptionalHeader.SizeOfHeapCommit);
}

//Returns heap size reserve for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_heap_size_reserve_32() const
{
	return static_cast<uint32_t>(nt_headers_.OptionalHeader.SizeOfHeapReserve);
}

//Returns heap size reserve for PE32/PE64
template<typename PEClassType>
uint64_t pe_properties_generic<PEClassType>::get_heap_size_reserve_64() const
{
	return static_cast<uint64_t>(nt_headers_.OptionalHeader.SizeOfHeapReserve);
}

//Returns stack size commit for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_stack_size_commit_32() const
{
	return static_cast<uint32_t>(nt_headers_.OptionalHeader.SizeOfStackCommit);
}

//Returns stack size commit for PE32/PE64
template<typename PEClassType>
uint64_t pe_properties_generic<PEClassType>::get_stack_size_commit_64() const
{
	return static_cast<uint64_t>(nt_headers_.OptionalHeader.SizeOfStackCommit);
}

//Returns stack size reserve for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_stack_size_reserve_32() const
{
	return static_cast<uint32_t>(nt_headers_.OptionalHeader.SizeOfStackReserve);
}

//Returns stack size reserve for PE32/PE64
template<typename PEClassType>
uint64_t pe_properties_generic<PEClassType>::get_stack_size_reserve_64() const
{
	return static_cast<uint64_t>(nt_headers_.OptionalHeader.SizeOfStackReserve);
}

//Returns virtual size of image
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_size_of_image() const
{
	return nt_headers_.OptionalHeader.SizeOfImage;
}

//Returns number of RVA and sizes (number of DATA_DIRECTORY entries)
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_number_of_rvas_and_sizes() const
{
	return nt_headers_.OptionalHeader.NumberOfRvaAndSizes;
}

//Sets number of RVA and sizes (number of DATA_DIRECTORY entries)
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_number_of_rvas_and_sizes(uint32_t number)
{
	nt_headers_.OptionalHeader.NumberOfRvaAndSizes = number;
}

//Returns PE characteristics
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_characteristics() const
{
	return nt_headers_.FileHeader.Characteristics;
}

//Returns checksum of PE file from header
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_checksum() const
{
	return nt_headers_.OptionalHeader.CheckSum;
}

//Sets checksum of PE file
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_checksum(uint32_t checksum)
{
	nt_headers_.OptionalHeader.CheckSum = checksum;
}

//Returns DLL Characteristics
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_dll_characteristics() const
{
	return nt_headers_.OptionalHeader.DllCharacteristics;
}

//Returns timestamp of PE file from header
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_time_date_stamp() const
{
	return nt_headers_.FileHeader.TimeDateStamp;
}

//Sets timestamp of PE file
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_time_date_stamp(uint32_t timestamp)
{
	nt_headers_.FileHeader.TimeDateStamp = timestamp;
}

//Sets DLL Characteristics
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_dll_characteristics(uint16_t characteristics)
{
	nt_headers_.OptionalHeader.DllCharacteristics = characteristics;
}

//Returns Machine field value of PE file from header
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_machine() const
{
	return nt_headers_.FileHeader.Machine;
}

//Sets Machine field value of PE file
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_machine(uint16_t machine)
{
	nt_headers_.FileHeader.Machine = machine;
}

//Sets PE characteristics
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_characteristics(uint16_t ch)
{
	nt_headers_.FileHeader.Characteristics = ch;
}

//Returns size of headers
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_size_of_headers() const
{
	return nt_headers_.OptionalHeader.SizeOfHeaders;
}

//Returns subsystem
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_subsystem() const
{
	return nt_headers_.OptionalHeader.Subsystem;
}

//Sets subsystem
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_subsystem(uint16_t subsystem)
{
	nt_headers_.OptionalHeader.Subsystem = subsystem;
}

//Returns size of optional header
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_size_of_optional_header() const
{
	return nt_headers_.FileHeader.SizeOfOptionalHeader;
}

//Returns PE signature
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_pe_signature() const
{
	return nt_headers_.Signature;
}

//Returns PE magic value
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_magic() const
{
	return nt_headers_.OptionalHeader.Magic;
}

//Sets required operation system version
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_os_version(uint16_t major, uint16_t minor)
{
	nt_headers_.OptionalHeader.MinorOperatingSystemVersion = minor;
	nt_headers_.OptionalHeader.MajorOperatingSystemVersion = major;
}

//Returns required operation system version (minor word)
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_minor_os_version() const
{
	return nt_headers_.OptionalHeader.MinorOperatingSystemVersion;
}

//Returns required operation system version (major word)
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_major_os_version() const
{
	return nt_headers_.OptionalHeader.MajorOperatingSystemVersion;
}

//Sets required subsystem version
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_subsystem_version(uint16_t major, uint16_t minor)
{
	nt_headers_.OptionalHeader.MinorSubsystemVersion = minor;
	nt_headers_.OptionalHeader.MajorSubsystemVersion = major;
}

//Returns required subsystem version (minor word)
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_minor_subsystem_version() const
{
	return nt_headers_.OptionalHeader.MinorSubsystemVersion;
}

//Returns required subsystem version (major word)
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_major_subsystem_version() const
{
	return nt_headers_.OptionalHeader.MajorSubsystemVersion;
}

//Virtual Address (VA) to Relative Virtual Address (RVA) convertions for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::va_to_rva(uint32_t va, bool bound_check) const
{
	if(bound_check && static_cast<uint64_t>(va) - nt_headers_.OptionalHeader.ImageBase > pe_utils::max_dword)
		throw pe_exception("Incorrect address conversion", pe_exception::incorrect_address_conversion);

	return static_cast<uint32_t>(va - nt_headers_.OptionalHeader.ImageBase);
}

//Virtual Address (VA) to Relative Virtual Address (RVA) convertions for PE32/PE64
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::va_to_rva(uint64_t va, bool bound_check) const
{
	if(bound_check && va - nt_headers_.OptionalHeader.ImageBase > pe_utils::max_dword)
		throw pe_exception("Incorrect address conversion", pe_exception::incorrect_address_conversion);

	return static_cast<uint32_t>(va - nt_headers_.OptionalHeader.ImageBase);
}

//Relative Virtual Address (RVA) to Virtual Address (VA) convertions for PE32
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::rva_to_va_32(uint32_t rva) const
{
	if(!pe_utils::is_sum_safe(rva, static_cast<uint32_t>(nt_headers_.OptionalHeader.ImageBase)))
		throw pe_exception("Incorrect address conversion", pe_exception::incorrect_address_conversion);

	return static_cast<uint32_t>(rva + nt_headers_.OptionalHeader.ImageBase);
}

//Relative Virtual Address (RVA) to Virtual Address (VA) convertions for PE32/PE64
template<typename PEClassType>
uint64_t pe_properties_generic<PEClassType>::rva_to_va_64(uint32_t rva) const
{
	return static_cast<uint64_t>(rva) + nt_headers_.OptionalHeader.ImageBase;
}

//Returns number of sections
template<typename PEClassType>
uint16_t pe_properties_generic<PEClassType>::get_number_of_sections() const
{
	return nt_headers_.FileHeader.NumberOfSections;
}

//Sets number of sections
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_number_of_sections(uint16_t number)
{
	nt_headers_.FileHeader.NumberOfSections = number;
}

//Sets virtual size of image
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_size_of_image(uint32_t size)
{
	nt_headers_.OptionalHeader.SizeOfImage = size;
}

//Sets size of headers
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_size_of_headers(uint32_t size)
{
	nt_headers_.OptionalHeader.SizeOfHeaders = size;
}

//Sets size of optional headers
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_size_of_optional_header(uint16_t size)
{
	nt_headers_.FileHeader.SizeOfOptionalHeader = size;
}

//Returns nt headers data pointer
template<typename PEClassType>
char* pe_properties_generic<PEClassType>::get_nt_headers_ptr()
{
	return reinterpret_cast<char*>(&nt_headers_);
}

//Returns nt headers data pointer
template<typename PEClassType>
const char* pe_properties_generic<PEClassType>::get_nt_headers_ptr() const
{
	return reinterpret_cast<const char*>(&nt_headers_);
}

//Returns size of NT header
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_sizeof_nt_header() const
{
	return sizeof(typename PEClassType::NtHeaders);
}

//Returns size of optional headers
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_sizeof_opt_headers() const
{
	return sizeof(typename PEClassType::OptHeaders);
}

//Sets file alignment (no checks)
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_file_alignment_unchecked(uint32_t alignment) 
{
	nt_headers_.OptionalHeader.FileAlignment = alignment;
}

//Sets base of code
template<typename PEClassType>
void pe_properties_generic<PEClassType>::set_base_of_code(uint32_t base)
{
	nt_headers_.OptionalHeader.BaseOfCode = base;
}

//Returns base of code
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_base_of_code() const
{
	return nt_headers_.OptionalHeader.BaseOfCode;
}

//Returns needed PE magic for PE or PE+ (from template parameters)
template<typename PEClassType>
uint32_t pe_properties_generic<PEClassType>::get_needed_magic() const
{
	return PEClassType::Id;
}

//Returns PE type of this image
template<typename PEClassType>
pe_type pe_properties_generic<PEClassType>::get_pe_type() const
{
	return PEClassType::Id == image_nt_optional_hdr32_magic ? pe_type_32 : pe_type_64;
}

//Two used instantiations for PE32 (PE) and PE64 (PE+)
template class pe_properties_generic<pe_types_class_32>;
template class pe_properties_generic<pe_types_class_64>;
}
