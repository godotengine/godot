#pragma once
#include <memory>
#include "pe_structures.h"

namespace pe_bliss
{
class pe_properties
{
public: //Constructors
	virtual std::auto_ptr<pe_properties> duplicate() const = 0;
	
	//Fills properly PE structures
	virtual void create_pe(uint32_t section_alignment, uint16_t subsystem) = 0;

public:
	//Destructor
	virtual ~pe_properties();


public: //DIRECTORIES
	//Returns true if directory exists
	virtual bool directory_exists(uint32_t id) const = 0;

	//Removes directory
	virtual void remove_directory(uint32_t id) = 0;

	//Returns directory RVA
	virtual uint32_t get_directory_rva(uint32_t id) const = 0;
	//Returns directory size
	virtual uint32_t get_directory_size(uint32_t id) const = 0;

	//Sets directory RVA (just a value of PE header, no moving occurs)
	virtual void set_directory_rva(uint32_t id, uint32_t rva) = 0;
	//Sets directory size (just a value of PE header, no moving occurs)
	virtual void set_directory_size(uint32_t id, uint32_t size) = 0;
	
	//Strips only zero DATA_DIRECTORY entries to count = min_count
	//Returns resulting number of data directories
	//strip_iat_directory - if true, even not empty IAT directory will be stripped
	virtual uint32_t strip_data_directories(uint32_t min_count = 1, bool strip_iat_directory = true) = 0;


public: //IMAGE
	//Returns PE type of this image
	virtual pe_type get_pe_type() const = 0;


public: //PE HEADER
	//Returns image base for PE32 and PE64 respectively
	virtual uint32_t get_image_base_32() const = 0;
	virtual uint64_t get_image_base_64() const = 0;

	//Sets new image base for PE32
	virtual void set_image_base(uint32_t base) = 0;
	//Sets new image base for PE32/PE+
	virtual void set_image_base_64(uint64_t base) = 0;

	//Returns image entry point
	virtual uint32_t get_ep() const = 0;
	//Sets image entry point
	virtual void set_ep(uint32_t new_ep) = 0;

	//Returns file alignment
	virtual uint32_t get_file_alignment() const = 0;
	//Returns section alignment
	virtual uint32_t get_section_alignment() const = 0;

	//Sets heap size commit for PE32 and PE64 respectively
	virtual void set_heap_size_commit(uint32_t size) = 0;
	virtual void set_heap_size_commit(uint64_t size) = 0;
	//Sets heap size reserve for PE32 and PE64 respectively
	virtual void set_heap_size_reserve(uint32_t size) = 0;
	virtual void set_heap_size_reserve(uint64_t size) = 0;
	//Sets stack size commit for PE32 and PE64 respectively
	virtual void set_stack_size_commit(uint32_t size) = 0;
	virtual void set_stack_size_commit(uint64_t size) = 0;
	//Sets stack size reserve for PE32 and PE64 respectively
	virtual void set_stack_size_reserve(uint32_t size) = 0;
	virtual void set_stack_size_reserve(uint64_t size) = 0;
	
	//Returns heap size commit for PE32 and PE64 respectively
	virtual uint32_t get_heap_size_commit_32() const = 0;
	virtual uint64_t get_heap_size_commit_64() const = 0;
	//Returns heap size reserve for PE32 and PE64 respectively
	virtual uint32_t get_heap_size_reserve_32() const = 0;
	virtual uint64_t get_heap_size_reserve_64() const = 0;
	//Returns stack size commit for PE32 and PE64 respectively
	virtual uint32_t get_stack_size_commit_32() const = 0;
	virtual uint64_t get_stack_size_commit_64() const = 0;
	//Returns stack size reserve for PE32 and PE64 respectively
	virtual uint32_t get_stack_size_reserve_32() const = 0;
	virtual uint64_t get_stack_size_reserve_64() const = 0;

	//Returns virtual size of image
	virtual uint32_t get_size_of_image() const = 0;

	//Returns number of RVA and sizes (number of DATA_DIRECTORY entries)
	virtual uint32_t get_number_of_rvas_and_sizes() const = 0;
	//Sets number of RVA and sizes (number of DATA_DIRECTORY entries)
	virtual void set_number_of_rvas_and_sizes(uint32_t number) = 0;

	//Returns PE characteristics
	virtual uint16_t get_characteristics() const = 0;
	//Sets PE characteristics
	virtual void set_characteristics(uint16_t ch) = 0;
	
	//Clears PE characteristics flag
	void clear_characteristics_flags(uint16_t flags);
	//Sets PE characteristics flag
	void set_characteristics_flags(uint16_t flags);

	//Returns size of headers
	virtual uint32_t get_size_of_headers() const = 0;

	//Returns subsystem
	virtual uint16_t get_subsystem() const = 0;

	//Sets subsystem
	virtual void set_subsystem(uint16_t subsystem) = 0;

	//Returns size of optional header
	virtual uint16_t get_size_of_optional_header() const = 0;

	//Returns PE signature
	virtual uint32_t get_pe_signature() const = 0;

	//Returns PE magic value
	virtual uint32_t get_magic() const = 0;

	//Returns checksum of PE file from header
	virtual uint32_t get_checksum() const = 0;
	
	//Sets checksum of PE file
	virtual void set_checksum(uint32_t checksum) = 0;
	
	//Returns timestamp of PE file from header
	virtual uint32_t get_time_date_stamp() const = 0;
	
	//Sets timestamp of PE file
	virtual void set_time_date_stamp(uint32_t timestamp) = 0;
	
	//Returns Machine field value of PE file from header
	virtual uint16_t get_machine() const = 0;

	//Sets Machine field value of PE file
	virtual void set_machine(uint16_t machine) = 0;

	//Returns DLL Characteristics
	virtual uint16_t get_dll_characteristics() const = 0;
	
	//Sets DLL Characteristics
	virtual void set_dll_characteristics(uint16_t characteristics) = 0;
	
	//Sets required operation system version
	virtual void set_os_version(uint16_t major, uint16_t minor) = 0;

	//Returns required operation system version (minor word)
	virtual uint16_t get_minor_os_version() const = 0;

	//Returns required operation system version (major word)
	virtual uint16_t get_major_os_version() const = 0;

	//Sets required subsystem version
	virtual void set_subsystem_version(uint16_t major, uint16_t minor) = 0;

	//Returns required subsystem version (minor word)
	virtual uint16_t get_minor_subsystem_version() const = 0;

	//Returns required subsystem version (major word)
	virtual uint16_t get_major_subsystem_version() const = 0;

public: //ADDRESS CONVERTIONS
	//Virtual Address (VA) to Relative Virtual Address (RVA) convertions
	//for PE32 and PE64 respectively
	//bound_check checks integer overflow
	virtual uint32_t va_to_rva(uint32_t va, bool bound_check = true) const = 0;
	virtual uint32_t va_to_rva(uint64_t va, bool bound_check = true) const = 0;
	
	//Relative Virtual Address (RVA) to Virtual Address (VA) convertions
	//for PE32 and PE64 respectively
	virtual uint32_t rva_to_va_32(uint32_t rva) const = 0;
	virtual uint64_t rva_to_va_64(uint32_t rva) const = 0;


public: //SECTIONS
	//Returns number of sections
	virtual uint16_t get_number_of_sections() const = 0;
	
public:
	//Sets number of sections
	virtual void set_number_of_sections(uint16_t number) = 0;
	//Sets virtual size of image
	virtual void set_size_of_image(uint32_t size) = 0;
	//Sets size of headers
	virtual void set_size_of_headers(uint32_t size) = 0;
	//Sets size of optional headers
	virtual void set_size_of_optional_header(uint16_t size) = 0;
	//Returns nt headers data pointer
	virtual char* get_nt_headers_ptr() = 0;
	//Returns nt headers data pointer
	virtual const char* get_nt_headers_ptr() const = 0;
	//Returns size of NT header
	virtual uint32_t get_sizeof_nt_header() const = 0;
	//Returns size of optional headers
	virtual uint32_t get_sizeof_opt_headers() const = 0;
	//Sets file alignment (no checks)
	virtual void set_file_alignment_unchecked(uint32_t alignment) = 0;
	//Sets base of code
	virtual void set_base_of_code(uint32_t base) = 0;
	//Returns base of code
	virtual uint32_t get_base_of_code() const = 0;
	//Returns needed PE magic for PE or PE+ (from template parameters)
	virtual uint32_t get_needed_magic() const = 0;
};
}
