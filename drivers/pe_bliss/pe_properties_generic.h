#include "pe_properties.h"

namespace pe_bliss
{
//Helper class to reduce code size and ease its editing
template<
	typename NtHeadersType,
	typename OptHeadersType,
	uint16_t IdVal,
	typename BaseSizeType,
	BaseSizeType ImportSnapFlagVal,
	typename TLSStructType,
	typename ConfigStructType>
class pe_types
{
public:
	typedef NtHeadersType NtHeaders; //NT HEADERS type
	typedef OptHeadersType OptHeaders; //NT OPTIONAL HEADER type
	typedef BaseSizeType BaseSize; //Base size of different values: DWORD or ULONGLONG
	typedef TLSStructType TLSStruct; //TLS structure type
	typedef ConfigStructType ConfigStruct; //Configuration structure type

	static const uint16_t Id = IdVal; //Magic of PE or PE+
	static const BaseSize ImportSnapFlag = ImportSnapFlagVal; //Import snap flag value
};

//Portable Executable derived class for PE and PE+
//Describes PE/PE+ dependent things
template<typename PEClassType>
class pe_properties_generic : public pe_properties
{
public: //Constructor
	virtual std::auto_ptr<pe_properties> duplicate() const;

	//Fills properly PE structures
	virtual void create_pe(uint32_t section_alignment, uint16_t subsystem);

public:
	//Destructor
	virtual ~pe_properties_generic();


public: //DIRECTORIES
	//Returns true if directory exists
	virtual bool directory_exists(uint32_t id) const;

	//Removes directory
	virtual void remove_directory(uint32_t id);

	//Returns directory RVA
	virtual uint32_t get_directory_rva(uint32_t id) const;
	//Returns directory size
	virtual uint32_t get_directory_size(uint32_t id) const;

	//Sets directory RVA (just a value of PE header, no moving occurs)
	virtual void set_directory_rva(uint32_t id, uint32_t rva);
	//Sets directory size (just a value of PE header, no moving occurs)
	virtual void set_directory_size(uint32_t id, uint32_t size);
	
	//Strips only zero DATA_DIRECTORY entries to count = min_count
	//Returns resulting number of data directories
	//strip_iat_directory - if true, even not empty IAT directory will be stripped
	virtual uint32_t strip_data_directories(uint32_t min_count = 1, bool strip_iat_directory = true);


public: //IMAGE
	//Returns PE type of this image
	virtual pe_type get_pe_type() const;


public: //PE HEADER
	//Returns image base for PE32 and PE64 respectively
	virtual uint32_t get_image_base_32() const;
	virtual uint64_t get_image_base_64() const;

	//Sets new image base for PE32
	virtual void set_image_base(uint32_t base);
	//Sets new image base for PE32/PE+
	virtual void set_image_base_64(uint64_t base);

	//Returns image entry point
	virtual uint32_t get_ep() const;
	//Sets image entry point
	virtual void set_ep(uint32_t new_ep);

	//Returns file alignment
	virtual uint32_t get_file_alignment() const;
	//Returns section alignment
	virtual uint32_t get_section_alignment() const;

	//Sets heap size commit for PE32 and PE64 respectively
	virtual void set_heap_size_commit(uint32_t size);
	virtual void set_heap_size_commit(uint64_t size);
	//Sets heap size reserve for PE32 and PE64 respectively
	virtual void set_heap_size_reserve(uint32_t size);
	virtual void set_heap_size_reserve(uint64_t size);
	//Sets stack size commit for PE32 and PE64 respectively
	virtual void set_stack_size_commit(uint32_t size);
	virtual void set_stack_size_commit(uint64_t size);
	//Sets stack size reserve for PE32 and PE64 respectively
	virtual void set_stack_size_reserve(uint32_t size);
	virtual void set_stack_size_reserve(uint64_t size);
	
	//Returns heap size commit for PE32 and PE64 respectively
	virtual uint32_t get_heap_size_commit_32() const;
	virtual uint64_t get_heap_size_commit_64() const;
	//Returns heap size reserve for PE32 and PE64 respectively
	virtual uint32_t get_heap_size_reserve_32() const;
	virtual uint64_t get_heap_size_reserve_64() const;
	//Returns stack size commit for PE32 and PE64 respectively
	virtual uint32_t get_stack_size_commit_32() const;
	virtual uint64_t get_stack_size_commit_64() const;
	//Returns stack size reserve for PE32 and PE64 respectively
	virtual uint32_t get_stack_size_reserve_32() const;
	virtual uint64_t get_stack_size_reserve_64() const;

	//Returns virtual size of image
	virtual uint32_t get_size_of_image() const;

	//Returns number of RVA and sizes (number of DATA_DIRECTORY entries)
	virtual uint32_t get_number_of_rvas_and_sizes() const;
	//Sets number of RVA and sizes (number of DATA_DIRECTORY entries)
	virtual void set_number_of_rvas_and_sizes(uint32_t number);

	//Returns PE characteristics
	virtual uint16_t get_characteristics() const;
	//Sets PE characteristics
	virtual void set_characteristics(uint16_t ch);
	
	//Returns size of headers
	virtual uint32_t get_size_of_headers() const;

	//Returns subsystem
	virtual uint16_t get_subsystem() const;

	//Sets subsystem
	virtual void set_subsystem(uint16_t subsystem);

	//Returns size of optional header
	virtual uint16_t get_size_of_optional_header() const;

	//Returns PE signature
	virtual uint32_t get_pe_signature() const;

	//Returns PE magic value
	virtual uint32_t get_magic() const;

	//Returns checksum of PE file from header
	virtual uint32_t get_checksum() const;
	
	//Sets checksum of PE file
	virtual void set_checksum(uint32_t checksum);
	
	//Returns timestamp of PE file from header
	virtual uint32_t get_time_date_stamp() const;
	
	//Sets timestamp of PE file
	virtual void set_time_date_stamp(uint32_t timestamp);
	
	//Returns Machine field value of PE file from header
	virtual uint16_t get_machine() const;

	//Sets Machine field value of PE file
	virtual void set_machine(uint16_t machine);

	//Returns DLL Characteristics
	virtual uint16_t get_dll_characteristics() const;
	
	//Sets DLL Characteristics
	virtual void set_dll_characteristics(uint16_t characteristics);
	
	//Sets required operation system version
	virtual void set_os_version(uint16_t major, uint16_t minor);

	//Returns required operation system version (minor word)
	virtual uint16_t get_minor_os_version() const;

	//Returns required operation system version (major word)
	virtual uint16_t get_major_os_version() const;

	//Sets required subsystem version
	virtual void set_subsystem_version(uint16_t major, uint16_t minor);

	//Returns required subsystem version (minor word)
	virtual uint16_t get_minor_subsystem_version() const;

	//Returns required subsystem version (major word)
	virtual uint16_t get_major_subsystem_version() const;

public: //ADDRESS CONVERTIONS
	//Virtual Address (VA) to Relative Virtual Address (RVA) convertions
	//for PE32 and PE64 respectively
	//bound_check checks integer overflow
	virtual uint32_t va_to_rva(uint32_t va, bool bound_check = true) const;
	virtual uint32_t va_to_rva(uint64_t va, bool bound_check = true) const;
	
	//Relative Virtual Address (RVA) to Virtual Address (VA) convertions
	//for PE32 and PE64 respectively
	virtual uint32_t rva_to_va_32(uint32_t rva) const;
	virtual uint64_t rva_to_va_64(uint32_t rva) const;


public: //SECTIONS
	//Returns number of sections
	virtual uint16_t get_number_of_sections() const;

protected:
	typename PEClassType::NtHeaders nt_headers_; //NT headers (PE32 or PE64)
	
public:
	//Sets number of sections
	virtual void set_number_of_sections(uint16_t number);
	//Sets virtual size of image
	virtual void set_size_of_image(uint32_t size);
	//Sets size of headers
	virtual void set_size_of_headers(uint32_t size);
	//Sets size of optional headers
	virtual void set_size_of_optional_header(uint16_t size);
	//Returns nt headers data pointer
	virtual char* get_nt_headers_ptr();
	//Returns nt headers data pointer
	virtual const char* get_nt_headers_ptr() const;
	//Returns size of NT header
	virtual uint32_t get_sizeof_nt_header() const;
	//Returns size of optional headers
	virtual uint32_t get_sizeof_opt_headers() const;
	//Sets file alignment (no checks)
	virtual void set_file_alignment_unchecked(uint32_t alignment);
	//Sets base of code
	virtual void set_base_of_code(uint32_t base);
	//Returns base of code
	virtual uint32_t get_base_of_code() const;
	//Returns needed PE magic for PE or PE+ (from template parameters)
	virtual uint32_t get_needed_magic() const;
};

//Two used typedefs for PE32 (PE) and PE64 (PE+)
typedef pe_types<pe_win::image_nt_headers32,
	pe_win::image_optional_header32,
	pe_win::image_nt_optional_hdr32_magic,
	uint32_t,
	pe_win::image_ordinal_flag32,
	pe_win::image_tls_directory32,
	pe_win::image_load_config_directory32> pe_types_class_32;

typedef pe_types<pe_win::image_nt_headers64,
	pe_win::image_optional_header64,
	pe_win::image_nt_optional_hdr64_magic,
	uint64_t,
	pe_win::image_ordinal_flag64,
	pe_win::image_tls_directory64,
	pe_win::image_load_config_directory64> pe_types_class_64;

typedef pe_properties_generic<pe_types_class_32> pe_properties_32;
typedef pe_properties_generic<pe_types_class_64> pe_properties_64;
}
