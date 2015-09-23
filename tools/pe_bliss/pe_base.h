#pragma once
#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <map>
#include "pe_exception.h"
#include "pe_structures.h"
#include "utils.h"
#include "pe_section.h"
#include "pe_properties.h"

//Please don't remove this information from header
//PEBliss 1.0.0
//(c) DX 2011 - 2012, http://kaimi.ru
//Free to use for commertial and non-commertial purposes, modification and distribution

// == more important ==
//TODO: compact import rebuilder
//TODO: remove sections in the middle
//== less important ==
//TODO: relocations that take more than one element (seems to be not possible in Windows PE, but anyway)
//TODO: delay import directory
//TODO: write message tables
//TODO: write string tables
//TODO: read security information
//TODO: read full .NET information

namespace pe_bliss
{
//Portable executable class
class pe_base
{
public: //CONSTRUCTORS
	//Constructor from stream
	pe_base(std::istream& file, const pe_properties& props, bool read_debug_raw_data = true);

	//Constructor of empty PE-file
	explicit pe_base(const pe_properties& props, uint32_t section_alignment = 0x1000, bool dll = false, uint16_t subsystem = pe_win::image_subsystem_windows_gui);

	pe_base(const pe_base& pe);
	pe_base& operator=(const pe_base& pe);

public:
	~pe_base();

public: //STUB
	//Strips stub MSVS overlay, if any
	void strip_stub_overlay();
	//Fills stub MSVS overlay with specified byte
	void fill_stub_overlay(char c);
	//Sets stub MSVS overlay
	void set_stub_overlay(const std::string& data);
	//Returns stub overlay contents
	const std::string& get_stub_overlay() const;


public: //DIRECTORIES
	//Returns true if directory exists
	bool directory_exists(uint32_t id) const;
	//Removes directory
	void remove_directory(uint32_t id);

	//Returns directory RVA
	uint32_t get_directory_rva(uint32_t id) const;
	//Returns directory size
	uint32_t get_directory_size(uint32_t id) const;

	//Sets directory RVA (just a value of PE header, no moving occurs)
	void set_directory_rva(uint32_t id, uint32_t rva);
	//Sets directory size (just a value of PE header, no moving occurs)
	void set_directory_size(uint32_t id, uint32_t size);

	//Strips only zero DATA_DIRECTORY entries to count = min_count
	//Returns resulting number of data directories
	//strip_iat_directory - if true, even not empty IAT directory will be stripped
	uint32_t strip_data_directories(uint32_t min_count = 1, bool strip_iat_directory = true);

	//Returns true if image has import directory
	bool has_imports() const;
	//Returns true if image has export directory
	bool has_exports() const;
	//Returns true if image has resource directory
	bool has_resources() const;
	//Returns true if image has security directory
	bool has_security() const;
	//Returns true if image has relocations
	bool has_reloc() const;
	//Returns true if image has TLS directory
	bool has_tls() const;
	//Returns true if image has config directory
	bool has_config() const;
	//Returns true if image has bound import directory
	bool has_bound_import() const;
	//Returns true if image has delay import directory
	bool has_delay_import() const;
	//Returns true if image has COM directory
	bool is_dotnet() const;
	//Returns true if image has exception directory
	bool has_exception_directory() const;
	//Returns true if image has debug directory
	bool has_debug() const;

	//Returns subsystem value
	uint16_t get_subsystem() const;
	//Sets subsystem value
	void set_subsystem(uint16_t subsystem);
	//Returns true if image has console subsystem
	bool is_console() const;
	//Returns true if image has Windows GUI subsystem
	bool is_gui() const;

	//Sets required operation system version
	void set_os_version(uint16_t major, uint16_t minor);
	//Returns required operation system version (minor word)
	uint16_t get_minor_os_version() const;
	//Returns required operation system version (major word)
	uint16_t get_major_os_version() const;

	//Sets required subsystem version
	void set_subsystem_version(uint16_t major, uint16_t minor);
	//Returns required subsystem version (minor word)
	uint16_t get_minor_subsystem_version() const;
	//Returns required subsystem version (major word)
	uint16_t get_major_subsystem_version() const;

public: //PE HEADER
	//Returns DOS header
	const pe_win::image_dos_header& get_dos_header() const;
	pe_win::image_dos_header& get_dos_header();

	//Returns PE header start (e_lfanew)
	int32_t get_pe_header_start() const;

	//Returns file alignment
	uint32_t get_file_alignment() const;
	//Sets file alignment, checking the correctness of its value
	void set_file_alignment(uint32_t alignment);

	//Returns size of image
	uint32_t get_size_of_image() const;

	//Returns image entry point
	uint32_t get_ep() const;
	//Sets image entry point (just a value of PE header)
	void set_ep(uint32_t new_ep);

	//Returns number of RVA and sizes (number of DATA_DIRECTORY entries)
	uint32_t get_number_of_rvas_and_sizes() const;
	//Sets number of RVA and sizes (number of DATA_DIRECTORY entries)
	void set_number_of_rvas_and_sizes(uint32_t number);

	//Returns PE characteristics
	uint16_t get_characteristics() const;
	//Sets PE characteristics (a value inside header)
	void set_characteristics(uint16_t ch);
	//Clears PE characteristics flag
	void clear_characteristics_flags(uint16_t flags);
	//Sets PE characteristics flag
	void set_characteristics_flags(uint16_t flags);
	//Returns true if PE characteristics flag set
	bool check_characteristics_flag(uint16_t flag) const;
	
	//Returns DLL Characteristics
	uint16_t get_dll_characteristics() const;
	//Sets DLL Characteristics
	void set_dll_characteristics(uint16_t characteristics);

	//Returns size of headers
	uint32_t get_size_of_headers() const;
	//Returns size of optional header
	uint16_t get_size_of_optional_header() const;

	//Returns PE signature
	uint32_t get_pe_signature() const;

	//Returns magic value
	uint32_t get_magic() const;

	//Returns image base for PE32 and PE64 respectively
	uint32_t get_image_base_32() const;
	void get_image_base(uint32_t& base) const;
	//Sets image base for PE32 and PE64 respectively
	uint64_t get_image_base_64() const;
	void get_image_base(uint64_t& base) const;

	//Sets new image base
	void set_image_base(uint32_t base);
	void set_image_base_64(uint64_t base);

	//Sets heap size commit for PE32 and PE64 respectively
	void set_heap_size_commit(uint32_t size);
	void set_heap_size_commit(uint64_t size);
	//Sets heap size reserve for PE32 and PE64 respectively
	void set_heap_size_reserve(uint32_t size);
	void set_heap_size_reserve(uint64_t size);
	//Sets stack size commit for PE32 and PE64 respectively
	void set_stack_size_commit(uint32_t size);
	void set_stack_size_commit(uint64_t size);
	//Sets stack size reserve for PE32 and PE64 respectively
	void set_stack_size_reserve(uint32_t size);
	void set_stack_size_reserve(uint64_t size);

	//Returns heap size commit for PE32 and PE64 respectively
	uint32_t get_heap_size_commit_32() const;
	void get_heap_size_commit(uint32_t& size) const;
	uint64_t get_heap_size_commit_64() const;
	void get_heap_size_commit(uint64_t& size) const;
	//Returns heap size reserve for PE32 and PE64 respectively
	uint32_t get_heap_size_reserve_32() const;
	void get_heap_size_reserve(uint32_t& size) const;
	uint64_t get_heap_size_reserve_64() const;
	void get_heap_size_reserve(uint64_t& size) const;
	//Returns stack size commit for PE32 and PE64 respectively
	uint32_t get_stack_size_commit_32() const;
	void get_stack_size_commit(uint32_t& size) const;
	uint64_t get_stack_size_commit_64() const;
	void get_stack_size_commit(uint64_t& size) const;
	//Returns stack size reserve for PE32 and PE64 respectively
	uint32_t get_stack_size_reserve_32() const;
	void get_stack_size_reserve(uint32_t& size) const;
	uint64_t get_stack_size_reserve_64() const;
	void get_stack_size_reserve(uint64_t& size) const;

	//Updates virtual size of image corresponding to section virtual sizes
	void update_image_size();

	//Returns checksum of PE file from header
	uint32_t get_checksum() const;
	//Sets checksum of PE file
	void set_checksum(uint32_t checksum);
	
	//Returns timestamp of PE file from header
	uint32_t get_time_date_stamp() const;
	//Sets timestamp of PE file
	void set_time_date_stamp(uint32_t timestamp);
	
	//Returns Machine field value of PE file from header
	uint16_t get_machine() const;
	//Sets Machine field value of PE file
	void set_machine(uint16_t machine);

	//Returns data from the beginning of image
	//Size = SizeOfHeaders
	const std::string& get_full_headers_data() const;
	
	typedef std::multimap<uint32_t, std::string> debug_data_list;
	//Returns raw list of debug data
	const debug_data_list& get_raw_debug_data_list() const;
	
	//Reads and checks DOS header
	static void read_dos_header(std::istream& file, pe_win::image_dos_header& header);
	
	//Returns sizeof() nt headers
	uint32_t get_sizeof_nt_header() const;
	//Returns sizeof() optional headers
	uint32_t get_sizeof_opt_headers() const;
	//Returns raw nt headers data pointer
	const char* get_nt_headers_ptr() const;
	
	//Sets size of headers (to NT headers)
	void set_size_of_headers(uint32_t size);
	//Sets size of optional headers (to NT headers)
	void set_size_of_optional_header(uint16_t size);
	
	//Sets base of code
	void set_base_of_code(uint32_t base);
	//Returns base of code
	uint32_t get_base_of_code() const;

public: //ADDRESS CONVERTIONS
	//Virtual Address (VA) to Relative Virtual Address (RVA) convertions
	//for PE32 and PE64 respectively
	//bound_check checks integer overflow
	uint32_t va_to_rva(uint32_t va, bool bound_check = true) const;
	uint32_t va_to_rva(uint64_t va, bool bound_check = true) const;

	//Relative Virtual Address (RVA) to Virtual Address (VA) convertions
	//for PE32 and PE64 respectively
	uint32_t rva_to_va_32(uint32_t rva) const;
	void rva_to_va(uint32_t rva, uint32_t& va) const;
	uint64_t rva_to_va_64(uint32_t rva) const;
	void rva_to_va(uint32_t rva, uint64_t& va) const;

	//RVA to RAW file offset convertion (4gb max)
	uint32_t rva_to_file_offset(uint32_t rva) const;
	//RAW file offset to RVA convertion (4gb max)
	uint32_t file_offset_to_rva(uint32_t offset) const;

	//RVA from section raw data offset
	static uint32_t rva_from_section_offset(const section& s, uint32_t raw_offset_from_section_start);

public: //IMAGE SECTIONS
	//Returns number of sections from PE header
	uint16_t get_number_of_sections() const;

	//Updates number of sections in PE header
	uint16_t update_number_of_sections();

	//Returns section alignment
	uint32_t get_section_alignment() const;

	//Returns section list
	section_list& get_image_sections();
	const section_list& get_image_sections() const;

	//Realigns all sections, if you made any changes to sections or alignments
	void realign_all_sections();
	//Resligns section with specified index
	void realign_section(uint32_t index);

	//Returns section from RVA inside it
	section& section_from_rva(uint32_t rva);
	const section& section_from_rva(uint32_t rva) const;
	//Returns section from directory ID
	section& section_from_directory(uint32_t directory_id);
	const section& section_from_directory(uint32_t directory_id) const;
	//Returns section from VA inside it for PE32 and PE64 respectively
	section& section_from_va(uint32_t va);
	const section& section_from_va(uint32_t va) const;
	section& section_from_va(uint64_t va);
	const section& section_from_va(uint64_t va) const;
	//Returns section from file offset (4gb max)
	section& section_from_file_offset(uint32_t offset);
	const section& section_from_file_offset(uint32_t offset) const;

	//Returns section TOTAL RAW/VIRTUAL data length from RVA inside section
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	uint32_t section_data_length_from_rva(uint32_t rva, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	//Returns section TOTAL RAW/VIRTUAL data length from VA inside section for PE32 and PE64 respectively
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	uint32_t section_data_length_from_va(uint32_t va, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	uint32_t section_data_length_from_va(uint64_t va, section_data_type datatype = section_data_raw, bool include_headers = false) const;

	//Returns section remaining RAW/VIRTUAL data length from RVA to the end of section "s" (checks bounds)
	uint32_t section_data_length_from_rva(const section& s, uint32_t rva_inside, section_data_type datatype = section_data_raw) const;
	//Returns section remaining RAW/VIRTUAL data length from VA to the end of section "s" for PE32 and PE64 respectively (checks bounds)
	uint32_t section_data_length_from_va(const section& s, uint64_t va_inside, section_data_type datatype = section_data_raw) const;
	uint32_t section_data_length_from_va(const section& s, uint32_t va_inside, section_data_type datatype = section_data_raw) const;

	//Returns section remaining RAW/VIRTUAL data length from RVA "rva_inside" to the end of section containing RVA "rva"
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	uint32_t section_data_length_from_rva(uint32_t rva, uint32_t rva_inside, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	//Returns section remaining RAW/VIRTUAL data length from VA "va_inside" to the end of section containing VA "va" for PE32 and PE64 respectively
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	uint32_t section_data_length_from_va(uint32_t va, uint32_t va_inside, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	uint32_t section_data_length_from_va(uint64_t va, uint64_t va_inside, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	//Returns corresponding section data pointer from RVA inside section
	char* section_data_from_rva(uint32_t rva, bool include_headers = false);
	const char* section_data_from_rva(uint32_t rva, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	//Returns corresponding section data pointer from VA inside section for PE32 and PE64 respectively
	char* section_data_from_va(uint32_t va, bool include_headers = false);
	const char* section_data_from_va(uint32_t va, section_data_type datatype = section_data_raw, bool include_headers = false) const;
	char* section_data_from_va(uint64_t va, bool include_headers = false);
	const char* section_data_from_va(uint64_t va, section_data_type datatype = section_data_raw, bool include_headers = false) const;

	//Returns corresponding section data pointer from RVA inside section "s" (checks bounds)
	char* section_data_from_rva(section& s, uint32_t rva);
	const char* section_data_from_rva(const section& s, uint32_t rva, section_data_type datatype = section_data_raw) const;
	//Returns corresponding section data pointer from VA inside section "s" for PE32 and PE64 respectively (checks bounds)
	char* section_data_from_va(section& s, uint32_t va); //Always returns raw data
	const char* section_data_from_va(const section& s, uint32_t va, section_data_type datatype = section_data_raw) const;
	char* section_data_from_va(section& s, uint64_t va); //Always returns raw data
	const char* section_data_from_va(const section& s, uint64_t va, section_data_type datatype = section_data_raw) const;

	//Returns corresponding section data pointer from RVA inside section "s" (checks bounds, checks sizes, the most safe function)
	template<typename T>
	T section_data_from_rva(const section& s, uint32_t rva, section_data_type datatype = section_data_raw) const
	{
		if(rva >= s.get_virtual_address() && rva < s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()) && pe_utils::is_sum_safe(rva, sizeof(T)))
		{
			const std::string& data = datatype == section_data_raw ? s.get_raw_data() : s.get_virtual_data(get_section_alignment());
			//Don't check for underflow here, comparsion is unsigned
			if(data.size() < rva - s.get_virtual_address() + sizeof(T))
				throw pe_exception("RVA and requested data size does not exist inside section", pe_exception::rva_not_exists);

			return *reinterpret_cast<const T*>(data.data() + rva - s.get_virtual_address());
		}

		throw pe_exception("RVA not found inside section", pe_exception::rva_not_exists);
	}

	//Returns corresponding section data pointer from RVA inside section (checks rva, checks sizes, the most safe function)
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	template<typename T>
	T section_data_from_rva(uint32_t rva, section_data_type datatype = section_data_raw, bool include_headers = false) const
	{
		//if RVA is inside of headers and we're searching them too...
		if(include_headers && pe_utils::is_sum_safe(rva, sizeof(T)) && (rva + sizeof(T) < full_headers_data_.length()))
			return *reinterpret_cast<const T*>(&full_headers_data_[rva]);

		const section& s = section_from_rva(rva);
		const std::string& data = datatype == section_data_raw ? s.get_raw_data() : s.get_virtual_data(get_section_alignment());
		//Don't check for underflow here, comparsion is unsigned
		if(data.size() < rva - s.get_virtual_address() + sizeof(T))
			throw pe_exception("RVA and requested data size does not exist inside section", pe_exception::rva_not_exists);

		return *reinterpret_cast<const T*>(data.data() + rva - s.get_virtual_address());
	}

	//Returns corresponding section data pointer from VA inside section "s" (checks bounds, checks sizes, the most safe function)
	template<typename T>
	T section_data_from_va(const section& s, uint32_t va, section_data_type datatype = section_data_raw) const
	{
		return section_data_from_rva<T>(s, va_to_rva(va), datatype);
	}

	template<typename T>
	T section_data_from_va(const section& s, uint64_t va, section_data_type datatype = section_data_raw) const
	{
		return section_data_from_rva<T>(s, va_to_rva(va), datatype);
	}

	//Returns corresponding section data pointer from VA inside section (checks rva, checks sizes, the most safe function)
	//If include_headers = true, data from the beginning of PE file to SizeOfHeaders will be searched, too
	template<typename T>
	T section_data_from_va(uint32_t va, section_data_type datatype = section_data_raw, bool include_headers = false) const
	{
		return section_data_from_rva<T>(va_to_rva(va), datatype, include_headers);
	}

	template<typename T>
	T section_data_from_va(uint64_t va, section_data_type datatype = section_data_raw, bool include_headers = false) const
	{
		return section_data_from_rva<T>(va_to_rva(va), datatype, include_headers);
	}

	//Returns section and offset (raw data only) from its start from RVA
	const std::pair<uint32_t, const section*> section_and_offset_from_rva(uint32_t rva) const;

	//Sets virtual size of section "s"
	//Section must be free (not bound to any image)
	//or the last section of this image
	//Function calls update_image_size automatically in second case
	void set_section_virtual_size(section& s, uint32_t vsize);

	//Represents section expand type for expand_section function
	enum section_expand_type
	{
		expand_section_raw, //Section raw data size will be expanded
		expand_section_virtual //Section virtual data size will be expanded
	};

	//Expands section raw or virtual size to hold data from specified RVA with specified size
	//Section must be free (not bound to any image)
	//or the last section of this image
	//Returns true if section was expanded
	bool expand_section(section& s, uint32_t needed_rva, uint32_t needed_size, section_expand_type expand);

	//Adds section to image
	//Returns last section
	section& add_section(section s);
	//Prepares section to later add it to image (checks and recalculates virtual and raw section size)
	//Section must be prepared by this function before calling add_section
	void prepare_section(section& s);

	//Returns true if sectios "s" is already attached to this PE file
	bool section_attached(const section& s) const;


public: //IMAGE
	//Returns PE type (PE or PE+) from pe_type enumeration (minimal correctness checks)
	static pe_type get_pe_type(std::istream& file);
	//Returns PE type of this image
	pe_type get_pe_type() const;

	//Returns true if image has overlay data at the end of file
	bool has_overlay() const;

	//Realigns file (changes file alignment)
	void realign_file(uint32_t new_file_alignment);
	
	//Helper function to recalculate RAW and virtual section sizes and strip it, if necessary
	//auto_strip = strip section, if necessary
	void recalculate_section_sizes(section& s, bool auto_strip);

	// ========== END OF PUBLIC MEMBERS AND STRUCTURES ========== //
private:
	//Image DOS header
	pe_win::image_dos_header dos_header_;
	//Rich (stub) overlay data (for MSVS)
	std::string rich_overlay_;
	//List of image sections
	section_list sections_;
	//True if image has overlay
	bool has_overlay_;
	//Raw SizeOfHeaders-sized data from the beginning of image
	std::string full_headers_data_;
	//Raw debug data for all directories
	//PointerToRawData; Data
	debug_data_list debug_data_;
	//PE or PE+ related properties
	pe_properties* props_;

	//Reads and checks DOS header
	void read_dos_header(std::istream& file);

	//Reads and checks PE headers and section headers, data
	void read_pe(std::istream& file, bool read_debug_raw_data);

	//Sets number of sections
	void set_number_of_sections(uint16_t number);
	//Sets size of image
	void set_size_of_image(uint32_t size);
	//Sets file alignment (no checks)
	void set_file_alignment_unchecked(uint32_t alignment);
	//Returns needed magic of image
	uint32_t get_needed_magic() const;
	//Returns nt headers data pointer
	char* get_nt_headers_ptr();

private:
	static const uint16_t maximum_number_of_sections = 0x60;
	static const uint32_t minimum_file_alignment = 512;

private:
	//RAW file offset to section convertion helpers (4gb max)
	section_list::const_iterator file_offset_to_section(uint32_t offset) const;
	section_list::iterator file_offset_to_section(uint32_t offset);
};
}
