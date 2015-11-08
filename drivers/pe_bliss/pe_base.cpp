#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <string.h>
#include "pe_exception.h"
#include "pe_base.h"

namespace pe_bliss
{
using namespace pe_win;

//Constructor
pe_base::pe_base(std::istream& file, const pe_properties& props, bool read_debug_raw_data)
{
	props_ = props.duplicate().release();

	//Save istream state
	std::ios_base::iostate state = file.exceptions();
	std::streamoff old_offset = file.tellg();

	try
	{
		file.exceptions(std::ios::goodbit);
		//Read DOS header, PE headers and section data
		read_dos_header(file);
		read_pe(file, read_debug_raw_data);
	}
	catch(const std::exception&)
	{
		//If something went wrong, restore istream state
		file.seekg(old_offset);
		file.exceptions(state);
		file.clear();
		//Rethrow
		throw;
	}

	//Restore istream state
	file.seekg(old_offset);
	file.exceptions(state);
	file.clear();
}

pe_base::pe_base(const pe_properties& props, uint32_t section_alignment, bool dll, uint16_t subsystem)
{
	props_ = props.duplicate().release();
	props_->create_pe(section_alignment, subsystem);

	has_overlay_ = false;
	memset(&dos_header_, 0, sizeof(dos_header_));

	dos_header_.e_magic = 0x5A4D; //"MZ"
	//Magic numbers from MSVC++ build
	dos_header_.e_maxalloc = 0xFFFF;
	dos_header_.e_cblp = 0x90;
	dos_header_.e_cp = 3;
	dos_header_.e_cparhdr = 4;
	dos_header_.e_sp = 0xB8;
	dos_header_.e_lfarlc = 64;

	set_characteristics(image_file_executable_image | image_file_relocs_stripped);

	if(get_pe_type() == pe_type_32)
		set_characteristics_flags(image_file_32bit_machine);

	if(dll)
		set_characteristics_flags(image_file_dll);

	set_subsystem_version(5, 1); //WinXP
	set_os_version(5, 1); //WinXP
}

pe_base::pe_base(const pe_base& pe)
	:dos_header_(pe.dos_header_),
	rich_overlay_(pe.rich_overlay_),
	sections_(pe.sections_),
	has_overlay_(pe.has_overlay_),
	full_headers_data_(pe.full_headers_data_),
	debug_data_(pe.debug_data_),
	props_(0)
{
	props_ = pe.props_->duplicate().release();
}

pe_base& pe_base::operator=(const pe_base& pe)
{
	dos_header_ = pe.dos_header_;
	rich_overlay_ = pe.rich_overlay_;
	sections_ = pe.sections_;
	has_overlay_ = pe.has_overlay_;
	full_headers_data_ = pe.full_headers_data_;
	debug_data_ = pe.debug_data_;
	delete props_;
	props_ = 0;
	props_ = pe.props_->duplicate().release();

	return *this;
}

pe_base::~pe_base()
{
	delete props_;
}

//Returns dos header
const image_dos_header& pe_base::get_dos_header() const
{
	return dos_header_;
}

//Returns dos header
image_dos_header& pe_base::get_dos_header()
{
	return dos_header_;
}

//Returns PE headers start position (e_lfanew)
int32_t pe_base::get_pe_header_start() const
{
	return dos_header_.e_lfanew;
}

//Strips MSVC stub overlay
void pe_base::strip_stub_overlay()
{
	rich_overlay_.clear();
}

//Fills MSVC stub overlay with character c
void pe_base::fill_stub_overlay(char c)
{
	if(rich_overlay_.length())
		rich_overlay_.assign(rich_overlay_.length(), c);
}

//Sets stub MSVS overlay
void pe_base::set_stub_overlay(const std::string& data)
{
	rich_overlay_ = data;
}

//Returns stub overlay
const std::string& pe_base::get_stub_overlay() const
{
	return rich_overlay_;
}

//Realigns all sections
void pe_base::realign_all_sections()
{
	for(unsigned int i = 0; i < sections_.size(); i++)
		realign_section(i);
}

//Returns number of sections from PE header
uint16_t pe_base::get_number_of_sections() const
{
	return props_->get_number_of_sections();
}

//Updates number of sections in PE header
uint16_t pe_base::update_number_of_sections()
{
	uint16_t new_number = static_cast<uint16_t>(sections_.size());
	props_->set_number_of_sections(new_number);
	return new_number;
}

//Returns section alignment
uint32_t pe_base::get_section_alignment() const
{
	return props_->get_section_alignment();
}

//Returns image sections list
section_list& pe_base::get_image_sections()
{
	return sections_;
}

//Returns image sections list
const section_list& pe_base::get_image_sections() const
{
	return sections_;
}

//Realigns section by index
void pe_base::realign_section(uint32_t index)
{
	//Check index
	if(sections_.size() <= index)
		throw pe_exception("Section not found", pe_exception::section_not_found);

	//Get section iterator
	section_list::iterator it = sections_.begin() + index;
	section& s = *it;

	//Calculate, how many null bytes we have in the end of raw section data
	std::size_t strip = 0;
	for(std::size_t i = (*it).get_raw_data().length(); i >= 1; --i)
	{
		if(s.get_raw_data()[i - 1] == 0)
			strip++;
		else
			break;
	}

	if(it == sections_.end() - 1) //If we're realigning the last section
	{
		//We can strip ending null bytes
		s.set_size_of_raw_data(static_cast<uint32_t>(s.get_raw_data().length() - strip));
		s.get_raw_data().resize(s.get_raw_data().length() - strip, 0);
	}
	else
	{
		//Else just set size of raw data
		uint32_t raw_size_aligned = s.get_aligned_raw_size(get_file_alignment());
		s.set_size_of_raw_data(raw_size_aligned);
		s.get_raw_data().resize(raw_size_aligned, 0);
	}
}

//Returns file alignment
uint32_t pe_base::get_file_alignment() const
{
	return props_->get_file_alignment();
}

//Sets file alignment
void pe_base::set_file_alignment(uint32_t alignment)
{
	//Check alignment
	if(alignment < minimum_file_alignment)
		throw pe_exception("File alignment can't be less than 512", pe_exception::incorrect_file_alignment);

	if(!pe_utils::is_power_of_2(alignment))
		throw pe_exception("File alignment must be a power of 2", pe_exception::incorrect_file_alignment);

	if(alignment > get_section_alignment())
		throw pe_exception("File alignment must be <= section alignment", pe_exception::incorrect_file_alignment);

	//Set file alignment without any additional checks
	set_file_alignment_unchecked(alignment);
}

//Returns size of image
uint32_t pe_base::get_size_of_image() const
{
	return props_->get_size_of_image();
}

//Returns image entry point
uint32_t pe_base::get_ep() const
{
	return props_->get_ep();
}

//Sets image entry point (just a value of PE header)
void pe_base::set_ep(uint32_t new_ep)
{
	props_->set_ep(new_ep);
}

//Returns number of RVA and sizes (number of DATA_DIRECTORY entries)
uint32_t pe_base::get_number_of_rvas_and_sizes() const
{
	return props_->get_number_of_rvas_and_sizes();
}

//Sets number of RVA and sizes (number of DATA_DIRECTORY entries)
void pe_base::set_number_of_rvas_and_sizes(uint32_t number)
{
	props_->set_number_of_rvas_and_sizes(number);
}

//Returns PE characteristics
uint16_t pe_base::get_characteristics() const
{
	return props_->get_characteristics();
}

//Sets PE characteristics (a value inside header)
void pe_base::set_characteristics(uint16_t ch)
{
	props_->set_characteristics(ch);
}

//Returns section from RVA
section& pe_base::section_from_rva(uint32_t rva)
{
	//Search for section
	for(section_list::iterator i = sections_.begin(); i != sections_.end(); ++i)
	{
		section& s = *i;
		//Return section if found
		if(rva >= s.get_virtual_address() && rva < s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()))
			return s;
	}

	throw pe_exception("No section found by presented address", pe_exception::no_section_found);
}

//Returns section from RVA
const section& pe_base::section_from_rva(uint32_t rva) const
{
	//Search for section
	for(section_list::const_iterator i = sections_.begin(); i != sections_.end(); ++i)
	{
		const section& s = *i;
		//Return section if found
		if(rva >= s.get_virtual_address() && rva < s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()))
			return s;
	}

	throw pe_exception("No section found by presented address", pe_exception::no_section_found);
}

//Returns section from directory ID
section& pe_base::section_from_directory(uint32_t directory_id)
{
	return section_from_rva(get_directory_rva(directory_id));		
}

//Returns section from directory ID
const section& pe_base::section_from_directory(uint32_t directory_id) const
{
	return section_from_rva(get_directory_rva(directory_id));	
}

//Sets section virtual size (actual for the last one of this PE or for unbound section)
void pe_base::set_section_virtual_size(section& s, uint32_t vsize)
{
	//Check if we're changing virtual size of the last section
	//Of course, we can change virtual size of section that's not bound to this PE file
	if(sections_.empty() || std::find_if(sections_.begin(), sections_.end() - 1, section_ptr_finder(s)) != sections_.end() - 1)
		throw pe_exception("Can't change virtual size of any section, except last one", pe_exception::error_changing_section_virtual_size);

	//If we're setting virtual size to zero
	if(vsize == 0)
	{
		//Check if section is empty
		if(s.empty())
			throw pe_exception("Cannot set virtual size of empty section to zero", pe_exception::error_changing_section_virtual_size);

		//Set virtual size equal to aligned size of raw data
		s.set_virtual_size(s.get_size_of_raw_data());
	}
	else
	{
		s.set_virtual_size(vsize);
	}

	//Update image size if we're changing virtual size for the last section of this PE
	if(!sections_.empty() || &s == &(*(sections_.end() - 1)))
		update_image_size();
}

//Expands section raw or virtual size to hold data from specified RVA with specified size
//Section must be free (not bound to any image)
//or the last section of this image
bool pe_base::expand_section(section& s, uint32_t needed_rva, uint32_t needed_size, section_expand_type expand)
{
	//Check if we're changing the last section
	//Of course, we can change the section that's not bound to this PE file
	if(sections_.empty() || std::find_if(sections_.begin(), sections_.end() - 1, section_ptr_finder(s)) != sections_.end() - 1)
		throw pe_exception("Can't expand any section, except last one", pe_exception::error_expanding_section);

	//Check if we should expand our section
	if(expand == expand_section_raw && section_data_length_from_rva(s, needed_rva, section_data_raw) < needed_size)
	{
		//Expand section raw data
		s.get_raw_data().resize(needed_rva - s.get_virtual_address() + needed_size);
		recalculate_section_sizes(s, false);
		return true;
	}
	else if(expand == expand_section_virtual && section_data_length_from_rva(s, needed_rva, section_data_virtual) < needed_size)
	{
		//Expand section virtual data
		set_section_virtual_size(s, needed_rva - s.get_virtual_address() + needed_size);
		return true;
	}
	
	return false;
}

//Updates image virtual size
void pe_base::update_image_size()
{
	//Write virtual size of image to headers
	if(!sections_.empty())
		set_size_of_image(sections_.back().get_virtual_address() + sections_.back().get_aligned_virtual_size(get_section_alignment()));
	else
		set_size_of_image(get_size_of_headers());
}

//Returns checksum of PE file from header
uint32_t pe_base::get_checksum() const
{
	return props_->get_checksum();
}

//Sets checksum of PE file
void pe_base::set_checksum(uint32_t checksum)
{
	props_->set_checksum(checksum);
}

//Returns timestamp of PE file from header
uint32_t pe_base::get_time_date_stamp() const
{
	return props_->get_time_date_stamp();
}

//Sets timestamp of PE file
void pe_base::set_time_date_stamp(uint32_t timestamp)
{
	props_->set_time_date_stamp(timestamp);
}

//Returns Machine field value of PE file from header
uint16_t pe_base::get_machine() const
{
	return props_->get_machine();
}

//Sets Machine field value of PE file
void pe_base::set_machine(uint16_t machine)
{
	props_->set_machine(machine);
}

//Prepares section before attaching it
void pe_base::prepare_section(section& s)
{
	//Calculate its size of raw data
	s.set_size_of_raw_data(static_cast<uint32_t>(pe_utils::align_up(s.get_raw_data().length(), get_file_alignment())));

	//Check section virtual and raw size
	if(!s.get_size_of_raw_data() && !s.get_virtual_size())
		throw pe_exception("Virtual and Physical sizes of section can't be 0 at the same time", pe_exception::zero_section_sizes);

	//If section virtual size is zero
	if(!s.get_virtual_size())
	{
		s.set_virtual_size(s.get_size_of_raw_data());
	}
	else
	{
		//Else calculate its virtual size
		s.set_virtual_size(
			std::max<uint32_t>(pe_utils::align_up(s.get_size_of_raw_data(), get_file_alignment()),
			pe_utils::align_up(s.get_virtual_size(), get_section_alignment())));
	}
}

//Adds section to image
section& pe_base::add_section(section s)
{
	if(sections_.size() >= maximum_number_of_sections)
		throw pe_exception("Maximum number of sections has been reached", pe_exception::no_more_sections_can_be_added);

	//Prepare section before adding it
	prepare_section(s);

	//Calculate section virtual address
	if(!sections_.empty())
	{
		s.set_virtual_address(pe_utils::align_up(sections_.back().get_virtual_address() + sections_.back().get_aligned_virtual_size(get_section_alignment()), get_section_alignment()));

		//We should align last section raw size, if it wasn't aligned
		section& last = sections_.back();
		last.set_size_of_raw_data(static_cast<uint32_t>(pe_utils::align_up(last.get_raw_data().length(), get_file_alignment())));
	}
	else
	{
		s.set_virtual_address(
			s.get_virtual_address() == 0
			? pe_utils::align_up(get_size_of_headers(), get_section_alignment())
			: pe_utils::align_up(s.get_virtual_address(), get_section_alignment()));
	}

	//Add section to the end of section list
	sections_.push_back(s);
	//Set number of sections in PE header
	set_number_of_sections(static_cast<uint16_t>(sections_.size()));
	//Recalculate virtual size of image
	set_size_of_image(get_size_of_image() + s.get_aligned_virtual_size(get_section_alignment()));
	//Return last section
	return sections_.back();
}

//Returns true if sectios "s" is already attached to this PE file
bool pe_base::section_attached(const section& s) const
{
	return sections_.end() != std::find_if(sections_.begin(), sections_.end(), section_ptr_finder(s));
}

//Returns true if directory exists
bool pe_base::directory_exists(uint32_t id) const
{
	return props_->directory_exists(id);
}

//Removes directory
void pe_base::remove_directory(uint32_t id)
{
	props_->remove_directory(id);
}

//Returns directory RVA
uint32_t pe_base::get_directory_rva(uint32_t id) const
{
	return props_->get_directory_rva(id);
}

//Returns directory size
uint32_t pe_base::get_directory_size(uint32_t id) const
{
	return props_->get_directory_size(id);
}

//Sets directory RVA (just a value of PE header, no moving occurs)
void pe_base::set_directory_rva(uint32_t id, uint32_t rva)
{
	return props_->set_directory_rva(id, rva);
}

//Sets directory size (just a value of PE header, no moving occurs)
void pe_base::set_directory_size(uint32_t id, uint32_t size)
{
	return props_->set_directory_size(id, size);
}

//Strips only zero DATA_DIRECTORY entries to count = min_count
//Returns resulting number of data directories
//strip_iat_directory - if true, even not empty IAT directory will be stripped
uint32_t pe_base::strip_data_directories(uint32_t min_count, bool strip_iat_directory)
{
	return props_->strip_data_directories(min_count, strip_iat_directory);
}

//Returns true if image has import directory
bool pe_base::has_imports() const
{
	return directory_exists(image_directory_entry_import);
}

//Returns true if image has export directory
bool pe_base::has_exports() const
{
	return directory_exists(image_directory_entry_export);
}

//Returns true if image has resource directory
bool pe_base::has_resources() const
{
	return directory_exists(image_directory_entry_resource);
}

//Returns true if image has security directory
bool pe_base::has_security() const
{
	return directory_exists(image_directory_entry_security);
}

//Returns true if image has relocations
bool pe_base::has_reloc() const
{
	return directory_exists(image_directory_entry_basereloc) && !(get_characteristics() & image_file_relocs_stripped);
}

//Returns true if image has TLS directory
bool pe_base::has_tls() const
{
	return directory_exists(image_directory_entry_tls);
}

//Returns true if image has config directory
bool pe_base::has_config() const
{
	return directory_exists(image_directory_entry_load_config);
}

//Returns true if image has bound import directory
bool pe_base::has_bound_import() const
{
	return directory_exists(image_directory_entry_bound_import);
}

//Returns true if image has delay import directory
bool pe_base::has_delay_import() const
{
	return directory_exists(image_directory_entry_delay_import);
}

//Returns true if image has COM directory
bool pe_base::is_dotnet() const
{
	return directory_exists(image_directory_entry_com_descriptor);
}

//Returns true if image has exception directory
bool pe_base::has_exception_directory() const
{
	return directory_exists(image_directory_entry_exception);
}

//Returns true if image has debug directory
bool pe_base::has_debug() const
{
	return directory_exists(image_directory_entry_debug);
}

//Returns corresponding section data pointer from RVA inside section "s" (checks bounds)
char* pe_base::section_data_from_rva(section& s, uint32_t rva)
{
	//Check if RVA is inside section "s"
	if(rva >= s.get_virtual_address() && rva < s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()))
	{
		if(s.get_raw_data().empty())
			throw pe_exception("Section raw data is empty and cannot be changed", pe_exception::section_is_empty);

		return &s.get_raw_data()[rva - s.get_virtual_address()];
	}

	throw pe_exception("RVA not found inside section", pe_exception::rva_not_exists);
}

//Returns corresponding section data pointer from RVA inside section "s" (checks bounds)
const char* pe_base::section_data_from_rva(const section& s, uint32_t rva, section_data_type datatype) const
{
	//Check if RVA is inside section "s"
	if(rva >= s.get_virtual_address() && rva < s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()))
		return (datatype == section_data_raw ? s.get_raw_data().data() : s.get_virtual_data(get_section_alignment()).c_str()) + rva - s.get_virtual_address();

	throw pe_exception("RVA not found inside section", pe_exception::rva_not_exists);
}

//Returns section TOTAL RAW/VIRTUAL data length from RVA inside section
uint32_t pe_base::section_data_length_from_rva(uint32_t rva, section_data_type datatype, bool include_headers) const
{
	//if RVA is inside of headers and we're searching them too...
	if(include_headers && rva < full_headers_data_.length())
		return static_cast<unsigned long>(full_headers_data_.length());

	const section& s = section_from_rva(rva);
	return static_cast<unsigned long>(datatype == section_data_raw ? s.get_raw_data().length() /* instead of SizeOfRawData */ : s.get_aligned_virtual_size(get_section_alignment()));
}

//Returns section TOTAL RAW/VIRTUAL data length from VA inside section for PE32
uint32_t pe_base::section_data_length_from_va(uint32_t va, section_data_type datatype, bool include_headers) const
{
	return section_data_length_from_rva(va_to_rva(va), datatype, include_headers);
}

//Returns section TOTAL RAW/VIRTUAL data length from VA inside section for PE32/PE64
uint32_t pe_base::section_data_length_from_va(uint64_t va, section_data_type datatype, bool include_headers) const
{
	return section_data_length_from_rva(va_to_rva(va), datatype, include_headers);
}

//Returns section remaining RAW/VIRTUAL data length from RVA "rva_inside" to the end of section containing RVA "rva"
uint32_t pe_base::section_data_length_from_rva(uint32_t rva, uint32_t rva_inside, section_data_type datatype, bool include_headers) const
{
	//if RVAs are inside of headers and we're searching them too...
	if(include_headers && rva < full_headers_data_.length() && rva_inside < full_headers_data_.length())
		return static_cast<unsigned long>(full_headers_data_.length() - rva_inside);

	const section& s = section_from_rva(rva);
	if(rva_inside < s.get_virtual_address())
		throw pe_exception("RVA not found inside section", pe_exception::rva_not_exists);

	//Calculate remaining length of section data from "rva" address
	long length = static_cast<long>(datatype == section_data_raw ? s.get_raw_data().length() /* instead of SizeOfRawData */ : s.get_aligned_virtual_size(get_section_alignment()))
		+ s.get_virtual_address() - rva_inside;

	if(length < 0)
		return 0;

	return static_cast<unsigned long>(length);
}

//Returns section remaining RAW/VIRTUAL data length from VA "va_inside" to the end of section containing VA "va" for PE32
uint32_t pe_base::section_data_length_from_va(uint32_t va, uint32_t va_inside, section_data_type datatype, bool include_headers) const
{
	return section_data_length_from_rva(va_to_rva(va), va_to_rva(va_inside), datatype, include_headers);
}

//Returns section remaining RAW/VIRTUAL data length from VA "va_inside" to the end of section containing VA "va" for PE32/PE64
uint32_t pe_base::section_data_length_from_va(uint64_t va, uint64_t va_inside, section_data_type datatype, bool include_headers) const
{
	return section_data_length_from_rva(va_to_rva(va), va_to_rva(va_inside), datatype, include_headers);
}

//Returns section remaining RAW/VIRTUAL data length from RVA to the end of section "s" (checks bounds)
uint32_t pe_base::section_data_length_from_rva(const section& s, uint32_t rva_inside, section_data_type datatype) const
{
	//Check rva_inside
	if(rva_inside >= s.get_virtual_address() && rva_inside < s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()))
	{
		//Calculate remaining length of section data from "rva" address
		int32_t length = static_cast<int32_t>(datatype == section_data_raw ? s.get_raw_data().length() /* instead of SizeOfRawData */ : s.get_aligned_virtual_size(get_section_alignment()))
			+ s.get_virtual_address() - rva_inside;

		if(length < 0)
			return 0;

		return static_cast<uint32_t>(length);
	}

	throw pe_exception("RVA not found inside section", pe_exception::rva_not_exists);
}

//Returns section remaining RAW/VIRTUAL data length from VA to the end of section "s" for PE32 (checks bounds)
uint32_t pe_base::section_data_length_from_va(const section& s, uint32_t va_inside, section_data_type datatype) const
{
	return section_data_length_from_rva(s, va_to_rva(va_inside), datatype);
}

//Returns section remaining RAW/VIRTUAL data length from VA to the end of section "s" for PE32/PE64 (checks bounds)
uint32_t pe_base::section_data_length_from_va(const section& s, uint64_t va_inside, section_data_type datatype) const
{
	return section_data_length_from_rva(s, va_to_rva(va_inside), datatype);
}

//Returns corresponding section data pointer from RVA inside section
char* pe_base::section_data_from_rva(uint32_t rva, bool include_headers)
{
	//if RVA is inside of headers and we're searching them too...
	if(include_headers && rva < full_headers_data_.length())
		return &full_headers_data_[rva];

	section& s = section_from_rva(rva);

	if(s.get_raw_data().empty())
		throw pe_exception("Section raw data is empty and cannot be changed", pe_exception::section_is_empty);

	return &s.get_raw_data()[rva - s.get_virtual_address()];
}

//Returns corresponding section data pointer from RVA inside section
const char* pe_base::section_data_from_rva(uint32_t rva, section_data_type datatype, bool include_headers) const
{
	//if RVA is inside of headers and we're searching them too...
	if(include_headers && rva < full_headers_data_.length())
		return &full_headers_data_[rva];

	const section& s = section_from_rva(rva);
	return (datatype == section_data_raw ? s.get_raw_data().data() : s.get_virtual_data(get_section_alignment()).c_str()) + rva - s.get_virtual_address();
}

//Reads DOS headers from istream
void pe_base::read_dos_header(std::istream& file, image_dos_header& header)
{
	//Check istream flags
	if(file.bad() || file.eof())
		throw pe_exception("PE file stream is bad or closed.", pe_exception::bad_pe_file);

	//Read DOS header and check istream
	file.read(reinterpret_cast<char*>(&header), sizeof(image_dos_header));
	if(file.bad() || file.eof())
		throw pe_exception("Unable to read IMAGE_DOS_HEADER", pe_exception::bad_dos_header);

	//Check DOS header magic
	if(header.e_magic != 0x5a4d) //"MZ"
		throw pe_exception("IMAGE_DOS_HEADER signature is incorrect", pe_exception::bad_dos_header);
}

//Reads DOS headers from istream
void pe_base::read_dos_header(std::istream& file)
{
	read_dos_header(file, dos_header_);
}

//Reads PE image from istream
void pe_base::read_pe(std::istream& file, bool read_debug_raw_data)
{
	//Get istream size
	std::streamoff filesize = pe_utils::get_file_size(file);

	//Check if PE header is DWORD-aligned
	if((dos_header_.e_lfanew % sizeof(uint32_t)) != 0)
		throw pe_exception("PE header is not DWORD-aligned", pe_exception::bad_dos_header);

	//Seek to NT headers
	file.seekg(dos_header_.e_lfanew);
	if(file.bad() || file.fail())
		throw pe_exception("Cannot reach IMAGE_NT_HEADERS", pe_exception::image_nt_headers_not_found);

	//Read NT headers
	file.read(get_nt_headers_ptr(), get_sizeof_nt_header() - sizeof(image_data_directory) * image_numberof_directory_entries);
	if(file.bad() || file.eof())
		throw pe_exception("Error reading IMAGE_NT_HEADERS", pe_exception::error_reading_image_nt_headers);

	//Check PE signature
	if(get_pe_signature() != 0x4550) //"PE"
		throw pe_exception("Incorrect PE signature", pe_exception::pe_signature_incorrect);

	//Check number of directories
	if(get_number_of_rvas_and_sizes() > image_numberof_directory_entries)
		set_number_of_rvas_and_sizes(image_numberof_directory_entries);

	if(get_number_of_rvas_and_sizes() > 0)
	{
		//Read data directory headers, if any
		file.read(get_nt_headers_ptr() + (get_sizeof_nt_header() - sizeof(image_data_directory) * image_numberof_directory_entries), sizeof(image_data_directory) * get_number_of_rvas_and_sizes());
		if(file.bad() || file.eof())
			throw pe_exception("Error reading DATA_DIRECTORY headers", pe_exception::error_reading_data_directories);
	}

	//Check section number
	//Images with zero section number accepted
	if(get_number_of_sections() > maximum_number_of_sections)
		throw pe_exception("Incorrect number of sections", pe_exception::section_number_incorrect);

	//Check PE magic
	if(get_magic() != get_needed_magic())
		throw pe_exception("Incorrect PE signature", pe_exception::pe_signature_incorrect);

	//Check section alignment
	if(!pe_utils::is_power_of_2(get_section_alignment()))
		throw pe_exception("Incorrect section alignment", pe_exception::incorrect_section_alignment);

	//Check file alignment
	if(!pe_utils::is_power_of_2(get_file_alignment()))
		throw pe_exception("Incorrect file alignment", pe_exception::incorrect_file_alignment);

	if(get_file_alignment() != get_section_alignment() && (get_file_alignment() < minimum_file_alignment || get_file_alignment() > get_section_alignment()))
		throw pe_exception("Incorrect file alignment", pe_exception::incorrect_file_alignment);

	//Check size of image
	if(pe_utils::align_up(get_size_of_image(), get_section_alignment()) == 0)
		throw pe_exception("Incorrect size of image", pe_exception::incorrect_size_of_image);
	
	//Read rich data overlay / DOS stub (if any)
	if(static_cast<uint32_t>(dos_header_.e_lfanew) > sizeof(image_dos_header))
	{
		rich_overlay_.resize(dos_header_.e_lfanew - sizeof(image_dos_header));
		file.seekg(sizeof(image_dos_header));
		file.read(&rich_overlay_[0], dos_header_.e_lfanew - sizeof(image_dos_header));
		if(file.bad() || file.eof())
			throw pe_exception("Error reading 'Rich' & 'DOS stub' overlay", pe_exception::error_reading_overlay);
	}

	//Calculate first section raw position
	//Sum is safe here
	uint32_t first_section = dos_header_.e_lfanew + get_size_of_optional_header() + sizeof(image_file_header) + sizeof(uint32_t) /* Signature */;

	if(get_number_of_sections() > 0)
	{
		//Go to first section
		file.seekg(first_section);
		if(file.bad() || file.fail())
			throw pe_exception("Cannot reach section headers", pe_exception::image_section_headers_not_found);
	}

	uint32_t last_raw_size = 0;

	//Read all sections
	for(int i = 0; i < get_number_of_sections(); i++)
	{
		section s;
		//Read section header
		file.read(reinterpret_cast<char*>(&s.get_raw_header()), sizeof(image_section_header));
		if(file.bad() || file.eof())
			throw pe_exception("Error reading section header", pe_exception::error_reading_section_header);

		//Save next section header position
		std::streamoff next_sect = file.tellg();

		//Check section virtual and raw sizes
		if(!s.get_size_of_raw_data() && !s.get_virtual_size())
			throw pe_exception("Virtual and Physical sizes of section can't be 0 at the same time", pe_exception::zero_section_sizes);

		//Check for adequate values of section fields
		if(!pe_utils::is_sum_safe(s.get_virtual_address(), s.get_virtual_size()) || s.get_virtual_size() > pe_utils::two_gb
			|| !pe_utils::is_sum_safe(s.get_pointer_to_raw_data(), s.get_size_of_raw_data()) || s.get_size_of_raw_data() > pe_utils::two_gb)
			throw pe_exception("Incorrect section address or size", pe_exception::section_incorrect_addr_or_size);

		if(s.get_size_of_raw_data() != 0)
		{
			//If section has raw data

			//If section raw data size is greater than virtual, fix it
			last_raw_size = s.get_size_of_raw_data();
			if(pe_utils::align_up(s.get_size_of_raw_data(), get_file_alignment()) > pe_utils::align_up(s.get_virtual_size(), get_section_alignment()))
				s.set_size_of_raw_data(s.get_virtual_size());

			//Check virtual and raw section sizes and addresses
			if(s.get_virtual_address() + pe_utils::align_up(s.get_virtual_size(), get_section_alignment()) > pe_utils::align_up(get_size_of_image(), get_section_alignment())
				||
				pe_utils::align_down(s.get_pointer_to_raw_data(), get_file_alignment()) + s.get_size_of_raw_data() > static_cast<uint32_t>(filesize))
				throw pe_exception("Incorrect section address or size", pe_exception::section_incorrect_addr_or_size);

			//Seek to section raw data
			file.seekg(pe_utils::align_down(s.get_pointer_to_raw_data(), get_file_alignment()));
			if(file.bad() || file.fail())
				throw pe_exception("Cannot reach section data", pe_exception::image_section_data_not_found);

			//Read section raw data
			s.get_raw_data().resize(s.get_size_of_raw_data());
			file.read(&s.get_raw_data()[0], s.get_size_of_raw_data());
			if(file.bad() || file.fail())
				throw pe_exception("Error reading section data", pe_exception::image_section_data_not_found);
		}

		//Check virtual address and size of section
		if(s.get_virtual_address() + s.get_aligned_virtual_size(get_section_alignment()) > pe_utils::align_up(get_size_of_image(), get_section_alignment()))
			throw pe_exception("Incorrect section address or size", pe_exception::section_incorrect_addr_or_size);

		//Save section
		sections_.push_back(s);

		//Seek to the next section header
		file.seekg(next_sect);
	}

	//Check size of headers: SizeOfHeaders can't be larger than first section VA
	if(!sections_.empty() && get_size_of_headers() > sections_.front().get_virtual_address())
		throw pe_exception("Incorrect size of headers", pe_exception::incorrect_size_of_headers);

	//If image has more than two sections
	if(sections_.size() >= 2)
	{
		//Check sections virtual sizes
		for(section_list::const_iterator i = sections_.begin() + 1; i != sections_.end(); ++i)
		{
			if((*i).get_virtual_address() != (*(i - 1)).get_virtual_address() + (*(i - 1)).get_aligned_virtual_size(get_section_alignment()))
				throw pe_exception("Section table is incorrect", pe_exception::image_section_table_incorrect);
		}
	}

	//Check if image has overlay in the end of file
	has_overlay_ = !sections_.empty() && filesize > static_cast<std::streamoff>(sections_.back().get_pointer_to_raw_data() + last_raw_size);

	{
		//Additionally, read data from the beginning of istream to size of headers
		file.seekg(0);
		uint32_t size_of_headers = std::min<uint32_t>(get_size_of_headers(), static_cast<uint32_t>(filesize));

		if(!sections_.empty())
		{
			for(section_list::const_iterator i = sections_.begin(); i != sections_.end(); ++i)
			{
				if(!(*i).empty())
				{
					size_of_headers = std::min<uint32_t>(get_size_of_headers(), (*i).get_pointer_to_raw_data());
					break;
				}
			}
		}

		full_headers_data_.resize(size_of_headers);
		file.read(&full_headers_data_[0], size_of_headers);
		if(file.bad() || file.eof())
			throw pe_exception("Error reading file", pe_exception::error_reading_file);
	}

	//Moreover, if there's debug directory, read its raw data for some debug info types
	while(read_debug_raw_data && has_debug())
	{
		try
		{
			//Check the length in bytes of the section containing debug directory
			if(section_data_length_from_rva(get_directory_rva(image_directory_entry_debug), get_directory_rva(image_directory_entry_debug), section_data_virtual, true) < sizeof(image_debug_directory))
				break;

			unsigned long current_pos = get_directory_rva(image_directory_entry_debug);

			//First IMAGE_DEBUG_DIRECTORY table
			image_debug_directory directory = section_data_from_rva<image_debug_directory>(current_pos, section_data_virtual, true);

			//Iterate over all IMAGE_DEBUG_DIRECTORY directories
			while(directory.PointerToRawData
				&& current_pos < get_directory_rva(image_directory_entry_debug) + get_directory_size(image_directory_entry_debug))
			{
				//If we have something to read
				if((directory.Type == image_debug_type_codeview
					|| directory.Type == image_debug_type_misc
					|| directory.Type == image_debug_type_coff)
					&& directory.SizeOfData)
				{
					std::string data;
					data.resize(directory.SizeOfData);
					file.seekg(directory.PointerToRawData);
					file.read(&data[0], directory.SizeOfData);
					if(file.bad() || file.eof())
						throw pe_exception("Error reading file", pe_exception::error_reading_file);

					debug_data_.insert(std::make_pair(directory.PointerToRawData, data));
				}

				//Go to next debug entry
				current_pos += sizeof(image_debug_directory);
				directory = section_data_from_rva<image_debug_directory>(current_pos, section_data_virtual, true);
			}

			break;
		}
		catch(const pe_exception&)
		{
			//Don't throw any exception here, if debug info is corrupted or incorrect
			break;
		}
		catch(const std::bad_alloc&)
		{
			//Don't throw any exception here, if debug info is corrupted or incorrect
			break;
		}
	}
}

//Returns PE type of this image
pe_type pe_base::get_pe_type() const
{
	return props_->get_pe_type();
}

//Returns PE type (PE or PE+) from pe_type enumeration (minimal correctness checks)
pe_type pe_base::get_pe_type(std::istream& file)
{
	//Save state of the istream
	std::ios_base::iostate state = file.exceptions();
	std::streamoff old_offset = file.tellg();
	image_nt_headers32 nt_headers;
	image_dos_header header;

	try
	{
		//Read dos header
		file.exceptions(std::ios::goodbit);
		read_dos_header(file, header);

		//Seek to the NT headers start
		file.seekg(header.e_lfanew);
		if(file.bad() || file.fail())
			throw pe_exception("Cannot reach IMAGE_NT_HEADERS", pe_exception::image_nt_headers_not_found);

		//Read NT headers (we're using 32-bit version, because there's no significant differencies between 32 and 64 bit version structures)
		file.read(reinterpret_cast<char*>(&nt_headers), sizeof(image_nt_headers32) - sizeof(image_data_directory) * image_numberof_directory_entries);
		if(file.bad() || file.eof())
			throw pe_exception("Error reading IMAGE_NT_HEADERS", pe_exception::error_reading_image_nt_headers);

		//Check NT headers signature
		if(nt_headers.Signature != 0x4550) //"PE"
			throw pe_exception("Incorrect PE signature", pe_exception::pe_signature_incorrect);

		//Check NT headers magic
		if(nt_headers.OptionalHeader.Magic != image_nt_optional_hdr32_magic && nt_headers.OptionalHeader.Magic != image_nt_optional_hdr64_magic)
			throw pe_exception("Incorrect PE signature", pe_exception::pe_signature_incorrect);
	}
	catch(const std::exception&)
	{
		//If something went wrong, restore istream state
		file.exceptions(state);
		file.seekg(old_offset);
		file.clear();
		//Retrhow exception
		throw;
	}

	//Restore stream state
	file.exceptions(state);
	file.seekg(old_offset);
	file.clear();

	//Determine PE type and return it
	return nt_headers.OptionalHeader.Magic == image_nt_optional_hdr64_magic ? pe_type_64 : pe_type_32;
}

//Returns true if image has overlay data at the end of file
bool pe_base::has_overlay() const
{
	return has_overlay_;
}

//Clears PE characteristics flag
void pe_base::clear_characteristics_flags(uint16_t flags)
{
	set_characteristics(get_characteristics() & ~flags);
}

//Sets PE characteristics flag
void pe_base::set_characteristics_flags(uint16_t flags)
{
	set_characteristics(get_characteristics() | flags);
}

//Returns true if PE characteristics flag set
bool pe_base::check_characteristics_flag(uint16_t flag) const
{
	return (get_characteristics() & flag) ? true : false;
}

//Returns subsystem value
uint16_t pe_base::get_subsystem() const
{
	return props_->get_subsystem();
}

//Sets subsystem value
void pe_base::set_subsystem(uint16_t subsystem)
{
	props_->set_subsystem(subsystem);
}

//Returns true if image has console subsystem
bool pe_base::is_console() const
{
	return get_subsystem() == image_subsystem_windows_cui;
}

//Returns true if image has Windows GUI subsystem
bool pe_base::is_gui() const
{
	return get_subsystem() == image_subsystem_windows_gui;
}

//Sets required operation system version
void pe_base::set_os_version(uint16_t major, uint16_t minor)
{
	props_->set_os_version(major, minor);
}

//Returns required operation system version (minor word)
uint16_t pe_base::get_minor_os_version() const
{
	return props_->get_minor_os_version();
}

//Returns required operation system version (major word)
uint16_t pe_base::get_major_os_version() const
{
	return props_->get_major_os_version();
}

//Sets required subsystem version
void pe_base::set_subsystem_version(uint16_t major, uint16_t minor)
{
	props_->set_subsystem_version(major, minor);
}

//Returns required subsystem version (minor word)
uint16_t pe_base::get_minor_subsystem_version() const
{
	return props_->get_minor_subsystem_version();
}

//Returns required subsystem version (major word)
uint16_t pe_base::get_major_subsystem_version() const
{
	return props_->get_major_subsystem_version();
}

//Returns corresponding section data pointer from VA inside section "s" for PE32 (checks bounds)
char* pe_base::section_data_from_va(section& s, uint32_t va) //Always returns raw data
{
	return section_data_from_rva(s, va_to_rva(va));
}

//Returns corresponding section data pointer from VA inside section "s" for PE32 (checks bounds)
const char* pe_base::section_data_from_va(const section& s, uint32_t va, section_data_type datatype) const
{
	return section_data_from_rva(s, va_to_rva(va), datatype);
}

//Returns corresponding section data pointer from VA inside section for PE32
char* pe_base::section_data_from_va(uint32_t va, bool include_headers) //Always returns raw data
{
	return section_data_from_rva(va_to_rva(va), include_headers);
}

//Returns corresponding section data pointer from VA inside section for PE32
const char* pe_base::section_data_from_va(uint32_t va, section_data_type datatype, bool include_headers) const
{
	return section_data_from_rva(va_to_rva(va), datatype, include_headers);
}

//Returns corresponding section data pointer from VA inside section "s" for PE32/PE64 (checks bounds)
char* pe_base::section_data_from_va(section& s, uint64_t va)  //Always returns raw data
{
	return section_data_from_rva(s, va_to_rva(va));
}

//Returns corresponding section data pointer from VA inside section "s" for PE32/PE64 (checks bounds)
const char* pe_base::section_data_from_va(const section& s, uint64_t va, section_data_type datatype) const
{
	return section_data_from_rva(s, va_to_rva(va), datatype);
}

//Returns corresponding section data pointer from VA inside section for PE32/PE64
char* pe_base::section_data_from_va(uint64_t va, bool include_headers)  //Always returns raw data
{
	return section_data_from_rva(va_to_rva(va), include_headers);
}

//Returns corresponding section data pointer from VA inside section for PE32/PE64
const char* pe_base::section_data_from_va(uint64_t va, section_data_type datatype, bool include_headers) const
{
	return section_data_from_rva(va_to_rva(va), datatype, include_headers);
}

//Returns section from VA inside it for PE32
section& pe_base::section_from_va(uint32_t va)
{
	return section_from_rva(va_to_rva(va));
}

//Returns section from VA inside it for PE32/PE64
section& pe_base::section_from_va(uint64_t va)
{
	return section_from_rva(va_to_rva(va));
}

//Returns section from RVA inside it for PE32
const section& pe_base::section_from_va(uint32_t va) const
{
	return section_from_rva(va_to_rva(va));
}

//Returns section from RVA inside it for PE32/PE64
const section& pe_base::section_from_va(uint64_t va) const
{
	return section_from_rva(va_to_rva(va));
}

uint32_t pe_base::va_to_rva(uint32_t va, bool bound_check) const
{
	return props_->va_to_rva(va, bound_check);
}

uint32_t pe_base::va_to_rva(uint64_t va, bool bound_check) const
{
	return props_->va_to_rva(va, bound_check);
}

uint32_t pe_base::rva_to_va_32(uint32_t rva) const
{
	return props_->rva_to_va_32(rva);
}

uint64_t pe_base::rva_to_va_64(uint32_t rva) const
{
	return props_->rva_to_va_64(rva);
}

//Relative Virtual Address (RVA) to Virtual Address (VA) convertion for PE32
void pe_base::rva_to_va(uint32_t rva, uint32_t& va) const
{
	va = rva_to_va_32(rva);
}

//Relative Virtual Address (RVA) to Virtual Address (VA) convertions for PE32/PE64
void pe_base::rva_to_va(uint32_t rva, uint64_t& va) const
{
	va = rva_to_va_64(rva);
}

//Returns section from file offset (4gb max)
section& pe_base::section_from_file_offset(uint32_t offset)
{
	return *file_offset_to_section(offset);
}

//Returns section from file offset (4gb max)
const section& pe_base::section_from_file_offset(uint32_t offset) const
{
	return *file_offset_to_section(offset);
}

//Returns section and offset (raw data only) from its start from RVA
const std::pair<uint32_t, const section*> pe_base::section_and_offset_from_rva(uint32_t rva) const
{
	const section& s = section_from_rva(rva);
	return std::make_pair(rva - s.get_virtual_address(), &s);
}

//Returns DLL Characteristics
uint16_t pe_base::get_dll_characteristics() const
{
	return props_->get_dll_characteristics();
}

//Sets DLL Characteristics
void pe_base::set_dll_characteristics(uint16_t characteristics)
{
	props_->set_dll_characteristics(characteristics);
}

//Returns size of headers
uint32_t pe_base::get_size_of_headers() const
{
	return props_->get_size_of_headers();
}

//Returns size of optional header
uint16_t pe_base::get_size_of_optional_header() const
{
	return props_->get_size_of_optional_header();
}

//Returns PE signature
uint32_t pe_base::get_pe_signature() const
{
	return props_->get_pe_signature();
}

//Returns magic value
uint32_t pe_base::get_magic() const
{
	return props_->get_magic();
}

//Returns image base for PE32
void pe_base::get_image_base(uint32_t& base) const
{
	base = get_image_base_32();
}

//Returns image base for PE32 and PE64 respectively
uint32_t pe_base::get_image_base_32() const
{
	return props_->get_image_base_32();
}

//Sets image base for PE32 and PE64 respectively
uint64_t pe_base::get_image_base_64() const
{
	return props_->get_image_base_64();
}

//RVA to RAW file offset convertion (4gb max)
uint32_t pe_base::rva_to_file_offset(uint32_t rva) const
{
	//Maybe, RVA is inside PE headers
	if(rva < get_size_of_headers())
		return rva;

	const section& s = section_from_rva(rva);
	return s.get_pointer_to_raw_data() + rva - s.get_virtual_address();
}

//RAW file offset to RVA convertion (4gb max)
uint32_t pe_base::file_offset_to_rva(uint32_t offset) const
{
	//Maybe, offset is inside PE headers
	if(offset < get_size_of_headers())
		return offset;

	const section_list::const_iterator it = file_offset_to_section(offset);
	return offset - (*it).get_pointer_to_raw_data() + (*it).get_virtual_address();
}

//RAW file offset to section convertion helper (4gb max)
section_list::const_iterator pe_base::file_offset_to_section(uint32_t offset) const
{
	section_list::const_iterator it = std::find_if(sections_.begin(), sections_.end(), section_by_raw_offset(offset));
	if(it == sections_.end())
		throw pe_exception("No section found by presented file offset", pe_exception::no_section_found);

	return it;
}

//RAW file offset to section convertion helper (4gb max)
section_list::iterator pe_base::file_offset_to_section(uint32_t offset)
{
	section_list::iterator it = std::find_if(sections_.begin(), sections_.end(), section_by_raw_offset(offset));
	if(it == sections_.end())
		throw pe_exception("No section found by presented file offset", pe_exception::no_section_found);
	
	return it;
}

//RVA from section raw data offset
uint32_t pe_base::rva_from_section_offset(const section& s, uint32_t raw_offset_from_section_start)
{
	return s.get_virtual_address() + raw_offset_from_section_start;
}

//Returns image base for PE32/PE64
void pe_base::get_image_base(uint64_t& base) const
{
	base = get_image_base_64();
}

//Sets new image base
void pe_base::set_image_base(uint32_t base)
{
	props_->set_image_base(base);
}

void pe_base::set_image_base_64(uint64_t base)
{
	props_->set_image_base_64(base);
}

//Sets heap size commit for PE32 and PE64 respectively
void pe_base::set_heap_size_commit(uint32_t size)
{
	props_->set_heap_size_commit(size);
}

void pe_base::set_heap_size_commit(uint64_t size)
{
	props_->set_heap_size_commit(size);
}

//Sets heap size reserve for PE32 and PE64 respectively
void pe_base::set_heap_size_reserve(uint32_t size)
{
	props_->set_heap_size_reserve(size);
}

void pe_base::set_heap_size_reserve(uint64_t size)
{
	props_->set_heap_size_reserve(size);
}

//Sets stack size commit for PE32 and PE64 respectively
void pe_base::set_stack_size_commit(uint32_t size)
{
	props_->set_stack_size_commit(size);
}

void pe_base::set_stack_size_commit(uint64_t size)
{
	props_->set_stack_size_commit(size);
}

//Sets stack size reserve for PE32 and PE64 respectively
void pe_base::set_stack_size_reserve(uint32_t size)
{
	props_->set_stack_size_reserve(size);
}

void pe_base::set_stack_size_reserve(uint64_t size)
{
	props_->set_stack_size_reserve(size);
}

//Returns heap size commit for PE32 and PE64 respectively
uint32_t pe_base::get_heap_size_commit_32() const
{
	return props_->get_heap_size_commit_32();
}

uint64_t pe_base::get_heap_size_commit_64() const
{
	return props_->get_heap_size_commit_64();
}

//Returns heap size reserve for PE32 and PE64 respectively
uint32_t pe_base::get_heap_size_reserve_32() const
{
	return props_->get_heap_size_reserve_32();
}

uint64_t pe_base::get_heap_size_reserve_64() const
{
	return props_->get_heap_size_reserve_64();
}

//Returns stack size commit for PE32 and PE64 respectively
uint32_t pe_base::get_stack_size_commit_32() const
{
	return props_->get_stack_size_commit_32();
}

uint64_t pe_base::get_stack_size_commit_64() const
{
	return props_->get_stack_size_commit_64();
}

//Returns stack size reserve for PE32 and PE64 respectively
uint32_t pe_base::get_stack_size_reserve_32() const
{
	return props_->get_stack_size_reserve_32();
}

uint64_t pe_base::get_stack_size_reserve_64() const
{
	return props_->get_stack_size_reserve_64();
}

//Returns heap size commit for PE32
void pe_base::get_heap_size_commit(uint32_t& size) const
{
	size = get_heap_size_commit_32();
}

//Returns heap size commit for PE32/PE64
void pe_base::get_heap_size_commit(uint64_t& size) const
{
	size = get_heap_size_commit_64();
}

//Returns heap size reserve for PE32
void pe_base::get_heap_size_reserve(uint32_t& size) const
{
	size = get_heap_size_reserve_32();
}

//Returns heap size reserve for PE32/PE64
void pe_base::get_heap_size_reserve(uint64_t& size) const
{
	size = get_heap_size_reserve_64();
}

//Returns stack size commit for PE32
void pe_base::get_stack_size_commit(uint32_t& size) const
{
	size = get_stack_size_commit_32();
}

//Returns stack size commit for PE32/PE64
void pe_base::get_stack_size_commit(uint64_t& size) const
{
	size = get_stack_size_commit_64();
}

//Returns stack size reserve for PE32
void pe_base::get_stack_size_reserve(uint32_t& size) const
{
	size = get_stack_size_reserve_32();
}

//Returns stack size reserve for PE32/PE64
void pe_base::get_stack_size_reserve(uint64_t& size) const
{
	size = get_stack_size_reserve_64();
}

//Realigns file (changes file alignment)
void pe_base::realign_file(uint32_t new_file_alignment)
{
	//Checks alignment for correctness
	set_file_alignment(new_file_alignment);
	realign_all_sections();
}

//Helper function to recalculate RAW and virtual section sizes and strip it, if necessary
void pe_base::recalculate_section_sizes(section& s, bool auto_strip)
{
	prepare_section(s); //Recalculate section raw addresses

	//Strip RAW size of section, if it is the last one
	//For all others it must be file-aligned and calculated by prepare_section() call
	if(auto_strip && !(sections_.empty() || &s == &*(sections_.end() - 1)))
	{
		//Strip ending raw data nullbytes to optimize size
		std::string& raw_data = s.get_raw_data();
		if(!raw_data.empty())
		{
			std::string::size_type i = raw_data.length();
			for(; i != 1; --i)
			{
				if(raw_data[i - 1] != 0)
					break;
			}
			
			raw_data.resize(i);
		}

		s.set_size_of_raw_data(static_cast<uint32_t>(raw_data.length()));
	}

	//Can occur only for last section
	if(pe_utils::align_up(s.get_virtual_size(), get_section_alignment()) < pe_utils::align_up(s.get_size_of_raw_data(), get_file_alignment()))
		set_section_virtual_size(s, pe_utils::align_up(s.get_size_of_raw_data(), get_section_alignment())); //Recalculate section virtual size
}

//Returns data from the beginning of image
//Size = SizeOfHeaders
const std::string& pe_base::get_full_headers_data() const
{
	return full_headers_data_;
}

const pe_base::debug_data_list& pe_base::get_raw_debug_data_list() const
{
	return debug_data_;
}

//Sets number of sections
void pe_base::set_number_of_sections(uint16_t number)
{
	props_->set_number_of_sections(number);
}

//Sets size of image
void pe_base::set_size_of_image(uint32_t size)
{
	props_->set_size_of_image(size);
}

//Sets size of headers
void pe_base::set_size_of_headers(uint32_t size)
{
	props_->set_size_of_headers(size);
}

//Sets size of optional headers
void pe_base::set_size_of_optional_header(uint16_t size)
{
	props_->set_size_of_optional_header(size);
}

//Returns nt headers data pointer
char* pe_base::get_nt_headers_ptr()
{
	return props_->get_nt_headers_ptr();
}

//Returns nt headers data pointer
const char* pe_base::get_nt_headers_ptr() const
{
	return props_->get_nt_headers_ptr();
}

//Returns sizeof() nt headers
uint32_t pe_base::get_sizeof_nt_header() const
{
	return props_->get_sizeof_nt_header();
}

//Returns sizeof() optional headers
uint32_t pe_base::get_sizeof_opt_headers() const
{
	return props_->get_sizeof_opt_headers();
}

//Sets file alignment (no checks)
void pe_base::set_file_alignment_unchecked(uint32_t alignment)
{
	props_->set_file_alignment_unchecked(alignment);
}

//Sets base of code
void pe_base::set_base_of_code(uint32_t base)
{
	props_->set_base_of_code(base);
}

//Returns base of code
uint32_t pe_base::get_base_of_code() const
{
	return props_->get_base_of_code();
}

//Returns needed magic of image
uint32_t pe_base::get_needed_magic() const
{
	return props_->get_needed_magic();
}
}
