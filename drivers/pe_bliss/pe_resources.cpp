#include <algorithm>
#include <string.h>
#include "pe_resources.h"

namespace pe_bliss
{
using namespace pe_win;

//RESOURCES
//Default constructor
resource_data_entry::resource_data_entry()
	:codepage_(0)
{}

//Constructor from data
resource_data_entry::resource_data_entry(const std::string& data, uint32_t codepage)
	:codepage_(codepage), data_(data)
{}

//Returns resource data codepage
uint32_t resource_data_entry::get_codepage() const
{
	return codepage_;
}

//Returns resource data
const std::string& resource_data_entry::get_data() const
{
	return data_;
}

//Sets resource data codepage
void resource_data_entry::set_codepage(uint32_t codepage)
{
	codepage_ = codepage;
}

//Sets resource data
void resource_data_entry::set_data(const std::string& data)
{
	data_ = data;
}

//Default constructor
resource_directory_entry::includes::includes()
	:data_(0)
{}

//Default constructor
resource_directory_entry::resource_directory_entry()
	:id_(0), includes_data_(false), named_(false)
{}

//Copy constructor
resource_directory_entry::resource_directory_entry(const resource_directory_entry& other)
	:id_(other.id_), name_(other.name_), includes_data_(other.includes_data_), named_(other.named_)
{
	//If union'ed pointer is not zero
	if(other.ptr_.data_)
	{
		if(other.includes_data())
			ptr_.data_ = new resource_data_entry(*other.ptr_.data_);
		else
			ptr_.dir_ = new resource_directory(*other.ptr_.dir_);
	}
}

//Copy assignment operator
resource_directory_entry& resource_directory_entry::operator=(const resource_directory_entry& other)
{
	release();

	id_ = other.id_;
	name_ = other.name_;
	includes_data_ = other.includes_data_;
	named_ = other.named_;

	//If other union'ed pointer is not zero
	if(other.ptr_.data_)
	{
		if(other.includes_data())
			ptr_.data_ = new resource_data_entry(*other.ptr_.data_);
		else
			ptr_.dir_ = new resource_directory(*other.ptr_.dir_);
	}

	return *this;
}

//Destroys included data
void resource_directory_entry::release()
{
	//If union'ed pointer is not zero
	if(ptr_.data_)
	{
		if(includes_data())
			delete ptr_.data_;
		else
			delete ptr_.dir_;

		ptr_.data_ = 0;
	}
}

//Destructor
resource_directory_entry::~resource_directory_entry()
{
	release();
}

//Returns entry ID
uint32_t resource_directory_entry::get_id() const
{
	return id_;
}

//Returns entry name
const std::wstring& resource_directory_entry::get_name() const
{
	return name_;
}

//Returns true, if entry has name
//Returns false, if entry has ID
bool resource_directory_entry::is_named() const
{
	return named_;
}

//Returns true, if entry includes resource_data_entry
//Returns false, if entry includes resource_directory
bool resource_directory_entry::includes_data() const
{
	return includes_data_;
}

//Returns resource_directory if entry includes it, otherwise throws an exception
const resource_directory& resource_directory_entry::get_resource_directory() const
{
	if(!ptr_.dir_ || includes_data_)
		throw pe_exception("Resource directory entry does not contain resource directory", pe_exception::resource_directory_entry_error);

	return *ptr_.dir_;
}

//Returns resource_data_entry if entry includes it, otherwise throws an exception
const resource_data_entry& resource_directory_entry::get_data_entry() const
{
	if(!ptr_.data_ || !includes_data_)
		throw pe_exception("Resource directory entry does not contain resource data entry", pe_exception::resource_directory_entry_error);

	return *ptr_.data_;
}

//Returns resource_directory if entry includes it, otherwise throws an exception
resource_directory& resource_directory_entry::get_resource_directory()
{
	if(!ptr_.dir_ || includes_data_)
		throw pe_exception("Resource directory entry does not contain resource directory", pe_exception::resource_directory_entry_error);

	return *ptr_.dir_;
}

//Returns resource_data_entry if entry includes it, otherwise throws an exception
resource_data_entry& resource_directory_entry::get_data_entry()
{
	if(!ptr_.data_ || !includes_data_)
		throw pe_exception("Resource directory entry does not contain resource data entry", pe_exception::resource_directory_entry_error);

	return *ptr_.data_;
}

//Sets entry name
void resource_directory_entry::set_name(const std::wstring& name)
{
	name_ = name;
	named_ = true;
	id_ = 0;
}

//Sets entry ID
void resource_directory_entry::set_id(uint32_t id)
{
	id_ = id;
	named_ = false;
	name_.clear();
}

//Adds resource_data_entry
void resource_directory_entry::add_data_entry(const resource_data_entry& entry)
{
	release();
	ptr_.data_ = new resource_data_entry(entry);
	includes_data_ = true;
}

//Adds resource_directory
void resource_directory_entry::add_resource_directory(const resource_directory& dir)
{
	release();
	ptr_.dir_ = new resource_directory(dir);
	includes_data_ = false;
}

//Default constructor
resource_directory::resource_directory()
	:characteristics_(0),
	timestamp_(0),
	major_version_(0), minor_version_(0),
	number_of_named_entries_(0), number_of_id_entries_(0)
{}

//Constructor from data
resource_directory::resource_directory(const image_resource_directory& dir)
	:characteristics_(dir.Characteristics),
	timestamp_(dir.TimeDateStamp),
	major_version_(dir.MajorVersion), minor_version_(dir.MinorVersion),
	number_of_named_entries_(0), number_of_id_entries_(0) //Set to zero here, calculate on add
{}

//Returns characteristics of directory
uint32_t resource_directory::get_characteristics() const
{
	return characteristics_;
}

//Returns date and time stamp of directory
uint32_t resource_directory::get_timestamp() const
{
	return timestamp_;
}

//Returns major version of directory
uint16_t resource_directory::get_major_version() const
{
	return major_version_;
}

//Returns minor version of directory
uint16_t resource_directory::get_minor_version() const
{
	return minor_version_;
}

//Returns number of named entries
uint32_t resource_directory::get_number_of_named_entries() const
{
	return number_of_named_entries_;
}

//Returns number of ID entries
uint32_t resource_directory::get_number_of_id_entries() const
{
	return number_of_id_entries_;
}

//Returns resource_directory_entry array
const resource_directory::entry_list& resource_directory::get_entry_list() const
{
	return entries_;
}

//Returns resource_directory_entry array
resource_directory::entry_list& resource_directory::get_entry_list()
{
	return entries_;
}

//Adds resource_directory_entry
void resource_directory::add_resource_directory_entry(const resource_directory_entry& entry)
{
	entries_.push_back(entry);
	if(entry.is_named())
		++number_of_named_entries_;
	else
		++number_of_id_entries_;
}

//Clears resource_directory_entry array
void resource_directory::clear_resource_directory_entry_list()
{
	entries_.clear();
	number_of_named_entries_ = 0;
	number_of_id_entries_ = 0;
}

//Sets characteristics of directory
void resource_directory::set_characteristics(uint32_t characteristics)
{
	characteristics_ = characteristics;
}

//Sets date and time stamp of directory
void resource_directory::set_timestamp(uint32_t timestamp)
{
	timestamp_ = timestamp;
}

//Sets number of named entries
void resource_directory::set_number_of_named_entries(uint32_t number)
{
	number_of_named_entries_ = number;
}

//Sets number of ID entries
void resource_directory::set_number_of_id_entries(uint32_t number)
{
	number_of_id_entries_ = number;
}

//Sets major version of directory
void resource_directory::set_major_version(uint16_t major_version)
{
	major_version_ = major_version;
}

//Sets minor version of directory
void resource_directory::get_minor_version(uint16_t minor_version)
{
	minor_version_ = minor_version;
}

//Processes resource directory
const resource_directory process_resource_directory(const pe_base& pe, uint32_t res_rva, uint32_t offset_to_directory, std::set<uint32_t>& processed)
{
	resource_directory ret;
	
	//Check for resource loops
	if(!processed.insert(offset_to_directory).second)
		throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

	if(!pe_utils::is_sum_safe(res_rva, offset_to_directory))
		throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

	//Get root IMAGE_RESOURCE_DIRECTORY
	image_resource_directory directory = pe.section_data_from_rva<image_resource_directory>(res_rva + offset_to_directory, section_data_virtual, true);

	ret = resource_directory(directory);

	//Check DWORDs for possible overflows
	if(!pe_utils::is_sum_safe(directory.NumberOfIdEntries, directory.NumberOfNamedEntries)
		|| directory.NumberOfIdEntries + directory.NumberOfNamedEntries >= pe_utils::max_dword / sizeof(image_resource_directory_entry) + sizeof(image_resource_directory))
		throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

	if(!pe_utils::is_sum_safe(offset_to_directory, sizeof(image_resource_directory) + (directory.NumberOfIdEntries + directory.NumberOfNamedEntries) * sizeof(image_resource_directory_entry))
		|| !pe_utils::is_sum_safe(res_rva, offset_to_directory + sizeof(image_resource_directory) + (directory.NumberOfIdEntries + directory.NumberOfNamedEntries) * sizeof(image_resource_directory_entry)))
		throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

	for(unsigned long i = 0; i != static_cast<unsigned long>(directory.NumberOfIdEntries) + directory.NumberOfNamedEntries; ++i)
	{
		//Read directory entries one by one
		image_resource_directory_entry dir_entry = pe.section_data_from_rva<image_resource_directory_entry>(
			res_rva + sizeof(image_resource_directory) + i * sizeof(image_resource_directory_entry) + offset_to_directory, section_data_virtual, true);

		//Create directory entry structure
		resource_directory_entry entry;

		//If directory is named
		if(dir_entry.NameIsString)
		{
			if(!pe_utils::is_sum_safe(res_rva + sizeof(uint16_t) /* safe */, dir_entry.NameOffset))
				throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

			//get directory name length
			uint16_t directory_name_length = pe.section_data_from_rva<uint16_t>(res_rva + dir_entry.NameOffset, section_data_virtual, true);

			//Check name length
			if(pe.section_data_length_from_rva(res_rva + dir_entry.NameOffset + sizeof(uint16_t), res_rva + dir_entry.NameOffset + sizeof(uint16_t), section_data_virtual, true)
				< directory_name_length)
				throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

#ifdef PE_BLISS_WINDOWS
			//Set entry UNICODE name
			entry.set_name(std::wstring(
				reinterpret_cast<const wchar_t*>(pe.section_data_from_rva(res_rva + dir_entry.NameOffset + sizeof(uint16_t), section_data_virtual, true)),
				directory_name_length));
#else
			//Set entry UNICODE name
			entry.set_name(pe_utils::from_ucs2(u16string(
				reinterpret_cast<const unicode16_t*>(pe.section_data_from_rva(res_rva + dir_entry.NameOffset + sizeof(uint16_t), section_data_virtual, true)),
				directory_name_length)));
#endif
		}
		else
		{
			//Else - set directory ID
			entry.set_id(dir_entry.Id);
		}

		//If directory entry has another resource directory
		if(dir_entry.DataIsDirectory)
		{
			entry.add_resource_directory(process_resource_directory(pe, res_rva, dir_entry.OffsetToDirectory, processed));
		}
		else
		{
			//If directory entry has data
			image_resource_data_entry data_entry = pe.section_data_from_rva<image_resource_data_entry>(
				res_rva + dir_entry.OffsetToData, section_data_virtual, true);

			//Check byte count that stated by data entry
			if(pe.section_data_length_from_rva(data_entry.OffsetToData, data_entry.OffsetToData, section_data_virtual, true) < data_entry.Size)
				throw pe_exception("Incorrect resource directory", pe_exception::incorrect_resource_directory);

			//Add data entry to directory entry
			entry.add_data_entry(resource_data_entry(
				std::string(pe.section_data_from_rva(data_entry.OffsetToData, section_data_virtual, true), data_entry.Size),
				data_entry.CodePage));
		}

		//Save directory entry
		ret.add_resource_directory_entry(entry);
	}

	//Return resource directory
	return ret;
}

//Helper function to calculate needed space for resource data
void calculate_resource_data_space(const resource_directory& root, uint32_t aligned_offset_from_section_start, uint32_t& needed_size_for_structures, uint32_t& needed_size_for_strings)
{
	needed_size_for_structures += sizeof(image_resource_directory);
	for(resource_directory::entry_list::const_iterator it = root.get_entry_list().begin(); it != root.get_entry_list().end(); ++it)
	{
		needed_size_for_structures += sizeof(image_resource_directory_entry);

		if((*it).is_named())
			needed_size_for_strings += static_cast<uint32_t>(((*it).get_name().length() + 1) * 2 /* unicode */ + sizeof(uint16_t) /* for string length */);

		if(!(*it).includes_data())
			calculate_resource_data_space((*it).get_resource_directory(), aligned_offset_from_section_start, needed_size_for_structures, needed_size_for_strings);
	}
}

//Helper function to calculate needed space for resource data
void calculate_resource_data_space(const resource_directory& root, uint32_t needed_size_for_structures, uint32_t needed_size_for_strings, uint32_t& needed_size_for_data, uint32_t& current_data_pos)
{
	for(resource_directory::entry_list::const_iterator it = root.get_entry_list().begin(); it != root.get_entry_list().end(); ++it)
	{
		if((*it).includes_data())
		{
			uint32_t data_size = static_cast<uint32_t>((*it).get_data_entry().get_data().length()
				+ sizeof(image_resource_data_entry)
				+ (pe_utils::align_up(current_data_pos, sizeof(uint32_t)) - current_data_pos) /* alignment */);
			needed_size_for_data += data_size;
			current_data_pos += data_size;
		}
		else
		{
			calculate_resource_data_space((*it).get_resource_directory(), needed_size_for_structures, needed_size_for_strings, needed_size_for_data, current_data_pos);
		}
	}
}

//Helper: sorts resource directory entries
struct entry_sorter
{
public:
	bool operator()(const resource_directory_entry& entry1, const resource_directory_entry& entry2) const;
};

//Helper function to rebuild resource directory
void rebuild_resource_directory(pe_base& pe, section& resource_section, resource_directory& root, uint32_t& current_structures_pos, uint32_t& current_data_pos, uint32_t& current_strings_pos, uint32_t offset_from_section_start)
{
	//Create resource directory
	image_resource_directory dir = {0};
	dir.Characteristics = root.get_characteristics();
	dir.MajorVersion = root.get_major_version();
	dir.MinorVersion = root.get_minor_version();
	dir.TimeDateStamp = root.get_timestamp();
	
	{
		resource_directory::entry_list& entries = root.get_entry_list();
		std::sort(entries.begin(), entries.end(), entry_sorter());
	}

	//Calculate number of named and ID entries
	for(resource_directory::entry_list::const_iterator it = root.get_entry_list().begin(); it != root.get_entry_list().end(); ++it)
	{
		if((*it).is_named())
			++dir.NumberOfNamedEntries;
		else
			++dir.NumberOfIdEntries;
	}
	
	std::string& raw_data = resource_section.get_raw_data();

	//Save resource directory
	memcpy(&raw_data[current_structures_pos], &dir, sizeof(dir));
	current_structures_pos += sizeof(dir);

	uint32_t this_current_structures_pos = current_structures_pos;

	current_structures_pos += sizeof(image_resource_directory_entry) * (dir.NumberOfNamedEntries + dir.NumberOfIdEntries);

	//Create all resource directory entries
	for(resource_directory::entry_list::iterator it = root.get_entry_list().begin(); it != root.get_entry_list().end(); ++it)
	{
		image_resource_directory_entry entry;
		if((*it).is_named())
		{
			entry.Name = 0x80000000 | (current_strings_pos - offset_from_section_start);
			uint16_t unicode_length = static_cast<uint16_t>((*it).get_name().length());
			memcpy(&raw_data[current_strings_pos], &unicode_length, sizeof(unicode_length));
			current_strings_pos += sizeof(unicode_length);

#ifdef PE_BLISS_WINDOWS
			memcpy(&raw_data[current_strings_pos], (*it).get_name().c_str(), (*it).get_name().length() * sizeof(uint16_t) + sizeof(uint16_t) /* unicode */);
#else
			{
				u16string str(pe_utils::to_ucs2((*it).get_name()));
				memcpy(&raw_data[current_strings_pos], str.c_str(), (*it).get_name().length() * sizeof(uint16_t) + sizeof(uint16_t) /* unicode */);
			}
#endif

			current_strings_pos += static_cast<unsigned long>((*it).get_name().length() * sizeof(uint16_t) + sizeof(uint16_t) /* unicode */);
		}
		else
		{
			entry.Name = (*it).get_id();
		}

		if((*it).includes_data())
		{
			current_data_pos = pe_utils::align_up(current_data_pos, sizeof(uint32_t));
			image_resource_data_entry data_entry = {0};
			data_entry.CodePage = (*it).get_data_entry().get_codepage();
			data_entry.Size = static_cast<uint32_t>((*it).get_data_entry().get_data().length());
			data_entry.OffsetToData = pe.rva_from_section_offset(resource_section, current_data_pos + sizeof(data_entry));
			
			entry.OffsetToData = current_data_pos - offset_from_section_start;

			memcpy(&raw_data[current_data_pos], &data_entry, sizeof(data_entry));
			current_data_pos += sizeof(data_entry);
			
			memcpy(&raw_data[current_data_pos], (*it).get_data_entry().get_data().data(), data_entry.Size);
			current_data_pos += data_entry.Size;

			memcpy(&raw_data[this_current_structures_pos], &entry, sizeof(entry));
			this_current_structures_pos += sizeof(entry);
		}
		else
		{
			entry.OffsetToData = 0x80000000 | (current_structures_pos - offset_from_section_start);

			memcpy(&raw_data[this_current_structures_pos], &entry, sizeof(entry));
			this_current_structures_pos += sizeof(entry);

			rebuild_resource_directory(pe, resource_section, (*it).get_resource_directory(), current_structures_pos, current_data_pos, current_strings_pos, offset_from_section_start);
		}
	}
}

//Helper function to rebuild resource directory
bool entry_sorter::operator()(const resource_directory_entry& entry1, const resource_directory_entry& entry2) const
{
	if(entry1.is_named() && entry2.is_named())
		return entry1.get_name() < entry2.get_name();
	else if(!entry1.is_named() && !entry2.is_named())
		return entry1.get_id() < entry2.get_id();
	else
		return entry1.is_named();
}

//Resources rebuilder
//resource_directory - root resource directory
//resources_section - section where resource directory will be placed (must be attached to PE image)
//offset_from_section_start - offset from resources_section raw data start
//resource_directory is non-constant, because it will be sorted
//save_to_pe_headers - if true, new resource directory information will be saved to PE image headers
//auto_strip_last_section - if true and resources are placed in the last section, it will be automatically stripped
//number_of_id_entries and number_of_named_entries for resource directories are recalculated and not used
const image_directory rebuild_resources(pe_base& pe, resource_directory& info, section& resources_section, uint32_t offset_from_section_start, bool save_to_pe_header, bool auto_strip_last_section)
{
	//Check that resources_section is attached to this PE image
	if(!pe.section_attached(resources_section))
		throw pe_exception("Resource section must be attached to PE file", pe_exception::section_is_not_attached);
	
	//Check resource directory correctness
	if(info.get_entry_list().empty())
		throw pe_exception("Empty resource directory", pe_exception::incorrect_resource_directory);
	
	uint32_t aligned_offset_from_section_start = pe_utils::align_up(offset_from_section_start, sizeof(uint32_t));
	uint32_t needed_size_for_structures = aligned_offset_from_section_start - offset_from_section_start; //Calculate needed size for resource tables and data
	uint32_t needed_size_for_strings = 0;
	uint32_t needed_size_for_data = 0;

	calculate_resource_data_space(info, aligned_offset_from_section_start, needed_size_for_structures, needed_size_for_strings);

	{
		uint32_t current_data_pos = aligned_offset_from_section_start + needed_size_for_structures + needed_size_for_strings;
		calculate_resource_data_space(info, needed_size_for_structures, needed_size_for_strings, needed_size_for_data, current_data_pos);
	}

	uint32_t needed_size = needed_size_for_structures + needed_size_for_strings + needed_size_for_data;

	//Check if resources_section is last one. If it's not, check if there's enough place for resource data
	if(&resources_section != &*(pe.get_image_sections().end() - 1) && 
		(resources_section.empty() || pe_utils::align_up(resources_section.get_size_of_raw_data(), pe.get_file_alignment())
		< needed_size + aligned_offset_from_section_start))
		throw pe_exception("Insufficient space for resource directory", pe_exception::insufficient_space);

	std::string& raw_data = resources_section.get_raw_data();

	//This will be done only if resources_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + aligned_offset_from_section_start)
		raw_data.resize(needed_size + aligned_offset_from_section_start); //Expand section raw data

	uint32_t current_structures_pos = aligned_offset_from_section_start;
	uint32_t current_strings_pos = current_structures_pos + needed_size_for_structures;
	uint32_t current_data_pos = current_strings_pos + needed_size_for_strings;
	rebuild_resource_directory(pe, resources_section, info, current_structures_pos, current_data_pos, current_strings_pos, aligned_offset_from_section_start);
	
	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(resources_section, auto_strip_last_section);

	image_directory ret(pe.rva_from_section_offset(resources_section, aligned_offset_from_section_start), needed_size);

	//If auto-rewrite of PE headers is required
	if(save_to_pe_header)
	{
		pe.set_directory_rva(image_directory_entry_resource, ret.get_rva());
		pe.set_directory_size(image_directory_entry_resource, ret.get_size());
	}

	return ret;
}

//Returns resources from PE file
const resource_directory get_resources(const pe_base& pe)
{
	resource_directory ret;

	if(!pe.has_resources())
		return ret;

	//Get resource directory RVA
	uint32_t res_rva = pe.get_directory_rva(image_directory_entry_resource);
	
	//Store already processed directories to avoid resource loops
	std::set<uint32_t> processed;
	
	//Process all directories (recursion)
	ret = process_resource_directory(pe, res_rva, 0, processed);

	return ret;
}

//Finds resource_directory_entry by ID
resource_directory::id_entry_finder::id_entry_finder(uint32_t id)
	:id_(id)
{}

bool resource_directory::id_entry_finder::operator()(const resource_directory_entry& entry) const
{
	return !entry.is_named() && entry.get_id() == id_;
}

//Finds resource_directory_entry by name
resource_directory::name_entry_finder::name_entry_finder(const std::wstring& name)
	:name_(name)
{}

bool resource_directory::name_entry_finder::operator()(const resource_directory_entry& entry) const
{
	return entry.is_named() && entry.get_name() == name_;
}

//Finds resource_directory_entry by name or ID (universal)
resource_directory::entry_finder::entry_finder(const std::wstring& name)
	:name_(name), named_(true)
{}

resource_directory::entry_finder::entry_finder(uint32_t id)
	:id_(id), named_(false)
{}

bool resource_directory::entry_finder::operator()(const resource_directory_entry& entry) const
{
	if(named_)
		return entry.is_named() && entry.get_name() == name_;
	else
		return !entry.is_named() && entry.get_id() == id_;
}

//Returns resource_directory_entry by ID. If not found - throws an exception
const resource_directory_entry& resource_directory::entry_by_id(uint32_t id) const
{
	entry_list::const_iterator i = std::find_if(entries_.begin(), entries_.end(), id_entry_finder(id));
	if(i == entries_.end())
		throw pe_exception("Resource directory entry not found", pe_exception::resource_directory_entry_not_found);

	return *i;
}

//Returns resource_directory_entry by name. If not found - throws an exception
const resource_directory_entry& resource_directory::entry_by_name(const std::wstring& name) const
{
	entry_list::const_iterator i = std::find_if(entries_.begin(), entries_.end(), name_entry_finder(name));
	if(i == entries_.end())
		throw pe_exception("Resource directory entry not found", pe_exception::resource_directory_entry_not_found);

	return *i;
}
}
