#include <string.h>
#include "pe_bound_import.h"
#include "utils.h"

namespace pe_bliss
{
using namespace pe_win;

//BOUND IMPORT
//Default constructor
bound_import_ref::bound_import_ref()
	:timestamp_(0)
{}

//Constructor from data
bound_import_ref::bound_import_ref(const std::string& module_name, uint32_t timestamp)
	:module_name_(module_name), timestamp_(timestamp)
{}

//Returns imported module name
const std::string& bound_import_ref::get_module_name() const
{
	return module_name_;
}

//Returns bound import date and time stamp
uint32_t bound_import_ref::get_timestamp() const
{
	return timestamp_;
}

//Sets module name
void bound_import_ref::set_module_name(const std::string& module_name)
{
	module_name_ = module_name;
}

//Sets timestamp
void bound_import_ref::set_timestamp(uint32_t timestamp)
{
	timestamp_ = timestamp;
}

//Default constructor
bound_import::bound_import()
	:timestamp_(0)
{}

//Constructor from data
bound_import::bound_import(const std::string& module_name, uint32_t timestamp)
	:module_name_(module_name), timestamp_(timestamp)
{}

//Returns imported module name
const std::string& bound_import::get_module_name() const
{
	return module_name_;
}

//Returns bound import date and time stamp
uint32_t bound_import::get_timestamp() const
{
	return timestamp_;
}

//Returns bound references cound
size_t bound_import::get_module_ref_count() const
{
	return refs_.size();
}

//Returns module references
const bound_import::ref_list& bound_import::get_module_ref_list() const
{
	return refs_;
}

//Adds module reference
void bound_import::add_module_ref(const bound_import_ref& ref)
{
	refs_.push_back(ref);
}

//Clears module references list
void bound_import::clear_module_refs()
{
	refs_.clear();
}

//Returns module references
bound_import::ref_list& bound_import::get_module_ref_list()
{
	return refs_;
}

//Sets module name
void bound_import::set_module_name(const std::string& module_name)
{
	module_name_ = module_name;
}

//Sets timestamp
void bound_import::set_timestamp(uint32_t timestamp)
{
	timestamp_ = timestamp;
}

const bound_import_module_list get_bound_import_module_list(const pe_base& pe)
{
	//Returned bound import modules list
	bound_import_module_list ret;

	//If image has no bound imports
	if(!pe.has_bound_import())
		return ret;

	uint32_t bound_import_data_len =
		pe.section_data_length_from_rva(pe.get_directory_rva(image_directory_entry_bound_import), pe.get_directory_rva(image_directory_entry_bound_import), section_data_raw, true);

	if(bound_import_data_len < pe.get_directory_size(image_directory_entry_bound_import))
		throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);
	
	const char* bound_import_data = pe.section_data_from_rva(pe.get_directory_rva(image_directory_entry_bound_import), section_data_raw, true);

	//Check read in "read_pe" function raw bound import data size
	if(bound_import_data_len < sizeof(image_bound_import_descriptor))
		throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

	//current bound_import_data_ in-string position
	unsigned long current_pos = 0;
	//first bound import descriptor
	//so, we're working with raw data here, no section helpers available
	const image_bound_import_descriptor* descriptor = reinterpret_cast<const image_bound_import_descriptor*>(&bound_import_data[current_pos]);

	//Enumerate until zero
	while(descriptor->OffsetModuleName)
	{
		//Check module name offset
		if(descriptor->OffsetModuleName >= bound_import_data_len)
			throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

		//Check module name for null-termination
		if(!pe_utils::is_null_terminated(&bound_import_data[descriptor->OffsetModuleName], bound_import_data_len - descriptor->OffsetModuleName))
			throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

		//Create bound import descriptor structure
		bound_import elem(&bound_import_data[descriptor->OffsetModuleName], descriptor->TimeDateStamp);

		//Check DWORDs
		if(descriptor->NumberOfModuleForwarderRefs >= pe_utils::max_dword / sizeof(image_bound_forwarder_ref)
			|| !pe_utils::is_sum_safe(current_pos, 2 /* this descriptor and the next one */ * sizeof(image_bound_import_descriptor) + descriptor->NumberOfModuleForwarderRefs * sizeof(image_bound_forwarder_ref)))
			throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

		//Move after current descriptor
		current_pos += sizeof(image_bound_import_descriptor);

		//Enumerate referenced bound import descriptors
		for(unsigned long i = 0; i != descriptor->NumberOfModuleForwarderRefs; ++i)
		{
			//They're just after parent descriptor
			//Check size of structure
			if(current_pos + sizeof(image_bound_forwarder_ref) > bound_import_data_len)
				throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

			//Get IMAGE_BOUND_FORWARDER_REF pointer
			const image_bound_forwarder_ref* ref_descriptor = reinterpret_cast<const image_bound_forwarder_ref*>(&bound_import_data[current_pos]);

			//Check referenced module name
			if(ref_descriptor->OffsetModuleName >= bound_import_data_len)
				throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

			//And its null-termination
			if(!pe_utils::is_null_terminated(&bound_import_data[ref_descriptor->OffsetModuleName], bound_import_data_len - ref_descriptor->OffsetModuleName))
				throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

			//Add referenced module to current bound import structure
			elem.add_module_ref(bound_import_ref(&bound_import_data[ref_descriptor->OffsetModuleName], ref_descriptor->TimeDateStamp));

			//Move after referenced bound import descriptor
			current_pos += sizeof(image_bound_forwarder_ref);
		}

		//Check structure size
		if(current_pos + sizeof(image_bound_import_descriptor) > bound_import_data_len)
			throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);

		//Move to next bound import descriptor
		descriptor = reinterpret_cast<const image_bound_import_descriptor*>(&bound_import_data[current_pos]);

		//Save created descriptor structure and references
		ret.push_back(elem);
	}

	//Return result
	return ret;
}

//imports - bound imported modules list
//imports_section - section where export directory will be placed (must be attached to PE image)
//offset_from_section_start - offset from imports_section raw data start
//save_to_pe_headers - if true, new bound import directory information will be saved to PE image headers
//auto_strip_last_section - if true and bound imports are placed in the last section, it will be automatically stripped
const image_directory rebuild_bound_imports(pe_base& pe, const bound_import_module_list& imports, section& imports_section, uint32_t offset_from_section_start, bool save_to_pe_header, bool auto_strip_last_section)
{
	//Check that exports_section is attached to this PE image
	if(!pe.section_attached(imports_section))
		throw pe_exception("Bound import section must be attached to PE file", pe_exception::section_is_not_attached);

	uint32_t directory_pos = pe_utils::align_up(offset_from_section_start, sizeof(uint32_t));
	uint32_t needed_size = sizeof(image_bound_import_descriptor) /* Ending null descriptor */;
	uint32_t needed_size_for_strings = 0;

	//Calculate needed size for bound import data
	for(bound_import_module_list::const_iterator it = imports.begin(); it != imports.end(); ++it)
	{
		const bound_import& import = *it;
		needed_size += sizeof(image_bound_import_descriptor);
		needed_size_for_strings += static_cast<uint32_t>((*it).get_module_name().length()) + 1 /* nullbyte */;

		const bound_import::ref_list& refs = import.get_module_ref_list();
		for(bound_import::ref_list::const_iterator ref_it = refs.begin(); ref_it != refs.end(); ++ref_it)
		{
			needed_size_for_strings += static_cast<uint32_t>((*ref_it).get_module_name().length()) + 1 /* nullbyte */;
			needed_size += sizeof(image_bound_forwarder_ref);
		}
	}
	
	needed_size += needed_size_for_strings;
	
	//Check if imports_section is last one. If it's not, check if there's enough place for bound import data
	if(&imports_section != &*(pe.get_image_sections().end() - 1) && 
		(imports_section.empty() || pe_utils::align_up(imports_section.get_size_of_raw_data(), pe.get_file_alignment()) < needed_size + directory_pos))
		throw pe_exception("Insufficient space for bound import directory", pe_exception::insufficient_space);

	std::string& raw_data = imports_section.get_raw_data();

	//This will be done only if imports_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + directory_pos)
		raw_data.resize(needed_size + directory_pos); //Expand section raw data
	
	uint32_t current_pos_for_structures = directory_pos;
	uint32_t current_pos_for_strings = current_pos_for_structures + needed_size - needed_size_for_strings;

	for(bound_import_module_list::const_iterator it = imports.begin(); it != imports.end(); ++it)
	{
		const bound_import& import = *it;
		image_bound_import_descriptor descriptor;
		descriptor.NumberOfModuleForwarderRefs = static_cast<uint16_t>(import.get_module_ref_list().size());
		descriptor.OffsetModuleName = static_cast<uint16_t>(current_pos_for_strings - directory_pos);
		descriptor.TimeDateStamp = import.get_timestamp();

		memcpy(&raw_data[current_pos_for_structures], &descriptor, sizeof(descriptor));
		current_pos_for_structures += sizeof(descriptor);
		
		size_t length = import.get_module_name().length() + 1 /* nullbyte */;
		memcpy(&raw_data[current_pos_for_strings], import.get_module_name().c_str(), length);
		current_pos_for_strings += static_cast<uint32_t>(length);

		const bound_import::ref_list& refs = import.get_module_ref_list();
		for(bound_import::ref_list::const_iterator ref_it = refs.begin(); ref_it != refs.end(); ++ref_it)
		{
			const bound_import_ref& ref = *ref_it;
			image_bound_forwarder_ref ref_descriptor = {0};
			ref_descriptor.OffsetModuleName = static_cast<uint16_t>(current_pos_for_strings - directory_pos);
			ref_descriptor.TimeDateStamp = ref.get_timestamp();

			memcpy(&raw_data[current_pos_for_structures], &ref_descriptor, sizeof(ref_descriptor));
			current_pos_for_structures += sizeof(ref_descriptor);

			length = ref.get_module_name().length() + 1 /* nullbyte */;
			memcpy(&raw_data[current_pos_for_strings], ref.get_module_name().c_str(), length);
			current_pos_for_strings += static_cast<uint32_t>(length);
		}
	}

	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(imports_section, auto_strip_last_section);
	
	image_directory ret(pe.rva_from_section_offset(imports_section, directory_pos), needed_size);

	//If auto-rewrite of PE headers is required
	if(save_to_pe_header)
	{
		pe.set_directory_rva(image_directory_entry_bound_import, ret.get_rva());
		pe.set_directory_size(image_directory_entry_bound_import, ret.get_size());
	}

	return ret;
}
}
