#include <set>
#include <algorithm>
#include <string.h>
#include "pe_exports.h"
#include "utils.h"

namespace pe_bliss
{
using namespace pe_win;

//EXPORTS
//Default constructor
exported_function::exported_function()
	:ordinal_(0), rva_(0), has_name_(false), name_ordinal_(0), forward_(false)
{}

//Returns ordinal of function (actually, ordinal = hint + ordinal base)
uint16_t exported_function::get_ordinal() const
{
	return ordinal_;
}

//Returns RVA of function
uint32_t exported_function::get_rva() const
{
	return rva_;
}

//Returns name of function
const std::string& exported_function::get_name() const
{
	return name_;
}

//Returns true if function has name and name ordinal
bool exported_function::has_name() const
{
	return has_name_;
}

//Returns name ordinal of function
uint16_t exported_function::get_name_ordinal() const
{
	return name_ordinal_;
}

//Returns true if function is forwarded to other library
bool exported_function::is_forwarded() const
{
	return forward_;
}

//Returns the name of forwarded function
const std::string& exported_function::get_forwarded_name() const
{
	return forward_name_;
}

//Sets ordinal of function
void exported_function::set_ordinal(uint16_t ordinal)
{
	ordinal_ = ordinal;
}

//Sets RVA of function
void exported_function::set_rva(uint32_t rva)
{
	rva_ = rva;
}

//Sets name of function (or clears it, if empty name is passed)
void exported_function::set_name(const std::string& name)
{
	name_ = name;
	has_name_ = !name.empty();
}

//Sets name ordinal
void exported_function::set_name_ordinal(uint16_t name_ordinal)
{
	name_ordinal_ = name_ordinal;
}

//Sets forwarded function name (or clears it, if empty name is passed)
void exported_function::set_forwarded_name(const std::string& name)
{
	forward_name_ = name;
	forward_ = !name.empty();
}

//Default constructor
export_info::export_info()
	:characteristics_(0),
	timestamp_(0),
	major_version_(0),
	minor_version_(0),
	ordinal_base_(0),
	number_of_functions_(0),
	number_of_names_(0),
	address_of_functions_(0),
	address_of_names_(0),
	address_of_name_ordinals_(0)
{}

//Returns characteristics
uint32_t export_info::get_characteristics() const
{
	return characteristics_;
}

//Returns timestamp
uint32_t export_info::get_timestamp() const
{
	return timestamp_;
}

//Returns major version
uint16_t export_info::get_major_version() const
{
	return major_version_;
}

//Returns minor version
uint16_t export_info::get_minor_version() const
{
	return minor_version_;
}

//Returns DLL name
const std::string& export_info::get_name() const
{
	return name_;
}

//Returns ordinal base
uint32_t export_info::get_ordinal_base() const
{
	return ordinal_base_;
}

//Returns number of functions
uint32_t export_info::get_number_of_functions() const
{
	return number_of_functions_;
}

//Returns number of function names
uint32_t export_info::get_number_of_names() const
{
	return number_of_names_;
}

//Returns RVA of function address table
uint32_t export_info::get_rva_of_functions() const
{
	return address_of_functions_;
}

//Returns RVA of function name address table
uint32_t export_info::get_rva_of_names() const
{
	return address_of_names_;
}

//Returns RVA of name ordinals table
uint32_t export_info::get_rva_of_name_ordinals() const
{
	return address_of_name_ordinals_;
}

//Sets characteristics
void export_info::set_characteristics(uint32_t characteristics)
{
	characteristics_ = characteristics;
}

//Sets timestamp
void export_info::set_timestamp(uint32_t timestamp)
{
	timestamp_ = timestamp;
}

//Sets major version
void export_info::set_major_version(uint16_t major_version)
{
	major_version_ = major_version;
}

//Sets minor version
void export_info::set_minor_version(uint16_t minor_version)
{
	minor_version_ = minor_version;
}

//Sets DLL name
void export_info::set_name(const std::string& name)
{
	name_ = name;
}

//Sets ordinal base
void export_info::set_ordinal_base(uint32_t ordinal_base)
{
	ordinal_base_ = ordinal_base;
}

//Sets number of functions
void export_info::set_number_of_functions(uint32_t number_of_functions)
{
	number_of_functions_ = number_of_functions;
}

//Sets number of function names
void export_info::set_number_of_names(uint32_t number_of_names)
{
	number_of_names_ = number_of_names;
}

//Sets RVA of function address table
void export_info::set_rva_of_functions(uint32_t rva_of_functions)
{
	address_of_functions_ = rva_of_functions;
}

//Sets RVA of function name address table
void export_info::set_rva_of_names(uint32_t rva_of_names)
{
	address_of_names_ = rva_of_names;
}

//Sets RVA of name ordinals table
void export_info::set_rva_of_name_ordinals(uint32_t rva_of_name_ordinals)
{
	address_of_name_ordinals_ = rva_of_name_ordinals;
}

const exported_functions_list get_exported_functions(const pe_base& pe, export_info* info);

//Returns array of exported functions
const exported_functions_list get_exported_functions(const pe_base& pe)
{
	return get_exported_functions(pe, 0);
}

//Returns array of exported functions and information about export
const exported_functions_list get_exported_functions(const pe_base& pe, export_info& info)
{
	return get_exported_functions(pe, &info);
}

//Helper: sorts exported function list by ordinals
struct ordinal_sorter
{
public:
		bool operator()(const exported_function& func1, const exported_function& func2) const;
};

//Returns array of exported functions and information about export (if info != 0)
const exported_functions_list get_exported_functions(const pe_base& pe, export_info* info)
{
	//Returned exported functions info array
	std::vector<exported_function> ret;

	if(pe.has_exports())
	{
		//Check the length in bytes of the section containing export directory
		if(pe.section_data_length_from_rva(pe.get_directory_rva(image_directory_entry_export),
			pe.get_directory_rva(image_directory_entry_export), section_data_virtual, true)
			< sizeof(image_export_directory))
			throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

		image_export_directory exports = pe.section_data_from_rva<image_export_directory>(pe.get_directory_rva(image_directory_entry_export), section_data_virtual, true);

		unsigned long max_name_length;

		if(info)
		{
			//Save some export info data
			info->set_characteristics(exports.Characteristics);
			info->set_major_version(exports.MajorVersion);
			info->set_minor_version(exports.MinorVersion);

			//Get byte count that we have for dll name
			if((max_name_length = pe.section_data_length_from_rva(exports.Name, exports.Name, section_data_virtual, true)) < 2)
				throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

			//Get dll name pointer
			const char* dll_name = pe.section_data_from_rva(exports.Name, section_data_virtual, true);

			//Check for null-termination
			if(!pe_utils::is_null_terminated(dll_name, max_name_length))
				throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

			//Save the rest of export information data
			info->set_name(dll_name);
			info->set_number_of_functions(exports.NumberOfFunctions);
			info->set_number_of_names(exports.NumberOfNames);
			info->set_ordinal_base(exports.Base);
			info->set_rva_of_functions(exports.AddressOfFunctions);
			info->set_rva_of_names(exports.AddressOfNames);
			info->set_rva_of_name_ordinals(exports.AddressOfNameOrdinals);
			info->set_timestamp(exports.TimeDateStamp);
		}

		if(!exports.NumberOfFunctions)
			return ret;

		//Check IMAGE_EXPORT_DIRECTORY fields
		if(exports.NumberOfNames > exports.NumberOfFunctions)
			throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

		//Check some export directory fields
		if((!exports.AddressOfNameOrdinals && exports.AddressOfNames) ||
			(exports.AddressOfNameOrdinals && !exports.AddressOfNames) ||
			!exports.AddressOfFunctions
			|| exports.NumberOfFunctions >= pe_utils::max_dword / sizeof(uint32_t)
			|| exports.NumberOfNames > pe_utils::max_dword / sizeof(uint32_t)
			|| !pe_utils::is_sum_safe(exports.AddressOfFunctions, exports.NumberOfFunctions * sizeof(uint32_t))
			|| !pe_utils::is_sum_safe(exports.AddressOfNames, exports.NumberOfNames * sizeof(uint32_t))
			|| !pe_utils::is_sum_safe(exports.AddressOfNameOrdinals, exports.NumberOfFunctions * sizeof(uint32_t))
			|| !pe_utils::is_sum_safe(pe.get_directory_rva(image_directory_entry_export), pe.get_directory_size(image_directory_entry_export)))
			throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

		//Check if it is enough bytes to hold AddressOfFunctions table
		if(pe.section_data_length_from_rva(exports.AddressOfFunctions, exports.AddressOfFunctions, section_data_virtual, true)
			< exports.NumberOfFunctions * sizeof(uint32_t))
			throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

		if(exports.AddressOfNames)
		{
			//Check if it is enough bytes to hold name and ordinal tables
			if(pe.section_data_length_from_rva(exports.AddressOfNameOrdinals, exports.AddressOfNameOrdinals, section_data_virtual, true)
				< exports.NumberOfNames * sizeof(uint16_t))
				throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

			if(pe.section_data_length_from_rva(exports.AddressOfNames, exports.AddressOfNames, section_data_virtual, true)
				< exports.NumberOfNames * sizeof(uint32_t))
				throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);
		}
		
		for(uint32_t ordinal = 0; ordinal < exports.NumberOfFunctions; ordinal++)
		{
			//Get function address
			//Sum and multiplication are safe (checked above)
			uint32_t rva = pe.section_data_from_rva<uint32_t>(exports.AddressOfFunctions + ordinal * sizeof(uint32_t), section_data_virtual, true);

			//If we have a skip
			if(!rva)
				continue;

			exported_function func;
			func.set_rva(rva);

			if(!pe_utils::is_sum_safe(exports.Base, ordinal) || exports.Base + ordinal > pe_utils::max_word)
				throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

			func.set_ordinal(static_cast<uint16_t>(ordinal + exports.Base));

			//Scan for function name ordinal
			for(uint32_t i = 0; i < exports.NumberOfNames; i++)
			{
				uint16_t ordinal2 = pe.section_data_from_rva<uint16_t>(exports.AddressOfNameOrdinals + i * sizeof(uint16_t), section_data_virtual, true);

				//If function has name (and name ordinal)
				if(ordinal == ordinal2)
				{
					//Get function name
					//Sum and multiplication are safe (checked above)
					uint32_t function_name_rva = pe.section_data_from_rva<uint32_t>(exports.AddressOfNames + i * sizeof(uint32_t), section_data_virtual, true);

					//Get byte count that we have for function name
					if((max_name_length = pe.section_data_length_from_rva(function_name_rva, function_name_rva, section_data_virtual, true)) < 2)
						throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

					//Get function name pointer
					const char* func_name = pe.section_data_from_rva(function_name_rva, section_data_virtual, true);

					//Check for null-termination
					if(!pe_utils::is_null_terminated(func_name, max_name_length))
						throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

					//Save function info
					func.set_name(func_name);
					func.set_name_ordinal(ordinal2);

					//If the function is just a redirect, save its name
					if(rva >= pe.get_directory_rva(image_directory_entry_export) + sizeof(image_directory_entry_export) &&
						rva < pe.get_directory_rva(image_directory_entry_export) + pe.get_directory_size(image_directory_entry_export))
					{
						if((max_name_length = pe.section_data_length_from_rva(rva, rva, section_data_virtual, true)) < 2)
							throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

						//Get forwarded function name pointer
						const char* forwarded_func_name = pe.section_data_from_rva(rva, section_data_virtual, true);

						//Check for null-termination
						if(!pe_utils::is_null_terminated(forwarded_func_name, max_name_length))
							throw pe_exception("Incorrect export directory", pe_exception::incorrect_export_directory);

						//Set the name of forwarded function
						func.set_forwarded_name(forwarded_func_name);
					}

					break;
				}
			}

			//Add function info to output array
			ret.push_back(func);
		}
	}

	return ret;
}

//Helper export functions
//Returns pair: <ordinal base for supplied functions; maximum ordinal value for supplied functions>
const std::pair<uint16_t, uint16_t> get_export_ordinal_limits(const exported_functions_list& exports)
{
	if(exports.empty())
		return std::make_pair(0, 0);

	uint16_t max_ordinal = 0; //Maximum ordinal number
	uint16_t ordinal_base = pe_utils::max_word; //Minimum ordinal value
	for(exported_functions_list::const_iterator it = exports.begin(); it != exports.end(); ++it)
	{
		const exported_function& func = (*it);

		//Calculate maximum and minimum ordinal numbers
		max_ordinal = std::max<uint16_t>(max_ordinal, func.get_ordinal());
		ordinal_base = std::min<uint16_t>(ordinal_base, func.get_ordinal());
	}

	return std::make_pair(ordinal_base, max_ordinal);
}

//Checks if exported function name already exists
bool exported_name_exists(const std::string& function_name, const exported_functions_list& exports)
{
	for(exported_functions_list::const_iterator it = exports.begin(); it != exports.end(); ++it)
	{
		if((*it).has_name() && (*it).get_name() == function_name)
			return true;
	}

	return false;
}

//Checks if exported function name already exists
bool exported_ordinal_exists(uint16_t ordinal, const exported_functions_list& exports)
{
	for(exported_functions_list::const_iterator it = exports.begin(); it != exports.end(); ++it)
	{
		if((*it).get_ordinal() == ordinal)
			return true;
	}

	return false;
}

//Helper: sorts exported function list by ordinals
bool ordinal_sorter::operator()(const exported_function& func1, const exported_function& func2) const
{
	return func1.get_ordinal() < func2.get_ordinal();
}

//Export directory rebuilder
//info - export information
//exported_functions_list - list of exported functions
//exports_section - section where export directory will be placed (must be attached to PE image)
//offset_from_section_start - offset from exports_section raw data start
//save_to_pe_headers - if true, new export directory information will be saved to PE image headers
//auto_strip_last_section - if true and exports are placed in the last section, it will be automatically stripped
//number_of_functions and number_of_names parameters don't matter in "info" when rebuilding, they're calculated independently
//characteristics, major_version, minor_version, timestamp and name are the only used members of "info" structure
//Returns new export directory information
//exported_functions_list is copied intentionally to be sorted by ordinal values later
//Name ordinals in exported function don't matter, they will be recalculated
const image_directory rebuild_exports(pe_base& pe, const export_info& info, exported_functions_list exports, section& exports_section, uint32_t offset_from_section_start, bool save_to_pe_header, bool auto_strip_last_section)
{
	//Check that exports_section is attached to this PE image
	if(!pe.section_attached(exports_section))
		throw pe_exception("Exports section must be attached to PE file", pe_exception::section_is_not_attached);

	//Needed space for strings
	uint32_t needed_size_for_strings = static_cast<uint32_t>(info.get_name().length() + 1);
	uint32_t number_of_names = 0; //Number of named functions
	uint32_t max_ordinal = 0; //Maximum ordinal number
	uint32_t ordinal_base = static_cast<uint32_t>(-1); //Minimum ordinal value
	
	if(exports.empty())
		ordinal_base = info.get_ordinal_base();

	uint32_t needed_size_for_function_names = 0; //Needed space for function name strings
	uint32_t needed_size_for_function_forwards = 0; //Needed space for function forwards names
	
	//List all exported functions
	//Calculate needed size for function list
	{
		//Also check that there're no duplicate names and ordinals
		std::set<std::string> used_function_names;
		std::set<uint16_t> used_function_ordinals;

		for(exported_functions_list::const_iterator it = exports.begin(); it != exports.end(); ++it)
		{
			const exported_function& func = (*it);
			//Calculate maximum and minimum ordinal numbers
			max_ordinal = std::max<uint32_t>(max_ordinal, func.get_ordinal());
			ordinal_base = std::min<uint32_t>(ordinal_base, func.get_ordinal());

			//Check if ordinal is unique
			if(!used_function_ordinals.insert(func.get_ordinal()).second)
				throw pe_exception("Duplicate exported function ordinal", pe_exception::duplicate_exported_function_ordinal);
			
			if(func.has_name())
			{
				//If function is named
				++number_of_names;
				needed_size_for_function_names += static_cast<uint32_t>(func.get_name().length() + 1);
				
				//Check if it's name and name ordinal are unique
				if(!used_function_names.insert(func.get_name()).second)
					throw pe_exception("Duplicate exported function name", pe_exception::duplicate_exported_function_name);
			}

			//If function is forwarded to another DLL
			if(func.is_forwarded())
				needed_size_for_function_forwards += static_cast<uint32_t>(func.get_forwarded_name().length() + 1);
		}
	}
	
	//Sort functions by ordinal value
	std::sort(exports.begin(), exports.end(), ordinal_sorter());

	//Calculate needed space for different things...
	needed_size_for_strings += needed_size_for_function_names;
	needed_size_for_strings += needed_size_for_function_forwards;
	uint32_t needed_size_for_function_name_ordinals = number_of_names * sizeof(uint16_t);
	uint32_t needed_size_for_function_name_rvas = number_of_names * sizeof(uint32_t);
	uint32_t needed_size_for_function_addresses = (max_ordinal - ordinal_base + 1) * sizeof(uint32_t);
	
	//Export directory header will be placed first
	uint32_t directory_pos = pe_utils::align_up(offset_from_section_start, sizeof(uint32_t));

	uint32_t needed_size = sizeof(image_export_directory); //Calculate needed size for export tables and strings
	//sizeof(IMAGE_EXPORT_DIRECTORY) = export directory header

	//Total needed space...
	needed_size += needed_size_for_function_name_ordinals; //For list of names ordinals
	needed_size += needed_size_for_function_addresses; //For function RVAs
	needed_size += needed_size_for_strings; //For all strings
	needed_size += needed_size_for_function_name_rvas; //For function name strings RVAs

	//Check if exports_section is last one. If it's not, check if there's enough place for exports data
	if(&exports_section != &*(pe.get_image_sections().end() - 1) && 
		(exports_section.empty() || pe_utils::align_up(exports_section.get_size_of_raw_data(), pe.get_file_alignment()) < needed_size + directory_pos))
		throw pe_exception("Insufficient space for export directory", pe_exception::insufficient_space);

	std::string& raw_data = exports_section.get_raw_data();

	//This will be done only if exports_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + directory_pos)
		raw_data.resize(needed_size + directory_pos); //Expand section raw data

	//Library name will be placed after it
	uint32_t current_pos_of_function_names = static_cast<uint32_t>(info.get_name().length() + 1 + directory_pos + sizeof(image_export_directory));
	//Next - function names
	uint32_t current_pos_of_function_name_ordinals = current_pos_of_function_names + needed_size_for_function_names;
	//Next - function name ordinals
	uint32_t current_pos_of_function_forwards = current_pos_of_function_name_ordinals + needed_size_for_function_name_ordinals;
	//Finally - function addresses
	uint32_t current_pos_of_function_addresses = current_pos_of_function_forwards + needed_size_for_function_forwards;
	//Next - function names RVAs
	uint32_t current_pos_of_function_names_rvas = current_pos_of_function_addresses + needed_size_for_function_addresses;

	{
		//Create export directory and fill it
		image_export_directory dir = {0};
		dir.Characteristics = info.get_characteristics();
		dir.MajorVersion = info.get_major_version();
		dir.MinorVersion = info.get_minor_version();
		dir.TimeDateStamp = info.get_timestamp();
		dir.NumberOfFunctions = max_ordinal - ordinal_base + 1;
		dir.NumberOfNames = number_of_names;
		dir.Base = ordinal_base;
		dir.AddressOfFunctions = pe.rva_from_section_offset(exports_section, current_pos_of_function_addresses);
		dir.AddressOfNameOrdinals = pe.rva_from_section_offset(exports_section, current_pos_of_function_name_ordinals);
		dir.AddressOfNames = pe.rva_from_section_offset(exports_section, current_pos_of_function_names_rvas);
		dir.Name = pe.rva_from_section_offset(exports_section, directory_pos + sizeof(image_export_directory));

		//Save it
		memcpy(&raw_data[directory_pos], &dir, sizeof(dir));
	}

	//Sve library name
	memcpy(&raw_data[directory_pos + sizeof(image_export_directory)], info.get_name().c_str(), info.get_name().length() + 1);

	//A map to sort function names alphabetically
	typedef std::map<std::string, uint16_t> funclist; //function name; function name ordinal
	funclist funcs;

	uint32_t last_ordinal = ordinal_base;
	//Enumerate all exported functions
	for(exported_functions_list::const_iterator it = exports.begin(); it != exports.end(); ++it)
	{
		const exported_function& func = (*it);

		//If we're skipping some ordinals...
		if(func.get_ordinal() > last_ordinal)
		{
			//Fill this function RVAs data with zeros
			uint32_t len = sizeof(uint32_t) * (func.get_ordinal() - last_ordinal - 1);
			if(len)
			{
				memset(&raw_data[current_pos_of_function_addresses], 0, len);
				current_pos_of_function_addresses += len;
			}
			
			//Save last encountered ordinal
			last_ordinal = func.get_ordinal();
		}
		
		//If function is named, save its name ordinal and name in sorted alphabetically order
		if(func.has_name())
			funcs.insert(std::make_pair(func.get_name(), static_cast<uint16_t>(func.get_ordinal() - ordinal_base))); //Calculate name ordinal

		//If function is forwarded to another DLL
		if(func.is_forwarded())
		{
			//Write its forwarded name and its RVA
			uint32_t function_rva = pe.rva_from_section_offset(exports_section, current_pos_of_function_forwards);
			memcpy(&raw_data[current_pos_of_function_addresses], &function_rva, sizeof(function_rva));
			current_pos_of_function_addresses += sizeof(function_rva);

			memcpy(&raw_data[current_pos_of_function_forwards], func.get_forwarded_name().c_str(), func.get_forwarded_name().length() + 1);
			current_pos_of_function_forwards += static_cast<uint32_t>(func.get_forwarded_name().length() + 1);
		}
		else
		{
			//Write actual function RVA
			uint32_t function_rva = func.get_rva();
			memcpy(&raw_data[current_pos_of_function_addresses], &function_rva, sizeof(function_rva));
			current_pos_of_function_addresses += sizeof(function_rva);
		}
	}
	
	//Enumerate sorted function names
	for(funclist::const_iterator it = funcs.begin(); it != funcs.end(); ++it)
	{
		//Save function name RVA
		uint32_t function_name_rva = pe.rva_from_section_offset(exports_section, current_pos_of_function_names);
		memcpy(&raw_data[current_pos_of_function_names_rvas], &function_name_rva, sizeof(function_name_rva));
		current_pos_of_function_names_rvas += sizeof(function_name_rva);

		//Save function name
		memcpy(&raw_data[current_pos_of_function_names], (*it).first.c_str(), (*it).first.length() + 1);
		current_pos_of_function_names += static_cast<uint32_t>((*it).first.length() + 1);

		//Save function name ordinal
		uint16_t name_ordinal = (*it).second;
		memcpy(&raw_data[current_pos_of_function_name_ordinals], &name_ordinal, sizeof(name_ordinal));
		current_pos_of_function_name_ordinals += sizeof(name_ordinal);
	}
	
	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(exports_section, auto_strip_last_section);
	
	image_directory ret(pe.rva_from_section_offset(exports_section, directory_pos), needed_size);

	//If auto-rewrite of PE headers is required
	if(save_to_pe_header)
	{
		pe.set_directory_rva(image_directory_entry_export, ret.get_rva());
		pe.set_directory_size(image_directory_entry_export, ret.get_size());
	}

	return ret;
}
}
