#include <string.h>
#include "pe_imports.h"
#include "pe_properties_generic.h"

namespace pe_bliss
{
using namespace pe_win;

//IMPORTS
//Default constructor
//If set_to_pe_headers = true, IMAGE_DIRECTORY_ENTRY_IMPORT entry will be reset
//to new value after import rebuilding
//If auto_zero_directory_entry_iat = true, IMAGE_DIRECTORY_ENTRY_IAT will be set to zero
//IMAGE_DIRECTORY_ENTRY_IAT is used by loader to temporarily make section, where IMAGE_DIRECTORY_ENTRY_IAT RVA points, writeable
//to be able to modify IAT thunks
import_rebuilder_settings::import_rebuilder_settings(bool set_to_pe_headers, bool auto_zero_directory_entry_iat)
	:offset_from_section_start_(0),
	build_original_iat_(true),
	save_iat_and_original_iat_rvas_(true),
	fill_missing_original_iats_(false),
	set_to_pe_headers_(set_to_pe_headers),
	zero_directory_entry_iat_(auto_zero_directory_entry_iat),
	rewrite_iat_and_original_iat_contents_(false),
	auto_strip_last_section_(true)
{}

//Returns offset from section start where import directory data will be placed
uint32_t import_rebuilder_settings::get_offset_from_section_start() const
{
	return offset_from_section_start_;
}

//Returns true if Original import address table (IAT) will be rebuilt
bool import_rebuilder_settings::build_original_iat() const
{
	return build_original_iat_;
}

//Returns true if Original import address and import address tables will not be rebuilt,
//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
bool import_rebuilder_settings::save_iat_and_original_iat_rvas() const
{
	return save_iat_and_original_iat_rvas_;
}

//Returns true if Original import address and import address tables contents will be rewritten
//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
//and save_iat_and_original_iat_rvas is true
bool import_rebuilder_settings::rewrite_iat_and_original_iat_contents() const
{
	return rewrite_iat_and_original_iat_contents_;
}

//Returns true if original missing IATs will be rebuilt
//(only if IATs are saved)
bool import_rebuilder_settings::fill_missing_original_iats() const
{
	return fill_missing_original_iats_;
}

//Returns true if PE headers should be updated automatically after rebuilding of imports
bool import_rebuilder_settings::auto_set_to_pe_headers() const
{
	return set_to_pe_headers_;
}

//Returns true if IMAGE_DIRECTORY_ENTRY_IAT must be zeroed, works only if auto_set_to_pe_headers = true
bool import_rebuilder_settings::zero_directory_entry_iat() const
{
	return zero_directory_entry_iat_;	
}

//Returns true if the last section should be stripped automatically, if imports are inside it
bool import_rebuilder_settings::auto_strip_last_section_enabled() const
{
	return auto_strip_last_section_;
}

//Sets offset from section start where import directory data will be placed
void import_rebuilder_settings::set_offset_from_section_start(uint32_t offset)
{
	offset_from_section_start_ = offset;
}

//Sets if Original import address table (IAT) will be rebuilt
void import_rebuilder_settings::build_original_iat(bool enable)
{
	build_original_iat_ = enable;
}

//Sets if Original import address and import address tables will not be rebuilt,
//works only if import descriptor IAT (and orig.IAT, if present) RVAs are not zero
void import_rebuilder_settings::save_iat_and_original_iat_rvas(bool enable, bool enable_rewrite_iat_and_original_iat_contents)
{
	save_iat_and_original_iat_rvas_ = enable;
	if(save_iat_and_original_iat_rvas_)
		rewrite_iat_and_original_iat_contents_ = enable_rewrite_iat_and_original_iat_contents;
	else
		rewrite_iat_and_original_iat_contents_ = false;
}

//Sets if original missing IATs will be rebuilt
//(only if IATs are saved)
void import_rebuilder_settings::fill_missing_original_iats(bool enable)
{
	fill_missing_original_iats_ = enable;
}

//Sets if PE headers should be updated automatically after rebuilding of imports
void import_rebuilder_settings::auto_set_to_pe_headers(bool enable)
{
	set_to_pe_headers_ = enable;
}

//Sets if IMAGE_DIRECTORY_ENTRY_IAT must be zeroed, works only if auto_set_to_pe_headers = true
void import_rebuilder_settings::zero_directory_entry_iat(bool enable)
{
	zero_directory_entry_iat_ = enable;
}

//Sets if the last section should be stripped automatically, if imports are inside it, default true
void import_rebuilder_settings::enable_auto_strip_last_section(bool enable)
{
	auto_strip_last_section_ = enable;
}

//Default constructor
imported_function::imported_function()
	:hint_(0), ordinal_(0), iat_va_(0)
{}

//Returns name of function
const std::string& imported_function::get_name() const
{
	return name_;
}

//Returns true if imported function has name (and hint)
bool imported_function::has_name() const
{
	return !name_.empty();
}

//Returns hint
uint16_t imported_function::get_hint() const
{
	return hint_;
}

//Returns ordinal of function
uint16_t imported_function::get_ordinal() const
{
	return ordinal_;
}

//Returns IAT entry VA (usable if image has both IAT and original IAT and is bound)
uint64_t imported_function::get_iat_va() const
{
	return iat_va_;
}

//Sets name of function
void imported_function::set_name(const std::string& name)
{
	name_ = name;
}

//Sets hint
void imported_function::set_hint(uint16_t hint)
{
	hint_ = hint;
}

//Sets ordinal
void imported_function::set_ordinal(uint16_t ordinal)
{
	ordinal_ = ordinal;
}

//Sets IAT entry VA (usable if image has both IAT and original IAT and is bound)
void imported_function::set_iat_va(uint64_t va)
{
	iat_va_ = va;
}

//Default constructor
import_library::import_library()
	:rva_to_iat_(0), rva_to_original_iat_(0), timestamp_(0)
{}

//Returns name of library
const std::string& import_library::get_name() const
{
	return name_;
}

//Returns RVA to Import Address Table (IAT)
uint32_t import_library::get_rva_to_iat() const
{
	return rva_to_iat_;
}

//Returns RVA to Original Import Address Table (Original IAT)
uint32_t import_library::get_rva_to_original_iat() const
{
	return rva_to_original_iat_;
}

//Returns timestamp
uint32_t import_library::get_timestamp() const
{
	return timestamp_;
}

//Sets name of library
void import_library::set_name(const std::string& name)
{
	name_ = name;
}

//Sets RVA to Import Address Table (IAT)
void import_library::set_rva_to_iat(uint32_t rva_to_iat)
{
	rva_to_iat_ = rva_to_iat;
}

//Sets RVA to Original Import Address Table (Original IAT)
void import_library::set_rva_to_original_iat(uint32_t rva_to_original_iat)
{
	rva_to_original_iat_ = rva_to_original_iat;
}

//Sets timestamp
void import_library::set_timestamp(uint32_t timestamp)
{
	timestamp_ = timestamp;
}

//Returns imported functions list
const import_library::imported_list& import_library::get_imported_functions() const
{
	return imports_;
}

//Adds imported function
void import_library::add_import(const imported_function& func)
{
	imports_.push_back(func);
}

//Clears imported functions list
void import_library::clear_imports()
{
	imports_.clear();
}

const imported_functions_list get_imported_functions(const pe_base& pe)
{
	return (pe.get_pe_type() == pe_type_32 ?
		get_imported_functions_base<pe_types_class_32>(pe)
		: get_imported_functions_base<pe_types_class_64>(pe));
}

const image_directory rebuild_imports(pe_base& pe, const imported_functions_list& imports, section& import_section, const import_rebuilder_settings& import_settings)
{
	return (pe.get_pe_type() == pe_type_32 ?
		rebuild_imports_base<pe_types_class_32>(pe, imports, import_section, import_settings)
		: rebuild_imports_base<pe_types_class_64>(pe, imports, import_section, import_settings));
}

//Returns imported functions list with related libraries info
template<typename PEClassType>
const imported_functions_list get_imported_functions_base(const pe_base& pe)
{
	imported_functions_list ret;

	//If image has no imports, return empty array
	if(!pe.has_imports())
		return ret;

	unsigned long current_descriptor_pos = pe.get_directory_rva(image_directory_entry_import);
	//Get first IMAGE_IMPORT_DESCRIPTOR
	image_import_descriptor import_descriptor = pe.section_data_from_rva<image_import_descriptor>(current_descriptor_pos, section_data_virtual, true);

	//Iterate them until we reach zero-element
	//We don't need to check correctness of this, because exception will be thrown
	//inside of loop if we go outsize of section
	while(import_descriptor.Name)
	{
		//Get imported library information
		import_library lib;

		unsigned long max_name_length;
		//Get byte count that we have for library name
		if((max_name_length = pe.section_data_length_from_rva(import_descriptor.Name, import_descriptor.Name, section_data_virtual, true)) < 2)
			throw pe_exception("Incorrect import directory", pe_exception::incorrect_import_directory);

		//Get DLL name pointer
		const char* dll_name = pe.section_data_from_rva(import_descriptor.Name, section_data_virtual, true);

		//Check for null-termination
		if(!pe_utils::is_null_terminated(dll_name, max_name_length))
			throw pe_exception("Incorrect import directory", pe_exception::incorrect_import_directory);

		//Set library name
		lib.set_name(dll_name);
		//Set library timestamp
		lib.set_timestamp(import_descriptor.TimeDateStamp);
		//Set library RVA to IAT and original IAT
		lib.set_rva_to_iat(import_descriptor.FirstThunk);
		lib.set_rva_to_original_iat(import_descriptor.OriginalFirstThunk);

		//Get RVA to IAT (it must be filled by loader when loading PE)
		uint32_t current_thunk_rva = import_descriptor.FirstThunk;
		typename PEClassType::BaseSize import_address_table = pe.section_data_from_rva<typename PEClassType::BaseSize>(current_thunk_rva, section_data_virtual, true);

		//Get RVA to original IAT (lookup table), which must handle imported functions names
		//Some linkers leave this pointer zero-filled
		//Such image is valid, but it is not possible to restore imported functions names
		//afted image was loaded, because IAT becomes the only one table
		//containing both function names and function RVAs after loading
		uint32_t current_original_thunk_rva = import_descriptor.OriginalFirstThunk;
		typename PEClassType::BaseSize import_lookup_table = current_original_thunk_rva == 0 ? import_address_table : pe.section_data_from_rva<typename PEClassType::BaseSize>(current_original_thunk_rva, section_data_virtual, true);
		if(current_original_thunk_rva == 0)
			current_original_thunk_rva = current_thunk_rva;

		//List all imported functions for current DLL
		if(import_lookup_table != 0 && import_address_table != 0)
		{
			while(true)
			{
				//Imported function description
				imported_function func;

				//Get VA from IAT
				typename PEClassType::BaseSize address = pe.section_data_from_rva<typename PEClassType::BaseSize>(current_thunk_rva, section_data_virtual, true);
				//Move pointer
				current_thunk_rva += sizeof(typename PEClassType::BaseSize);

				//Jump to next DLL if we finished with this one
				if(!address)
					break;

				func.set_iat_va(address);

				//Get VA from original IAT
				typename PEClassType::BaseSize lookup = pe.section_data_from_rva<typename PEClassType::BaseSize>(current_original_thunk_rva, section_data_virtual, true);
				//Move pointer
				current_original_thunk_rva += sizeof(typename PEClassType::BaseSize);

				//Check if function is imported by ordinal
				if((lookup & PEClassType::ImportSnapFlag) != 0)
				{
					//Set function ordinal
					func.set_ordinal(static_cast<uint16_t>(lookup & 0xffff));
				}
				else
				{
					//Get byte count that we have for function name
					if(lookup > static_cast<uint32_t>(-1) - sizeof(uint16_t))
						throw pe_exception("Incorrect import directory", pe_exception::incorrect_import_directory);

					//Get maximum available length of function name
					if((max_name_length = pe.section_data_length_from_rva(static_cast<uint32_t>(lookup + sizeof(uint16_t)), static_cast<uint32_t>(lookup + sizeof(uint16_t)), section_data_virtual, true)) < 2)
						throw pe_exception("Incorrect import directory", pe_exception::incorrect_import_directory);

					//Get imported function name
					const char* func_name = pe.section_data_from_rva(static_cast<uint32_t>(lookup + sizeof(uint16_t)), section_data_virtual, true);

					//Check for null-termination
					if(!pe_utils::is_null_terminated(func_name, max_name_length))
						throw pe_exception("Incorrect import directory", pe_exception::incorrect_import_directory);

					//HINT in import table is ORDINAL in export table
					uint16_t hint = pe.section_data_from_rva<uint16_t>(static_cast<uint32_t>(lookup), section_data_virtual, true);

					//Save hint and name
					func.set_name(func_name);
					func.set_hint(hint);
				}

				//Add function to list
				lib.add_import(func);
			}
		}

		//Check possible overflow
		if(!pe_utils::is_sum_safe(current_descriptor_pos, sizeof(image_import_descriptor)))
			throw pe_exception("Incorrect import directory", pe_exception::incorrect_import_directory);

		//Go to next library
		current_descriptor_pos += sizeof(image_import_descriptor);
		import_descriptor = pe.section_data_from_rva<image_import_descriptor>(current_descriptor_pos, section_data_virtual, true);

		//Save import information
		ret.push_back(lib);
	}

	//Return resulting list
	return ret;
}


//Simple import directory rebuilder
//You can get all image imports with get_imported_functions() function
//You can use returned value to, for example, add new imported library with some functions
//to the end of list of imported libraries
//To keep PE file working, rebuild its imports with save_iat_and_original_iat_rvas = true (default)
//Don't add new imported functions to existing imported library entries, because this can cause
//rewriting of some used memory (or other IAT/orig.IAT fields) by system loader
//The safest way is just adding import libraries with functions to the end of imported_functions_list array
template<typename PEClassType>
const image_directory rebuild_imports_base(pe_base& pe, const imported_functions_list& imports, section& import_section, const import_rebuilder_settings& import_settings)
{
	//Check that import_section is attached to this PE image
	if(!pe.section_attached(import_section))
		throw pe_exception("Import section must be attached to PE file", pe_exception::section_is_not_attached);

	uint32_t needed_size = 0; //Calculate needed size for import structures and strings
	uint32_t needed_size_for_strings = 0; //Calculate needed size for import strings (library and function names and hints)
	uint32_t size_of_iat = 0; //Size of IAT structures

	needed_size += static_cast<uint32_t>((1 /* ending null descriptor */ + imports.size()) * sizeof(image_import_descriptor));
	
	//Enumerate imported functions
	for(imported_functions_list::const_iterator it = imports.begin(); it != imports.end(); ++it)
	{
		needed_size_for_strings += static_cast<uint32_t>((*it).get_name().length() + 1 /* nullbyte */);

		const import_library::imported_list& funcs = (*it).get_imported_functions();

		//IMAGE_THUNK_DATA
		size_of_iat += static_cast<uint32_t>(sizeof(typename PEClassType::BaseSize) * (1 /*ending null */ + funcs.size()));

		//Enumerate all imported functions in library
		for(import_library::imported_list::const_iterator f = funcs.begin(); f != funcs.end(); ++f)
		{
			if((*f).has_name())
				needed_size_for_strings += static_cast<uint32_t>((*f).get_name().length() + 1 /* nullbyte */ + sizeof(uint16_t) /* hint */);
		}
	}

	if(import_settings.build_original_iat() || import_settings.fill_missing_original_iats())
		needed_size += size_of_iat * 2; //We'll have two similar-sized IATs if we're building original IAT
	else
		needed_size += size_of_iat;

	needed_size += sizeof(typename PEClassType::BaseSize); //Maximum align for IAT and original IAT
	
	//Total needed size for import structures and strings
	needed_size += needed_size_for_strings;

	//Check if import_section is last one. If it's not, check if there's enough place for import data
	if(&import_section != &*(pe.get_image_sections().end() - 1) && 
		(import_section.empty() || pe_utils::align_up(import_section.get_size_of_raw_data(), pe.get_file_alignment()) < needed_size + import_settings.get_offset_from_section_start()))
		throw pe_exception("Insufficient space for import directory", pe_exception::insufficient_space);

	std::string& raw_data = import_section.get_raw_data();

	//This will be done only if image_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + import_settings.get_offset_from_section_start())
		raw_data.resize(needed_size + import_settings.get_offset_from_section_start()); //Expand section raw data
	
	uint32_t current_string_pointer = import_settings.get_offset_from_section_start();/* we will paste structures after strings */
	
	//Position for IAT
	uint32_t current_pos_for_iat = pe_utils::align_up(static_cast<uint32_t>(needed_size_for_strings + import_settings.get_offset_from_section_start() + (1 + imports.size()) * sizeof(image_import_descriptor)), sizeof(typename PEClassType::BaseSize));
	//Position for original IAT
	uint32_t current_pos_for_original_iat = current_pos_for_iat + size_of_iat;
	//Position for import descriptors
	uint32_t current_pos_for_descriptors = needed_size_for_strings + import_settings.get_offset_from_section_start();

	//Build imports
	for(imported_functions_list::const_iterator it = imports.begin(); it != imports.end(); ++it)
	{
		//Create import descriptor
		image_import_descriptor descr;
		memset(&descr, 0, sizeof(descr));
		descr.TimeDateStamp = (*it).get_timestamp(); //Restore timestamp
		descr.Name = pe.rva_from_section_offset(import_section, current_string_pointer); //Library name RVA

		//If we should save IAT for current import descriptor
		bool save_iats_for_this_descriptor = import_settings.save_iat_and_original_iat_rvas() && (*it).get_rva_to_iat() != 0;
		//If we should write original IAT
		bool write_original_iat = (!save_iats_for_this_descriptor && import_settings.build_original_iat()) || import_settings.fill_missing_original_iats();

		//If we should rewrite saved original IAT for current import descriptor (without changing its position)
		bool rewrite_saved_original_iat = save_iats_for_this_descriptor && import_settings.rewrite_iat_and_original_iat_contents() && import_settings.build_original_iat();
		//If we should rewrite saved IAT for current import descriptor (without changing its position)
		bool rewrite_saved_iat = save_iats_for_this_descriptor && import_settings.rewrite_iat_and_original_iat_contents() && (*it).get_rva_to_iat() != 0;

		//Helper values if we're rewriting existing IAT or orig.IAT
		uint32_t original_first_thunk = 0;
		uint32_t first_thunk = 0;

		if(save_iats_for_this_descriptor)
		{
			//If there's no original IAT and we're asked to rebuild missing original IATs
			if(!(*it).get_rva_to_original_iat() && import_settings.fill_missing_original_iats())
				descr.OriginalFirstThunk = import_settings.build_original_iat() ? pe.rva_from_section_offset(import_section, current_pos_for_original_iat) : 0;
			else
				descr.OriginalFirstThunk = import_settings.build_original_iat() ? (*it).get_rva_to_original_iat() : 0;
			
			descr.FirstThunk = (*it).get_rva_to_iat();

			original_first_thunk = descr.OriginalFirstThunk;
			first_thunk = descr.FirstThunk;

			if(rewrite_saved_original_iat)
			{
				if((*it).get_rva_to_original_iat())
					write_original_iat = true;
				else
					rewrite_saved_original_iat = false;
			}

			if(rewrite_saved_iat)
				save_iats_for_this_descriptor = false;
		}
		else
		{
			//We are creating new IAT and original IAT (if needed)
			descr.OriginalFirstThunk = import_settings.build_original_iat() ? pe.rva_from_section_offset(import_section, current_pos_for_original_iat) : 0;
			descr.FirstThunk = pe.rva_from_section_offset(import_section, current_pos_for_iat);
		}
		
		//Save import descriptor
		memcpy(&raw_data[current_pos_for_descriptors], &descr, sizeof(descr));
		current_pos_for_descriptors += sizeof(descr);

		//Save library name
		memcpy(&raw_data[current_string_pointer], (*it).get_name().c_str(), (*it).get_name().length() + 1 /* nullbyte */);
		current_string_pointer += static_cast<uint32_t>((*it).get_name().length() + 1 /* nullbyte */);
		
		//List all imported functions
		const import_library::imported_list& funcs = (*it).get_imported_functions();
		for(import_library::imported_list::const_iterator f = funcs.begin(); f != funcs.end(); ++f)
		{
			if((*f).has_name()) //If function is imported by name
			{
				//Get RVA of IMAGE_IMPORT_BY_NAME
				typename PEClassType::BaseSize rva_of_named_import = pe.rva_from_section_offset(import_section, current_string_pointer);

				if(!save_iats_for_this_descriptor)
				{
					if(write_original_iat)
					{
						//We're creating original IATs - so we can write to IAT saved VA (because IMAGE_IMPORT_BY_NAME will be read
						//by PE loader from original IAT)
						typename PEClassType::BaseSize iat_value = static_cast<typename PEClassType::BaseSize>((*f).get_iat_va());

						if(rewrite_saved_iat)
						{
							if(pe.section_data_length_from_rva(first_thunk, first_thunk, section_data_raw, true) <= sizeof(iat_value))
								throw pe_exception("Insufficient space inside initial IAT", pe_exception::insufficient_space);

							memcpy(pe.section_data_from_rva(first_thunk, true), &iat_value, sizeof(iat_value));

							first_thunk += sizeof(iat_value);
						}
						else
						{
							memcpy(&raw_data[current_pos_for_iat], &iat_value, sizeof(iat_value));
							current_pos_for_iat += sizeof(rva_of_named_import);
						}
					}
					else
					{
						//Else - write to IAT RVA of IMAGE_IMPORT_BY_NAME
						if(rewrite_saved_iat)
						{
							if(pe.section_data_length_from_rva(first_thunk, first_thunk, section_data_raw, true) <= sizeof(rva_of_named_import))
								throw pe_exception("Insufficient space inside initial IAT", pe_exception::insufficient_space);

							memcpy(pe.section_data_from_rva(first_thunk, true), &rva_of_named_import, sizeof(rva_of_named_import));

							first_thunk += sizeof(rva_of_named_import);
						}
						else
						{
							memcpy(&raw_data[current_pos_for_iat], &rva_of_named_import, sizeof(rva_of_named_import));
							current_pos_for_iat += sizeof(rva_of_named_import);
						}
					}
				}

				if(write_original_iat)
				{
					if(rewrite_saved_original_iat)
					{
						if(pe.section_data_length_from_rva(original_first_thunk, original_first_thunk, section_data_raw, true) <= sizeof(rva_of_named_import))
							throw pe_exception("Insufficient space inside initial original IAT", pe_exception::insufficient_space);

						memcpy(pe.section_data_from_rva(original_first_thunk, true), &rva_of_named_import, sizeof(rva_of_named_import));

						original_first_thunk += sizeof(rva_of_named_import);
					}
					else
					{
						//We're creating original IATs
						memcpy(&raw_data[current_pos_for_original_iat], &rva_of_named_import, sizeof(rva_of_named_import));
						current_pos_for_original_iat += sizeof(rva_of_named_import);
					}
				}

				//Write IMAGE_IMPORT_BY_NAME (WORD hint + string function name)
				uint16_t hint = (*f).get_hint();
				memcpy(&raw_data[current_string_pointer], &hint, sizeof(hint));
				memcpy(&raw_data[current_string_pointer + sizeof(uint16_t)], (*f).get_name().c_str(), (*f).get_name().length() + 1 /* nullbyte */);
				current_string_pointer += static_cast<uint32_t>((*f).get_name().length() + 1 /* nullbyte */ + sizeof(uint16_t) /* hint */);
			}
			else //Function is imported by ordinal
			{
				uint16_t ordinal = (*f).get_ordinal();
				typename PEClassType::BaseSize thunk_value = ordinal;
				thunk_value |= PEClassType::ImportSnapFlag; //Imported by ordinal

				if(!save_iats_for_this_descriptor)
				{
					if(write_original_iat)
					{
						//We're creating original IATs - so we can wtire to IAT saved VA (because ordinal will be read
						//by PE loader from original IAT)
						typename PEClassType::BaseSize iat_value = static_cast<typename PEClassType::BaseSize>((*f).get_iat_va());
						if(rewrite_saved_iat)
						{
							if(pe.section_data_length_from_rva(first_thunk, first_thunk, section_data_raw, true) <= sizeof(iat_value))
								throw pe_exception("Insufficient space inside initial IAT", pe_exception::insufficient_space);

							memcpy(pe.section_data_from_rva(first_thunk, true), &iat_value, sizeof(iat_value));

							first_thunk += sizeof(iat_value);
						}
						else
						{
							memcpy(&raw_data[current_pos_for_iat], &iat_value, sizeof(iat_value));
							current_pos_for_iat += sizeof(thunk_value);
						}
					}
					else
					{
						//Else - write ordinal to IAT
						if(rewrite_saved_iat)
						{
							if(pe.section_data_length_from_rva(first_thunk, first_thunk, section_data_raw, true) <= sizeof(thunk_value))
								throw pe_exception("Insufficient space inside initial IAT", pe_exception::insufficient_space);

							memcpy(pe.section_data_from_rva(first_thunk, true), &thunk_value, sizeof(thunk_value));

							first_thunk += sizeof(thunk_value);
						}
						else
						{
							memcpy(&raw_data[current_pos_for_iat], &thunk_value, sizeof(thunk_value));
						}
					}
				}

				//We're writing ordinal to original IAT slot
				if(write_original_iat)
				{
					if(rewrite_saved_original_iat)
					{
						if(pe.section_data_length_from_rva(original_first_thunk, original_first_thunk, section_data_raw, true) <= sizeof(thunk_value))
							throw pe_exception("Insufficient space inside initial original IAT", pe_exception::insufficient_space);

						memcpy(pe.section_data_from_rva(original_first_thunk, true), &thunk_value, sizeof(thunk_value));

						original_first_thunk += sizeof(thunk_value);
					}
					else
					{
						memcpy(&raw_data[current_pos_for_original_iat], &thunk_value, sizeof(thunk_value));
						current_pos_for_original_iat += sizeof(thunk_value);
					}
				}
			}
		}

		if(!save_iats_for_this_descriptor)
		{
			//Ending null thunks
			typename PEClassType::BaseSize thunk_value = 0;

			if(rewrite_saved_iat)
			{
				if(pe.section_data_length_from_rva(first_thunk, first_thunk, section_data_raw, true) <= sizeof(thunk_value))
					throw pe_exception("Insufficient space inside initial IAT", pe_exception::insufficient_space);

				memcpy(pe.section_data_from_rva(first_thunk, true), &thunk_value, sizeof(thunk_value));

				first_thunk += sizeof(thunk_value);
			}
			else
			{
				memcpy(&raw_data[current_pos_for_iat], &thunk_value, sizeof(thunk_value));
				current_pos_for_iat += sizeof(thunk_value);
			}
		}

		if(write_original_iat)
		{
			//Ending null thunks
			typename PEClassType::BaseSize thunk_value = 0;

			if(rewrite_saved_original_iat)
			{
				if(pe.section_data_length_from_rva(original_first_thunk, original_first_thunk, section_data_raw, true) <= sizeof(thunk_value))
					throw pe_exception("Insufficient space inside initial original IAT", pe_exception::insufficient_space);

				memcpy(pe.section_data_from_rva(original_first_thunk, true), &thunk_value, sizeof(thunk_value));

				original_first_thunk += sizeof(thunk_value);
			}
			else
			{
				memcpy(&raw_data[current_pos_for_original_iat], &thunk_value, sizeof(thunk_value));
				current_pos_for_original_iat += sizeof(thunk_value);
			}
		}
	}

	{
		//Null ending descriptor
		image_import_descriptor descr;
		memset(&descr, 0, sizeof(descr));
		memcpy(&raw_data[current_pos_for_descriptors], &descr, sizeof(descr));
	}

	//Strip data a little, if we saved some place
	//We're allocating more space than needed, if present original IAT and IAT are saved
	raw_data.resize(current_pos_for_original_iat);

	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(import_section, import_settings.auto_strip_last_section_enabled());

	//Return information about rebuilt import directory
	image_directory ret(pe.rva_from_section_offset(import_section, import_settings.get_offset_from_section_start() + needed_size_for_strings), needed_size - needed_size_for_strings);

	//If auto-rewrite of PE headers is required
	if(import_settings.auto_set_to_pe_headers())
	{
		pe.set_directory_rva(image_directory_entry_import, ret.get_rva());
		pe.set_directory_size(image_directory_entry_import, ret.get_size());

		//If we are requested to zero IMAGE_DIRECTORY_ENTRY_IAT also
		if(import_settings.zero_directory_entry_iat())
		{
			pe.set_directory_rva(image_directory_entry_iat, 0);
			pe.set_directory_size(image_directory_entry_iat, 0);
		}
	}

	return ret;
}
}
