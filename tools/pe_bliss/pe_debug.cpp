#include <string.h>
#include "pe_debug.h"
#include "utils.h"

namespace pe_bliss
{
using namespace pe_win;
//DEBUG
//Default constructor
debug_info::debug_info()
	:characteristics_(0),
	time_stamp_(0),
	major_version_(0), minor_version_(0),
	type_(0),
	size_of_data_(0),
	address_of_raw_data_(0),
	pointer_to_raw_data_(0),
	advanced_info_type_(advanced_info_none)
{}

//Constructor from data
debug_info::debug_info(const image_debug_directory& debug)
	:characteristics_(debug.Characteristics),
	time_stamp_(debug.TimeDateStamp),
	major_version_(debug.MajorVersion), minor_version_(debug.MinorVersion),
	type_(debug.Type),
	size_of_data_(debug.SizeOfData),
	address_of_raw_data_(debug.AddressOfRawData),
	pointer_to_raw_data_(debug.PointerToRawData),
	advanced_info_type_(advanced_info_none)
{}

//Returns debug characteristics
uint32_t debug_info::get_characteristics() const
{
	return characteristics_;
}

//Returns debug datetimestamp
uint32_t debug_info::get_time_stamp() const
{
	return time_stamp_;
}

//Returns major version
uint32_t debug_info::get_major_version() const
{
	return major_version_;
}

//Returns minor version
uint32_t debug_info::get_minor_version() const
{
	return minor_version_;
}

//Returns type of debug info (unchecked)
uint32_t debug_info::get_type_raw() const
{
	return type_;
}

//Returns type of debug info from debug_info_type enumeration
debug_info::debug_info_type debug_info::get_type() const
{
	//Determine debug type
	switch(type_)
	{
	case image_debug_type_coff:
		return debug_type_coff;

	case image_debug_type_codeview:
		return debug_type_codeview;

	case image_debug_type_fpo:
		return debug_type_fpo;

	case image_debug_type_misc:
		return debug_type_misc;

	case image_debug_type_exception:
		return debug_type_exception;

	case image_debug_type_fixup:
		return debug_type_fixup;

	case image_debug_type_omap_to_src:
		return debug_type_omap_to_src;

	case image_debug_type_omap_from_src:
		return debug_type_omap_from_src;

	case image_debug_type_borland:
		return debug_type_borland;

	case image_debug_type_clsid:
		return debug_type_clsid;

	case image_debug_type_reserved10:
		return debug_type_reserved10;
	}

	return debug_type_unknown;
}

//Returns size of debug data (internal, .pdb or other file doesn't count)
uint32_t debug_info::get_size_of_data() const
{
	return size_of_data_;
}

//Returns RVA of debug info when mapped to memory or zero, if info is not mapped
uint32_t debug_info::get_rva_of_raw_data() const
{
	return address_of_raw_data_;
}

//Returns raw file pointer to raw data
uint32_t debug_info::get_pointer_to_raw_data() const
{
	return pointer_to_raw_data_;
}

//Copy constructor
debug_info::debug_info(const debug_info& info)
	:characteristics_(info.characteristics_),
	time_stamp_(info.time_stamp_),
	major_version_(info.major_version_), minor_version_(info.minor_version_),
	type_(info.type_),
	size_of_data_(info.size_of_data_),
	address_of_raw_data_(info.address_of_raw_data_),
	pointer_to_raw_data_(info.pointer_to_raw_data_),
	advanced_info_type_(info.advanced_info_type_)
{
	copy_advanced_info(info);
}

//Copy assignment operator
debug_info& debug_info::operator=(const debug_info& info)
{
	copy_advanced_info(info);

	characteristics_ = info.characteristics_;
	time_stamp_ = info.time_stamp_;
	major_version_ = info.major_version_;
	minor_version_ = info.minor_version_;
	type_ = info.type_;
	size_of_data_ = info.size_of_data_;
	address_of_raw_data_ = info.address_of_raw_data_;
	pointer_to_raw_data_ = info.pointer_to_raw_data_;
	advanced_info_type_ = info.advanced_info_type_;

	return *this;
}

//Default constructor
debug_info::advanced_info::advanced_info()
	:adv_pdb_7_0_info(0) //Zero pointer to advanced data
{}

//Returns true if advanced debug info is present
bool debug_info::advanced_info::is_present() const
{
	return adv_pdb_7_0_info != 0;
}

//Helper for advanced debug information copying
void debug_info::copy_advanced_info(const debug_info& info)
{
	free_present_advanced_info();

	switch(info.advanced_info_type_)
	{
	case advanced_info_pdb_7_0:
		advanced_debug_info_.adv_pdb_7_0_info = new pdb_7_0_info(*info.advanced_debug_info_.adv_pdb_7_0_info);
		break;
	case advanced_info_pdb_2_0:
		advanced_debug_info_.adv_pdb_2_0_info = new pdb_2_0_info(*info.advanced_debug_info_.adv_pdb_2_0_info);
		break;
	case advanced_info_misc:
		advanced_debug_info_.adv_misc_info = new misc_debug_info(*info.advanced_debug_info_.adv_misc_info);
		break;
	case advanced_info_coff:
		advanced_debug_info_.adv_coff_info = new coff_debug_info(*info.advanced_debug_info_.adv_coff_info);
		break;
	default:
		break;
	}

	advanced_info_type_ = info.advanced_info_type_;
}

//Helper for clearing any present advanced debug information
void debug_info::free_present_advanced_info()
{
	switch(advanced_info_type_)
	{
	case advanced_info_pdb_7_0:
		delete advanced_debug_info_.adv_pdb_7_0_info;
		break;
	case advanced_info_pdb_2_0:
		delete advanced_debug_info_.adv_pdb_2_0_info;
		break;
	case advanced_info_misc:
		delete advanced_debug_info_.adv_misc_info;
		break;
	case advanced_info_coff:
		delete advanced_debug_info_.adv_coff_info;
		break;
	default:
		break;
	}

	advanced_debug_info_.adv_pdb_7_0_info = 0;
	advanced_info_type_ = advanced_info_none;
}

//Destructor
debug_info::~debug_info()
{
	free_present_advanced_info();
}

//Sets advanced debug information
void debug_info::set_advanced_debug_info(const pdb_7_0_info& info)
{
	free_present_advanced_info();
	advanced_debug_info_.adv_pdb_7_0_info = new pdb_7_0_info(info);
	advanced_info_type_ = advanced_info_pdb_7_0;
}

void debug_info::set_advanced_debug_info(const pdb_2_0_info& info)
{
	free_present_advanced_info();
	advanced_debug_info_.adv_pdb_2_0_info = new pdb_2_0_info(info);
	advanced_info_type_ = advanced_info_pdb_2_0;
}

void debug_info::set_advanced_debug_info(const misc_debug_info& info)
{
	free_present_advanced_info();
	advanced_debug_info_.adv_misc_info = new misc_debug_info(info);
	advanced_info_type_ = advanced_info_misc;
}

void debug_info::set_advanced_debug_info(const coff_debug_info& info)
{
	free_present_advanced_info();
	advanced_debug_info_.adv_coff_info = new coff_debug_info(info);
	advanced_info_type_ = advanced_info_coff;
}

//Returns advanced debug information type
debug_info::advanced_info_type debug_info::get_advanced_info_type() const
{
	return advanced_info_type_;
}

//Returns advanced debug information or throws an exception,
//if requested information type is not contained by structure
template<>
const pdb_7_0_info debug_info::get_advanced_debug_info<pdb_7_0_info>() const
{
	if(advanced_info_type_ != advanced_info_pdb_7_0)
		throw pe_exception("Debug info structure does not contain PDB 7.0 data", pe_exception::advanced_debug_information_request_error);

	return *advanced_debug_info_.adv_pdb_7_0_info;
}

template<>
const pdb_2_0_info debug_info::get_advanced_debug_info<pdb_2_0_info>() const
{
	if(advanced_info_type_ != advanced_info_pdb_2_0)
		throw pe_exception("Debug info structure does not contain PDB 2.0 data", pe_exception::advanced_debug_information_request_error);

	return *advanced_debug_info_.adv_pdb_2_0_info;
}

template<>
const misc_debug_info debug_info::get_advanced_debug_info<misc_debug_info>() const
{
	if(advanced_info_type_ != advanced_info_misc)
		throw pe_exception("Debug info structure does not contain MISC data", pe_exception::advanced_debug_information_request_error);

	return *advanced_debug_info_.adv_misc_info;
}

template<>
const coff_debug_info debug_info::get_advanced_debug_info<coff_debug_info>() const
{
	if(advanced_info_type_ != advanced_info_coff)
		throw pe_exception("Debug info structure does not contain COFF data", pe_exception::advanced_debug_information_request_error);

	return *advanced_debug_info_.adv_coff_info;
}

//Sets advanced debug information type, if no advanced info structure available
void debug_info::set_advanced_info_type(advanced_info_type type)
{
	free_present_advanced_info();
	if(advanced_info_type_ >= advanced_info_codeview_4_0) //Don't set info type for those types, which have advanced info structures
		advanced_info_type_ = type;
}

//Default constructor
pdb_7_0_info::pdb_7_0_info()
	:age_(0)
{
	memset(&guid_, 0, sizeof(guid_));
}

//Constructor from data
pdb_7_0_info::pdb_7_0_info(const CV_INFO_PDB70* info)
	:age_(info->Age), guid_(info->Signature),
	pdb_file_name_(reinterpret_cast<const char*>(info->PdbFileName)) //Must be checked before for null-termination
{}

//Returns debug PDB 7.0 structure GUID
const guid pdb_7_0_info::get_guid() const
{
	return guid_;
}

//Returns age of build
uint32_t pdb_7_0_info::get_age() const
{
	return age_;
}

//Returns PDB file name / path
const std::string& pdb_7_0_info::get_pdb_file_name() const
{
	return pdb_file_name_;
}

//Default constructor
pdb_2_0_info::pdb_2_0_info()
	:age_(0), signature_(0)
{}

//Constructor from data
pdb_2_0_info::pdb_2_0_info(const CV_INFO_PDB20* info)
	:age_(info->Age), signature_(info->Signature),
	pdb_file_name_(reinterpret_cast<const char*>(info->PdbFileName)) //Must be checked before for null-termination
{}

//Returns debug PDB 2.0 structure signature
uint32_t pdb_2_0_info::get_signature() const
{
	return signature_;
}

//Returns age of build
uint32_t pdb_2_0_info::get_age() const
{
	return age_;
}

//Returns PDB file name / path
const std::string& pdb_2_0_info::get_pdb_file_name() const
{
	return pdb_file_name_;
}

//Default constructor
misc_debug_info::misc_debug_info()
	:data_type_(0), unicode_(false)
{}

//Constructor from data
misc_debug_info::misc_debug_info(const image_debug_misc* info)
	:data_type_(info->DataType), unicode_(info->Unicode ? true : false)
{
	//IMAGE_DEBUG_MISC::Data must be checked before!
	if(info->Unicode)
	{
#ifdef PE_BLISS_WINDOWS
		debug_data_unicode_ = std::wstring(reinterpret_cast<const wchar_t*>(info->Data), (info->Length - sizeof(image_debug_misc) + 1 /* BYTE[1] in the end of structure */) / 2);
#else
		debug_data_unicode_ = pe_utils::from_ucs2(u16string(reinterpret_cast<const unicode16_t*>(info->Data), (info->Length - sizeof(image_debug_misc) + 1 /* BYTE[1] in the end of structure */) / 2));
#endif
		
		pe_utils::strip_nullbytes(debug_data_unicode_); //Strip nullbytes in the end of string
	}
	else
	{
		debug_data_ansi_ = std::string(reinterpret_cast<const char*>(info->Data), info->Length - sizeof(image_debug_misc) + 1 /* BYTE[1] in the end of structure */);
		pe_utils::strip_nullbytes(debug_data_ansi_); //Strip nullbytes in the end of string
	}
}

//Returns debug data type
uint32_t misc_debug_info::get_data_type() const
{
	return data_type_;
}

//Returns true if data type is exe name
bool misc_debug_info::is_exe_name() const
{
	return data_type_ == image_debug_misc_exename;
}

//Returns true if debug data is UNICODE
bool misc_debug_info::is_unicode() const
{
	return unicode_;
}

//Returns debug data (ANSI)
const std::string& misc_debug_info::get_data_ansi() const
{
	return debug_data_ansi_;
}

//Returns debug data (UNICODE)
const std::wstring& misc_debug_info::get_data_unicode() const
{
	return debug_data_unicode_;
}

//Default constructor
coff_debug_info::coff_debug_info()
	:number_of_symbols_(0),
	lva_to_first_symbol_(0),
	number_of_line_numbers_(0),
	lva_to_first_line_number_(0),
	rva_to_first_byte_of_code_(0),
	rva_to_last_byte_of_code_(0),
	rva_to_first_byte_of_data_(0),
	rva_to_last_byte_of_data_(0)
{}

//Constructor from data
coff_debug_info::coff_debug_info(const image_coff_symbols_header* info)
	:number_of_symbols_(info->NumberOfSymbols),
	lva_to_first_symbol_(info->LvaToFirstSymbol),
	number_of_line_numbers_(info->NumberOfLinenumbers),
	lva_to_first_line_number_(info->LvaToFirstLinenumber),
	rva_to_first_byte_of_code_(info->RvaToFirstByteOfCode),
	rva_to_last_byte_of_code_(info->RvaToLastByteOfCode),
	rva_to_first_byte_of_data_(info->RvaToFirstByteOfData),
	rva_to_last_byte_of_data_(info->RvaToLastByteOfData)
{}

//Returns number of symbols
uint32_t coff_debug_info::get_number_of_symbols() const
{
	return number_of_symbols_;
}

//Returns virtual address of the first symbol
uint32_t coff_debug_info::get_lva_to_first_symbol() const
{
	return lva_to_first_symbol_;
}

//Returns number of line-number entries
uint32_t coff_debug_info::get_number_of_line_numbers() const
{
	return number_of_line_numbers_;
}

//Returns virtual address of the first line-number entry
uint32_t coff_debug_info::get_lva_to_first_line_number() const
{
	return lva_to_first_line_number_;
}

//Returns relative virtual address of the first byte of code
uint32_t coff_debug_info::get_rva_to_first_byte_of_code() const
{
	return rva_to_first_byte_of_code_;
}

//Returns relative virtual address of the last byte of code
uint32_t coff_debug_info::get_rva_to_last_byte_of_code() const
{
	return rva_to_last_byte_of_code_;
}

//Returns relative virtual address of the first byte of data
uint32_t coff_debug_info::get_rva_to_first_byte_of_data() const
{
	return rva_to_first_byte_of_data_;
}

//Returns relative virtual address of the last byte of data
uint32_t coff_debug_info::get_rva_to_last_byte_of_data() const
{
	return rva_to_last_byte_of_data_;
}

//Returns COFF symbols list
const coff_debug_info::coff_symbols_list& coff_debug_info::get_symbols() const
{
	return symbols_;
}

//Adds COFF symbol
void coff_debug_info::add_symbol(const coff_symbol& sym)
{
	symbols_.push_back(sym);
}

//Default constructor
coff_debug_info::coff_symbol::coff_symbol()
	:storage_class_(0),
	index_(0),
	section_number_(0), rva_(0),
	type_(0),
	is_filename_(false)
{}

//Returns storage class
uint32_t coff_debug_info::coff_symbol::get_storage_class() const
{
	return storage_class_;
}

//Returns symbol index
uint32_t coff_debug_info::coff_symbol::get_index() const
{
	return index_;
}

//Returns section number
uint32_t coff_debug_info::coff_symbol::get_section_number() const
{
	return section_number_;
}

//Returns RVA
uint32_t coff_debug_info::coff_symbol::get_rva() const
{
	return rva_;
}

//Returns true if structure contains file name
bool coff_debug_info::coff_symbol::is_file() const
{
	return is_filename_;
}

//Returns text data (symbol or file name)
const std::string& coff_debug_info::coff_symbol::get_symbol() const
{
	return name_;
}

//Sets storage class
void coff_debug_info::coff_symbol::set_storage_class(uint32_t storage_class)
{
	storage_class_ = storage_class;
}

//Sets symbol index
void coff_debug_info::coff_symbol::set_index(uint32_t index)
{
	index_ = index;
}

//Sets section number
void coff_debug_info::coff_symbol::set_section_number(uint32_t section_number)
{
	section_number_ = section_number;
}

//Sets RVA
void coff_debug_info::coff_symbol::set_rva(uint32_t rva)
{
	rva_ = rva;
}

//Sets file name
void coff_debug_info::coff_symbol::set_file_name(const std::string& file_name)
{
	name_ = file_name;
	is_filename_ = true;
}

//Sets symbol name
void coff_debug_info::coff_symbol::set_symbol_name(const std::string& symbol_name)
{
	name_ = symbol_name;
	is_filename_ = false;
}

//Returns type
uint16_t coff_debug_info::coff_symbol::get_type() const
{
	return type_;
}

//Sets type
void coff_debug_info::coff_symbol::set_type(uint16_t type)
{
	type_ = type;
}

//Returns debug information list
const debug_info_list get_debug_information(const pe_base& pe)
{
	debug_info_list ret;

	//If there's no debug directory, return empty list
	if(!pe.has_debug())
		return ret;

	//Check the length in bytes of the section containing debug directory
	if(pe.section_data_length_from_rva(pe.get_directory_rva(image_directory_entry_debug), pe.get_directory_rva(image_directory_entry_debug), section_data_virtual, true)
		< sizeof(image_debug_directory))
		throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

	unsigned long current_pos = pe.get_directory_rva(image_directory_entry_debug);

	//First IMAGE_DEBUG_DIRECTORY table
	image_debug_directory directory = pe.section_data_from_rva<image_debug_directory>(current_pos, section_data_virtual, true);

	if(!pe_utils::is_sum_safe(pe.get_directory_rva(image_directory_entry_debug), pe.get_directory_size(image_directory_entry_debug)))
		throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

	//Iterate over all IMAGE_DEBUG_DIRECTORY directories
	while(directory.PointerToRawData
		&& current_pos < pe.get_directory_rva(image_directory_entry_debug) + pe.get_directory_size(image_directory_entry_debug))
	{
		//Create debug information structure
		debug_info info(directory);

		//Find raw debug data
		const pe_base::debug_data_list& debug_datas = pe.get_raw_debug_data_list();
		pe_base::debug_data_list::const_iterator it = debug_datas.find(directory.PointerToRawData);
		if(it != debug_datas.end()) //If it exists, we'll do some detailed debug info research
		{
			const std::string& debug_data = (*it).second;
			switch(directory.Type)
			{
			case image_debug_type_coff:
				{
					//Check data length
					if(debug_data.length() < sizeof(image_coff_symbols_header))
						throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

					//Get coff header structure pointer
					const image_coff_symbols_header* coff = reinterpret_cast<const image_coff_symbols_header*>(debug_data.data());

					//Check possible overflows
					if(coff->NumberOfSymbols >= pe_utils::max_dword / sizeof(image_symbol)
						|| !pe_utils::is_sum_safe(coff->NumberOfSymbols * sizeof(image_symbol), coff->LvaToFirstSymbol))
						throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

					//Check data length again
					if(debug_data.length() < coff->NumberOfSymbols * sizeof(image_symbol) + coff->LvaToFirstSymbol)
						throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

					//Create COFF debug info structure
					coff_debug_info coff_info(coff);

					//Enumerate debug symbols data
					for(uint32_t i = 0; i < coff->NumberOfSymbols; ++i)
					{
						//Safe sum (checked above)
						const image_symbol* sym = reinterpret_cast<const image_symbol*>(debug_data.data() + i * sizeof(image_symbol) + coff->LvaToFirstSymbol);

						coff_debug_info::coff_symbol symbol;
						symbol.set_index(i); //Save symbol index
						symbol.set_storage_class(sym->StorageClass); //Save storage class
						symbol.set_type(sym->Type); //Save storage class

						//Check data length again
						if(!pe_utils::is_sum_safe(i, sym->NumberOfAuxSymbols)
							|| (i + sym->NumberOfAuxSymbols) > coff->NumberOfSymbols
							|| debug_data.length() < (i + 1) * sizeof(image_symbol) + coff->LvaToFirstSymbol + sym->NumberOfAuxSymbols * sizeof(image_symbol))
							throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

						//If symbol is filename
						if(sym->StorageClass == image_sym_class_file)
						{
							//Save file name, it is situated just after this IMAGE_SYMBOL structure
							std::string file_name(reinterpret_cast<const char*>(debug_data.data() + (i + 1) * sizeof(image_symbol)), sym->NumberOfAuxSymbols * sizeof(image_symbol));
							pe_utils::strip_nullbytes(file_name);
							symbol.set_file_name(file_name);

							//Save symbol info
							coff_info.add_symbol(symbol);

							//Move to next symbol
							i += sym->NumberOfAuxSymbols;
							continue;
						}

						//Dump some other symbols
						if(((sym->StorageClass == image_sym_class_static)
							&& (sym->NumberOfAuxSymbols == 0)
							&& (sym->SectionNumber == 1))
							||
							((sym->StorageClass == image_sym_class_external)
							&& ISFCN(sym->Type)
							&& (sym->SectionNumber > 0))
							)
						{
							//Save RVA and section number
							symbol.set_section_number(sym->SectionNumber);
							symbol.set_rva(sym->Value);

							//If symbol has short name
							if(sym->N.Name.Short)
							{
								//Copy and save symbol name
								char name_buff[9];
								memcpy(name_buff, sym->N.ShortName, 8);
								name_buff[8] = '\0';
								symbol.set_symbol_name(name_buff);
							}
							else
							{
								//Symbol has long name

								//Check possible overflows
								if(!pe_utils::is_sum_safe(coff->LvaToFirstSymbol + coff->NumberOfSymbols * sizeof(image_symbol), sym->N.Name.Long))
									throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

								//Here we have an offset to the string table
								uint32_t symbol_offset = coff->LvaToFirstSymbol + coff->NumberOfSymbols * sizeof(image_symbol) + sym->N.Name.Long;

								//Check data length
								if(debug_data.length() < symbol_offset)
									throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

								//Check symbol name for null-termination
								if(!pe_utils::is_null_terminated(debug_data.data() + symbol_offset, debug_data.length() - symbol_offset))
									throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

								//Save symbol name
								symbol.set_symbol_name(debug_data.data() + symbol_offset);
							}

							//Save symbol info
							coff_info.add_symbol(symbol);

							//Move to next symbol
							i += sym->NumberOfAuxSymbols;
							continue;
						}
					}

					info.set_advanced_debug_info(coff_info);
				}
				break;

			case image_debug_type_codeview:
				{
					//Check data length
					if(debug_data.length() < sizeof(OMFSignature*))
						throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

					//Get POMFSignature structure pointer from the very beginning of debug data
					const OMFSignature* sig = reinterpret_cast<const OMFSignature*>(debug_data.data());
					if(!memcmp(sig->Signature, "RSDS", 4))
					{
						//Signature is "RSDS" - PDB 7.0

						//Check data length
						if(debug_data.length() < sizeof(CV_INFO_PDB70))
							throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

						const CV_INFO_PDB70* pdb_data = reinterpret_cast<const CV_INFO_PDB70*>(debug_data.data());

						//Check PDB file name null-termination
						if(!pe_utils::is_null_terminated(pdb_data->PdbFileName, debug_data.length() - (sizeof(CV_INFO_PDB70) - 1 /* BYTE of filename in structure */)))
							throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

						info.set_advanced_debug_info(pdb_7_0_info(pdb_data));
					}
					else if(!memcmp(sig->Signature, "NB10", 4))
					{
						//Signature is "NB10" - PDB 2.0

						//Check data length
						if(debug_data.length() < sizeof(CV_INFO_PDB20))
							throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

						const CV_INFO_PDB20* pdb_data = reinterpret_cast<const CV_INFO_PDB20*>(debug_data.data());

						//Check PDB file name null-termination
						if(!pe_utils::is_null_terminated(pdb_data->PdbFileName, debug_data.length() - (sizeof(CV_INFO_PDB20) - 1 /* BYTE of filename in structure */)))
							throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

						info.set_advanced_debug_info(pdb_2_0_info(pdb_data));
					}
					else if(!memcmp(sig->Signature, "NB09", 4))
					{
						//CodeView 4.0, no structures available
						info.set_advanced_info_type(debug_info::advanced_info_codeview_4_0);
					}
					else if(!memcmp(sig->Signature, "NB11", 4))
					{
						//CodeView 5.0, no structures available
						info.set_advanced_info_type(debug_info::advanced_info_codeview_5_0);
					}
					else if(!memcmp(sig->Signature, "NB05", 4))
					{
						//Other CodeView, no structures available
						info.set_advanced_info_type(debug_info::advanced_info_codeview);
					}
				}

				break;

			case image_debug_type_misc:
				{
					//Check data length
					if(debug_data.length() < sizeof(image_debug_misc))
						throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

					//Get misc structure pointer
					const image_debug_misc* misc_data = reinterpret_cast<const image_debug_misc*>(debug_data.data());

					//Check misc data length
					if(debug_data.length() < misc_data->Length /* Total length of record */)
						throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

					//Save advanced information
					info.set_advanced_debug_info(misc_debug_info(misc_data));
				}
				break;
			}
		}

		//Save debug information structure
		ret.push_back(info);

		//Check possible overflow
		if(!pe_utils::is_sum_safe(current_pos, sizeof(image_debug_directory)))
			throw pe_exception("Incorrect debug directory", pe_exception::incorrect_debug_directory);

		//Go to next debug entry
		current_pos += sizeof(image_debug_directory);
		directory = pe.section_data_from_rva<image_debug_directory>(current_pos, section_data_virtual, true);
	}

	return ret;
}
}
