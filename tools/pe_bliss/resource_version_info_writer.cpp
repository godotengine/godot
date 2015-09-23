#include <string.h>
#include "resource_version_info_writer.h"
#include "pe_structures.h"
#include "resource_internal.h"
#include "utils.h"
#include "pe_resource_manager.h"
#include "resource_version_info_reader.h"

namespace pe_bliss
{
using namespace pe_win;

resource_version_info_writer::resource_version_info_writer(pe_resource_manager& res)
	:res_(res)
{}

//Sets/replaces full version information:
//file_version_info: versions and file info
//lang_string_values_map: map of version info strings with encodings
//translation_values_map: map of translations
void resource_version_info_writer::set_version_info(const file_version_info& file_info,
	const lang_string_values_map& string_values,
	const translation_values_map& translations,
	uint32_t language,
	uint32_t codepage,
	uint32_t timestamp)
{
	std::string version_data;

	//Calculate total size of version resource data
	uint32_t total_version_info_length =
		static_cast<uint32_t>(sizeof(version_info_block) - sizeof(uint16_t) + sizeof(uint16_t) /* pading */
		+ (resource_version_info_reader::version_info_key.length() + 1) * 2
		+ sizeof(vs_fixedfileinfo));

	//If we have any strings values
	if(!string_values.empty())
	{
		total_version_info_length += sizeof(version_info_block) - sizeof(uint16_t); //StringFileInfo block
		total_version_info_length += SizeofStringFileInfo; //Name of block (key)

		//Add required size for version strings
		for(lang_string_values_map::const_iterator table_it = string_values.begin(); table_it != string_values.end(); ++table_it)
		{
			total_version_info_length += pe_utils::align_up(static_cast<uint32_t>(sizeof(uint16_t) * 3 + ((*table_it).first.length() + 1) * 2), sizeof(uint32_t)); //Name of child block and block size (key of string table block)

			const string_values_map& values = (*table_it).second;
			for(string_values_map::const_iterator it = values.begin(); it != values.end(); ++it)
			{
				total_version_info_length += pe_utils::align_up(static_cast<uint32_t>(sizeof(uint16_t) * 3 + ((*it).first.length() + 1) * 2), sizeof(uint32_t));
				total_version_info_length += pe_utils::align_up(static_cast<uint32_t>(((*it).second.length() + 1) * 2), sizeof(uint32_t));
			}
		}
	}

	//If we have translations
	if(!translations.empty())
	{
		total_version_info_length += (sizeof(version_info_block) - sizeof(uint16_t)) * 2; //VarFileInfo and Translation blocks
		total_version_info_length += SizeofVarFileInfoAligned; //DWORD-aligned VarFileInfo block name
		total_version_info_length += SizeofTranslationAligned; //DWORD-aligned Translation block name
		total_version_info_length += static_cast<uint32_t>(translations.size() * sizeof(uint16_t) * 2);
	}

	//Resize version data buffer
	version_data.resize(total_version_info_length);

	//Create root version block
	version_info_block root_block = {0};
	root_block.ValueLength = sizeof(vs_fixedfileinfo);
	root_block.Length = static_cast<uint16_t>(total_version_info_length);

	//Fill fixed file info
	vs_fixedfileinfo fixed_info = {0};
	fixed_info.dwFileDateLS = file_info.get_file_date_ls();
	fixed_info.dwFileDateMS = file_info.get_file_date_ms();
	fixed_info.dwFileFlags = file_info.get_file_flags();
	fixed_info.dwFileFlagsMask = vs_ffi_fileflagsmask;
	fixed_info.dwFileOS = file_info.get_file_os_raw();
	fixed_info.dwFileSubtype = file_info.get_file_subtype();
	fixed_info.dwFileType = file_info.get_file_type_raw();
	fixed_info.dwFileVersionLS = file_info.get_file_version_ls();
	fixed_info.dwFileVersionMS = file_info.get_file_version_ms();
	fixed_info.dwSignature = vs_ffi_signature;
	fixed_info.dwStrucVersion = vs_ffi_strucversion;
	fixed_info.dwProductVersionLS = file_info.get_product_version_ls();
	fixed_info.dwProductVersionMS = file_info.get_product_version_ms();

	//Write root block and fixed file info to buffer
	uint32_t data_ptr = 0;
	memcpy(&version_data[data_ptr], &root_block, sizeof(version_info_block) - sizeof(uint16_t));
	data_ptr += sizeof(version_info_block) - sizeof(uint16_t);
	memcpy(&version_data[data_ptr], resource_version_info_reader::version_info_key.c_str(), (resource_version_info_reader::version_info_key.length() + 1) * sizeof(uint16_t));
	data_ptr += static_cast<uint32_t>((resource_version_info_reader::version_info_key.length() + 1) * sizeof(uint16_t));
	memset(&version_data[data_ptr], 0, sizeof(uint16_t));
	data_ptr += sizeof(uint16_t);
	memcpy(&version_data[data_ptr], &fixed_info, sizeof(fixed_info));
	data_ptr += sizeof(fixed_info);

	//Write string values, if any
	if(!string_values.empty())
	{
		//Create string file info root block
		version_info_block string_file_info_block = {0};
		string_file_info_block.Type = 1; //Block type is string
		memcpy(&version_data[data_ptr], &string_file_info_block, sizeof(version_info_block) - sizeof(uint16_t));
		//We will calculate its length later
		version_info_block* string_file_info_block_ptr = reinterpret_cast<version_info_block*>(&version_data[data_ptr]);
		data_ptr += sizeof(version_info_block) - sizeof(uint16_t);

		uint32_t old_ptr1 = data_ptr; //Used to calculate string file info block length later
		memcpy(&version_data[data_ptr], StringFileInfo, SizeofStringFileInfo); //Write block name
		data_ptr += SizeofStringFileInfo;

		//Create string table root block (child of string file info)
		version_info_block string_table_block = {0};
		string_table_block.Type = 1; //Block type is string

		for(lang_string_values_map::const_iterator table_it = string_values.begin(); table_it != string_values.end(); ++table_it)
		{
			const string_values_map& values = (*table_it).second;

			memcpy(&version_data[data_ptr], &string_table_block, sizeof(version_info_block) - sizeof(uint16_t));
			//We will calculate its length later
			version_info_block* string_table_block_ptr = reinterpret_cast<version_info_block*>(&version_data[data_ptr]);
			data_ptr += sizeof(version_info_block) - sizeof(uint16_t);

			uint32_t old_ptr2 = data_ptr; //Used to calculate string table block length later
			uint32_t lang_key_length = static_cast<uint32_t>(((*table_it).first.length() + 1) * sizeof(uint16_t));

#ifdef PE_BLISS_WINDOWS
			memcpy(&version_data[data_ptr], (*table_it).first.c_str(), lang_key_length); //Write block key
#else
			{
				u16string str(pe_utils::to_ucs2((*table_it).first));
				memcpy(&version_data[data_ptr], str.c_str(), lang_key_length); //Write block key
			}
#endif

			data_ptr += lang_key_length;
			//Align key if necessary
			if((sizeof(uint16_t) * 3 + lang_key_length) % sizeof(uint32_t))
			{
				memset(&version_data[data_ptr], 0, sizeof(uint16_t));
				data_ptr += sizeof(uint16_t);
			}

			//Create string block (child of string table block)
			version_info_block string_block = {0};
			string_block.Type = 1; //Block type is string
			for(string_values_map::const_iterator it = values.begin(); it != values.end(); ++it)
			{
				//Calculate value length and key length of string block
				string_block.ValueLength = static_cast<uint16_t>((*it).second.length() + 1);
				uint32_t key_length = static_cast<uint32_t>(((*it).first.length() + 1) * sizeof(uint16_t));
				//Calculate length of block
				string_block.Length = static_cast<uint16_t>(pe_utils::align_up(sizeof(uint16_t) * 3 + key_length, sizeof(uint32_t)) + string_block.ValueLength * sizeof(uint16_t));

				//Write string block
				memcpy(&version_data[data_ptr], &string_block, sizeof(version_info_block) - sizeof(uint16_t));
				data_ptr += sizeof(version_info_block) - sizeof(uint16_t);

#ifdef PE_BLISS_WINDOWS
				memcpy(&version_data[data_ptr], (*it).first.c_str(), key_length); //Write block key
#else
				{
					u16string str(pe_utils::to_ucs2((*it).first));
					memcpy(&version_data[data_ptr], str.c_str(), key_length); //Write block key
				}
#endif

				data_ptr += key_length;
				//Align key if necessary
				if((sizeof(uint16_t) * 3 + key_length) % sizeof(uint32_t))
				{
					memset(&version_data[data_ptr], 0, sizeof(uint16_t));
					data_ptr += sizeof(uint16_t);
				}

				//Write block data (value)
#ifdef PE_BLISS_WINDOWS
				memcpy(&version_data[data_ptr], (*it).second.c_str(), string_block.ValueLength * sizeof(uint16_t));
#else
				{
					u16string str(pe_utils::to_ucs2((*it).second));
					memcpy(&version_data[data_ptr], str.c_str(), string_block.ValueLength * sizeof(uint16_t));
				}
#endif

				data_ptr += string_block.ValueLength * 2;
				//Align data if necessary
				if((string_block.ValueLength * 2) % sizeof(uint32_t))
				{
					memset(&version_data[data_ptr], 0, sizeof(uint16_t));
					data_ptr += sizeof(uint16_t);
				}
			}

			//Calculate string table and string file info blocks lengths
			string_table_block_ptr->Length = static_cast<uint16_t>(data_ptr - old_ptr2 + sizeof(uint16_t) * 3);
		}

		string_file_info_block_ptr->Length = static_cast<uint16_t>(data_ptr - old_ptr1 + sizeof(uint16_t) * 3);
	}

	//If we have transactions
	if(!translations.empty())
	{
		//Create root var file info block
		version_info_block var_file_info_block = {0};
		var_file_info_block.Type = 1; //Type of block is string
		//Write block header
		memcpy(&version_data[data_ptr], &var_file_info_block, sizeof(version_info_block) - sizeof(uint16_t));
		//We will calculate its length later
		version_info_block* var_file_info_block_ptr = reinterpret_cast<version_info_block*>(&version_data[data_ptr]);
		data_ptr += sizeof(version_info_block) - sizeof(uint16_t);

		uint32_t old_ptr1 = data_ptr; //Used to calculate var file info block length later
		memcpy(&version_data[data_ptr], VarFileInfoAligned, SizeofVarFileInfoAligned); //Write block key (aligned)
		data_ptr += SizeofVarFileInfoAligned;

		//Create root translation block (child of var file info block)
		version_info_block translation_block = {0};
		//Write block header
		memcpy(&version_data[data_ptr], &translation_block, sizeof(version_info_block) - sizeof(uint16_t));
		//We will calculate its length later
		version_info_block* translation_block_ptr = reinterpret_cast<version_info_block*>(&version_data[data_ptr]);
		data_ptr += sizeof(version_info_block) - sizeof(uint16_t);

		uint32_t old_ptr2 = data_ptr; //Used to calculate var file info block length later
		memcpy(&version_data[data_ptr], TranslationAligned, SizeofTranslationAligned); //Write block key (aligned)
		data_ptr += SizeofTranslationAligned;

		//Calculate translation block value length
		translation_block_ptr->ValueLength = static_cast<uint16_t>(sizeof(uint16_t) * 2 * translations.size());

		//Write translation values to block
		for(translation_values_map::const_iterator it = translations.begin(); it != translations.end(); ++it)
		{
			uint16_t lang_id = (*it).first; //Language ID
			uint16_t codepage_id = (*it).second; //Codepage ID
			memcpy(&version_data[data_ptr], &lang_id, sizeof(lang_id));
			data_ptr += sizeof(lang_id);
			memcpy(&version_data[data_ptr], &codepage_id, sizeof(codepage_id));
			data_ptr += sizeof(codepage_id);
		}

		//Calculate Translation and VarFileInfo blocks lengths
		translation_block_ptr->Length = static_cast<uint16_t>(data_ptr - old_ptr2 + sizeof(uint16_t) * 3);
		var_file_info_block_ptr->Length = static_cast<uint16_t>(data_ptr - old_ptr1 + sizeof(uint16_t) * 3);
	}

	//Add/replace version info resource
	res_.add_resource(version_data, pe_resource_viewer::resource_version, 1, language, codepage, timestamp);
}

//Removes version info by language (ID = 1)
bool resource_version_info_writer::remove_version_info(uint32_t language)
{
	return res_.remove_resource(pe_resource_viewer::resource_version, 1, language);
}
}
