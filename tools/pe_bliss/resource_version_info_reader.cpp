#include "resource_version_info_reader.h"
#include "utils.h"
#include "pe_exception.h"
#include "resource_internal.h"
#include "pe_resource_viewer.h"

namespace pe_bliss
{
using namespace pe_win;

//Root version info block key value
const u16string resource_version_info_reader::version_info_key(U16TEXT("V\0S\0_\0V\0E\0R\0S\0I\0O\0N\0_\0I\0N\0F\0O\0\0"));

resource_version_info_reader::resource_version_info_reader(const pe_resource_viewer& res)
	:res_(res)
{}

//Returns aligned version block value position
uint32_t resource_version_info_reader::get_version_block_value_pos(uint32_t base_pos, const unicode16_t* key)
{
	uint32_t string_length = static_cast<uint32_t>(u16string(key).length());
	uint32_t ret = pe_utils::align_up(static_cast<uint32_t>(sizeof(uint16_t) * 3 /* headers before Key data */
		+ base_pos
		+ (string_length + 1 /* nullbyte */) * 2),
		sizeof(uint32_t));

	//Check possible overflows
	if(ret < base_pos || ret < sizeof(uint16_t) * 3 || ret < (string_length + 1) * 2)
		throw_incorrect_version_info();

	return ret;
}

//Returns aligned version block first child position
uint32_t resource_version_info_reader::get_version_block_first_child_pos(uint32_t base_pos, uint32_t value_length, const unicode16_t* key)
{
	uint32_t string_length = static_cast<uint32_t>(u16string(key).length());
	uint32_t ret =  pe_utils::align_up(static_cast<uint32_t>(sizeof(uint16_t) * 3 /* headers before Key data */
		+ base_pos
		+ (string_length + 1 /* nullbyte */) * 2),
		sizeof(uint32_t))
		+ pe_utils::align_up(value_length, sizeof(uint32_t));

	//Check possible overflows
	if(ret < base_pos || ret < value_length || ret < sizeof(uint16_t) * 3 || ret < (string_length + 1) * 2)
		throw_incorrect_version_info();

	return ret;
}

//Throws an exception (id = resource_incorrect_version_info)
void resource_version_info_reader::throw_incorrect_version_info()
{
	throw pe_exception("Incorrect resource version info", pe_exception::resource_incorrect_version_info);
}

//Returns full version information:
//file_version_info: versions and file info
//lang_string_values_map: map of version info strings with encodings
//translation_values_map: map of translations
const file_version_info resource_version_info_reader::get_version_info(lang_string_values_map& string_values, translation_values_map& translations, const std::string& resource_data) const
{
	//Fixed file version info
	file_version_info ret;

	//Check resource data length
	if(resource_data.length() < sizeof(version_info_block))
		throw_incorrect_version_info();

	//Root version info block
	const version_info_block* root_block = reinterpret_cast<const version_info_block*>(resource_data.data());

	//Check root block key for null-termination and its name
	if(!pe_utils::is_null_terminated(root_block->Key, resource_data.length() - sizeof(uint16_t) * 3 /* headers before Key data */)
		|| version_info_key != reinterpret_cast<const unicode16_t*>(root_block->Key))
		throw_incorrect_version_info();

	//If file has fixed version info
	if(root_block->ValueLength)
	{
		//Get root block value position
		uint32_t value_pos = get_version_block_value_pos(0, reinterpret_cast<const unicode16_t*>(root_block->Key));
		//Check value length
		if(resource_data.length() < value_pos + sizeof(vs_fixedfileinfo))
			throw_incorrect_version_info();

		//Get VS_FIXEDFILEINFO structure pointer
		const vs_fixedfileinfo* file_info = reinterpret_cast<const vs_fixedfileinfo*>(resource_data.data() + value_pos);
		//Check its signature and some other fields
		if(file_info->dwSignature != vs_ffi_signature || file_info->dwStrucVersion != vs_ffi_strucversion) //Don't check if file_info->dwFileFlagsMask == VS_FFI_FILEFLAGSMASK
			throw_incorrect_version_info();

		//Save fixed version info
		ret = file_version_info(*file_info);
	}

	//Iterate over child elements of VS_VERSIONINFO (StringFileInfo or VarFileInfo)
	for(uint32_t child_pos = get_version_block_first_child_pos(0, root_block->ValueLength, reinterpret_cast<const unicode16_t*>(root_block->Key));
		child_pos < root_block->Length;)
	{
		//Check block position
		if(!pe_utils::is_sum_safe(child_pos, sizeof(version_info_block))
			|| resource_data.length() < child_pos + sizeof(version_info_block))
			throw_incorrect_version_info();

		//Get VERSION_INFO_BLOCK structure pointer
		const version_info_block* block = reinterpret_cast<const version_info_block*>(resource_data.data() + child_pos);

		//Check its length
		if(block->Length == 0)
			throw_incorrect_version_info();

		//Check block key for null-termination
		if(!pe_utils::is_null_terminated(block->Key, resource_data.length() - child_pos - sizeof(uint16_t) * 3 /* headers before Key data */))
			throw_incorrect_version_info();

		u16string info_type(reinterpret_cast<const unicode16_t*>(block->Key));
		//If we encountered StringFileInfo...
		if(info_type == StringFileInfo)
		{
			//Enumerate all string tables
			for(uint32_t string_table_pos = get_version_block_first_child_pos(child_pos, block->ValueLength, reinterpret_cast<const unicode16_t*>(block->Key));
				string_table_pos - child_pos < block->Length;)
			{
				//Check string table block position
				if(resource_data.length() < string_table_pos + sizeof(version_info_block))
					throw_incorrect_version_info();

				//Get VERSION_INFO_BLOCK structure pointer for string table
				const version_info_block* string_table = reinterpret_cast<const version_info_block*>(resource_data.data() + string_table_pos);

				//Check its length
				if(string_table->Length == 0)
					throw_incorrect_version_info();

				//Check string table key for null-termination
				if(!pe_utils::is_null_terminated(string_table->Key, resource_data.length() - string_table_pos - sizeof(uint16_t) * 3 /* headers before Key data */))	
					throw_incorrect_version_info();

				string_values_map new_values;

				//Enumerate all strings in the string table
				for(uint32_t string_pos = get_version_block_first_child_pos(string_table_pos, string_table->ValueLength, reinterpret_cast<const unicode16_t*>(string_table->Key));
					string_pos - string_table_pos < string_table->Length;)
				{
					//Check string block position
					if(resource_data.length() < string_pos + sizeof(version_info_block))
						throw_incorrect_version_info();

					//Get VERSION_INFO_BLOCK structure pointer for string block
					const version_info_block* string_block = reinterpret_cast<const version_info_block*>(resource_data.data() + string_pos);

					//Check its length
					if(string_block->Length == 0)
						throw_incorrect_version_info();

					//Check string block key for null-termination
					if(!pe_utils::is_null_terminated(string_block->Key, resource_data.length() - string_pos - sizeof(uint16_t) * 3 /* headers before Key data */))
						throw_incorrect_version_info();

					u16string data;
					//If string block has value
					if(string_block->ValueLength != 0)
					{
						//Get value position
						uint32_t value_pos = get_version_block_value_pos(string_pos, reinterpret_cast<const unicode16_t*>(string_block->Key));
						//Check it
						if(resource_data.length() < value_pos + string_block->ValueLength)
							throw pe_exception("Incorrect resource version info", pe_exception::resource_incorrect_version_info);

						//Get UNICODE string value
						data = u16string(reinterpret_cast<const unicode16_t*>(resource_data.data() + value_pos), string_block->ValueLength);
						pe_utils::strip_nullbytes(data);
					}

					//Save name-value pair
#ifdef PE_BLISS_WINDOWS
					new_values.insert(std::make_pair(reinterpret_cast<const unicode16_t*>(string_block->Key), data));
#else
					new_values.insert(std::make_pair(pe_utils::from_ucs2(reinterpret_cast<const unicode16_t*>(string_block->Key)),
						pe_utils::from_ucs2(data)));
#endif

					//Navigate to next string block
					string_pos += pe_utils::align_up(string_block->Length, sizeof(uint32_t));
				}

#ifdef PE_BLISS_WINDOWS
				string_values.insert(std::make_pair(reinterpret_cast<const unicode16_t*>(string_table->Key), new_values));
#else
				string_values.insert(std::make_pair(pe_utils::from_ucs2(reinterpret_cast<const unicode16_t*>(string_table->Key)), new_values));
#endif

				//Navigate to next string table block
				string_table_pos += pe_utils::align_up(string_table->Length, sizeof(uint32_t));
			}
		}
		else if(info_type == VarFileInfo) //If we encountered VarFileInfo
		{
			for(uint32_t var_table_pos = get_version_block_first_child_pos(child_pos, block->ValueLength, reinterpret_cast<const unicode16_t*>(block->Key));
				var_table_pos - child_pos < block->Length;)
			{
				//Check var block position
				if(resource_data.length() < var_table_pos + sizeof(version_info_block))
					throw_incorrect_version_info();

				//Get VERSION_INFO_BLOCK structure pointer for var block
				const version_info_block* var_table = reinterpret_cast<const version_info_block*>(resource_data.data() + var_table_pos);

				//Check its length
				if(var_table->Length == 0)
					throw_incorrect_version_info();

				//Check its key for null-termination
				if(!pe_utils::is_null_terminated(var_table->Key, resource_data.length() - var_table_pos - sizeof(uint16_t) * 3 /* headers before Key data */))
					throw_incorrect_version_info();

				//If block is "Translation" (actually, there's no other types possible in VarFileInfo) and it has value
				if(u16string(reinterpret_cast<const unicode16_t*>(var_table->Key)) == Translation && var_table->ValueLength)
				{
					//Get its value position
					uint32_t value_pos = get_version_block_value_pos(var_table_pos, reinterpret_cast<const unicode16_t*>(var_table->Key));
					//Cherck value length
					if(resource_data.length() < value_pos + var_table->ValueLength)
						throw_incorrect_version_info();

					//Get list of translations: pairs of LANGUAGE_ID - CODEPAGE_ID
					for(unsigned long i = 0; i < var_table->ValueLength; i += sizeof(uint16_t) * 2)
					{
						//Pair of WORDs
						uint16_t lang_id = *reinterpret_cast<const uint16_t*>(resource_data.data() + value_pos + i);
						uint16_t codepage_id = *reinterpret_cast<const uint16_t*>(resource_data.data() + value_pos + sizeof(uint16_t) + i);
						//Save translation
						translations.insert(std::make_pair(lang_id, codepage_id));
					}
				}

				//Navigate to next var block
				var_table_pos += pe_utils::align_up(var_table->Length, sizeof(uint32_t));
			}
		}
		else
		{
			throw_incorrect_version_info();
		}

		//Navigate to next element in root block
		child_pos += pe_utils::align_up(block->Length, sizeof(uint32_t));
	}

	return ret;
}

//Returns full version information:
//file_version info: versions and file info
//lang_string_values_map: map of version info strings with encodings
//translation_values_map: map of translations
const file_version_info resource_version_info_reader::get_version_info_by_lang(lang_string_values_map& string_values, translation_values_map& translations, uint32_t language) const
{
	const std::string& resource_data = res_.get_root_directory() //Type directory
		.entry_by_id(pe_resource_viewer::resource_version)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(1)
		.get_resource_directory() //Language directory
		.entry_by_id(language)
		.get_data_entry() //Data directory
		.get_data();

	return get_version_info(string_values, translations, resource_data);
}

//Returns full version information:
//file_version_info: versions and file info
//lang_string_values_map: map of version info strings with encodings
//translation_values_map: map of translations
const file_version_info resource_version_info_reader::get_version_info(lang_string_values_map& string_values, translation_values_map& translations, uint32_t index) const
{
	const resource_directory::entry_list& entries = res_.get_root_directory() //Type directory
		.entry_by_id(pe_resource_viewer::resource_version)
		.get_resource_directory() //Name/ID directory
		.entry_by_id(1)
		.get_resource_directory() //Language directory
		.get_entry_list();

	if(entries.size() <= index)
		throw pe_exception("Resource data entry not found", pe_exception::resource_data_entry_not_found);

	return get_version_info(string_values, translations, entries.at(index).get_data_entry().get_data()); //Data directory
}
}
