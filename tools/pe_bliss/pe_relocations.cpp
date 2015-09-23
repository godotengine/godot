#include <string.h>
#include "pe_relocations.h"
#include "pe_properties_generic.h"

namespace pe_bliss
{
using namespace pe_win;

//RELOCATIONS
//Default constructor
relocation_entry::relocation_entry()
	:rva_(0), type_(0)
{}

//Constructor from relocation item (WORD)
relocation_entry::relocation_entry(uint16_t relocation_value)
	:rva_(relocation_value & ((1 << 12) - 1)), type_(relocation_value >> 12)
{}

//Constructor from relative rva and relocation type
relocation_entry::relocation_entry(uint16_t rrva, uint16_t type)
	:rva_(rrva), type_(type)
{}

//Returns RVA of relocation
uint16_t relocation_entry::get_rva() const
{
	return rva_;
}

//Returns type of relocation
uint16_t relocation_entry::get_type() const
{
	return type_;
}

//Sets RVA of relocation
void relocation_entry::set_rva(uint16_t rva)
{
	rva_ = rva;
}

//Sets type of relocation
void relocation_entry::set_type(uint16_t type)
{
	type_ = type;
}

//Returns relocation item (rrva + type)
uint16_t relocation_entry::get_item() const
{
	return rva_ | (type_ << 12);
}

//Sets relocation item (rrva + type)
void relocation_entry::set_item(uint16_t item)
{
	rva_ = item & ((1 << 12) - 1);
	type_ = item >> 12;
}

//Returns relocation list
const relocation_table::relocation_list& relocation_table::get_relocations() const
{
	return relocations_;
}

//Adds relocation to table
void relocation_table::add_relocation(const relocation_entry& entry)
{
	relocations_.push_back(entry);
}

//Default constructor
relocation_table::relocation_table()
	:rva_(0)
{}

//Constructor from RVA of relocation table
relocation_table::relocation_table(uint32_t rva)
	:rva_(rva)
{}

//Returns RVA of block
uint32_t relocation_table::get_rva() const
{
	return rva_;
}

//Sets RVA of block
void relocation_table::set_rva(uint32_t rva)
{
	rva_ = rva;
}

//Returns changeable relocation list
relocation_table::relocation_list& relocation_table::get_relocations()
{
	return relocations_;
}

//Get relocation list of pe file, supports one-word sized relocations only
//If list_absolute_entries = true, IMAGE_REL_BASED_ABSOLUTE will be listed
const relocation_table_list get_relocations(const pe_base& pe, bool list_absolute_entries)
{
	relocation_table_list ret;

	//If image does not have relocations
	if(!pe.has_reloc())
		return ret;

	//Check the length in bytes of the section containing relocation directory
	if(pe.section_data_length_from_rva(pe.get_directory_rva(image_directory_entry_basereloc),
		pe.get_directory_rva(image_directory_entry_basereloc), section_data_virtual, true)
		< sizeof(image_base_relocation))
		throw pe_exception("Incorrect relocation directory", pe_exception::incorrect_relocation_directory);

	unsigned long current_pos = pe.get_directory_rva(image_directory_entry_basereloc);
	//First IMAGE_BASE_RELOCATION table
	image_base_relocation reloc_table = pe.section_data_from_rva<image_base_relocation>(current_pos, section_data_virtual, true);

	if(reloc_table.SizeOfBlock % 2)
		throw pe_exception("Incorrect relocation directory", pe_exception::incorrect_relocation_directory);

	unsigned long reloc_size = pe.get_directory_size(image_directory_entry_basereloc);
	unsigned long read_size = 0;

	//reloc_table.VirtualAddress is not checked (not so important)
	while(reloc_table.SizeOfBlock && read_size < reloc_size)
	{
		//Create relocation table
		relocation_table table;
		//Save RVA
		table.set_rva(reloc_table.VirtualAddress);

		if(!pe_utils::is_sum_safe(current_pos, reloc_table.SizeOfBlock))
			throw pe_exception("Incorrect relocation directory", pe_exception::incorrect_relocation_directory);

		//List all relocations
		for(unsigned long i = sizeof(image_base_relocation); i < reloc_table.SizeOfBlock; i += sizeof(uint16_t))
		{
			relocation_entry entry(pe.section_data_from_rva<uint16_t>(current_pos + i, section_data_virtual, true));
			if(list_absolute_entries || entry.get_type() != image_rel_based_absolute)
				table.add_relocation(entry);
		}

		//Save table
		ret.push_back(table);
		
		//Go to next relocation block
		if(!pe_utils::is_sum_safe(current_pos, reloc_table.SizeOfBlock))
			throw pe_exception("Incorrect relocation directory", pe_exception::incorrect_relocation_directory);

		current_pos += reloc_table.SizeOfBlock;
		read_size += reloc_table.SizeOfBlock;
		reloc_table = pe.section_data_from_rva<image_base_relocation>(current_pos, section_data_virtual, true);
	}

	return ret;
}

//Simple relocations rebuilder
//To keep PE file working, don't remove any of existing relocations in
//relocation_table_list returned by a call to get_relocations() function
//auto_strip_last_section - if true and relocations are placed in the last section, it will be automatically stripped
//offset_from_section_start - offset from the beginning of reloc_section, where relocations data will be situated
//If save_to_pe_header is true, PE header will be modified automatically
const image_directory rebuild_relocations(pe_base& pe, const relocation_table_list& relocs, section& reloc_section, uint32_t offset_from_section_start, bool save_to_pe_header, bool auto_strip_last_section)
{
	//Check that reloc_section is attached to this PE image
	if(!pe.section_attached(reloc_section))
		throw pe_exception("Relocations section must be attached to PE file", pe_exception::section_is_not_attached);
	
	uint32_t current_reloc_data_pos = pe_utils::align_up(offset_from_section_start, sizeof(uint32_t));

	uint32_t needed_size = current_reloc_data_pos - offset_from_section_start; //Calculate needed size for relocation tables
	uint32_t size_delta = needed_size;

	uint32_t start_reloc_pos = current_reloc_data_pos;

	//Enumerate relocation tables
	for(relocation_table_list::const_iterator it = relocs.begin(); it != relocs.end(); ++it)
	{
		needed_size += static_cast<uint32_t>((*it).get_relocations().size() * sizeof(uint16_t) /* relocations */ + sizeof(image_base_relocation) /* table header */);
		//End of each table will be DWORD-aligned
		if((start_reloc_pos + needed_size - size_delta) % sizeof(uint32_t))
			needed_size += sizeof(uint16_t); //Align it with IMAGE_REL_BASED_ABSOLUTE relocation
	}

	//Check if reloc_section is last one. If it's not, check if there's enough place for relocations data
	if(&reloc_section != &*(pe.get_image_sections().end() - 1) && 
		(reloc_section.empty() || pe_utils::align_up(reloc_section.get_size_of_raw_data(), pe.get_file_alignment()) < needed_size + current_reloc_data_pos))
		throw pe_exception("Insufficient space for relocations directory", pe_exception::insufficient_space);

	std::string& raw_data = reloc_section.get_raw_data();

	//This will be done only if reloc_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + current_reloc_data_pos)
		raw_data.resize(needed_size + current_reloc_data_pos); //Expand section raw data

	//Enumerate relocation tables
	for(relocation_table_list::const_iterator it = relocs.begin(); it != relocs.end(); ++it)
	{
		//Create relocation table header
		image_base_relocation reloc;
		reloc.VirtualAddress = (*it).get_rva();
		const relocation_table::relocation_list& reloc_list = (*it).get_relocations();
		reloc.SizeOfBlock = static_cast<uint32_t>(sizeof(image_base_relocation) + sizeof(uint16_t) * reloc_list.size());
		if((reloc_list.size() * sizeof(uint16_t)) % sizeof(uint32_t)) //If we must align end of relocation table
			reloc.SizeOfBlock += sizeof(uint16_t);

		memcpy(&raw_data[current_reloc_data_pos], &reloc, sizeof(reloc));
		current_reloc_data_pos += sizeof(reloc);

		//Enumerate relocations in table
		for(relocation_table::relocation_list::const_iterator r = reloc_list.begin(); r != reloc_list.end(); ++r)
		{
			//Save relocations
			uint16_t reloc_value = (*r).get_item();
			memcpy(&raw_data[current_reloc_data_pos], &reloc_value, sizeof(reloc_value));
			current_reloc_data_pos += sizeof(reloc_value);
		}

		if(current_reloc_data_pos % sizeof(uint32_t)) //If end of table is not DWORD-aligned
		{
			memset(&raw_data[current_reloc_data_pos], 0, sizeof(uint16_t)); //Align it with IMAGE_REL_BASED_ABSOLUTE relocation
			current_reloc_data_pos += sizeof(uint16_t);
		}
	}

	image_directory ret(pe.rva_from_section_offset(reloc_section, start_reloc_pos), needed_size - size_delta);
	
	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(reloc_section, auto_strip_last_section);

	//If auto-rewrite of PE headers is required
	if(save_to_pe_header)
	{
		pe.set_directory_rva(image_directory_entry_basereloc, ret.get_rva());
		pe.set_directory_size(image_directory_entry_basereloc, ret.get_size());

		pe.clear_characteristics_flags(image_file_relocs_stripped);
		pe.set_dll_characteristics(pe.get_dll_characteristics() | image_dllcharacteristics_dynamic_base);
	}

	return ret;
}

//Recalculates image base with the help of relocation tables
void rebase_image(pe_base& pe, const relocation_table_list& tables, uint64_t new_base)
{
	pe.get_pe_type() == pe_type_32
		? rebase_image_base<pe_types_class_32>(pe, tables, new_base)
		: rebase_image_base<pe_types_class_64>(pe, tables, new_base);
}

//RELOCATIONS
//Recalculates image base with the help of relocation tables
//Recalculates VAs of DWORDS/QWORDS in image according to relocations
//Notice: if you move some critical structures like TLS, image relocations will not fix new
//positions of TLS VAs. Instead, some bytes that now doesn't belong to TLS will be fixed.
//It is recommended to rebase image in the very beginning and move all structures afterwards.
template<typename PEClassType>
void rebase_image_base(pe_base& pe, const relocation_table_list& tables, uint64_t new_base)
{
	//Get current image base value
	typename PEClassType::BaseSize image_base;
	pe.get_image_base(image_base);

	//ImageBase difference
	typename PEClassType::BaseSize base_rel = static_cast<typename PEClassType::BaseSize>(static_cast<int64_t>(new_base) - image_base);

	//We need to fix addresses from relocation tables
	//Enumerate relocation tables
	for(relocation_table_list::const_iterator it = tables.begin(); it != tables.end(); ++it)
	{
		const relocation_table::relocation_list& relocs = (*it).get_relocations();

		uint32_t base_rva = (*it).get_rva();

		//Enumerate relocations
		for(relocation_table::relocation_list::const_iterator rel = relocs.begin(); rel != relocs.end(); ++rel)
		{
			//Skip ABSOLUTE entries
			if((*rel).get_type() == pe_win::image_rel_based_absolute)
				continue;
			
			//Recalculate value by RVA and rewrite it
			uint32_t current_rva = base_rva + (*rel).get_rva();
			typename PEClassType::BaseSize value = pe.section_data_from_rva<typename PEClassType::BaseSize>(current_rva, section_data_raw, true);
			value += base_rel;
			memcpy(pe.section_data_from_rva(current_rva, true), &value, sizeof(value));
		}
	}

	//Finally, save new image base
	pe.set_image_base_64(new_base);
}
}
