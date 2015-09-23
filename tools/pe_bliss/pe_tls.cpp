#include <string.h>
#include "pe_tls.h"
#include "pe_properties_generic.h"

namespace pe_bliss
{
using namespace pe_win;

//TLS
//Default constructor
tls_info::tls_info()
	:start_rva_(0), end_rva_(0), index_rva_(0), callbacks_rva_(0),
	size_of_zero_fill_(0), characteristics_(0)
{}

//Returns start RVA of TLS raw data
uint32_t tls_info::get_raw_data_start_rva() const
{
	return start_rva_;
}

//Returns end RVA of TLS raw data
uint32_t tls_info::get_raw_data_end_rva() const
{
	return end_rva_;
}

//Returns TLS index RVA
uint32_t tls_info::get_index_rva() const
{
	return index_rva_;
}

//Returns TLS callbacks RVA
uint32_t tls_info::get_callbacks_rva() const
{
	return callbacks_rva_;
}

//Returns size of zero fill
uint32_t tls_info::get_size_of_zero_fill() const
{
	return size_of_zero_fill_;
}

//Returns characteristics
uint32_t tls_info::get_characteristics() const
{
	return characteristics_;
}

//Returns raw TLS data
const std::string& tls_info::get_raw_data() const
{
	return raw_data_;
}

//Returns TLS callbacks addresses
const tls_info::tls_callback_list& tls_info::get_tls_callbacks() const
{
	return callbacks_;
}

//Returns TLS callbacks addresses
tls_info::tls_callback_list& tls_info::get_tls_callbacks()
{
	return callbacks_;
}

//Adds TLS callback
void tls_info::add_tls_callback(uint32_t rva)
{
	callbacks_.push_back(rva);
}

//Clears TLS callbacks list
void tls_info::clear_tls_callbacks()
{
	callbacks_.clear();
}

//Recalculates end address of raw TLS data
void tls_info::recalc_raw_data_end_rva()
{
	end_rva_ = static_cast<uint32_t>(start_rva_ + raw_data_.length());
}

//Sets start RVA of TLS raw data
void tls_info::set_raw_data_start_rva(uint32_t rva)
{
	start_rva_ = rva;
}

//Sets end RVA of TLS raw data
void tls_info::set_raw_data_end_rva(uint32_t rva)
{
	end_rva_ = rva;
}

//Sets TLS index RVA
void tls_info::set_index_rva(uint32_t rva)
{
	index_rva_ = rva;
}

//Sets TLS callbacks RVA
void tls_info::set_callbacks_rva(uint32_t rva)
{
	callbacks_rva_ = rva;
}

//Sets size of zero fill
void tls_info::set_size_of_zero_fill(uint32_t size)
{
	size_of_zero_fill_ = size;
}

//Sets characteristics
void tls_info::set_characteristics(uint32_t characteristics)
{
	characteristics_ = characteristics;
}

//Sets raw TLS data
void tls_info::set_raw_data(const std::string& data)
{
	raw_data_ = data;
}

//If image does not have TLS, throws an exception
const tls_info get_tls_info(const pe_base& pe)
{
	return pe.get_pe_type() == pe_type_32
		? get_tls_info_base<pe_types_class_32>(pe)
		: get_tls_info_base<pe_types_class_64>(pe);
}

//TLS Rebuilder
const image_directory rebuild_tls(pe_base& pe, const tls_info& info, section& tls_section, uint32_t offset_from_section_start, bool write_tls_callbacks, bool write_tls_data, tls_data_expand_type expand, bool save_to_pe_header, bool auto_strip_last_section)
{
	return pe.get_pe_type() == pe_type_32
		? rebuild_tls_base<pe_types_class_32>(pe, info, tls_section, offset_from_section_start, write_tls_callbacks, write_tls_data, expand, save_to_pe_header, auto_strip_last_section)
		: rebuild_tls_base<pe_types_class_64>(pe, info, tls_section, offset_from_section_start, write_tls_callbacks, write_tls_data, expand, save_to_pe_header, auto_strip_last_section);
}

//Get TLS info
//If image does not have TLS, throws an exception
template<typename PEClassType>
const tls_info get_tls_info_base(const pe_base& pe)
{
	tls_info ret;

	//If there's no TLS directory, throw an exception
	if(!pe.has_tls())
		throw pe_exception("Image does not have TLS directory", pe_exception::directory_does_not_exist);

	//Get TLS directory data
	typename PEClassType::TLSStruct tls_directory_data = pe.section_data_from_rva<typename PEClassType::TLSStruct>(pe.get_directory_rva(image_directory_entry_tls), section_data_virtual, true);

	//Check data addresses
	if(tls_directory_data.EndAddressOfRawData == tls_directory_data.StartAddressOfRawData)
	{
		try
		{
			pe.va_to_rva(static_cast<typename PEClassType::BaseSize>(tls_directory_data.EndAddressOfRawData));
		}
		catch(const pe_exception&)
		{
			//Fix addressess on incorrect conversion
			tls_directory_data.EndAddressOfRawData = tls_directory_data.StartAddressOfRawData = 0;
		}
	}

	if(tls_directory_data.StartAddressOfRawData &&
		pe.section_data_length_from_va(static_cast<typename PEClassType::BaseSize>(tls_directory_data.StartAddressOfRawData),
		static_cast<typename PEClassType::BaseSize>(tls_directory_data.StartAddressOfRawData), section_data_virtual, true)
		< (tls_directory_data.EndAddressOfRawData - tls_directory_data.StartAddressOfRawData))
		throw pe_exception("Incorrect TLS directory", pe_exception::incorrect_tls_directory);

	//Fill TLS info
	//VAs are not checked
	ret.set_raw_data_start_rva(tls_directory_data.StartAddressOfRawData ? pe.va_to_rva(static_cast<typename PEClassType::BaseSize>(tls_directory_data.StartAddressOfRawData)) : 0);
	ret.set_raw_data_end_rva(tls_directory_data.EndAddressOfRawData ? pe.va_to_rva(static_cast<typename PEClassType::BaseSize>(tls_directory_data.EndAddressOfRawData)) : 0);
	ret.set_index_rva(tls_directory_data.AddressOfIndex ? pe.va_to_rva(static_cast<typename PEClassType::BaseSize>(tls_directory_data.AddressOfIndex)) : 0);
	ret.set_callbacks_rva(tls_directory_data.AddressOfCallBacks ? pe.va_to_rva(static_cast<typename PEClassType::BaseSize>(tls_directory_data.AddressOfCallBacks)) : 0);
	ret.set_size_of_zero_fill(tls_directory_data.SizeOfZeroFill);
	ret.set_characteristics(tls_directory_data.Characteristics);

	if(tls_directory_data.StartAddressOfRawData && tls_directory_data.StartAddressOfRawData != tls_directory_data.EndAddressOfRawData)
	{
		//Read and save TLS RAW data
		ret.set_raw_data(std::string(
			pe.section_data_from_va(static_cast<typename PEClassType::BaseSize>(tls_directory_data.StartAddressOfRawData), section_data_virtual, true),
			static_cast<uint32_t>(tls_directory_data.EndAddressOfRawData - tls_directory_data.StartAddressOfRawData)));
	}

	//If file has TLS callbacks
	if(ret.get_callbacks_rva())
	{
		//Read callbacks VAs
		uint32_t current_tls_callback = 0;

		while(true)
		{
			//Read TLS callback VA
			typename PEClassType::BaseSize va = pe.section_data_from_va<typename PEClassType::BaseSize>(static_cast<typename PEClassType::BaseSize>(tls_directory_data.AddressOfCallBacks + current_tls_callback), section_data_virtual, true);
			if(va == 0)
				break;

			//Save it
			ret.add_tls_callback(pe.va_to_rva(va, false));

			//Move to next callback VA
			current_tls_callback += sizeof(va);
		}
	}

	return ret;
}

//Rebuilder of TLS structures
//If write_tls_callbacks = true, TLS callbacks VAs will be written to their place
//If write_tls_data = true, TLS data will be written to its place
//If you have chosen to rewrite raw data, only (EndAddressOfRawData - StartAddressOfRawData) bytes will be written, not the full length of string
//representing raw data content
//auto_strip_last_section - if true and TLS are placed in the last section, it will be automatically stripped
//Note/TODO: TLS Callbacks array is not DWORD-aligned (seems to work on WinXP - Win7)
template<typename PEClassType>
const image_directory rebuild_tls_base(pe_base& pe, const tls_info& info, section& tls_section, uint32_t offset_from_section_start, bool write_tls_callbacks, bool write_tls_data, tls_data_expand_type expand, bool save_to_pe_header, bool auto_strip_last_section)
{
	//Check that tls_section is attached to this PE image
	if(!pe.section_attached(tls_section))
		throw pe_exception("TLS section must be attached to PE file", pe_exception::section_is_not_attached);
	
	uint32_t tls_data_pos = pe_utils::align_up(offset_from_section_start, sizeof(typename PEClassType::BaseSize));
	uint32_t needed_size = sizeof(typename PEClassType::TLSStruct); //Calculate needed size for TLS table
	
	//Check if tls_section is last one. If it's not, check if there's enough place for TLS data
	if(&tls_section != &*(pe.get_image_sections().end() - 1) && 
		(tls_section.empty() || pe_utils::align_up(tls_section.get_size_of_raw_data(), pe.get_file_alignment()) < needed_size + tls_data_pos))
		throw pe_exception("Insufficient space for TLS directory", pe_exception::insufficient_space);

	//Check raw data positions
	if(info.get_raw_data_end_rva() < info.get_raw_data_start_rva() || info.get_index_rva() == 0)
		throw pe_exception("Incorrect TLS directory", pe_exception::incorrect_tls_directory);

	std::string& raw_data = tls_section.get_raw_data();

	//This will be done only if tls_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + tls_data_pos)
		raw_data.resize(needed_size + tls_data_pos); //Expand section raw data

	//Create and fill TLS structure
	typename PEClassType::TLSStruct tls_struct = {0};
	
	typename PEClassType::BaseSize va;
	if(info.get_raw_data_start_rva())
	{
		pe.rva_to_va(info.get_raw_data_start_rva(), va);
		tls_struct.StartAddressOfRawData = va;
		tls_struct.SizeOfZeroFill = info.get_size_of_zero_fill();
	}

	if(info.get_raw_data_end_rva())
	{
		pe.rva_to_va(info.get_raw_data_end_rva(), va);
		tls_struct.EndAddressOfRawData = va;
	}

	pe.rva_to_va(info.get_index_rva(), va);
	tls_struct.AddressOfIndex = va;

	if(info.get_callbacks_rva())
	{
		pe.rva_to_va(info.get_callbacks_rva(), va);
		tls_struct.AddressOfCallBacks = va;
	}

	tls_struct.Characteristics = info.get_characteristics();

	//Save TLS structure
	memcpy(&raw_data[tls_data_pos], &tls_struct, sizeof(tls_struct));

	//If we are asked to rewrite TLS raw data
	if(write_tls_data && info.get_raw_data_start_rva() && info.get_raw_data_start_rva() != info.get_raw_data_end_rva())
	{
		try
		{
			//Check if we're going to write TLS raw data to an existing section (not to PE headers)
			section& raw_data_section = pe.section_from_rva(info.get_raw_data_start_rva());
			pe.expand_section(raw_data_section, info.get_raw_data_start_rva(), info.get_raw_data_end_rva() - info.get_raw_data_start_rva(), expand == tls_data_expand_raw ? pe_base::expand_section_raw : pe_base::expand_section_virtual);
		}
		catch(const pe_exception&)
		{
			//If no section is presented by StartAddressOfRawData, just go to next step
		}

		unsigned long write_raw_data_size = info.get_raw_data_end_rva() - info.get_raw_data_start_rva();
		unsigned long available_raw_length = 0;

		//Check if there's enough RAW space to write raw TLS data...
		if((available_raw_length = pe.section_data_length_from_rva(info.get_raw_data_start_rva(), info.get_raw_data_start_rva(), section_data_raw, true))
			< info.get_raw_data_end_rva() - info.get_raw_data_start_rva())
		{
			//Check if there's enough virtual space for it...
			if(pe.section_data_length_from_rva(info.get_raw_data_start_rva(), info.get_raw_data_start_rva(), section_data_virtual, true)
				< info.get_raw_data_end_rva() - info.get_raw_data_start_rva())
				throw pe_exception("Insufficient space for TLS raw data", pe_exception::insufficient_space);
			else
				write_raw_data_size = available_raw_length; //We'll write just a part of full raw data
		}

		//Write raw TLS data, if any
		if(write_raw_data_size != 0)
			memcpy(pe.section_data_from_rva(info.get_raw_data_start_rva(), true), info.get_raw_data().data(), write_raw_data_size);
	}

	//If we are asked to rewrite TLS callbacks addresses
	if(write_tls_callbacks && info.get_callbacks_rva())
	{
		unsigned long needed_callback_size = static_cast<unsigned long>((info.get_tls_callbacks().size() + 1 /* last null element */) * sizeof(typename PEClassType::BaseSize));

		try
		{
			//Check if we're going to write TLS callbacks VAs to an existing section (not to PE headers)
			section& raw_data_section = pe.section_from_rva(info.get_callbacks_rva());
			pe.expand_section(raw_data_section, info.get_callbacks_rva(), needed_callback_size, pe_base::expand_section_raw);
		}
		catch(const pe_exception&)
		{
			//If no section is presented by RVA of callbacks, just go to next step
		}

		//Check if there's enough space to write callbacks TLS data...
		if(pe.section_data_length_from_rva(info.get_callbacks_rva(), info.get_callbacks_rva(), section_data_raw, true)
			< needed_callback_size - sizeof(typename PEClassType::BaseSize) /* last zero element can be virtual only */)
			throw pe_exception("Insufficient space for TLS callbacks data", pe_exception::insufficient_space);
		
		if(pe.section_data_length_from_rva(info.get_callbacks_rva(), info.get_callbacks_rva(), section_data_virtual, true)
			< needed_callback_size /* check here full virtual data length available */)
			throw pe_exception("Insufficient space for TLS callbacks data", pe_exception::insufficient_space);

		std::vector<typename PEClassType::BaseSize> callbacks_virtual_addresses;
		callbacks_virtual_addresses.reserve(info.get_tls_callbacks().size() + 1 /* last null element */);

		//Convert TLS RVAs to VAs
		for(tls_info::tls_callback_list::const_iterator it = info.get_tls_callbacks().begin(); it != info.get_tls_callbacks().end(); ++it)
		{
			typename PEClassType::BaseSize cb_va = 0;
			pe.rva_to_va(*it, cb_va);
			callbacks_virtual_addresses.push_back(cb_va);
		}

		//Ending null element
		callbacks_virtual_addresses.push_back(0);

		//Write callbacks TLS data
		memcpy(pe.section_data_from_rva(info.get_callbacks_rva(), true), &callbacks_virtual_addresses[0], needed_callback_size);
	}
	
	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(tls_section, auto_strip_last_section);

	image_directory ret(pe.rva_from_section_offset(tls_section, tls_data_pos), needed_size);

	//If auto-rewrite of PE headers is required
	if(save_to_pe_header)
	{
		pe.set_directory_rva(image_directory_entry_tls, ret.get_rva());
		pe.set_directory_size(image_directory_entry_tls, ret.get_size());
	}

	return ret;
}
}
