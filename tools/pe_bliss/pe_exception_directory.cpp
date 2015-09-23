#include "pe_exception_directory.h"

namespace pe_bliss
{
using namespace pe_win;

//EXCEPTION DIRECTORY (exists on PE+ only)
//Default constructor
exception_entry::exception_entry()
	:begin_address_(0), end_address_(0), unwind_info_address_(0),
	unwind_info_version_(0),
	flags_(0),
	size_of_prolog_(0),
	count_of_codes_(0),
	frame_register_(0),
	frame_offset_(0)
{}

//Constructor from data
exception_entry::exception_entry(const image_runtime_function_entry& entry, const unwind_info& unwind_info)
	:begin_address_(entry.BeginAddress), end_address_(entry.EndAddress), unwind_info_address_(entry.UnwindInfoAddress),
	unwind_info_version_(unwind_info.Version),
	flags_(unwind_info.Flags),
	size_of_prolog_(unwind_info.SizeOfProlog),
	count_of_codes_(unwind_info.CountOfCodes),
	frame_register_(unwind_info.FrameRegister),
	frame_offset_(unwind_info.FrameOffset)
{}

//Returns starting address of function, affected by exception unwinding
uint32_t exception_entry::get_begin_address() const
{
	return begin_address_;
}

//Returns ending address of function, affected by exception unwinding
uint32_t exception_entry::get_end_address() const
{
	return end_address_;
}

//Returns unwind info address
uint32_t exception_entry::get_unwind_info_address() const
{
	return unwind_info_address_;
}

//Returns UNWIND_INFO structure version
uint8_t exception_entry::get_unwind_info_version() const
{
	return unwind_info_version_;
}

//Returns unwind info flags
uint8_t exception_entry::get_flags() const
{
	return flags_;
}

//The function has an exception handler that should be called
//when looking for functions that need to examine exceptions
bool exception_entry::has_exception_handler() const
{
	return (flags_ & unw_flag_ehandler) ? true : false;
}

//The function has a termination handler that should be called
//when unwinding an exception
bool exception_entry::has_termination_handler() const
{
	return (flags_ & unw_flag_uhandler) ? true : false;
}

//The unwind info structure is not the primary one for the procedure
bool exception_entry::is_chaininfo() const
{
	return (flags_ & unw_flag_chaininfo) ? true : false;
}

//Returns size of function prolog
uint8_t exception_entry::get_size_of_prolog() const
{
	return size_of_prolog_;
}

//Returns number of unwind slots
uint8_t exception_entry::get_number_of_unwind_slots() const
{
	return count_of_codes_;
}

//If the function uses frame pointer
bool exception_entry::uses_frame_pointer() const
{
	return frame_register_ != 0;
}

//Number of the nonvolatile register used as the frame pointer,
//using the same encoding for the operation info field of UNWIND_CODE nodes
uint8_t exception_entry::get_frame_pointer_register_number() const
{
	return frame_register_;
}

//The scaled offset from RSP that is applied to the FP reg when it is established.
//The actual FP reg is set to RSP + 16 * this number, allowing offsets from 0 to 240
uint8_t exception_entry::get_scaled_rsp_offset() const
{
	return frame_offset_;
}

//Returns exception directory data (exists on PE+ only)
//Unwind opcodes are not listed, because their format and list are subject to change
const exception_entry_list get_exception_directory_data(const pe_base& pe)
{
	exception_entry_list ret;

	//If image doesn't have exception directory, return empty list
	if(!pe.has_exception_directory())
		return ret;

	//Check the length in bytes of the section containing exception directory
	if(pe.section_data_length_from_rva(pe.get_directory_rva(image_directory_entry_exception), pe.get_directory_rva(image_directory_entry_exception), section_data_virtual, true)
		< sizeof(image_runtime_function_entry))
		throw pe_exception("Incorrect exception directory", pe_exception::incorrect_exception_directory);

	unsigned long current_pos = pe.get_directory_rva(image_directory_entry_exception);

	//Check if structures are DWORD-aligned
	if(current_pos % sizeof(uint32_t))
		throw pe_exception("Incorrect exception directory", pe_exception::incorrect_exception_directory);

	//First IMAGE_RUNTIME_FUNCTION_ENTRY table
	image_runtime_function_entry exception_table = pe.section_data_from_rva<image_runtime_function_entry>(current_pos, section_data_virtual, true);

	//todo: virtual addresses BeginAddress and EndAddress are not checked to be inside image
	while(exception_table.BeginAddress)
	{
		//Check addresses
		if(exception_table.BeginAddress > exception_table.EndAddress)
			throw pe_exception("Incorrect exception directory", pe_exception::incorrect_exception_directory);

		//Get unwind information
		unwind_info info = pe.section_data_from_rva<unwind_info>(exception_table.UnwindInfoAddress, section_data_virtual, true);

		//Create exception entry and save it
		ret.push_back(exception_entry(exception_table, info));

		//Go to next exception entry
		current_pos += sizeof(image_runtime_function_entry);
		exception_table = pe.section_data_from_rva<image_runtime_function_entry>(current_pos, section_data_virtual, true);
	}

	return ret;
}
}
