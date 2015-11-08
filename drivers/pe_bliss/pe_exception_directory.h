#pragma once
#include <vector>
#include "pe_structures.h"
#include "pe_base.h"

namespace pe_bliss
{
//Class representing exception directory entry
class exception_entry
{
public:
	//Default constructor
	exception_entry();
	//Constructor from data
	exception_entry(const pe_win::image_runtime_function_entry& entry, const pe_win::unwind_info& unwind_info);

	//Returns starting address of function, affected by exception unwinding
	uint32_t get_begin_address() const;
	//Returns ending address of function, affected by exception unwinding
	uint32_t get_end_address() const;
	//Returns unwind info address
	uint32_t get_unwind_info_address() const;

	//Returns UNWIND_INFO structure version
	uint8_t get_unwind_info_version() const;

	//Returns unwind info flags
	uint8_t get_flags() const;
	//The function has an exception handler that should be called
	//when looking for functions that need to examine exceptions
	bool has_exception_handler() const;
	//The function has a termination handler that should be called
	//when unwinding an exception
	bool has_termination_handler() const;
	//The unwind info structure is not the primary one for the procedure
	bool is_chaininfo() const;

	//Returns size of function prolog
	uint8_t get_size_of_prolog() const;

	//Returns number of unwind slots
	uint8_t get_number_of_unwind_slots() const;

	//If the function uses frame pointer
	bool uses_frame_pointer() const;
	//Number of the nonvolatile register used as the frame pointer,
	//using the same encoding for the operation info field of UNWIND_CODE nodes
	uint8_t get_frame_pointer_register_number() const;
	//The scaled offset from RSP that is applied to the FP reg when it is established.
	//The actual FP reg is set to RSP + 16 * this number, allowing offsets from 0 to 240
	uint8_t get_scaled_rsp_offset() const;

private:
	uint32_t begin_address_, end_address_, unwind_info_address_;
	uint8_t unwind_info_version_;
	uint8_t flags_;
	uint8_t size_of_prolog_;
	uint8_t count_of_codes_;
	uint8_t frame_register_, frame_offset_;
};

typedef std::vector<exception_entry> exception_entry_list;

//Returns exception directory data (exists on PE+ only)
//Unwind opcodes are not listed, because their format and list are subject to change
const exception_entry_list get_exception_directory_data(const pe_base& pe);
}
