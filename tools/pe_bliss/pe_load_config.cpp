#include <algorithm>
#include <string.h>
#include "pe_load_config.h"
#include "pe_properties_generic.h"

namespace pe_bliss
{
using namespace pe_win;

//IMAGE CONFIG
//Default constructor
image_config_info::image_config_info()
	:time_stamp_(0),
	major_version_(0), minor_version_(0),
	global_flags_clear_(0), global_flags_set_(0),
	critical_section_default_timeout_(0),
	decommit_free_block_threshold_(0), decommit_total_free_threshold_(0),
	lock_prefix_table_va_(0),
	max_allocation_size_(0),
	virtual_memory_threshold_(0),
	process_affinity_mask_(0),
	process_heap_flags_(0),
	service_pack_version_(0),
	edit_list_va_(0),
	security_cookie_va_(0),
	se_handler_table_va_(0),
	se_handler_count_(0)
{}

//Constructors from PE structures
template<typename ConfigStructure>
image_config_info::image_config_info(const ConfigStructure& info)
	:time_stamp_(info.TimeDateStamp),
	major_version_(info.MajorVersion), minor_version_(info.MinorVersion),
	global_flags_clear_(info.GlobalFlagsClear), global_flags_set_(info.GlobalFlagsSet),
	critical_section_default_timeout_(info.CriticalSectionDefaultTimeout),
	decommit_free_block_threshold_(info.DeCommitFreeBlockThreshold), decommit_total_free_threshold_(info.DeCommitTotalFreeThreshold),
	lock_prefix_table_va_(info.LockPrefixTable),
	max_allocation_size_(info.MaximumAllocationSize),
	virtual_memory_threshold_(info.VirtualMemoryThreshold),
	process_affinity_mask_(info.ProcessAffinityMask),
	process_heap_flags_(info.ProcessHeapFlags),
	service_pack_version_(info.CSDVersion),
	edit_list_va_(info.EditList),
	security_cookie_va_(info.SecurityCookie),
	se_handler_table_va_(info.SEHandlerTable),
	se_handler_count_(info.SEHandlerCount)
{}

//Instantiate template constructor with needed structures
template image_config_info::image_config_info(const image_load_config_directory32& info);
template image_config_info::image_config_info(const image_load_config_directory64& info);

//Returns the date and time stamp value
uint32_t image_config_info::get_time_stamp() const
{
	return time_stamp_;
}

//Returns major version number
uint16_t image_config_info::get_major_version() const
{
	return major_version_;
}

//Returns minor version number
uint16_t image_config_info::get_minor_version() const
{
	return minor_version_;
}

//Returns clear global flags
uint32_t image_config_info::get_global_flags_clear() const
{
	return global_flags_clear_;
}

//Returns set global flags
uint32_t image_config_info::get_global_flags_set() const
{
	return global_flags_set_;
}

//Returns critical section default timeout
uint32_t image_config_info::get_critical_section_default_timeout() const
{
	return critical_section_default_timeout_;
}

//Get the size of the minimum block that
//must be freed before it is freed (de-committed), in bytes
uint64_t image_config_info::get_decommit_free_block_threshold() const
{
	return decommit_free_block_threshold_;
}

//Returns the size of the minimum total memory
//that must be freed in the process heap before it is freed (de-committed), in bytes
uint64_t image_config_info::get_decommit_total_free_threshold() const
{
	return decommit_total_free_threshold_;
}

//Returns VA of a list of addresses where the LOCK prefix is used
uint64_t image_config_info::get_lock_prefix_table_va() const
{
	return lock_prefix_table_va_;
}

//Returns the maximum allocation size, in bytes
uint64_t image_config_info::get_max_allocation_size() const
{
	return max_allocation_size_;
}

//Returns the maximum block size that can be allocated from heap segments, in bytes
uint64_t image_config_info::get_virtual_memory_threshold() const
{
	return virtual_memory_threshold_;
}

//Returns process affinity mask
uint64_t image_config_info::get_process_affinity_mask() const
{
	return process_affinity_mask_;
}

//Returns process heap flags
uint32_t image_config_info::get_process_heap_flags() const
{
	return process_heap_flags_;
}

//Returns service pack version (CSDVersion)
uint16_t image_config_info::get_service_pack_version() const
{
	return service_pack_version_;
}

//Returns VA of edit list (reserved by system)
uint64_t image_config_info::get_edit_list_va() const
{
	return edit_list_va_;
}

//Returns a pointer to a cookie that is used by Visual C++ or GS implementation
uint64_t image_config_info::get_security_cookie_va() const
{
	return security_cookie_va_;
}

//Returns VA of the sorted table of RVAs of each valid, unique handler in the image
uint64_t image_config_info::get_se_handler_table_va() const
{
	return se_handler_table_va_;
}

//Returns the count of unique handlers in the table
uint64_t image_config_info::get_se_handler_count() const
{
	return se_handler_count_;
}

//Returns SE Handler RVA list
const image_config_info::se_handler_list& image_config_info::get_se_handler_rvas() const
{
	return se_handlers_;
}

//Returns Lock Prefix RVA list
const image_config_info::lock_prefix_rva_list& image_config_info::get_lock_prefix_rvas() const
{
	return lock_prefixes_;
}

//Adds SE Handler RVA to list
void image_config_info::add_se_handler_rva(uint32_t rva)
{
	se_handlers_.push_back(rva);
}

//Clears SE Handler list
void image_config_info::clear_se_handler_list()
{
	se_handlers_.clear();
}

//Adds Lock Prefix RVA to list
void image_config_info::add_lock_prefix_rva(uint32_t rva)
{
	lock_prefixes_.push_back(rva);
}

//Clears Lock Prefix list
void image_config_info::clear_lock_prefix_list()
{
	lock_prefixes_.clear();
}

//Sets the date and time stamp value
void image_config_info::set_time_stamp(uint32_t time_stamp)
{
	time_stamp_ = time_stamp;
}

//Sets major version number
void image_config_info::set_major_version(uint16_t major_version)
{
	major_version_ = major_version;
}

//Sets minor version number
void image_config_info::set_minor_version(uint16_t minor_version)
{
	minor_version_ = minor_version;
}

//Sets clear global flags
void image_config_info::set_global_flags_clear(uint32_t global_flags_clear)
{
	global_flags_clear_ = global_flags_clear;
}

//Sets set global flags
void image_config_info::set_global_flags_set(uint32_t global_flags_set)
{
	global_flags_set_ = global_flags_set;
}

//Sets critical section default timeout
void image_config_info::set_critical_section_default_timeout(uint32_t critical_section_default_timeout)
{
	critical_section_default_timeout_ = critical_section_default_timeout;
}

//Sets the size of the minimum block that
//must be freed before it is freed (de-committed), in bytes
void image_config_info::set_decommit_free_block_threshold(uint64_t decommit_free_block_threshold)
{
	decommit_free_block_threshold_ = decommit_free_block_threshold;
}

//Sets the size of the minimum total memory
//that must be freed in the process heap before it is freed (de-committed), in bytes
void image_config_info::set_decommit_total_free_threshold(uint64_t decommit_total_free_threshold)
{
	decommit_total_free_threshold_ = decommit_total_free_threshold;
}

//Sets VA of a list of addresses where the LOCK prefix is used
//If you rebuild this list, VA will be re-assigned automatically
void image_config_info::set_lock_prefix_table_va(uint64_t lock_prefix_table_va)
{
	lock_prefix_table_va_ = lock_prefix_table_va;
}

//Sets the maximum allocation size, in bytes
void image_config_info::set_max_allocation_size(uint64_t max_allocation_size)
{
	max_allocation_size_ = max_allocation_size;
}

//Sets the maximum block size that can be allocated from heap segments, in bytes
void image_config_info::set_virtual_memory_threshold(uint64_t virtual_memory_threshold)
{
	virtual_memory_threshold_ = virtual_memory_threshold;
}

//Sets process affinity mask
void image_config_info::set_process_affinity_mask(uint64_t process_affinity_mask)
{
	process_affinity_mask_ = process_affinity_mask;
}

//Sets process heap flags
void image_config_info::set_process_heap_flags(uint32_t process_heap_flags)
{
	process_heap_flags_ = process_heap_flags;
}

//Sets service pack version (CSDVersion)
void image_config_info::set_service_pack_version(uint16_t service_pack_version)
{
	service_pack_version_ = service_pack_version;
}

//Sets VA of edit list (reserved by system)
void image_config_info::set_edit_list_va(uint64_t edit_list_va)
{
	edit_list_va_ = edit_list_va;
}

//Sets a pointer to a cookie that is used by Visual C++ or GS implementation
void image_config_info::set_security_cookie_va(uint64_t security_cookie_va)
{
	security_cookie_va_ = security_cookie_va;
}

//Sets VA of the sorted table of RVAs of each valid, unique handler in the image
//If you rebuild this list, VA will be re-assigned automatically
void image_config_info::set_se_handler_table_va(uint64_t se_handler_table_va)
{
	se_handler_table_va_ = se_handler_table_va;
}

//Returns SE Handler RVA list
image_config_info::se_handler_list& image_config_info::get_se_handler_rvas()
{
	return se_handlers_;
}

//Returns Lock Prefix RVA list
image_config_info::lock_prefix_rva_list& image_config_info::get_lock_prefix_rvas()
{
	return lock_prefixes_;
}

//Returns image config info
//If image does not have config info, throws an exception
const image_config_info get_image_config(const pe_base& pe)
{
	return pe.get_pe_type() == pe_type_32
		? get_image_config_base<pe_types_class_32>(pe)
		: get_image_config_base<pe_types_class_64>(pe);
}

//Image config rebuilder
const image_directory rebuild_image_config(pe_base& pe, const image_config_info& info, section& image_config_section, uint32_t offset_from_section_start, bool write_se_handlers, bool write_lock_prefixes, bool save_to_pe_header, bool auto_strip_last_section)
{
	return pe.get_pe_type() == pe_type_32
		? rebuild_image_config_base<pe_types_class_32>(pe, info, image_config_section, offset_from_section_start, write_se_handlers, write_lock_prefixes, save_to_pe_header, auto_strip_last_section)
		: rebuild_image_config_base<pe_types_class_64>(pe, info, image_config_section, offset_from_section_start, write_se_handlers, write_lock_prefixes, save_to_pe_header, auto_strip_last_section);
}


//Returns image config info
//If image does not have config info, throws an exception
template<typename PEClassType>
const image_config_info get_image_config_base(const pe_base& pe)
{
	//Check if image has config directory
	if(!pe.has_config())
		throw pe_exception("Image does not have load config directory", pe_exception::directory_does_not_exist);

	//Get load config structure
	typename PEClassType::ConfigStruct config_info = pe.section_data_from_rva<typename PEClassType::ConfigStruct>(pe.get_directory_rva(image_directory_entry_load_config), section_data_virtual);

	//Check size of config directory
	if(config_info.Size != sizeof(config_info))
		throw pe_exception("Incorrect (or old) load config directory", pe_exception::incorrect_config_directory);

	//Fill return structure
	image_config_info ret(config_info);

	//Check possible overflow
	if(config_info.SEHandlerCount >= pe_utils::max_dword / sizeof(uint32_t)
		|| config_info.SEHandlerTable >= static_cast<typename PEClassType::BaseSize>(-1) - config_info.SEHandlerCount * sizeof(uint32_t))
		throw pe_exception("Incorrect load config directory", pe_exception::incorrect_config_directory);

	//Read sorted SE handler RVA list (if any)
	for(typename PEClassType::BaseSize i = 0; i != config_info.SEHandlerCount; ++i)
		ret.add_se_handler_rva(pe.section_data_from_va<uint32_t>(static_cast<typename PEClassType::BaseSize>(config_info.SEHandlerTable + i * sizeof(uint32_t))));

	if(config_info.LockPrefixTable)
	{
		//Read Lock Prefix VA list (if any)
		unsigned long current = 0;
		while(true)
		{
			typename PEClassType::BaseSize lock_prefix_va = pe.section_data_from_va<typename PEClassType::BaseSize>(static_cast<typename PEClassType::BaseSize>(config_info.LockPrefixTable + current * sizeof(typename PEClassType::BaseSize)));
			if(!lock_prefix_va)
				break;

			ret.add_lock_prefix_rva(pe.va_to_rva(lock_prefix_va));

			++current;
		}
	}

	return ret;
}

//Image config directory rebuilder
//auto_strip_last_section - if true and TLS are placed in the last section, it will be automatically stripped
//If write_se_handlers = true, SE Handlers list will be written just after image config directory structure
//If write_lock_prefixes = true, Lock Prefixes address list will be written just after image config directory structure
template<typename PEClassType>
const image_directory rebuild_image_config_base(pe_base& pe, const image_config_info& info, section& image_config_section, uint32_t offset_from_section_start, bool write_se_handlers, bool write_lock_prefixes, bool save_to_pe_header, bool auto_strip_last_section)
{
	//Check that image_config_section is attached to this PE image
	if(!pe.section_attached(image_config_section))
		throw pe_exception("Image Config section must be attached to PE file", pe_exception::section_is_not_attached);
	
	uint32_t alignment = pe_utils::align_up(offset_from_section_start, sizeof(typename PEClassType::BaseSize)) - offset_from_section_start;

	uint32_t needed_size = sizeof(typename PEClassType::ConfigStruct); //Calculate needed size for Image Config table
	
	uint32_t image_config_data_pos = offset_from_section_start + alignment;

	uint32_t current_pos_of_se_handlers = 0;
	uint32_t current_pos_of_lock_prefixes = 0;
	
	if(write_se_handlers)
	{
		current_pos_of_se_handlers = needed_size + image_config_data_pos;
		needed_size += static_cast<uint32_t>(info.get_se_handler_rvas().size()) * sizeof(uint32_t); //RVAs of SE Handlers
	}
	
	if(write_lock_prefixes)
	{
		current_pos_of_lock_prefixes = needed_size + image_config_data_pos;
		needed_size += static_cast<uint32_t>((info.get_lock_prefix_rvas().size() + 1) * sizeof(typename PEClassType::BaseSize)); //VAs of Lock Prefixes (and ending null element)
	}

	//Check if image_config_section is last one. If it's not, check if there's enough place for Image Config data
	if(&image_config_section != &*(pe.get_image_sections().end() - 1) && 
		(image_config_section.empty() || pe_utils::align_up(image_config_section.get_size_of_raw_data(), pe.get_file_alignment()) < needed_size + image_config_data_pos))
		throw pe_exception("Insufficient space for TLS directory", pe_exception::insufficient_space);

	std::string& raw_data = image_config_section.get_raw_data();

	//This will be done only if image_config_section is the last section of image or for section with unaligned raw length of data
	if(raw_data.length() < needed_size + image_config_data_pos)
		raw_data.resize(needed_size + image_config_data_pos); //Expand section raw data

	//Create and fill Image Config structure
	typename PEClassType::ConfigStruct image_config_section_struct = {0};
	image_config_section_struct.Size = sizeof(image_config_section_struct);
	image_config_section_struct.TimeDateStamp = info.get_time_stamp();
	image_config_section_struct.MajorVersion = info.get_major_version();
	image_config_section_struct.MinorVersion = info.get_minor_version();
	image_config_section_struct.GlobalFlagsClear = info.get_global_flags_clear();
	image_config_section_struct.GlobalFlagsSet = info.get_global_flags_set();
	image_config_section_struct.CriticalSectionDefaultTimeout = info.get_critical_section_default_timeout();
	image_config_section_struct.DeCommitFreeBlockThreshold = static_cast<typename PEClassType::BaseSize>(info.get_decommit_free_block_threshold());
	image_config_section_struct.DeCommitTotalFreeThreshold = static_cast<typename PEClassType::BaseSize>(info.get_decommit_total_free_threshold());
	image_config_section_struct.MaximumAllocationSize = static_cast<typename PEClassType::BaseSize>(info.get_max_allocation_size());
	image_config_section_struct.VirtualMemoryThreshold = static_cast<typename PEClassType::BaseSize>(info.get_virtual_memory_threshold());
	image_config_section_struct.ProcessHeapFlags = info.get_process_heap_flags();
	image_config_section_struct.ProcessAffinityMask = static_cast<typename PEClassType::BaseSize>(info.get_process_affinity_mask());
	image_config_section_struct.CSDVersion = info.get_service_pack_version();
	image_config_section_struct.EditList = static_cast<typename PEClassType::BaseSize>(info.get_edit_list_va());
	image_config_section_struct.SecurityCookie = static_cast<typename PEClassType::BaseSize>(info.get_security_cookie_va());
	image_config_section_struct.SEHandlerCount = static_cast<typename PEClassType::BaseSize>(info.get_se_handler_rvas().size());
	

	if(write_se_handlers)
	{
		if(info.get_se_handler_rvas().empty())
		{
			write_se_handlers = false;
			image_config_section_struct.SEHandlerTable = 0;
		}
		else
		{
			typename PEClassType::BaseSize va;
			pe.rva_to_va(pe.rva_from_section_offset(image_config_section, current_pos_of_se_handlers), va);
			image_config_section_struct.SEHandlerTable = va;
		}
	}
	else
	{
		image_config_section_struct.SEHandlerTable = static_cast<typename PEClassType::BaseSize>(info.get_se_handler_table_va());
	}

	if(write_lock_prefixes)
	{
		if(info.get_lock_prefix_rvas().empty())
		{
			write_lock_prefixes = false;
			image_config_section_struct.LockPrefixTable = 0;
		}
		else
		{
			typename PEClassType::BaseSize va;
			pe.rva_to_va(pe.rva_from_section_offset(image_config_section, current_pos_of_lock_prefixes), va);
			image_config_section_struct.LockPrefixTable = va;
		}
	}
	else
	{
		image_config_section_struct.LockPrefixTable = static_cast<typename PEClassType::BaseSize>(info.get_lock_prefix_table_va());
	}

	//Write image config section
	memcpy(&raw_data[image_config_data_pos], &image_config_section_struct, sizeof(image_config_section_struct));

	if(write_se_handlers)
	{
		//Sort SE Handlers list
		image_config_info::se_handler_list sorted_list = info.get_se_handler_rvas();
		std::sort(sorted_list.begin(), sorted_list.end());

		//Write SE Handlers table
		for(image_config_info::se_handler_list::const_iterator it = sorted_list.begin(); it != sorted_list.end(); ++it)
		{
			uint32_t se_handler_rva = *it;
			memcpy(&raw_data[current_pos_of_se_handlers], &se_handler_rva, sizeof(se_handler_rva));
			current_pos_of_se_handlers += sizeof(se_handler_rva);
		}
	}

	if(write_lock_prefixes)
	{
		//Write Lock Prefixes VA list
		for(image_config_info::lock_prefix_rva_list::const_iterator it = info.get_lock_prefix_rvas().begin(); it != info.get_lock_prefix_rvas().end(); ++it)
		{
			typename PEClassType::BaseSize lock_prefix_va;
			pe.rva_to_va(*it, lock_prefix_va);
			memcpy(&raw_data[current_pos_of_lock_prefixes], &lock_prefix_va, sizeof(lock_prefix_va));
			current_pos_of_lock_prefixes += sizeof(lock_prefix_va);
		}

		{
			//Ending null VA
			typename PEClassType::BaseSize lock_prefix_va = 0;
			memcpy(&raw_data[current_pos_of_lock_prefixes], &lock_prefix_va, sizeof(lock_prefix_va));
		}
	}

	//Adjust section raw and virtual sizes
	pe.recalculate_section_sizes(image_config_section, auto_strip_last_section);

	image_directory ret(pe.rva_from_section_offset(image_config_section, image_config_data_pos), sizeof(typename PEClassType::ConfigStruct));

	//If auto-rewrite of PE headers is required
	if(save_to_pe_header)
	{
		pe.set_directory_rva(image_directory_entry_load_config, ret.get_rva());
		pe.set_directory_size(image_directory_entry_load_config, ret.get_size());
	}

	return ret;
}

}
