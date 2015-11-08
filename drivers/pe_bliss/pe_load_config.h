#pragma once
#include <vector>
#include "pe_structures.h"
#include "pe_base.h"
#include "pe_directory.h"

namespace pe_bliss
{
//Class representing image configuration information
class image_config_info
{
public:
	typedef std::vector<uint32_t> se_handler_list;
	typedef std::vector<uint32_t> lock_prefix_rva_list;

public:
	//Default constructor
	image_config_info();
	//Constructors from PE structures (no checks)
	template<typename ConfigStructure>
	explicit image_config_info(const ConfigStructure& info);

	//Returns the date and time stamp value
	uint32_t get_time_stamp() const;
	//Returns major version number
	uint16_t get_major_version() const;
	//Returns minor version number
	uint16_t get_minor_version() const;
	//Returns clear global flags
	uint32_t get_global_flags_clear() const;
	//Returns set global flags
	uint32_t get_global_flags_set() const;
	//Returns critical section default timeout
	uint32_t get_critical_section_default_timeout() const;
	//Get the size of the minimum block that
	//must be freed before it is freed (de-committed), in bytes
	uint64_t get_decommit_free_block_threshold() const;
	//Returns the size of the minimum total memory
	//that must be freed in the process heap before it is freed (de-committed), in bytes
	uint64_t get_decommit_total_free_threshold() const;
	//Returns VA of a list of addresses where the LOCK prefix is used
	uint64_t get_lock_prefix_table_va() const;
	//Returns the maximum allocation size, in bytes
	uint64_t get_max_allocation_size() const;
	//Returns the maximum block size that can be allocated from heap segments, in bytes
	uint64_t get_virtual_memory_threshold() const;
	//Returns process affinity mask
	uint64_t get_process_affinity_mask() const;
	//Returns process heap flags
	uint32_t get_process_heap_flags() const;
	//Returns service pack version (CSDVersion)
	uint16_t get_service_pack_version() const;
	//Returns VA of edit list (reserved by system)
	uint64_t get_edit_list_va() const;
	//Returns a pointer to a cookie that is used by Visual C++ or GS implementation
	uint64_t get_security_cookie_va() const;
	//Returns VA of the sorted table of RVAs of each valid, unique handler in the image
	uint64_t get_se_handler_table_va() const;
	//Returns the count of unique handlers in the table
	uint64_t get_se_handler_count() const;

	//Returns SE Handler RVA list
	const se_handler_list& get_se_handler_rvas() const;
		
	//Returns Lock Prefix RVA list
	const lock_prefix_rva_list& get_lock_prefix_rvas() const;

public: //These functions do not change everything inside image, they are used by PE class
	//Also you can use these functions to rebuild image config directory

	//Adds SE Handler RVA to list
	void add_se_handler_rva(uint32_t rva);
	//Clears SE Handler list
	void clear_se_handler_list();
		
	//Adds Lock Prefix RVA to list
	void add_lock_prefix_rva(uint32_t rva);
	//Clears Lock Prefix list
	void clear_lock_prefix_list();
		
	//Sets the date and time stamp value
	void set_time_stamp(uint32_t time_stamp);
	//Sets major version number
	void set_major_version(uint16_t major_version);
	//Sets minor version number
	void set_minor_version(uint16_t minor_version);
	//Sets clear global flags
	void set_global_flags_clear(uint32_t global_flags_clear);
	//Sets set global flags
	void set_global_flags_set(uint32_t global_flags_set);
	//Sets critical section default timeout
	void set_critical_section_default_timeout(uint32_t critical_section_default_timeout);
	//Sets the size of the minimum block that
	//must be freed before it is freed (de-committed), in bytes
	void set_decommit_free_block_threshold(uint64_t decommit_free_block_threshold);
	//Sets the size of the minimum total memory
	//that must be freed in the process heap before it is freed (de-committed), in bytes
	void set_decommit_total_free_threshold(uint64_t decommit_total_free_threshold);
	//Sets VA of a list of addresses where the LOCK prefix is used
	//If you rebuild this list, VA will be re-assigned automatically
	void set_lock_prefix_table_va(uint64_t lock_prefix_table_va);
	//Sets the maximum allocation size, in bytes
	void set_max_allocation_size(uint64_t max_allocation_size);
	//Sets the maximum block size that can be allocated from heap segments, in bytes
	void set_virtual_memory_threshold(uint64_t virtual_memory_threshold);
	//Sets process affinity mask
	void set_process_affinity_mask(uint64_t process_affinity_mask);
	//Sets process heap flags
	void set_process_heap_flags(uint32_t process_heap_flags);
	//Sets service pack version (CSDVersion)
	void set_service_pack_version(uint16_t service_pack_version);
	//Sets VA of edit list (reserved by system)
	void set_edit_list_va(uint64_t edit_list_va);
	//Sets a pointer to a cookie that is used by Visual C++ or GS implementation
	void set_security_cookie_va(uint64_t security_cookie_va);
	//Sets VA of the sorted table of RVAs of each valid, unique handler in the image
	//If you rebuild this list, VA will be re-assigned automatically
	void set_se_handler_table_va(uint64_t se_handler_table_va);

	//Returns SE Handler RVA list
	se_handler_list& get_se_handler_rvas();

	//Returns Lock Prefix RVA list
	lock_prefix_rva_list& get_lock_prefix_rvas();

private:
	uint32_t time_stamp_;
	uint16_t major_version_, minor_version_;
	uint32_t global_flags_clear_, global_flags_set_;
	uint32_t critical_section_default_timeout_;
	uint64_t decommit_free_block_threshold_, decommit_total_free_threshold_;
	uint64_t lock_prefix_table_va_;
	uint64_t max_allocation_size_;
	uint64_t virtual_memory_threshold_;
	uint64_t process_affinity_mask_;
	uint32_t process_heap_flags_;
	uint16_t service_pack_version_;
	uint64_t edit_list_va_;
	uint64_t security_cookie_va_;
	uint64_t se_handler_table_va_;
	uint64_t se_handler_count_;

	se_handler_list se_handlers_;
	lock_prefix_rva_list lock_prefixes_;
};

//Returns image config info
//If image does not have config info, throws an exception
const image_config_info get_image_config(const pe_base& pe);

template<typename PEClassType>
const image_config_info get_image_config_base(const pe_base& pe);


//Image config directory rebuilder
//auto_strip_last_section - if true and TLS are placed in the last section, it will be automatically stripped
//If write_se_handlers = true, SE Handlers list will be written just after image config directory structure
//If write_lock_prefixes = true, Lock Prefixes address list will be written just after image config directory structure
const image_directory rebuild_image_config(pe_base& pe, const image_config_info& info, section& image_config_section, uint32_t offset_from_section_start = 0, bool write_se_handlers = true, bool write_lock_prefixes = true, bool save_to_pe_header = true, bool auto_strip_last_section = true);

template<typename PEClassType>
const image_directory rebuild_image_config_base(pe_base& pe, const image_config_info& info, section& image_config_section, uint32_t offset_from_section_start = 0, bool write_se_handlers = true, bool write_lock_prefixes = true, bool save_to_pe_header = true, bool auto_strip_last_section = true);
}
