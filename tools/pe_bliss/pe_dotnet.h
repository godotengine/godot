#pragma once
#include "pe_structures.h"
#include "pe_base.h"

namespace pe_bliss
{
//Class representing basic .NET header information
class basic_dotnet_info
{
public:
	//Default constructor
	basic_dotnet_info();
	//Constructor from data
	explicit basic_dotnet_info(const pe_win::image_cor20_header& header);

	//Returns major runtime version
	uint16_t get_major_runtime_version() const;
	//Returns minor runtime version
	uint16_t get_minor_runtime_version() const;

	//Returns RVA of metadata (symbol table and startup information)
	uint32_t get_rva_of_metadata() const;
	//Returns size of metadata (symbol table and startup information)
	uint32_t get_size_of_metadata() const;

	//Returns flags
	uint32_t get_flags() const;

	//Returns true if entry point is native
	bool is_native_entry_point() const;
	//Returns true if 32 bit required
	bool is_32bit_required() const;
	//Returns true if image is IL library
	bool is_il_library() const;
	//Returns true if image uses IL only
	bool is_il_only() const;

	//Returns entry point RVA (if entry point is native)
	//Returns entry point managed token (if entry point is managed)
	uint32_t get_entry_point_rva_or_token() const;

	//Returns RVA of managed resources
	uint32_t get_rva_of_resources() const;
	//Returns size of managed resources
	uint32_t get_size_of_resources() const;
	//Returns RVA of strong name signature
	uint32_t get_rva_of_strong_name_signature() const;
	//Returns size of strong name signature
	uint32_t get_size_of_strong_name_signature() const;
	//Returns RVA of code manager table
	uint32_t get_rva_of_code_manager_table() const;
	//Returns size of code manager table
	uint32_t get_size_of_code_manager_table() const;
	//Returns RVA of VTable fixups
	uint32_t get_rva_of_vtable_fixups() const;
	//Returns size of VTable fixups
	uint32_t get_size_of_vtable_fixups() const;
	//Returns RVA of export address table jumps
	uint32_t get_rva_of_export_address_table_jumps() const;
	//Returns size of export address table jumps
	uint32_t get_size_of_export_address_table_jumps() const;
	//Returns RVA of managed native header
	//(precompiled header info, usually set to zero, for internal use)
	uint32_t get_rva_of_managed_native_header() const;
	//Returns size of managed native header
	//(precompiled header info, usually set to zero, for internal use)
	uint32_t get_size_of_managed_native_header() const;

private:
	pe_win::image_cor20_header header_;
};

//Returns basic .NET information
//If image is not native, throws an exception
const basic_dotnet_info get_basic_dotnet_info(const pe_base& pe);
}
