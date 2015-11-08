#include <string.h>
#include "pe_dotnet.h"

namespace pe_bliss
{
using namespace pe_win;

//.NET
basic_dotnet_info::basic_dotnet_info()
{
	memset(&header_, 0, sizeof(header_));
}

//Constructor from data
basic_dotnet_info::basic_dotnet_info(const image_cor20_header& header)
	:header_(header)
{}

//Returns major runtime version
uint16_t basic_dotnet_info::get_major_runtime_version() const
{
	return header_.MajorRuntimeVersion;
}

//Returns minor runtime version
uint16_t basic_dotnet_info::get_minor_runtime_version() const
{
	return header_.MinorRuntimeVersion;
}

//Returns RVA of metadata (symbol table and startup information)
uint32_t basic_dotnet_info::get_rva_of_metadata() const
{
	return header_.MetaData.VirtualAddress;
}

//Returns size of metadata (symbol table and startup information)
uint32_t basic_dotnet_info::get_size_of_metadata() const
{
	return header_.MetaData.Size;
}

//Returns flags
uint32_t basic_dotnet_info::get_flags() const
{
	return header_.Flags;
}

//Returns true if entry point is native
bool basic_dotnet_info::is_native_entry_point() const
{
	return (header_.Flags & comimage_flags_native_entrypoint) ? true : false;
}

//Returns true if 32 bit required
bool basic_dotnet_info::is_32bit_required() const
{
	return (header_.Flags & comimage_flags_32bitrequired) ? true : false;
}

//Returns true if image is IL library
bool basic_dotnet_info::is_il_library() const
{
	return (header_.Flags & comimage_flags_il_library) ? true : false;
}

//Returns true if image uses IL only
bool basic_dotnet_info::is_il_only() const
{
	return (header_.Flags & comimage_flags_ilonly) ? true : false;
}

//Returns entry point RVA (if entry point is native)
//Returns entry point managed token (if entry point is managed)
uint32_t basic_dotnet_info::get_entry_point_rva_or_token() const
{
	return header_.EntryPointToken;
}

//Returns RVA of managed resources
uint32_t basic_dotnet_info::get_rva_of_resources() const
{
	return header_.Resources.VirtualAddress;
}

//Returns size of managed resources
uint32_t basic_dotnet_info::get_size_of_resources() const
{
	return header_.Resources.Size;
}

//Returns RVA of strong name signature
uint32_t basic_dotnet_info::get_rva_of_strong_name_signature() const
{
	return header_.StrongNameSignature.VirtualAddress;
}

//Returns size of strong name signature
uint32_t basic_dotnet_info::get_size_of_strong_name_signature() const
{
	return header_.StrongNameSignature.Size;
}

//Returns RVA of code manager table
uint32_t basic_dotnet_info::get_rva_of_code_manager_table() const
{
	return header_.CodeManagerTable.VirtualAddress;
}

//Returns size of code manager table
uint32_t basic_dotnet_info::get_size_of_code_manager_table() const
{
	return header_.CodeManagerTable.Size;
}

//Returns RVA of VTable fixups
uint32_t basic_dotnet_info::get_rva_of_vtable_fixups() const
{
	return header_.VTableFixups.VirtualAddress;
}

//Returns size of VTable fixups
uint32_t basic_dotnet_info::get_size_of_vtable_fixups() const
{
	return header_.VTableFixups.Size;
}

//Returns RVA of export address table jumps
uint32_t basic_dotnet_info::get_rva_of_export_address_table_jumps() const
{
	return header_.ExportAddressTableJumps.VirtualAddress;
}

//Returns size of export address table jumps
uint32_t basic_dotnet_info::get_size_of_export_address_table_jumps() const
{
	return header_.ExportAddressTableJumps.Size;
}

//Returns RVA of managed native header
//(precompiled header info, usually set to zero, for internal use)
uint32_t basic_dotnet_info::get_rva_of_managed_native_header() const
{
	return header_.ManagedNativeHeader.VirtualAddress;
}

//Returns size of managed native header
//(precompiled header info, usually set to zero, for internal use)
uint32_t basic_dotnet_info::get_size_of_managed_native_header() const
{
	return header_.ManagedNativeHeader.Size;
}

//Returns basic .NET information
//If image is not native, throws an exception
const basic_dotnet_info get_basic_dotnet_info(const pe_base& pe)
{
	//If there's no debug directory, return empty list
	if(!pe.is_dotnet())
		throw pe_exception("Image does not have managed code", pe_exception::image_does_not_have_managed_code);

	//Return basic .NET information
	return basic_dotnet_info(pe.section_data_from_rva<image_cor20_header>(pe.get_directory_rva(image_directory_entry_com_descriptor), section_data_virtual, true));
}
}
