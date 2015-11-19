/*************************************************************************/
/* Copyright (c) 2015 dx, http://kaimi.ru                                */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person           */
/* obtaining a copy of this software and associated documentation        */
/* files (the "Software"), to deal in the Software without               */
/* restriction, including without limitation the rights to use,          */
/* copy, modify, merge, publish, distribute, sublicense, and/or          */
/* sell copies of the Software, and to permit persons to whom the        */
/* Software is furnished to do so, subject to the following conditions:  */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#pragma once
#include <string>
#include <sstream>
#include "stdint_defs.h"
#if defined(_MSC_VER) or defined(__MINGW32__)
#define PE_BLISS_WINDOWS
#endif

namespace pe_bliss
{
//Enumeration of PE types
enum pe_type
{
	pe_type_32,
	pe_type_64
};

namespace pe_win
{
const uint32_t image_numberof_directory_entries = 16;
const uint32_t image_nt_optional_hdr32_magic = 0x10b;
const uint32_t image_nt_optional_hdr64_magic = 0x20b;
const uint32_t image_resource_name_is_string = 0x80000000;
const uint32_t image_resource_data_is_directory = 0x80000000;

const uint32_t image_dllcharacteristics_dynamic_base = 0x0040;     // DLL can move.
const uint32_t image_dllcharacteristics_force_integrity = 0x0080;     // Code Integrity Image
const uint32_t image_dllcharacteristics_nx_compat = 0x0100;     // Image is NX compatible
const uint32_t image_dllcharacteristics_no_isolation = 0x0200;     // Image understands isolation and doesn't want it
const uint32_t image_dllcharacteristics_no_seh = 0x0400;     // Image does not use SEH.  No SE handler may reside in this image
const uint32_t image_dllcharacteristics_no_bind = 0x0800;     // Do not bind this image.
const uint32_t image_dllcharacteristics_wdm_driver = 0x2000;     // Driver uses WDM model
const uint32_t image_dllcharacteristics_terminal_server_aware = 0x8000;

const uint32_t image_sizeof_file_header = 20;

const uint32_t image_file_relocs_stripped = 0x0001;  // Relocation info stripped from file.
const uint32_t image_file_executable_image = 0x0002;  // File is executable  (i.e. no unresolved externel references).
const uint32_t image_file_line_nums_stripped = 0x0004;  // Line nunbers stripped from file.
const uint32_t image_file_local_syms_stripped = 0x0008;  // Local symbols stripped from file.
const uint32_t image_file_aggresive_ws_trim = 0x0010;  // Agressively trim working set
const uint32_t image_file_large_address_aware = 0x0020;  // App can handle >2gb addresses
const uint32_t image_file_bytes_reversed_lo = 0x0080;  // Bytes of machine word are reversed.
const uint32_t image_file_32bit_machine = 0x0100;  // 32 bit word machine.
const uint32_t image_file_debug_stripped = 0x0200;  // Debugging info stripped from file in .DBG file
const uint32_t image_file_removable_run_from_swap = 0x0400;  // If Image is on removable media, copy and run from the swap file.
const uint32_t image_file_net_run_from_swap = 0x0800;  // If Image is on Net, copy and run from the swap file.
const uint32_t image_file_system = 0x1000;  // System File.
const uint32_t image_file_dll = 0x2000;  // File is a DLL.
const uint32_t image_file_up_system_only = 0x4000;  // File should only be run on a UP machine
const uint32_t image_file_bytes_reversed_hi = 0x8000;  // Bytes of machine word are reversed.

const uint32_t image_scn_lnk_nreloc_ovfl = 0x01000000;  // Section contains extended relocations.
const uint32_t image_scn_mem_discardable = 0x02000000;  // Section can be discarded.
const uint32_t image_scn_mem_not_cached = 0x04000000;  // Section is not cachable.
const uint32_t image_scn_mem_not_paged = 0x08000000;  // Section is not pageable.
const uint32_t image_scn_mem_shared = 0x10000000;  // Section is shareable.
const uint32_t image_scn_mem_execute = 0x20000000;  // Section is executable.
const uint32_t image_scn_mem_read = 0x40000000;  // Section is readable.
const uint32_t image_scn_mem_write = 0x80000000;  // Section is writeable.

const uint32_t image_scn_cnt_code = 0x00000020;  // Section contains code.
const uint32_t image_scn_cnt_initialized_data = 0x00000040;  // Section contains initialized data.
const uint32_t image_scn_cnt_uninitialized_data = 0x00000080;  // Section contains uninitialized data.

//Directory Entries
const uint32_t image_directory_entry_export = 0;   // Export Directory
const uint32_t image_directory_entry_import = 1;   // Import Directory
const uint32_t image_directory_entry_resource = 2;   // Resource Directory
const uint32_t image_directory_entry_exception = 3;   // Exception Directory
const uint32_t image_directory_entry_security = 4;   // Security Directory
const uint32_t image_directory_entry_basereloc = 5;   // Base Relocation Table
const uint32_t image_directory_entry_debug = 6;   // Debug Directory
const uint32_t image_directory_entry_architecture = 7;   // Architecture Specific Data
const uint32_t image_directory_entry_globalptr = 8;   // RVA of GP
const uint32_t image_directory_entry_tls = 9;   // TLS Directory
const uint32_t image_directory_entry_load_config = 10;   // Load Configuration Directory
const uint32_t image_directory_entry_bound_import = 11;   // Bound Import Directory in headers
const uint32_t image_directory_entry_iat = 12;   // Import Address Table
const uint32_t image_directory_entry_delay_import = 13;   // Delay Load Import Descriptors
const uint32_t image_directory_entry_com_descriptor = 14;   // COM Runtime descriptor

//Subsystem Values
const uint32_t image_subsystem_unknown = 0;   // Unknown subsystem.
const uint32_t image_subsystem_native = 1;   // Image doesn't require a subsystem.
const uint32_t image_subsystem_windows_gui = 2;   // Image runs in the Windows GUI subsystem.
const uint32_t image_subsystem_windows_cui = 3;   // Image runs in the Windows character subsystem.
const uint32_t image_subsystem_os2_cui = 5;   // image runs in the OS/2 character subsystem.
const uint32_t image_subsystem_posix_cui = 7;   // image runs in the Posix character subsystem.
const uint32_t image_subsystem_native_windows = 8;   // image is a native Win9x driver.
const uint32_t image_subsystem_windows_ce_gui = 9;   // Image runs in the Windows CE subsystem.
const uint32_t image_subsystem_efi_application = 10;  //
const uint32_t image_subsystem_efi_boot_service_driver = 11;   //
const uint32_t image_subsystem_efi_runtime_driver = 12;  //
const uint32_t image_subsystem_efi_rom = 13;
const uint32_t image_subsystem_xbox = 14;
const uint32_t image_subsystem_windows_boot_application = 16;

//Imports
const uint64_t image_ordinal_flag64 = 0x8000000000000000ull;
const uint32_t image_ordinal_flag32 = 0x80000000;

//Based relocation types
const uint32_t image_rel_based_absolute = 0;
const uint32_t image_rel_based_high =  1;
const uint32_t image_rel_based_low = 2;
const uint32_t image_rel_based_highlow = 3;
const uint32_t image_rel_based_highadj = 4;
const uint32_t image_rel_based_mips_jmpaddr = 5;
const uint32_t image_rel_based_mips_jmpaddr16 = 9;
const uint32_t image_rel_based_ia64_imm64 = 9;
const uint32_t image_rel_based_dir64 = 10;

//Exception directory
//The function has an exception handler that should be called when looking for functions that need to examine exceptions
const uint32_t unw_flag_ehandler = 0x01;
//The function has a termination handler that should be called when unwinding an exception
const uint32_t unw_flag_uhandler = 0x02;
//This unwind info structure is not the primary one for the procedure.
//Instead, the chained unwind info entry is the contents of a previous RUNTIME_FUNCTION entry.
//If this flag is set, then the UNW_FLAG_EHANDLER and UNW_FLAG_UHANDLER flags must be cleared.
//Also, the frame register and fixed-stack allocation fields must have the same values as in the primary unwind info
const uint32_t unw_flag_chaininfo = 0x04;

//Debug
const uint32_t image_debug_misc_exename = 1;
const uint32_t image_debug_type_unknown = 0;
const uint32_t image_debug_type_coff = 1;
const uint32_t image_debug_type_codeview = 2;
const uint32_t image_debug_type_fpo = 3;
const uint32_t image_debug_type_misc = 4;
const uint32_t image_debug_type_exception = 5;
const uint32_t image_debug_type_fixup = 6;
const uint32_t image_debug_type_omap_to_src = 7;
const uint32_t image_debug_type_omap_from_src = 8;
const uint32_t image_debug_type_borland = 9;
const uint32_t image_debug_type_reserved10 = 10;
const uint32_t image_debug_type_clsid = 11;


//Storage classes
const uint32_t image_sym_class_end_of_function = static_cast<uint8_t>(-1);
const uint32_t image_sym_class_null = 0x0000;
const uint32_t image_sym_class_automatic = 0x0001;
const uint32_t image_sym_class_external = 0x0002;
const uint32_t image_sym_class_static = 0x0003;
const uint32_t image_sym_class_register = 0x0004;
const uint32_t image_sym_class_external_def = 0x0005;
const uint32_t image_sym_class_label = 0x0006;
const uint32_t image_sym_class_undefined_label = 0x0007;
const uint32_t image_sym_class_member_of_struct = 0x0008;
const uint32_t image_sym_class_argument = 0x0009;
const uint32_t image_sym_class_struct_tag = 0x000a;
const uint32_t image_sym_class_member_of_union = 0x000b;
const uint32_t image_sym_class_union_tag = 0x000c;
const uint32_t image_sym_class_type_definition = 0x000d;
const uint32_t image_sym_class_undefined_static = 0x000e;
const uint32_t image_sym_class_enum_tag = 0x000f;
const uint32_t image_sym_class_member_of_enum = 0x0010;
const uint32_t image_sym_class_register_param = 0x0011;
const uint32_t image_sym_class_bit_field = 0x0012;

const uint32_t image_sym_class_far_external = 0x0044;

const uint32_t image_sym_class_block = 0x0064;
const uint32_t image_sym_class_function = 0x0065;
const uint32_t image_sym_class_end_of_struct = 0x0066;
const uint32_t image_sym_class_file = 0x0067;

const uint32_t image_sym_class_section = 0x0068;
const uint32_t image_sym_class_weak_external = 0x0069;

const uint32_t image_sym_class_clr_token = 0x006b;

//type packing constants
const uint32_t n_btmask = 0x000f;
const uint32_t n_tmask = 0x0030;
const uint32_t n_tmask1 = 0x00c0;
const uint32_t n_tmask2 = 0x00f0;
const uint32_t n_btshft = 4;
const uint32_t n_tshift = 2;

//Type (derived) values.
const uint32_t image_sym_dtype_null = 0;          // no derived type.
const uint32_t image_sym_dtype_pointer = 1;       // pointer.
const uint32_t image_sym_dtype_function = 2;      // function.
const uint32_t image_sym_dtype_array = 3;         // array.

// Is x a function?
//TODO
#ifndef ISFCN
#define ISFCN(x) (((x) & n_tmask) == (image_sym_dtype_function << n_btshft))
#endif

//Version info
const uint32_t vs_ffi_fileflagsmask = 0x0000003FL;

const uint32_t vs_ffi_signature = 0xFEEF04BDL;
const uint32_t vs_ffi_strucversion = 0x00010000L;

/* ----- VS_VERSION.dwFileFlags ----- */
const uint32_t vs_ff_debug = 0x00000001L;
const uint32_t vs_ff_prerelease = 0x00000002L;
const uint32_t vs_ff_patched = 0x00000004L;
const uint32_t vs_ff_privatebuild = 0x00000008L;
const uint32_t vs_ff_infoinferred = 0x00000010L;
const uint32_t vs_ff_specialbuild = 0x00000020L;

/* ----- VS_VERSION.dwFileOS ----- */
const uint32_t vos_unknown = 0x00000000L;
const uint32_t vos_dos = 0x00010000L;
const uint32_t vos_os216 = 0x00020000L;
const uint32_t vos_os232 = 0x00030000L;
const uint32_t vos_nt = 0x00040000L;
const uint32_t vos_wince = 0x00050000L;

const uint32_t vos__base = 0x00000000L;
const uint32_t vos__windows16 = 0x00000001L;
const uint32_t vos__pm16 = 0x00000002L;
const uint32_t vos__pm32 = 0x00000003L;
const uint32_t vos__windows32 = 0x00000004L;

const uint32_t vos_dos_windows16 = 0x00010001L;
const uint32_t vos_dos_windows32 = 0x00010004L;
const uint32_t vos_os216_pm16 = 0x00020002L;
const uint32_t vos_os232_pm32 = 0x00030003L;
const uint32_t vos_nt_windows32 = 0x00040004L;

/* ----- VS_VERSION.dwFileType ----- */
const uint32_t vft_unknown = 0x00000000L;
const uint32_t vft_app = 0x00000001L;
const uint32_t vft_dll = 0x00000002L;
const uint32_t vft_drv = 0x00000003L;
const uint32_t vft_font =  0x00000004L;
const uint32_t vft_vxd = 0x00000005L;
const uint32_t vft_static_lib = 0x00000007L;

const uint32_t message_resource_unicode = 0x0001;

#pragma pack(push, 1)

//Windows GUID structure
struct guid
{
	uint32_t Data1;
	uint16_t Data2;
	uint16_t Data3;
	uint8_t Data4[8];
};

//DOS .EXE header
struct image_dos_header
{
	uint16_t e_magic;                     // Magic number
	uint16_t e_cblp;                      // Bytes on last page of file
	uint16_t e_cp;                        // Pages in file
	uint16_t e_crlc;                      // Relocations
	uint16_t e_cparhdr;                   // Size of header in paragraphs
	uint16_t e_minalloc;                  // Minimum extra paragraphs needed
	uint16_t e_maxalloc;                  // Maximum extra paragraphs needed
	uint16_t e_ss;                        // Initial (relative) SS value
	uint16_t e_sp;                        // Initial SP value
	uint16_t e_csum;                      // Checksum
	uint16_t e_ip;                        // Initial IP value
	uint16_t e_cs;                        // Initial (relative) CS value
	uint16_t e_lfarlc;                    // File address of relocation table
	uint16_t e_ovno;                      // Overlay number
	uint16_t e_res[4];                    // Reserved words
	uint16_t e_oemid;                     // OEM identifier (for e_oeminfo)
	uint16_t e_oeminfo;                   // OEM information; e_oemid specific
	uint16_t e_res2[10];                  // Reserved words
	int32_t  e_lfanew;                    // File address of new exe header
};

//Directory format
struct image_data_directory
{
	uint32_t VirtualAddress;
	uint32_t Size;
};

//Optional header format
struct image_optional_header32
{
	//Standard fields
	uint16_t Magic;
	uint8_t  MajorLinkerVersion;
	uint8_t  MinorLinkerVersion;
	uint32_t SizeOfCode;
	uint32_t SizeOfInitializedData;
	uint32_t SizeOfUninitializedData;
	uint32_t AddressOfEntryPoint;
	uint32_t BaseOfCode;
	uint32_t BaseOfData;

	//NT additional fields
	uint32_t ImageBase;
	uint32_t SectionAlignment;
	uint32_t FileAlignment;
	uint16_t MajorOperatingSystemVersion;
	uint16_t MinorOperatingSystemVersion;
	uint16_t MajorImageVersion;
	uint16_t MinorImageVersion;
	uint16_t MajorSubsystemVersion;
	uint16_t MinorSubsystemVersion;
	uint32_t Win32VersionValue;
	uint32_t SizeOfImage;
	uint32_t SizeOfHeaders;
	uint32_t CheckSum;
	uint16_t Subsystem;
	uint16_t DllCharacteristics;
	uint32_t SizeOfStackReserve;
	uint32_t SizeOfStackCommit;
	uint32_t SizeOfHeapReserve;
	uint32_t SizeOfHeapCommit;
	uint32_t LoaderFlags;
	uint32_t NumberOfRvaAndSizes;
	image_data_directory DataDirectory[image_numberof_directory_entries];
};

struct image_optional_header64
{
	uint16_t Magic;
	uint8_t  MajorLinkerVersion;
	uint8_t  MinorLinkerVersion;
	uint32_t SizeOfCode;
	uint32_t SizeOfInitializedData;
	uint32_t SizeOfUninitializedData;
	uint32_t AddressOfEntryPoint;
	uint32_t BaseOfCode;
	uint64_t ImageBase;
	uint32_t SectionAlignment;
	uint32_t FileAlignment;
	uint16_t MajorOperatingSystemVersion;
	uint16_t MinorOperatingSystemVersion;
	uint16_t MajorImageVersion;
	uint16_t MinorImageVersion;
	uint16_t MajorSubsystemVersion;
	uint16_t MinorSubsystemVersion;
	uint32_t Win32VersionValue;
	uint32_t SizeOfImage;
	uint32_t SizeOfHeaders;
	uint32_t CheckSum;
	uint16_t Subsystem;
	uint16_t DllCharacteristics;
	uint64_t SizeOfStackReserve;
	uint64_t SizeOfStackCommit;
	uint64_t SizeOfHeapReserve;
	uint64_t SizeOfHeapCommit;
	uint32_t LoaderFlags;
	uint32_t NumberOfRvaAndSizes;
	image_data_directory DataDirectory[image_numberof_directory_entries];
};

struct image_file_header
{
	uint16_t Machine;
	uint16_t NumberOfSections;
	uint32_t TimeDateStamp;
	uint32_t PointerToSymbolTable;
	uint32_t NumberOfSymbols;
	uint16_t SizeOfOptionalHeader;
	uint16_t Characteristics;
};

struct image_nt_headers64
{
	uint32_t Signature;
	image_file_header FileHeader;
	image_optional_header64 OptionalHeader;
};

struct image_nt_headers32
{
	uint32_t Signature;
	image_file_header FileHeader;
	image_optional_header32 OptionalHeader;
};

//Section header format
struct image_section_header
{
	uint8_t Name[8];
	union
	{
		uint32_t PhysicalAddress;
		uint32_t VirtualSize;
	} Misc;

	uint32_t VirtualAddress;
	uint32_t SizeOfRawData;
	uint32_t PointerToRawData;
	uint32_t PointerToRelocations;
	uint32_t PointerToLinenumbers;
	uint16_t NumberOfRelocations;
	uint16_t NumberOfLinenumbers;
	uint32_t Characteristics;
};


/// RESOURCES ///
struct image_resource_directory
{
	uint32_t Characteristics;
	uint32_t TimeDateStamp;
	uint16_t MajorVersion;
	uint16_t MinorVersion;
	uint16_t NumberOfNamedEntries;
	uint16_t NumberOfIdEntries;
	//  IMAGE_RESOURCE_DIRECTORY_ENTRY DirectoryEntries[];
};

struct vs_fixedfileinfo
{
	uint32_t dwSignature;            /* e.g. 0xfeef04bd */
	uint32_t dwStrucVersion;         /* e.g. 0x00000042 = "0.42" */
	uint32_t dwFileVersionMS;        /* e.g. 0x00030075 = "3.75" */
	uint32_t dwFileVersionLS;        /* e.g. 0x00000031 = "0.31" */
	uint32_t dwProductVersionMS;     /* e.g. 0x00030010 = "3.10" */
	uint32_t dwProductVersionLS;     /* e.g. 0x00000031 = "0.31" */
	uint32_t dwFileFlagsMask;        /* = 0x3F for version "0.42" */
	uint32_t dwFileFlags;            /* e.g. VFF_DEBUG | VFF_PRERELEASE */
	uint32_t dwFileOS;               /* e.g. VOS_DOS_WINDOWS16 */
	uint32_t dwFileType;             /* e.g. VFT_DRIVER */
	uint32_t dwFileSubtype;          /* e.g. VFT2_DRV_KEYBOARD */
	uint32_t dwFileDateMS;           /* e.g. 0 */
	uint32_t dwFileDateLS;           /* e.g. 0 */
};

struct bitmapinfoheader
{
	uint32_t biSize;
	int32_t  biWidth;
	int32_t  biHeight;
	uint16_t biPlanes;
	uint16_t biBitCount;
	uint32_t biCompression;
	uint32_t biSizeImage;
	int32_t  biXPelsPerMeter;
	int32_t  biYPelsPerMeter;
	uint32_t biClrUsed;
	uint32_t biClrImportant;
};

struct message_resource_entry
{
	uint16_t Length;
	uint16_t Flags;
	uint8_t  Text[1];
};

struct message_resource_block
{
	uint32_t LowId;
	uint32_t HighId;
	uint32_t OffsetToEntries;
};

struct message_resource_data
{
	uint32_t NumberOfBlocks;
	message_resource_block Blocks[1];
};

struct image_resource_directory_entry
{
	union
	{
		struct
		{
			uint32_t NameOffset:31;
			uint32_t NameIsString:1;
		};
		uint32_t Name;
		uint16_t Id;
	};

	union
	{
		uint32_t OffsetToData;
		struct
		{
			uint32_t OffsetToDirectory:31;
			uint32_t DataIsDirectory:1;
		};
	};
};

struct image_resource_data_entry
{
	uint32_t OffsetToData;
	uint32_t Size;
	uint32_t CodePage;
	uint32_t Reserved;
};

#pragma pack(push, 2)
struct bitmapfileheader
{
	uint16_t bfType;
	uint32_t bfSize;
	uint16_t bfReserved1;
	uint16_t bfReserved2;
	uint32_t bfOffBits;
};
#pragma pack(pop)



//Structure representing ICON file header
struct ico_header
{
	uint16_t Reserved;
	uint16_t Type; //1
	uint16_t Count; //Count of icons included in icon group
};

//Structure that is stored in icon group directory in PE resources
struct icon_group
{
	uint8_t Width;
	uint8_t Height;
	uint8_t ColorCount;
	uint8_t Reserved;
	uint16_t Planes;
	uint16_t BitCount;
	uint32_t SizeInBytes;
	uint16_t Number; //Represents resource ID in PE icon list
};

//Structure representing ICON directory entry inside ICON file
struct icondirentry
{
	uint8_t Width;
	uint8_t Height;
	uint8_t ColorCount;
	uint8_t Reserved;
	uint16_t Planes;
	uint16_t BitCount;
	uint32_t SizeInBytes;
	uint32_t ImageOffset; //Offset from start of header to the image
};

//Structure representing CURSOR file header
struct cursor_header
{
	uint16_t Reserved;
	uint16_t Type; //2
	uint16_t Count; //Count of cursors included in cursor group
};

struct cursor_group
{
	uint16_t Width;
	uint16_t Height; //Divide by 2 to get the actual height.
	uint16_t Planes;
	uint16_t BitCount;
	uint32_t SizeInBytes;
	uint16_t Number; //Represents resource ID in PE icon list
};

//Structure representing CURSOR directory entry inside CURSOR file
struct cursordirentry
{
	uint8_t Width; //Set to CURSOR_GROUP::Height/2.
	uint8_t Height;
	uint8_t ColorCount;
	uint8_t Reserved;
	uint16_t HotspotX;
	uint16_t HotspotY;
	uint32_t SizeInBytes;
	uint32_t ImageOffset; //Offset from start of header to the image
};

//Structure representing BLOCK in version info resource
struct version_info_block //(always aligned on 32-bit (DWORD) boundary)
{
	uint16_t Length; //Length of this block (doesn't include padding)
	uint16_t ValueLength; //Value length (if any)
	uint16_t Type; //Value type (0 = binary, 1 = text)
	uint16_t Key[1]; //Value name (block key) (always NULL terminated)

	//////////
	//WORD padding1[]; //Padding, if any (ALIGNMENT)
	//xxxxx Value[]; //Value data, if any (*ALIGNED*)
	//WORD padding2[]; //Padding, if any (ALIGNMENT)
	//xxxxx Child[]; //Child block(s), if any (*ALIGNED*)
	//////////
};


/// IMPORTS ///
#pragma pack(push, 8)
struct image_thunk_data64
{
	union
	{
		uint64_t ForwarderString;  // PBYTE 
		uint64_t Function;         // PDWORD
		uint64_t Ordinal;
		uint64_t AddressOfData;    // PIMAGE_IMPORT_BY_NAME
	} u1;
};
#pragma pack(pop)

struct image_thunk_data32
{
	union
	{
		uint32_t ForwarderString;      // PBYTE 
		uint32_t Function;             // PDWORD
		uint32_t Ordinal;
		uint32_t AddressOfData;        // PIMAGE_IMPORT_BY_NAME
	} u1;
};

struct image_import_descriptor
{
	union
	{
		uint32_t Characteristics;           // 0 for terminating null import descriptor
		uint32_t OriginalFirstThunk;        // RVA to original unbound IAT (PIMAGE_THUNK_DATA)
	};

	uint32_t TimeDateStamp;                 // 0 if not bound,
											// -1 if bound, and real date\time stamp
											//     in IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT (new BIND)
											// O.W. date/time stamp of DLL bound to (Old BIND)

	uint32_t ForwarderChain;                // -1 if no forwarders
	uint32_t Name;
	uint32_t FirstThunk;                    // RVA to IAT (if bound this IAT has actual addresses)
};


/// TLS ///
struct image_tls_directory64
{
	uint64_t StartAddressOfRawData;
	uint64_t EndAddressOfRawData;
	uint64_t AddressOfIndex;         // PDWORD
	uint64_t AddressOfCallBacks;     // PIMAGE_TLS_CALLBACK *;
	uint32_t SizeOfZeroFill;
	uint32_t Characteristics;
};

struct image_tls_directory32
{
	uint32_t StartAddressOfRawData;
	uint32_t EndAddressOfRawData;
	uint32_t AddressOfIndex;             // PDWORD
	uint32_t AddressOfCallBacks;         // PIMAGE_TLS_CALLBACK *
	uint32_t SizeOfZeroFill;
	uint32_t Characteristics;
};


/// Export Format ///
struct image_export_directory
{
	uint32_t Characteristics;
	uint32_t TimeDateStamp;
	uint16_t MajorVersion;
	uint16_t MinorVersion;
	uint32_t Name;
	uint32_t Base;
	uint32_t NumberOfFunctions;
	uint32_t NumberOfNames;
	uint32_t AddressOfFunctions;     // RVA from base of image
	uint32_t AddressOfNames;         // RVA from base of image
	uint32_t AddressOfNameOrdinals;  // RVA from base of image
};


/// Based relocation format ///
struct image_base_relocation
{
	uint32_t VirtualAddress;
	uint32_t SizeOfBlock;
	//  uint16_t TypeOffset[1];
};


/// New format import descriptors pointed to by DataDirectory[ IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT ] ///
struct image_bound_import_descriptor
{
	uint32_t TimeDateStamp;
	uint16_t OffsetModuleName;
	uint16_t NumberOfModuleForwarderRefs;
	// Array of zero or more IMAGE_BOUND_FORWARDER_REF follows
};

struct image_bound_forwarder_ref
{
	uint32_t TimeDateStamp;
	uint16_t OffsetModuleName;
	uint16_t Reserved;
};


/// Exception directory ///
struct image_runtime_function_entry
{
	uint32_t BeginAddress;
	uint32_t EndAddress;
	uint32_t UnwindInfoAddress;
};

enum unwind_op_codes
{
	uwop_push_nonvol = 0, /* info == register number */
	uwop_alloc_large,     /* no info, alloc size in next 2 slots */
	uwop_alloc_small,     /* info == size of allocation / 8 - 1 */
	uwop_set_fpreg,       /* no info, FP = RSP + UNWIND_INFO.FPRegOffset*16 */
	uwop_save_nonvol,     /* info == register number, offset in next slot */
	uwop_save_nonvol_far, /* info == register number, offset in next 2 slots */
	uwop_save_xmm128,     /* info == XMM reg number, offset in next slot */
	uwop_save_xmm128_far, /* info == XMM reg number, offset in next 2 slots */
	uwop_push_machframe   /* info == 0: no error-code, 1: error-code */
};

union unwind_code
{
	struct s
	{
		uint8_t CodeOffset;
		uint8_t UnwindOp : 4;
		uint8_t OpInfo   : 4;
	};

	uint16_t FrameOffset;
};

struct unwind_info
{
	uint8_t Version       : 3;
	uint8_t Flags         : 5;
	uint8_t SizeOfProlog;
	uint8_t CountOfCodes;
	uint8_t FrameRegister : 4;
	uint8_t FrameOffset   : 4;
	unwind_code UnwindCode[1];
	/*  unwind_code MoreUnwindCode[((CountOfCodes + 1) & ~1) - 1];
	*   union {
	*       OPTIONAL ULONG ExceptionHandler;
	*       OPTIONAL ULONG FunctionEntry;
	*   };
	*   OPTIONAL ULONG ExceptionData[]; */
};



/// Debug ///
struct image_debug_misc
{
	uint32_t DataType;               // type of misc data, see defines
	uint32_t Length;                 // total length of record, rounded to four
	// byte multiple.
	uint8_t  Unicode;                // TRUE if data is unicode string
	uint8_t  Reserved[3];
	uint8_t  Data[1];                // Actual data
};

struct image_coff_symbols_header
{
	uint32_t NumberOfSymbols;
	uint32_t LvaToFirstSymbol;
	uint32_t NumberOfLinenumbers;
	uint32_t LvaToFirstLinenumber;
	uint32_t RvaToFirstByteOfCode;
	uint32_t RvaToLastByteOfCode;
	uint32_t RvaToFirstByteOfData;
	uint32_t RvaToLastByteOfData;
};

struct image_debug_directory
{
	uint32_t Characteristics;
	uint32_t TimeDateStamp;
	uint16_t MajorVersion;
	uint16_t MinorVersion;
	uint32_t Type;
	uint32_t SizeOfData;
	uint32_t AddressOfRawData;
	uint32_t PointerToRawData;
};


#pragma pack(push, 2)
struct image_symbol
{
	union
	{
		uint8_t ShortName[8];
		struct
		{
			uint32_t Short;     // if 0, use LongName
			uint32_t Long;      // offset into string table
		} Name;
		uint32_t LongName[2];    // PBYTE [2]
	} N;
	uint32_t Value;
	int16_t  SectionNumber;
	uint16_t Type;
	uint8_t  StorageClass;
	uint8_t  NumberOfAuxSymbols;
};
#pragma pack(pop)

//CodeView Debug OMF signature. The signature at the end of the file is
//a negative offset from the end of the file to another signature.  At
//the negative offset (base address) is another signature whose filepos
//field points to the first OMFDirHeader in a chain of directories.
//The NB05 signature is used by the link utility to indicated a completely
//unpacked file. The NB06 signature is used by ilink to indicate that the
//executable has had CodeView information from an incremental link appended
//to the executable. The NB08 signature is used by cvpack to indicate that
//the CodeView Debug OMF has been packed. CodeView will only process
//executables with the NB08 signature.
struct OMFSignature
{
	char Signature[4];   // "NBxx"
	uint32_t filepos;    // offset in file
};

struct CV_INFO_PDB20
{
	OMFSignature CvHeader;
	uint32_t Signature;
	uint32_t Age;
	uint8_t PdbFileName[1];
};

struct CV_INFO_PDB70
{
	uint32_t CvSignature;
	guid Signature;
	uint32_t Age;
	uint8_t PdbFileName[1];
};

//  directory information structure
//  This structure contains the information describing the directory.
//  It is pointed to by the signature at the base address or the directory
//  link field of a preceeding directory.  The directory entries immediately
//  follow this structure.
struct OMFDirHeader
{
	uint16_t cbDirHeader;    // length of this structure
	uint16_t cbDirEntry;     // number of bytes in each directory entry
	uint32_t cDir;           // number of directorie entries
	int32_t  lfoNextDir;     // offset from base of next directory
	uint32_t flags;          // status flags
};

//  directory structure
//  The data in this structure is used to reference the data for each
//  subsection of the CodeView Debug OMF information.  Tables that are
//  not associated with a specific module will have a module index of
//  oxffff.  These tables are the global types table, the global symbol
//  table, the global public table and the library table.
struct OMFDirEntry
{
	uint16_t SubSection;     // subsection type (sst...)
	uint16_t iMod;           // module index
	int32_t  lfo;            // large file offset of subsection
	uint32_t cb;             // number of bytes in subsection
};


/// CLR 2.0 header structure ///
struct image_cor20_header
{
	//Header versioning
	uint32_t cb;
	uint16_t MajorRuntimeVersion;
	uint16_t MinorRuntimeVersion;

	// Symbol table and startup information
	image_data_directory MetaData;
	uint32_t Flags;

	// If COMIMAGE_FLAGS_NATIVE_ENTRYPOINT is not set, EntryPointToken represents a managed entrypoint.
	// If COMIMAGE_FLAGS_NATIVE_ENTRYPOINT is set, EntryPointRVA represents an RVA to a native entrypoint.
	union
	{
		uint32_t EntryPointToken;
		uint32_t EntryPointRVA;
	};

	// Binding information
	image_data_directory Resources;
	image_data_directory StrongNameSignature;

	// Regular fixup and binding information
	image_data_directory CodeManagerTable;
	image_data_directory VTableFixups;
	image_data_directory ExportAddressTableJumps;

	// Precompiled image info (internal use only - set to zero)
	image_data_directory ManagedNativeHeader;
};

enum replaces_cor_hdr_numeric_defines
{
	// COM+ Header entry point flags.
	comimage_flags_ilonly               =0x00000001,
	comimage_flags_32bitrequired        =0x00000002,
	comimage_flags_il_library           =0x00000004,
	comimage_flags_strongnamesigned     =0x00000008,
	comimage_flags_native_entrypoint    =0x00000010,
	comimage_flags_trackdebugdata       =0x00010000,

	// Version flags for image.
	cor_version_major_v2                =2,
	cor_version_major                   =cor_version_major_v2,
	cor_version_minor                   =0,
	cor_deleted_name_length             =8,
	cor_vtablegap_name_length           =8,

	// Maximum size of a NativeType descriptor.
	native_type_max_cb                  =1,
	cor_ilmethod_sect_small_max_datasize=0xff,

	// #defines for the MIH FLAGS
	image_cor_mih_methodrva             =0x01,
	image_cor_mih_ehrva                 =0x02,
	image_cor_mih_basicblock            =0x08,

	// V-table constants
	cor_vtable_32bit                    =0x01,          // V-table slots are 32-bits in size.
	cor_vtable_64bit                    =0x02,          // V-table slots are 64-bits in size.
	cor_vtable_from_unmanaged           =0x04,          // If set, transition from unmanaged.
	cor_vtable_from_unmanaged_retain_appdomain  =0x08,  // If set, transition from unmanaged with keeping the current appdomain.
	cor_vtable_call_most_derived        =0x10,          // Call most derived method described by

	// EATJ constants
	image_cor_eatj_thunk_size           =32,            // Size of a jump thunk reserved range.

	// Max name lengths
	//@todo: Change to unlimited name lengths.
	max_class_name                      =1024,
	max_package_name                    =1024
};

/// Load Configuration Directory Entry ///
struct image_load_config_directory32
{
	uint32_t Size;
	uint32_t TimeDateStamp;
	uint16_t MajorVersion;
	uint16_t MinorVersion;
	uint32_t GlobalFlagsClear;
	uint32_t GlobalFlagsSet;
	uint32_t CriticalSectionDefaultTimeout;
	uint32_t DeCommitFreeBlockThreshold;
	uint32_t DeCommitTotalFreeThreshold;
	uint32_t LockPrefixTable;            // VA
	uint32_t MaximumAllocationSize;
	uint32_t VirtualMemoryThreshold;
	uint32_t ProcessHeapFlags;
	uint32_t ProcessAffinityMask;
	uint16_t CSDVersion;
	uint16_t Reserved1;
	uint32_t EditList;                   // VA
	uint32_t SecurityCookie;             // VA
	uint32_t SEHandlerTable;             // VA
	uint32_t SEHandlerCount;
};

struct image_load_config_directory64
{
	uint32_t Size;
	uint32_t TimeDateStamp;
	uint16_t MajorVersion;
	uint16_t MinorVersion;
	uint32_t GlobalFlagsClear;
	uint32_t GlobalFlagsSet;
	uint32_t CriticalSectionDefaultTimeout;
	uint64_t DeCommitFreeBlockThreshold;
	uint64_t DeCommitTotalFreeThreshold;
	uint64_t LockPrefixTable;         // VA
	uint64_t MaximumAllocationSize;
	uint64_t VirtualMemoryThreshold;
	uint64_t ProcessAffinityMask;
	uint32_t ProcessHeapFlags;
	uint16_t CSDVersion;
	uint16_t Reserved1;
	uint64_t EditList;                // VA
	uint64_t SecurityCookie;          // VA
	uint64_t SEHandlerTable;          // VA
	uint64_t SEHandlerCount;
};

#pragma pack(pop)
} //namespace pe_win

#ifdef PE_BLISS_WINDOWS
typedef wchar_t unicode16_t;
typedef std::basic_string<unicode16_t> u16string;
#else
//Instead of wchar_t for windows
typedef unsigned short unicode16_t;
typedef std::basic_string<unicode16_t> u16string;
#endif

} //namespace pe_bliss
