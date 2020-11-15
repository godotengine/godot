// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_COMPAT_NON_WIN_WINNT_H_
#define CRASHPAD_COMPAT_NON_WIN_WINNT_H_

#include <stdint.h>

//! \file

//! \anchor VER_SUITE_x
//! \name VER_SUITE_*
//!
//! \brief Installable product values for MINIDUMP_SYSTEM_INFO::SuiteMask.
//! \{
#define VER_SUITE_SMALLBUSINESS 0x0001
#define VER_SUITE_ENTERPRISE 0x0002
#define VER_SUITE_BACKOFFICE 0x0004
#define VER_SUITE_COMMUNICATIONS 0x0008
#define VER_SUITE_TERMINAL 0x0010
#define VER_SUITE_SMALLBUSINESS_RESTRICTED 0x0020
#define VER_SUITE_EMBEDDEDNT 0x0040
#define VER_SUITE_DATACENTER 0x0080
#define VER_SUITE_SINGLEUSERTS 0x0100
#define VER_SUITE_PERSONAL 0x0200
#define VER_SUITE_BLADE 0x0400
#define VER_SUITE_EMBEDDED_RESTRICTED 0x0800
#define VER_SUITE_SECURITY_APPLIANCE 0x1000
#define VER_SUITE_STORAGE_SERVER 0x2000
#define VER_SUITE_COMPUTE_SERVER 0x4000
#define VER_SUITE_WH_SERVER 0x8000
//! \}

//! \brief The maximum number of exception parameters present in the
//!     MINIDUMP_EXCEPTION::ExceptionInformation array.
#define EXCEPTION_MAXIMUM_PARAMETERS 15

//! \anchor PROCESSOR_ARCHITECTURE_x
//! \name PROCESSOR_ARCHITECTURE_*
//!
//! \brief CPU type values for MINIDUMP_SYSTEM_INFO::ProcessorArchitecture.
//!
//! \sa crashpad::MinidumpCPUArchitecture
//! \{
#define PROCESSOR_ARCHITECTURE_INTEL 0
#define PROCESSOR_ARCHITECTURE_MIPS 1
#define PROCESSOR_ARCHITECTURE_ALPHA 2
#define PROCESSOR_ARCHITECTURE_PPC 3
#define PROCESSOR_ARCHITECTURE_SHX 4
#define PROCESSOR_ARCHITECTURE_ARM 5
#define PROCESSOR_ARCHITECTURE_IA64 6
#define PROCESSOR_ARCHITECTURE_ALPHA64 7
#define PROCESSOR_ARCHITECTURE_MSIL 8
#define PROCESSOR_ARCHITECTURE_AMD64 9
#define PROCESSOR_ARCHITECTURE_IA32_ON_WIN64 10
#define PROCESSOR_ARCHITECTURE_NEUTRAL 11
#define PROCESSOR_ARCHITECTURE_ARM64 12
#define PROCESSOR_ARCHITECTURE_ARM32_ON_WIN64 13
#define PROCESSOR_ARCHITECTURE_UNKNOWN 0xffff
//! \}

//! \anchor PF_x
//! \name PF_*
//!
//! \brief CPU feature values for \ref CPU_INFORMATION::ProcessorFeatures
//!     "CPU_INFORMATION::OtherCpuInfo::ProcessorFeatures".
//!
//! \{
#define PF_FLOATING_POINT_PRECISION_ERRATA 0
#define PF_FLOATING_POINT_EMULATED 1
#define PF_COMPARE_EXCHANGE_DOUBLE 2
#define PF_MMX_INSTRUCTIONS_AVAILABLE 3
#define PF_PPC_MOVEMEM_64BIT_OK 4
#define PF_ALPHA_BYTE_INSTRUCTIONS 5
#define PF_XMMI_INSTRUCTIONS_AVAILABLE 6
#define PF_3DNOW_INSTRUCTIONS_AVAILABLE 7
#define PF_RDTSC_INSTRUCTION_AVAILABLE 8
#define PF_PAE_ENABLED 9
#define PF_XMMI64_INSTRUCTIONS_AVAILABLE 10
#define PF_SSE_DAZ_MODE_AVAILABLE 11
#define PF_NX_ENABLED 12
#define PF_SSE3_INSTRUCTIONS_AVAILABLE 13
#define PF_COMPARE_EXCHANGE128 14
#define PF_COMPARE64_EXCHANGE128 15
#define PF_CHANNELS_ENABLED 16
#define PF_XSAVE_ENABLED 17
#define PF_ARM_VFP_32_REGISTERS_AVAILABLE 18
#define PF_ARM_NEON_INSTRUCTIONS_AVAILABLE 19
#define PF_SECOND_LEVEL_ADDRESS_TRANSLATION 20
#define PF_VIRT_FIRMWARE_ENABLED 21
#define PF_RDWRFSGSBASE_AVAILABLE 22
#define PF_FASTFAIL_AVAILABLE 23
#define PF_ARM_DIVIDE_INSTRUCTION_AVAILABLE 24
#define PF_ARM_64BIT_LOADSTORE_ATOMIC 25
#define PF_ARM_EXTERNAL_CACHE_AVAILABLE 26
#define PF_ARM_FMAC_INSTRUCTIONS_AVAILABLE 27
#define PF_RDRAND_INSTRUCTION_AVAILABLE 28
#define PF_ARM_V8_INSTRUCTIONS_AVAILABLE 29
#define PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE 30
#define PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE 31
#define PF_RDTSCP_INSTRUCTION_AVAILABLE 32
//! \}

//! \anchor PAGE_x
//! \name PAGE_*
//!
//! \brief Memory protection constants for MINIDUMP_MEMORY_INFO::Protect and
//!     MINIDUMP_MEMORY_INFO::AllocationProtect.
//! \{
#define PAGE_NOACCESS 0x1
#define PAGE_READONLY 0x2
#define PAGE_READWRITE 0x4
#define PAGE_WRITECOPY 0x8
#define PAGE_EXECUTE 0x10
#define PAGE_EXECUTE_READ 0x20
#define PAGE_EXECUTE_READWRITE 0x40
#define PAGE_EXECUTE_WRITECOPY 0x80
#define PAGE_GUARD 0x100
#define PAGE_NOCACHE 0x200
#define PAGE_WRITECOMBINE 0x400
//! \}

//! \anchor MEM_x
//! \name MEM_*
//!
//! \brief Memory state and type constants for MINIDUMP_MEMORY_INFO::State and
//!     MINIDUMP_MEMORY_INFO::Type.
//! \{
#define MEM_COMMIT 0x1000
#define MEM_RESERVE 0x2000
#define MEM_DECOMMIT 0x4000
#define MEM_RELEASE 0x8000
#define MEM_FREE 0x10000
#define MEM_PRIVATE 0x20000
#define MEM_MAPPED 0x40000
#define MEM_RESET 0x80000
//! \}

//! \brief The maximum number of distinct identifiable features that could
//!     possibly be carried in an XSAVE area.
//!
//! This corresponds to the number of bits in the XSAVE state-component bitmap,
//! XSAVE_BV. See Intel Software Developer’s Manual, Volume 1: Basic
//! Architecture (253665-060), 13.4.2 “XSAVE Header”.
#define MAXIMUM_XSTATE_FEATURES (64)

//! \brief The location of a single state component within an XSAVE area.
struct XSTATE_FEATURE {
  //! \brief The location of a state component within a CPU-specific context
  //!     structure.
  //!
  //! This is equivalent to the difference (`ptrdiff_t`) between the return
  //! value of `LocateXStateFeature()` and its \a Context argument.
  uint32_t Offset;

  //! \brief The size of a state component with a CPU-specific context
  //!     structure.
  //!
  //! This is equivalent to the size returned by `LocateXStateFeature()` in \a
  //!     Length.
  uint32_t Size;
};

//! \anchor IMAGE_DEBUG_MISC_x
//! \name IMAGE_DEBUG_MISC_*
//!
//! Data type values for IMAGE_DEBUG_MISC::DataType.
//! \{

//! \brief A pointer to a `.dbg` file.
//!
//! IMAGE_DEBUG_MISC::Data will contain the path or file name of the `.dbg` file
//! associated with the module.
#define IMAGE_DEBUG_MISC_EXENAME 1

//! \}

//! \brief Miscellaneous debugging record.
//!
//! This structure is referenced by MINIDUMP_MODULE::MiscRecord. It is obsolete,
//! superseded by the CodeView record.
struct IMAGE_DEBUG_MISC {
  //! \brief The type of data carried in the #Data field.
  //!
  //! This is a value of \ref IMAGE_DEBUG_MISC_x "IMAGE_DEBUG_MISC_*".
  uint32_t DataType;

  //! \brief The length of this structure in bytes, including the entire #Data
  //!     field and its `NUL` terminator.
  //!
  //! \note The Windows documentation states that this field is rounded up to
  //!     nearest nearest 4-byte multiple.
  uint32_t Length;

  //! \brief The encoding of the #Data field.
  //!
  //! If this field is `0`, #Data contains narrow or multibyte character data.
  //! If this field is `1`, #Data is UTF-16-encoded.
  //!
  //! On Windows, with this field set to `0`, #Data will be encoded in the code
  //! page of the system that linked the module. On other operating systems,
  //! UTF-8 may be used.
  uint8_t Unicode;

  uint8_t Reserved[3];

  //! \brief The data carried within this structure.
  //!
  //! For string data, this field will be `NUL`-terminated. If #Unicode is `1`,
  //! this field is UTF-16-encoded, and will be terminated by a UTF-16 `NUL`
  //! code unit (two `NUL` bytes).
  uint8_t Data[1];
};

//! \anchor VER_NT_x
//! \name VER_NT_*
//!
//! \brief Operating system type values for MINIDUMP_SYSTEM_INFO::ProductType.
//!
//! \sa crashpad::MinidumpOSType
//! \{
#define VER_NT_WORKSTATION 1
#define VER_NT_DOMAIN_CONTROLLER 2
#define VER_NT_SERVER 3
//! \}

//! \anchor VER_PLATFORM_x
//! \name VER_PLATFORM_*
//!
//! \brief Operating system family values for MINIDUMP_SYSTEM_INFO::PlatformId.
//!
//! \sa crashpad::MinidumpOS
//! \{
#define VER_PLATFORM_WIN32s 0
#define VER_PLATFORM_WIN32_WINDOWS 1
#define VER_PLATFORM_WIN32_NT 2
//! \}

#endif  // CRASHPAD_COMPAT_NON_WIN_WINNT_H_
