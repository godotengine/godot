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

#ifndef CRASHPAD_COMPAT_NON_WIN_DBGHELP_H_
#define CRASHPAD_COMPAT_NON_WIN_DBGHELP_H_

#include <stdint.h>

#include "base/strings/string16.h"
#include "compat/non_win/timezoneapi.h"
#include "compat/non_win/verrsrc.h"
#include "compat/non_win/winnt.h"

//! \file

//! \brief The magic number for a minidump file, stored in
//!     MINIDUMP_HEADER::Signature.
//!
//! A hex dump of a little-endian minidump file will begin with the string
//! “MDMP”.
#define MINIDUMP_SIGNATURE ('PMDM')  // 0x4d444d50

//! \brief The version of a minidump file, stored in MINIDUMP_HEADER::Version.
#define MINIDUMP_VERSION (42899)

//! \brief An offset within a minidump file, relative to the start of its
//!     MINIDUMP_HEADER.
//!
//! RVA stands for “relative virtual address”. Within a minidump file, RVAs are
//! used as pointers to link structures together.
//!
//! \sa MINIDUMP_LOCATION_DESCRIPTOR
typedef uint32_t RVA;

//! \brief A pointer to a structure or union within a minidump file.
struct __attribute__((packed, aligned(4))) MINIDUMP_LOCATION_DESCRIPTOR {
  //! \brief The size of the referenced structure or union, in bytes.
  uint32_t DataSize;

  //! \brief The relative virtual address of the structure or union within the
  //!     minidump file.
  RVA Rva;
};

//! \brief A pointer to a snapshot of a region of memory contained within a
//!     minidump file.
//!
//! \sa MINIDUMP_MEMORY_LIST
struct __attribute__((packed, aligned(4))) MINIDUMP_MEMORY_DESCRIPTOR {
  //! \brief The base address of the memory region in the address space of the
  //!     process that the minidump file contains a snapshot of.
  uint64_t StartOfMemoryRange;

  //! \brief The contents of the memory region.
  MINIDUMP_LOCATION_DESCRIPTOR Memory;
};

//! \brief The top-level structure identifying a minidump file.
//!
//! This structure contains a pointer to the stream directory, a second-level
//! structure which in turn contains pointers to third-level structures
//! (“streams”) containing the data within the minidump file. This structure
//! also contains the minidump file’s magic numbers, and other bookkeeping data.
//!
//! This structure must be present at the beginning of a minidump file (at ::RVA
//! 0).
struct __attribute__((packed, aligned(4))) MINIDUMP_HEADER {
  //! \brief The minidump file format magic number, ::MINIDUMP_SIGNATURE.
  uint32_t Signature;

  //! \brief The minidump file format version number, ::MINIDUMP_VERSION.
  uint32_t Version;

  //! \brief The number of MINIDUMP_DIRECTORY elements present in the directory
  //!     referenced by #StreamDirectoryRva.
  uint32_t NumberOfStreams;

  //! \brief A pointer to an array of MINIDUMP_DIRECTORY structures that
  //!     identify all of the streams within this minidump file. The array has
  //!     #NumberOfStreams elements present.
  RVA StreamDirectoryRva;

  //! \brief The minidump file’s checksum. This can be `0`, and in practice, `0`
  //!     is the only value that has ever been seen in this field.
  uint32_t CheckSum;

  //! \brief The time that the minidump file was generated, in `time_t` format,
  //!     the number of seconds since the POSIX epoch.
  uint32_t TimeDateStamp;

  //! \brief A bitfield containing members of ::MINIDUMP_TYPE, describing the
  //!     types of data carried within this minidump file.
  uint64_t Flags;
};

//! \brief A pointer to a stream within a minidump file.
//!
//! Each stream present in a minidump file will have a corresponding
//! MINIDUMP_DIRECTORY entry in the stream directory referenced by
//! MINIDUMP_HEADER::StreamDirectoryRva.
struct __attribute__((packed, aligned(4))) MINIDUMP_DIRECTORY {
  //! \brief The type of stream referenced, a value of ::MINIDUMP_STREAM_TYPE.
  uint32_t StreamType;

  //! \brief A pointer to the stream data within the minidump file.
  MINIDUMP_LOCATION_DESCRIPTOR Location;
};

//! \brief A variable-length UTF-16-encoded string carried within a minidump
//!     file.
//!
//! The UTF-16 string is stored as UTF-16LE or UTF-16BE according to the byte
//! ordering of the minidump file itself.
//!
//! \sa crashpad::MinidumpUTF8String
struct __attribute__((packed, aligned(4))) MINIDUMP_STRING {
  //! \brief The length of the #Buffer field in bytes, not including the `NUL`
  //!     terminator.
  //!
  //! \note This field is interpreted as a byte count, not a count of UTF-16
  //!     code units or Unicode code points.
  uint32_t Length;

  //! \brief The string, encoded in UTF-16, and terminated with a UTF-16 `NUL`
  //!     code unit (two `NUL` bytes).
  base::char16 Buffer[0];
};

//! \brief Minidump stream type values for MINIDUMP_DIRECTORY::StreamType. Each
//!     stream structure has a corresponding stream type value to identify it.
//!
//! \sa crashpad::MinidumpStreamType
enum MINIDUMP_STREAM_TYPE {
  //! \brief The stream type for MINIDUMP_THREAD_LIST.
  ThreadListStream = 3,

  //! \brief The stream type for MINIDUMP_MODULE_LIST.
  ModuleListStream = 4,

  //! \brief The stream type for MINIDUMP_MEMORY_LIST.
  MemoryListStream = 5,

  //! \brief The stream type for MINIDUMP_EXCEPTION_STREAM.
  ExceptionStream = 6,

  //! \brief The stream type for MINIDUMP_SYSTEM_INFO.
  SystemInfoStream = 7,

  //! \brief The stream contains information about active `HANDLE`s.
  HandleDataStream = 12,

  //! \brief The stream type for MINIDUMP_UNLOADED_MODULE_LIST.
  UnloadedModuleListStream = 14,

  //! \brief The stream type for MINIDUMP_MISC_INFO, MINIDUMP_MISC_INFO_2,
  //!     MINIDUMP_MISC_INFO_3, MINIDUMP_MISC_INFO_4, and MINIDUMP_MISC_INFO_5.
  //!
  //! More recent versions of this stream are supersets of earlier versions.
  //!
  //! The exact version of the stream that is present is implied by the stream’s
  //! size. Furthermore, this stream contains a field,
  //! MINIDUMP_MISC_INFO::Flags1, that indicates which data is present and
  //! valid.
  MiscInfoStream = 15,

  //! \brief The stream type for MINIDUMP_MEMORY_INFO_LIST.
  MemoryInfoListStream = 16,

  //! \brief Values greater than this value will not be used by the system
  //!     and can be used for custom user data streams.
  LastReservedStream = 0xffff,
};

//! \brief Information about the CPU (or CPUs) that ran the process that the
//!     minidump file contains a snapshot of.
//!
//! This union only appears as MINIDUMP_SYSTEM_INFO::Cpu. Its interpretation is
//! controlled by MINIDUMP_SYSTEM_INFO::ProcessorArchitecture.
union __attribute__((packed, aligned(4))) CPU_INFORMATION {
  //! \brief Information about 32-bit x86 CPUs, or x86_64 CPUs when running
  //!     32-bit x86 processes.
  struct __attribute__((packed, aligned(4))) {
    //! \brief The CPU’s vendor identification string as encoded in `cpuid 0`
    //!     `ebx`, `edx`, and `ecx`, represented as it appears in these
    //!     registers.
    //!
    //! For Intel CPUs, `[0]` will encode “Genu”, `[1]` will encode “ineI”, and
    //! `[2]` will encode “ntel”, for a vendor ID string “GenuineIntel”.
    //!
    //! \note The Windows documentation incorrectly states that these fields are
    //!     to be interpreted as `cpuid 0` `eax`, `ebx`, and `ecx`.
    uint32_t VendorId[3];

    //! \brief Family, model, and stepping ID values as encoded in `cpuid 1`
    //!     `eax`.
    uint32_t VersionInformation;

    //! \brief A bitfield containing supported CPU capabilities as encoded in
    //!     `cpuid 1` `edx`.
    uint32_t FeatureInformation;

    //! \brief A bitfield containing supported CPU capabalities as encoded in
    //!     `cpuid 0x80000001` `edx`.
    //!
    //! This field is only valid if #VendorId identifies the CPU vendor as
    //! “AuthenticAMD”.
    uint32_t AMDExtendedCpuFeatures;
  } X86CpuInfo;

  //! \brief Information about non-x86 CPUs, and x86_64 CPUs when not running
  //!     32-bit x86 processes.
  struct __attribute__((packed, aligned(4))) {
    //! \brief Bitfields containing supported CPU capabilities as identified by
    //!     bits corresponding to \ref PF_x "PF_*" values passed to
    //!     `IsProcessorFeaturePresent()`.
    uint64_t ProcessorFeatures[2];
  } OtherCpuInfo;
};

//! \brief Information about the system that hosted the process that the
//!     minidump file contains a snapshot of.
struct __attribute__((packed, aligned(4))) MINIDUMP_SYSTEM_INFO {
  // The next 4 fields are from the SYSTEM_INFO structure returned by
  // GetSystemInfo().

  //! \brief The system’s CPU architecture. This may be a \ref
  //!     PROCESSOR_ARCHITECTURE_x "PROCESSOR_ARCHITECTURE_*" value, or a member
  //!     of crashpad::MinidumpCPUArchitecture.
  //!
  //! In some cases, a system may be able to run processes of multiple specific
  //! architecture types. For example, systems based on 64-bit architectures
  //! such as x86_64 are often able to run 32-bit code of another architecture
  //! in the same family, such as 32-bit x86. On these systems, this field will
  //! identify the architecture of the process that the minidump file contains a
  //! snapshot of.
  uint16_t ProcessorArchitecture;

  //! \brief General CPU version information.
  //!
  //! The precise interpretation of this field is specific to each CPU
  //! architecture. For x86-family CPUs (including x86_64 and 32-bit x86), this
  //! field contains the CPU family ID value from `cpuid 1` `eax`, adjusted to
  //! take the extended family ID into account.
  uint16_t ProcessorLevel;

  //! \brief Specific CPU version information.
  //!
  //! The precise interpretation of this field is specific to each CPU
  //! architecture. For x86-family CPUs (including x86_64 and 32-bit x86), this
  //! field contains values obtained from `cpuid 1` `eax`: the high byte
  //! contains the CPU model ID value adjusted to take the extended model ID
  //! into account, and the low byte contains the CPU stepping ID value.
  uint16_t ProcessorRevision;

  //! \brief The total number of CPUs present in the system.
  uint8_t NumberOfProcessors;

  // The next 7 fields are from the OSVERSIONINFOEX structure returned by
  // GetVersionEx().

  //! \brief The system’s operating system type, which distinguishes between
  //!     “desktop” or “workstation” systems and “server” systems. This may be a
  //!     \ref VER_NT_x "VER_NT_*" value, or a member of
  //!     crashpad::MinidumpOSType.
  uint8_t ProductType;

  //! \brief The system’s operating system version number’s first (major)
  //!     component.
  //!
  //!  - For Windows 7 (NT 6.1) SP1, version 6.1.7601, this would be `6`.
  //!  - For macOS 10.12.1, this would be `10`.
  uint32_t MajorVersion;

  //! \brief The system’s operating system version number’s second (minor)
  //!     component.
  //!
  //!  - For Windows 7 (NT 6.1) SP1, version 6.1.7601, this would be `1`.
  //!  - For macOS 10.12.1, this would be `12`.
  uint32_t MinorVersion;

  //! \brief The system’s operating system version number’s third (build or
  //!     patch) component.
  //!
  //!  - For Windows 7 (NT 6.1) SP1, version 6.1.7601, this would be `7601`.
  //!  - For macOS 10.12.1, this would be `1`.
  uint32_t BuildNumber;

  //! \brief The system’s operating system family. This may be a \ref
  //!     VER_PLATFORM_x "VER_PLATFORM_*" value, or a member of
  //!     crashpad::MinidumpOS.
  uint32_t PlatformId;

  //! \brief ::RVA of a MINIDUMP_STRING containing operating system-specific
  //!     version information.
  //!
  //! This field further identifies an operating system version beyond its
  //! version number fields. Historically, “CSD” stands for “corrective service
  //! diskette.”
  //!
  //!  - On Windows, this is the name of the installed operating system service
  //!    pack, such as “Service Pack 1”. If no service pack is installed, this
  //!    field references an empty string.
  //!  - On macOS, this is the operating system build number from `sw_vers
  //!    -buildVersion`. For macOS 10.12.1 on most hardware types, this would
  //!    be `16B2657`.
  //!  - On Linux and other Unix-like systems, this is the kernel version from
  //!    `uname -srvm`, possibly with additional information appended. On
  //!    Android, the `ro.build.fingerprint` system property is appended.
  RVA CSDVersionRva;

  //! \brief A bitfield identifying products installed on the system. This is
  //!     composed of \ref VER_SUITE_x "VER_SUITE_*" values.
  //!
  //! This field is Windows-specific, and has no meaning on other operating
  //! systems.
  uint16_t SuiteMask;

  uint16_t Reserved2;

  //! \brief Information about the system’s CPUs.
  //!
  //! This field is a union. Which of its members should be expressed is
  //! controlled by the #ProcessorArchitecture field. If it is set to
  //! crashpad::kMinidumpCPUArchitectureX86, the CPU_INFORMATION::X86CpuInfo
  //! field is expressed. Otherwise, the CPU_INFORMATION::OtherCpuInfo field is
  //! expressed.
  //!
  //! \note Older Breakpad implementations produce minidump files that express
  //!     CPU_INFORMATION::X86CpuInfo when #ProcessorArchitecture is set to
  //!     crashpad::kMinidumpCPUArchitectureAMD64. Minidump files produced by
  //!     `dbghelp.dll` on Windows express CPU_INFORMATION::OtherCpuInfo in this
  //!     case.
  CPU_INFORMATION Cpu;
};

//! \brief Information about a specific thread within the process.
//!
//! \sa MINIDUMP_THREAD_LIST
struct __attribute__((packed, aligned(4))) MINIDUMP_THREAD {
  //! \brief The thread’s ID. This may be referenced by
  //!     MINIDUMP_EXCEPTION_STREAM::ThreadId.
  uint32_t ThreadId;

  //! \brief The thread’s suspend count.
  //!
  //! This field will be `0` if the thread is schedulable (not suspended).
  uint32_t SuspendCount;

  //! \brief The thread’s priority class.
  //!
  //! On Windows, this is a `*_PRIORITY_CLASS` value. `NORMAL_PRIORITY_CLASS`
  //! has value `0x20`; higher priority classes have higher values.
  uint32_t PriorityClass;

  //! \brief The thread’s priority level.
  //!
  //! On Windows, this is a `THREAD_PRIORITY_*` value. `THREAD_PRIORITY_NORMAL`
  //! has value `0`; higher priorities have higher values, and lower priorities
  //! have lower (negative) values.
  uint32_t Priority;

  //! \brief The address of the thread’s thread environment block in the address
  //!     space of the process that the minidump file contains a snapshot of.
  //!
  //! The thread environment block contains thread-local data.
  //!
  //! A MINIDUMP_MEMORY_DESCRIPTOR may be present in the MINIDUMP_MEMORY_LIST
  //! stream containing the thread-local data pointed to by this field.
  uint64_t Teb;

  //! \brief A snapshot of the thread’s stack.
  //!
  //! A MINIDUMP_MEMORY_DESCRIPTOR may be present in the MINIDUMP_MEMORY_LIST
  //! stream containing a pointer to the same memory range referenced by this
  //! field.
  MINIDUMP_MEMORY_DESCRIPTOR Stack;

  //! \brief A pointer to a CPU-specific CONTEXT structure containing the
  //!     thread’s context at the time the snapshot was taken.
  //!
  //! If the minidump file was generated as a result of an exception taken on
  //! this thread, this field may identify a different context than the
  //! exception context. For these minidump files, a MINIDUMP_EXCEPTION_STREAM
  //! stream will be present, and the context contained within that stream will
  //! be the exception context.
  //!
  //! The interpretation of the context structure is dependent on the CPU
  //! architecture identified by MINIDUMP_SYSTEM_INFO::ProcessorArchitecture.
  //! For crashpad::kMinidumpCPUArchitectureX86, this will be
  //! crashpad::MinidumpContextX86. For crashpad::kMinidumpCPUArchitectureAMD64,
  //! this will be crashpad::MinidumpContextAMD64.
  MINIDUMP_LOCATION_DESCRIPTOR ThreadContext;
};

//! \brief Information about all threads within the process.
struct __attribute__((packed, aligned(4))) MINIDUMP_THREAD_LIST {
  //! \brief The number of threads present in the #Threads array.
  uint32_t NumberOfThreads;

  //! \brief Structures identifying each thread within the process.
  MINIDUMP_THREAD Threads[0];
};

//! \brief Information about an exception that occurred in the process.
struct __attribute__((packed, aligned(4))) MINIDUMP_EXCEPTION {
  //! \brief The top-level exception code identifying the exception, in
  //!     operating system-specific values.
  //!
  //! For macOS minidumps, this will be an \ref EXC_x "EXC_*" exception type,
  //! such as `EXC_BAD_ACCESS`. `EXC_CRASH` will not appear here for exceptions
  //! processed as `EXC_CRASH` when generated from another preceding exception:
  //! the original exception code will appear instead. The exception type as it
  //! was received will appear at index 0 of #ExceptionInformation.
  //!
  //! For Windows minidumps, this will be an `EXCEPTION_*` exception type, such
  //! as `EXCEPTION_ACCESS_VIOLATION`.
  //!
  //! \note This field is named ExceptionCode, but what is known as the
  //!     “exception code” on macOS/Mach is actually stored in the
  //!     #ExceptionFlags field of a minidump file.
  //!
  //! \todo Document the possible values by OS. There may be OS-specific enums
  //!     in minidump_extensions.h.
  uint32_t ExceptionCode;

  //! \brief Additional exception flags that further identify the exception, in
  //!     operating system-specific values.
  //!
  //! For macOS minidumps, this will be the value of the exception code at index
  //! 0 as received by a Mach exception handler, except:
  //!  * For exception type `EXC_CRASH` generated from another preceding
  //!    exception, the original exception code will appear here, not the code
  //!    as received by the Mach exception handler.
  //!  * For exception types `EXC_RESOURCE` and `EXC_GUARD`, the high 32 bits of
  //!    the code received by the Mach exception handler will appear here.
  //!
  //! In all cases for macOS minidumps, the code as it was received by the Mach
  //! exception handler will appear at index 1 of #ExceptionInformation.
  //!
  //! For Windows minidumps, this will either be `0` if the exception is
  //! continuable, or `EXCEPTION_NONCONTINUABLE` to indicate a noncontinuable
  //! exception.
  //!
  //! \todo Document the possible values by OS. There may be OS-specific enums
  //!     in minidump_extensions.h.
  uint32_t ExceptionFlags;

  //! \brief An address, in the address space of the process that this minidump
  //!     file contains a snapshot of, of another MINIDUMP_EXCEPTION. This field
  //!     is used for nested exceptions.
  uint64_t ExceptionRecord;

  //! \brief The address that caused the exception.
  //!
  //! This may be the address that caused a fault on data access, or it may be
  //! the instruction pointer that contained an offending instruction.
  uint64_t ExceptionAddress;

  //! \brief The number of valid elements in #ExceptionInformation.
  uint32_t NumberParameters;

  uint32_t __unusedAlignment;

  //! \brief Additional information about the exception, specific to the
  //!     operating system and possibly the #ExceptionCode.
  //!
  //! For macOS minidumps, this will contain the exception type as received by a
  //! Mach exception handler and the values of the `codes[0]` and `codes[1]`
  //! (exception code and subcode) parameters supplied to the Mach exception
  //! handler. Unlike #ExceptionCode and #ExceptionFlags, the values received by
  //! a Mach exception handler are used directly here even for the `EXC_CRASH`,
  //! `EXC_RESOURCE`, and `EXC_GUARD` exception types.

  //! For Windows, these are additional arguments (if any) as provided to
  //! `RaiseException()`.
  uint64_t ExceptionInformation[EXCEPTION_MAXIMUM_PARAMETERS];
};

//! \brief Information about the exception that triggered a minidump file’s
//!     generation.
struct __attribute__((packed, aligned(4))) MINIDUMP_EXCEPTION_STREAM {
  //! \brief The ID of the thread that caused the exception.
  //!
  //! \sa MINIDUMP_THREAD::ThreadId
  uint32_t ThreadId;

  uint32_t __alignment;

  //! \brief Information about the exception.
  MINIDUMP_EXCEPTION ExceptionRecord;

  //! \brief A pointer to a CPU-specific CONTEXT structure containing the
  //!     thread’s context at the time the exception was caused.
  //!
  //! The interpretation of the context structure is dependent on the CPU
  //! architecture identified by MINIDUMP_SYSTEM_INFO::ProcessorArchitecture.
  //! For crashpad::kMinidumpCPUArchitectureX86, this will be
  //! crashpad::MinidumpContextX86. For crashpad::kMinidumpCPUArchitectureAMD64,
  //! this will be crashpad::MinidumpContextAMD64.
  MINIDUMP_LOCATION_DESCRIPTOR ThreadContext;
};

//! \brief Information about a specific module loaded within the process at the
//!     time the snapshot was taken.
//!
//! A module may be the main executable, a shared library, or a loadable module.
//!
//! \sa MINIDUMP_MODULE_LIST
struct __attribute__((packed, aligned(4))) MINIDUMP_MODULE {
  //! \brief The base address of the loaded module in the address space of the
  //!     process that the minidump file contains a snapshot of.
  uint64_t BaseOfImage;

  //! \brief The size of the loaded module.
  uint32_t SizeOfImage;

  //! \brief The loaded module’s checksum, or `0` if unknown.
  //!
  //! On Windows, this field comes from the `CheckSum` field of the module’s
  //! `IMAGE_OPTIONAL_HEADER` structure, if present. It reflects the checksum at
  //! the time the module was linked.
  uint32_t CheckSum;

  //! \brief The module’s timestamp, in `time_t` units, seconds since the POSIX
  //!     epoch, or `0` if unknown.
  //!
  //! On Windows, this field comes from the `TimeDateStamp` field of the
  //! module’s `IMAGE_FILE_HEADER` structure. It reflects the timestamp at the
  //! time the module was linked.
  uint32_t TimeDateStamp;

  //! \brief ::RVA of a MINIDUMP_STRING containing the module’s path or file
  //!     name.
  RVA ModuleNameRva;

  //! \brief The module’s version information.
  VS_FIXEDFILEINFO VersionInfo;

  //! \brief A pointer to the module’s CodeView record, typically a link to its
  //!     debugging information in crashpad::CodeViewRecordPDB70 format.
  //!
  //! The specific format of the CodeView record is indicated by its signature,
  //! the first 32-bit value in the structure. For links to debugging
  //! information in contemporary usage, this is normally a
  //! crashpad::CodeViewRecordPDB70 structure, but may be a
  //! crashpad::CodeViewRecordPDB20 structure instead. These structures identify
  //! a link to debugging data within a `.pdb` (Program Database) file. See <a
  //! href="http://www.debuginfo.com/articles/debuginfomatch.html#pdbfiles">Matching
  //! Debug Information</a>, PDB Files.
  //!
  //! On Windows, it is also possible for the CodeView record to contain
  //! debugging information itself, as opposed to a link to a `.pdb` file. See
  //! <a
  //! href="http://pierrelib.pagesperso-orange.fr/exec_formats/MS_Symbol_Type_v1.0.pdf#page=71">Microsoft
  //! Symbol and Type Information</a>, section 7.2, “Debug Information Format”
  //! for a list of debug information formats, and <i>Undocumented Windows 2000
  //! Secrets</i>, Windows 2000 Debugging Support/Microsoft Symbol File
  //! Internals/CodeView Subsections for an in-depth description of the CodeView
  //! 4.1 format. Signatures seen in the wild include “NB09” (0x3930424e) for
  //! CodeView 4.1 and “NB11” (0x3131424e) for CodeView 5.0. This form of
  //! debugging information within the module, as opposed to a link to an
  //! external `.pdb` file, is chosen by building with `/Z7` in Visual Studio
  //! 6.0 (1998) and earlier. This embedded form of debugging information is now
  //! considered obsolete.
  //!
  //! On Windows, the CodeView record is taken from a module’s
  //! IMAGE_DEBUG_DIRECTORY entry whose Type field has the value
  //! IMAGE_DEBUG_TYPE_CODEVIEW (`2`), if any. Records in
  //! crashpad::CodeViewRecordPDB70 format are generated by Visual Studio .NET
  //! (2002) (version 7.0) and later.
  //!
  //! When the CodeView record is not present, the fields of this
  //! MINIDUMP_LOCATION_DESCRIPTOR will be `0`.
  MINIDUMP_LOCATION_DESCRIPTOR CvRecord;

  //! \brief A pointer to the module’s miscellaneous debugging record, a
  //!     structure of type IMAGE_DEBUG_MISC.
  //!
  //! This field is Windows-specific, and has no meaning on other operating
  //! systems. It is largely obsolete on Windows, where it was used to link to
  //! debugging information stored in a `.dbg` file. `.dbg` files have been
  //! superseded by `.pdb` files.
  //!
  //! On Windows, the miscellaneous debugging record is taken from module’s
  //! IMAGE_DEBUG_DIRECTORY entry whose Type field has the value
  //! IMAGE_DEBUG_TYPE_MISC (`4`), if any.
  //!
  //! When the miscellaneous debugging record is not present, the fields of this
  //! MINIDUMP_LOCATION_DESCRIPTOR will be `0`.
  //!
  //! \sa #CvRecord
  MINIDUMP_LOCATION_DESCRIPTOR MiscRecord;

  uint64_t Reserved0;
  uint64_t Reserved1;
};

//! \brief Information about all modules loaded within the process at the time
//!     the snapshot was taken.
struct __attribute__((packed, aligned(4))) MINIDUMP_MODULE_LIST {
  //! \brief The number of modules present in the #Modules array.
  uint32_t NumberOfModules;

  //! \brief Structures identifying each module present in the minidump file.
  MINIDUMP_MODULE Modules[0];
};

//! \brief Information about memory regions within the process.
//!
//! Typically, a minidump file will not contain a snapshot of a process’ entire
//! memory image. For minidump files identified as ::MiniDumpNormal in
//! MINIDUMP_HEADER::Flags, memory regions are limited to those referenced by
//! MINIDUMP_THREAD::Stack fields, and a small number of others possibly related
//! to the exception that triggered the snapshot to be taken.
struct __attribute__((packed, aligned(4))) MINIDUMP_MEMORY_LIST {
  //! \brief The number of memory regions present in the #MemoryRanges array.
  uint32_t NumberOfMemoryRanges;

  //! \brief Structures identifying each memory region present in the minidump
  //!     file.
  MINIDUMP_MEMORY_DESCRIPTOR MemoryRanges[0];
};

//! \brief Contains the state of an individual system handle at the time the
//!     snapshot was taken. This structure is Windows-specific.
//!
//! \sa MINIDUMP_HANDLE_DESCRIPTOR_2
struct __attribute__((packed, aligned(4))) MINIDUMP_HANDLE_DESCRIPTOR {
  //! \brief The Windows `HANDLE` value.
  uint64_t Handle;

  //! \brief An RVA to a MINIDUMP_STRING structure that specifies the object
  //!     type of the handle. This member can be zero.
  RVA TypeNameRva;

  //! \brief An RVA to a MINIDUMP_STRING structure that specifies the object
  //!     name of the handle. This member can be zero.
  RVA ObjectNameRva;

  //! \brief The attributes for the handle, this corresponds to `OBJ_INHERIT`,
  //!     `OBJ_CASE_INSENSITIVE`, etc.
  uint32_t Attributes;

  //! \brief The `ACCESS_MASK` for the handle.
  uint32_t GrantedAccess;

  //! \brief This is the number of open handles to the object that this handle
  //!     refers to.
  uint32_t HandleCount;

  //! \brief This is the number kernel references to the object that this
  //!     handle refers to.
  uint32_t PointerCount;
};

//! \brief Contains the state of an individual system handle at the time the
//!     snapshot was taken. This structure is Windows-specific.
//!
//! \sa MINIDUMP_HANDLE_DESCRIPTOR
struct __attribute__((packed, aligned(4))) MINIDUMP_HANDLE_DESCRIPTOR_2
    : public MINIDUMP_HANDLE_DESCRIPTOR {
  //! \brief An RVA to a MINIDUMP_HANDLE_OBJECT_INFORMATION structure that
  //!     specifies object-specific information. This member can be zero if
  //!     there is no extra information.
  RVA ObjectInfoRva;

  //! \brief Must be zero.
  uint32_t Reserved0;
};

//! \brief Represents the header for a handle data stream.
//!
//! A list of MINIDUMP_HANDLE_DESCRIPTOR or MINIDUMP_HANDLE_DESCRIPTOR_2
//! structures will immediately follow in the stream.
struct __attribute((packed, aligned(4))) MINIDUMP_HANDLE_DATA_STREAM {
  //! \brief The size of the header information for the stream, in bytes. This
  //!     value is `sizeof(MINIDUMP_HANDLE_DATA_STREAM)`.
  uint32_t SizeOfHeader;

  //! \brief The size of a descriptor in the stream, in bytes. This value is
  //!     `sizeof(MINIDUMP_HANDLE_DESCRIPTOR)` or
  //!     `sizeof(MINIDUMP_HANDLE_DESCRIPTOR_2)`.
  uint32_t SizeOfDescriptor;

  //! \brief The number of descriptors in the stream.
  uint32_t NumberOfDescriptors;

  //! \brief Must be zero.
  uint32_t Reserved;
};

//! \brief Information about a specific module that was recorded as being
//!     unloaded at the time the snapshot was taken.
//!
//! An unloaded module may be a shared library or a loadable module.
//!
//! \sa MINIDUMP_UNLOADED_MODULE_LIST
struct __attribute__((packed, aligned(4))) MINIDUMP_UNLOADED_MODULE {
  //! \brief The base address where the module was loaded in the address space
  //!     of the process that the minidump file contains a snapshot of.
  uint64_t BaseOfImage;

  //! \brief The size of the unloaded module.
  uint32_t SizeOfImage;

  //! \brief The module’s checksum, or `0` if unknown.
  //!
  //! On Windows, this field comes from the `CheckSum` field of the module’s
  //! `IMAGE_OPTIONAL_HEADER` structure, if present. It reflects the checksum at
  //! the time the module was linked.
  uint32_t CheckSum;

  //! \brief The module’s timestamp, in `time_t` units, seconds since the POSIX
  //!     epoch, or `0` if unknown.
  //!
  //! On Windows, this field comes from the `TimeDateStamp` field of the
  //! module’s `IMAGE_FILE_HEADER` structure. It reflects the timestamp at the
  //! time the module was linked.
  uint32_t TimeDateStamp;

  //! \brief ::RVA of a MINIDUMP_STRING containing the module’s path or file
  //!     name.
  RVA ModuleNameRva;
};

//! \brief Information about all modules recorded as unloaded when the snapshot
//!     was taken.
//!
//! A list of MINIDUMP_UNLOADED_MODULE structures will immediately follow in the
//! stream.
struct __attribute__((packed, aligned(4))) MINIDUMP_UNLOADED_MODULE_LIST {
  //! \brief The size of the header information for the stream, in bytes. This
  //!     value is `sizeof(MINIDUMP_UNLOADED_MODULE_LIST)`.
  uint32_t SizeOfHeader;

  //! \brief The size of a descriptor in the stream, in bytes. This value is
  //!     `sizeof(MINIDUMP_UNLOADED_MODULE)`.
  uint32_t SizeOfEntry;

  //! \brief The number of entries in the stream.
  uint32_t NumberOfEntries;
};

//! \brief Information about XSAVE-managed state stored within CPU-specific
//!     context structures.
struct __attribute__((packed, aligned(4))) XSTATE_CONFIG_FEATURE_MSC_INFO {
  //! \brief The size of this structure, in bytes. This value is
  //!     `sizeof(XSTATE_CONFIG_FEATURE_MSC_INFO)`.
  uint32_t SizeOfInfo;

  //! \brief The size of a CPU-specific context structure carrying all XSAVE
  //!     state components described by this structure.
  //!
  //! Equivalent to the value returned by `InitializeContext()` in \a
  //! ContextLength.
  uint32_t ContextSize;

  //! \brief The XSAVE state-component bitmap, XSAVE_BV.
  //!
  //! See Intel Software Developer’s Manual, Volume 1: Basic Architecture
  //! (253665-060), 13.4.2 “XSAVE Header”.
  uint64_t EnabledFeatures;

  //! \brief The location of each state component within a CPU-specific context
  //!     structure.
  //!
  //! This array is indexed by bit position numbers used in #EnabledFeatures.
  XSTATE_FEATURE Features[MAXIMUM_XSTATE_FEATURES];
};

//! \anchor MINIDUMP_MISCx
//! \name MINIDUMP_MISC*
//!
//! \brief Field validity flag values for MINIDUMP_MISC_INFO::Flags1.
//! \{

//! \brief MINIDUMP_MISC_INFO::ProcessId is valid.
#define MINIDUMP_MISC1_PROCESS_ID 0x00000001

//! \brief The time-related fields in MINIDUMP_MISC_INFO are valid.
//!
//! The following fields are valid:
//!  - MINIDUMP_MISC_INFO::ProcessCreateTime
//!  - MINIDUMP_MISC_INFO::ProcessUserTime
//!  - MINIDUMP_MISC_INFO::ProcessKernelTime
#define MINIDUMP_MISC1_PROCESS_TIMES 0x00000002

//! \brief The CPU-related fields in MINIDUMP_MISC_INFO_2 are valid.
//!
//! The following fields are valid:
//!  - MINIDUMP_MISC_INFO_2::ProcessorMaxMhz
//!  - MINIDUMP_MISC_INFO_2::ProcessorCurrentMhz
//!  - MINIDUMP_MISC_INFO_2::ProcessorMhzLimit
//!  - MINIDUMP_MISC_INFO_2::ProcessorMaxIdleState
//!  - MINIDUMP_MISC_INFO_2::ProcessorCurrentIdleState
//!
//! \note This macro should likely have been named
//!     MINIDUMP_MISC2_PROCESSOR_POWER_INFO.
#define MINIDUMP_MISC1_PROCESSOR_POWER_INFO 0x00000004

//! \brief MINIDUMP_MISC_INFO_3::ProcessIntegrityLevel is valid.
#define MINIDUMP_MISC3_PROCESS_INTEGRITY 0x00000010

//! \brief MINIDUMP_MISC_INFO_3::ProcessExecuteFlags is valid.
#define MINIDUMP_MISC3_PROCESS_EXECUTE_FLAGS 0x00000020

//! \brief The time zone-related fields in MINIDUMP_MISC_INFO_3 are valid.
//!
//! The following fields are valid:
//!  - MINIDUMP_MISC_INFO_3::TimeZoneId
//!  - MINIDUMP_MISC_INFO_3::TimeZone
#define MINIDUMP_MISC3_TIMEZONE 0x00000040

//! \brief MINIDUMP_MISC_INFO_3::ProtectedProcess is valid.
#define MINIDUMP_MISC3_PROTECTED_PROCESS 0x00000080

//! \brief The build string-related fields in MINIDUMP_MISC_INFO_4 are valid.
//!
//! The following fields are valid:
//!  - MINIDUMP_MISC_INFO_4::BuildString
//!  - MINIDUMP_MISC_INFO_4::DbgBldStr
#define MINIDUMP_MISC4_BUILDSTRING 0x00000100

//! \brief MINIDUMP_MISC_INFO_5::ProcessCookie is valid.
#define MINIDUMP_MISC5_PROCESS_COOKIE 0x00000200

//! \}

//! \brief Information about the process that the minidump file contains a
//!     snapshot of, as well as the system that hosted that process.
//!
//! \sa \ref MINIDUMP_MISCx "MINIDUMP_MISC*"
//! \sa MINIDUMP_MISC_INFO_2
//! \sa MINIDUMP_MISC_INFO_3
//! \sa MINIDUMP_MISC_INFO_4
//! \sa MINIDUMP_MISC_INFO_5
//! \sa MINIDUMP_MISC_INFO_N
struct __attribute__((packed, aligned(4))) MINIDUMP_MISC_INFO {
  //! \brief The size of the structure.
  //!
  //! This field can be used to distinguish between different versions of this
  //! structure: MINIDUMP_MISC_INFO, MINIDUMP_MISC_INFO_2, MINIDUMP_MISC_INFO_3,
  //! and MINIDUMP_MISC_INFO_4.
  //!
  //! \sa Flags1
  uint32_t SizeOfInfo;

  //! \brief A bit field of \ref MINIDUMP_MISCx "MINIDUMP_MISC*" values
  //!     indicating which fields of this structure contain valid data.
  uint32_t Flags1;

  //! \brief The process ID of the process.
  uint32_t ProcessId;

  //! \brief The time that the process started, in `time_t` units, seconds since
  //!     the POSIX epoch.
  uint32_t ProcessCreateTime;

  //! \brief The amount of user-mode CPU time used by the process, in seconds,
  //!     at the time of the snapshot.
  uint32_t ProcessUserTime;

  //! \brief The amount of system-mode (kernel) CPU time used by the process, in
  //!     seconds, at the time of the snapshot.
  uint32_t ProcessKernelTime;
};

//! \brief Information about the process that the minidump file contains a
//!     snapshot of, as well as the system that hosted that process.
//!
//! This structure variant is used on Windows Vista (NT 6.0) and later.
//!
//! \sa \ref MINIDUMP_MISCx "MINIDUMP_MISC*"
//! \sa MINIDUMP_MISC_INFO
//! \sa MINIDUMP_MISC_INFO_3
//! \sa MINIDUMP_MISC_INFO_4
//! \sa MINIDUMP_MISC_INFO_5
//! \sa MINIDUMP_MISC_INFO_N
struct __attribute__((packed, aligned(4))) MINIDUMP_MISC_INFO_2
    : public MINIDUMP_MISC_INFO {
  //! \brief The maximum clock rate of the system’s CPU or CPUs, in MHz.
  uint32_t ProcessorMaxMhz;

  //! \brief The clock rate of the system’s CPU or CPUs, in MHz, at the time of
  //!     the snapshot.
  uint32_t ProcessorCurrentMhz;

  //! \brief The maximum clock rate of the system’s CPU or CPUs, in MHz, reduced
  //!     by any thermal limitations, at the time of the snapshot.
  uint32_t ProcessorMhzLimit;

  //! \brief The maximum idle state of the system’s CPU or CPUs.
  uint32_t ProcessorMaxIdleState;

  //! \brief The idle state of the system’s CPU or CPUs at the time of the
  //!     snapshot.
  uint32_t ProcessorCurrentIdleState;
};

//! \brief Information about the process that the minidump file contains a
//!     snapshot of, as well as the system that hosted that process.
//!
//! This structure variant is used on Windows 7 (NT 6.1) and later.
//!
//! \sa \ref MINIDUMP_MISCx "MINIDUMP_MISC*"
//! \sa MINIDUMP_MISC_INFO
//! \sa MINIDUMP_MISC_INFO_2
//! \sa MINIDUMP_MISC_INFO_4
//! \sa MINIDUMP_MISC_INFO_5
//! \sa MINIDUMP_MISC_INFO_N
struct __attribute__((packed, aligned(4))) MINIDUMP_MISC_INFO_3
    : public MINIDUMP_MISC_INFO_2 {
  //! \brief The process’ integrity level.
  //!
  //! Windows typically uses `SECURITY_MANDATORY_MEDIUM_RID` (0x2000) for
  //! processes belonging to normal authenticated users and
  //! `SECURITY_MANDATORY_HIGH_RID` (0x3000) for elevated processes.
  //!
  //! This field is Windows-specific, and has no meaning on other operating
  //! systems.
  uint32_t ProcessIntegrityLevel;

  //! \brief The process’ execute flags.
  //!
  //! On Windows, this appears to be returned by `NtQueryInformationProcess()`
  //! with an argument of `ProcessExecuteFlags` (34).
  //!
  //! This field is Windows-specific, and has no meaning on other operating
  //! systems.
  uint32_t ProcessExecuteFlags;

  //! \brief Whether the process is protected.
  //!
  //! This field is Windows-specific, and has no meaning on other operating
  //! systems.
  uint32_t ProtectedProcess;

  //! \brief Whether daylight saving time was being observed in the system’s
  //!     location at the time of the snapshot.
  //!
  //! This field can contain the following values:
  //!  - `0` if the location does not observe daylight saving time at all. The
  //!    TIME_ZONE_INFORMATION::StandardName field of #TimeZoneId contains the
  //!    time zone name.
  //!  - `1` if the location observes daylight saving time, but standard time
  //!    was in effect at the time of the snapshot. The
  //!    TIME_ZONE_INFORMATION::StandardName field of #TimeZoneId contains the
  //!    time zone name.
  //!  - `2` if the location observes daylight saving time, and it was in effect
  //!    at the time of the snapshot. The TIME_ZONE_INFORMATION::DaylightName
  //!    field of #TimeZoneId contains the time zone name.
  //!
  //! \sa #TimeZone
  uint32_t TimeZoneId;

  //! \brief Information about the time zone at the system’s location.
  //!
  //! \sa #TimeZoneId
  TIME_ZONE_INFORMATION TimeZone;
};

//! \brief Information about the process that the minidump file contains a
//!     snapshot of, as well as the system that hosted that process.
//!
//! This structure variant is used on Windows 8 (NT 6.2) and later.
//!
//! \sa \ref MINIDUMP_MISCx "MINIDUMP_MISC*"
//! \sa MINIDUMP_MISC_INFO
//! \sa MINIDUMP_MISC_INFO_2
//! \sa MINIDUMP_MISC_INFO_3
//! \sa MINIDUMP_MISC_INFO_5
//! \sa MINIDUMP_MISC_INFO_N
struct __attribute__((packed, aligned(4))) MINIDUMP_MISC_INFO_4
    : public MINIDUMP_MISC_INFO_3 {
  //! \brief The operating system’s “build string”, a string identifying a
  //!     specific build of the operating system.
  //!
  //! This string is UTF-16-encoded and terminated by a UTF-16 `NUL` code unit.
  //!
  //! On Windows 8.1 (NT 6.3), this is “6.3.9600.17031
  //! (winblue_gdr.140221-1952)”.
  base::char16 BuildString[260];

  //! \brief The minidump producer’s “build string”, a string identifying the
  //!     module that produced a minidump file.
  //!
  //! This string is UTF-16-encoded and terminated by a UTF-16 `NUL` code unit.
  //!
  //! On Windows 8.1 (NT 6.3), this may be “dbghelp.i386,6.3.9600.16520” or
  //! “dbghelp.amd64,6.3.9600.16520” depending on CPU architecture.
  base::char16 DbgBldStr[40];
};

//! \brief Information about the process that the minidump file contains a
//!     snapshot of, as well as the system that hosted that process.
//!
//! This structure variant is used on Windows 10 and later.
//!
//! \sa \ref MINIDUMP_MISCx "MINIDUMP_MISC*"
//! \sa MINIDUMP_MISC_INFO
//! \sa MINIDUMP_MISC_INFO_2
//! \sa MINIDUMP_MISC_INFO_3
//! \sa MINIDUMP_MISC_INFO_4
//! \sa MINIDUMP_MISC_INFO_N
struct __attribute__((packed, aligned(4))) MINIDUMP_MISC_INFO_5
    : public MINIDUMP_MISC_INFO_4 {
  //! \brief Information about XSAVE-managed state stored within CPU-specific
  //!     context structures.
  //!
  //! This information can be used to locate state components within
  //! CPU-specific context structures.
  XSTATE_CONFIG_FEATURE_MSC_INFO XStateData;

  uint32_t ProcessCookie;
};

//! \brief The latest known version of the MINIDUMP_MISC_INFO structure.
typedef MINIDUMP_MISC_INFO_5 MINIDUMP_MISC_INFO_N;

//! \brief Describes a region of memory.
struct __attribute__((packed, aligned(4))) MINIDUMP_MEMORY_INFO {
  //! \brief The base address of the region of pages.
  uint64_t BaseAddress;

  //! \brief The base address of a range of pages in this region. The page is
  //!     contained within this memory region.
  uint64_t AllocationBase;

  //! \brief The memory protection when the region was initially allocated. This
  //!     member can be one of the memory protection options (such as
  //!     \ref PAGE_x "PAGE_EXECUTE", \ref PAGE_x "PAGE_NOACCESS", etc.), along
  //!     with \ref PAGE_x "PAGE_GUARD" or \ref PAGE_x "PAGE_NOCACHE", as
  //!     needed.
  uint32_t AllocationProtect;

  uint32_t __alignment1;

  //! \brief The size of the region beginning at the base address in which all
  //!     pages have identical attributes, in bytes.
  uint64_t RegionSize;

  //! \brief The state of the pages in the region. This can be one of
  //!     \ref MEM_x "MEM_COMMIT", \ref MEM_x "MEM_FREE", or \ref MEM_x
  //!     "MEM_RESERVE".
  uint32_t State;

  //! \brief The access protection of the pages in the region. This member is
  //!     one of the values listed for the #AllocationProtect member.
  uint32_t Protect;

  //! \brief The type of pages in the region. This can be one of \ref MEM_x
  //!     "MEM_IMAGE", \ref MEM_x "MEM_MAPPED", or \ref MEM_x "MEM_PRIVATE".
  uint32_t Type;

  uint32_t __alignment2;
};

//! \brief Contains a list of memory regions.
struct __attribute__((packed, aligned(4))) MINIDUMP_MEMORY_INFO_LIST {
  //! \brief The size of the header data for the stream, in bytes. This is
  //!     generally sizeof(MINIDUMP_MEMORY_INFO_LIST).
  uint32_t SizeOfHeader;

  //! \brief The size of each entry following the header, in bytes. This is
  //!     generally sizeof(MINIDUMP_MEMORY_INFO).
  uint32_t SizeOfEntry;

  //! \brief The number of entries in the stream. These are generally
  //!     MINIDUMP_MEMORY_INFO structures. The entries follow the header.
  uint64_t NumberOfEntries;
};

//! \brief Minidump file type values for MINIDUMP_HEADER::Flags. These bits
//!     describe the types of data carried within a minidump file.
enum MINIDUMP_TYPE {
  //! \brief A minidump file without any additional data.
  //!
  //! This type of minidump file contains:
  //!  - A MINIDUMP_SYSTEM_INFO stream.
  //!  - A MINIDUMP_MISC_INFO, MINIDUMP_MISC_INFO_2, MINIDUMP_MISC_INFO_3, or
  //!    MINIDUMP_MISC_INFO_4 stream, depending on which fields are present.
  //!  - A MINIDUMP_THREAD_LIST stream. All threads are present, along with a
  //!    snapshot of each thread’s stack memory sufficient to obtain backtraces.
  //!  - If the minidump file was generated as a result of an exception, a
  //!    MINIDUMP_EXCEPTION_STREAM describing the exception.
  //!  - A MINIDUMP_MODULE_LIST stream. All loaded modules are present.
  //!  - Typically, a MINIDUMP_MEMORY_LIST stream containing duplicate pointers
  //!    to the stack memory regions also referenced by the MINIDUMP_THREAD_LIST
  //!    stream. This type of minidump file also includes a
  //!    MINIDUMP_MEMORY_DESCRIPTOR containing the 256 bytes centered around
  //!    the exception address or the instruction pointer.
  MiniDumpNormal = 0x00000000,
};

#endif  // CRASHPAD_COMPAT_NON_WIN_DBGHELP_H_
