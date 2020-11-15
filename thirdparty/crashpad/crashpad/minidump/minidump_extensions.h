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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_EXTENSIONS_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_EXTENSIONS_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <winnt.h>

#include "base/compiler_specific.h"
#include "build/build_config.h"
#include "util/misc/pdb_structures.h"
#include "util/misc/uuid.h"

// C4200 is "nonstandard extension used : zero-sized array in struct/union".
// We would like to globally disable this warning, but unfortunately, the
// compiler is buggy and only supports disabling it with a pragma, so we can't
// disable it with other silly warnings in build/common.gypi. See:
//   https://connect.microsoft.com/VisualStudio/feedback/details/1114440
MSVC_PUSH_DISABLE_WARNING(4200);

#if defined(COMPILER_MSVC)
#define PACKED
#pragma pack(push, 1)
#else
#define PACKED __attribute__((packed))
#endif  // COMPILER_MSVC

namespace crashpad {

//! \brief Minidump stream type values for MINIDUMP_DIRECTORY::StreamType. Each
//!     stream structure has a corresponding stream type value to identify it.
//!
//! \sa MINIDUMP_STREAM_TYPE
enum MinidumpStreamType : uint32_t {
  //! \brief The stream type for MINIDUMP_THREAD_LIST.
  //!
  //! \sa ThreadListStream
  kMinidumpStreamTypeThreadList = ThreadListStream,

  //! \brief The stream type for MINIDUMP_MODULE_LIST.
  //!
  //! \sa ModuleListStream
  kMinidumpStreamTypeModuleList = ModuleListStream,

  //! \brief The stream type for MINIDUMP_MEMORY_LIST.
  //!
  //! \sa MemoryListStream
  kMinidumpStreamTypeMemoryList = MemoryListStream,

  //! \brief The stream type for MINIDUMP_EXCEPTION_STREAM.
  //!
  //! \sa ExceptionStream
  kMinidumpStreamTypeException = ExceptionStream,

  //! \brief The stream type for MINIDUMP_SYSTEM_INFO.
  //!
  //! \sa SystemInfoStream
  kMinidumpStreamTypeSystemInfo = SystemInfoStream,

  //! \brief The stream type for MINIDUMP_HANDLE_DATA_STREAM.
  //!
  //! \sa HandleDataStream
  kMinidumpStreamTypeHandleData = HandleDataStream,

  //! \brief The stream type for MINIDUMP_UNLOADED_MODULE_LIST.
  //!
  //! \sa UnloadedModuleListStream
  kMinidumpStreamTypeUnloadedModuleList = UnloadedModuleListStream,

  //! \brief The stream type for MINIDUMP_MISC_INFO, MINIDUMP_MISC_INFO_2,
  //!     MINIDUMP_MISC_INFO_3, and MINIDUMP_MISC_INFO_4.
  //!
  //! \sa MiscInfoStream
  kMinidumpStreamTypeMiscInfo = MiscInfoStream,

  //! \brief The stream type for MINIDUMP_MEMORY_INFO_LIST.
  //!
  //! \sa MemoryInfoListStream
  kMinidumpStreamTypeMemoryInfoList = MemoryInfoListStream,

  // 0x4350 = "CP"

  //! \brief The stream type for MinidumpCrashpadInfo.
  kMinidumpStreamTypeCrashpadInfo = 0x43500001,
};

//! \brief A variable-length UTF-8-encoded string carried within a minidump
//!     file.
//!
//! \sa MINIDUMP_STRING
struct ALIGNAS(4) PACKED MinidumpUTF8String {
  // The field names do not conform to typical style, they match the names used
  // in MINIDUMP_STRING. This makes it easier to operate on MINIDUMP_STRING (for
  // UTF-16 strings) and MinidumpUTF8String using templates.

  //! \brief The length of the #Buffer field in bytes, not including the `NUL`
  //!     terminator.
  //!
  //! \note This field is interpreted as a byte count, not a count of Unicode
  //!     code points.
  uint32_t Length;

  //! \brief The string, encoded in UTF-8, and terminated with a `NUL` byte.
  uint8_t Buffer[0];
};

//! \brief A variable-length array of bytes carried within a minidump file.
//!     The data have no intrinsic type and should be interpreted according
//!     to their referencing context.
struct ALIGNAS(4) PACKED MinidumpByteArray {
  //! \brief The length of the #data field.
  uint32_t length;

  //! \brief The bytes of data.
  uint8_t data[0];
};

//! \brief CPU type values for MINIDUMP_SYSTEM_INFO::ProcessorArchitecture.
//!
//! \sa \ref PROCESSOR_ARCHITECTURE_x "PROCESSOR_ARCHITECTURE_*"
enum MinidumpCPUArchitecture : uint16_t {
  //! \brief 32-bit x86.
  //!
  //! These systems identify their CPUs generically as “x86” or “ia32”, or with
  //! more specific names such as “i386”, “i486”, “i586”, and “i686”.
  kMinidumpCPUArchitectureX86 = PROCESSOR_ARCHITECTURE_INTEL,

  kMinidumpCPUArchitectureMIPS = PROCESSOR_ARCHITECTURE_MIPS,
  kMinidumpCPUArchitectureAlpha = PROCESSOR_ARCHITECTURE_ALPHA,

  //! \brief 32-bit PowerPC.
  //!
  //! These systems identify their CPUs generically as “ppc”, or with more
  //! specific names such as “ppc6xx”, “ppc7xx”, and “ppc74xx”.
  kMinidumpCPUArchitecturePPC = PROCESSOR_ARCHITECTURE_PPC,

  kMinidumpCPUArchitectureSHx = PROCESSOR_ARCHITECTURE_SHX,

  //! \brief 32-bit ARM.
  //!
  //! These systems identify their CPUs generically as “arm”, or with more
  //! specific names such as “armv6” and “armv7”.
  kMinidumpCPUArchitectureARM = PROCESSOR_ARCHITECTURE_ARM,

  kMinidumpCPUArchitectureIA64 = PROCESSOR_ARCHITECTURE_IA64,
  kMinidumpCPUArchitectureAlpha64 = PROCESSOR_ARCHITECTURE_ALPHA64,
  kMinidumpCPUArchitectureMSIL = PROCESSOR_ARCHITECTURE_MSIL,

  //! \brief 64-bit x86.
  //!
  //! These systems identify their CPUs as “x86_64”, “amd64”, or “x64”.
  kMinidumpCPUArchitectureAMD64 = PROCESSOR_ARCHITECTURE_AMD64,

  //! \brief A 32-bit x86 process running on IA-64 (Itanium).
  //!
  //! \note This value is not used in minidump files for 32-bit x86 processes
  //!     running on a 64-bit-capable x86 CPU and operating system. In that
  //!     configuration, #kMinidumpCPUArchitectureX86 is used instead.
  kMinidumpCPUArchitectureX86Win64 = PROCESSOR_ARCHITECTURE_IA32_ON_WIN64,

  kMinidumpCPUArchitectureNeutral = PROCESSOR_ARCHITECTURE_NEUTRAL,

  //! \brief 64-bit ARM.
  //!
  //! These systems identify their CPUs generically as “arm64” or “aarch64”, or
  //! with more specific names such as “armv8”.
  //!
  //! \sa #kMinidumpCPUArchitectureARM64Breakpad
  kMinidumpCPUArchitectureARM64 = PROCESSOR_ARCHITECTURE_ARM64,

  kMinidumpCPUArchitectureARM32Win64 = PROCESSOR_ARCHITECTURE_ARM32_ON_WIN64,
  kMinidumpCPUArchitectureSPARC = 0x8001,

  //! \brief 64-bit PowerPC.
  //!
  //! These systems identify their CPUs generically as “ppc64”, or with more
  //! specific names such as “ppc970”.
  kMinidumpCPUArchitecturePPC64 = 0x8002,

  //! \brief Used by Breakpad for 64-bit ARM.
  //!
  //! \deprecated Use #kMinidumpCPUArchitectureARM64 instead.
  kMinidumpCPUArchitectureARM64Breakpad = 0x8003,

  //! \brief Unknown CPU architecture.
  kMinidumpCPUArchitectureUnknown = PROCESSOR_ARCHITECTURE_UNKNOWN,
};

//! \brief Operating system type values for MINIDUMP_SYSTEM_INFO::ProductType.
//!
//! \sa \ref VER_NT_x "VER_NT_*"
enum MinidumpOSType : uint8_t {
  //! \brief A “desktop” or “workstation” system.
  kMinidumpOSTypeWorkstation = VER_NT_WORKSTATION,

  //! \brief A “domain controller” system. Windows-specific.
  kMinidumpOSTypeDomainController = VER_NT_DOMAIN_CONTROLLER,

  //! \brief A “server” system.
  kMinidumpOSTypeServer = VER_NT_SERVER,
};

//! \brief Operating system family values for MINIDUMP_SYSTEM_INFO::PlatformId.
//!
//! \sa \ref VER_PLATFORM_x "VER_PLATFORM_*"
enum MinidumpOS : uint32_t {
  //! \brief Windows 3.1.
  kMinidumpOSWin32s = VER_PLATFORM_WIN32s,

  //! \brief Windows 95, Windows 98, and Windows Me.
  kMinidumpOSWin32Windows = VER_PLATFORM_WIN32_WINDOWS,

  //! \brief Windows NT, Windows 2000, and later.
  kMinidumpOSWin32NT = VER_PLATFORM_WIN32_NT,

  kMinidumpOSUnix = 0x8000,

  //! \brief macOS, Darwin for traditional systems.
  kMinidumpOSMacOSX = 0x8101,

  //! \brief iOS, Darwin for mobile devices.
  kMinidumpOSiOS = 0x8102,

  //! \brief Linux, not including Android.
  kMinidumpOSLinux = 0x8201,

  kMinidumpOSSolaris = 0x8202,

  //! \brief Android.
  kMinidumpOSAndroid = 0x8203,

  kMinidumpOSPS3 = 0x8204,

  //! \brief Native Client (NaCl).
  kMinidumpOSNaCl = 0x8205,

  //! \brief Fuchsia.
  kMinidumpOSFuchsia = 0x8206,

  //! \brief Unknown operating system.
  kMinidumpOSUnknown = 0xffffffff,
};


//! \brief A list of ::RVA pointers.
struct ALIGNAS(4) PACKED MinidumpRVAList {
  //! \brief The number of children present in the #children array.
  uint32_t count;

  //! \brief Pointers to other structures in the minidump file.
  RVA children[0];
};

//! \brief A key-value pair.
struct ALIGNAS(4) PACKED MinidumpSimpleStringDictionaryEntry {
  //! \brief ::RVA of a MinidumpUTF8String containing the key of a key-value
  //!     pair.
  RVA key;

  //! \brief ::RVA of a MinidumpUTF8String containing the value of a key-value
  //!     pair.
  RVA value;
};

//! \brief A list of key-value pairs.
struct ALIGNAS(4) PACKED MinidumpSimpleStringDictionary {
  //! \brief The number of key-value pairs present.
  uint32_t count;

  //! \brief A list of MinidumpSimpleStringDictionaryEntry entries.
  MinidumpSimpleStringDictionaryEntry entries[0];
};

//! \brief A typed annotation object.
struct ALIGNAS(4) PACKED MinidumpAnnotation {
  //! \brief ::RVA of a MinidumpUTF8String containing the name of the
  //!     annotation.
  RVA name;

  //! \brief The type of data stored in the \a value of the annotation. This
  //!     may correspond to an \a Annotation::Type or it may be user-defined.
  uint16_t type;

  //! \brief This field is always `0`.
  uint16_t reserved;

  //! \brief ::RVA of a MinidumpByteArray to the data for the annotation.
  RVA value;
};

//! \brief A list of annotation objects.
struct ALIGNAS(4) PACKED MinidumpAnnotationList {
  //! \brief The number of annotation objects present.
  uint32_t count;

  //! \brief A list of MinidumpAnnotation objects.
  MinidumpAnnotation objects[0];
};

//! \brief Additional Crashpad-specific information about a module carried
//!     within a minidump file.
//!
//! This structure augments the information provided by MINIDUMP_MODULE. The
//! minidump file must contain a module list stream
//! (::kMinidumpStreamTypeModuleList) in order for this structure to appear.
//!
//! This structure is versioned. When changing this structure, leave the
//! existing structure intact so that earlier parsers will be able to understand
//! the fields they are aware of, and make additions at the end of the
//! structure. Revise #kVersion and document each field’s validity based on
//! #version, so that newer parsers will be able to determine whether the added
//! fields are valid or not.
//!
//! \sa MinidumpModuleCrashpadInfoList
struct ALIGNAS(4) PACKED MinidumpModuleCrashpadInfo {
  //! \brief The structure’s currently-defined version number.
  //!
  //! \sa version
  static constexpr uint32_t kVersion = 1;

  //! \brief The structure’s version number.
  //!
  //! Readers can use this field to determine which other fields in the
  //! structure are valid. Upon encountering a value greater than #kVersion, a
  //! reader should assume that the structure’s layout is compatible with the
  //! structure defined as having value #kVersion.
  //!
  //! Writers may produce values less than #kVersion in this field if there is
  //! no need for any fields present in later versions.
  uint32_t version;

  //! \brief A MinidumpRVAList pointing to MinidumpUTF8String objects. The
  //!     module controls the data that appears here.
  //!
  //! These strings correspond to ModuleSnapshot::AnnotationsVector() and do not
  //! duplicate anything in #simple_annotations or #annotation_objects.
  //!
  //! This field is present when #version is at least `1`.
  MINIDUMP_LOCATION_DESCRIPTOR list_annotations;

  //! \brief A MinidumpSimpleStringDictionary pointing to strings interpreted as
  //!     key-value pairs. The module controls the data that appears here.
  //!
  //! These key-value pairs correspond to
  //! ModuleSnapshot::AnnotationsSimpleMap() and do not duplicate anything in
  //! #list_annotations or #annotation_objects.
  //!
  //! This field is present when #version is at least `1`.
  MINIDUMP_LOCATION_DESCRIPTOR simple_annotations;

  //! \brief A MinidumpAnnotationList object containing the annotation objects
  //!     stored within the module. The module controls the data that appears
  //!     here.
  //!
  //! These key-value pairs correspond to ModuleSnapshot::AnnotationObjects()
  //! and do not duplicate anything in #list_annotations or #simple_annotations.
  //!
  //! This field may be present when #version is at least `1`.
  MINIDUMP_LOCATION_DESCRIPTOR annotation_objects;
};

//! \brief A link between a MINIDUMP_MODULE structure and additional
//!     Crashpad-specific information about a module carried within a minidump
//!     file.
struct ALIGNAS(4) PACKED MinidumpModuleCrashpadInfoLink {
  //! \brief A link to a MINIDUMP_MODULE structure in the module list stream.
  //!
  //! This field is an index into MINIDUMP_MODULE_LIST::Modules. This field’s
  //! value must be in the range of MINIDUMP_MODULE_LIST::NumberOfEntries.
  uint32_t minidump_module_list_index;

  //! \brief A link to a MinidumpModuleCrashpadInfo structure.
  //!
  //! MinidumpModuleCrashpadInfo structures are accessed indirectly through
  //! MINIDUMP_LOCATION_DESCRIPTOR pointers to allow for future growth of the
  //! MinidumpModuleCrashpadInfo structure.
  MINIDUMP_LOCATION_DESCRIPTOR location;
};

//! \brief Additional Crashpad-specific information about modules carried within
//!     a minidump file.
//!
//! This structure augments the information provided by
//! MINIDUMP_MODULE_LIST. The minidump file must contain a module list stream
//! (::kMinidumpStreamTypeModuleList) in order for this structure to appear.
//!
//! MinidumpModuleCrashpadInfoList::count may be less than the value of
//! MINIDUMP_MODULE_LIST::NumberOfModules because not every MINIDUMP_MODULE
//! structure carried within the minidump file will necessarily have
//! Crashpad-specific information provided by a MinidumpModuleCrashpadInfo
//! structure.
struct ALIGNAS(4) PACKED MinidumpModuleCrashpadInfoList {
  //! \brief The number of children present in the #modules array.
  uint32_t count;

  //! \brief Crashpad-specific information about modules, along with links to
  //!     MINIDUMP_MODULE structures that contain module information
  //!     traditionally carried within minidump files.
  MinidumpModuleCrashpadInfoLink modules[0];
};

//! \brief Additional Crashpad-specific information carried within a minidump
//!     file.
//!
//! This structure is versioned. When changing this structure, leave the
//! existing structure intact so that earlier parsers will be able to understand
//! the fields they are aware of, and make additions at the end of the
//! structure. Revise #kVersion and document each field’s validity based on
//! #version, so that newer parsers will be able to determine whether the added
//! fields are valid or not.
struct ALIGNAS(4) PACKED MinidumpCrashpadInfo {
  // UUID has a constructor, which makes it non-POD, which makes this structure
  // non-POD. In order for the default constructor to zero-initialize other
  // members, an explicit constructor must be provided.
  MinidumpCrashpadInfo()
      : version(),
        report_id(),
        client_id(),
        simple_annotations(),
        module_list() {
  }

  //! \brief The structure’s currently-defined version number.
  //!
  //! \sa version
  static constexpr uint32_t kVersion = 1;

  //! \brief The structure’s version number.
  //!
  //! Readers can use this field to determine which other fields in the
  //! structure are valid. Upon encountering a value greater than #kVersion, a
  //! reader should assume that the structure’s layout is compatible with the
  //! structure defined as having value #kVersion.
  //!
  //! Writers may produce values less than #kVersion in this field if there is
  //! no need for any fields present in later versions.
  uint32_t version;

  //! \brief A %UUID identifying an individual crash report.
  //!
  //! This provides a stable identifier for a crash even as the report is
  //! converted to different formats, provided that all formats support storing
  //! a crash report ID.
  //!
  //! If no identifier is available, this field will contain zeroes.
  //!
  //! This field is present when #version is at least `1`.
  UUID report_id;

  //! \brief A %UUID identifying the client that crashed.
  //!
  //! Client identification is within the scope of the application, but it is
  //! expected that the identifier will be unique for an instance of Crashpad
  //! monitoring an application or set of applications for a user. The
  //! identifier shall remain stable over time.
  //!
  //! If no identifier is available, this field will contain zeroes.
  //!
  //! This field is present when #version is at least `1`.
  UUID client_id;

  //! \brief A MinidumpSimpleStringDictionary pointing to strings interpreted as
  //!     key-value pairs.
  //!
  //! These key-value pairs correspond to
  //! ProcessSnapshot::AnnotationsSimpleMap().
  //!
  //! This field is present when #version is at least `1`.
  MINIDUMP_LOCATION_DESCRIPTOR simple_annotations;

  //! \brief A pointer to a MinidumpModuleCrashpadInfoList structure.
  //!
  //! This field is present when #version is at least `1`.
  MINIDUMP_LOCATION_DESCRIPTOR module_list;
};

#if defined(COMPILER_MSVC)
#pragma pack(pop)
#endif  // COMPILER_MSVC
#undef PACKED

MSVC_POP_WARNING();  // C4200

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_EXTENSIONS_H_
