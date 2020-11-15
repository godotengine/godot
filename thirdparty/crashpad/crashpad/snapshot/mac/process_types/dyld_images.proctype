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

// This file corresponds to the system’s <mach-o/dyld_images.h>.
//
// This file is intended to be included multiple times in the same translation
// unit, so #include guards are intentionally absent.
//
// This file is included by snapshot/mac/process_types.h and
// snapshot/mac/process_types.cc to produce process type struct definitions and
// accessors.

PROCESS_TYPE_STRUCT_BEGIN(dyld_image_info)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, imageLoadAddress)  // const mach_header*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, imageFilePath)  // const char*
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, imageFileModDate)
PROCESS_TYPE_STRUCT_END(dyld_image_info)

PROCESS_TYPE_STRUCT_BEGIN(dyld_uuid_info)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, imageLoadAddress)  // const mach_header*
  PROCESS_TYPE_STRUCT_MEMBER(uuid_t, imageUUID)
PROCESS_TYPE_STRUCT_END(dyld_uuid_info)

// dyld_all_image_infos is variable-length. Its length dictated by its |version|
// field which is always present. A custom implementation of the flavored
// ReadSpecificInto function that understands how to map this field to the
// structure’s actual size is provided in snapshot/mac/process_types/custom.cc.
// No implementation of ReadArrayInto is provided because dyld_all_image_infos
// structs are singletons in a process and are never present in arrays, so the
// functionality is unnecessary.

#if !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO) && \
    !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY)

PROCESS_TYPE_STRUCT_BEGIN(dyld_all_image_infos)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, version)
  PROCESS_TYPE_STRUCT_VERSIONED(dyld_all_image_infos, version)

  // Version 1 (Mac OS X 10.4)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, infoArrayCount)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, infoArray)  // const dyld_image_info*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, notification)  // function pointer
  PROCESS_TYPE_STRUCT_MEMBER(bool, processDetachedFromSharedRegion)

  // Version 2 (Mac OS X 10.6)
  PROCESS_TYPE_STRUCT_MEMBER(bool, libSystemInitialized)

  // This field does not appear in the system’s structure definition but is
  // necessary to ensure that when building in 32-bit mode, the 64-bit version
  // of the process_types structure matches the genuine 64-bit structure. This
  // is required because the alignment constraints on 64-bit types are more
  // stringent in 64-bit mode.
  PROCESS_TYPE_STRUCT_MEMBER(Reserved32_64Only, alignment)

  // const mach_header*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, dyldImageLoadAddress)

  // Version 3 (Mac OS X 10.6)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, jitInfo)  // void*

  // Version 5 (Mac OS X 10.6)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, dyldVersion)  // const char*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, errorMessage)  // const char*
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, terminationFlags)

  // Version 6 (Mac OS X 10.6)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, coreSymbolicationShmPage)  // void*

  // Version 7 (Mac OS X 10.6)
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, systemOrderFlag)

  // Version 8 (OS X 10.7)
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, uuidArrayCount)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, uuidArray)  // const dyld_uuid_info*

  // Version 9 (OS X 10.7)
  // dyld_all_image_infos*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, dyldAllImageInfosAddress)

  // Version 10 (OS X 10.7)
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, initialImageCount)

  // Version 11 (OS X 10.7)
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, errorKind)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, errorClientOfDylibPath)  // const char*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, errorTargetDylibPath)  // const char*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, errorSymbol)  // const char*

  // Version 12 (OS X 10.7)
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, sharedCacheSlide)

  // Version 13 (OS X 10.9)
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, sharedCacheUUID, [16])

  // Version 15 (macOS 10.12)
  // This space is also allocated in version 14 (OS X 10.9) as part of the
  // “reserved” member.
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, sharedCacheBaseAddress)
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, infoArrayChangeTimestamp)
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, dyldPath)  // const char*

  // These should be considered mach_port_name_t when interacting with them from
  // another Mach IPC namespace (process).
  PROCESS_TYPE_STRUCT_MEMBER(mach_port_t, notifyPorts, [8])

  // Version 14 (OS X 10.9)
  // As of the 10.12 SDK, this is declared as reserved[9] for 64-bit platforms
  // and reserved[4] for 32-bit platforms. It was expanded to reserved[5] for
  // 32-bit platforms in the 10.13 SDK to provide proper padding, but because
  // the runtimes that use versions 14 and 15 were built with SDKs that did not
  // have this extra padding, it’s necessary to treat the element at index 4 on
  // 32-bit systems as outside of the version 14 and 15 structure. This is why
  // |reserved| is only declared a 4-element array, with a special end_14_15
  // member (not present in the native definition) available to indicate the
  // end of the native version 14 and 15 structure, preceding the padding in the
  // 32-bit structure that would natively be addressed at index 4 of |reserved|.
  // Treat reserved_4_32 as only available in version 16 of the structure.
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, reserved, [4])
  PROCESS_TYPE_STRUCT_MEMBER(Reserved64_64Only, reserved_4_64)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved64_64Only, reserved_5)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved64_64Only, reserved_6)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved64_64Only, reserved_7)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved64_64Only, reserved_8)
  PROCESS_TYPE_STRUCT_MEMBER(Nothing, end_14_15)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved32_32Only, reserved_4_32)

  // Version 16 (macOS 10.13)
  PROCESS_TYPE_STRUCT_MEMBER(UIntPtr, compact_dyld_image_info_addr)
  PROCESS_TYPE_STRUCT_MEMBER(ULong, compact_dyld_image_info_size)  // size_t
PROCESS_TYPE_STRUCT_END(dyld_all_image_infos)

#endif  // ! PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO &&
        // ! PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY
