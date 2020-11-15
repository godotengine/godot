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

// This file corresponds to the system’s <mach-o/loader.h>.
//
// This file is intended to be included multiple times in the same translation
// unit, so #include guards are intentionally absent.
//
// This file is included by snapshot/mac/process_types.h and
// snapshot/mac/process_types.cc to produce process type struct definitions and
// accessors.

PROCESS_TYPE_STRUCT_BEGIN(mach_header)  // 64-bit: mach_header_64
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, magic)

  // cputype is really cpu_type_t, a typedef for integer_t, itself a typedef for
  // int. It is currently reasonable to assume that int is int32_t.
  PROCESS_TYPE_STRUCT_MEMBER(int32_t, cputype)

  // cpusubtype is really cpu_subtype_t, a typedef for integer_t, itself a
  // typedef for int. It is currently reasonable to assume that int is int32_t.
  PROCESS_TYPE_STRUCT_MEMBER(int32_t, cpusubtype)

  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, filetype)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, ncmds)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, sizeofcmds)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, flags)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved32_64Only, reserved)
PROCESS_TYPE_STRUCT_END(mach_header)

PROCESS_TYPE_STRUCT_BEGIN(load_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)
PROCESS_TYPE_STRUCT_END(load_command)

PROCESS_TYPE_STRUCT_BEGIN(segment_command)  // 64-bit: segment_command_64
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)

  // This string is not necessarily NUL-terminated.
  PROCESS_TYPE_STRUCT_MEMBER(char, segname, [16])

  PROCESS_TYPE_STRUCT_MEMBER(ULong, vmaddr)
  PROCESS_TYPE_STRUCT_MEMBER(ULong, vmsize)
  PROCESS_TYPE_STRUCT_MEMBER(ULong, fileoff)
  PROCESS_TYPE_STRUCT_MEMBER(ULong, filesize)
  PROCESS_TYPE_STRUCT_MEMBER(vm_prot_t, maxprot)
  PROCESS_TYPE_STRUCT_MEMBER(vm_prot_t, initprot)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nsects)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, flags)
PROCESS_TYPE_STRUCT_END(segment_command)

PROCESS_TYPE_STRUCT_BEGIN(dylib_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)

  // The following come from the dylib struct member of dylib_command.
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, dylib_name)  // lc_str
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, dylib_timestamp)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, dylib_current_version)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, dylib_compatibility_version)
PROCESS_TYPE_STRUCT_END(dylib_command)

PROCESS_TYPE_STRUCT_BEGIN(dylinker_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, name)  // lc_str
PROCESS_TYPE_STRUCT_END(dylinker_command)

PROCESS_TYPE_STRUCT_BEGIN(symtab_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, symoff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nsyms)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, stroff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, strsize)
PROCESS_TYPE_STRUCT_END(symtab_command)

PROCESS_TYPE_STRUCT_BEGIN(dysymtab_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, ilocalsym)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nlocalsym)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, iextdefsym)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nextdefsym)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, iundefsym)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nundefsym)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, tocoff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, ntoc)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, modtaboff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nmodtab)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, extrefsymoff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nextrefsyms)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, indirectsymoff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nindirectsyms)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, extreloff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nextrel)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, locreloff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nlocrel)
PROCESS_TYPE_STRUCT_END(dysymtab_command)

PROCESS_TYPE_STRUCT_BEGIN(uuid_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, uuid, [16])
PROCESS_TYPE_STRUCT_END(uuid_command)

PROCESS_TYPE_STRUCT_BEGIN(source_version_command)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmd)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, cmdsize)
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, version)
PROCESS_TYPE_STRUCT_END(source_version_command)

PROCESS_TYPE_STRUCT_BEGIN(section)  // 64-bit: section_64
  // These strings are not necessarily NUL-terminated.
  PROCESS_TYPE_STRUCT_MEMBER(char, sectname, [16])
  PROCESS_TYPE_STRUCT_MEMBER(char, segname, [16])

  PROCESS_TYPE_STRUCT_MEMBER(ULong, addr)
  PROCESS_TYPE_STRUCT_MEMBER(ULong, size)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, offset)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, align)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, reloff)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, nreloc)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, flags)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, reserved1)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, reserved2)
  PROCESS_TYPE_STRUCT_MEMBER(Reserved32_64Only, reserved3)
PROCESS_TYPE_STRUCT_END(section)
