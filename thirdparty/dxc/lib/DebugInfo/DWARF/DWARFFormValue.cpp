//===-- DWARFFormValue.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SyntaxHighlighting.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <climits>
using namespace llvm;
using namespace dwarf;
using namespace syntax;

namespace {
uint8_t getRefAddrSize(uint8_t AddrSize, uint16_t Version) {
  // FIXME: Support DWARF64.
  return (Version == 2) ? AddrSize : 4;
}

template <uint8_t AddrSize, uint8_t RefAddrSize>
ArrayRef<uint8_t> makeFixedFormSizesArrayRef() {
  static const uint8_t sizes[] = {
    0,           // 0x00 unused
    AddrSize,    // 0x01 DW_FORM_addr
    0,           // 0x02 unused
    0,           // 0x03 DW_FORM_block2
    0,           // 0x04 DW_FORM_block4
    2,           // 0x05 DW_FORM_data2
    4,           // 0x06 DW_FORM_data4
    8,           // 0x07 DW_FORM_data8
    0,           // 0x08 DW_FORM_string
    0,           // 0x09 DW_FORM_block
    0,           // 0x0a DW_FORM_block1
    1,           // 0x0b DW_FORM_data1
    1,           // 0x0c DW_FORM_flag
    0,           // 0x0d DW_FORM_sdata
    4,           // 0x0e DW_FORM_strp
    0,           // 0x0f DW_FORM_udata
    RefAddrSize, // 0x10 DW_FORM_ref_addr
    1,           // 0x11 DW_FORM_ref1
    2,           // 0x12 DW_FORM_ref2
    4,           // 0x13 DW_FORM_ref4
    8,           // 0x14 DW_FORM_ref8
    0,           // 0x15 DW_FORM_ref_udata
    0,           // 0x16 DW_FORM_indirect
    4,           // 0x17 DW_FORM_sec_offset
    0,           // 0x18 DW_FORM_exprloc
    0,           // 0x19 DW_FORM_flag_present
  };
  return makeArrayRef(sizes);
}
}

ArrayRef<uint8_t> DWARFFormValue::getFixedFormSizes(uint8_t AddrSize,
                                                    uint16_t Version) {
  uint8_t RefAddrSize = getRefAddrSize(AddrSize, Version);
  if (AddrSize == 4 && RefAddrSize == 4)
    return makeFixedFormSizesArrayRef<4, 4>();
  if (AddrSize == 4 && RefAddrSize == 8)
    return makeFixedFormSizesArrayRef<4, 8>();
  if (AddrSize == 8 && RefAddrSize == 4)
    return makeFixedFormSizesArrayRef<8, 4>();
  if (AddrSize == 8 && RefAddrSize == 8)
    return makeFixedFormSizesArrayRef<8, 8>();
  return None;
}

static const DWARFFormValue::FormClass DWARF4FormClasses[] = {
  DWARFFormValue::FC_Unknown,       // 0x0
  DWARFFormValue::FC_Address,       // 0x01 DW_FORM_addr
  DWARFFormValue::FC_Unknown,       // 0x02 unused
  DWARFFormValue::FC_Block,         // 0x03 DW_FORM_block2
  DWARFFormValue::FC_Block,         // 0x04 DW_FORM_block4
  DWARFFormValue::FC_Constant,      // 0x05 DW_FORM_data2
  // --- These can be FC_SectionOffset in DWARF3 and below:
  DWARFFormValue::FC_Constant,      // 0x06 DW_FORM_data4
  DWARFFormValue::FC_Constant,      // 0x07 DW_FORM_data8
  // ---
  DWARFFormValue::FC_String,        // 0x08 DW_FORM_string
  DWARFFormValue::FC_Block,         // 0x09 DW_FORM_block
  DWARFFormValue::FC_Block,         // 0x0a DW_FORM_block1
  DWARFFormValue::FC_Constant,      // 0x0b DW_FORM_data1
  DWARFFormValue::FC_Flag,          // 0x0c DW_FORM_flag
  DWARFFormValue::FC_Constant,      // 0x0d DW_FORM_sdata
  DWARFFormValue::FC_String,        // 0x0e DW_FORM_strp
  DWARFFormValue::FC_Constant,      // 0x0f DW_FORM_udata
  DWARFFormValue::FC_Reference,     // 0x10 DW_FORM_ref_addr
  DWARFFormValue::FC_Reference,     // 0x11 DW_FORM_ref1
  DWARFFormValue::FC_Reference,     // 0x12 DW_FORM_ref2
  DWARFFormValue::FC_Reference,     // 0x13 DW_FORM_ref4
  DWARFFormValue::FC_Reference,     // 0x14 DW_FORM_ref8
  DWARFFormValue::FC_Reference,     // 0x15 DW_FORM_ref_udata
  DWARFFormValue::FC_Indirect,      // 0x16 DW_FORM_indirect
  DWARFFormValue::FC_SectionOffset, // 0x17 DW_FORM_sec_offset
  DWARFFormValue::FC_Exprloc,       // 0x18 DW_FORM_exprloc
  DWARFFormValue::FC_Flag,          // 0x19 DW_FORM_flag_present
};

bool DWARFFormValue::isFormClass(DWARFFormValue::FormClass FC) const {
  // First, check DWARF4 form classes.
  if (Form < ArrayRef<FormClass>(DWARF4FormClasses).size() &&
      DWARF4FormClasses[Form] == FC)
    return true;
  // Check more forms from DWARF4 and DWARF5 proposals.
  switch (Form) {
  case DW_FORM_ref_sig8:
  case DW_FORM_GNU_ref_alt:
    return (FC == FC_Reference);
  case DW_FORM_GNU_addr_index:
    return (FC == FC_Address);
  case DW_FORM_GNU_str_index:
  case DW_FORM_GNU_strp_alt:
    return (FC == FC_String);
  }
  // In DWARF3 DW_FORM_data4 and DW_FORM_data8 served also as a section offset.
  // Don't check for DWARF version here, as some producers may still do this
  // by mistake.
  return (Form == DW_FORM_data4 || Form == DW_FORM_data8) &&
         FC == FC_SectionOffset;
}

bool DWARFFormValue::extractValue(DataExtractor data, uint32_t *offset_ptr,
                                  const DWARFUnit *cu) {
  bool indirect = false;
  bool is_block = false;
  Value.data = nullptr;
  // Read the value for the form into value and follow and DW_FORM_indirect
  // instances we run into
  do {
    indirect = false;
    switch (Form) {
    case DW_FORM_addr:
    case DW_FORM_ref_addr: {
      if (!cu)
        return false;
      uint16_t AddrSize =
          (Form == DW_FORM_addr)
              ? cu->getAddressByteSize()
              : getRefAddrSize(cu->getAddressByteSize(), cu->getVersion());
      RelocAddrMap::const_iterator AI = cu->getRelocMap()->find(*offset_ptr);
      if (AI != cu->getRelocMap()->end()) {
        const std::pair<uint8_t, int64_t> &R = AI->second;
        Value.uval = data.getUnsigned(offset_ptr, AddrSize) + R.second;
      } else
        Value.uval = data.getUnsigned(offset_ptr, AddrSize);
      break;
    }
    case DW_FORM_exprloc:
    case DW_FORM_block:
      Value.uval = data.getULEB128(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_block1:
      Value.uval = data.getU8(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_block2:
      Value.uval = data.getU16(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_block4:
      Value.uval = data.getU32(offset_ptr);
      is_block = true;
      break;
    case DW_FORM_data1:
    case DW_FORM_ref1:
    case DW_FORM_flag:
      Value.uval = data.getU8(offset_ptr);
      break;
    case DW_FORM_data2:
    case DW_FORM_ref2:
      Value.uval = data.getU16(offset_ptr);
      break;
    case DW_FORM_data4:
    case DW_FORM_ref4: {
      Value.uval = data.getU32(offset_ptr);
      if (!cu)
        break;
      RelocAddrMap::const_iterator AI = cu->getRelocMap()->find(*offset_ptr-4);
      if (AI != cu->getRelocMap()->end())
        Value.uval += AI->second.second;
      break;
    }
    case DW_FORM_data8:
    case DW_FORM_ref8:
      Value.uval = data.getU64(offset_ptr);
      break;
    case DW_FORM_sdata:
      Value.sval = data.getSLEB128(offset_ptr);
      break;
    case DW_FORM_udata:
    case DW_FORM_ref_udata:
      Value.uval = data.getULEB128(offset_ptr);
      break;
    case DW_FORM_string:
      Value.cstr = data.getCStr(offset_ptr);
      break;
    case DW_FORM_indirect:
      Form = data.getULEB128(offset_ptr);
      indirect = true;
      break;
    case DW_FORM_sec_offset:
    case DW_FORM_strp:
    case DW_FORM_GNU_ref_alt:
    case DW_FORM_GNU_strp_alt: {
      // FIXME: This is 64-bit for DWARF64.
      Value.uval = data.getU32(offset_ptr);
      if (!cu)
        break;
      RelocAddrMap::const_iterator AI =
          cu->getRelocMap()->find(*offset_ptr - 4);
      if (AI != cu->getRelocMap()->end())
        Value.uval += AI->second.second;
      break;
    }
    case DW_FORM_flag_present:
      Value.uval = 1;
      break;
    case DW_FORM_ref_sig8:
      Value.uval = data.getU64(offset_ptr);
      break;
    case DW_FORM_GNU_addr_index:
    case DW_FORM_GNU_str_index:
      Value.uval = data.getULEB128(offset_ptr);
      break;
    default:
      return false;
    }
  } while (indirect);

  if (is_block) {
    StringRef str = data.getData().substr(*offset_ptr, Value.uval);
    Value.data = nullptr;
    if (!str.empty()) {
      Value.data = reinterpret_cast<const uint8_t *>(str.data());
      *offset_ptr += Value.uval;
    }
  }

  return true;
}

bool
DWARFFormValue::skipValue(DataExtractor debug_info_data, uint32_t* offset_ptr,
                          const DWARFUnit *cu) const {
  return DWARFFormValue::skipValue(Form, debug_info_data, offset_ptr, cu);
}

bool
DWARFFormValue::skipValue(uint16_t form, DataExtractor debug_info_data,
                          uint32_t *offset_ptr, const DWARFUnit *cu) {
  bool indirect = false;
  do {
    switch (form) {
    // Blocks if inlined data that have a length field and the data bytes
    // inlined in the .debug_info
    case DW_FORM_exprloc:
    case DW_FORM_block: {
      uint64_t size = debug_info_data.getULEB128(offset_ptr);
      *offset_ptr += size;
      return true;
    }
    case DW_FORM_block1: {
      uint8_t size = debug_info_data.getU8(offset_ptr);
      *offset_ptr += size;
      return true;
    }
    case DW_FORM_block2: {
      uint16_t size = debug_info_data.getU16(offset_ptr);
      *offset_ptr += size;
      return true;
    }
    case DW_FORM_block4: {
      uint32_t size = debug_info_data.getU32(offset_ptr);
      *offset_ptr += size;
      return true;
    }

    // Inlined NULL terminated C-strings
    case DW_FORM_string:
      debug_info_data.getCStr(offset_ptr);
      return true;

    // Compile unit address sized values
    case DW_FORM_addr:
      *offset_ptr += cu->getAddressByteSize();
      return true;
    case DW_FORM_ref_addr:
      *offset_ptr += getRefAddrSize(cu->getAddressByteSize(), cu->getVersion());
      return true;

    // 0 byte values - implied from the form.
    case DW_FORM_flag_present:
      return true;

    // 1 byte values
    case DW_FORM_data1:
    case DW_FORM_flag:
    case DW_FORM_ref1:
      *offset_ptr += 1;
      return true;

    // 2 byte values
    case DW_FORM_data2:
    case DW_FORM_ref2:
      *offset_ptr += 2;
      return true;

    // 4 byte values
    case DW_FORM_data4:
    case DW_FORM_ref4:
      *offset_ptr += 4;
      return true;

    // 8 byte values
    case DW_FORM_data8:
    case DW_FORM_ref8:
    case DW_FORM_ref_sig8:
      *offset_ptr += 8;
      return true;

    // signed or unsigned LEB 128 values
    //  case DW_FORM_APPLE_db_str:
    case DW_FORM_sdata:
    case DW_FORM_udata:
    case DW_FORM_ref_udata:
    case DW_FORM_GNU_str_index:
    case DW_FORM_GNU_addr_index:
      debug_info_data.getULEB128(offset_ptr);
      return true;

    case DW_FORM_indirect:
      indirect = true;
      form = debug_info_data.getULEB128(offset_ptr);
      break;

    // FIXME: 4 for DWARF32, 8 for DWARF64.
    case DW_FORM_sec_offset:
    case DW_FORM_strp:
    case DW_FORM_GNU_ref_alt:
    case DW_FORM_GNU_strp_alt:
      *offset_ptr += 4;
      return true;

    default:
      return false;
    }
  } while (indirect);
  return true;
}

void
DWARFFormValue::dump(raw_ostream &OS, const DWARFUnit *cu) const {
  uint64_t uvalue = Value.uval;
  bool cu_relative_offset = false;

  switch (Form) {
  case DW_FORM_addr:      OS << format("0x%016" PRIx64, uvalue); break;
  case DW_FORM_GNU_addr_index: {
    OS << format(" indexed (%8.8x) address = ", (uint32_t)uvalue);
    uint64_t Address;
    if (cu->getAddrOffsetSectionItem(uvalue, Address))
      OS << format("0x%016" PRIx64, Address);
    else
      OS << "<no .debug_addr section>";
    break;
  }
  case DW_FORM_flag_present: OS << "true"; break;
  case DW_FORM_flag:
  case DW_FORM_data1:     OS << format("0x%02x", (uint8_t)uvalue); break;
  case DW_FORM_data2:     OS << format("0x%04x", (uint16_t)uvalue); break;
  case DW_FORM_data4:     OS << format("0x%08x", (uint32_t)uvalue); break;
  case DW_FORM_ref_sig8:
  case DW_FORM_data8:     OS << format("0x%016" PRIx64, uvalue); break;
  case DW_FORM_string:
    OS << '"';
    OS.write_escaped(Value.cstr);
    OS << '"';
    break;
  case DW_FORM_exprloc:
  case DW_FORM_block:
  case DW_FORM_block1:
  case DW_FORM_block2:
  case DW_FORM_block4:
    if (uvalue > 0) {
      switch (Form) {
      case DW_FORM_exprloc:
      case DW_FORM_block:  OS << format("<0x%" PRIx64 "> ", uvalue);     break;
      case DW_FORM_block1: OS << format("<0x%2.2x> ", (uint8_t)uvalue);  break;
      case DW_FORM_block2: OS << format("<0x%4.4x> ", (uint16_t)uvalue); break;
      case DW_FORM_block4: OS << format("<0x%8.8x> ", (uint32_t)uvalue); break;
      default: break;
      }

      const uint8_t* data_ptr = Value.data;
      if (data_ptr) {
        // uvalue contains size of block
        const uint8_t* end_data_ptr = data_ptr + uvalue;
        while (data_ptr < end_data_ptr) {
          OS << format("%2.2x ", *data_ptr);
          ++data_ptr;
        }
      }
      else
        OS << "NULL";
    }
    break;

  case DW_FORM_sdata:     OS << Value.sval; break;
  case DW_FORM_udata:     OS << Value.uval; break;
  case DW_FORM_strp: {
    OS << format(" .debug_str[0x%8.8x] = ", (uint32_t)uvalue);
    dumpString(OS, cu);
    break;
  }
  case DW_FORM_GNU_str_index: {
    OS << format(" indexed (%8.8x) string = ", (uint32_t)uvalue);
    dumpString(OS, cu);
    break;
  }
  case DW_FORM_GNU_strp_alt: {
    OS << format("alt indirect string, offset: 0x%" PRIx64 "", uvalue);
    dumpString(OS, cu);
    break;
  }
  case DW_FORM_ref_addr:
    OS << format("0x%016" PRIx64, uvalue);
    break;
  case DW_FORM_ref1:
    cu_relative_offset = true;
    OS << format("cu + 0x%2.2x", (uint8_t)uvalue);
    break;
  case DW_FORM_ref2:
    cu_relative_offset = true;
    OS << format("cu + 0x%4.4x", (uint16_t)uvalue);
    break;
  case DW_FORM_ref4:
    cu_relative_offset = true;
    OS << format("cu + 0x%4.4x", (uint32_t)uvalue);
    break;
  case DW_FORM_ref8:
    cu_relative_offset = true;
    OS << format("cu + 0x%8.8" PRIx64, uvalue);
    break;
  case DW_FORM_ref_udata:
    cu_relative_offset = true;
    OS << format("cu + 0x%" PRIx64, uvalue);
    break;
  case DW_FORM_GNU_ref_alt:
    OS << format("<alt 0x%" PRIx64 ">", uvalue);
    break;

    // All DW_FORM_indirect attributes should be resolved prior to calling
    // this function
  case DW_FORM_indirect:
    OS << "DW_FORM_indirect";
    break;

    // Should be formatted to 64-bit for DWARF64.
  case DW_FORM_sec_offset:
    OS << format("0x%08x", (uint32_t)uvalue);
    break;

  default:
    OS << format("DW_FORM(0x%4.4x)", Form);
    break;
  }

  if (cu_relative_offset) {
    OS << " => {";
    WithColor(OS, syntax::Address).get()
      << format("0x%8.8" PRIx64, uvalue + (cu ? cu->getOffset() : 0));
    OS << "}";
  }
}

void DWARFFormValue::dumpString(raw_ostream &OS, const DWARFUnit *U) const {
  Optional<const char *> DbgStr = getAsCString(U);
  if (DbgStr.hasValue()) {
    raw_ostream &COS = WithColor(OS, syntax::String);
    COS << '"';
    COS.write_escaped(DbgStr.getValue());
    COS << '"';
  }
}

Optional<const char *> DWARFFormValue::getAsCString(const DWARFUnit *U) const {
  if (!isFormClass(FC_String))
    return None;
  if (Form == DW_FORM_string)
    return Value.cstr;
  // FIXME: Add support for DW_FORM_GNU_strp_alt
  if (Form == DW_FORM_GNU_strp_alt || U == nullptr)
    return None;
  uint32_t Offset = Value.uval;
  if (Form == DW_FORM_GNU_str_index) {
    uint32_t StrOffset;
    if (!U->getStringOffsetSectionItem(Offset, StrOffset))
      return None;
    Offset = StrOffset;
  }
  if (const char *Str = U->getStringExtractor().getCStr(&Offset)) {
    return Str;
  }
  return None;
}

Optional<uint64_t> DWARFFormValue::getAsAddress(const DWARFUnit *U) const {
  if (!isFormClass(FC_Address))
    return None;
  if (Form == DW_FORM_GNU_addr_index) {
    uint32_t Index = Value.uval;
    uint64_t Result;
    if (!U || !U->getAddrOffsetSectionItem(Index, Result))
      return None;
    return Result;
  }
  return Value.uval;
}

Optional<uint64_t> DWARFFormValue::getAsReference(const DWARFUnit *U) const {
  if (!isFormClass(FC_Reference))
    return None;
  switch (Form) {
  case DW_FORM_ref1:
  case DW_FORM_ref2:
  case DW_FORM_ref4:
  case DW_FORM_ref8:
  case DW_FORM_ref_udata:
    if (!U)
      return None;
    return Value.uval + U->getOffset();
  case DW_FORM_ref_addr:
    return Value.uval;
  // FIXME: Add proper support for DW_FORM_ref_sig8 and DW_FORM_GNU_ref_alt.
  default:
    return None;
  }
}

Optional<uint64_t> DWARFFormValue::getAsSectionOffset() const {
  if (!isFormClass(FC_SectionOffset))
    return None;
  return Value.uval;
}

Optional<uint64_t> DWARFFormValue::getAsUnsignedConstant() const {
  if ((!isFormClass(FC_Constant) && !isFormClass(FC_Flag))
      || Form == DW_FORM_sdata)
    return None;
  return Value.uval;
}

Optional<int64_t> DWARFFormValue::getAsSignedConstant() const {
  if ((!isFormClass(FC_Constant) && !isFormClass(FC_Flag)) ||
      (Form == DW_FORM_udata && uint64_t(LLONG_MAX) < Value.uval))
    return None;
  switch (Form) {
  case DW_FORM_data4:
    return int32_t(Value.uval);
  case DW_FORM_data2:
    return int16_t(Value.uval);
  case DW_FORM_data1:
    return int8_t(Value.uval);
  case DW_FORM_sdata:
  case DW_FORM_data8:
  default:
    return Value.sval;
  }
}

Optional<ArrayRef<uint8_t>> DWARFFormValue::getAsBlock() const {
  if (!isFormClass(FC_Block) && !isFormClass(FC_Exprloc))
    return None;
  return ArrayRef<uint8_t>(Value.data, Value.uval);
}

