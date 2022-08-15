//===-- DataLayout.cpp - Data size & alignment routines --------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines layout properties related to datatype size/offset/alignment
// information.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(StructType *ST, const DataLayout &DL) {
  assert(!ST->isOpaque() && "Cannot get layout of opaque structs");
  StructAlignment = 0;
  StructSize = 0;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    Type *Ty = ST->getElementType(i);
    unsigned TyAlign = ST->isPacked() ? 1 : DL.getABITypeAlignment(Ty);

    // Add padding if necessary to align the data element properly.
    if ((StructSize & (TyAlign-1)) != 0)
      StructSize = RoundUpToAlignment(StructSize, TyAlign);

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    MemberOffsets[i] = StructSize;
    StructSize += DL.getTypeAllocSize(Ty); // Consume space for this data item
  }

  // Empty structures have alignment of 1 byte.
  if (StructAlignment == 0) StructAlignment = 1;

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if ((StructSize & (StructAlignment-1)) != 0)
    StructSize = RoundUpToAlignment(StructSize, StructAlignment);
}


/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t Offset) const {
  const uint64_t *SI =
    std::upper_bound(&MemberOffsets[0], &MemberOffsets[NumElements], Offset);
  assert(SI != &MemberOffsets[0] && "Offset not in structure type!");
  --SI;
  assert(*SI <= Offset && "upper_bound didn't work");
  assert((SI == &MemberOffsets[0] || *(SI-1) <= Offset) &&
         (SI+1 == &MemberOffsets[NumElements] || *(SI+1) > Offset) &&
         "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI-&MemberOffsets[0];
}

//===----------------------------------------------------------------------===//
// LayoutAlignElem, LayoutAlign support
//===----------------------------------------------------------------------===//

LayoutAlignElem
LayoutAlignElem::get(AlignTypeEnum align_type, unsigned abi_align,
                     unsigned pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  LayoutAlignElem retval;
  retval.AlignType = align_type;
  retval.ABIAlign = abi_align;
  retval.PrefAlign = pref_align;
  retval.TypeBitWidth = bit_width;
  return retval;
}

bool
LayoutAlignElem::operator==(const LayoutAlignElem &rhs) const {
  return (AlignType == rhs.AlignType
          && ABIAlign == rhs.ABIAlign
          && PrefAlign == rhs.PrefAlign
          && TypeBitWidth == rhs.TypeBitWidth);
}

const LayoutAlignElem
DataLayout::InvalidAlignmentElem = { INVALID_ALIGN, 0, 0, 0 };

//===----------------------------------------------------------------------===//
// PointerAlignElem, PointerAlign support
//===----------------------------------------------------------------------===//

PointerAlignElem
PointerAlignElem::get(uint32_t AddressSpace, unsigned ABIAlign,
                      unsigned PrefAlign, uint32_t TypeByteWidth) {
  assert(ABIAlign <= PrefAlign && "Preferred alignment worse than ABI!");
  PointerAlignElem retval;
  retval.AddressSpace = AddressSpace;
  retval.ABIAlign = ABIAlign;
  retval.PrefAlign = PrefAlign;
  retval.TypeByteWidth = TypeByteWidth;
  return retval;
}

bool
PointerAlignElem::operator==(const PointerAlignElem &rhs) const {
  return (ABIAlign == rhs.ABIAlign
          && AddressSpace == rhs.AddressSpace
          && PrefAlign == rhs.PrefAlign
          && TypeByteWidth == rhs.TypeByteWidth);
}

const PointerAlignElem
DataLayout::InvalidPointerElem = { 0U, 0U, 0U, ~0U };

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

const char *DataLayout::getManglingComponent(const Triple &T) {
  if (T.isOSBinFormatMachO())
    return "-m:o";
  if (T.isOSWindows() && T.isOSBinFormatCOFF())
    return T.getArch() == Triple::x86 ? "-m:x" : "-m:w";
  return "-m:e";
}

static const LayoutAlignElem DefaultAlignments[] = {
  { INTEGER_ALIGN, 1, 1, 1 },    // i1
  { INTEGER_ALIGN, 8, 1, 1 },    // i8
  { INTEGER_ALIGN, 16, 2, 2 },   // i16
  { INTEGER_ALIGN, 32, 4, 4 },   // i32
  { INTEGER_ALIGN, 64, 4, 8 },   // i64
  { FLOAT_ALIGN, 16, 2, 2 },     // half
  { FLOAT_ALIGN, 32, 4, 4 },     // float
  { FLOAT_ALIGN, 64, 8, 8 },     // double
  { FLOAT_ALIGN, 128, 16, 16 },  // ppcf128, quad, ...
  { VECTOR_ALIGN, 64, 8, 8 },    // v2i32, v1i64, ...
  { VECTOR_ALIGN, 128, 16, 16 }, // v16i8, v8i16, v4i32, ...
  { AGGREGATE_ALIGN, 0, 0, 8 }   // struct
};

void DataLayout::reset(StringRef Desc) {
  clear();

  LayoutMap = nullptr;
  BigEndian = false;
  StackNaturalAlign = 0;
  ManglingMode = MM_None;

  for (const LayoutAlignElem &E : DefaultAlignments) {
    setAlignment((AlignTypeEnum)E.AlignType, E.ABIAlign, E.PrefAlign,
                 E.TypeBitWidth);
  }

  setPointerAlignment(0, 8, 8, 8);

  parseSpecifier(Desc);
}

/// Checked version of split, to ensure mandatory subparts.
static std::pair<StringRef, StringRef> split(StringRef Str, char Separator) {
  assert(!Str.empty() && "parse error, string can't be empty here");
  std::pair<StringRef, StringRef> Split = Str.split(Separator);
  if (Split.second.empty() && Split.first != Str)
    report_fatal_error("Trailing separator in datalayout string");
  if (!Split.second.empty() && Split.first.empty())
    report_fatal_error("Expected token before separator in datalayout string");
  return Split;
}

/// Get an unsigned integer, including error checks.
static unsigned getInt(StringRef R) {
  unsigned Result;
  bool error = R.getAsInteger(10, Result); (void)error;
  if (error)
    report_fatal_error("not a number, or does not fit in an unsigned int");
  return Result;
}

/// Convert bits into bytes. Assert if not a byte width multiple.
static unsigned inBytes(unsigned Bits) {
  if (Bits % 8)
    report_fatal_error("number of bits must be a byte width multiple");
  return Bits / 8;
}

void DataLayout::parseSpecifier(StringRef Desc) {
  StringRepresentation = Desc;
  while (!Desc.empty()) {
    // Split at '-'.
    std::pair<StringRef, StringRef> Split = split(Desc, '-');
    Desc = Split.second;

    // Split at ':'.
    Split = split(Split.first, ':');

    // Aliases used below.
    StringRef &Tok  = Split.first;  // Current token.
    StringRef &Rest = Split.second; // The rest of the string.

    char Specifier = Tok.front();
    Tok = Tok.substr(1);

    switch (Specifier) {
    case 's':
      // Ignored for backward compatibility.
      // FIXME: remove this on LLVM 4.0.
      break;
    case 'E':
      BigEndian = true;
      break;
    case 'e':
      BigEndian = false;
      break;
    case 'p': {
      // Address space.
      unsigned AddrSpace = Tok.empty() ? 0 : getInt(Tok);
      if (!isUInt<24>(AddrSpace))
        report_fatal_error("Invalid address space, must be a 24bit integer");

      // Size.
      if (Rest.empty())
        report_fatal_error(
            "Missing size specification for pointer in datalayout string");
      Split = split(Rest, ':');
      unsigned PointerMemSize = inBytes(getInt(Tok));
      if (!PointerMemSize)
        report_fatal_error("Invalid pointer size of 0 bytes");

      // ABI alignment.
      if (Rest.empty())
        report_fatal_error(
            "Missing alignment specification for pointer in datalayout string");
      Split = split(Rest, ':');
      unsigned PointerABIAlign = inBytes(getInt(Tok));
      if (!isPowerOf2_64(PointerABIAlign))
        report_fatal_error(
            "Pointer ABI alignment must be a power of 2");

      // Preferred alignment.
      unsigned PointerPrefAlign = PointerABIAlign;
      if (!Rest.empty()) {
        Split = split(Rest, ':');
        PointerPrefAlign = inBytes(getInt(Tok));
        if (!isPowerOf2_64(PointerPrefAlign))
          report_fatal_error(
            "Pointer preferred alignment must be a power of 2");
      }

      setPointerAlignment(AddrSpace, PointerABIAlign, PointerPrefAlign,
                          PointerMemSize);
      break;
    }
    case 'i':
    case 'v':
    case 'f':
    case 'a': {
      AlignTypeEnum AlignType;
      switch (Specifier) {
      default:
      case 'i': AlignType = INTEGER_ALIGN; break;
      case 'v': AlignType = VECTOR_ALIGN; break;
      case 'f': AlignType = FLOAT_ALIGN; break;
      case 'a': AlignType = AGGREGATE_ALIGN; break;
      }

      // Bit size.
      unsigned Size = Tok.empty() ? 0 : getInt(Tok);

      if (AlignType == AGGREGATE_ALIGN && Size != 0)
        report_fatal_error(
            "Sized aggregate specification in datalayout string");

      // ABI alignment.
      if (Rest.empty())
        report_fatal_error(
            "Missing alignment specification in datalayout string");
      Split = split(Rest, ':');
      unsigned ABIAlign = inBytes(getInt(Tok));
      if (AlignType != AGGREGATE_ALIGN && !ABIAlign)
        report_fatal_error(
            "ABI alignment specification must be >0 for non-aggregate types");

      // Preferred alignment.
      unsigned PrefAlign = ABIAlign;
      if (!Rest.empty()) {
        Split = split(Rest, ':');
        PrefAlign = inBytes(getInt(Tok));
      }

      setAlignment(AlignType, ABIAlign, PrefAlign, Size);

      break;
    }
    case 'n':  // Native integer types.
      for (;;) {
        unsigned Width = getInt(Tok);
        if (Width == 0)
          report_fatal_error(
              "Zero width native integer type in datalayout string");
        LegalIntWidths.push_back(Width);
        if (Rest.empty())
          break;
        Split = split(Rest, ':');
      }
      break;
    case 'S': { // Stack natural alignment.
      StackNaturalAlign = inBytes(getInt(Tok));
      break;
    }
    case 'm':
      if (!Tok.empty())
        report_fatal_error("Unexpected trailing characters after mangling specifier in datalayout string");
      if (Rest.empty())
        report_fatal_error("Expected mangling specifier in datalayout string");
      if (Rest.size() > 1)
        report_fatal_error("Unknown mangling specifier in datalayout string");
      switch(Rest[0]) {
      default:
        report_fatal_error("Unknown mangling in datalayout string");
      case 'e':
        ManglingMode = MM_ELF;
        break;
      case 'o':
        ManglingMode = MM_MachO;
        break;
      case 'm':
        ManglingMode = MM_Mips;
        break;
      case 'w':
        ManglingMode = MM_WinCOFF;
        break;
      case 'x':
        ManglingMode = MM_WinCOFFX86;
        break;
      }
      break;
    default:
      report_fatal_error("Unknown specifier in datalayout string");
      break;
    }
  }
}

DataLayout::DataLayout(const Module *M) : LayoutMap(nullptr) {
  init(M);
}

void DataLayout::init(const Module *M) { *this = M->getDataLayout(); }

bool DataLayout::operator==(const DataLayout &Other) const {
  bool Ret = BigEndian == Other.BigEndian &&
             StackNaturalAlign == Other.StackNaturalAlign &&
             ManglingMode == Other.ManglingMode &&
             LegalIntWidths == Other.LegalIntWidths &&
             Alignments == Other.Alignments && Pointers == Other.Pointers;
  // Note: getStringRepresentation() might differs, it is not canonicalized
  return Ret;
}

void
DataLayout::setAlignment(AlignTypeEnum align_type, unsigned abi_align,
                         unsigned pref_align, uint32_t bit_width) {
  if (!isUInt<24>(bit_width))
    report_fatal_error("Invalid bit width, must be a 24bit integer");
  if (!isUInt<16>(abi_align))
    report_fatal_error("Invalid ABI alignment, must be a 16bit integer");
  if (!isUInt<16>(pref_align))
    report_fatal_error("Invalid preferred alignment, must be a 16bit integer");
  if (abi_align != 0 && !isPowerOf2_64(abi_align))
    report_fatal_error("Invalid ABI alignment, must be a power of 2");
  if (pref_align != 0 && !isPowerOf2_64(pref_align))
    report_fatal_error("Invalid preferred alignment, must be a power of 2");

  if (pref_align < abi_align)
    report_fatal_error(
        "Preferred alignment cannot be less than the ABI alignment");

  for (LayoutAlignElem &Elem : Alignments) {
    if (Elem.AlignType == (unsigned)align_type &&
        Elem.TypeBitWidth == bit_width) {
      // Update the abi, preferred alignments.
      Elem.ABIAlign = abi_align;
      Elem.PrefAlign = pref_align;
      return;
    }
  }

  Alignments.push_back(LayoutAlignElem::get(align_type, abi_align,
                                            pref_align, bit_width));
}

DataLayout::PointersTy::iterator
DataLayout::findPointerLowerBound(uint32_t AddressSpace) {
  return std::lower_bound(Pointers.begin(), Pointers.end(), AddressSpace,
                          [](const PointerAlignElem &A, uint32_t AddressSpace) {
    return A.AddressSpace < AddressSpace;
  });
}

void DataLayout::setPointerAlignment(uint32_t AddrSpace, unsigned ABIAlign,
                                     unsigned PrefAlign,
                                     uint32_t TypeByteWidth) {
  if (PrefAlign < ABIAlign)
    report_fatal_error(
        "Preferred alignment cannot be less than the ABI alignment");

  PointersTy::iterator I = findPointerLowerBound(AddrSpace);
  if (I == Pointers.end() || I->AddressSpace != AddrSpace) {
    Pointers.insert(I, PointerAlignElem::get(AddrSpace, ABIAlign, PrefAlign,
                                             TypeByteWidth));
  } else {
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
    I->TypeByteWidth = TypeByteWidth;
  }
}

/// getAlignmentInfo - Return the alignment (either ABI if ABIInfo = true or
/// preferred if ABIInfo = false) the layout wants for the specified datatype.
unsigned DataLayout::getAlignmentInfo(AlignTypeEnum AlignType,
                                      uint32_t BitWidth, bool ABIInfo,
                                      Type *Ty) const {
  // Check to see if we have an exact match and remember the best match we see.
  int BestMatchIdx = -1;
  int LargestInt = -1;
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == (unsigned)AlignType &&
        Alignments[i].TypeBitWidth == BitWidth)
      return ABIInfo ? Alignments[i].ABIAlign : Alignments[i].PrefAlign;

    // The best match so far depends on what we're looking for.
     if (AlignType == INTEGER_ALIGN &&
         Alignments[i].AlignType == INTEGER_ALIGN) {
      // The "best match" for integers is the smallest size that is larger than
      // the BitWidth requested.
      if (Alignments[i].TypeBitWidth > BitWidth && (BestMatchIdx == -1 ||
          Alignments[i].TypeBitWidth < Alignments[BestMatchIdx].TypeBitWidth))
        BestMatchIdx = i;
      // However, if there isn't one that's larger, then we must use the
      // largest one we have (see below)
      if (LargestInt == -1 ||
          Alignments[i].TypeBitWidth > Alignments[LargestInt].TypeBitWidth)
        LargestInt = i;
    }
  }

  // Okay, we didn't find an exact solution.  Fall back here depending on what
  // is being looked for.
  if (BestMatchIdx == -1) {
    // If we didn't find an integer alignment, fall back on most conservative.
    if (AlignType == INTEGER_ALIGN) {
      BestMatchIdx = LargestInt;
    } else if (AlignType == VECTOR_ALIGN) {
      // By default, use natural alignment for vector types. This is consistent
      // with what clang and llvm-gcc do.
      unsigned Align = getTypeAllocSize(cast<VectorType>(Ty)->getElementType());
      Align *= cast<VectorType>(Ty)->getNumElements();
      // If the alignment is not a power of 2, round up to the next power of 2.
      // This happens for non-power-of-2 length vectors.
      if (Align & (Align-1))
        Align = NextPowerOf2(Align);
      return Align;
    }
  }

  // If we still couldn't find a reasonable default alignment, fall back
  // to a simple heuristic that the alignment is the first power of two
  // greater-or-equal to the store size of the type.  This is a reasonable
  // approximation of reality, and if the user wanted something less
  // less conservative, they should have specified it explicitly in the data
  // layout.
  if (BestMatchIdx == -1) {
    unsigned Align = getTypeStoreSize(Ty);
    if (Align & (Align-1))
      Align = NextPowerOf2(Align);
    return Align;
  }

  // Since we got a "best match" index, just return it.
  return ABIInfo ? Alignments[BestMatchIdx].ABIAlign
                 : Alignments[BestMatchIdx].PrefAlign;
}

namespace {

class StructLayoutMap {
  typedef DenseMap<StructType*, StructLayout*> LayoutInfoTy;
  LayoutInfoTy LayoutInfo;

public:
  ~StructLayoutMap() {
    // Remove any layouts.
    for (const auto &I : LayoutInfo) {
      StructLayout *Value = I.second;
      Value->~StructLayout();
      ::operator delete(Value); // HLSL Change: Use overridable operator delete
    }
  }

  StructLayout *&operator[](StructType *STy) {
    return LayoutInfo[STy];
  }
};

} // end anonymous namespace

void DataLayout::clear() {
  LegalIntWidths.clear();
  Alignments.clear();
  Pointers.clear();
  delete static_cast<StructLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
}

DataLayout::~DataLayout() {
  clear();
}

const StructLayout *DataLayout::getStructLayout(StructType *Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();

  StructLayoutMap *STM = static_cast<StructLayoutMap*>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL) return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  int NumElts = Ty->getNumElements();
  StructLayout *L =
    (StructLayout *)::operator new(sizeof(StructLayout)+(NumElts-1) * sizeof(uint64_t)); // HLSL Change: Use overridable operator new

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}


unsigned DataLayout::getPointerABIAlignment(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->ABIAlign;
}

unsigned DataLayout::getPointerPrefAlignment(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->PrefAlign;
}

unsigned DataLayout::getPointerSize(unsigned AS) const {
  PointersTy::const_iterator I = findPointerLowerBound(AS);
  if (I == Pointers.end() || I->AddressSpace != AS) {
    I = findPointerLowerBound(0);
    assert(I->AddressSpace == 0);
  }
  return I->TypeByteWidth;
}

unsigned DataLayout::getPointerTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "This should only be called with a pointer or pointer vector type");

  if (Ty->isPointerTy())
    return getTypeSizeInBits(Ty);

  return getTypeSizeInBits(Ty->getScalarType());
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
unsigned DataLayout::getAlignment(Type *Ty, bool abi_or_pref) const {
  int AlignType = -1;

  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  // Early escape for the non-numeric types.
  case Type::LabelTyID:
    return (abi_or_pref
            ? getPointerABIAlignment(0)
            : getPointerPrefAlignment(0));
  case Type::PointerTyID: {
    unsigned AS = cast<PointerType>(Ty)->getAddressSpace();
    return (abi_or_pref
            ? getPointerABIAlignment(AS)
            : getPointerPrefAlignment(AS));
    }
  case Type::ArrayTyID:
    return getAlignment(cast<ArrayType>(Ty)->getElementType(), abi_or_pref);

  case Type::StructTyID: {
    // Packed structure types always have an ABI alignment of one.
    if (cast<StructType>(Ty)->isPacked() && abi_or_pref)
      return 1;

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    unsigned Align = getAlignmentInfo(AGGREGATE_ALIGN, 0, abi_or_pref, Ty);
    return std::max(Align, Layout->getAlignment());
  }
  case Type::IntegerTyID:
    AlignType = INTEGER_ALIGN;
    break;
  case Type::HalfTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  // PPC_FP128TyID and FP128TyID have different data contents, but the
  // same size and alignment, so they look the same here.
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
  case Type::X86_FP80TyID:
    AlignType = FLOAT_ALIGN;
    break;
  case Type::X86_MMXTyID:
  case Type::VectorTyID:
    AlignType = VECTOR_ALIGN;
    // HLSL Change Begins.
    return getAlignment(Ty->getVectorElementType(), abi_or_pref);
    // HLSL Change Ends.
    break;
  default:
    llvm_unreachable("Bad type for getAlignment!!!");
  }

  return getAlignmentInfo((AlignTypeEnum)AlignType, getTypeSizeInBits(Ty),
                          abi_or_pref, Ty);
}

unsigned DataLayout::getABITypeAlignment(Type *Ty) const {
  return getAlignment(Ty, true);
}

/// getABIIntegerTypeAlignment - Return the minimum ABI-required alignment for
/// an integer type of the specified bitwidth.
unsigned DataLayout::getABIIntegerTypeAlignment(unsigned BitWidth) const {
  return getAlignmentInfo(INTEGER_ALIGN, BitWidth, true, nullptr);
}

unsigned DataLayout::getPrefTypeAlignment(Type *Ty) const {
  return getAlignment(Ty, false);
}

unsigned DataLayout::getPreferredTypeAlignmentShift(Type *Ty) const {
  unsigned Align = getPrefTypeAlignment(Ty);
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");
  return Log2_32(Align);
}

IntegerType *DataLayout::getIntPtrType(LLVMContext &C,
                                       unsigned AddressSpace) const {
  return IntegerType::get(C, getPointerSizeInBits(AddressSpace));
}

Type *DataLayout::getIntPtrType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getPointerTypeSizeInBits(Ty);
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return VectorType::get(IntTy, VecTy->getNumElements());
  return IntTy;
}

Type *DataLayout::getSmallestLegalIntType(LLVMContext &C, unsigned Width) const {
  for (unsigned LegalIntWidth : LegalIntWidths)
    if (Width <= LegalIntWidth)
      return Type::getIntNTy(C, LegalIntWidth);
  return nullptr;
}

unsigned DataLayout::getLargestLegalIntTypeSize() const {
  auto Max = std::max_element(LegalIntWidths.begin(), LegalIntWidths.end());
  return Max != LegalIntWidths.end() ? *Max : 0;
}

uint64_t DataLayout::getIndexedOffset(Type *ptrTy,
                                      ArrayRef<Value *> Indices) const {
  Type *Ty = ptrTy;
  assert(Ty->isPointerTy() && "Illegal argument for getIndexedOffset()");
  uint64_t Result = 0;

  generic_gep_type_iterator<Value* const*>
    TI = gep_type_begin(ptrTy, Indices);
  for (unsigned CurIDX = 0, EndIDX = Indices.size(); CurIDX != EndIDX;
       ++CurIDX, ++TI) {
    if (StructType *STy = dyn_cast<StructType>(*TI)) {
      assert(Indices[CurIDX]->getType() ==
             Type::getInt32Ty(ptrTy->getContext()) &&
             "Illegal struct idx");
      unsigned FieldNo = cast<ConstantInt>(Indices[CurIDX])->getZExtValue();

      // Get structure layout information...
      const StructLayout *Layout = getStructLayout(STy);

      // Add in the offset, as calculated by the structure layout info...
      Result += Layout->getElementOffset(FieldNo);

      // Update Ty to refer to current element
      Ty = STy->getElementType(FieldNo);
    } else {
      // Update Ty to refer to current element
      Ty = cast<SequentialType>(Ty)->getElementType();

      // Get the array index and the size of each array element.
      if (int64_t arrayIdx = cast<ConstantInt>(Indices[CurIDX])->getSExtValue())
        Result += (uint64_t)arrayIdx * getTypeAllocSize(Ty);
    }
  }

  return Result;
}

/// getPreferredAlignment - Return the preferred alignment of the specified
/// global.  This includes an explicitly requested alignment (if the global
/// has one).
unsigned DataLayout::getPreferredAlignment(const GlobalVariable *GV) const {
  Type *ElemType = GV->getType()->getElementType();
  unsigned Alignment = getPrefTypeAlignment(ElemType);
  unsigned GVAlignment = GV->getAlignment();
  if (GVAlignment >= Alignment) {
    Alignment = GVAlignment;
  } else if (GVAlignment != 0) {
    Alignment = std::max(GVAlignment, getABITypeAlignment(ElemType));
  }

  if (GV->hasInitializer() && GVAlignment == 0) {
    if (Alignment < 16) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (getTypeSizeInBits(ElemType) > 128)
        Alignment = 16;    // 16-byte alignment.
    }
  }
  return Alignment;
}

/// getPreferredAlignmentLog - Return the preferred alignment of the
/// specified global, returned in log form.  This includes an explicitly
/// requested alignment (if the global has one).
unsigned DataLayout::getPreferredAlignmentLog(const GlobalVariable *GV) const {
  return Log2_32(getPreferredAlignment(GV));
}

