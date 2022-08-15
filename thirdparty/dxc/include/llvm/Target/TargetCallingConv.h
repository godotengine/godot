//===-- llvm/Target/TargetCallingConv.h - Calling Convention ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines types for working with calling-convention information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETCALLINGCONV_H
#define LLVM_TARGET_TARGETCALLINGCONV_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include <string>
#include <limits.h>

namespace llvm {

namespace ISD {
  struct ArgFlagsTy {
  private:
    static const uint64_t NoFlagSet      = 0ULL;
    static const uint64_t ZExt           = 1ULL<<0;  ///< Zero extended
    static const uint64_t ZExtOffs       = 0;
    static const uint64_t SExt           = 1ULL<<1;  ///< Sign extended
    static const uint64_t SExtOffs       = 1;
    static const uint64_t InReg          = 1ULL<<2;  ///< Passed in register
    static const uint64_t InRegOffs      = 2;
    static const uint64_t SRet           = 1ULL<<3;  ///< Hidden struct-ret ptr
    static const uint64_t SRetOffs       = 3;
    static const uint64_t ByVal          = 1ULL<<4;  ///< Struct passed by value
    static const uint64_t ByValOffs      = 4;
    static const uint64_t Nest           = 1ULL<<5;  ///< Nested fn static chain
    static const uint64_t NestOffs       = 5;
    static const uint64_t Returned       = 1ULL<<6;  ///< Always returned
    static const uint64_t ReturnedOffs   = 6;
    static const uint64_t ByValAlign     = 0xFULL<<7; ///< Struct alignment
    static const uint64_t ByValAlignOffs = 7;
    static const uint64_t Split          = 1ULL<<11;
    static const uint64_t SplitOffs      = 11;
    static const uint64_t InAlloca       = 1ULL<<12; ///< Passed with inalloca
    static const uint64_t InAllocaOffs   = 12;
    static const uint64_t OrigAlign      = 0x1FULL<<27;
    static const uint64_t OrigAlignOffs  = 27;
    static const uint64_t ByValSize      = 0x3fffffffULL<<32; ///< Struct size
    static const uint64_t ByValSizeOffs  = 32;
    static const uint64_t InConsecutiveRegsLast      = 0x1ULL<<62; ///< Struct size
    static const uint64_t InConsecutiveRegsLastOffs  = 62;
    static const uint64_t InConsecutiveRegs      = 0x1ULL<<63; ///< Struct size
    static const uint64_t InConsecutiveRegsOffs  = 63;

    static const uint64_t One            = 1ULL; ///< 1 of this type, for shifts

    uint64_t Flags;
  public:
    ArgFlagsTy() : Flags(0) { }

    bool isZExt()      const { return Flags & ZExt; }
    void setZExt()     { Flags |= One << ZExtOffs; }

    bool isSExt()      const { return Flags & SExt; }
    void setSExt()     { Flags |= One << SExtOffs; }

    bool isInReg()     const { return Flags & InReg; }
    void setInReg()    { Flags |= One << InRegOffs; }

    bool isSRet()      const { return Flags & SRet; }
    void setSRet()     { Flags |= One << SRetOffs; }

    bool isByVal()     const { return Flags & ByVal; }
    void setByVal()    { Flags |= One << ByValOffs; }

    bool isInAlloca()  const { return Flags & InAlloca; }
    void setInAlloca() { Flags |= One << InAllocaOffs; }

    bool isNest()      const { return Flags & Nest; }
    void setNest()     { Flags |= One << NestOffs; }

    bool isReturned()  const { return Flags & Returned; }
    void setReturned() { Flags |= One << ReturnedOffs; }

    bool isInConsecutiveRegs()  const { return Flags & InConsecutiveRegs; }
    void setInConsecutiveRegs() { Flags |= One << InConsecutiveRegsOffs; }

    bool isInConsecutiveRegsLast()  const { return Flags & InConsecutiveRegsLast; }
    void setInConsecutiveRegsLast() { Flags |= One << InConsecutiveRegsLastOffs; }

    unsigned getByValAlign() const {
      return (unsigned)
        ((One << ((Flags & ByValAlign) >> ByValAlignOffs)) / 2);
    }
    void setByValAlign(unsigned A) {
      Flags = (Flags & ~ByValAlign) |
        (uint64_t(Log2_32(A) + 1) << ByValAlignOffs);
    }

    bool isSplit()   const { return Flags & Split; }
    void setSplit()  { Flags |= One << SplitOffs; }

    unsigned getOrigAlign() const {
      return (unsigned)
        ((One << ((Flags & OrigAlign) >> OrigAlignOffs)) / 2);
    }
    void setOrigAlign(unsigned A) {
      Flags = (Flags & ~OrigAlign) |
        (uint64_t(Log2_32(A) + 1) << OrigAlignOffs);
    }

    unsigned getByValSize() const {
      return (unsigned)((Flags & ByValSize) >> ByValSizeOffs);
    }
    void setByValSize(unsigned S) {
      Flags = (Flags & ~ByValSize) | (uint64_t(S) << ByValSizeOffs);
    }

    /// getRawBits - Represent the flags as a bunch of bits.
    uint64_t getRawBits() const { return Flags; }
  };

  /// InputArg - This struct carries flags and type information about a
  /// single incoming (formal) argument or incoming (from the perspective
  /// of the caller) return value virtual register.
  ///
  struct InputArg {
    ArgFlagsTy Flags;
    MVT VT;
    EVT ArgVT;
    bool Used;

    /// Index original Function's argument.
    unsigned OrigArgIndex;
    /// Sentinel value for implicit machine-level input arguments.
    static const unsigned NoArgIndex = UINT_MAX;

    /// Offset in bytes of current input value relative to the beginning of
    /// original argument. E.g. if argument was splitted into four 32 bit
    /// registers, we got 4 InputArgs with PartOffsets 0, 4, 8 and 12.
    unsigned PartOffset;

    InputArg() : VT(MVT::Other), Used(false) {}
    InputArg(ArgFlagsTy flags, EVT vt, EVT argvt, bool used,
             unsigned origIdx, unsigned partOffs)
      : Flags(flags), Used(used), OrigArgIndex(origIdx), PartOffset(partOffs) {
      VT = vt.getSimpleVT();
      ArgVT = argvt;
    }

    bool isOrigArg() const {
      return OrigArgIndex != NoArgIndex;
    }

    unsigned getOrigArgIndex() const {
      assert(OrigArgIndex != NoArgIndex && "Implicit machine-level argument");
      return OrigArgIndex;
    }
  };

  /// OutputArg - This struct carries flags and a value for a
  /// single outgoing (actual) argument or outgoing (from the perspective
  /// of the caller) return value virtual register.
  ///
  struct OutputArg {
    ArgFlagsTy Flags;
    MVT VT;
    EVT ArgVT;

    /// IsFixed - Is this a "fixed" value, ie not passed through a vararg "...".
    bool IsFixed;

    /// Index original Function's argument.
    unsigned OrigArgIndex;

    /// Offset in bytes of current output value relative to the beginning of
    /// original argument. E.g. if argument was splitted into four 32 bit
    /// registers, we got 4 OutputArgs with PartOffsets 0, 4, 8 and 12.
    unsigned PartOffset;

    OutputArg() : IsFixed(false) {}
    OutputArg(ArgFlagsTy flags, EVT vt, EVT argvt, bool isfixed,
              unsigned origIdx, unsigned partOffs)
      : Flags(flags), IsFixed(isfixed), OrigArgIndex(origIdx),
        PartOffset(partOffs) {
      VT = vt.getSimpleVT();
      ArgVT = argvt;
    }
  };
}

} // end llvm namespace

#endif
