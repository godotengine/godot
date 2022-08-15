//=== OutputBuffer.h - Output Buffer ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods to output values to a data buffer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_OUTPUTBUFFER_H
#define LLVM_SUPPORT_OUTPUTBUFFER_H

#include <cassert>
#include <string>
#include <vector>

namespace llvm {

  class OutputBuffer {
    /// Output buffer.
    std::vector<unsigned char> &Output;

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating what header values and flags to set.
    bool is64Bit, isLittleEndian;
  public:
    OutputBuffer(std::vector<unsigned char> &Out,
                 bool is64bit, bool le)
      : Output(Out), is64Bit(is64bit), isLittleEndian(le) {}

    // align - Emit padding into the file until the current output position is
    // aligned to the specified power of two boundary.
    void align(unsigned Boundary) {
      assert(Boundary && (Boundary & (Boundary - 1)) == 0 &&
             "Must align to 2^k boundary");
      size_t Size = Output.size();

      if (Size & (Boundary - 1)) {
        // Add padding to get alignment to the correct place.
        size_t Pad = Boundary - (Size & (Boundary - 1));
        Output.resize(Size + Pad);
      }
    }

    //===------------------------------------------------------------------===//
    // Out Functions - Output the specified value to the data buffer.

    void outbyte(unsigned char X) {
      Output.push_back(X);
    }
    void outhalf(unsigned short X) {
      if (isLittleEndian) {
        Output.push_back(X & 255);
        Output.push_back(X >> 8);
      } else {
        Output.push_back(X >> 8);
        Output.push_back(X & 255);
      }
    }
    void outword(unsigned X) {
      if (isLittleEndian) {
        Output.push_back((X >>  0) & 255);
        Output.push_back((X >>  8) & 255);
        Output.push_back((X >> 16) & 255);
        Output.push_back((X >> 24) & 255);
      } else {
        Output.push_back((X >> 24) & 255);
        Output.push_back((X >> 16) & 255);
        Output.push_back((X >>  8) & 255);
        Output.push_back((X >>  0) & 255);
      }
    }
    void outxword(uint64_t X) {
      if (isLittleEndian) {
        Output.push_back(unsigned(X >>  0) & 255);
        Output.push_back(unsigned(X >>  8) & 255);
        Output.push_back(unsigned(X >> 16) & 255);
        Output.push_back(unsigned(X >> 24) & 255);
        Output.push_back(unsigned(X >> 32) & 255);
        Output.push_back(unsigned(X >> 40) & 255);
        Output.push_back(unsigned(X >> 48) & 255);
        Output.push_back(unsigned(X >> 56) & 255);
      } else {
        Output.push_back(unsigned(X >> 56) & 255);
        Output.push_back(unsigned(X >> 48) & 255);
        Output.push_back(unsigned(X >> 40) & 255);
        Output.push_back(unsigned(X >> 32) & 255);
        Output.push_back(unsigned(X >> 24) & 255);
        Output.push_back(unsigned(X >> 16) & 255);
        Output.push_back(unsigned(X >>  8) & 255);
        Output.push_back(unsigned(X >>  0) & 255);
      }
    }
    void outaddr32(unsigned X) {
      outword(X);
    }
    void outaddr64(uint64_t X) {
      outxword(X);
    }
    void outaddr(uint64_t X) {
      if (!is64Bit)
        outword((unsigned)X);
      else
        outxword(X);
    }
    void outstring(const std::string &S, unsigned Length) {
      unsigned len_to_copy = static_cast<unsigned>(S.length()) < Length
        ? static_cast<unsigned>(S.length()) : Length;
      unsigned len_to_fill = static_cast<unsigned>(S.length()) < Length
        ? Length - static_cast<unsigned>(S.length()) : 0;

      for (unsigned i = 0; i < len_to_copy; ++i)
        outbyte(S[i]);

      for (unsigned i = 0; i < len_to_fill; ++i)
        outbyte(0);
    }

    //===------------------------------------------------------------------===//
    // Fix Functions - Replace an existing entry at an offset.

    void fixhalf(unsigned short X, unsigned Offset) {
      unsigned char *P = &Output[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 8)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 0)) & 255;
    }
    void fixword(unsigned X, unsigned Offset) {
      unsigned char *P = &Output[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 24)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 16)) & 255;
      P[2] = (X >> (isLittleEndian ? 16 :  8)) & 255;
      P[3] = (X >> (isLittleEndian ? 24 :  0)) & 255;
    }
    void fixxword(uint64_t X, unsigned Offset) {
      unsigned char *P = &Output[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 56)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 48)) & 255;
      P[2] = (X >> (isLittleEndian ? 16 : 40)) & 255;
      P[3] = (X >> (isLittleEndian ? 24 : 32)) & 255;
      P[4] = (X >> (isLittleEndian ? 32 : 24)) & 255;
      P[5] = (X >> (isLittleEndian ? 40 : 16)) & 255;
      P[6] = (X >> (isLittleEndian ? 48 :  8)) & 255;
      P[7] = (X >> (isLittleEndian ? 56 :  0)) & 255;
    }
    void fixaddr(uint64_t X, unsigned Offset) {
      if (!is64Bit)
        fixword((unsigned)X, Offset);
      else
        fixxword(X, Offset);
    }

    unsigned char &operator[](unsigned Index) {
      return Output[Index];
    }
    const unsigned char &operator[](unsigned Index) const {
      return Output[Index];
    }
  };

} // end llvm namespace

#endif // LLVM_SUPPORT_OUTPUTBUFFER_H
