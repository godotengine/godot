//===-- llvm/Bitcode/ReaderWriter.h - Bitcode reader/writers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to read and write LLVM bitcode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_READERWRITER_H
#define LLVM_BITCODE_READERWRITER_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <string>

namespace llvm {
  class BitstreamWriter;
  class DataStreamer;
  class LLVMContext;
  class Module;
  class ModulePass;
  class raw_ostream;

  /// Read the header of the specified bitcode buffer and prepare for lazy
  /// deserialization of function bodies. If ShouldLazyLoadMetadata is true,
  /// lazily load metadata as well. If successful, this moves Buffer. On
  /// error, this *does not* move Buffer.
  ErrorOr<std::unique_ptr<Module>>
  getLazyBitcodeModule(std::unique_ptr<MemoryBuffer> &&Buffer,
                       LLVMContext &Context,
                       DiagnosticHandlerFunction DiagnosticHandler = nullptr,
                       bool ShouldLazyLoadMetadata = false,
                       bool ShouldTrackBitstreamUsage = false);

  /// Read the header of the specified stream and prepare for lazy
  /// deserialization and streaming of function bodies.
  ErrorOr<std::unique_ptr<Module>> getStreamedBitcodeModule(
      StringRef Name, std::unique_ptr<DataStreamer> Streamer,
      LLVMContext &Context,
      DiagnosticHandlerFunction DiagnosticHandler = nullptr);

  /// Read the header of the specified bitcode buffer and extract just the
  /// triple information. If successful, this returns a string. On error, this
  /// returns "".
  std::string
  getBitcodeTargetTriple(MemoryBufferRef Buffer, LLVMContext &Context,
                         DiagnosticHandlerFunction DiagnosticHandler = nullptr);

  /// Read the specified bitcode file, returning the module.
  ErrorOr<std::unique_ptr<Module>>
  parseBitcodeFile(MemoryBufferRef Buffer, LLVMContext &Context,
                   DiagnosticHandlerFunction DiagnosticHandler = nullptr,
                   bool ShouldTrackBitstreamUsage = false); // HLSL Change

  /// \brief Write the specified module to the specified raw output stream.
  ///
  /// For streams where it matters, the given stream should be in "binary"
  /// mode.
  ///
  /// If \c ShouldPreserveUseListOrder, encode the use-list order for each \a
  /// Value in \c M.  These will be reconstructed exactly when \a M is
  /// deserialized.
  void WriteBitcodeToFile(const Module *M, raw_ostream &Out,
                          bool ShouldPreserveUseListOrder = false);

  /// isBitcodeWrapper - Return true if the given bytes are the magic bytes
  /// for an LLVM IR bitcode wrapper.
  ///
  inline bool isBitcodeWrapper(const unsigned char *BufPtr,
                               const unsigned char *BufEnd) {
    // See if you can find the hidden message in the magic bytes :-).
    // (Hint: it's a little-endian encoding.)
    return BufPtr != BufEnd &&
           BufPtr[0] == 0xDE &&
           BufPtr[1] == 0xC0 &&
           BufPtr[2] == 0x17 &&
           BufPtr[3] == 0x0B;
  }

  /// isRawBitcode - Return true if the given bytes are the magic bytes for
  /// raw LLVM IR bitcode (without a wrapper).
  ///
  inline bool isRawBitcode(const unsigned char *BufPtr,
                           const unsigned char *BufEnd) {
    // These bytes sort of have a hidden message, but it's not in
    // little-endian this time, and it's a little redundant.
    return BufPtr != BufEnd &&
           BufPtr[0] == 'B' &&
           BufPtr[1] == 'C' &&
           BufPtr[2] == 0xc0 &&
           BufPtr[3] == 0xde;
  }

  /// isBitcode - Return true if the given bytes are the magic bytes for
  /// LLVM IR bitcode, either with or without a wrapper.
  ///
  inline bool isBitcode(const unsigned char *BufPtr,
                        const unsigned char *BufEnd) {
    return isBitcodeWrapper(BufPtr, BufEnd) ||
           isRawBitcode(BufPtr, BufEnd);
  }

  /// SkipBitcodeWrapperHeader - Some systems wrap bc files with a special
  /// header for padding or other reasons.  The format of this header is:
  ///
  /// struct bc_header {
  ///   uint32_t Magic;         // 0x0B17C0DE
  ///   uint32_t Version;       // Version, currently always 0.
  ///   uint32_t BitcodeOffset; // Offset to traditional bitcode file.
  ///   uint32_t BitcodeSize;   // Size of traditional bitcode file.
  ///   ... potentially other gunk ...
  /// };
  ///
  /// This function is called when we find a file with a matching magic number.
  /// In this case, skip down to the subsection of the file that is actually a
  /// BC file.
  /// If 'VerifyBufferSize' is true, check that the buffer is large enough to
  /// contain the whole bitcode file.
  inline bool SkipBitcodeWrapperHeader(const unsigned char *&BufPtr,
                                       const unsigned char *&BufEnd,
                                       bool VerifyBufferSize) {
    enum {
      KnownHeaderSize = 4*4,  // Size of header we read.
      OffsetField = 2*4,      // Offset in bytes to Offset field.
      SizeField = 3*4         // Offset in bytes to Size field.
    };

    // Must contain the header!
    if (BufEnd-BufPtr < KnownHeaderSize) return true;

    unsigned Offset = support::endian::read32le(&BufPtr[OffsetField]);
    unsigned Size = support::endian::read32le(&BufPtr[SizeField]);

    // Verify that Offset+Size fits in the file.
    if (VerifyBufferSize && Offset+Size > unsigned(BufEnd-BufPtr))
      return true;
    BufPtr += Offset;
    BufEnd = BufPtr+Size;
    return false;
  }

  const std::error_category &BitcodeErrorCategory();
  enum class BitcodeError { InvalidBitcodeSignature = 1, CorruptedBitcode };
  inline std::error_code make_error_code(BitcodeError E) {
    return std::error_code(static_cast<int>(E), BitcodeErrorCategory());
  }

  class BitcodeDiagnosticInfo : public DiagnosticInfo {
    const Twine &Msg;
    std::error_code EC;

  public:
    BitcodeDiagnosticInfo(std::error_code EC, DiagnosticSeverity Severity,
                          const Twine &Msg);
    void print(DiagnosticPrinter &DP) const override;
    std::error_code getError() const { return EC; };

    static bool classof(const DiagnosticInfo *DI) {
      return DI->getKind() == DK_Bitcode;
    }
  };

} // End llvm namespace

namespace std {
template <> struct is_error_code_enum<llvm::BitcodeError> : std::true_type {};
}

#endif
