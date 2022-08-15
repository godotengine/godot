//===- BitstreamWriter.h - Low-level bitstream writer interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamWriter class.  This class can be used to
// write an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_BITSTREAMWRITER_H
#define LLVM_BITCODE_BITSTREAMWRITER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitCodes.h"
#include "llvm/Support/Endian.h"
#include <vector>

extern thread_local bool emitting_sc_sentinel;
extern thread_local std::vector<uint64_t> sc_sentinel_offsets;

namespace llvm {

class BitstreamWriter {
  SmallVectorImpl<char> &Out;

  /// CurBit - Always between 0 and 31 inclusive, specifies the next bit to use.
  unsigned CurBit;

  /// CurValue - The current value.  Only bits < CurBit are valid.
  uint32_t CurValue;

  /// CurCodeSize - This is the declared size of code values used for the
  /// current block, in bits.
  unsigned CurCodeSize;

  /// BlockInfoCurBID - When emitting a BLOCKINFO_BLOCK, this is the currently
  /// selected BLOCK ID.
  unsigned BlockInfoCurBID;

  /// CurAbbrevs - Abbrevs installed at in this block.
  std::vector<IntrusiveRefCntPtr<BitCodeAbbrev>> CurAbbrevs;

  struct Block {
    unsigned PrevCodeSize;
    unsigned StartSizeWord;
    std::vector<IntrusiveRefCntPtr<BitCodeAbbrev>> PrevAbbrevs;
    Block(unsigned PCS, unsigned SSW) : PrevCodeSize(PCS), StartSizeWord(SSW) {}
  };

  /// BlockScope - This tracks the current blocks that we have entered.
  std::vector<Block> BlockScope;

  /// BlockInfo - This contains information emitted to BLOCKINFO_BLOCK blocks.
  /// These describe abbreviations that all blocks of the specified ID inherit.
  struct BlockInfo {
    unsigned BlockID;
    std::vector<IntrusiveRefCntPtr<BitCodeAbbrev>> Abbrevs;
  };
  std::vector<BlockInfo> BlockInfoRecords;

  // BackpatchWord - Backpatch a 32-bit word in the output with the specified
  // value.
  void BackpatchWord(unsigned ByteNo, unsigned NewWord) {
    support::endian::write32le(&Out[ByteNo], NewWord);
  }

  void WriteByte(unsigned char Value) {
    Out.push_back(Value);
  }

  void WriteWord(unsigned Value) {
    Value = support::endian::byte_swap<uint32_t, support::little>(Value);
    Out.append(reinterpret_cast<const char *>(&Value),
               reinterpret_cast<const char *>(&Value + 1));
  }

  unsigned GetBufferOffset() const {
    return Out.size();
  }

  unsigned GetWordIndex() const {
    unsigned Offset = GetBufferOffset();
    assert((Offset & 3) == 0 && "Not 32-bit aligned");
    return Offset / 4;
  }

public:
  explicit BitstreamWriter(SmallVectorImpl<char> &O)
    : Out(O), CurBit(0), CurValue(0), CurCodeSize(2) {}

  ~BitstreamWriter() {
#if 0 // HLSL Change - these are not true when recovering from OOM
    assert(CurBit == 0 && "Unflushed data remaining");
    assert(BlockScope.empty() && CurAbbrevs.empty() && "Block imbalance");
#endif
  }

  /// \brief Retrieve the current position in the stream, in bits.
  uint64_t GetCurrentBitNo() const { return GetBufferOffset() * 8 + CurBit; }

  //===--------------------------------------------------------------------===//
  // Basic Primitives for emitting bits to the stream.
  //===--------------------------------------------------------------------===//

  void Emit(uint32_t Val, unsigned NumBits) {
    assert(NumBits && NumBits <= 32 && "Invalid value size!");
    assert((Val & ~(~0U >> (32-NumBits))) == 0 && "High bits set!");
    CurValue |= Val << CurBit;
    if (CurBit + NumBits < 32) {
      CurBit += NumBits;
      return;
    }

    // Add the current word.
    WriteWord(CurValue);

    if (CurBit)
      CurValue = Val >> (32-CurBit);
    else
      CurValue = 0;
    CurBit = (CurBit+NumBits) & 31;
  }

  void Emit64(uint64_t Val, unsigned NumBits) {
    if (NumBits <= 32)
      Emit((uint32_t)Val, NumBits);
    else {
      Emit((uint32_t)Val, 32);
      Emit((uint32_t)(Val >> 32), NumBits-32);
    }
  }

  void FlushToWord() {
    if (CurBit) {
      WriteWord(CurValue);
      CurBit = 0;
      CurValue = 0;
    }
  }

  void EmitVBR(uint32_t Val, unsigned NumBits) {
    assert(NumBits <= 32 && "Too many bits to emit!");
    uint32_t Threshold = 1U << (NumBits-1);

    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit((Val & ((1 << (NumBits-1))-1)) | (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }

    Emit(Val, NumBits);
  }

  void EmitVBR64(uint64_t Val, unsigned NumBits) {
    assert(NumBits <= 32 && "Too many bits to emit!");
    if ((uint32_t)Val == Val)
      return EmitVBR((uint32_t)Val, NumBits);

    uint32_t Threshold = 1U << (NumBits-1);

    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit(((uint32_t)Val & ((1 << (NumBits-1))-1)) |
           (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }

    Emit((uint32_t)Val, NumBits);
  }

  /// EmitCode - Emit the specified code.
  void EmitCode(unsigned Val) {
    Emit(Val, CurCodeSize);
  }

  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//

  /// getBlockInfo - If there is block info for the specified ID, return it,
  /// otherwise return null.
  BlockInfo *getBlockInfo(unsigned BlockID) {
    // Common case, the most recent entry matches BlockID.
    if (!BlockInfoRecords.empty() && BlockInfoRecords.back().BlockID == BlockID)
      return &BlockInfoRecords.back();

    for (unsigned i = 0, e = static_cast<unsigned>(BlockInfoRecords.size());
         i != e; ++i)
      if (BlockInfoRecords[i].BlockID == BlockID)
        return &BlockInfoRecords[i];
    return nullptr;
  }

  void EnterSubblock(unsigned BlockID, unsigned CodeLen) {
    // Block header:
    //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]
    EmitCode(bitc::ENTER_SUBBLOCK);
    EmitVBR(BlockID, bitc::BlockIDWidth);
    EmitVBR(CodeLen, bitc::CodeLenWidth);
    FlushToWord();

    unsigned BlockSizeWordIndex = GetWordIndex();
    unsigned OldCodeSize = CurCodeSize;

    // Emit a placeholder, which will be replaced when the block is popped.
    Emit(0, bitc::BlockSizeWidth);

    CurCodeSize = CodeLen;

    // Push the outer block's abbrev set onto the stack, start out with an
    // empty abbrev set.
    BlockScope.emplace_back(OldCodeSize, BlockSizeWordIndex);
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);

    // If there is a blockinfo for this BlockID, add all the predefined abbrevs
    // to the abbrev list.
    if (BlockInfo *Info = getBlockInfo(BlockID)) {
      CurAbbrevs.insert(CurAbbrevs.end(), Info->Abbrevs.begin(),
                        Info->Abbrevs.end());
    }
  }

  void ExitBlock() {
    assert(!BlockScope.empty() && "Block scope imbalance!");
    const Block &B = BlockScope.back();

    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    EmitCode(bitc::END_BLOCK);
    FlushToWord();

    // Compute the size of the block, in words, not counting the size field.
    unsigned SizeInWords = GetWordIndex() - B.StartSizeWord - 1;
    unsigned ByteNo = B.StartSizeWord*4;

    // Update the block size field in the header of this sub-block.
    BackpatchWord(ByteNo, SizeInWords);

    // Restore the inner block's code size and abbrev table.
    CurCodeSize = B.PrevCodeSize;
    CurAbbrevs = std::move(B.PrevAbbrevs);
    BlockScope.pop_back();
  }

  //===--------------------------------------------------------------------===//
  // Record Emission
  //===--------------------------------------------------------------------===//

private:
  /// EmitAbbreviatedLiteral - Emit a literal value according to its abbrev
  /// record.  This is a no-op, since the abbrev specifies the literal to use.
  template<typename uintty>
  void EmitAbbreviatedLiteral(const BitCodeAbbrevOp &Op, uintty V) {
    assert(Op.isLiteral() && "Not a literal");
    // If the abbrev specifies the literal value to use, don't emit
    // anything.
    assert(V == Op.getLiteralValue() &&
           "Invalid abbrev for record!");
  }

  /// EmitAbbreviatedField - Emit a single scalar field value with the specified
  /// encoding.
  template<typename uintty>
  void EmitAbbreviatedField(const BitCodeAbbrevOp &Op, uintty V) {
    assert(!Op.isLiteral() && "Literals should use EmitAbbreviatedLiteral!");

    // Encode the value as we are commanded.
    switch (Op.getEncoding()) {
    default: llvm_unreachable("Unknown encoding!");
    case BitCodeAbbrevOp::Fixed:
      if (Op.getEncodingData())
        Emit((unsigned)V, (unsigned)Op.getEncodingData());
      break;
    case BitCodeAbbrevOp::VBR:
      if (Op.getEncodingData())
        EmitVBR64(V, (unsigned)Op.getEncodingData());
      break;
    case BitCodeAbbrevOp::Char6:
      Emit(BitCodeAbbrevOp::EncodeChar6((char)V), 6);
      break;
    }
  }

  /// EmitRecordWithAbbrevImpl - This is the core implementation of the record
  /// emission code.  If BlobData is non-null, then it specifies an array of
  /// data that should be emitted as part of the Blob or Array operand that is
  /// known to exist at the end of the record.
  template<typename uintty>
  void EmitRecordWithAbbrevImpl(unsigned Abbrev, SmallVectorImpl<uintty> &Vals,
                                StringRef Blob) {
    const char *BlobData = Blob.data();
    unsigned BlobLen = (unsigned) Blob.size();
    unsigned AbbrevNo = Abbrev-bitc::FIRST_APPLICATION_ABBREV;
    assert(AbbrevNo < CurAbbrevs.size() && "Invalid abbrev #!");
    const BitCodeAbbrev *Abbv = CurAbbrevs[AbbrevNo].get();

    EmitCode(Abbrev);

    unsigned RecordIdx = 0;
    for (unsigned i = 0, e = static_cast<unsigned>(Abbv->getNumOperandInfos());
         i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      if (Op.isLiteral()) {
        assert(RecordIdx < Vals.size() && "Invalid abbrev/record");
        EmitAbbreviatedLiteral(Op, Vals[RecordIdx]);
        ++RecordIdx;
      } else if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
        // Array case.
        assert(i+2 == e && "array op not second to last?");
        const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);

        // If this record has blob data, emit it, otherwise we must have record
        // entries to encode this way.
        if (BlobData) {
          assert(RecordIdx == Vals.size() &&
                 "Blob data and record entries specified for array!");
          // Emit a vbr6 to indicate the number of elements present.
          EmitVBR(static_cast<uint32_t>(BlobLen), 6);

          // Emit each field.
          for (unsigned i = 0; i != BlobLen; ++i)
            EmitAbbreviatedField(EltEnc, (unsigned char)BlobData[i]);

          // Know that blob data is consumed for assertion below.
          BlobData = nullptr;
        } else {
          // Emit a vbr6 to indicate the number of elements present.
          EmitVBR(static_cast<uint32_t>(Vals.size()-RecordIdx), 6);

          // Emit each field.
          for (unsigned e = Vals.size(); RecordIdx != e; ++RecordIdx)
            EmitAbbreviatedField(EltEnc, Vals[RecordIdx]);
        }
      } else if (Op.getEncoding() == BitCodeAbbrevOp::Blob) {
        // If this record has blob data, emit it, otherwise we must have record
        // entries to encode this way.

        // Emit a vbr6 to indicate the number of elements present.
        if (BlobData) {
          EmitVBR(static_cast<uint32_t>(BlobLen), 6);
          assert(RecordIdx == Vals.size() &&
                 "Blob data and record entries specified for blob operand!");
        } else {
          EmitVBR(static_cast<uint32_t>(Vals.size()-RecordIdx), 6);
        }

        // Flush to a 32-bit alignment boundary.
        FlushToWord();

        // Emit each field as a literal byte.
        if (BlobData) {
          for (unsigned i = 0; i != BlobLen; ++i)
            WriteByte((unsigned char)BlobData[i]);

          // Know that blob data is consumed for assertion below.
          BlobData = nullptr;
        } else {
          for (unsigned e = Vals.size(); RecordIdx != e; ++RecordIdx) {
            assert(isUInt<8>(Vals[RecordIdx]) &&
                   "Value too large to emit as blob");
            WriteByte((unsigned char)Vals[RecordIdx]);
          }
        }

        // Align end to 32-bits.
        while (GetBufferOffset() & 3)
          WriteByte(0);
      } else {  // Single scalar field.
        assert(RecordIdx < Vals.size() && "Invalid abbrev/record");
        if (emitting_sc_sentinel && RecordIdx == 1) {
          uint64_t sc_bit_offset = (uint64_t)Out.size_in_bytes() * 8 + CurBit;
          uint32_t sc_index = (Vals[RecordIdx] >> 1) % 0x45678900; // It has been left-shifted on a previous step
#ifndef NDEBUG
          assert(sc_index < sc_sentinel_offsets.size());
#endif
          sc_sentinel_offsets[sc_index] = sc_bit_offset;
          Vals[RecordIdx] = 0b1000001; // Explained somewhere else, we can't emit here more than 7 bits for a working patching
          emitting_sc_sentinel = false;
        }
        EmitAbbreviatedField(Op, Vals[RecordIdx]);
        ++RecordIdx;
      }
    }
    assert(RecordIdx == Vals.size() && "Not all record operands emitted!");
    assert(BlobData == nullptr &&
           "Blob data specified for record that doesn't use it!");
  }

public:

  /// EmitRecord - Emit the specified record to the stream, using an abbrev if
  /// we have one to compress the output.
  template<typename uintty>
  void EmitRecord(unsigned Code, SmallVectorImpl<uintty> &Vals,
                  unsigned Abbrev = 0) {
    if (!Abbrev) {
      // If we don't have an abbrev to use, emit this in its fully unabbreviated
      // form.
      EmitCode(bitc::UNABBREV_RECORD);
      EmitVBR(Code, 6);
      EmitVBR(static_cast<uint32_t>(Vals.size()), 6);
      for (unsigned i = 0, e = static_cast<unsigned>(Vals.size()); i != e; ++i)
        EmitVBR64(Vals[i], 6);
      return;
    }

    // Insert the code into Vals to treat it uniformly.
    Vals.insert(Vals.begin(), Code);

    EmitRecordWithAbbrev(Abbrev, Vals);
  }

  /// EmitRecordWithAbbrev - Emit a record with the specified abbreviation.
  /// Unlike EmitRecord, the code for the record should be included in Vals as
  /// the first entry.
  template<typename uintty>
  void EmitRecordWithAbbrev(unsigned Abbrev, SmallVectorImpl<uintty> &Vals) {
    EmitRecordWithAbbrevImpl(Abbrev, Vals, StringRef());
  }

  /// EmitRecordWithBlob - Emit the specified record to the stream, using an
  /// abbrev that includes a blob at the end.  The blob data to emit is
  /// specified by the pointer and length specified at the end.  In contrast to
  /// EmitRecord, this routine expects that the first entry in Vals is the code
  /// of the record.
  template<typename uintty>
  void EmitRecordWithBlob(unsigned Abbrev, SmallVectorImpl<uintty> &Vals,
                          StringRef Blob) {
    EmitRecordWithAbbrevImpl(Abbrev, Vals, Blob);
  }
  template<typename uintty>
  void EmitRecordWithBlob(unsigned Abbrev, SmallVectorImpl<uintty> &Vals,
                          const char *BlobData, unsigned BlobLen) {
    return EmitRecordWithAbbrevImpl(Abbrev, Vals, StringRef(BlobData, BlobLen));
  }

  /// EmitRecordWithArray - Just like EmitRecordWithBlob, works with records
  /// that end with an array.
  template<typename uintty>
  void EmitRecordWithArray(unsigned Abbrev, SmallVectorImpl<uintty> &Vals,
                          StringRef Array) {
    EmitRecordWithAbbrevImpl(Abbrev, Vals, Array);
  }
  template<typename uintty>
  void EmitRecordWithArray(unsigned Abbrev, SmallVectorImpl<uintty> &Vals,
                          const char *ArrayData, unsigned ArrayLen) {
    return EmitRecordWithAbbrevImpl(Abbrev, Vals, StringRef(ArrayData,
                                                            ArrayLen));
  }

  //===--------------------------------------------------------------------===//
  // Abbrev Emission
  //===--------------------------------------------------------------------===//

private:
  // Emit the abbreviation as a DEFINE_ABBREV record.
  void EncodeAbbrev(BitCodeAbbrev *Abbv) {
    EmitCode(bitc::DEFINE_ABBREV);
    EmitVBR(Abbv->getNumOperandInfos(), 5);
    for (unsigned i = 0, e = static_cast<unsigned>(Abbv->getNumOperandInfos());
         i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      Emit(Op.isLiteral(), 1);
      if (Op.isLiteral()) {
        EmitVBR64(Op.getLiteralValue(), 8);
      } else {
        Emit(Op.getEncoding(), 3);
        if (Op.hasEncodingData())
          EmitVBR64(Op.getEncodingData(), 5);
      }
    }
  }
public:

  /// EmitAbbrev - This emits an abbreviation to the stream.  Note that this
  /// method takes ownership of the specified abbrev.
  unsigned EmitAbbrev(BitCodeAbbrev *Abbv) {
    // Emit the abbreviation as a record.
    EncodeAbbrev(Abbv);
    CurAbbrevs.push_back(Abbv);
    return static_cast<unsigned>(CurAbbrevs.size())-1 +
      bitc::FIRST_APPLICATION_ABBREV;
  }

  //===--------------------------------------------------------------------===//
  // BlockInfo Block Emission
  //===--------------------------------------------------------------------===//

  /// EnterBlockInfoBlock - Start emitting the BLOCKINFO_BLOCK.
  void EnterBlockInfoBlock(unsigned CodeWidth) {
    EnterSubblock(bitc::BLOCKINFO_BLOCK_ID, CodeWidth);
    BlockInfoCurBID = ~0U;
  }
private:
  /// SwitchToBlockID - If we aren't already talking about the specified block
  /// ID, emit a BLOCKINFO_CODE_SETBID record.
  void SwitchToBlockID(unsigned BlockID) {
    if (BlockInfoCurBID == BlockID) return;
    SmallVector<unsigned, 2> V;
    V.push_back(BlockID);
    EmitRecord(bitc::BLOCKINFO_CODE_SETBID, V);
    BlockInfoCurBID = BlockID;
  }

  BlockInfo &getOrCreateBlockInfo(unsigned BlockID) {
    if (BlockInfo *BI = getBlockInfo(BlockID))
      return *BI;

    // Otherwise, add a new record.
    BlockInfoRecords.emplace_back();
    BlockInfoRecords.back().BlockID = BlockID;
    return BlockInfoRecords.back();
  }

public:

  /// EmitBlockInfoAbbrev - Emit a DEFINE_ABBREV record for the specified
  /// BlockID.
  unsigned EmitBlockInfoAbbrev(unsigned BlockID, BitCodeAbbrev *Abbv) {
    SwitchToBlockID(BlockID);
    EncodeAbbrev(Abbv);

    // Add the abbrev to the specified block record.
    BlockInfo &Info = getOrCreateBlockInfo(BlockID);
    Info.Abbrevs.push_back(Abbv);

    return Info.Abbrevs.size()-1+bitc::FIRST_APPLICATION_ABBREV;
  }
};


} // End llvm namespace

#endif
