//===- BitstreamReader.cpp - BitstreamReader implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitstreamReader.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  BitstreamCursor implementation
//===----------------------------------------------------------------------===//

void BitstreamCursor::freeState() {
  // Free all the Abbrevs.
  CurAbbrevs.clear();

  // Free all the Abbrevs in the block scope.
  BlockScope.clear();
}

/// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, enter
/// the block, and return true if the block has an error.
bool BitstreamCursor::EnterSubBlock(unsigned BlockID, unsigned *NumWordsP) {
  // Save the current block's state on BlockScope.
  BlockScope.push_back(Block(CurCodeSize));
  BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);

  // Add the abbrevs specific to this block to the CurAbbrevs list.
  if (const BitstreamReader::BlockInfo *Info =
      BitStream->getBlockInfo(BlockID)) {
    CurAbbrevs.insert(CurAbbrevs.end(), Info->Abbrevs.begin(),
                      Info->Abbrevs.end());
  }

  // Get the codesize of this block.
  CurCodeSize = ReadVBR(bitc::CodeLenWidth);
  // We can't read more than MaxChunkSize at a time
  if (CurCodeSize > MaxChunkSize)
    return true;

  SkipToFourByteBoundary();
  unsigned NumWords = Read(bitc::BlockSizeWidth);
  if (NumWordsP) *NumWordsP = NumWords;

  // Validate that this block is sane.
  return CurCodeSize == 0 || AtEndOfStream();
}

static uint64_t readAbbreviatedField(BitstreamCursor &Cursor,
                                     const BitCodeAbbrevOp &Op) {
  assert(!Op.isLiteral() && "Not to be used with literals!");

  // Decode the value as we are commanded.
  switch (Op.getEncoding()) {
  case BitCodeAbbrevOp::Array:
  case BitCodeAbbrevOp::Blob:
    llvm_unreachable("Should not reach here");
  case BitCodeAbbrevOp::Fixed:
    assert((unsigned)Op.getEncodingData() <= Cursor.MaxChunkSize);
    return Cursor.Read((unsigned)Op.getEncodingData());
  case BitCodeAbbrevOp::VBR:
    assert((unsigned)Op.getEncodingData() <= Cursor.MaxChunkSize);
    return Cursor.ReadVBR64((unsigned)Op.getEncodingData());
  case BitCodeAbbrevOp::Char6:
    return BitCodeAbbrevOp::DecodeChar6(Cursor.Read(6));
  }
  llvm_unreachable("invalid abbreviation encoding");
}

static void skipAbbreviatedField(BitstreamCursor &Cursor,
                                 const BitCodeAbbrevOp &Op) {
  assert(!Op.isLiteral() && "Not to be used with literals!");

  // Decode the value as we are commanded.
  switch (Op.getEncoding()) {
  case BitCodeAbbrevOp::Array:
  case BitCodeAbbrevOp::Blob:
    llvm_unreachable("Should not reach here");
  case BitCodeAbbrevOp::Fixed:
    assert((unsigned)Op.getEncodingData() <= Cursor.MaxChunkSize);
    Cursor.Read((unsigned)Op.getEncodingData());
    break;
  case BitCodeAbbrevOp::VBR:
    assert((unsigned)Op.getEncodingData() <= Cursor.MaxChunkSize);
    Cursor.ReadVBR64((unsigned)Op.getEncodingData());
    break;
  case BitCodeAbbrevOp::Char6:
    Cursor.Read(6);
    break;
  }
}



/// skipRecord - Read the current record and discard it.
void BitstreamCursor::skipRecord(unsigned AbbrevID) {
  // Skip unabbreviated records by reading past their entries.
  if (AbbrevID == bitc::UNABBREV_RECORD) {
    unsigned Code = ReadVBR(6);
    (void)Code;
    unsigned NumElts = ReadVBR(6);
    for (unsigned i = 0; i != NumElts; ++i)
      (void)ReadVBR64(6);
    return;
  }

  const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);

  for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
    const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
    if (Op.isLiteral())
      continue;

    if (Op.getEncoding() != BitCodeAbbrevOp::Array &&
        Op.getEncoding() != BitCodeAbbrevOp::Blob) {
      skipAbbreviatedField(*this, Op);
      continue;
    }

    if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
      // Array case.  Read the number of elements as a vbr6.
      unsigned NumElts = ReadVBR(6);

      // Get the element encoding.
      assert(i+2 == e && "array op not second to last?");
      const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);

#if 1 // HLSL Change - Make skipping go brrrrrrrrrrr
      {
        const auto &Op = EltEnc;
        auto &Cursor = *this;
        auto CurBit = Cursor.GetCurrentBitNo();
        // Decode the value as we are commanded.
        switch (EltEnc.getEncoding()) {
        case BitCodeAbbrevOp::Array:
        case BitCodeAbbrevOp::Blob:
          llvm_unreachable("Should not reach here");
        case BitCodeAbbrevOp::Fixed:
          assert((unsigned)Op.getEncodingData() <= Cursor.MaxChunkSize);
          Cursor.JumpToBit(CurBit + NumElts * Op.getEncodingData());
          break;
        case BitCodeAbbrevOp::VBR:
          assert((unsigned)Op.getEncodingData() <= Cursor.MaxChunkSize);
          for (; NumElts; --NumElts)
            Cursor.ReadVBR64((unsigned)Op.getEncodingData());
          break;
        case BitCodeAbbrevOp::Char6:
          Cursor.JumpToBit(CurBit + NumElts * 6);
          break;
        }
      }
#else
      // Read all the elements.
      for (; NumElts; --NumElts)
        skipAbbreviatedField(*this, EltEnc);
#endif
      continue;
    }

    assert(Op.getEncoding() == BitCodeAbbrevOp::Blob);
    // Blob case.  Read the number of bytes as a vbr6.
    unsigned NumElts = ReadVBR(6);
    SkipToFourByteBoundary();  // 32-bit alignment

    // Figure out where the end of this blob will be including tail padding.
    size_t NewEnd = GetCurrentBitNo()+((NumElts+3)&~3)*8;

    // If this would read off the end of the bitcode file, just set the
    // record to empty and return.
    if (!canSkipToPos(NewEnd/8)) {
      NextChar = BitStream->getBitcodeBytes().getExtent();
      break;
    }

    // Skip over the blob.
    JumpToBit(NewEnd);
  }
}

// HLSL Change - Begin
unsigned BitstreamCursor::peekRecord(unsigned AbbrevID) {
  auto last_bit_pos = GetCurrentBitNo();
  if (AbbrevID == bitc::UNABBREV_RECORD) {
    unsigned Code = ReadVBR(6);
    this->JumpToBit(last_bit_pos);
    return Code;
  }

  const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);

  // Read the record code first.
  assert(Abbv->getNumOperandInfos() != 0 && "no record code in abbreviation?");
  const BitCodeAbbrevOp &CodeOp = Abbv->getOperandInfo(0);
  unsigned Code;
  if (CodeOp.isLiteral())
    Code = CodeOp.getLiteralValue();
  else {
    if (CodeOp.getEncoding() == BitCodeAbbrevOp::Array ||
        CodeOp.getEncoding() == BitCodeAbbrevOp::Blob)
      report_fatal_error("Abbreviation starts with an Array or a Blob");
    Code = readAbbreviatedField(*this, CodeOp);
  }
  this->JumpToBit(last_bit_pos);
  return Code;
}

template<typename T>
void BitstreamCursor::AddRecordElements(BitCodeAbbrevOp::Encoding enc, uint64_t encData, unsigned NumElts, SmallVectorImpl<T> &Vals) {
  const unsigned size = (unsigned)encData;
  if (enc == BitCodeAbbrevOp::VBR) {
    assert((unsigned)encData <= MaxChunkSize);
    for (; NumElts; --NumElts) {
      Vals.push_back((T)ReadVBR64(size));
    }
  }
  else if (enc == BitCodeAbbrevOp::Char6) {
    assert((unsigned)encData <= MaxChunkSize);
    for (; NumElts; --NumElts) {
      Vals.push_back(BitCodeAbbrevOp::DecodeChar6(Read(6)));
    }
  }
  else {
    llvm_unreachable("Unknown kind of thing");
  }
}
// HLSL Change - End

unsigned BitstreamCursor::readRecord(unsigned AbbrevID,
                                     SmallVectorImpl<uint64_t> &Vals,
                                     StringRef *Blob,
                                     SmallVectorImpl<uint8_t> *Uint8Vals // HLSL Change
  ) {
  if (AbbrevID == bitc::UNABBREV_RECORD) {
    unsigned Code = ReadVBR(6);
    unsigned NumElts = ReadVBR(6);
    if (Uint8Vals) {
      for (unsigned i = 0; i != NumElts; ++i)
        Uint8Vals->push_back((uint8_t)ReadVBR64(6));
    }
    else {
      for (unsigned i = 0; i != NumElts; ++i)
        Vals.push_back(ReadVBR64(6));
    }
    return Code;
  }

  const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);

  // Read the record code first.
  assert(Abbv->getNumOperandInfos() != 0 && "no record code in abbreviation?");
  const BitCodeAbbrevOp &CodeOp = Abbv->getOperandInfo(0);
  unsigned Code;
  if (CodeOp.isLiteral())
    Code = CodeOp.getLiteralValue();
  else {
    if (CodeOp.getEncoding() == BitCodeAbbrevOp::Array ||
        CodeOp.getEncoding() == BitCodeAbbrevOp::Blob)
      report_fatal_error("Abbreviation starts with an Array or a Blob");
    Code = readAbbreviatedField(*this, CodeOp);
  }

  for (unsigned i = 1, e = Abbv->getNumOperandInfos(); i != e; ++i) {
    const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
    if (Op.isLiteral()) {
      Vals.push_back(Op.getLiteralValue());
      continue;
    }

    if (Op.getEncoding() != BitCodeAbbrevOp::Array &&
        Op.getEncoding() != BitCodeAbbrevOp::Blob) {
      Vals.push_back(readAbbreviatedField(*this, Op));
      continue;
    }

    if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
      // Array case.  Read the number of elements as a vbr6.
      unsigned NumElts = ReadVBR(6);

      // Get the element encoding.
      if (i + 2 != e)
        report_fatal_error("Array op not second to last");
      const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);
      if (!EltEnc.isEncoding())
        report_fatal_error(
            "Array element type has to be an encoding of a type");
      if (EltEnc.getEncoding() == BitCodeAbbrevOp::Array ||
          EltEnc.getEncoding() == BitCodeAbbrevOp::Blob)
        report_fatal_error("Array element type can't be an Array or a Blob");

#if 1 // HLSL Change
      // Read all the elements a little faster.
      {
        BitCodeAbbrevOp::Encoding enc = EltEnc.getEncoding();
        uint64_t encData = 0;
        if (EltEnc.hasEncodingData())
          encData = EltEnc.getEncodingData();
        unsigned size = (unsigned)encData;
        if (Uint8Vals) {
          if (enc == BitCodeAbbrevOp::Fixed) {
            assert((unsigned)encData <= MaxChunkSize);
            assert((unsigned)encData == 8);
            // Special optimization for fixed elements that are 8 bits
            Uint8Vals->resize(NumElts);
            uint8_t *ptr = Uint8Vals->data();
            unsigned i = 0;
            constexpr unsigned BytesInWord = sizeof(size_t);
            // First, read word by word instead of byte by byte
            for (; NumElts >= BytesInWord; NumElts -= BytesInWord) {
              const size_t e = Read(BytesInWord * 8);
              memcpy(ptr + i, &e, sizeof(e));
              i += BytesInWord;
            }
            for (; NumElts; --NumElts)
              Uint8Vals->operator[](i++) = (uint8_t)Read(8);
          }
          else {
            AddRecordElements(enc, encData, NumElts, *Uint8Vals);
          }
        }
        else {
          if (enc == BitCodeAbbrevOp::Fixed) {
            assert((unsigned)encData <= MaxChunkSize);
            Vals.reserve(Vals.size() + NumElts);
            for (; NumElts; --NumElts)
              Vals.push_back(Read(size));
          }
          else {
            AddRecordElements(enc, encData, NumElts, Vals);
          }
        }
      }
#else // HLSL Change
      // Read all the elements.
      for (; NumElts; --NumElts)
        Vals.push_back(readAbbreviatedField(*this, EltEnc));

#endif // HLSL Change
      continue;
    }

    assert(Op.getEncoding() == BitCodeAbbrevOp::Blob);
    // Blob case.  Read the number of bytes as a vbr6.
    unsigned NumElts = ReadVBR(6);
    SkipToFourByteBoundary();  // 32-bit alignment

    // Figure out where the end of this blob will be including tail padding.
    size_t CurBitPos = GetCurrentBitNo();
    size_t NewEnd = CurBitPos+((NumElts+3)&~3)*8;

    // If this would read off the end of the bitcode file, just set the
    // record to empty and return.
    if (!canSkipToPos(NewEnd/8)) {
      Vals.append(NumElts, 0);
      NextChar = BitStream->getBitcodeBytes().getExtent();
      break;
    }

    // Otherwise, inform the streamer that we need these bytes in memory.
    const char *Ptr = (const char*)
      BitStream->getBitcodeBytes().getPointer(CurBitPos/8, NumElts);

    // If we can return a reference to the data, do so to avoid copying it.
    if (Blob) {
      *Blob = StringRef(Ptr, NumElts);
    } else {
      // Otherwise, unpack into Vals with zero extension.
      for (; NumElts; --NumElts)
        Vals.push_back((unsigned char)*Ptr++);
    }
    // Skip over tail padding.
    JumpToBit(NewEnd);
  }

  return Code;
}


void BitstreamCursor::ReadAbbrevRecord() {
  BitCodeAbbrev *Abbv = new BitCodeAbbrev();
  unsigned NumOpInfo = ReadVBR(5);
  for (unsigned i = 0; i != NumOpInfo; ++i) {
    bool IsLiteral = Read(1);
    if (IsLiteral) {
      Abbv->Add(BitCodeAbbrevOp(ReadVBR64(8)));
      continue;
    }

    BitCodeAbbrevOp::Encoding E = (BitCodeAbbrevOp::Encoding)Read(3);
    if (BitCodeAbbrevOp::hasEncodingData(E)) {
      uint64_t Data = ReadVBR64(5);

      // As a special case, handle fixed(0) (i.e., a fixed field with zero bits)
      // and vbr(0) as a literal zero.  This is decoded the same way, and avoids
      // a slow path in Read() to have to handle reading zero bits.
      if ((E == BitCodeAbbrevOp::Fixed || E == BitCodeAbbrevOp::VBR) &&
          Data == 0) {
        Abbv->Add(BitCodeAbbrevOp(0));
        continue;
      }

      if ((E == BitCodeAbbrevOp::Fixed || E == BitCodeAbbrevOp::VBR) &&
          Data > MaxChunkSize)
        report_fatal_error(
            "Fixed or VBR abbrev record with size > MaxChunkData");

      Abbv->Add(BitCodeAbbrevOp(E, Data));
    } else
      Abbv->Add(BitCodeAbbrevOp(E));
  }

  if (Abbv->getNumOperandInfos() == 0)
    report_fatal_error("Abbrev record with no operands");
  CurAbbrevs.push_back(Abbv);
}

bool BitstreamCursor::ReadBlockInfoBlock(unsigned *pCount) {
  // If this is the second stream to get to the block info block, skip it.
  if (BitStream->hasBlockInfoRecords())
    return SkipBlock();

  if (EnterSubBlock(bitc::BLOCKINFO_BLOCK_ID)) return true;

  SmallVector<uint64_t, 64> Record;
  BitstreamReader::BlockInfo *CurBlockInfo = nullptr;

  // Read all the records for this module.
  while (1) {
    BitstreamEntry Entry = advanceSkippingSubblocks(AF_DontAutoprocessAbbrevs, pCount);

    switch (Entry.Kind) {
    case llvm::BitstreamEntry::SubBlock: // Handled for us already.
    case llvm::BitstreamEntry::Error:
      return true;
    case llvm::BitstreamEntry::EndBlock:
      return false;
    case llvm::BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read abbrev records, associate them with CurBID.
    if (Entry.ID == bitc::DEFINE_ABBREV) {
      if (!CurBlockInfo) return true;
      ReadAbbrevRecord();

      // ReadAbbrevRecord installs the abbrev in CurAbbrevs.  Move it to the
      // appropriate BlockInfo.
      CurBlockInfo->Abbrevs.push_back(std::move(CurAbbrevs.back()));
      CurAbbrevs.pop_back();
      continue;
    }

    // Read a record.
    Record.clear();
    switch (readRecord(Entry.ID, Record)) {
      default: break;  // Default behavior, ignore unknown content.
      case bitc::BLOCKINFO_CODE_SETBID:
        if (Record.size() < 1) return true;
        CurBlockInfo = &BitStream->getOrCreateBlockInfo((unsigned)Record[0]);
        break;
      case bitc::BLOCKINFO_CODE_BLOCKNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 0, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->Name = Name;
        break;
      }
      case bitc::BLOCKINFO_CODE_SETRECORDNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 1, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->RecordNames.push_back(std::make_pair((unsigned)Record[0],
                                                           Name));
        break;
      }
    }
  }
}

// HLSL Change Starts
void BitstreamUseTracker::track(BitstreamUseTracker *BT, uint64_t begin,
                                   uint64_t end) {
  if (BT)
    BT->insert(begin, end);
}

BitstreamUseTracker::ExtendResult
BitstreamUseTracker::extendRange(UseRange &Curr, UseRange &NewRange) {
  // Most likely case first.
  if (Curr.first <= NewRange.first && Curr.second < NewRange.second) {
    Curr.second = NewRange.second;
    return ExtendedEnd;
  }
  if (Curr.first <= NewRange.first && NewRange.second <= Curr.second) {
    return Included; // already included.
  }
  if (NewRange.first < Curr.first && NewRange.second <= Curr.second) {
    return ExtendedBegin;
  }
  if (NewRange.first < Curr.first && Curr.second < NewRange.second) {
    return ExtendedBoth;
  }
  return Exclusive;
}

bool BitstreamUseTracker::isDense(uint64_t endBitoffset) const {
  return Ranges.size() == 1 && Ranges[0].first == 0 &&
         Ranges[0].second == endBitoffset;
}

bool BitstreamUseTracker::considerMergeRight(size_t idx) {
  bool changed = false;
  while (idx < Ranges.size() - 1) {
    if (Ranges[idx].second >= Ranges[idx + 1].first) {
      Ranges[idx].second = Ranges[idx + 1].second;
      Ranges.erase(&Ranges[idx + 1]);
      changed = true;
    }
  }
  return changed;
}

void BitstreamUseTracker::insert(uint64_t begin, uint64_t end) {
  UseRange IR(begin, end);
  for (size_t i = 0, E = Ranges.size(); i < E; ++i) {
    ExtendResult ER = extendRange(Ranges[i], IR);
    switch (ER) {
    case Included:
      return;
    case ExtendedEnd:
      considerMergeRight(i);
      return;
    case ExtendedBegin:
      if (i > 0)
        considerMergeRight(i - 1);
      return;
    case ExtendedBoth:
      if (i > 0) {
        if (!considerMergeRight(i - 1))
          considerMergeRight(i);
      } else
        considerMergeRight(i);
      return;
    case Exclusive:
      // If completely to the left, then insert there; otherwise,
      // keep traversing in order.
      if (end <= Ranges[i].first) {
        Ranges.insert(&Ranges[i], IR);
        return;
      }
    }
  }

  // This range goes at the end.
  Ranges.push_back(IR);
}

BitstreamUseTracker::ScopeTrack
BitstreamUseTracker::scope_track(BitstreamCursor *BC) {
  ScopeTrack Result;
  Result.BC = BC;
  Result.begin = BC->GetCurrentBitNo();
  return Result;
}

// HLSL Change Ends
