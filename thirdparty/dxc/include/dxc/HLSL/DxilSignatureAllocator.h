///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSignatureAllocator.h                                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Classes used for allocating signature elements.                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/DXIL/DxilConstants.h"
#include <vector>

namespace hlsl {

class DxilSignatureAllocator {
public:
  class PackElement {
  public:
    virtual ~PackElement() {}
    virtual uint32_t GetID() const = 0;
    virtual DXIL::SemanticKind GetKind() const = 0;
    virtual DXIL::InterpolationMode GetInterpolationMode() const = 0;
    virtual DXIL::SemanticInterpretationKind GetInterpretation() const = 0;
    virtual DXIL::SignatureDataWidth GetDataBitWidth() const = 0;
    virtual uint32_t GetRows() const = 0;
    virtual uint32_t GetCols() const = 0;
    virtual bool IsAllocated() const = 0;
    virtual uint32_t GetStartRow() const = 0;
    virtual uint32_t GetStartCol() const = 0;

    virtual void ClearLocation() = 0;
    virtual void SetLocation(uint32_t StartRow, uint32_t StartCol) = 0;
  };
  class DummyElement : public PackElement {
  public:
    uint32_t id;
    uint32_t rows, cols;
    uint32_t row, col;
    DXIL::SemanticKind kind;
    DXIL::InterpolationMode interpolation;
    DXIL::SemanticInterpretationKind interpretation;
    DXIL::SignatureDataWidth dataBitWidth;
    uint32_t indexFlags;

  public:
    DummyElement(uint32_t index = 0) : id(index), rows(1), cols(1), row((uint32_t)-1), col((uint32_t)-1),
      kind(DXIL::SemanticKind::Arbitrary),
      interpolation(DXIL::InterpolationMode::Undefined),
      interpretation(DXIL::SemanticInterpretationKind::Arb),
      dataBitWidth(DXIL::SignatureDataWidth::Undefined),
      indexFlags(0)
    {}
    ~DummyElement() override {}
    uint32_t GetID() const override { return id; }
    DXIL::SemanticKind GetKind() const override { return kind; }
    DXIL::InterpolationMode GetInterpolationMode() const override { return interpolation; }
    DXIL::SemanticInterpretationKind GetInterpretation() const override { return interpretation; }
    DXIL::SignatureDataWidth GetDataBitWidth() const override { return dataBitWidth; }
    uint32_t GetRows() const override { return rows; }
    uint32_t GetCols() const override { return cols; }
    bool IsAllocated() const override { return row != (uint32_t)-1; }
    uint32_t GetStartRow() const override { return row; }
    uint32_t GetStartCol() const override { return col; }

    void ClearLocation() override { row = col = (uint32_t)-1; }
    void SetLocation(uint32_t Row, uint32_t Col) override { row = Row; col = Col; }
  };

  // index flags
  static const uint8_t kIndexedUp = 1 << 0;     // Indexing continues upwards
  static const uint8_t kIndexedDown = 1 << 1;   // Indexing continues downwards
  static uint8_t GetIndexFlags(unsigned row, unsigned rows) {
    return ((row > 0) ? kIndexedUp : 0) | ((row < rows - 1) ? kIndexedDown : 0);
  }
  // element flags
  static const uint8_t kEFOccupied = 1 << 0;
  static const uint8_t kEFArbitrary = 1 << 1;
  static const uint8_t kEFSGV = 1 << 2;
  static const uint8_t kEFSV = 1 << 3;
  static const uint8_t kEFTessFactor = 1 << 4;
  static const uint8_t kEFClipCull = 1 << 5;
  static const uint8_t kEFConflictsWithIndexed = kEFSGV | kEFSV;
  static uint8_t GetElementFlags(const PackElement *SE);

  // The following two functions enforce the rules of component ordering when packing different
  // kinds of elements into the same register.

  // given element flags, return element flags that conflict when placed to the left of the element
  static uint8_t GetConflictFlagsLeft(uint8_t flags);
  // given element flags, return element flags that conflict when placed to the right of the element
  static uint8_t GetConflictFlagsRight(uint8_t flags);

  enum ConflictType {
    kNoConflict = 0,
    kConflictsWithIndexed,
    kConflictsWithIndexedTessFactor,
    kConflictsWithInterpolationMode,
    kInsufficientFreeComponents,
    kOverlapElement,
    kIllegalComponentOrder,
    kConflictFit,
    kConflictDataWidth,
  };

  struct PackedRegister {
    // Flags:
    // - for occupied components, they signify element flags
    // - for unoccupied components, they signify conflict flags
    uint8_t Flags[4];
    DXIL::InterpolationMode Interp : 8;
    uint8_t IndexFlags : 2;
    uint8_t IndexingFixed : 1;
    DXIL::SignatureDataWidth DataWidth; // length of each scalar type in bytes. (2 or 4 for now)

    PackedRegister();
    ConflictType DetectRowConflict(uint8_t flags, uint8_t indexFlags, DXIL::InterpolationMode interp, unsigned width, DXIL::SignatureDataWidth dataWidth);
    ConflictType DetectColConflict(uint8_t flags, unsigned col, unsigned width);
    void PlaceElement(uint8_t flags, uint8_t indexFlags, DXIL::InterpolationMode interp, unsigned col, unsigned width, DXIL::SignatureDataWidth dataWidth);
  };

  DxilSignatureAllocator(unsigned numRegisters, bool useMinPrecision);

  bool GetIgnoreIndexing() const { return m_bIgnoreIndexing; }
  void SetIgnoreIndexing(bool ignoreIndexing) { m_bIgnoreIndexing  = ignoreIndexing; }

  ConflictType DetectRowConflict(const PackElement *SE, unsigned row);
  ConflictType DetectColConflict(const PackElement *SE, unsigned row, unsigned col);
  void PlaceElement(const PackElement *SE, unsigned row, unsigned col);

  // FindNext/PackNext return found/packed location + element rows if found,
  // otherwise, they return 0.
  unsigned FindNext(unsigned &foundRow, unsigned &foundCol,
                    PackElement* SE, unsigned startRow, unsigned numRows, unsigned startCol = 0);
  unsigned PackNext(PackElement* SE, unsigned startRow, unsigned numRows, unsigned startCol = 0);

  // Simple greedy in-order packer used by PackOptimized
  unsigned PackGreedy(std::vector<PackElement*> elements, unsigned startRow, unsigned numRows, unsigned startCol = 0);

  // Optimized packing algorithm - appended elements may affect positions of prior elements.
  unsigned PackOptimized(std::vector<PackElement*> elements, unsigned startRow, unsigned numRows);

  // Pack in a prefix-stable way - appended elements do not affect positions of prior elements.
  unsigned PackPrefixStable(std::vector<PackElement*> elements, unsigned startRow, unsigned numRows);

  bool UseMinPrecision() const { return m_bUseMinPrecision; }

protected:
  std::vector<PackedRegister> m_Registers;
  bool m_bIgnoreIndexing;
  bool m_bUseMinPrecision;
};


} // namespace hlsl
