//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_QUALIFIER_TYPES_H_
#define COMPILER_TRANSLATOR_QUALIFIER_TYPES_H_

#include "common/angleutils.h"
#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Types.h"

namespace sh
{
class TDiagnostics;

TLayoutQualifier JoinLayoutQualifiers(TLayoutQualifier leftQualifier,
                                      TLayoutQualifier rightQualifier,
                                      const TSourceLoc &rightQualifierLocation,
                                      TDiagnostics *diagnostics);

enum TQualifierType
{
    QtInvariant,
    QtInterpolation,
    QtLayout,
    QtStorage,
    QtPrecision,
    QtMemory
};

class TQualifierWrapperBase : angle::NonCopyable
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    TQualifierWrapperBase(const TSourceLoc &line) : mLine(line) {}
    virtual ~TQualifierWrapperBase() {}
    virtual TQualifierType getType() const             = 0;
    virtual ImmutableString getQualifierString() const = 0;
    virtual unsigned int getRank() const               = 0;
    const TSourceLoc &getLine() const { return mLine; }

  private:
    TSourceLoc mLine;
};

class TInvariantQualifierWrapper final : public TQualifierWrapperBase
{
  public:
    TInvariantQualifierWrapper(const TSourceLoc &line) : TQualifierWrapperBase(line) {}
    ~TInvariantQualifierWrapper() {}

    TQualifierType getType() const override { return QtInvariant; }
    ImmutableString getQualifierString() const override { return ImmutableString("invariant"); }
    unsigned int getRank() const override;
};

class TInterpolationQualifierWrapper final : public TQualifierWrapperBase
{
  public:
    TInterpolationQualifierWrapper(TQualifier interpolationQualifier, const TSourceLoc &line)
        : TQualifierWrapperBase(line), mInterpolationQualifier(interpolationQualifier)
    {}
    ~TInterpolationQualifierWrapper() {}

    TQualifierType getType() const override { return QtInterpolation; }
    ImmutableString getQualifierString() const override
    {
        return ImmutableString(sh::getQualifierString(mInterpolationQualifier));
    }
    TQualifier getQualifier() const { return mInterpolationQualifier; }
    unsigned int getRank() const override;

  private:
    TQualifier mInterpolationQualifier;
};

class TLayoutQualifierWrapper final : public TQualifierWrapperBase
{
  public:
    TLayoutQualifierWrapper(TLayoutQualifier layoutQualifier, const TSourceLoc &line)
        : TQualifierWrapperBase(line), mLayoutQualifier(layoutQualifier)
    {}
    ~TLayoutQualifierWrapper() {}

    TQualifierType getType() const override { return QtLayout; }
    ImmutableString getQualifierString() const override { return ImmutableString("layout"); }
    const TLayoutQualifier &getQualifier() const { return mLayoutQualifier; }
    unsigned int getRank() const override;

  private:
    TLayoutQualifier mLayoutQualifier;
};

class TStorageQualifierWrapper final : public TQualifierWrapperBase
{
  public:
    TStorageQualifierWrapper(TQualifier storageQualifier, const TSourceLoc &line)
        : TQualifierWrapperBase(line), mStorageQualifier(storageQualifier)
    {}
    ~TStorageQualifierWrapper() {}

    TQualifierType getType() const override { return QtStorage; }
    ImmutableString getQualifierString() const override
    {
        return ImmutableString(sh::getQualifierString(mStorageQualifier));
    }
    TQualifier getQualifier() const { return mStorageQualifier; }
    unsigned int getRank() const override;

  private:
    TQualifier mStorageQualifier;
};

class TPrecisionQualifierWrapper final : public TQualifierWrapperBase
{
  public:
    TPrecisionQualifierWrapper(TPrecision precisionQualifier, const TSourceLoc &line)
        : TQualifierWrapperBase(line), mPrecisionQualifier(precisionQualifier)
    {}
    ~TPrecisionQualifierWrapper() {}

    TQualifierType getType() const override { return QtPrecision; }
    ImmutableString getQualifierString() const override
    {
        return ImmutableString(sh::getPrecisionString(mPrecisionQualifier));
    }
    TPrecision getQualifier() const { return mPrecisionQualifier; }
    unsigned int getRank() const override;

  private:
    TPrecision mPrecisionQualifier;
};

class TMemoryQualifierWrapper final : public TQualifierWrapperBase
{
  public:
    TMemoryQualifierWrapper(TQualifier memoryQualifier, const TSourceLoc &line)
        : TQualifierWrapperBase(line), mMemoryQualifier(memoryQualifier)
    {}
    ~TMemoryQualifierWrapper() {}

    TQualifierType getType() const override { return QtMemory; }
    ImmutableString getQualifierString() const override
    {
        return ImmutableString(sh::getQualifierString(mMemoryQualifier));
    }
    TQualifier getQualifier() const { return mMemoryQualifier; }
    unsigned int getRank() const override;

  private:
    TQualifier mMemoryQualifier;
};

// TTypeQualifier tightly covers type_qualifier from the grammar
struct TTypeQualifier
{
    // initializes all of the qualifiers and sets the scope
    TTypeQualifier(TQualifier scope, const TSourceLoc &loc);

    TLayoutQualifier layoutQualifier;
    TMemoryQualifier memoryQualifier;
    TPrecision precision;
    TQualifier qualifier;
    bool invariant;
    TSourceLoc line;
};

// TTypeQualifierBuilder contains all of the qualifiers when type_qualifier gets parsed.
// It is to be used to validate the qualifier sequence and build a TTypeQualifier from it.
class TTypeQualifierBuilder : angle::NonCopyable
{
  public:
    using QualifierSequence = TVector<const TQualifierWrapperBase *>;

  public:
    POOL_ALLOCATOR_NEW_DELETE
    TTypeQualifierBuilder(const TStorageQualifierWrapper *scope, int shaderVersion);
    // Adds the passed qualifier to the end of the sequence.
    void appendQualifier(const TQualifierWrapperBase *qualifier);
    // Checks for the order of qualification and repeating qualifiers.
    bool checkSequenceIsValid(TDiagnostics *diagnostics) const;
    // Goes over the qualifier sequence and parses it to form a type qualifier for a function
    // parameter.
    // The returned object is initialized even if the parsing fails.
    TTypeQualifier getParameterTypeQualifier(TDiagnostics *diagnostics) const;
    // Goes over the qualifier sequence and parses it to form a type qualifier for a variable.
    // The returned object is initialized even if the parsing fails.
    TTypeQualifier getVariableTypeQualifier(TDiagnostics *diagnostics) const;

  private:
    QualifierSequence mQualifiers;
    int mShaderVersion;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_QUALIFIER_TYPES_H_
