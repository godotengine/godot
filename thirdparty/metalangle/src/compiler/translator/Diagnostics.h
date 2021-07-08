//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_DIAGNOSTICS_H_
#define COMPILER_TRANSLATOR_DIAGNOSTICS_H_

#include "common/angleutils.h"
#include "compiler/preprocessor/DiagnosticsBase.h"
#include "compiler/translator/Severity.h"

namespace sh
{

class TInfoSinkBase;
struct TSourceLoc;

class TDiagnostics : public angle::pp::Diagnostics, angle::NonCopyable
{
  public:
    TDiagnostics(TInfoSinkBase &infoSink);
    ~TDiagnostics() override;

    int numErrors() const { return mNumErrors; }
    int numWarnings() const { return mNumWarnings; }

    void error(const angle::pp::SourceLocation &loc, const char *reason, const char *token);
    void warning(const angle::pp::SourceLocation &loc, const char *reason, const char *token);

    void error(const TSourceLoc &loc, const char *reason, const char *token);
    void warning(const TSourceLoc &loc, const char *reason, const char *token);

    void globalError(const char *message);

    void resetErrorCount();

  protected:
    void writeInfo(Severity severity,
                   const angle::pp::SourceLocation &loc,
                   const char *reason,
                   const char *token);

    void print(ID id, const angle::pp::SourceLocation &loc, const std::string &text) override;

  private:
    TInfoSinkBase &mInfoSink;
    int mNumErrors;
    int mNumWarnings;
};

// Diagnostics wrapper to use when the code is only allowed to generate warnings.
class PerformanceDiagnostics : public angle::NonCopyable
{
  public:
    PerformanceDiagnostics(TDiagnostics *diagnostics);

    void warning(const TSourceLoc &loc, const char *reason, const char *token);

  private:
    TDiagnostics *mDiagnostics;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_DIAGNOSTICS_H_
