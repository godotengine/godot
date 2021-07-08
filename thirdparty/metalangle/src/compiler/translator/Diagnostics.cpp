//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/Diagnostics.h"

#include "common/debug.h"
#include "compiler/preprocessor/SourceLocation.h"
#include "compiler/translator/Common.h"
#include "compiler/translator/InfoSink.h"

namespace sh
{

TDiagnostics::TDiagnostics(TInfoSinkBase &infoSink)
    : mInfoSink(infoSink), mNumErrors(0), mNumWarnings(0)
{}

TDiagnostics::~TDiagnostics() {}

void TDiagnostics::writeInfo(Severity severity,
                             const angle::pp::SourceLocation &loc,
                             const char *reason,
                             const char *token)
{
    switch (severity)
    {
        case SH_ERROR:
            ++mNumErrors;
            break;
        case SH_WARNING:
            ++mNumWarnings;
            break;
        default:
            UNREACHABLE();
            break;
    }

    /* VC++ format: file(linenum) : error #: 'token' : extrainfo */
    mInfoSink.prefix(severity);
    mInfoSink.location(loc.file, loc.line);
    mInfoSink << "'" << token << "' : " << reason << "\n";
}

void TDiagnostics::globalError(const char *message)
{
    ++mNumErrors;
    mInfoSink.prefix(SH_ERROR);
    mInfoSink << message << "\n";
}

void TDiagnostics::error(const angle::pp::SourceLocation &loc,
                         const char *reason,
                         const char *token)
{
    writeInfo(SH_ERROR, loc, reason, token);
}

void TDiagnostics::warning(const angle::pp::SourceLocation &loc,
                           const char *reason,
                           const char *token)
{
    writeInfo(SH_WARNING, loc, reason, token);
}

void TDiagnostics::error(const TSourceLoc &loc, const char *reason, const char *token)
{
    angle::pp::SourceLocation srcLoc;
    srcLoc.file = loc.first_file;
    srcLoc.line = loc.first_line;
    error(srcLoc, reason, token);
}

void TDiagnostics::warning(const TSourceLoc &loc, const char *reason, const char *token)
{
    angle::pp::SourceLocation srcLoc;
    srcLoc.file = loc.first_file;
    srcLoc.line = loc.first_line;
    warning(srcLoc, reason, token);
}

void TDiagnostics::print(ID id, const angle::pp::SourceLocation &loc, const std::string &text)
{
    writeInfo(isError(id) ? SH_ERROR : SH_WARNING, loc, message(id), text.c_str());
}

void TDiagnostics::resetErrorCount()
{
    mNumErrors   = 0;
    mNumWarnings = 0;
}

PerformanceDiagnostics::PerformanceDiagnostics(TDiagnostics *diagnostics)
    : mDiagnostics(diagnostics)
{
    ASSERT(diagnostics);
}

void PerformanceDiagnostics::warning(const TSourceLoc &loc, const char *reason, const char *token)
{
    mDiagnostics->warning(loc, reason, token);
}

}  // namespace sh
