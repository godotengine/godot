//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_PRAGMA_H_
#define COMPILER_TRANSLATOR_PRAGMA_H_

struct TPragma
{
    struct STDGL
    {
        STDGL() : invariantAll(false) {}

        bool invariantAll;
    };

    // By default optimization is turned on and debug is turned off.
    // Precision emulation is turned on by default, but has no effect unless
    // the extension is enabled.
    TPragma() : optimize(true), debug(false), debugShaderPrecision(true) {}
    TPragma(bool o, bool d) : optimize(o), debug(d), debugShaderPrecision(true) {}

    bool optimize;
    bool debug;
    bool debugShaderPrecision;
    STDGL stdgl;
};

#endif  // COMPILER_TRANSLATOR_PRAGMA_H_
