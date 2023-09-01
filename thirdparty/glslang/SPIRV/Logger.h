//
// Copyright (C) 2016 Google, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of Google Inc. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef GLSLANG_SPIRV_LOGGER_H
#define GLSLANG_SPIRV_LOGGER_H

#include <string>
#include <vector>

namespace spv {

// A class for holding all SPIR-V build status messages, including
// missing/TBD functionalities, warnings, and errors.
class SpvBuildLogger {
public:
    SpvBuildLogger() {}

    // Registers a TBD functionality.
    void tbdFunctionality(const std::string& f);
    // Registers a missing functionality.
    void missingFunctionality(const std::string& f);

    // Logs a warning.
    void warning(const std::string& w) { warnings.push_back(w); }
    // Logs an error.
    void error(const std::string& e) { errors.push_back(e); }

    // Returns all messages accumulated in the order of:
    // TBD functionalities, missing functionalities, warnings, errors.
    std::string getAllMessages() const;

private:
    SpvBuildLogger(const SpvBuildLogger&);

    std::vector<std::string> tbdFeatures;
    std::vector<std::string> missingFeatures;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

} // end spv namespace

#endif // GLSLANG_SPIRV_LOGGER_H
