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

#include "Logger.h"

#include <algorithm>
#include <iterator>
#include <sstream>

namespace spv {

void SpvBuildLogger::tbdFunctionality(const std::string& f)
{
    if (std::find(std::begin(tbdFeatures), std::end(tbdFeatures), f) == std::end(tbdFeatures))
        tbdFeatures.push_back(f);
}

void SpvBuildLogger::missingFunctionality(const std::string& f)
{
    if (std::find(std::begin(missingFeatures), std::end(missingFeatures), f) == std::end(missingFeatures))
        missingFeatures.push_back(f);
}

std::string SpvBuildLogger::getAllMessages() const {
    std::ostringstream messages;
    for (auto it = tbdFeatures.cbegin(); it != tbdFeatures.cend(); ++it)
        messages << "TBD functionality: " << *it << "\n";
    for (auto it = missingFeatures.cbegin(); it != missingFeatures.cend(); ++it)
        messages << "Missing functionality: " << *it << "\n";
    for (auto it = warnings.cbegin(); it != warnings.cend(); ++it)
        messages << "warning: " << *it << "\n";
    for (auto it = errors.cbegin(); it != errors.cend(); ++it)
        messages << "error: " << *it << "\n";
    return messages.str();
}

} // end spv namespace
