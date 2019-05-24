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

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "Initializer.h"
#include "Settings.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::unique_ptr<glslangtest::GlslangInitializer> initializer(
        new glslangtest::GlslangInitializer);

    glslangtest::GlobalTestSettings.initializer = initializer.get();

    for (int i = 1; i < argc; ++i) {
        if (std::string("--update-mode") == argv[i]) {
            glslangtest::GlobalTestSettings.updateMode = true;
        }
        if (std::string("--test-root") == argv[i]) {
            // Allow the user set the test root directory.  This is useful
            // for testing with files from another source tree.
            if (i + 1 < argc) {
                glslangtest::GlobalTestSettings.testRoot = argv[i + 1];
                i++;
            } else {
                printf("error: --test-root requires an argument\n");
                return 1;
            }
        }
        if (std::string("--help") == argv[i]) {
            printf("\nExtra options:\n\n");
            printf("  --update-mode\n      Update the golden results for the tests.\n");
            printf("  --test-root <arg>\n      Specify the test root directory (useful for testing with\n      files from another source tree).\n");
        }
    }

    const int result = RUN_ALL_TESTS();

    glslangtest::GlobalTestSettings.initializer = nullptr;

    return result;
}
