//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#define CATCH_CONFIG_RUNNER

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXFormat/File.h>

namespace mx = MaterialX;

int main(int argc, char* const argv[])
{
    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;

#ifdef CATCH_PLATFORM_WINDOWS
    BOOL inDebugger = IsDebuggerPresent();
    if (inDebugger)
    {
        session.configData().outputFilename = "%debug";
    }
    else
    {
        session.configData().outputFilename = "";
    }
#endif

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0)
    {
        return returnCode;
    }

    return session.run();
}
