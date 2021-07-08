//
// Copyright (c) 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// system_utils_cocoa.mm: Implementation of OS-specific functions for Apple's
// iOS and macOS apps running on Cocoa frameworks.
// This file is different from system_utils_mac.cpp is that system_utils_mac.cpp
// is used for traditional command line apps using .dylib as shared library.
// While this file is used for Cocoa app bundles that embed .framework as shared libraries.

#include "system_utils.h"

#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <vector>

#include <array>

namespace angle
{
std::string GetExecutablePath()
{
    return [[NSBundle mainBundle] executablePath].UTF8String;
}

std::string GetExecutableDirectory()
{
    std::string executablePath = GetExecutablePath();
    size_t lastPathSepLoc      = executablePath.find_last_of("/");
    return (lastPathSepLoc != std::string::npos) ? executablePath.substr(0, lastPathSepLoc) : "";
}

std::string GetResourceDirectory()
{
    return [[NSBundle mainBundle] resourcePath].UTF8String;
}

const char *GetSharedLibraryExtension()
{
    return "framework";
}

Optional<std::string> GetCWD()
{
    std::array<char, 4096> pathBuf;
    char *result = getcwd(pathBuf.data(), pathBuf.size());
    if (result == nullptr)
    {
        return Optional<std::string>::Invalid();
    }
    return std::string(pathBuf.data());
}

bool SetCWD(const char *dirName)
{
    return (chdir(dirName) == 0);
}

bool UnsetEnvironmentVar(const char *variableName)
{
    return (unsetenv(variableName) == 0);
}

bool SetEnvironmentVar(const char *variableName, const char *value)
{
    return (setenv(variableName, value, 1) == 0);
}

std::string GetEnvironmentVar(const char *variableName)
{
    const char *value = getenv(variableName);
    return (value == nullptr ? std::string() : std::string(value));
}

const char *GetPathSeparator()
{
    return "/";
}

bool RunApp(const std::vector<const char *> &args,
            std::string *stdoutOut,
            std::string *stderrOut,
            int *exitCodeOut)
{
    // Not supported.
    return false;
}

class FrameworkLibrary : public Library
{
  public:
    FrameworkLibrary(const char *libraryName)
    {
        char buffer[4096];
        int ret = snprintf(buffer, 4096, "%s/%s.framework/%s",
                           [[NSBundle mainBundle] privateFrameworksPath].UTF8String, libraryName,
                           libraryName);
        if (ret > 0 && ret < 4096)
        {
            mModule = dlopen(buffer, RTLD_NOW);
        }
    }

    ~FrameworkLibrary() override
    {
        if (mModule)
        {
            dlclose(mModule);
        }
    }

    void *getSymbol(const char *symbolName) override
    {
        if (!mModule)
        {
            return nullptr;
        }

        return dlsym(mModule, symbolName);
    }

    void *getNative() const override { return mModule; }

  private:
    void *mModule = nullptr;
};

Library *OpenSharedLibrary(const char *libraryName, SearchType searchType)
{
    return new FrameworkLibrary(libraryName);
}

bool IsDirectory(const char *filename)
{
    struct stat st;
    int result = stat(filename, &st);
    return result == 0 && ((st.st_mode & S_IFDIR) == S_IFDIR);
}

bool IsDebuggerAttached()
{
    // TODO(hqle).
    return false;
}

void BreakDebugger()
{
    // TODO(hqle).
    abort();
}

double GetCurrentTime()
{
    mach_timebase_info_data_t timebaseInfo;
    mach_timebase_info(&timebaseInfo);

    double secondCoeff = timebaseInfo.numer * 1e-9 / timebaseInfo.denom;
    return secondCoeff * mach_absolute_time();
}

}  // namespace angle
