//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// system_utils.h: declaration of OS-specific utility functions

#ifndef COMMON_SYSTEM_UTILS_H_
#define COMMON_SYSTEM_UTILS_H_

#include "common/Optional.h"
#include "common/angleutils.h"

namespace angle
{
std::string GetExecutablePath();
std::string GetExecutableDirectory();
std::string GetResourceDirectory();
const char *GetSharedLibraryExtension();
Optional<std::string> GetCWD();
bool SetCWD(const char *dirName);
bool SetEnvironmentVar(const char *variableName, const char *value);
bool UnsetEnvironmentVar(const char *variableName);
std::string GetEnvironmentVar(const char *variableName);
const char *GetPathSeparator();
bool PrependPathToEnvironmentVar(const char *variableName, const char *path);
bool IsDirectory(const char *filename);

// Get absolute time in seconds.  Use this function to get an absolute time with an unknown origin.
double GetCurrentTime();

// Run an application and get the output.  Gets a nullptr-terminated set of args to execute the
// application with, and returns the stdout and stderr outputs as well as the exit code.
//
// Pass nullptr for stdoutOut/stderrOut if you don't need to capture. exitCodeOut is required.
//
// Returns false if it fails to actually execute the application.
bool RunApp(const std::vector<const char *> &args,
            std::string *stdoutOut,
            std::string *stderrOut,
            int *exitCodeOut);

class Library : angle::NonCopyable
{
  public:
    virtual ~Library() {}
    virtual void *getSymbol(const char *symbolName) = 0;
    virtual void *getNative() const                 = 0;

    template <typename FuncT>
    void getAs(const char *symbolName, FuncT *funcOut)
    {
        *funcOut = reinterpret_cast<FuncT>(getSymbol(symbolName));
    }
};

// Use SYSTEM_DIR to bypass loading ANGLE libraries with the same name as system DLLS
// (e.g. opengl32.dll)
enum class SearchType
{
    ApplicationDir,
    SystemDir
};

Library *OpenSharedLibrary(const char *libraryName, SearchType searchType);

// Returns true if the process is currently being debugged.
bool IsDebuggerAttached();

// Calls system APIs to break into the debugger.
void BreakDebugger();
}  // namespace angle

#endif  // COMMON_SYSTEM_UTILS_H_
