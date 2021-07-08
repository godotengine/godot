//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// system_utils_posix.cpp: Implementation of POSIX OS-specific functions.

#include "system_utils.h"

#include <array>

#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace angle
{

namespace
{
struct ScopedPipe
{
    ~ScopedPipe()
    {
        closeEndPoint(0);
        closeEndPoint(1);
    }
    void closeEndPoint(int index)
    {
        if (fds[index] >= 0)
        {
            close(fds[index]);
            fds[index] = -1;
        }
    }
    int fds[2] = {
        -1,
        -1,
    };
};

void ReadEntireFile(int fd, std::string *out)
{
    out->clear();

    while (true)
    {
        char buffer[256];
        ssize_t bytesRead = read(fd, buffer, sizeof(buffer));

        // If interrupted, retry.
        if (bytesRead < 0 && errno == EINTR)
        {
            continue;
        }

        // If failed, or nothing to read, we are done.
        if (bytesRead <= 0)
        {
            break;
        }

        out->append(buffer, bytesRead);
    }
}
}  // anonymous namespace

std::string GetResourceDirectory()
{
    return GetExecutableDirectory();
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
    return ":";
}

bool RunApp(const std::vector<const char *> &args,
            std::string *stdoutOut,
            std::string *stderrOut,
            int *exitCodeOut)
{
    if (args.size() == 0 || args.back() != nullptr)
    {
        return false;
    }

    ScopedPipe stdoutPipe;
    ScopedPipe stderrPipe;

    // Create pipes for stdout and stderr.
    if (stdoutOut && pipe(stdoutPipe.fds) != 0)
    {
        return false;
    }
    if (stderrOut && pipe(stderrPipe.fds) != 0)
    {
        return false;
    }

    pid_t pid = fork();
    if (pid < 0)
    {
        return false;
    }

    if (pid == 0)
    {
        // Child.  Execute the application.

        // Redirect stdout and stderr to the pipe fds.
        if (stdoutOut)
        {
            if (dup2(stdoutPipe.fds[1], STDOUT_FILENO) < 0)
            {
                _exit(errno);
            }
        }
        if (stderrOut)
        {
            if (dup2(stderrPipe.fds[1], STDERR_FILENO) < 0)
            {
                _exit(errno);
            }
        }

        // Execute the application, which doesn't return unless failed.  Note: execv takes argv as
        // `char * const *` for historical reasons.  It is safe to const_cast it:
        //
        // http://pubs.opengroup.org/onlinepubs/9699919799/functions/exec.html
        //
        // > The statement about argv[] and envp[] being constants is included to make explicit to
        // future writers of language bindings that these objects are completely constant. Due to a
        // limitation of the ISO C standard, it is not possible to state that idea in standard C.
        // Specifying two levels of const- qualification for the argv[] and envp[] parameters for
        // the exec functions may seem to be the natural choice, given that these functions do not
        // modify either the array of pointers or the characters to which the function points, but
        // this would disallow existing correct code. Instead, only the array of pointers is noted
        // as constant.
        execv(args[0], const_cast<char *const *>(args.data()));
        _exit(errno);
    }

    // Parent.  Read child output from the pipes and clean it up.

    // Close the write end of the pipes, so EOF can be generated when child exits.
    stdoutPipe.closeEndPoint(1);
    stderrPipe.closeEndPoint(1);

    // Read back the output of the child.
    if (stdoutOut)
    {
        ReadEntireFile(stdoutPipe.fds[0], stdoutOut);
    }
    if (stderrOut)
    {
        ReadEntireFile(stderrPipe.fds[0], stderrOut);
    }

    // Cleanup the child.
    int status = 0;
    do
    {
        pid_t changedPid = waitpid(pid, &status, 0);
        if (changedPid < 0 && errno == EINTR)
        {
            continue;
        }
        if (changedPid < 0)
        {
            return false;
        }
    } while (!WIFEXITED(status) && !WIFSIGNALED(status));

    // Retrieve the error code.
    if (exitCodeOut)
    {
        *exitCodeOut = WEXITSTATUS(status);
    }

    return true;
}

class PosixLibrary : public Library
{
  public:
    PosixLibrary(const char *libraryName, SearchType searchType)
    {
        std::string directory;
        if (searchType == SearchType::ApplicationDir)
        {
            static int dummySymbol = 0;
            Dl_info dlInfo;
            if (dladdr(&dummySymbol, &dlInfo) != 0)
            {
                std::string moduleName = dlInfo.dli_fname;
                directory              = moduleName.substr(0, moduleName.find_last_of('/') + 1);
            }
        }

        std::string fullPath = directory + libraryName + "." + GetSharedLibraryExtension();
        mModule              = dlopen(fullPath.c_str(), RTLD_NOW);
    }

    ~PosixLibrary() override
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
    return new PosixLibrary(libraryName, searchType);
}

bool IsDirectory(const char *filename)
{
    struct stat st;
    int result = stat(filename, &st);
    return result == 0 && ((st.st_mode & S_IFDIR) == S_IFDIR);
}

bool IsDebuggerAttached()
{
    // This could have a fuller implementation.
    // See https://cs.chromium.org/chromium/src/base/debug/debugger_posix.cc
    return false;
}

void BreakDebugger()
{
    // This could have a fuller implementation.
    // See https://cs.chromium.org/chromium/src/base/debug/debugger_posix.cc
    abort();
}
}  // namespace angle
