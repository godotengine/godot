/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_PROCESS_POSIX

#include <dirent.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include "../SDL_sysprocess.h"
#include "../../io/SDL_iostream_c.h"


#if defined(HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR_NP) && \
    !defined(HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR)
#define HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR
#define posix_spawn_file_actions_addchdir posix_spawn_file_actions_addchdir_np
#endif

#define READ_END 0
#define WRITE_END 1

struct SDL_ProcessData {
    pid_t pid;
};

static void CleanupStream(void *userdata, void *value)
{
    SDL_Process *process = (SDL_Process *)value;
    const char *property = (const char *)userdata;

    SDL_ClearProperty(process->props, property);
}

static bool SetupStream(SDL_Process *process, int fd, const char *mode, const char *property)
{
    // Set the file descriptor to non-blocking mode
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

    SDL_IOStream *io = SDL_IOFromFD(fd, true);
    if (!io) {
        return false;
    }

    SDL_SetPointerPropertyWithCleanup(SDL_GetIOProperties(io), "SDL.internal.process", process, CleanupStream, (void *)property);
    SDL_SetPointerProperty(process->props, property, io);
    return true;
}

static void IgnoreSignal(int sig)
{
    struct sigaction action;

    sigaction(SIGPIPE, NULL, &action);
#ifdef HAVE_SA_SIGACTION
    if (action.sa_handler == SIG_DFL && (void (*)(int))action.sa_sigaction == SIG_DFL) {
#else
    if (action.sa_handler == SIG_DFL) {
#endif
        action.sa_handler = SIG_IGN;
        sigaction(sig, &action, NULL);
    }
}

static bool CreatePipe(int fds[2])
{
    if (pipe(fds) < 0) {
        return false;
    }

    // Make sure the pipe isn't accidentally inherited by another thread creating a process
    fcntl(fds[READ_END], F_SETFD, fcntl(fds[READ_END], F_GETFD) | FD_CLOEXEC);
    fcntl(fds[WRITE_END], F_SETFD, fcntl(fds[WRITE_END], F_GETFD) | FD_CLOEXEC);

    // Make sure we don't crash if we write when the pipe is closed
    IgnoreSignal(SIGPIPE);

    return true;
}

static bool GetStreamFD(SDL_PropertiesID props, const char *property, int *result)
{
    SDL_IOStream *io = (SDL_IOStream *)SDL_GetPointerProperty(props, property, NULL);
    if (!io) {
        SDL_SetError("%s is not set", property);
        return false;
    }

    int fd = (int)SDL_GetNumberProperty(SDL_GetIOProperties(io), SDL_PROP_IOSTREAM_FILE_DESCRIPTOR_NUMBER, -1);
    if (fd < 0) {
        SDL_SetError("%s doesn't have SDL_PROP_IOSTREAM_FILE_DESCRIPTOR_NUMBER available", property);
        return false;
    }

    *result = fd;
    return true;
}

static bool AddFileDescriptorCloseActions(posix_spawn_file_actions_t *fa)
{
    DIR *dir = opendir("/proc/self/fd");
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            int fd = SDL_atoi(entry->d_name);
            if (fd <= STDERR_FILENO) {
                continue;
            }

            int flags = fcntl(fd, F_GETFD);
            if (flags < 0 || (flags & FD_CLOEXEC)) {
                continue;
            }
            if (posix_spawn_file_actions_addclose(fa, fd) != 0) {
                closedir(dir);
                return SDL_SetError("posix_spawn_file_actions_addclose failed: %s", strerror(errno));
            }
        }
        closedir(dir);
    } else {
        for (int fd = (int)(sysconf(_SC_OPEN_MAX) - 1); fd > STDERR_FILENO; --fd) {
            int flags = fcntl(fd, F_GETFD);
            if (flags < 0 || (flags & FD_CLOEXEC)) {
                continue;
            }
            if (posix_spawn_file_actions_addclose(fa, fd) != 0) {
                return SDL_SetError("posix_spawn_file_actions_addclose failed: %s", strerror(errno));
            }
        }
    }
    return true;
}

bool SDL_SYS_CreateProcessWithProperties(SDL_Process *process, SDL_PropertiesID props)
{
    char * const *args = SDL_GetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, NULL);
    SDL_Environment *env = SDL_GetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ENVIRONMENT_POINTER, SDL_GetEnvironment());
    char **envp = NULL;
    const char *working_directory = SDL_GetStringProperty(props, SDL_PROP_PROCESS_CREATE_WORKING_DIRECTORY_STRING, NULL);
    SDL_ProcessIO stdin_option = (SDL_ProcessIO)SDL_GetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_NULL);
    SDL_ProcessIO stdout_option = (SDL_ProcessIO)SDL_GetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_INHERITED);
    SDL_ProcessIO stderr_option = (SDL_ProcessIO)SDL_GetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER, SDL_PROCESS_STDIO_INHERITED);
    bool redirect_stderr = SDL_GetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_TO_STDOUT_BOOLEAN, false) &&
                           !SDL_HasProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER);
    int stdin_pipe[2] = { -1, -1 };
    int stdout_pipe[2] = { -1, -1 };
    int stderr_pipe[2] = { -1, -1 };
    int fd = -1;

    // Keep the malloc() before exec() so that an OOM won't run a process at all
    envp = SDL_GetEnvironmentVariables(env);
    if (!envp) {
        return false;
    }

    SDL_ProcessData *data = SDL_calloc(1, sizeof(*data));
    if (!data) {
        SDL_free(envp);
        return false;
    }
    process->internal = data;

    posix_spawnattr_t attr;
    posix_spawn_file_actions_t fa;

    if (posix_spawnattr_init(&attr) != 0) {
        SDL_SetError("posix_spawnattr_init failed: %s", strerror(errno));
        goto posix_spawn_fail_none;
    }

    if (posix_spawn_file_actions_init(&fa) != 0) {
        SDL_SetError("posix_spawn_file_actions_init failed: %s", strerror(errno));
        goto posix_spawn_fail_attr;
    }

    if (working_directory) {
#ifdef HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR
#ifdef SDL_PLATFORM_APPLE
        if (__builtin_available(macOS 10.15, *)) {
            if (posix_spawn_file_actions_addchdir(&fa, working_directory) != 0) {
                SDL_SetError("posix_spawn_file_actions_addchdir failed: %s", strerror(errno));
                goto posix_spawn_fail_all;
            }
        } else {
            SDL_SetError("Setting the working directory is only supported on macOS 10.15 and newer");
            goto posix_spawn_fail_all;
        }
#else
        if (posix_spawn_file_actions_addchdir(&fa, working_directory) != 0) {
            SDL_SetError("posix_spawn_file_actions_addchdir failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
#endif // SDL_PLATFORM_APPLE
#else
        SDL_SetError("Setting the working directory is not supported");
        goto posix_spawn_fail_all;
#endif
    }

    // Background processes don't have access to the terminal
    if (process->background) {
        if (stdin_option == SDL_PROCESS_STDIO_INHERITED) {
            stdin_option = SDL_PROCESS_STDIO_NULL;
        }
        if (stdout_option == SDL_PROCESS_STDIO_INHERITED) {
            stdout_option = SDL_PROCESS_STDIO_NULL;
        }
        if (stderr_option == SDL_PROCESS_STDIO_INHERITED) {
            stderr_option = SDL_PROCESS_STDIO_NULL;
        }
    }

    switch (stdin_option) {
    case SDL_PROCESS_STDIO_REDIRECT:
        if (!GetStreamFD(props, SDL_PROP_PROCESS_CREATE_STDIN_POINTER, &fd)) {
            goto posix_spawn_fail_all;
        }
        if (posix_spawn_file_actions_adddup2(&fa, fd, STDIN_FILENO) != 0) {
            SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
        break;
    case SDL_PROCESS_STDIO_APP:
        if (!CreatePipe(stdin_pipe)) {
            goto posix_spawn_fail_all;
        }
        if (posix_spawn_file_actions_adddup2(&fa, stdin_pipe[READ_END], STDIN_FILENO) != 0) {
            SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
        break;
    case SDL_PROCESS_STDIO_NULL:
        if (posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null", O_RDONLY, 0) != 0) {
            SDL_SetError("posix_spawn_file_actions_addopen failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
        break;
    case SDL_PROCESS_STDIO_INHERITED:
    default:
        break;
    }

    switch (stdout_option) {
    case SDL_PROCESS_STDIO_REDIRECT:
        if (!GetStreamFD(props, SDL_PROP_PROCESS_CREATE_STDOUT_POINTER, &fd)) {
            goto posix_spawn_fail_all;
        }
        if (posix_spawn_file_actions_adddup2(&fa, fd, STDOUT_FILENO) != 0) {
            SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
        break;
    case SDL_PROCESS_STDIO_APP:
        if (!CreatePipe(stdout_pipe)) {
            goto posix_spawn_fail_all;
        }
        if (posix_spawn_file_actions_adddup2(&fa, stdout_pipe[WRITE_END], STDOUT_FILENO) != 0) {
            SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
        break;
    case SDL_PROCESS_STDIO_NULL:
        if (posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, "/dev/null", O_WRONLY, 0644) != 0) {
            SDL_SetError("posix_spawn_file_actions_addopen failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
        break;
    case SDL_PROCESS_STDIO_INHERITED:
    default:
        break;
    }

    if (redirect_stderr) {
        if (posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO) != 0) {
            SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
    } else {
        switch (stderr_option) {
        case SDL_PROCESS_STDIO_REDIRECT:
            if (!GetStreamFD(props, SDL_PROP_PROCESS_CREATE_STDERR_POINTER, &fd)) {
                goto posix_spawn_fail_all;
            }
            if (posix_spawn_file_actions_adddup2(&fa, fd, STDERR_FILENO) != 0) {
                SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
                goto posix_spawn_fail_all;
            }
            break;
        case SDL_PROCESS_STDIO_APP:
            if (!CreatePipe(stderr_pipe)) {
                goto posix_spawn_fail_all;
            }
            if (posix_spawn_file_actions_adddup2(&fa, stderr_pipe[WRITE_END], STDERR_FILENO) != 0) {
                SDL_SetError("posix_spawn_file_actions_adddup2 failed: %s", strerror(errno));
                goto posix_spawn_fail_all;
            }
            break;
        case SDL_PROCESS_STDIO_NULL:
            if (posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0644) != 0) {
                SDL_SetError("posix_spawn_file_actions_addopen failed: %s", strerror(errno));
                goto posix_spawn_fail_all;
            }
            break;
        case SDL_PROCESS_STDIO_INHERITED:
        default:
            break;
        }
    }

    if (!AddFileDescriptorCloseActions(&fa)) {
        goto posix_spawn_fail_all;
    }

    // Spawn the new process
    if (process->background) {
        int status = -1;
        #ifdef SDL_PLATFORM_APPLE  // Apple has vfork marked as deprecated and (as of macOS 10.12) is almost identical to calling fork() anyhow.
        const pid_t pid = fork();
        const char *forkname = "fork";
        #else
        const pid_t pid = vfork();
        const char *forkname = "vfork";
        #endif
        switch (pid) {
        case -1:
            SDL_SetError("%s() failed: %s", forkname, strerror(errno));
            goto posix_spawn_fail_all;

        case 0:
            // Detach from the terminal and launch the process
            setsid();
            if (posix_spawnp(&data->pid, args[0], &fa, &attr, args, envp) != 0) {
                _exit(errno);
            }
            _exit(0);

        default:
            if (waitpid(pid, &status, 0) < 0) {
                SDL_SetError("waitpid() failed: %s", strerror(errno));
                goto posix_spawn_fail_all;
            }
            if (status != 0) {
                SDL_SetError("posix_spawn() failed: %s", strerror(status));
                goto posix_spawn_fail_all;
            }
            break;
        }
    } else {
        if (posix_spawnp(&data->pid, args[0], &fa, &attr, args, envp) != 0) {
            SDL_SetError("posix_spawn() failed: %s", strerror(errno));
            goto posix_spawn_fail_all;
        }
    }
    SDL_SetNumberProperty(process->props, SDL_PROP_PROCESS_PID_NUMBER, data->pid);

    if (stdin_option == SDL_PROCESS_STDIO_APP) {
        if (!SetupStream(process, stdin_pipe[WRITE_END], "wb", SDL_PROP_PROCESS_STDIN_POINTER)) {
            close(stdin_pipe[WRITE_END]);
        }
        close(stdin_pipe[READ_END]);
    }

    if (stdout_option == SDL_PROCESS_STDIO_APP) {
        if (!SetupStream(process, stdout_pipe[READ_END], "rb", SDL_PROP_PROCESS_STDOUT_POINTER)) {
            close(stdout_pipe[READ_END]);
        }
        close(stdout_pipe[WRITE_END]);
    }

    if (stderr_option == SDL_PROCESS_STDIO_APP) {
        if (!SetupStream(process, stderr_pipe[READ_END], "rb", SDL_PROP_PROCESS_STDERR_POINTER)) {
            close(stderr_pipe[READ_END]);
        }
        close(stderr_pipe[WRITE_END]);
    }

    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&attr);
    SDL_free(envp);

    return true;

    /* --------------------------------------------------------------------- */

posix_spawn_fail_all:
    posix_spawn_file_actions_destroy(&fa);

posix_spawn_fail_attr:
    posix_spawnattr_destroy(&attr);

posix_spawn_fail_none:
    if (stdin_pipe[READ_END] >= 0) {
        close(stdin_pipe[READ_END]);
    }
    if (stdin_pipe[WRITE_END] >= 0) {
        close(stdin_pipe[WRITE_END]);
    }
    if (stdout_pipe[READ_END] >= 0) {
        close(stdout_pipe[READ_END]);
    }
    if (stdout_pipe[WRITE_END] >= 0) {
        close(stdout_pipe[WRITE_END]);
    }
    if (stderr_pipe[READ_END] >= 0) {
        close(stderr_pipe[READ_END]);
    }
    if (stderr_pipe[WRITE_END] >= 0) {
        close(stderr_pipe[WRITE_END]);
    }
    SDL_free(envp);
    return false;
}

bool SDL_SYS_KillProcess(SDL_Process *process, bool force)
{
    int ret = kill(process->internal->pid, force ? SIGKILL : SIGTERM);
    if (ret == 0) {
        return true;
    } else {
        return SDL_SetError("Could not kill(): %s", strerror(errno));
    }
}

bool SDL_SYS_WaitProcess(SDL_Process *process, bool block, int *exitcode)
{
    int wstatus = 0;
    int ret;
    pid_t pid = process->internal->pid;

    if (process->background) {
        // We can't wait on the status, so we'll poll to see if it's alive
        if (block) {
            while (kill(pid, 0) == 0) {
                SDL_Delay(10);
            }
        } else {
            if (kill(pid, 0) == 0) {
                return false;
            }
        }
        *exitcode = 0;
        return true;
    } else {
        ret = waitpid(pid, &wstatus, block ? 0 : WNOHANG);
        if (ret < 0) {
            return SDL_SetError("Could not waitpid(): %s", strerror(errno));
        }

        if (ret == 0) {
            SDL_ClearError();
            return false;
        }

        if (WIFEXITED(wstatus)) {
            *exitcode = WEXITSTATUS(wstatus);
        } else if (WIFSIGNALED(wstatus)) {
            *exitcode = -WTERMSIG(wstatus);
        } else {
            *exitcode = -255;
        }

        return true;
    }
}

void SDL_SYS_DestroyProcess(SDL_Process *process)
{
    SDL_IOStream *io;

    io = (SDL_IOStream *)SDL_GetPointerProperty(process->props, SDL_PROP_PROCESS_STDIN_POINTER, NULL);
    if (io) {
        SDL_CloseIO(io);
    }

    io = (SDL_IOStream *)SDL_GetPointerProperty(process->props, SDL_PROP_PROCESS_STDOUT_POINTER, NULL);
    if (io) {
        SDL_CloseIO(io);
    }

    io = (SDL_IOStream *)SDL_GetPointerProperty(process->props, SDL_PROP_PROCESS_STDERR_POINTER, NULL);
    if (io) {
        SDL_CloseIO(io);
    }

    SDL_free(process->internal);
}

#endif // SDL_PROCESS_POSIX
