//
//  Portable File Dialogs
//
//  Copyright © 2018—2020 Sam Hocevar <sam@hocevar.net>
//
//  This library is free software. It comes without any warranty, to
//  the extent permitted by applicable law. You can redistribute it
//  and/or modify it under the terms of the Do What the Fuck You Want
//  to Public License, Version 2, as published by the WTFPL Task Force.
//  See http://www.wtfpl.net/ for more details.
//

#pragma once

#if _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#   define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#include <shobjidl.h> // IFileDialog
#include <shellapi.h>
#include <strsafe.h>
#include <future>     // std::async

#elif __EMSCRIPTEN__
#include <emscripten.h>

#else
#ifndef _POSIX_C_SOURCE
#   define _POSIX_C_SOURCE 2 // for popen()
#endif
#ifdef __APPLE__
#   ifndef _DARWIN_C_SOURCE
#       define _DARWIN_C_SOURCE
#   endif
#endif
#include <cstdio>     // popen()
#include <cstdlib>    // std::getenv()
#include <fcntl.h>    // fcntl()
#include <unistd.h>   // read(), pipe(), dup2()
#include <csignal>    // ::kill, std::signal
#include <sys/wait.h> // waitpid()
#endif

#include <string>   // std::string
#include <memory>   // std::shared_ptr
#include <iostream> // std::ostream
#include <map>      // std::map
#include <set>      // std::set
#include <regex>    // std::regex
#include <thread>   // std::mutex, std::this_thread
#include <chrono>   // std::chrono

// Versions of mingw64 g++ up to 9.3.0 do not have a complete IFileDialog
#ifndef PFD_HAS_IFILEDIALOG
#   define PFD_HAS_IFILEDIALOG 1
#   if (defined __MINGW64__ || defined __MINGW32__) && defined __GXX_ABI_VERSION
#       if __GXX_ABI_VERSION <= 1013
#           undef PFD_HAS_IFILEDIALOG
#           define PFD_HAS_IFILEDIALOG 0
#       endif
#   endif
#endif

namespace pfd
{

enum class button
{
    cancel = -1,
    ok,
    yes,
    no,
    abort,
    retry,
    ignore,
};

enum class choice
{
    ok = 0,
    ok_cancel,
    yes_no,
    yes_no_cancel,
    retry_cancel,
    abort_retry_ignore,
};

enum class icon
{
    info = 0,
    warning,
    error,
    question,
};

// Additional option flags for various dialog constructors
enum class opt : uint8_t
{
    none = 0,
    // For file open, allow multiselect.
    multiselect     = 0x1,
    // For file save, force overwrite and disable the confirmation dialog.
    force_overwrite = 0x2,
    // For folder select, force path to be the provided argument instead
    // of the last opened directory, which is the Microsoft-recommended,
    // user-friendly behaviour.
    force_path      = 0x4,
};

inline opt operator |(opt a, opt b) { return opt(uint8_t(a) | uint8_t(b)); }
inline bool operator &(opt a, opt b) { return bool(uint8_t(a) & uint8_t(b)); }

// The settings class, only exposing to the user a way to set verbose mode
// and to force a rescan of installed desktop helpers (zenity, kdialog…).
class settings
{
public:
    static bool available();

    static void verbose(bool value);
    static void rescan();

protected:
    explicit settings(bool resync = false);

    bool check_program(std::string const &program);

    inline bool is_osascript() const;
    inline bool is_zenity() const;
    inline bool is_kdialog() const;

    enum class flag
    {
        is_scanned = 0,
        is_verbose,

        has_zenity,
        has_matedialog,
        has_qarma,
        has_kdialog,
        is_vista,

        max_flag,
    };

    // Static array of flags for internal state
    bool const &flags(flag in_flag) const;

    // Non-const getter for the static array of flags
    bool &flags(flag in_flag);
};

// Internal classes, not to be used by client applications
namespace internal
{

// Process wait timeout, in milliseconds
static int const default_wait_timeout = 20;

class executor
{
    friend class dialog;

public:
    // High level function to get the result of a command
    std::string result(int *exit_code = nullptr);

    // High level function to abort
    bool kill();

#if _WIN32
    void start_func(std::function<std::string(int *)> const &fun);
    static BOOL CALLBACK enum_windows_callback(HWND hwnd, LPARAM lParam);
#elif __EMSCRIPTEN__
    void start(int exit_code);
#else
    void start_process(std::vector<std::string> const &command);
#endif

    ~executor();

protected:
    bool ready(int timeout = default_wait_timeout);
    void stop();

private:
    bool m_running = false;
    std::string m_stdout;
    int m_exit_code = -1;
#if _WIN32
    std::future<std::string> m_future;
    std::set<HWND> m_windows;
    std::condition_variable m_cond;
    std::mutex m_mutex;
    DWORD m_tid;
#elif __EMSCRIPTEN__ || __NX__
    // FIXME: do something
#else
    pid_t m_pid = 0;
    int m_fd = -1;
#endif
};

class platform
{
protected:
#if _WIN32
    // Helper class around LoadLibraryA() and GetProcAddress() with some safety
    class dll
    {
    public:
        dll(std::string const &name);
        ~dll();

        template<typename T> class proc
        {
        public:
            proc(dll const &lib, std::string const &sym)
              : m_proc(reinterpret_cast<T *>(::GetProcAddress(lib.handle, sym.c_str())))
            {}

            operator bool() const { return m_proc != nullptr; }
            operator T *() const { return m_proc; }

        private:
            T *m_proc;
        };

    private:
        HMODULE handle;
    };

    // Helper class around CoInitialize() and CoUnInitialize()
    class ole32_dll : public dll
    {
    public:
        ole32_dll();
        ~ole32_dll();
        bool is_initialized();

    private:
        HRESULT m_state;
    };

    // Helper class around CreateActCtx() and ActivateActCtx()
    class new_style_context
    {
    public:
        new_style_context();
        ~new_style_context();

    private:
        HANDLE create();
        ULONG_PTR m_cookie = 0;
    };
#endif
};

class dialog : protected settings, protected platform
{
public:
    bool ready(int timeout = default_wait_timeout) const;
    bool kill() const;

protected:
    explicit dialog();

    std::vector<std::string> desktop_helper() const;
    static std::string buttons_to_name(choice _choice);
    static std::string get_icon_name(icon _icon);

    std::string powershell_quote(std::string const &str) const;
    std::string osascript_quote(std::string const &str) const;
    std::string shell_quote(std::string const &str) const;

    // Keep handle to executing command
    std::shared_ptr<executor> m_async;
};

class file_dialog : public dialog
{
protected:
    enum type
    {
        open,
        save,
        folder,
    };

    file_dialog(type in_type,
                std::string const &title,
                std::string const &default_path = "",
                std::vector<std::string> const &filters = {},
                opt options = opt::none);

protected:
    std::string string_result();
    std::vector<std::string> vector_result();

#if _WIN32
    static int CALLBACK bffcallback(HWND hwnd, UINT uMsg, LPARAM, LPARAM pData);
#if PFD_HAS_IFILEDIALOG
    std::string select_folder_vista(IFileDialog *ifd, bool force_path);
#endif

    std::wstring m_wtitle;
    std::wstring m_wdefault_path;

    std::vector<std::string> m_vector_result;
#endif
};

} // namespace internal

//
// The notify widget
//

class notify : public internal::dialog
{
public:
    notify(std::string const &title,
           std::string const &message,
           icon _icon = icon::info);
};

//
// The message widget
//

class message : public internal::dialog
{
public:
    message(std::string const &title,
            std::string const &text,
            choice _choice = choice::ok_cancel,
            icon _icon = icon::info);

    button result();

private:
    // Some extra logic to map the exit code to button number
    std::map<int, button> m_mappings;
};

//
// The open_file, save_file, and open_folder widgets
//

class open_file : public internal::file_dialog
{
public:
    open_file(std::string const &title,
              std::string const &default_path = "",
              std::vector<std::string> const &filters = { "All Files", "*" },
              opt options = opt::none);

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
    // Backwards compatibility
    [[deprecated("Use pfd::opt::multiselect instead of allow_multiselect")]]
#endif
#endif
    open_file(std::string const &title,
              std::string const &default_path,
              std::vector<std::string> const &filters,
              bool allow_multiselect);

    std::vector<std::string> result();
};

class save_file : public internal::file_dialog
{
public:
    save_file(std::string const &title,
              std::string const &default_path = "",
              std::vector<std::string> const &filters = { "All Files", "*" },
              opt options = opt::none);

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
    // Backwards compatibility
    [[deprecated("Use pfd::opt::force_overwrite instead of confirm_overwrite")]]
#endif
#endif
    save_file(std::string const &title,
              std::string const &default_path,
              std::vector<std::string> const &filters,
              bool confirm_overwrite);

    std::string result();
};

class select_folder : public internal::file_dialog
{
public:
    select_folder(std::string const &title,
                  std::string const &default_path = "",
                  opt options = opt::none);

    std::string result();
};

//
// Below this are all the method implementations. You may choose to define the
// macro PFD_SKIP_IMPLEMENTATION everywhere before including this header except
// in one place. This may reduce compilation times.
//

#if !defined PFD_SKIP_IMPLEMENTATION

// internal free functions implementations

namespace internal
{

#if _WIN32
static inline std::wstring str2wstr(std::string const &str)
{
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0);
    std::wstring ret(len, '\0');
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), (LPWSTR)ret.data(), (int)ret.size());
    return ret;
}

static inline std::string wstr2str(std::wstring const &str)
{
    int len = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0, nullptr, nullptr);
    std::string ret(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), (LPSTR)ret.data(), (int)ret.size(), nullptr, nullptr);
    return ret;
}

static inline bool is_vista()
{
    OSVERSIONINFOEXW osvi;
    memset(&osvi, 0, sizeof(osvi));
    DWORDLONG const mask = VerSetConditionMask(
            VerSetConditionMask(
                    VerSetConditionMask(
                            0, VER_MAJORVERSION, VER_GREATER_EQUAL),
                    VER_MINORVERSION, VER_GREATER_EQUAL),
            VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    osvi.dwMajorVersion = HIBYTE(_WIN32_WINNT_VISTA);
    osvi.dwMinorVersion = LOBYTE(_WIN32_WINNT_VISTA);
    osvi.wServicePackMajor = 0;

    return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, mask) != FALSE;
}
#endif

// This is necessary until C++20 which will have std::string::ends_with() etc.

static inline bool ends_with(std::string const &str, std::string const &suffix)
{
    return suffix.size() <= str.size() &&
        str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static inline bool starts_with(std::string const &str, std::string const &prefix)
{
    return prefix.size() <= str.size() &&
        str.compare(0, prefix.size(), prefix) == 0;
}

} // namespace internal

// settings implementation

inline settings::settings(bool resync)
{
    flags(flag::is_scanned) &= !resync;

    if (flags(flag::is_scanned))
        return;

#if _WIN32
    flags(flag::is_vista) = internal::is_vista();
#elif !__APPLE__
    flags(flag::has_zenity) = check_program("zenity");
    flags(flag::has_matedialog) = check_program("matedialog");
    flags(flag::has_qarma) = check_program("qarma");
    flags(flag::has_kdialog) = check_program("kdialog");

    // If multiple helpers are available, try to default to the best one
    if (flags(flag::has_zenity) && flags(flag::has_kdialog))
    {
        auto desktop_name = std::getenv("XDG_SESSION_DESKTOP");
        if (desktop_name && desktop_name == std::string("gnome"))
            flags(flag::has_kdialog) = false;
        else if (desktop_name && desktop_name == std::string("KDE"))
            flags(flag::has_zenity) = false;
    }
#endif

    flags(flag::is_scanned) = true;
}

inline bool settings::available()
{
#if _WIN32
    return true;
#elif __APPLE__
    return true;
#elif __EMSCRIPTEN__
    // FIXME: Return true after implementation is complete.
    return false;
#else
    settings tmp;
    return tmp.flags(flag::has_zenity) ||
           tmp.flags(flag::has_matedialog) ||
           tmp.flags(flag::has_qarma) ||
           tmp.flags(flag::has_kdialog);
#endif
}

inline void settings::verbose(bool value)
{
    settings().flags(flag::is_verbose) = value;
}

inline void settings::rescan()
{
    settings(/* resync = */ true);
}

// Check whether a program is present using “which”.
inline bool settings::check_program(std::string const &program)
{
#if _WIN32
    (void)program;
    return false;
#elif __EMSCRIPTEN__
    (void)program;
    return false;
#else
    int exit_code = -1;
    internal::executor async;
    async.start_process({"/bin/sh", "-c", "which " + program});
    async.result(&exit_code);
    return exit_code == 0;
#endif
}

inline bool settings::is_osascript() const
{
#if __APPLE__
    return true;
#else
    return false;
#endif
}

inline bool settings::is_zenity() const
{
    return flags(flag::has_zenity) ||
           flags(flag::has_matedialog) ||
           flags(flag::has_qarma);
}

inline bool settings::is_kdialog() const
{
    return flags(flag::has_kdialog);
}

inline bool const &settings::flags(flag in_flag) const
{
    static bool flags[size_t(flag::max_flag)];
    return flags[size_t(in_flag)];
}

inline bool &settings::flags(flag in_flag)
{
    return const_cast<bool &>(static_cast<settings const *>(this)->flags(in_flag));
}

// executor implementation

inline std::string internal::executor::result(int *exit_code /* = nullptr */)
{
    stop();
    if (exit_code)
        *exit_code = m_exit_code;
    return m_stdout;
}

inline bool internal::executor::kill()
{
#if _WIN32
    if (m_future.valid())
    {
        // Close all windows that weren’t open when we started the future
        auto previous_windows = m_windows;
        EnumWindows(&enum_windows_callback, (LPARAM)this);
        for (auto hwnd : m_windows)
            if (previous_windows.find(hwnd) == previous_windows.end())
                SendMessage(hwnd, WM_CLOSE, 0, 0);
    }
#elif __EMSCRIPTEN__ || __NX__
    // FIXME: do something
    return false; // cannot kill
#else
    ::kill(m_pid, SIGKILL);
#endif
    stop();
    return true;
}

#if _WIN32
inline BOOL CALLBACK internal::executor::enum_windows_callback(HWND hwnd, LPARAM lParam)
{
    auto that = (executor *)lParam;

    DWORD pid;
    auto tid = GetWindowThreadProcessId(hwnd, &pid);
    if (tid == that->m_tid)
        that->m_windows.insert(hwnd);
    return TRUE;
}
#endif

#if _WIN32
inline void internal::executor::start_func(std::function<std::string(int *)> const &fun)
{
    stop();

    auto trampoline = [fun, this]()
    {
        // Save our thread id so that the caller can cancel us
        m_tid = GetCurrentThreadId();
        EnumWindows(&enum_windows_callback, (LPARAM)this);
        m_cond.notify_all();
        return fun(&m_exit_code);
    };

    std::unique_lock<std::mutex> lock(m_mutex);
    m_future = std::async(std::launch::async, trampoline);
    m_cond.wait(lock);
    m_running = true;
}

#elif __EMSCRIPTEN__
inline void internal::executor::start(int exit_code)
{
    m_exit_code = exit_code;
}

#else
inline void internal::executor::start_process(std::vector<std::string> const &command)
{
    stop();
    m_stdout.clear();
    m_exit_code = -1;

    int in[2], out[2];
    if (pipe(in) != 0 || pipe(out) != 0)
        return;

    m_pid = fork();
    if (m_pid < 0)
        return;

    close(in[m_pid ? 0 : 1]);
    close(out[m_pid ? 1 : 0]);

    if (m_pid == 0)
    {
        dup2(in[0], STDIN_FILENO);
        dup2(out[1], STDOUT_FILENO);

        // Ignore stderr so that it doesn’t pollute the console (e.g. GTK+ errors from zenity)
        int fd = open("/dev/null", O_WRONLY);
        dup2(fd, STDERR_FILENO);
        close(fd);

        std::vector<char *> args;
        std::transform(command.cbegin(), command.cend(), std::back_inserter(args),
                       [](std::string const &s) { return const_cast<char *>(s.c_str()); });
        args.push_back(nullptr); // null-terminate argv[]

        execvp(args[0], args.data());
        exit(1);
    }

    close(in[1]);
    m_fd = out[0];
    auto flags = fcntl(m_fd, F_GETFL);
    fcntl(m_fd, F_SETFL, flags | O_NONBLOCK);

    m_running = true;
}
#endif

inline internal::executor::~executor()
{
    stop();
}

inline bool internal::executor::ready(int timeout /* = default_wait_timeout */)
{
    if (!m_running)
        return true;

#if _WIN32
    if (m_future.valid())
    {
        auto status = m_future.wait_for(std::chrono::milliseconds(timeout));
        if (status != std::future_status::ready)
        {
            // On Windows, we need to run the message pump. If the async
            // thread uses a Windows API dialog, it may be attached to the
            // main thread and waiting for messages that only we can dispatch.
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            return false;
        }

        m_stdout = m_future.get();
    }
#elif __EMSCRIPTEN__ || __NX__
    // FIXME: do something
    (void)timeout;
#else
    char buf[BUFSIZ];
    ssize_t received = read(m_fd, buf, BUFSIZ); // Flawfinder: ignore
    if (received > 0)
    {
        m_stdout += std::string(buf, received);
        return false;
    }

    // Reap child process if it is dead. It is possible that the system has already reaped it
    // (this happens when the calling application handles or ignores SIG_CHLD) and results in
    // waitpid() failing with ECHILD. Otherwise we assume the child is running and we sleep for
    // a little while.
    int status;
    pid_t child = waitpid(m_pid, &status, WNOHANG);
    if (child != m_pid && (child >= 0 || errno != ECHILD))
    {
        // FIXME: this happens almost always at first iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
        return false;
    }

    close(m_fd);
    m_exit_code = WEXITSTATUS(status);
#endif

    m_running = false;
    return true;
}

inline void internal::executor::stop()
{
    // Loop until the user closes the dialog
    while (!ready())
        ;
}

// dll implementation

#if _WIN32
inline internal::platform::dll::dll(std::string const &name)
  : handle(::LoadLibraryA(name.c_str()))
{}

inline internal::platform::dll::~dll()
{
    if (handle)
        ::FreeLibrary(handle);
}
#endif // _WIN32

// ole32_dll implementation

#if _WIN32
inline internal::platform::ole32_dll::ole32_dll()
    : dll("ole32.dll")
{
    // Use COINIT_MULTITHREADED because COINIT_APARTMENTTHREADED causes crashes.
    // See https://github.com/samhocevar/portable-file-dialogs/issues/51
    auto coinit = proc<HRESULT WINAPI (LPVOID, DWORD)>(*this, "CoInitializeEx");
    m_state = coinit(nullptr, COINIT_MULTITHREADED);
}

inline internal::platform::ole32_dll::~ole32_dll()
{
    if (is_initialized())
        proc<void WINAPI ()>(*this, "CoUninitialize")();
}

inline bool internal::platform::ole32_dll::is_initialized()
{
    return m_state == S_OK || m_state == S_FALSE;
}
#endif

// new_style_context implementation

#if _WIN32
inline internal::platform::new_style_context::new_style_context()
{
    // Only create one activation context for the whole app lifetime.
    static HANDLE hctx = create();

    if (hctx != INVALID_HANDLE_VALUE)
        ActivateActCtx(hctx, &m_cookie);
}

inline internal::platform::new_style_context::~new_style_context()
{
    DeactivateActCtx(0, m_cookie);
}

inline HANDLE internal::platform::new_style_context::create()
{
    // This “hack” seems to be necessary for this code to work on windows XP.
    // Without it, dialogs do not show and close immediately. GetError()
    // returns 0 so I don’t know what causes this. I was not able to reproduce
    // this behavior on Windows 7 and 10 but just in case, let it be here for
    // those versions too.
    // This hack is not required if other dialogs are used (they load comdlg32
    // automatically), only if message boxes are used.
    dll comdlg32("comdlg32.dll");

    // Using approach as shown here: https://stackoverflow.com/a/10444161
    UINT len = ::GetSystemDirectoryA(nullptr, 0);
    std::string sys_dir(len, '\0');
    ::GetSystemDirectoryA(&sys_dir[0], len);

    ACTCTXA act_ctx =
    {
        // Do not set flag ACTCTX_FLAG_SET_PROCESS_DEFAULT, since it causes a
        // crash with error “default context is already set”.
        sizeof(act_ctx),
        ACTCTX_FLAG_RESOURCE_NAME_VALID | ACTCTX_FLAG_ASSEMBLY_DIRECTORY_VALID,
        "shell32.dll", 0, 0, sys_dir.c_str(), (LPCSTR)124,
    };

    return ::CreateActCtxA(&act_ctx);
}
#endif // _WIN32

// dialog implementation

inline bool internal::dialog::ready(int timeout /* = default_wait_timeout */) const
{
    return m_async->ready(timeout);
}

inline bool internal::dialog::kill() const
{
    return m_async->kill();
}

inline internal::dialog::dialog()
  : m_async(std::make_shared<executor>())
{
}

inline std::vector<std::string> internal::dialog::desktop_helper() const
{
#if __APPLE__
    return { "osascript" };
#else
    return { flags(flag::has_zenity) ? "zenity"
           : flags(flag::has_matedialog) ? "matedialog"
           : flags(flag::has_qarma) ? "qarma"
           : flags(flag::has_kdialog) ? "kdialog"
           : "echo" };
#endif
}

inline std::string internal::dialog::buttons_to_name(choice _choice)
{
    switch (_choice)
    {
        case choice::ok_cancel: return "okcancel";
        case choice::yes_no: return "yesno";
        case choice::yes_no_cancel: return "yesnocancel";
        case choice::retry_cancel: return "retrycancel";
        case choice::abort_retry_ignore: return "abortretryignore";
        /* case choice::ok: */ default: return "ok";
    }
}

inline std::string internal::dialog::get_icon_name(icon _icon)
{
    switch (_icon)
    {
        case icon::warning: return "warning";
        case icon::error: return "error";
        case icon::question: return "question";
        // Zenity wants "information" but WinForms wants "info"
        /* case icon::info: */ default:
#if _WIN32
            return "info";
#else
            return "information";
#endif
    }
}

// THis is only used for debugging purposes
inline std::ostream& operator <<(std::ostream &s, std::vector<std::string> const &v)
{
    int not_first = 0;
    for (auto &e : v)
        s << (not_first++ ? " " : "") << e;
    return s;
}

// Properly quote a string for Powershell: replace ' or " with '' or ""
// FIXME: we should probably get rid of newlines!
// FIXME: the \" sequence seems unsafe, too!
// XXX: this is no longer used but I would like to keep it around just in case
inline std::string internal::dialog::powershell_quote(std::string const &str) const
{
    return "'" + std::regex_replace(str, std::regex("['\"]"), "$&$&") + "'";
}

// Properly quote a string for osascript: replace \ or " with \\ or \"
// XXX: this also used to replace ' with \' when popen was used, but it would be
// smarter to do shell_quote(osascript_quote(...)) if this is needed again.
inline std::string internal::dialog::osascript_quote(std::string const &str) const
{
    return "\"" + std::regex_replace(str, std::regex("[\\\\\"]"), "\\$&") + "\"";
}

// Properly quote a string for the shell: just replace ' with '\''
// XXX: this is no longer used but I would like to keep it around just in case
inline std::string internal::dialog::shell_quote(std::string const &str) const
{
    return "'" + std::regex_replace(str, std::regex("'"), "'\\''") + "'";
}

// file_dialog implementation

inline internal::file_dialog::file_dialog(type in_type,
            std::string const &title,
            std::string const &default_path /* = "" */,
            std::vector<std::string> const &filters /* = {} */,
            opt options /* = opt::none */)
{
#if _WIN32
    std::string filter_list;
    std::regex whitespace("  *");
    for (size_t i = 0; i + 1 < filters.size(); i += 2)
    {
        filter_list += filters[i] + '\0';
        filter_list += std::regex_replace(filters[i + 1], whitespace, ";") + '\0';
    }
    filter_list += '\0';

    m_async->start_func([this, in_type, title, default_path, filter_list,
                         options](int *exit_code) -> std::string
    {
        (void)exit_code;
        m_wtitle = internal::str2wstr(title);
        m_wdefault_path = internal::str2wstr(default_path);
        auto wfilter_list = internal::str2wstr(filter_list);

        // Initialise COM. This is required for the new folder selection window,
        // (see https://github.com/samhocevar/portable-file-dialogs/pull/21)
        // and to avoid random crashes with GetOpenFileNameW() (see
        // https://github.com/samhocevar/portable-file-dialogs/issues/51)
        ole32_dll ole32;

        // Folder selection uses a different method
        if (in_type == type::folder)
        {
#if PFD_HAS_IFILEDIALOG
            if (flags(flag::is_vista))
            {
                // On Vista and higher we should be able to use IFileDialog for folder selection
                IFileDialog *ifd;
                HRESULT hr = dll::proc<HRESULT WINAPI (REFCLSID, LPUNKNOWN, DWORD, REFIID, LPVOID *)>(ole32, "CoCreateInstance")
                                 (CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&ifd));

                // In case CoCreateInstance fails (which it should not), try legacy approach
                if (SUCCEEDED(hr))
                    return select_folder_vista(ifd, options & opt::force_path);
            }
#endif

            BROWSEINFOW bi;
            memset(&bi, 0, sizeof(bi));

            bi.lpfn = &bffcallback;
            bi.lParam = (LPARAM)this;

            if (flags(flag::is_vista))
            {
                if (ole32.is_initialized())
                    bi.ulFlags |= BIF_NEWDIALOGSTYLE;
                bi.ulFlags |= BIF_EDITBOX;
                bi.ulFlags |= BIF_STATUSTEXT;
            }

            auto *list = SHBrowseForFolderW(&bi);
            std::string ret;
            if (list)
            {
                auto buffer = new wchar_t[MAX_PATH];
                SHGetPathFromIDListW(list, buffer);
                dll::proc<void WINAPI (LPVOID)>(ole32, "CoTaskMemFree")(list);
                ret = internal::wstr2str(buffer);
                delete[] buffer;
            }
            return ret;
        }

        OPENFILENAMEW ofn;
        memset(&ofn, 0, sizeof(ofn));
        ofn.lStructSize = sizeof(OPENFILENAMEW);
        ofn.hwndOwner = GetActiveWindow();

        ofn.lpstrFilter = wfilter_list.c_str();

        auto woutput = std::wstring(MAX_PATH * 256, L'\0');
        ofn.lpstrFile = (LPWSTR)woutput.data();
        ofn.nMaxFile = (DWORD)woutput.size();
        if (!m_wdefault_path.empty())
        {
            // If a directory was provided, use it as the initial directory. If
            // a valid path was provided, use it as the initial file. Otherwise,
            // let the Windows API decide.
            auto path_attr = GetFileAttributesW(m_wdefault_path.c_str());
            if (path_attr != INVALID_FILE_ATTRIBUTES && (path_attr & FILE_ATTRIBUTE_DIRECTORY))
                ofn.lpstrInitialDir = m_wdefault_path.c_str();
            else if (m_wdefault_path.size() <= woutput.size())
                //second argument is size of buffer, not length of string
                StringCchCopyW(ofn.lpstrFile, MAX_PATH*256+1, m_wdefault_path.c_str());
            else
            {
                ofn.lpstrFileTitle = (LPWSTR)m_wdefault_path.data();
                ofn.nMaxFileTitle = (DWORD)m_wdefault_path.size();
            }
        }
        ofn.lpstrTitle = m_wtitle.c_str();
        ofn.Flags = OFN_NOCHANGEDIR | OFN_EXPLORER;

        dll comdlg32("comdlg32.dll");

        // Apply new visual style (required for windows XP)
        new_style_context ctx;

        if (in_type == type::save)
        {
            if (!(options & opt::force_overwrite))
                ofn.Flags |= OFN_OVERWRITEPROMPT;

            dll::proc<BOOL WINAPI (LPOPENFILENAMEW)> get_save_file_name(comdlg32, "GetSaveFileNameW");
            if (get_save_file_name(&ofn) == 0)
                return "";
            return internal::wstr2str(woutput.c_str());
        }
        else
        {
            if (options & opt::multiselect)
                ofn.Flags |= OFN_ALLOWMULTISELECT;
            ofn.Flags |= OFN_PATHMUSTEXIST;

            dll::proc<BOOL WINAPI (LPOPENFILENAMEW)> get_open_file_name(comdlg32, "GetOpenFileNameW");
            if (get_open_file_name(&ofn) == 0)
                return "";
        }

        std::string prefix;
        for (wchar_t const *p = woutput.c_str(); *p; )
        {
            auto filename = internal::wstr2str(p);
            p += wcslen(p);
            // In multiselect mode, we advance p one wchar further and
            // check for another filename. If there is one and the
            // prefix is empty, it means we just read the prefix.
            if ((options & opt::multiselect) && *++p && prefix.empty())
            {
                prefix = filename + "/";
                continue;
            }

            m_vector_result.push_back(prefix + filename);
        }

        return "";
    });
#elif __EMSCRIPTEN__
    // FIXME: do something
    (void)in_type;
    (void)title;
    (void)default_path;
    (void)filters;
    (void)options;
#else
    auto command = desktop_helper();

    if (is_osascript())
    {
        std::string script = "set ret to choose";
        switch (in_type)
        {
            case type::save:
                script += " file name";
                break;
            case type::open: default:
                script += " file";
                if (options & opt::multiselect)
                    script += " with multiple selections allowed";
                break;
            case type::folder:
                script += " folder";
                break;
        }

        if (default_path.size())
            script += " default location " + osascript_quote(default_path);
        script += " with prompt " + osascript_quote(title);

        if (in_type == type::open)
        {
            // Concatenate all user-provided filter patterns
            std::string patterns;
            for (size_t i = 0; i < filters.size() / 2; ++i)
                patterns += " " + filters[2 * i + 1];

            // Split the pattern list to check whether "*" is in there; if it
            // is, we have to disable filters because there is no mechanism in
            // OS X for the user to override the filter.
            std::regex sep("\\s+");
            std::string filter_list;
            bool has_filter = true;
            std::sregex_token_iterator iter(patterns.begin(), patterns.end(), sep, -1);
            std::sregex_token_iterator end;
            for ( ; iter != end; ++iter)
            {
                auto pat = iter->str();
                if (pat == "*" || pat == "*.*")
                    has_filter = false;
                else if (internal::starts_with(pat, "*."))
                    filter_list += (filter_list.size() == 0 ? "" : ",") +
                                   osascript_quote(pat.substr(2, pat.size() - 2));
            }
            if (has_filter && filter_list.size() > 0)
                script += " of type {" + filter_list + "}";
        }

        if (in_type == type::open && (options & opt::multiselect))
        {
            script += "\nset s to \"\"";
            script += "\nrepeat with i in ret";
            script += "\n  set s to s & (POSIX path of i) & \"\\n\"";
            script += "\nend repeat";
            script += "\ncopy s to stdout";
        }
        else
        {
            script += "\nPOSIX path of ret";
        }

        command.push_back("-e");
        command.push_back(script);
    }
    else if (is_zenity())
    {
        command.push_back("--file-selection");
        command.push_back("--filename=" + default_path);
        command.push_back("--title");
        command.push_back(title);
        command.push_back("--separator=\n");

        for (size_t i = 0; i < filters.size() / 2; ++i)
        {
            command.push_back("--file-filter");
            command.push_back(filters[2 * i] + "|" + filters[2 * i + 1]);
        }

        if (in_type == type::save)
            command.push_back("--save");
        if (in_type == type::folder)
            command.push_back("--directory");
        if (!(options & opt::force_overwrite))
            command.push_back("--confirm-overwrite");
        if (options & opt::multiselect)
            command.push_back("--multiple");
    }
    else if (is_kdialog())
    {
        switch (in_type)
        {
            case type::save: command.push_back("--getsavefilename"); break;
            case type::open: command.push_back("--getopenfilename"); break;
            case type::folder: command.push_back("--getexistingdirectory"); break;
        }
        if (options & opt::multiselect)
        {
            command.push_back("--multiple");
            command.push_back("--separate-output");
        }

        command.push_back(default_path);

        std::string filter;
        for (size_t i = 0; i < filters.size() / 2; ++i)
            filter += (i == 0 ? "" : " | ") + filters[2 * i] + "(" + filters[2 * i + 1] + ")";
        command.push_back(filter);

        command.push_back("--title");
        command.push_back(title);
    }

    if (flags(flag::is_verbose))
        std::cerr << "pfd: " << command << std::endl;

    m_async->start_process(command);
#endif
}

inline std::string internal::file_dialog::string_result()
{
#if _WIN32
    return m_async->result();
#else
    auto ret = m_async->result();
    // Strip potential trailing newline (zenity). Also strip trailing slash
    // added by osascript for consistency with other backends.
    while (!ret.empty() && (ret.back() == '\n' || ret.back() == '/'))
        ret.pop_back();
    return ret;
#endif
}

inline std::vector<std::string> internal::file_dialog::vector_result()
{
#if _WIN32
    m_async->result();
    return m_vector_result;
#else
    std::vector<std::string> ret;
    auto result = m_async->result();
    for (;;)
    {
        // Split result along newline characters
        auto i = result.find('\n');
        if (i == 0 || i == std::string::npos)
            break;
        ret.push_back(result.substr(0, i));
        result = result.substr(i + 1, result.size());
    }
    return ret;
#endif
}

#if _WIN32
// Use a static function to pass as BFFCALLBACK for legacy folder select
inline int CALLBACK internal::file_dialog::bffcallback(HWND hwnd, UINT uMsg,
                                                       LPARAM, LPARAM pData)
{
    auto inst = (file_dialog *)pData;
    switch (uMsg)
    {
        case BFFM_INITIALIZED:
            SendMessage(hwnd, BFFM_SETSELECTIONW, TRUE, (LPARAM)inst->m_wdefault_path.c_str());
            break;
    }
    return 0;
}

#if PFD_HAS_IFILEDIALOG
inline std::string internal::file_dialog::select_folder_vista(IFileDialog *ifd, bool force_path)
{
    std::string result;

    IShellItem *folder;

    // Load library at runtime so app doesn't link it at load time (which will fail on windows XP)
    dll shell32("shell32.dll");
    dll::proc<HRESULT WINAPI (PCWSTR, IBindCtx*, REFIID, void**)>
        create_item(shell32, "SHCreateItemFromParsingName");

    if (!create_item)
        return "";

    auto hr = create_item(m_wdefault_path.c_str(),
                          nullptr,
                          IID_PPV_ARGS(&folder));

    // Set default folder if found. This only sets the default folder. If
    // Windows has any info about the most recently selected folder, it
    // will display it instead. Generally, calling SetFolder() to set the
    // current directory “is not a good or expected user experience and
    // should therefore be avoided”:
    // https://docs.microsoft.com/windows/win32/api/shobjidl_core/nf-shobjidl_core-ifiledialog-setfolder
    if (SUCCEEDED(hr))
    {
        if (force_path)
            ifd->SetFolder(folder);
        else
            ifd->SetDefaultFolder(folder);
        folder->Release();
    }

    // Set the dialog title and option to select folders
    ifd->SetOptions(FOS_PICKFOLDERS);
    ifd->SetTitle(m_wtitle.c_str());

    hr = ifd->Show(GetActiveWindow());
    if (SUCCEEDED(hr))
    {
        IShellItem* item;
        hr = ifd->GetResult(&item);
        if (SUCCEEDED(hr))
        {
            wchar_t* wselected = nullptr;
            item->GetDisplayName(SIGDN_FILESYSPATH, &wselected);
            item->Release();

            if (wselected)
            {
                result = internal::wstr2str(std::wstring(wselected));
                dll::proc<void WINAPI (LPVOID)>(ole32_dll(), "CoTaskMemFree")(wselected);
            }
        }
    }

    ifd->Release();

    return result;
}
#endif
#endif

// notify implementation

inline notify::notify(std::string const &title,
                      std::string const &message,
                      icon _icon /* = icon::info */)
{
    if (_icon == icon::question) // Not supported by notifications
        _icon = icon::info;

#if _WIN32
    // Use a static shared pointer for notify_icon so that we can delete
    // it whenever we need to display a new one, and we can also wait
    // until the program has finished running.
    struct notify_icon_data : public NOTIFYICONDATAW
    {
        ~notify_icon_data() { Shell_NotifyIconW(NIM_DELETE, this); }
    };

    static std::shared_ptr<notify_icon_data> nid;

    // Release the previous notification icon, if any, and allocate a new
    // one. Note that std::make_shared() does value initialization, so there
    // is no need to memset the structure.
    nid = nullptr;
    nid = std::make_shared<notify_icon_data>();

    // For XP support
    nid->cbSize = NOTIFYICONDATAW_V2_SIZE;
    nid->hWnd = nullptr;
    nid->uID = 0;

    // Flag Description:
    // - NIF_ICON    The hIcon member is valid.
    // - NIF_MESSAGE The uCallbackMessage member is valid.
    // - NIF_TIP     The szTip member is valid.
    // - NIF_STATE   The dwState and dwStateMask members are valid.
    // - NIF_INFO    Use a balloon ToolTip instead of a standard ToolTip. The szInfo, uTimeout, szInfoTitle, and dwInfoFlags members are valid.
    // - NIF_GUID    Reserved.
    nid->uFlags = NIF_MESSAGE | NIF_ICON | NIF_INFO;

    // Flag Description
    // - NIIF_ERROR     An error icon.
    // - NIIF_INFO      An information icon.
    // - NIIF_NONE      No icon.
    // - NIIF_WARNING   A warning icon.
    // - NIIF_ICON_MASK Version 6.0. Reserved.
    // - NIIF_NOSOUND   Version 6.0. Do not play the associated sound. Applies only to balloon ToolTips
    switch (_icon)
    {
        case icon::warning: nid->dwInfoFlags = NIIF_WARNING; break;
        case icon::error: nid->dwInfoFlags = NIIF_ERROR; break;
        /* case icon::info: */ default: nid->dwInfoFlags = NIIF_INFO; break;
    }

    ENUMRESNAMEPROC icon_enum_callback = [](HMODULE, LPCTSTR, LPTSTR lpName, LONG_PTR lParam) -> BOOL
    {
        ((NOTIFYICONDATAW *)lParam)->hIcon = ::LoadIcon(GetModuleHandle(nullptr), lpName);
        return false;
    };

    nid->hIcon = ::LoadIcon(nullptr, IDI_APPLICATION);
    ::EnumResourceNames(nullptr, RT_GROUP_ICON, icon_enum_callback, (LONG_PTR)nid.get());

    nid->uTimeout = 5000;

    StringCchCopyW(nid->szInfoTitle, ARRAYSIZE(nid->szInfoTitle), internal::str2wstr(title).c_str());
    StringCchCopyW(nid->szInfo, ARRAYSIZE(nid->szInfo), internal::str2wstr(message).c_str());

    // Display the new icon
    Shell_NotifyIconW(NIM_ADD, nid.get());
#elif __EMSCRIPTEN__
    // FIXME: do something
    (void)title;
    (void)message;
#else
    auto command = desktop_helper();

    if (is_osascript())
    {
        command.push_back("-e");
        command.push_back("display notification " + osascript_quote(message) +
                          " with title " + osascript_quote(title));
    }
    else if (is_zenity())
    {
        command.push_back("--notification");
        command.push_back("--window-icon");
        command.push_back(get_icon_name(_icon));
        command.push_back("--text");
        command.push_back(title + "\n" + message);
    }
    else if (is_kdialog())
    {
        command.push_back("--icon");
        command.push_back(get_icon_name(_icon));
        command.push_back("--title");
        command.push_back(title);
        command.push_back("--passivepopup");
        command.push_back(message);
        command.push_back("5");
    }

    if (flags(flag::is_verbose))
        std::cerr << "pfd: " << command << std::endl;

    m_async->start_process(command);
#endif
}

// message implementation

inline message::message(std::string const &title,
                        std::string const &text,
                        choice _choice /* = choice::ok_cancel */,
                        icon _icon /* = icon::info */)
{
#if _WIN32
    // Use MB_SYSTEMMODAL rather than MB_TOPMOST to ensure the message window is brought
    // to front. See https://github.com/samhocevar/portable-file-dialogs/issues/52
    UINT style = MB_SYSTEMMODAL;
    switch (_icon)
    {
        case icon::warning: style |= MB_ICONWARNING; break;
        case icon::error: style |= MB_ICONERROR; break;
        case icon::question: style |= MB_ICONQUESTION; break;
        /* case icon::info: */ default: style |= MB_ICONINFORMATION; break;
    }

    switch (_choice)
    {
        case choice::ok_cancel: style |= MB_OKCANCEL; break;
        case choice::yes_no: style |= MB_YESNO; break;
        case choice::yes_no_cancel: style |= MB_YESNOCANCEL; break;
        case choice::retry_cancel: style |= MB_RETRYCANCEL; break;
        case choice::abort_retry_ignore: style |= MB_ABORTRETRYIGNORE; break;
        /* case choice::ok: */ default: style |= MB_OK; break;
    }

    m_mappings[IDCANCEL] = button::cancel;
    m_mappings[IDOK] = button::ok;
    m_mappings[IDYES] = button::yes;
    m_mappings[IDNO] = button::no;
    m_mappings[IDABORT] = button::abort;
    m_mappings[IDRETRY] = button::retry;
    m_mappings[IDIGNORE] = button::ignore;

    m_async->start_func([text, title, style](int* exit_code) -> std::string
    {
        auto wtext = internal::str2wstr(text);
        auto wtitle = internal::str2wstr(title);
        // Apply new visual style (required for all Windows versions)
        new_style_context ctx;
        *exit_code = MessageBoxW(GetActiveWindow(), wtext.c_str(), wtitle.c_str(), style);
        return "";
    });

#elif __EMSCRIPTEN__
    std::string full_message;
    switch (_icon)
    {
        case icon::warning: full_message = "⚠️"; break;
        case icon::error: full_message = "⛔"; break;
        case icon::question: full_message = "❓"; break;
        /* case icon::info: */ default: full_message = "ℹ"; break;
    }

    full_message += ' ' + title + "\n\n" + text;

    // This does not really start an async task; it just passes the
    // EM_ASM_INT return value to a fake start() function.
    m_async->start(EM_ASM_INT(
    {
        if ($1)
            return window.confirm(UTF8ToString($0)) ? 0 : -1;
        alert(UTF8ToString($0));
        return 0;
    }, full_message.c_str(), _choice == choice::ok_cancel));
#else
    auto command = desktop_helper();

    if (is_osascript())
    {
        std::string script = "display dialog " + osascript_quote(text) +
                             " with title " + osascript_quote(title);
        switch (_choice)
        {
            case choice::ok_cancel:
                script += "buttons {\"OK\", \"Cancel\"}"
                          " default button \"OK\""
                          " cancel button \"Cancel\"";
                m_mappings[256] = button::cancel;
                break;
            case choice::yes_no:
                script += "buttons {\"Yes\", \"No\"}"
                          " default button \"Yes\""
                          " cancel button \"No\"";
                m_mappings[256] = button::no;
                break;
            case choice::yes_no_cancel:
                script += "buttons {\"Yes\", \"No\", \"Cancel\"}"
                          " default button \"Yes\""
                          " cancel button \"Cancel\"";
                m_mappings[256] = button::cancel;
                break;
            case choice::retry_cancel:
                script += "buttons {\"Retry\", \"Cancel\"}"
                          " default button \"Retry\""
                          " cancel button \"Cancel\"";
                m_mappings[256] = button::cancel;
                break;
            case choice::abort_retry_ignore:
                script += "buttons {\"Abort\", \"Retry\", \"Ignore\"}"
                          " default button \"Retry\""
                          " cancel button \"Retry\"";
                m_mappings[256] = button::cancel;
                break;
            case choice::ok: default:
                script += "buttons {\"OK\"}"
                          " default button \"OK\""
                          " cancel button \"OK\"";
                m_mappings[256] = button::ok;
                break;
        }
        script += " with icon ";
        switch (_icon)
        {
            #define PFD_OSX_ICON(n) "alias ((path to library folder from system domain) as text " \
                "& \"CoreServices:CoreTypes.bundle:Contents:Resources:" n ".icns\")"
            case icon::info: default: script += PFD_OSX_ICON("ToolBarInfo"); break;
            case icon::warning: script += "caution"; break;
            case icon::error: script += "stop"; break;
            case icon::question: script += PFD_OSX_ICON("GenericQuestionMarkIcon"); break;
            #undef PFD_OSX_ICON
        }

        command.push_back("-e");
        command.push_back(script);
    }
    else if (is_zenity())
    {
        switch (_choice)
        {
            case choice::ok_cancel:
                command.insert(command.end(), { "--question", "--cancel-label=Cancel", "--ok-label=OK" }); break;
            case choice::yes_no:
                // Do not use standard --question because it causes “No” to return -1,
                // which is inconsistent with the “Yes/No/Cancel” mode below.
                command.insert(command.end(), { "--question", "--switch", "--extra-button=No", "--extra-button=Yes" }); break;
            case choice::yes_no_cancel:
                command.insert(command.end(), { "--question", "--switch", "--extra-button=Cancel", "--extra-button=No", "--extra-button=Yes" }); break;
            case choice::retry_cancel:
                command.insert(command.end(), { "--question", "--switch", "--extra-button=Cancel", "--extra-button=Retry" }); break;
            case choice::abort_retry_ignore:
                command.insert(command.end(), { "--question", "--switch", "--extra-button=Ignore", "--extra-button=Abort", "--extra-button=Retry" }); break;
            case choice::ok:
            default:
                switch (_icon)
                {
                    case icon::error: command.push_back("--error"); break;
                    case icon::warning: command.push_back("--warning"); break;
                    default: command.push_back("--info"); break;
                }
        }

        command.insert(command.end(), { "--title", title,
                                        "--width=300", "--height=0", // sensible defaults
                                        "--text", text,
                                        "--icon-name=dialog-" + get_icon_name(_icon) });
    }
    else if (is_kdialog())
    {
        if (_choice == choice::ok)
        {
            switch (_icon)
            {
                case icon::error: command.push_back("--error"); break;
                case icon::warning: command.push_back("--sorry"); break;
                default: command.push_back("--msgbox"); break;
            }
        }
        else
        {
            std::string flag = "--";
            if (_icon == icon::warning || _icon == icon::error)
                flag += "warning";
            flag += "yesno";
            if (_choice == choice::yes_no_cancel)
                flag += "cancel";
            command.push_back(flag);
            if (_choice == choice::yes_no || _choice == choice::yes_no_cancel)
            {
                m_mappings[0] = button::yes;
                m_mappings[256] = button::no;
            }
        }

        command.push_back(text);
        command.push_back("--title");
        command.push_back(title);

        // Must be after the above part
        if (_choice == choice::ok_cancel)
            command.insert(command.end(), { "--yes-label", "OK", "--no-label", "Cancel" });
    }

    if (flags(flag::is_verbose))
        std::cerr << "pfd: " << command << std::endl;

    m_async->start_process(command);
#endif
}

inline button message::result()
{
    int exit_code;
    auto ret = m_async->result(&exit_code);
    // osascript will say "button returned:Cancel\n"
    // and others will just say "Cancel\n"
    if (exit_code < 0 || // this means cancel
        internal::ends_with(ret, "Cancel\n"))
        return button::cancel;
    if (internal::ends_with(ret, "OK\n"))
        return button::ok;
    if (internal::ends_with(ret, "Yes\n"))
        return button::yes;
    if (internal::ends_with(ret, "No\n"))
        return button::no;
    if (internal::ends_with(ret, "Abort\n"))
        return button::abort;
    if (internal::ends_with(ret, "Retry\n"))
        return button::retry;
    if (internal::ends_with(ret, "Ignore\n"))
        return button::ignore;
    if (m_mappings.count(exit_code) != 0)
        return m_mappings[exit_code];
    return exit_code == 0 ? button::ok : button::cancel;
}

// open_file implementation

inline open_file::open_file(std::string const &title,
                            std::string const &default_path /* = "" */,
                            std::vector<std::string> const &filters /* = { "All Files", "*" } */,
                            opt options /* = opt::none */)
  : file_dialog(type::open, title, default_path, filters, options)
{
}

inline open_file::open_file(std::string const &title,
                            std::string const &default_path,
                            std::vector<std::string> const &filters,
                            bool allow_multiselect)
  : open_file(title, default_path, filters,
              (allow_multiselect ? opt::multiselect : opt::none))
{
}

inline std::vector<std::string> open_file::result()
{
    return vector_result();
}

// save_file implementation

inline save_file::save_file(std::string const &title,
                            std::string const &default_path /* = "" */,
                            std::vector<std::string> const &filters /* = { "All Files", "*" } */,
                            opt options /* = opt::none */)
  : file_dialog(type::save, title, default_path, filters, options)
{
}

inline save_file::save_file(std::string const &title,
                            std::string const &default_path,
                            std::vector<std::string> const &filters,
                            bool confirm_overwrite)
  : save_file(title, default_path, filters,
              (confirm_overwrite ? opt::none : opt::force_overwrite))
{
}

inline std::string save_file::result()
{
    return string_result();
}

// select_folder implementation

inline select_folder::select_folder(std::string const &title,
                                    std::string const &default_path /* = "" */,
                                    opt options /* = opt::none */)
  : file_dialog(type::folder, title, default_path, {}, options)
{
}

inline std::string select_folder::result()
{
    return string_result();
}

#endif // PFD_SKIP_IMPLEMENTATION

} // namespace pfd

