#include "error.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define C4_LOGF_ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)
#define C4_LOGF_WARN(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)
#define C4_LOGP(msg, ...) printf(msg)

#if defined(C4_XBOX) || (defined(C4_WIN) && defined(C4_MSVC))
#   include "windows.hpp"
#elif defined(C4_PS4)
#   include <libdbg.h>
#elif defined(C4_UNIX) || defined(C4_LINUX)
#   include <sys/stat.h>
#   include <cstring>
#   include <fcntl.h>
#elif defined(C4_MACOS) || defined(C4_IOS)
#   include <assert.h>
#   include <stdbool.h>
#   include <sys/types.h>
#   include <sys/sysctl.h>
#endif
// the amalgamation tool is dumb and was omitting this include under MACOS.
// So do it only once:
#if defined(C4_UNIX) || defined(C4_LINUX) || defined(C4_MACOS) || defined(C4_IOS)
#   include <unistd.h>
#endif

#if defined(C4_EXCEPTIONS_ENABLED) && defined(C4_ERROR_THROWS_EXCEPTION)
#   include <exception>
#endif

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wformat-nonliteral"
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wformat-nonliteral"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif


//-----------------------------------------------------------------------------
namespace c4 {

static error_flags         s_error_flags = ON_ERROR_DEFAULTS;
static error_callback_type s_error_callback = nullptr;

//-----------------------------------------------------------------------------

error_flags get_error_flags()
{
    return s_error_flags;
}
void set_error_flags(error_flags flags)
{
    s_error_flags = flags;
}

error_callback_type get_error_callback()
{
    return s_error_callback;
}
/** Set the function which is called when an error occurs. */
void set_error_callback(error_callback_type cb)
{
    s_error_callback = cb;
}

//-----------------------------------------------------------------------------

void handle_error(srcloc where, const char *fmt, ...)
{
    char buf[1024];
    size_t msglen = 0;
    if(s_error_flags & (ON_ERROR_LOG|ON_ERROR_CALLBACK))
    {
        va_list args;
        va_start(args, fmt);
        int ilen = vsnprintf(buf, sizeof(buf), fmt, args); // ss.vprintf(fmt, args);
        va_end(args);
        msglen = ilen >= 0 && ilen < (int)sizeof(buf) ? static_cast<size_t>(ilen) : sizeof(buf)-1;
    }

    if(s_error_flags & ON_ERROR_LOG)
    {
        C4_LOGF_ERR("\n");
#if defined(C4_ERROR_SHOWS_FILELINE) && defined(C4_ERROR_SHOWS_FUNC)
        C4_LOGF_ERR("%s:%d: ERROR: %s\n", where.file, where.line, buf);
        C4_LOGF_ERR("%s:%d: ERROR here: %s\n", where.file, where.line, where.func);
#elif defined(C4_ERROR_SHOWS_FILELINE)
        C4_LOGF_ERR("%s:%d: ERROR: %s\n", where.file, where.line, buf);
#elif ! defined(C4_ERROR_SHOWS_FUNC)
        C4_LOGF_ERR("ERROR: %s\n", buf);
#endif
    }

    if(s_error_flags & ON_ERROR_CALLBACK)
    {
        if(s_error_callback)
        {
            s_error_callback(buf, msglen/*ss.c_strp(), ss.tellp()*/);
        }
    }

    if(s_error_flags & ON_ERROR_ABORT)
    {
        abort();
    }

    if(s_error_flags & ON_ERROR_THROW)
    {
#if defined(C4_EXCEPTIONS_ENABLED) && defined(C4_ERROR_THROWS_EXCEPTION)
        throw Exception(buf);
#else
        abort();
#endif
    }
}

//-----------------------------------------------------------------------------

void handle_warning(srcloc where, const char *fmt, ...)
{
    va_list args;
    char buf[1024]; //sstream<c4::string> ss;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    C4_LOGF_WARN("\n");
#if defined(C4_ERROR_SHOWS_FILELINE) && defined(C4_ERROR_SHOWS_FUNC)
    C4_LOGF_WARN("%s:%d: WARNING: %s\n", where.file, where.line, buf/*ss.c_strp()*/);
    C4_LOGF_WARN("%s:%d: WARNING: here: %s\n", where.file, where.line, where.func);
#elif defined(C4_ERROR_SHOWS_FILELINE)
    C4_LOGF_WARN("%s:%d: WARNING: %s\n", where.file, where.line, buf/*ss.c_strp()*/);
#elif ! defined(C4_ERROR_SHOWS_FUNC)
    C4_LOGF_WARN("WARNING: %s\n", buf/*ss.c_strp()*/);
#endif
    //c4::log.flush();
}

//-----------------------------------------------------------------------------
bool is_debugger_attached()
{
#if defined(C4_UNIX) || defined(C4_LINUX)
    static bool first_call = true;
    static bool first_call_result = false;
    if(first_call)
    {
        first_call = false;
        C4_SUPPRESS_WARNING_GCC_PUSH
        #if defined(__GNUC__) && __GNUC__ > 9
        C4_SUPPRESS_WARNING_GCC("-Wanalyzer-fd-leak")
        #endif
        //! @see http://stackoverflow.com/questions/3596781/how-to-detect-if-the-current-process-is-being-run-by-gdb
        //! (this answer: http://stackoverflow.com/a/24969863/3968589 )
        char buf[1024] = "";
        int status_fd = open("/proc/self/status", O_RDONLY);
        if (status_fd == -1)
        {
            return 0;
        }
        else
        {
            ssize_t num_read = ::read(status_fd, buf, sizeof(buf));
            if (num_read > 0)
            {
                static const char TracerPid[] = "TracerPid:";
                char *tracer_pid;
                if(num_read < 1024)
                {
                    buf[num_read] = 0;
                }
                tracer_pid = strstr(buf, TracerPid);
                if (tracer_pid)
                {
                    first_call_result = !!::atoi(tracer_pid + sizeof(TracerPid) - 1);
                }
            }
            close(status_fd);
        }
        C4_SUPPRESS_WARNING_GCC_POP
    }
    return first_call_result;
#elif defined(C4_PS4)
    return (sceDbgIsDebuggerAttached() != 0);
#elif defined(C4_XBOX) || (defined(C4_WIN) && defined(C4_MSVC))
    return IsDebuggerPresent() != 0;
#elif defined(C4_MACOS) || defined(C4_IOS)
    // https://stackoverflow.com/questions/2200277/detecting-debugger-on-mac-os-x
    // Returns true if the current process is being debugged (either
    // running under the debugger or has a debugger attached post facto).
    int                 junk;
    int                 mib[4];
    struct kinfo_proc   info;
    size_t              size;

    // Initialize the flags so that, if sysctl fails for some bizarre
    // reason, we get a predictable result.

    info.kp_proc.p_flag = 0;

    // Initialize mib, which tells sysctl the info we want, in this case
    // we're looking for information about a specific process ID.

    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = getpid();

    // Call sysctl.

    size = sizeof(info);
    junk = sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, NULL, 0);
    assert(junk == 0);

    // We're being debugged if the P_TRACED flag is set.
    return ((info.kp_proc.p_flag & P_TRACED) != 0);
#else
    return false;
#endif
} // is_debugger_attached()

} // namespace c4


#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif
