#pragma once
#include <cctype>

namespace plog
{
    enum Severity
    {
        none = 0,
        fatal = 1,
        error = 2,
        warning = 3,
        info = 4,
        debug = 5,
        verbose = 6
    };

#ifdef _MSC_VER
#   pragma warning(suppress: 26812) //  Prefer 'enum class' over 'enum'
#endif
    inline const char* severityToString(Severity severity)
    {
        switch (severity)
        {
        case fatal:
            return "FATAL";
        case error:
            return "ERROR";
        case warning:
            return "WARN";
        case info:
            return "INFO";
        case debug:
            return "DEBUG";
        case verbose:
            return "VERB";
        default:
            return "NONE";
        }
    }

    inline Severity severityFromString(const char* str)
    {
        switch (std::toupper(str[0]))
        {
        case 'F':
            return fatal;
        case 'E':
            return error;
        case 'W':
            return warning;
        case 'I':
            return info;
        case 'D':
            return debug;
        case 'V':
            return verbose;
        default:
            return none;
        }
    }
}
