#pragma once
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Formatters/CsvFormatter.h>
#include <plog/Init.h>
#include <cstring>

namespace plog
{
    //////////////////////////////////////////////////////////////////////////
    // RollingFileAppender with any Formatter

    template<class Formatter, int instanceId>
    PLOG_LINKAGE_HIDDEN inline Logger<instanceId>& init(Severity maxSeverity, const util::nchar* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        static RollingFileAppender<Formatter> rollingFileAppender(fileName, maxFileSize, maxFiles);
        return init<instanceId>(maxSeverity, &rollingFileAppender);
    }

    template<class Formatter>
    inline Logger<PLOG_DEFAULT_INSTANCE_ID>& init(Severity maxSeverity, const util::nchar* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return init<Formatter, PLOG_DEFAULT_INSTANCE_ID>(maxSeverity, fileName, maxFileSize, maxFiles);
    }

    //////////////////////////////////////////////////////////////////////////
    // RollingFileAppender with TXT/CSV chosen by file extension

    namespace
    {
        inline bool isCsv(const util::nchar* fileName)
        {
            const util::nchar* dot = util::findExtensionDot(fileName);
#if PLOG_CHAR_IS_UTF8
            return dot && 0 == std::strcmp(dot, ".csv");
#else
            return dot && 0 == std::wcscmp(dot, L".csv");
#endif
        }
    }

    template<int instanceId>
    inline Logger<instanceId>& init(Severity maxSeverity, const util::nchar* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return isCsv(fileName) ? init<CsvFormatter, instanceId>(maxSeverity, fileName, maxFileSize, maxFiles) : init<TxtFormatter, instanceId>(maxSeverity, fileName, maxFileSize, maxFiles);
    }

    inline Logger<PLOG_DEFAULT_INSTANCE_ID>& init(Severity maxSeverity, const util::nchar* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return init<PLOG_DEFAULT_INSTANCE_ID>(maxSeverity, fileName, maxFileSize, maxFiles);
    }

    //////////////////////////////////////////////////////////////////////////
    // CHAR variants for Windows

#if defined(_WIN32) && !PLOG_CHAR_IS_UTF8
    template<class Formatter, int instanceId>
    inline Logger<instanceId>& init(Severity maxSeverity, const char* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return init<Formatter, instanceId>(maxSeverity, util::toWide(fileName).c_str(), maxFileSize, maxFiles);
    }

    template<class Formatter>
    inline Logger<PLOG_DEFAULT_INSTANCE_ID>& init(Severity maxSeverity, const char* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return init<Formatter, PLOG_DEFAULT_INSTANCE_ID>(maxSeverity, fileName, maxFileSize, maxFiles);
    }

    template<int instanceId>
    inline Logger<instanceId>& init(Severity maxSeverity, const char* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return init<instanceId>(maxSeverity, util::toWide(fileName).c_str(), maxFileSize, maxFiles);
    }

    inline Logger<PLOG_DEFAULT_INSTANCE_ID>& init(Severity maxSeverity, const char* fileName, size_t maxFileSize = 0, int maxFiles = 0)
    {
        return init<PLOG_DEFAULT_INSTANCE_ID>(maxSeverity, fileName, maxFileSize, maxFiles);
    }
#endif
}
