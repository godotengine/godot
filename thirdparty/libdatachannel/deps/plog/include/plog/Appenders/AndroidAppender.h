#pragma once
#include <plog/Appenders/IAppender.h>
#include <android/log.h>

namespace plog
{
    template<class Formatter>
    class PLOG_LINKAGE_HIDDEN AndroidAppender : public IAppender
    {
    public:
        AndroidAppender(const char* tag) : m_tag(tag)
        {
        }

        virtual void write(const Record& record) PLOG_OVERRIDE
        {
            std::string str = Formatter::format(record);

            __android_log_print(toPriority(record.getSeverity()), m_tag, "%s", str.c_str());
        }

    private:
        static android_LogPriority toPriority(Severity severity)
        {
            switch (severity)
            {
            case fatal:
                return ANDROID_LOG_FATAL;
            case error:
                return ANDROID_LOG_ERROR;
            case warning:
                return ANDROID_LOG_WARN;
            case info:
                return ANDROID_LOG_INFO;
            case debug:
                return ANDROID_LOG_DEBUG;
            case verbose:
                return ANDROID_LOG_VERBOSE;
            default:
                return ANDROID_LOG_UNKNOWN;
            }
        }

    private:
        const char* const m_tag;
    };
}
