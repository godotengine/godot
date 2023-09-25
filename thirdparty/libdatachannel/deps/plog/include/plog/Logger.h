#pragma once
#include <plog/Appenders/IAppender.h>
#include <plog/Util.h>
#include <vector>

#ifdef PLOG_DEFAULT_INSTANCE // for backward compatibility
#   define PLOG_DEFAULT_INSTANCE_ID PLOG_DEFAULT_INSTANCE
#endif

#ifndef PLOG_DEFAULT_INSTANCE_ID
#   define PLOG_DEFAULT_INSTANCE_ID 0
#endif

namespace plog
{
    template<int instanceId>
    class PLOG_LINKAGE Logger : public util::Singleton<Logger<instanceId> >, public IAppender
    {
    public:
        Logger(Severity maxSeverity = none) : m_maxSeverity(maxSeverity)
        {
        }

        Logger& addAppender(IAppender* appender)
        {
            assert(appender != this);
            m_appenders.push_back(appender);
            return *this;
        }

        Severity getMaxSeverity() const
        {
            return m_maxSeverity;
        }

        void setMaxSeverity(Severity severity)
        {
            m_maxSeverity = severity;
        }

        bool checkSeverity(Severity severity) const
        {
            return severity <= m_maxSeverity;
        }

        virtual void write(const Record& record) PLOG_OVERRIDE
        {
            if (checkSeverity(record.getSeverity()))
            {
                *this += record;
            }
        }

        void operator+=(const Record& record)
        {
            for (std::vector<IAppender*>::iterator it = m_appenders.begin(); it != m_appenders.end(); ++it)
            {
                (*it)->write(record);
            }
        }

    private:
        Severity m_maxSeverity;
#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable:4251) // needs to have dll-interface to be used by clients of class
#endif
        std::vector<IAppender*> m_appenders;
#ifdef _MSC_VER
#   pragma warning(pop)
#endif
    };

    template<int instanceId>
    inline Logger<instanceId>* get()
    {
        return Logger<instanceId>::getInstance();
    }

    inline Logger<PLOG_DEFAULT_INSTANCE_ID>* get()
    {
        return Logger<PLOG_DEFAULT_INSTANCE_ID>::getInstance();
    }
}
