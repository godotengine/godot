#pragma once
#include <plog/Appenders/IAppender.h>
#include <set>

namespace plog
{
    class PLOG_LINKAGE_HIDDEN DynamicAppender : public IAppender
    {
    public:
        DynamicAppender& addAppender(IAppender* appender)
        {
            assert(appender != this);

            util::MutexLock lock(m_mutex);
            m_appenders.insert(appender);

            return *this;
        }

        DynamicAppender& removeAppender(IAppender* appender)
        {
            util::MutexLock lock(m_mutex);
            m_appenders.erase(appender);

            return *this;
        }

        virtual void write(const Record& record) PLOG_OVERRIDE
        {
            util::MutexLock lock(m_mutex);

            for (std::set<IAppender*>::iterator it = m_appenders.begin(); it != m_appenders.end(); ++it)
            {
                (*it)->write(record);
            }
        }

    private:
        mutable util::Mutex     m_mutex;
        std::set<IAppender*>    m_appenders;
    };
}
