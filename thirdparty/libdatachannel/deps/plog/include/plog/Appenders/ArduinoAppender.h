#pragma once
#include <plog/Appenders/IAppender.h>
#include <Arduino.h>

namespace plog
{
    template<class Formatter>
    class PLOG_LINKAGE_HIDDEN ArduinoAppender : public IAppender
    {
    public:
        ArduinoAppender(Stream &stream) : m_stream(stream)
        {
        }

        virtual void write(const Record &record) PLOG_OVERRIDE
        {
            m_stream.print(Formatter::format(record).c_str());
        }

    private:
        Stream &m_stream;
    };
}
