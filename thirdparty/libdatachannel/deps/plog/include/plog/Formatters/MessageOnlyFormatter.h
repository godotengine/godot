#pragma once
#include <plog/Record.h>
#include <plog/Util.h>

namespace plog
{
    class MessageOnlyFormatter
    {
    public:
        static util::nstring header()
        {
            return util::nstring();
        }

        static util::nstring format(const Record& record)
        {
            util::nostringstream ss;
            ss << record.getMessage() << PLOG_NSTR("\n");

            return ss.str();
        }
    };
}
