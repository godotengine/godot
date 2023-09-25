#pragma once
#include <plog/Util.h>

namespace plog
{
    class UTF8Converter
    {
    public:
        static std::string header(const util::nstring& str)
        {
            const char kBOM[] = "\xEF\xBB\xBF";

            return std::string(kBOM) + convert(str);
        }

#if PLOG_CHAR_IS_UTF8
        static const std::string& convert(const util::nstring& str)
        {
            return str;
        }
#else
        static std::string convert(const util::nstring& str)
        {
            return util::toNarrow(str, codePage::kUTF8);
        }
#endif
    };
}
