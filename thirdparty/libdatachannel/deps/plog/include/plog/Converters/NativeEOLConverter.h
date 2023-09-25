#pragma once
#include <plog/Converters/UTF8Converter.h>
#include <plog/Util.h>

namespace plog
{
    template<class NextConverter = UTF8Converter>
    class NativeEOLConverter : public NextConverter
    {
#ifdef _WIN32
    public:
        static std::string header(const util::nstring& str)
        {
            return NextConverter::header(fixLineEndings(str));
        }

        static std::string convert(const util::nstring& str)
        {
            return NextConverter::convert(fixLineEndings(str));
        }

    private:
        static util::nstring fixLineEndings(const util::nstring& str)
        {
            util::nstring output;
            output.reserve(str.length() * 2); // the worst case requires 2x chars

            for (size_t i = 0; i < str.size(); ++i)
            {
                util::nchar ch = str[i];

                if (ch == PLOG_NSTR('\n'))
                {
                    output.push_back(PLOG_NSTR('\r'));
                }

                output.push_back(ch);
            }

            return output;
        }
#endif
    };
}
