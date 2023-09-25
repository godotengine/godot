#pragma once
#include <plog/Util.h>
#include <iomanip>

namespace plog
{
    class HexDump
    {
    public:
        HexDump(const void* ptr, size_t size)
            : m_ptr(static_cast<const unsigned char*>(ptr))
            , m_size(size)
            , m_group(8)
            , m_digitSeparator(" ")
            , m_groupSeparator("  ")
        {
        }

        HexDump& group(size_t group)
        {
            m_group = group;
            return *this;
        }

        HexDump& separator(const char* digitSeparator)
        {
            m_digitSeparator = digitSeparator;
            return *this;
        }

        HexDump& separator(const char* digitSeparator, const char* groupSeparator)
        {
            m_digitSeparator = digitSeparator;
            m_groupSeparator = groupSeparator;
            return *this;
        }

        friend util::nostringstream& operator<<(util::nostringstream& stream, const HexDump&);

    private:
        const unsigned char* m_ptr;
        size_t m_size;
        size_t m_group;
        const char* m_digitSeparator;
        const char* m_groupSeparator;
    };

    inline util::nostringstream& operator<<(util::nostringstream& stream, const HexDump& hexDump)
    {
        stream << std::hex << std::setfill(PLOG_NSTR('0'));

        for (size_t i = 0; i < hexDump.m_size;)
        {
            stream << std::setw(2) << static_cast<int>(hexDump.m_ptr[i]);

            if (++i < hexDump.m_size)
            {
                if (hexDump.m_group > 0 && i % hexDump.m_group == 0)
                {
                    stream << hexDump.m_groupSeparator;
                }
                else
                {
                    stream << hexDump.m_digitSeparator;
                }
            }
        }

        return stream;
    }

    inline HexDump hexdump(const void* ptr, size_t size) { return HexDump(ptr, size); }

    template<class Container>
    inline HexDump hexdump(const Container& container) { return HexDump(container.data(), container.size() * sizeof(*container.data())); }

    template<class T, size_t N>
    inline HexDump hexdump(const T (&arr)[N]) { return HexDump(arr, N * sizeof(*arr)); }
}
