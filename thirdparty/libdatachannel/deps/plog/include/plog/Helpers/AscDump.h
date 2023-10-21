#pragma once
#include <plog/Util.h>
#include <cctype>

namespace plog
{
    class AscDump
    {
    public:
        AscDump(const void* ptr, size_t size)
            : m_ptr(static_cast<const char*>(ptr))
            , m_size(size)
        {
        }

        friend util::nostringstream& operator<<(util::nostringstream& stream, const AscDump& ascDump);

    private:
        const char* m_ptr;
        size_t m_size;
    };

    inline util::nostringstream& operator<<(util::nostringstream& stream, const AscDump& ascDump)
    {
        for (size_t i = 0; i < ascDump.m_size; ++i)
        {
            stream << (std::isprint(ascDump.m_ptr[i]) ? ascDump.m_ptr[i] : '.');
        }

        return stream;
    }

    inline AscDump ascdump(const void* ptr, size_t size) { return AscDump(ptr, size); }

    template<class Container>
    inline AscDump ascdump(const Container& container) { return AscDump(container.data(), container.size() * sizeof(*container.data())); }

    template<class T, size_t N>
    inline AscDump ascdump(const T (&arr)[N]) { return AscDump(arr, N * sizeof(*arr)); }
}
