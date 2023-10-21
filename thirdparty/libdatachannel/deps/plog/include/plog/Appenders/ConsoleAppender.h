#pragma once
#include <plog/Appenders/IAppender.h>
#include <plog/Util.h>
#include <plog/WinApi.h>
#include <iostream>

namespace plog
{
    enum OutputStream
    {
        streamStdOut,
        streamStdErr
    };

    template<class Formatter>
    class PLOG_LINKAGE_HIDDEN ConsoleAppender : public IAppender
    {
    public:
#ifdef _WIN32
#   ifdef _MSC_VER
#       pragma warning(suppress: 26812) //  Prefer 'enum class' over 'enum'
#   endif
        ConsoleAppender(OutputStream outStream = streamStdOut)
            : m_isatty(!!_isatty(_fileno(outStream == streamStdOut ? stdout : stderr)))
            , m_outputStream(outStream == streamStdOut ? std::cout : std::cerr)
            , m_outputHandle()
        {
            if (m_isatty)
            {
                m_outputHandle = GetStdHandle(outStream == streamStdOut ? stdHandle::kOutput : stdHandle::kErrorOutput);
            }
        }
#else
        ConsoleAppender(OutputStream outStream = streamStdOut)
            : m_isatty(!!isatty(fileno(outStream == streamStdOut ? stdout : stderr)))
            , m_outputStream(outStream == streamStdOut ? std::cout : std::cerr)
        {}
#endif

        virtual void write(const Record& record) PLOG_OVERRIDE
        {
            util::nstring str = Formatter::format(record);
            util::MutexLock lock(m_mutex);

            writestr(str);
        }

    protected:
        void writestr(const util::nstring& str)
        {
#ifdef _WIN32
            if (m_isatty)
            {
                const std::wstring& wstr = util::toWide(str);
                WriteConsoleW(m_outputHandle, wstr.c_str(), static_cast<DWORD>(wstr.size()), NULL, NULL);
            }
            else
            {
#   if PLOG_CHAR_IS_UTF8
                m_outputStream << str << std::flush;
#   else
                m_outputStream << util::toNarrow(str, codePage::kActive) << std::flush;
#   endif
            }
#else
            m_outputStream << str << std::flush;
#endif
        }

    private:
#ifdef __BORLANDC__
        static int _isatty(int fd) { return ::isatty(fd); }
#endif

    protected:
        util::Mutex m_mutex;
        const bool  m_isatty;
        std::ostream& m_outputStream;
#ifdef _WIN32
        HANDLE      m_outputHandle;
#endif
    };
}
