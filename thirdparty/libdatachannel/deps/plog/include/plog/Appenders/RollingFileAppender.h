#pragma once
#include <plog/Appenders/IAppender.h>
#include <plog/Converters/UTF8Converter.h>
#include <plog/Converters/NativeEOLConverter.h>
#include <plog/Util.h>
#include <algorithm>

namespace plog
{
    template<class Formatter, class Converter = NativeEOLConverter<UTF8Converter> >
    class PLOG_LINKAGE_HIDDEN RollingFileAppender : public IAppender
    {
    public:
        RollingFileAppender(const util::nchar* fileName, size_t maxFileSize = 0, int maxFiles = 0)
            : m_fileSize()
            , m_maxFileSize()
            , m_maxFiles(maxFiles)
            , m_firstWrite(true)
        {
            setFileName(fileName);
            setMaxFileSize(maxFileSize);
        }

#if defined(_WIN32) && !PLOG_CHAR_IS_UTF8
        RollingFileAppender(const char* fileName, size_t maxFileSize = 0, int maxFiles = 0)
            : m_fileSize()
            , m_maxFileSize()
            , m_maxFiles(maxFiles)
            , m_firstWrite(true)
        {
            setFileName(fileName);
            setMaxFileSize(maxFileSize);
        }
#endif

        virtual void write(const Record& record) PLOG_OVERRIDE
        {
            util::MutexLock lock(m_mutex);

            if (m_firstWrite)
            {
                openLogFile();
                m_firstWrite = false;
            }
            else if (m_maxFiles > 0 && m_fileSize > m_maxFileSize && static_cast<size_t>(-1) != m_fileSize)
            {
                rollLogFiles();
            }

            size_t bytesWritten = m_file.write(Converter::convert(Formatter::format(record)));

            if (static_cast<size_t>(-1) != bytesWritten)
            {
                m_fileSize += bytesWritten;
            }
        }

        void setFileName(const util::nchar* fileName)
        {
            util::MutexLock lock(m_mutex);

            util::splitFileName(fileName, m_fileNameNoExt, m_fileExt);

            m_file.close();
            m_firstWrite = true;
        }

#if defined(_WIN32) && !PLOG_CHAR_IS_UTF8
        void setFileName(const char* fileName)
        {
            setFileName(util::toWide(fileName).c_str());
        }
#endif

        void setMaxFiles(int maxFiles)
        {
            m_maxFiles = maxFiles;
        }

        void setMaxFileSize(size_t maxFileSize)
        {
            m_maxFileSize = (std::max)(maxFileSize, static_cast<size_t>(1000)); // set a lower limit for the maxFileSize
        }

        void rollLogFiles()
        {
            m_file.close();

            util::nstring lastFileName = buildFileName(m_maxFiles - 1);
            util::File::unlink(lastFileName);

            for (int fileNumber = m_maxFiles - 2; fileNumber >= 0; --fileNumber)
            {
                util::nstring currentFileName = buildFileName(fileNumber);
                util::nstring nextFileName = buildFileName(fileNumber + 1);

                util::File::rename(currentFileName, nextFileName);
            }

            openLogFile();
            m_firstWrite = false;
        }

    private:
        void openLogFile()
        {
            m_fileSize = m_file.open(buildFileName());

            if (0 == m_fileSize)
            {
                size_t bytesWritten = m_file.write(Converter::header(Formatter::header()));

                if (static_cast<size_t>(-1) != bytesWritten)
                {
                    m_fileSize += bytesWritten;
                }
            }
        }

        util::nstring buildFileName(int fileNumber = 0)
        {
            util::nostringstream ss;
            ss << m_fileNameNoExt;

            if (fileNumber > 0)
            {
                ss << '.' << fileNumber;
            }

            if (!m_fileExt.empty())
            {
                ss << '.' << m_fileExt;
            }

            return ss.str();
        }

    private:
        util::Mutex     m_mutex;
        util::File      m_file;
        size_t          m_fileSize;
        size_t          m_maxFileSize;
        int             m_maxFiles;
        util::nstring   m_fileExt;
        util::nstring   m_fileNameNoExt;
        bool            m_firstWrite;
    };
}
