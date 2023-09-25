#pragma once
#include <plog/Appenders/IAppender.h>
#include <plog/WinApi.h>

namespace plog
{
    template <class Formatter>
    class PLOG_LINKAGE_HIDDEN EventLogAppender : public IAppender
    {
    public:
        EventLogAppender(const util::nchar* sourceName) : m_eventSource(RegisterEventSourceW(NULL, util::toWide(sourceName).c_str()))
        {
        }

        ~EventLogAppender()
        {
            DeregisterEventSource(m_eventSource);
        }

        virtual void write(const Record& record) PLOG_OVERRIDE
        {
            util::nstring str = Formatter::format(record);

            write(record.getSeverity(), util::toWide(str).c_str());
        }

    private:
        void write(Severity severity, const wchar_t* str)
        {
            const wchar_t* logMessagePtr[] = { str };

            ReportEventW(m_eventSource, logSeverityToType(severity), static_cast<WORD>(severity), 0, NULL, 1, 0, logMessagePtr, NULL);
        }

        static WORD logSeverityToType(plog::Severity severity)
        {
            switch (severity)
            {
            case plog::fatal:
            case plog::error:
                return eventLog::kErrorType;

            case plog::warning:
                return eventLog::kWarningType;

            case plog::info:
            case plog::debug:
            case plog::verbose:
            default:
                return eventLog::kInformationType;
            }
        }

    private:
        HANDLE m_eventSource;
    };

    class EventLogAppenderRegistry
    {
    public:
        static bool add(const util::nchar* sourceName, const util::nchar* logName = PLOG_NSTR("Application"))
        {
            std::wstring logKeyName;
            std::wstring sourceKeyName;
            getKeyNames(sourceName, logName, sourceKeyName, logKeyName);

            HKEY sourceKey;
            if (0 != RegCreateKeyExW(hkey::kLocalMachine, sourceKeyName.c_str(), 0, NULL, 0, regSam::kSetValue, NULL, &sourceKey, NULL))
            {
                return false;
            }

            const DWORD kTypesSupported = eventLog::kErrorType | eventLog::kWarningType | eventLog::kInformationType;
            RegSetValueExW(sourceKey, L"TypesSupported", 0, regType::kDword, reinterpret_cast<const BYTE*>(&kTypesSupported), sizeof(kTypesSupported));

            const wchar_t kEventMessageFile[] = L"%windir%\\Microsoft.NET\\Framework\\v4.0.30319\\EventLogMessages.dll;%windir%\\Microsoft.NET\\Framework\\v2.0.50727\\EventLogMessages.dll";
            RegSetValueExW(sourceKey, L"EventMessageFile", 0, regType::kExpandSz, reinterpret_cast<const BYTE*>(kEventMessageFile), sizeof(kEventMessageFile) - sizeof(*kEventMessageFile));

            RegCloseKey(sourceKey);
            return true;
        }

        static bool exists(const util::nchar* sourceName, const util::nchar* logName = PLOG_NSTR("Application"))
        {
            std::wstring logKeyName;
            std::wstring sourceKeyName;
            getKeyNames(sourceName, logName, sourceKeyName, logKeyName);

            HKEY sourceKey;
            if (0 != RegOpenKeyExW(hkey::kLocalMachine, sourceKeyName.c_str(), 0, regSam::kQueryValue, &sourceKey))
            {
                return false;
            }

            RegCloseKey(sourceKey);
            return true;
        }

        static void remove(const util::nchar* sourceName, const util::nchar* logName = PLOG_NSTR("Application"))
        {
            std::wstring logKeyName;
            std::wstring sourceKeyName;
            getKeyNames(sourceName, logName, sourceKeyName, logKeyName);

            RegDeleteKeyW(hkey::kLocalMachine, sourceKeyName.c_str());
            RegDeleteKeyW(hkey::kLocalMachine, logKeyName.c_str());
        }

    private:
        static void getKeyNames(const util::nchar* sourceName, const util::nchar* logName, std::wstring& sourceKeyName, std::wstring& logKeyName)
        {
            const std::wstring kPrefix = L"SYSTEM\\CurrentControlSet\\Services\\EventLog\\";
            logKeyName = kPrefix + util::toWide(logName);
            sourceKeyName = logKeyName + L"\\" + util::toWide(sourceName);
        }
    };
}
