using System;

namespace Godot.NativeInterop
{
    internal static class ExceptionUtils
    {
        public static void PushError(string message)
        {
            GD.PushError(message);
        }

        private static void OnExceptionLoggerException(Exception loggerException, Exception exceptionToLog)
        {
            // This better not throw
            PushError("Exception thrown when trying to log another exception...");
            PushError("Exception:");
            PushError(exceptionToLog.ToString());
            PushError("Logger exception:");
            PushError(loggerException.ToString());
        }

        public static void DebugPrintUnhandledException(Exception e)
        {
            try
            {
                // TODO Not implemented (debug_print_unhandled_exception)
                GD.PushError(e.ToString());
            }
            catch (Exception unexpected)
            {
                OnExceptionLoggerException(unexpected, e);
            }
        }

        public static void DebugSendUnhandledExceptionError(Exception e)
        {
            try
            {
                // TODO Not implemented (debug_send_unhandled_exception_error)
                GD.PushError(e.ToString());
            }
            catch (Exception unexpected)
            {
                OnExceptionLoggerException(unexpected, e);
            }
        }

        public static void DebugUnhandledException(Exception e)
        {
            try
            {
                // TODO Not implemented (debug_unhandled_exception)
                GD.PushError(e.ToString());
            }
            catch (Exception unexpected)
            {
                OnExceptionLoggerException(unexpected, e);
            }
        }

        public static void PrintUnhandledException(Exception e)
        {
            try
            {
                // TODO Not implemented (print_unhandled_exception)
                GD.PushError(e.ToString());
            }
            catch (Exception unexpected)
            {
                OnExceptionLoggerException(unexpected, e);
            }
        }
    }
}
