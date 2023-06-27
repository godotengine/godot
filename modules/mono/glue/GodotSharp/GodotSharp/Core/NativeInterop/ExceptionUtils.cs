using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

#nullable enable

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
            try
            {
                // This better not throw
                PushError(string.Concat("Exception thrown while trying to log another exception...",
                    "\n### Exception ###\n", exceptionToLog.ToString(),
                    "\n### Logger exception ###\n", loggerException.ToString()));
            }
            catch (Exception)
            {
                // Well, too bad...
            }
        }

        private record struct StackInfoTuple(string? File, string Func, int Line);

        private static void CollectExceptionInfo(Exception exception, List<StackInfoTuple> globalFrames,
            StringBuilder excMsg)
        {
            if (excMsg.Length > 0)
                excMsg.Append(" ---> ");
            excMsg.Append(exception.GetType().FullName);
            excMsg.Append(": ");
            excMsg.Append(exception.Message);

            var innerExc = exception.InnerException;

            if (innerExc != null)
            {
                CollectExceptionInfo(innerExc, globalFrames, excMsg);
                globalFrames.Add(new("", "--- End of inner exception stack trace ---", 0));
            }

            var stackTrace = new StackTrace(exception, fNeedFileInfo: true);

            foreach (StackFrame frame in stackTrace.GetFrames())
            {
                DebuggingUtils.GetStackFrameMethodDecl(frame, out string methodDecl);
                globalFrames.Add(new(frame.GetFileName(), methodDecl, frame.GetFileLineNumber()));
            }
        }

        private static void SendToScriptDebugger(Exception e)
        {
            var globalFrames = new List<StackInfoTuple>();

            var excMsg = new StringBuilder();

            CollectExceptionInfo(e, globalFrames, excMsg);

            string file = globalFrames.Count > 0 ? globalFrames[0].File ?? "" : "";
            string func = globalFrames.Count > 0 ? globalFrames[0].Func : "";
            int line = globalFrames.Count > 0 ? globalFrames[0].Line : 0;
            string errorMsg = e.GetType().FullName ?? "";

            using godot_string nFile = Marshaling.ConvertStringToNative(file);
            using godot_string nFunc = Marshaling.ConvertStringToNative(func);
            using godot_string nErrorMsg = Marshaling.ConvertStringToNative(errorMsg);
            using godot_string nExcMsg = Marshaling.ConvertStringToNative(excMsg.ToString());

            using DebuggingUtils.godot_stack_info_vector stackInfoVector = default;

            stackInfoVector.Resize(globalFrames.Count);

            unsafe
            {
                for (int i = 0; i < globalFrames.Count; i++)
                {
                    DebuggingUtils.godot_stack_info* stackInfo = &stackInfoVector.Elements[i];

                    var globalFrame = globalFrames[i];

                    // Assign directly to element in Vector. This way we don't need to worry
                    // about disposal if an exception is thrown. The Vector takes care of it.
                    stackInfo->File = Marshaling.ConvertStringToNative(globalFrame.File);
                    stackInfo->Func = Marshaling.ConvertStringToNative(globalFrame.Func);
                    stackInfo->Line = globalFrame.Line;
                }

                NativeFuncs.godotsharp_internal_script_debugger_send_error(nFunc, nFile, line,
                    nErrorMsg, nExcMsg, p_warning: godot_bool.False, stackInfoVector);
            }
        }

        public static void LogException(Exception e)
        {
            try
            {
                if (NativeFuncs.godotsharp_internal_script_debugger_is_active().ToBool())
                {
                    SendToScriptDebugger(e);
                }
                else
                {
                    GD.PushError(e.ToString());
                }
            }
            catch (Exception unexpected)
            {
                OnExceptionLoggerException(unexpected, e);
            }
        }

        public static void LogUnhandledException(Exception e)
        {
            try
            {
                if (NativeFuncs.godotsharp_internal_script_debugger_is_active().ToBool())
                {
                    SendToScriptDebugger(e);
                }

                // In this case, print it as well in addition to sending it to the script debugger
                GD.PushError("Unhandled exception\n" + e);
            }
            catch (Exception unexpected)
            {
                OnExceptionLoggerException(unexpected, e);
            }
        }
    }
}
