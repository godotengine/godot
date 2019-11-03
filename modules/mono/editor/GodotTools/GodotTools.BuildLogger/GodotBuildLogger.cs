using System;
using System.IO;
using System.Security;
using Microsoft.Build.Framework;
using GodotTools.Core;

namespace GodotTools.BuildLogger
{
    public class GodotBuildLogger : ILogger
    {
        public static readonly string AssemblyPath = Path.GetFullPath(typeof(GodotBuildLogger).Assembly.Location);

        public string Parameters { get; set; }
        public LoggerVerbosity Verbosity { get; set; }

        public void Initialize(IEventSource eventSource)
        {
            if (null == Parameters)
                throw new LoggerException("Log directory was not set.");

            var parameters = Parameters.Split(new[] {';'});

            string logDir = parameters[0];

            if (string.IsNullOrEmpty(logDir))
                throw new LoggerException("Log directory was not set.");

            if (parameters.Length > 1)
                throw new LoggerException("Too many parameters passed.");

            string logFile = Path.Combine(logDir, "msbuild_log.txt");
            string issuesFile = Path.Combine(logDir, "msbuild_issues.csv");

            try
            {
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);

                logStreamWriter = new StreamWriter(logFile);
                issuesStreamWriter = new StreamWriter(issuesFile);
            }
            catch (Exception ex)
            {
                if (ex is UnauthorizedAccessException
                    || ex is ArgumentNullException
                    || ex is PathTooLongException
                    || ex is DirectoryNotFoundException
                    || ex is NotSupportedException
                    || ex is ArgumentException
                    || ex is SecurityException
                    || ex is IOException)
                {
                    throw new LoggerException("Failed to create log file: " + ex.Message);
                }
                else
                {
                    // Unexpected failure
                    throw;
                }
            }

            eventSource.ProjectStarted += eventSource_ProjectStarted;
            eventSource.TaskStarted += eventSource_TaskStarted;
            eventSource.MessageRaised += eventSource_MessageRaised;
            eventSource.WarningRaised += eventSource_WarningRaised;
            eventSource.ErrorRaised += eventSource_ErrorRaised;
            eventSource.ProjectFinished += eventSource_ProjectFinished;
        }

        void eventSource_ErrorRaised(object sender, BuildErrorEventArgs e)
        {
            string line = $"{e.File}({e.LineNumber},{e.ColumnNumber}): error {e.Code}: {e.Message}";

            if (e.ProjectFile.Length > 0)
                line += $" [{e.ProjectFile}]";

            WriteLine(line);

            string errorLine = $@"error,{e.File.CsvEscape()},{e.LineNumber},{e.ColumnNumber}," +
                               $@"{e.Code.CsvEscape()},{e.Message.CsvEscape()},{e.ProjectFile.CsvEscape()}";
            issuesStreamWriter.WriteLine(errorLine);
        }

        void eventSource_WarningRaised(object sender, BuildWarningEventArgs e)
        {
            string line = $"{e.File}({e.LineNumber},{e.ColumnNumber}): warning {e.Code}: {e.Message}";

            if (!string.IsNullOrEmpty(e.ProjectFile))
                line += $" [{e.ProjectFile}]";

            WriteLine(line);

            string warningLine = $@"warning,{e.File.CsvEscape()},{e.LineNumber},{e.ColumnNumber},{e.Code.CsvEscape()}," +
                                 $@"{e.Message.CsvEscape()},{(e.ProjectFile != null ? e.ProjectFile.CsvEscape() : string.Empty)}";
            issuesStreamWriter.WriteLine(warningLine);
        }

        private void eventSource_MessageRaised(object sender, BuildMessageEventArgs e)
        {
            // BuildMessageEventArgs adds Importance to BuildEventArgs
            // Let's take account of the verbosity setting we've been passed in deciding whether to log the message
            if (e.Importance == MessageImportance.High && IsVerbosityAtLeast(LoggerVerbosity.Minimal)
                || e.Importance == MessageImportance.Normal && IsVerbosityAtLeast(LoggerVerbosity.Normal)
                || e.Importance == MessageImportance.Low && IsVerbosityAtLeast(LoggerVerbosity.Detailed))
            {
                WriteLineWithSenderAndMessage(string.Empty, e);
            }
        }

        private void eventSource_TaskStarted(object sender, TaskStartedEventArgs e)
        {
            // TaskStartedEventArgs adds ProjectFile, TaskFile, TaskName
            // To keep this log clean, this logger will ignore these events.
        }

        private void eventSource_ProjectStarted(object sender, ProjectStartedEventArgs e)
        {
            WriteLine(e.Message);
            indent++;
        }

        private void eventSource_ProjectFinished(object sender, ProjectFinishedEventArgs e)
        {
            indent--;
            WriteLine(e.Message);
        }

        /// <summary>
        /// Write a line to the log, adding the SenderName
        /// </summary>
        private void WriteLineWithSender(string line, BuildEventArgs e)
        {
            if (0 == string.Compare(e.SenderName, "MSBuild", StringComparison.OrdinalIgnoreCase))
            {
                // Well, if the sender name is MSBuild, let's leave it out for prettiness
                WriteLine(line);
            }
            else
            {
                WriteLine(e.SenderName + ": " + line);
            }
        }

        /// <summary>
        /// Write a line to the log, adding the SenderName and Message
        /// (these parameters are on all MSBuild event argument objects)
        /// </summary>
        private void WriteLineWithSenderAndMessage(string line, BuildEventArgs e)
        {
            if (0 == string.Compare(e.SenderName, "MSBuild", StringComparison.OrdinalIgnoreCase))
            {
                // Well, if the sender name is MSBuild, let's leave it out for prettiness
                WriteLine(line + e.Message);
            }
            else
            {
                WriteLine(e.SenderName + ": " + line + e.Message);
            }
        }

        private void WriteLine(string line)
        {
            for (int i = indent; i > 0; i--)
            {
                logStreamWriter.Write("\t");
            }

            logStreamWriter.WriteLine(line);
        }

        public void Shutdown()
        {
            logStreamWriter.Close();
            issuesStreamWriter.Close();
        }

        private bool IsVerbosityAtLeast(LoggerVerbosity checkVerbosity)
        {
            return Verbosity >= checkVerbosity;
        }

        private StreamWriter logStreamWriter;
        private StreamWriter issuesStreamWriter;
        private int indent;
    }
}
