using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.Build.Framework;

namespace GodotSharpTools.Build
{
    public class BuildInstance : IDisposable
    {
        [MethodImpl(MethodImplOptions.InternalCall)]
        private extern static void godot_icall_BuildInstance_ExitCallback(string solution, string config, int exitCode);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private extern static void godot_icall_BuildInstance_get_MSBuildInfo(ref string msbuildPath, ref string frameworkPath);

        private struct MSBuildInfo
        {
            public string path;
            public string frameworkPathOverride;
        }

        private static MSBuildInfo GetMSBuildInfo()
        {
            MSBuildInfo msbuildInfo = new MSBuildInfo();

            godot_icall_BuildInstance_get_MSBuildInfo(ref msbuildInfo.path, ref msbuildInfo.frameworkPathOverride);

            if (msbuildInfo.path == null)
                throw new FileNotFoundException("Cannot find the MSBuild executable.");

            return msbuildInfo;
        }

        private string solution;
        private string config;

        private Process process;

        private int exitCode;
        public int ExitCode { get { return exitCode; } }

        public bool IsRunning { get { return process != null && !process.HasExited; } }

        public BuildInstance(string solution, string config)
        {
            this.solution = solution;
            this.config = config;
        }

        public bool Build(string loggerAssemblyPath, string loggerOutputDir, string[] customProperties = null)
        {
            MSBuildInfo msbuildInfo = GetMSBuildInfo();

            List<string> customPropertiesList = new List<string>();

            if (customProperties != null)
                customPropertiesList.AddRange(customProperties);

            if (msbuildInfo.frameworkPathOverride != null)
                customPropertiesList.Add("FrameworkPathOverride=" + msbuildInfo.frameworkPathOverride);

            string compilerArgs = BuildArguments(loggerAssemblyPath, loggerOutputDir, customPropertiesList);

            ProcessStartInfo startInfo = new ProcessStartInfo(msbuildInfo.path, compilerArgs);

            // No console output, thanks
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.UseShellExecute = false;

            // Needed when running from Developer Command Prompt for VS
            RemovePlatformVariable(startInfo.EnvironmentVariables);

            using (Process process = new Process())
            {
                process.StartInfo = startInfo;

                process.Start();

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                process.WaitForExit();

                exitCode = process.ExitCode;
            }

            return true;
        }

        public bool BuildAsync(string loggerAssemblyPath, string loggerOutputDir, string[] customProperties = null)
        {
            if (process != null)
                throw new InvalidOperationException("Already in use");

            MSBuildInfo msbuildInfo = GetMSBuildInfo();

            List<string> customPropertiesList = new List<string>();

            if (customProperties != null)
                customPropertiesList.AddRange(customProperties);

            if (msbuildInfo.frameworkPathOverride.Length > 0)
                customPropertiesList.Add("FrameworkPathOverride=" + msbuildInfo.frameworkPathOverride);

            string compilerArgs = BuildArguments(loggerAssemblyPath, loggerOutputDir, customPropertiesList);

            ProcessStartInfo startInfo = new ProcessStartInfo(msbuildInfo.path, compilerArgs);

            // No console output, thanks
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.UseShellExecute = false;

            // Needed when running from Developer Command Prompt for VS
            RemovePlatformVariable(startInfo.EnvironmentVariables);

            process = new Process();
            process.StartInfo = startInfo;
            process.EnableRaisingEvents = true;
            process.Exited += new EventHandler(BuildProcess_Exited);

            process.Start();

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            return true;
        }

        private string BuildArguments(string loggerAssemblyPath, string loggerOutputDir, List<string> customProperties)
        {
            string arguments = string.Format(@"""{0}"" /v:normal /t:Build ""/p:{1}"" ""/l:{2},{3};{4}""",
                solution,
                "Configuration=" + config,
                typeof(GodotBuildLogger).FullName,
                loggerAssemblyPath,
                loggerOutputDir
            );

            foreach (string customProperty in customProperties)
            {
                arguments += " \"/p:" + customProperty + "\"";
            }

            return arguments;
        }

        private void RemovePlatformVariable(StringDictionary environmentVariables)
        {
            // EnvironmentVariables is case sensitive? Seriously?

            List<string> platformEnvironmentVariables = new List<string>();

            foreach (string env in environmentVariables.Keys)
            {
                if (env.ToUpper() == "PLATFORM")
                    platformEnvironmentVariables.Add(env);
            }

            foreach (string env in platformEnvironmentVariables)
                environmentVariables.Remove(env);
        }

        private void BuildProcess_Exited(object sender, System.EventArgs e)
        {
            exitCode = process.ExitCode;

            godot_icall_BuildInstance_ExitCallback(solution, config, exitCode);

            Dispose();
        }

        public void Dispose()
        {
            if (process != null)
            {
                process.Dispose();
                process = null;
            }
        }
    }

    public class GodotBuildLogger : ILogger
    {
        public string Parameters { get; set; }
        public LoggerVerbosity Verbosity { get; set; }

        public void Initialize(IEventSource eventSource)
        {
            if (null == Parameters)
                throw new LoggerException("Log directory was not set.");

            string[] parameters = Parameters.Split(';');

            string logDir = parameters[0];

            if (String.IsNullOrEmpty(logDir))
                throw new LoggerException("Log directory was not set.");

            if (parameters.Length > 1)
                throw new LoggerException("Too many parameters passed.");

            string logFile = Path.Combine(logDir, "msbuild_log.txt");
            string issuesFile = Path.Combine(logDir, "msbuild_issues.csv");

            try
            {
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);

                this.logStreamWriter = new StreamWriter(logFile);
                this.issuesStreamWriter = new StreamWriter(issuesFile);
            }
            catch (Exception ex)
            {
                if
                (
                    ex is UnauthorizedAccessException
                    || ex is ArgumentNullException
                    || ex is PathTooLongException
                    || ex is DirectoryNotFoundException
                    || ex is NotSupportedException
                    || ex is ArgumentException
                    || ex is SecurityException
                    || ex is IOException
                )
                {
                    throw new LoggerException("Failed to create log file: " + ex.Message);
                }
                else
                {
                    // Unexpected failure
                    throw;
                }
            }

            eventSource.ProjectStarted += new ProjectStartedEventHandler(eventSource_ProjectStarted);
            eventSource.TaskStarted += new TaskStartedEventHandler(eventSource_TaskStarted);
            eventSource.MessageRaised += new BuildMessageEventHandler(eventSource_MessageRaised);
            eventSource.WarningRaised += new BuildWarningEventHandler(eventSource_WarningRaised);
            eventSource.ErrorRaised += new BuildErrorEventHandler(eventSource_ErrorRaised);
            eventSource.ProjectFinished += new ProjectFinishedEventHandler(eventSource_ProjectFinished);
        }

        void eventSource_ErrorRaised(object sender, BuildErrorEventArgs e)
        {
            string line = String.Format("{0}({1},{2}): error {3}: {4}", e.File, e.LineNumber, e.ColumnNumber, e.Code, e.Message);

            if (e.ProjectFile.Length > 0)
                line += string.Format(" [{0}]", e.ProjectFile);

            WriteLine(line);

            string errorLine = String.Format(@"error,{0},{1},{2},{3},{4},{5}",
                                    e.File.CsvEscape(), e.LineNumber, e.ColumnNumber,
                                    e.Code.CsvEscape(), e.Message.CsvEscape(), e.ProjectFile.CsvEscape());
            issuesStreamWriter.WriteLine(errorLine);
        }

        void eventSource_WarningRaised(object sender, BuildWarningEventArgs e)
        {
            string line = String.Format("{0}({1},{2}): warning {3}: {4}", e.File, e.LineNumber, e.ColumnNumber, e.Code, e.Message, e.ProjectFile);

            if (e.ProjectFile != null && e.ProjectFile.Length > 0)
                line += string.Format(" [{0}]", e.ProjectFile);

            WriteLine(line);

            string warningLine = String.Format(@"warning,{0},{1},{2},{3},{4},{5}",
                                    e.File.CsvEscape(), e.LineNumber, e.ColumnNumber,
                                    e.Code.CsvEscape(), e.Message.CsvEscape(), e.ProjectFile != null ? e.ProjectFile.CsvEscape() : string.Empty);
            issuesStreamWriter.WriteLine(warningLine);
        }

        void eventSource_MessageRaised(object sender, BuildMessageEventArgs e)
        {
            // BuildMessageEventArgs adds Importance to BuildEventArgs
            // Let's take account of the verbosity setting we've been passed in deciding whether to log the message
            if ((e.Importance == MessageImportance.High && IsVerbosityAtLeast(LoggerVerbosity.Minimal))
                || (e.Importance == MessageImportance.Normal && IsVerbosityAtLeast(LoggerVerbosity.Normal))
                || (e.Importance == MessageImportance.Low && IsVerbosityAtLeast(LoggerVerbosity.Detailed))
                )
            {
                WriteLineWithSenderAndMessage(String.Empty, e);
            }
        }

        void eventSource_TaskStarted(object sender, TaskStartedEventArgs e)
        {
            // TaskStartedEventArgs adds ProjectFile, TaskFile, TaskName
            // To keep this log clean, this logger will ignore these events.
        }

        void eventSource_ProjectStarted(object sender, ProjectStartedEventArgs e)
        {
            WriteLine(e.Message);
            indent++;
        }

        void eventSource_ProjectFinished(object sender, ProjectFinishedEventArgs e)
        {
            indent--;
            WriteLine(e.Message);
        }

        /// <summary>
        /// Write a line to the log, adding the SenderName
        /// </summary>
        private void WriteLineWithSender(string line, BuildEventArgs e)
        {
            if (0 == String.Compare(e.SenderName, "MSBuild", true /*ignore case*/))
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
            if (0 == String.Compare(e.SenderName, "MSBuild", true /*ignore case*/))
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

        public bool IsVerbosityAtLeast(LoggerVerbosity checkVerbosity)
        {
            return this.Verbosity >= checkVerbosity;
        }

        private StreamWriter logStreamWriter;
        private StreamWriter issuesStreamWriter;
        private int indent;
    }
}
