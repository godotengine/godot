using System;

namespace GodotTools.IdeConnection
{
    public class ConsoleLogger : ILogger
    {
        public void LogDebug(string message)
        {
            Console.WriteLine("DEBUG: " + message);
        }

        public void LogInfo(string message)
        {
            Console.WriteLine("INFO: " + message);
        }

        public void LogWarning(string message)
        {
            Console.WriteLine("WARN: " + message);
        }

        public void LogError(string message)
        {
            Console.WriteLine("ERROR: " + message);
        }

        public void LogError(string message, Exception e)
        {
            Console.WriteLine("EXCEPTION: " + message);
            Console.WriteLine(e);
        }
    }
}
