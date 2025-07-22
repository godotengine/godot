using System;

namespace GodotTools.IdeMessaging
{
    public interface ILogger
    {
        public void LogDebug(string message);
        public void LogInfo(string message);
        public void LogWarning(string message);
        public void LogError(string message);
        public void LogError(string message, Exception e);
    }
}
