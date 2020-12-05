using System.IO;

namespace GodotTools.Core
{
    public static class FileUtils
    {
        public static void SaveBackupCopy(string filePath)
        {
            string backupPathBase = filePath + ".old";
            string backupPath = backupPathBase;

            const int maxAttempts = 5;
            int attempt = 1;

            while (File.Exists(backupPath) && attempt <= maxAttempts)
            {
                backupPath = backupPathBase + "." + (attempt);
                attempt++;
            }

            if (attempt > maxAttempts + 1)
            {
                // Overwrite the oldest one
                backupPath = backupPathBase;
            }

            File.Copy(filePath, backupPath, overwrite: true);
        }
    }
}
