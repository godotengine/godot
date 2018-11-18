using System;
using System.Linq;
using System.Runtime.CompilerServices;

namespace GodotSharpTools.Utils
{
    public static class OS
    {
        [MethodImpl(MethodImplOptions.InternalCall)]
        extern static string GetPlatformName();

        const string HaikuName = "Haiku";
        const string OSXName = "OSX";
        const string ServerName = "Server";
        const string UWPName = "UWP";
        const string WindowsName = "Windows";
        const string X11Name = "X11";

        public static bool IsHaiku()
        {
            return HaikuName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsOSX()
        {
            return OSXName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsServer()
        {
            return ServerName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsUWP()
        {
            return UWPName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsWindows()
        {
            return WindowsName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsX11()
        {
            return X11Name.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        static bool? IsUnixCache = null;
        static readonly string[] UnixPlatforms = new string[] { HaikuName, OSXName, ServerName, X11Name };

        public static bool IsUnix()
        {
            if (IsUnixCache.HasValue)
                return IsUnixCache.Value;

            string osName = GetPlatformName();
            IsUnixCache = UnixPlatforms.Any(p => p.Equals(osName, StringComparison.OrdinalIgnoreCase));
            return IsUnixCache.Value;
        }
    }
}
