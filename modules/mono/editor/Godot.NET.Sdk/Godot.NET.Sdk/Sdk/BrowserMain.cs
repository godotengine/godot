using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace GodotPlugins.Game
{
    internal static class BrowserMain
    {
        [DllImport("__Internal")]
        private static extern unsafe int libgodot_web_main(int argc, byte** argv);

        [DllImport("__Internal")]
        private static extern byte libgodot_web_iteration();

        public static unsafe async Task<int> Main(string[] args)
        {
            int argc = args.Length + 1;
            byte** argv = stackalloc byte*[argc];
            IntPtr[] allocatedArgs = new IntPtr[argc];

            try
            {
                allocatedArgs[0] = StringToNativeUtf8("godot");
                argv[0] = (byte*)allocatedArgs[0];
                for (int i = 0; i < args.Length; i++)
                {
                    allocatedArgs[i + 1] = StringToNativeUtf8(args[i]);
                    argv[i + 1] = (byte*)allocatedArgs[i + 1];
                }

                int exitCode = libgodot_web_main(argc, argv);
                if (exitCode != 0)
                    return exitCode;

                while (libgodot_web_iteration() != 0)
                {
                    await Task.Delay(1);
                }

                return 0;
            }
            finally
            {
                foreach (IntPtr allocatedArg in allocatedArgs)
                {
                    if (allocatedArg != IntPtr.Zero)
                        Marshal.FreeHGlobal(allocatedArg);
                }
            }
        }

        private static IntPtr StringToNativeUtf8(string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            IntPtr buffer = Marshal.AllocHGlobal(bytes.Length + 1);
            Marshal.Copy(bytes, 0, buffer, bytes.Length);
            Marshal.WriteByte(buffer, bytes.Length, 0);
            return buffer;
        }
    }
}
