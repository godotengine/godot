// GodotNative.cs
// P/Invoke surface for the godot_uwp_* C ABI exported by godot.dll
// (built from the modified Godot 4.6 engine: platform/windows/godot_uwp_embed.cpp).
//
// Every function here is thread-affine to the single engine thread, except
// the configuration setters which must be called BEFORE engine setup.
// Host code should use GodotEngineHost, which marshals everything onto the
// dedicated engine thread.
//
// UWP/.NET Native constraints honored here:
//  - No Marshal.PtrToStringUTF8 (manual UTF-8 marshalling).
//  - All native callback delegates are kept alive in static fields.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Godot.Uwp.Embedding
{
    internal enum GodotLogLevel
    {
        Print = 0,
        Warning = 1,
        Error = 2,
    }

    /// <summary>Godot MouseButton values (core/input/input_enums.h).</summary>
    internal enum GodotMouseButton
    {
        None = 0,
        Left = 1,
        Right = 2,
        Middle = 3,
        WheelUp = 4,
        WheelDown = 5,
        WheelLeft = 6,
        WheelRight = 7,
        XButton1 = 8,
        XButton2 = 9,
    }

    internal static class GodotNative
    {
        // godot.dll must be deployed in the appx package root (x64).
        private const string DLL = "godot.dll";

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void LogCallback(IntPtr messageUtf8, int level);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void WorkFunc(IntPtr userdata);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void UiDispatchFunc(IntPtr work, IntPtr userdata);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_set_log_callback(LogCallback callback);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_set_swap_chain_panel(IntPtr panelNativeIUnknown);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_set_ui_dispatcher(UiDispatchFunc dispatch);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int godot_uwp_engine_setup(int argc, IntPtr argv);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int godot_uwp_engine_start();

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int godot_uwp_engine_iteration();

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_engine_shutdown();

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_notify_panel_resize(int widthPx, int heightPx);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_set_composition_scale(float scaleX, float scaleY);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_inject_mouse_button(int button, int pressed, float x, float y, int doubleClick);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_inject_mouse_motion(float x, float y, float relX, float relY);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_inject_mouse_wheel(float x, float y, float deltaX, float deltaY);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_inject_key(uint winVk, int pressed, int echo, uint unicode);

        // --- Host <-> engine JSON message bus -----------------------------

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void HostMsgCallback(IntPtr methodUtf8, IntPtr argsJsonUtf8);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_set_host_message_callback(HostMsgCallback callback);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_set_call_return(IntPtr jsonUtf8);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int godot_uwp_call_engine(IntPtr methodUtf8, IntPtr argsJsonUtf8, out IntPtr retJsonUtf8);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void godot_uwp_free_string(IntPtr str);

        // -----------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------

        /// <summary>Reads a null-terminated UTF-8 string from native memory.</summary>
        internal static string Utf8ToString(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                return string.Empty;
            }
            int len = 0;
            while (Marshal.ReadByte(ptr, len) != 0)
            {
                len++;
            }
            if (len == 0)
            {
                return string.Empty;
            }
            byte[] buffer = new byte[len];
            Marshal.Copy(ptr, buffer, 0, len);
            return Encoding.UTF8.GetString(buffer, 0, len);
        }

        /// <summary>Allocates a null-terminated UTF-8 copy of <paramref name="s"/> (HGlobal).</summary>
        internal static IntPtr Utf8Alloc(string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s + '\0');
            IntPtr ptr = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, ptr, bytes.Length);
            return ptr;
        }

        /// <summary>
        /// Marshals managed args to a native UTF-8 char** and invokes setup.
        /// </summary>
        internal static bool EngineSetup(string[] args)
        {
            if (args == null || args.Length < 1)
            {
                throw new ArgumentException("args must contain at least argv[0].", "args");
            }

            IntPtr[] utf8Ptrs = new IntPtr[args.Length];
            IntPtr argv = Marshal.AllocHGlobal(IntPtr.Size * args.Length);
            try
            {
                for (int i = 0; i < args.Length; i++)
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(args[i] + '\0');
                    utf8Ptrs[i] = Marshal.AllocHGlobal(bytes.Length);
                    Marshal.Copy(bytes, 0, utf8Ptrs[i], bytes.Length);
                    Marshal.WriteIntPtr(argv, i * IntPtr.Size, utf8Ptrs[i]);
                }
                return godot_uwp_engine_setup(args.Length, argv) != 0;
            }
            finally
            {
                for (int i = 0; i < utf8Ptrs.Length; i++)
                {
                    if (utf8Ptrs[i] != IntPtr.Zero)
                    {
                        Marshal.FreeHGlobal(utf8Ptrs[i]);
                    }
                }
                Marshal.FreeHGlobal(argv);
            }
        }
    }
}
