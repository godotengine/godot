using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GodotTools.Internals
{
    public class BindingsGenerator : IDisposable
    {
        class BindingsGeneratorSafeHandle : SafeHandle
        {
            public BindingsGeneratorSafeHandle(IntPtr handle) : base(IntPtr.Zero, true)
            {
                this.handle = handle;
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            protected override bool ReleaseHandle()
            {
                internal_Dtor(handle);
                return true;
            }
        }

        private BindingsGeneratorSafeHandle safeHandle;
        private bool disposed = false;

        public bool LogPrintEnabled
        {
            get => internal_LogPrintEnabled(GetPtr());
            set => internal_SetLogPrintEnabled(GetPtr(), value);
        }

        public static uint Version => internal_Version();
        public static uint CsGlueVersion => internal_CsGlueVersion();

        public Godot.Error GenerateCsApi(string outputDir) => internal_GenerateCsApi(GetPtr(), outputDir);

        internal IntPtr GetPtr()
        {
            if (disposed)
                throw new ObjectDisposedException(GetType().FullName);

            return safeHandle.DangerousGetHandle();
        }

        public void Dispose()
        {
            if (disposed)
                return;

            if (safeHandle != null && !safeHandle.IsInvalid)
            {
                safeHandle.Dispose();
                safeHandle = null;
            }

            disposed = true;
        }

        public BindingsGenerator()
        {
            safeHandle = new BindingsGeneratorSafeHandle(internal_Ctor());
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr internal_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_Dtor(IntPtr handle);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_LogPrintEnabled(IntPtr handle);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_SetLogPrintEnabled(IntPtr handle, bool enabled);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Godot.Error internal_GenerateCsApi(IntPtr handle, string outputDir);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern uint internal_Version();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern uint internal_CsGlueVersion();
    }
}
