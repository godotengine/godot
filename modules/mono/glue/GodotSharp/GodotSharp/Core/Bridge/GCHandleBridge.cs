using System;
using System.Runtime.InteropServices;

namespace Godot.Bridge
{
    internal static class GCHandleBridge
    {
        private static void FreeGCHandle(IntPtr gcHandlePtr)
            => GCHandle.FromIntPtr(gcHandlePtr).Free();
    }
}
