using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    internal static class GCHandleBridge
    {
        [UnmanagedCallersOnly]
        internal static void FreeGCHandle(nint gcHandlePtr)
        {
            try
            {
                CustomGCHandle.Free(GCHandle.FromIntPtr(gcHandlePtr));
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }
    }
}
