using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    [StructLayout(LayoutKind.Sequential)]
    public struct RID
    {
        private ulong _id; // Default is 0

        internal RID(ulong id)
        {
            _id = id;
        }

        public RID(Object from)
            => _id = from is Resource res ? res.GetRid()._id : default;

        public ulong Id => _id;

        public override string ToString() => $"RID({Id})";
    }
}
