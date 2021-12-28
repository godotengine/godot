using System.Runtime.CompilerServices;

namespace Godot.NativeInterop
{
    // Our source generators will add trampolines methods that access variant arguments.
    // This struct makes that possible without having to enable `AllowUnsafeBlocks` in game projects.

    public unsafe struct NativeVariantPtrArgs
    {
        private godot_variant** _args;

        internal NativeVariantPtrArgs(godot_variant** args) => _args = args;

        public ref godot_variant this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref *_args[index];
        }
    }
}
