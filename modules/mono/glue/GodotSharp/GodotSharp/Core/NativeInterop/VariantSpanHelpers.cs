using System;

namespace Godot.NativeInterop
{
    internal readonly ref struct VariantSpanDisposer
    {
        private readonly Span<godot_variant.movable> _variantSpan;

        // IMPORTANT: The span element must be default initialized.
        // Make sure call Clear() on the span if it was created with stackalloc.
        public VariantSpanDisposer(Span<godot_variant.movable> variantSpan)
        {
            _variantSpan = variantSpan;
        }

        public void Dispose()
        {
            for (int i = 0; i < _variantSpan.Length; i++)
                _variantSpan[i].DangerousSelfRef.Dispose();
        }
    }

    internal static class VariantSpanExtensions
    {
        // Used to make sure we always initialize the span values to the default,
        // as we need that in order to safely dispose all elements after.
        public static Span<godot_variant.movable> Cleared(this Span<godot_variant.movable> span)
        {
            span.Clear();
            return span;
        }
    }
}
