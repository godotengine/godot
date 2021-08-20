using System;

namespace Godot.NativeInterop
{
    internal ref struct VariantSpanDisposer
    {
        private readonly Span<godot_variant> _variantSpan;

        // IMPORTANT: The span element must be default initialized.
        // Make sure call Clear() on the span if it was created with stackalloc.
        public VariantSpanDisposer(Span<godot_variant> variantSpan)
        {
            _variantSpan = variantSpan;
        }

        public void Dispose()
        {
            for (int i = 0; i < _variantSpan.Length; i++)
                _variantSpan[i].Dispose();
        }
    }

    internal static class VariantSpanExtensions
    {
        // Used to make sure we always initialize the span values to the default,
        // as we need that in order to safely dispose all elements after.
        public static Span<godot_variant> Cleared(this Span<godot_variant> span)
        {
            span.Clear();
            return span;
        }
    }
}
