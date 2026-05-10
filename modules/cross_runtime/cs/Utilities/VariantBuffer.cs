using System;
using System.Collections;
using System.Text;

public class VariantBuffer
{
    // Useful for Tracking
    public int Cursor { get; private set; }

    public VariantBuffer(int startOffset) => Cursor = startOffset;

    // Relies on VariantHandling Encoder to write correctly
    // Follows the cursor tracking is shaping what is being written where and where in the relevant files
    public void Write(object value)
    {
        Cursor += VariantHandling.Encode(Cursor, value);
    }

    // Uses the Decoding from VariantHandler to understand how and where to read
    public object Read()
    {
        object result = VariantHandling.DecodeInternal(Cursor, out int bytesRead);
        Cursor += bytesRead;
        return result;
    }


    public void WriteCallable(CustomCallable callable)
    {
        Helpers.WriteUInt64(Cursor, callable.TargetId);
        Helpers.WriteString(Cursor + 8, callable.Method);
        Cursor += 12 + Math.Min(Encoding.UTF8.GetByteCount(callable.Method ?? ""), 256);
    }

    public void WriteSignal(CustomSignal signal)
    {
        Helpers.WriteUInt64(Cursor, signal.TargetId);
        Helpers.WriteString(Cursor + 8, signal.Name);
        Cursor += 12 + Math.Min(Encoding.UTF8.GetByteCount(signal.Name ?? ""), 256);
    }
}
