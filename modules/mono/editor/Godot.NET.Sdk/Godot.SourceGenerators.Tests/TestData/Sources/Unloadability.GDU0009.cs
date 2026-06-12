using System.Text;
using Godot;

public class MyEncodingProvider : EncodingProvider
{
    public override Encoding GetEncoding(int codepage) => null;
    public override Encoding GetEncoding(string name) => null;
}

// Positive: [Tool] class calling Encoding.RegisterProvider with a user-defined provider triggers GDU0009
[Tool]
public class ToolClassWithEncodingProvider
{
    public void Register()
    {
        {|GDU0009:Encoding.RegisterProvider(new MyEncodingProvider())|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithEncodingProvider
{
    public void Register()
    {
        Encoding.RegisterProvider(new MyEncodingProvider());
    }
}
