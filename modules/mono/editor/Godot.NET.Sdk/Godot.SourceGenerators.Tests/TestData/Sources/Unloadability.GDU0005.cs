using Godot;

// Stubs for Newtonsoft.Json (not available in test reference assemblies)
namespace Newtonsoft.Json
{
    public static class JsonConvert
    {
        public static string SerializeObject(object value) => "";
        public static T DeserializeObject<T>(string value) => default;
    }
}

public class MyData
{
    public int Value { get; set; }
}

// Positive: [Tool] class calling JsonConvert.SerializeObject triggers GDU0005
[Tool]
public class ToolClassWithNewtonsoft
{
    public void Serialize()
    {
        var json = {|GDU0005:Newtonsoft.Json.JsonConvert.SerializeObject(new MyData())|};
    }

    public void Deserialize()
    {
        var obj = {|GDU0005:Newtonsoft.Json.JsonConvert.DeserializeObject<MyData>("{}")|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithNewtonsoft
{
    public void Serialize()
    {
        var json = Newtonsoft.Json.JsonConvert.SerializeObject(new MyData());
    }
}
