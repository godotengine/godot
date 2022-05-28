using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace Godot.Bridge;

public class GodotSerializationInfo
{
    private readonly Collections.Dictionary<StringName, object> _properties = new();
    private readonly Collections.Dictionary<StringName, Collections.Array> _signalEvents = new();

    internal GodotSerializationInfo()
    {
    }

    internal GodotSerializationInfo(
        Collections.Dictionary<StringName, object> properties,
        Collections.Dictionary<StringName, Collections.Array> signalEvents
    )
    {
        _properties = properties;
        _signalEvents = signalEvents;
    }

    public void AddProperty(StringName name, object value)
    {
        _properties[name] = value;
    }

    public bool TryGetProperty<T>(StringName name, [MaybeNullWhen(false)] out T value)
    {
        return _properties.TryGetValueAsType(name, out value);
    }

    public void AddSignalEventDelegate(StringName name, Delegate eventDelegate)
    {
        var serializedData = new Collections.Array();

        if (DelegateUtils.TrySerializeDelegate(eventDelegate, serializedData))
        {
            _signalEvents[name] = serializedData;
        }
        else if (OS.IsStdoutVerbose())
        {
            Console.WriteLine($"Failed to serialize event signal delegate: {name}");
        }
    }

    public Delegate GetSignalEventDelegate(StringName name)
    {
        if (DelegateUtils.TryDeserializeDelegate(_signalEvents[name], out var eventDelegate))
        {
            return eventDelegate;
        }
        else if (OS.IsStdoutVerbose())
        {
            Console.WriteLine($"Failed to deserialize event signal delegate: {name}");
        }

        return null;
    }

    public IEnumerable<StringName> GetSignalEventsList()
    {
        return _signalEvents.Keys;
    }
}
