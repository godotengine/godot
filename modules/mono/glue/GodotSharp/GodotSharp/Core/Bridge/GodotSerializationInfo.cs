using System;
using System.Diagnostics.CodeAnalysis;
using Godot.NativeInterop;
using JetBrains.Annotations;

namespace Godot.Bridge;

public sealed class GodotSerializationInfo : IDisposable
{
    private readonly Collections.Dictionary _properties;
    private readonly Collections.Dictionary _signalEvents;

    public void Dispose()
    {
        _properties?.Dispose();
        _signalEvents?.Dispose();
    }

    private GodotSerializationInfo(in godot_dictionary properties, in godot_dictionary signalEvents)
    {
        _properties = Collections.Dictionary.CreateConsuming(properties);
        _signalEvents = Collections.Dictionary.CreateConsuming(signalEvents);
    }

    internal static GodotSerializationInfo CreateCopyingBorrowed(
        in godot_dictionary properties, in godot_dictionary signalEvents)
    {
        return new(NativeFuncs.godotsharp_dictionary_new_copy(properties),
            NativeFuncs.godotsharp_dictionary_new_copy(signalEvents));
    }

    [PublicAPI("Source generators depend on this for serialization.")]
    public void AddProperty(StringName name, Variant value)
    {
        _properties[name] = value;
    }

    [PublicAPI("Source generators depend on this for deserialization.")]
    public bool TryGetProperty(StringName name, out Variant value)
    {
        return _properties.TryGetValue(name, out value);
    }

    // NOTE: Currently, this doesn't actually require unreferenced code,
    //       but we annotate it preemptively in case that changes in the future.
    [RequiresUnreferencedCode(
        "This method might not be compatible with trimming in the future. "
        + "It's for use within 'GodotObject.SaveGodotObjectData' only, "
        + "which itself is meant for use by the Godot editor only.")]
    [PublicAPI("Source generators depend on this for serialization.")]
    public void AddSignalEventDelegate(StringName name, Delegate eventDelegate)
    {
        var serializedData = new Collections.Array();

        if (DelegateUtils.TrySerializeDelegate(eventDelegate, serializedData))
        {
            _signalEvents[name] = serializedData;
        }
        else if (OS.IsStdOutVerbose())
        {
            Console.WriteLine($"Failed to serialize event signal delegate: {name}");
        }
    }

    [RequiresUnreferencedCode(
        "This method for use within 'GodotObject.RestoreGodotObjectData' only, which itself "
        + "is meant for use by the Godot editor only. It uses dynamic reflection to deserialize "
        + "types and to search for methods, which is not compatible with trimming.")]
    [RequiresDynamicCode(
        "This method for use within 'GodotObject.RestoreGodotObjectData' only, which itself "
        + "is meant for use by the Godot editor only. It uses MakeGenericType to instantiate "
        + "generic types from the method signature and target object, and the native code for this"
        + "instantiation might not be available at runtime.")]
    [PublicAPI("Source generators depend on this for deserialization.")]
    public bool TryGetSignalEventDelegate<T>(StringName name, [MaybeNullWhen(false)] out T value)
        where T : Delegate
    {
        if (_signalEvents.TryGetValue(name, out Variant serializedData))
        {
            if (DelegateUtils.TryDeserializeDelegate(serializedData.AsGodotArray(), out var eventDelegate))
            {
                value = eventDelegate as T;

                if (value == null)
                {
                    Console.WriteLine($"Cannot cast the deserialized event signal delegate: {name}. " +
                                      $"Expected '{typeof(T).FullName}'; got '{eventDelegate.GetType().FullName}'.");
                    return false;
                }

                return true;
            }
            else if (OS.IsStdOutVerbose())
            {
                Console.WriteLine($"Failed to deserialize event signal delegate: {name}");
            }

            value = null;
            return false;
        }

        value = null;
        return false;
    }
}
