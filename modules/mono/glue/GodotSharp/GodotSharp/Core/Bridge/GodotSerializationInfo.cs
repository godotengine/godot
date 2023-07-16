using System;
using System.Diagnostics.CodeAnalysis;
using Godot.NativeInterop;

namespace Godot.Bridge;

/// <summary>
/// Serialized information container for handling <see cref="Godot.GodotObject"/> data.
/// </summary>
public sealed class GodotSerializationInfo : IDisposable
{
    private readonly Collections.Dictionary _properties;
    private readonly Collections.Dictionary _signalEvents;

    /// <summary>
    /// Disposes of this <see cref="GodotSerializationInfo"/>.
    /// </summary>
    public void Dispose()
    {
        _properties?.Dispose();
        _signalEvents?.Dispose();

        GC.SuppressFinalize(this);
    }

    private GodotSerializationInfo(in godot_dictionary properties, in godot_dictionary signalEvents)
    {
        _properties = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(properties);
        _signalEvents = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(signalEvents);
    }

    internal static GodotSerializationInfo CreateCopyingBorrowed(
        in godot_dictionary properties, in godot_dictionary signalEvents)
    {
        return new(NativeFuncs.godotsharp_dictionary_new_copy(properties),
            NativeFuncs.godotsharp_dictionary_new_copy(signalEvents));
    }

    /// <summary>
    /// Adds a property to this <see cref="GodotSerializationInfo"/>.
    /// </summary>
    /// <param name="name">The name of the property to add.</param>
    /// <param name="value">The value to set for this property.</param>
    public void AddProperty(StringName name, Variant value)
    {
        _properties[name] = value;
    }

    /// <summary>
    /// Tries to get a property from this <see cref="GodotSerializationInfo"/>.
    /// </summary>
    /// <param name="name">The name of the property to parse.</param>
    /// <param name="value">The value retrieved from this property.</param>
    /// <returns><see langword="true"/> if a property was successfully retrieved;
    /// otherwise, <see langword="false"/>.</returns>
    public bool TryGetProperty(StringName name, out Variant value)
    {
        return _properties.TryGetValue(name, out value);
    }

    /// <summary>
    /// Adds a signal event to this <see cref="GodotSerializationInfo"/>.
    /// </summary>
    /// <param name="name">The name of the signal to add.</param>
    /// <param name="eventDelegate">The delegate to assign to this signal.</param>
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

    /// <summary>
    /// Tries to get a signal from this <see cref="GodotSerializationInfo"/>.
    /// </summary>
    /// <typeparam name="T">The type of <see cref="Delegate"/> being retrieved.</typeparam>
    /// <param name="name">The name of the signal to parse.</param>
    /// <param name="value">The value retrieved from this signal.</param>
    /// <returns><see langword="true"/> if a signal was successfully retrieved;
    /// otherwise, <see langword="false"/>.</returns>
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
