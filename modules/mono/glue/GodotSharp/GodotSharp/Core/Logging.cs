
using System;

#nullable enable

namespace Godot;

/// <summary>
/// Functionality to intercept and handle engine log messages.
/// </summary>
public static class Logging
{
    public static event Action<Exception>? UnhandledExceptionReporter;

    internal static void InvokeUnhandledExceptionReporter(Exception e)
    {
        UnhandledExceptionReporter?.Invoke(e);
    }
}
