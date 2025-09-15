using System;

namespace Godot;

/// <summary>
/// Functionality to intercept and handle engine log messages.
/// </summary>
public static class Logging
{
    public static event Action<Exception> UnhandledExceptionReporter;
    public static event Action<Exception> ExceptionReporter;

    internal static void InvokeUnhandledExceptionReporter(Exception e)
    {
        UnhandledExceptionReporter?.Invoke(e);
    }

    internal static void InvokeExceptionReporter(Exception e)
    {
        ExceptionReporter?.Invoke(e);
    }
}
