using System;
using Godot;
using JetBrains.Rider.PathLocator;
using Newtonsoft.Json;
using OS = GodotTools.Utils.OS;

namespace GodotTools.Ides.Rider;

public class RiderLocatorEnvironment : IRiderLocatorEnvironment
{
    public JetBrains.Rider.PathLocator.OS CurrentOS
    {
        get
        {
            if (OS.IsWindows)
                return JetBrains.Rider.PathLocator.OS.Windows;
            if (OS.IsMacOS) return JetBrains.Rider.PathLocator.OS.MacOSX;
            if (OS.IsUnixLike) return JetBrains.Rider.PathLocator.OS.Linux;
            return JetBrains.Rider.PathLocator.OS.Other;
        }
    }

    public T FromJson<T>(string json)
    {
        return JsonConvert.DeserializeObject<T>(json);
    }

    public void Info(string message, Exception e = null)
    {
        if (e == null)
            GD.Print(message);
        else
            GD.Print(message, e);
    }

    public void Warn(string message, Exception e = null)
    {
        if (e == null)
            GD.PushWarning(message);
        else
            GD.PushWarning(message, e);
    }

    public void Error(string message, Exception e = null)
    {
        if (e == null)
            GD.PushError(message);
        else
            GD.PushError(message, e);
    }

    public void Verbose(string message, Exception e = null)
    {
        // do nothing, since IDK how to write only to the log, without spamming the output
    }
}
