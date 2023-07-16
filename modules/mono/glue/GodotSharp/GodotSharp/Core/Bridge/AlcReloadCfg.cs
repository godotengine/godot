namespace Godot.Bridge;

/// <summary>
/// A reload configuration for an assembly load context.
/// </summary>
public static class AlcReloadCfg
{
    private static bool _configured = false;

    /// <summary>
    /// Configure this assembly load context.
    /// </summary>
    /// <param name="alcReloadEnabled">Specifies if a reload is enabled.</param>
    public static void Configure(bool alcReloadEnabled)
    {
        if (_configured)
            return;

        _configured = true;

        IsAlcReloadingEnabled = alcReloadEnabled;
    }

    internal static bool IsAlcReloadingEnabled = false;
}
