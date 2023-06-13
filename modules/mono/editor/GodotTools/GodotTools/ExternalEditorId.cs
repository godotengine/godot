namespace GodotTools
{
    public enum ExternalEditorId : long
    {
        None,
        VisualStudio, // TODO (Windows-only)
        VisualStudioForMac, // Mac-only
        MonoDevelop,
        VsCode,
        Rider,
        CustomEditor
    }
}
