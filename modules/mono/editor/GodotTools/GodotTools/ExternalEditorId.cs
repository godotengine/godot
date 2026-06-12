namespace GodotTools
{
    // Keep in sync with editor/settings/editor_settings.cpp.
    public enum ExternalEditorId : long
    {
        None,
        VisualStudio, // Windows-only
        VisualStudioForMac, // Mac-only
        MonoDevelop,
        VsCode,
        Rider,
        CustomEditor,
        Fleet,
    }
}
