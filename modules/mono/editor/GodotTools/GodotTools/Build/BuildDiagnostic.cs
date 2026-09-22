namespace GodotTools.Build
{
    public class BuildDiagnostic
    {
        public enum DiagnosticType
        {
            Hidden,
            Info,
            Warning,
            Error,
        }

        public DiagnosticType Type { get; set; }
        public string? File { get; set; }
        public int Line { get; set; }
        public int Column { get; set; }
        public string? Code { get; set; }
        public string Message { get; set; } = "";
        public string? ProjectFile { get; set; }
    }
}
