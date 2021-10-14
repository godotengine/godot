using System.Diagnostics;

namespace Godot
{
    internal class GodotTraceListener : TraceListener
    {
        public override void Write(string message)
        {
            GD.PrintRaw(message);
        }

        public override void WriteLine(string message)
        {
            GD.Print(message);
        }

        public override void Fail(string message, string detailMessage)
        {
            GD.PrintErr("Assertion failed: ", message);
            if (detailMessage != null)
            {
                GD.PrintErr("  Details: ", detailMessage);
            }

            try
            {
                string stackTrace = new StackTrace(true).ToString();
                GD.PrintErr(stackTrace);
            }
            catch
            {
                // ignored
            }
        }
    }
}
