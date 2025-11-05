using System.Diagnostics.CodeAnalysis;

namespace GodotTools.IdeMessaging
{
    public interface IHandshake
    {
        public string GetHandshakeLine(string identity);
        public bool IsValidPeerHandshake(string handshake, [NotNullWhen(true)] out string? identity, ILogger logger);
    }
}
