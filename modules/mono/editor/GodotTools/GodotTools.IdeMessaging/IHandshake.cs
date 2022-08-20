namespace GodotTools.IdeMessaging
{
    public interface IHandshake
    {
        string GetHandshakeLine(string identity);
        bool IsValidPeerHandshake(string handshake, out string identity, ILogger logger);
    }
}
