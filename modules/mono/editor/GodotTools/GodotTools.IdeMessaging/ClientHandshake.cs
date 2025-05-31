using System.Diagnostics.CodeAnalysis;
using System.Text.RegularExpressions;

namespace GodotTools.IdeMessaging
{
    public class ClientHandshake : IHandshake
    {
        private static readonly string ClientHandshakeBase = $"{Peer.ClientHandshakeName},Version={Peer.ProtocolVersionMajor}.{Peer.ProtocolVersionMinor}.{Peer.ProtocolVersionRevision}";
        private static readonly string ServerHandshakePattern = $@"{Regex.Escape(Peer.ServerHandshakeName)},Version=([0-9]+)\.([0-9]+)\.([0-9]+),([_a-zA-Z][_a-zA-Z0-9]{{0,63}})";

        public string GetHandshakeLine(string identity) => $"{ClientHandshakeBase},{identity}";

        public bool IsValidPeerHandshake(string handshake, [NotNullWhen(true)] out string? identity, ILogger logger)
        {
            identity = null;

            var match = Regex.Match(handshake, ServerHandshakePattern);

            if (!match.Success)
                return false;

            if (!uint.TryParse(match.Groups[1].Value, out uint serverMajor) || Peer.ProtocolVersionMajor != serverMajor)
            {
                logger.LogDebug("Incompatible major version: " + match.Groups[1].Value);
                return false;
            }

            if (!uint.TryParse(match.Groups[2].Value, out uint serverMinor) || Peer.ProtocolVersionMinor < serverMinor)
            {
                logger.LogDebug("Incompatible minor version: " + match.Groups[2].Value);
                return false;
            }

            if (!uint.TryParse(match.Groups[3].Value, out uint _)) // Revision
            {
                logger.LogDebug("Incompatible revision build: " + match.Groups[3].Value);
                return false;
            }

            identity = match.Groups[4].Value;

            return true;
        }
    }
}
