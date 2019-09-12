using System;
using System.Net.Sockets;
using System.Threading.Tasks;

namespace GodotTools.IdeConnection
{
    public class GodotIdeConnectionClient : GodotIdeConnection
    {
        public GodotIdeConnectionClient(TcpClient tcpClient, Func<Message, bool> messageHandler)
            : base(tcpClient, messageHandler)
        {
        }

        protected override bool WriteHandshake()
        {
            return WriteLine(ClientHandshake);
        }

        protected override bool IsValidResponseHandshake(string handshakeLine)
        {
            return handshakeLine == ServerHandshake;
        }
    }
}
