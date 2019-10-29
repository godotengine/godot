using System;
using System.Net.Sockets;
using System.Threading.Tasks;

namespace GodotTools.IdeConnection
{
    public class GodotIdeConnectionServer : GodotIdeConnection
    {
        public GodotIdeConnectionServer(TcpClient tcpClient, Func<Message, bool> messageHandler)
            : base(tcpClient, messageHandler)
        {
        }

        protected override bool WriteHandshake()
        {
            return WriteLine(ServerHandshake);
        }

        protected override bool IsValidResponseHandshake(string handshakeLine)
        {
            return handshakeLine == ClientHandshake;
        }
    }
}
