using System.Threading.Tasks;

namespace GodotTools.IdeMessaging
{
    public interface IMessageHandler
    {
        public Task<MessageContent> HandleRequest(Peer peer, string id, MessageContent content, ILogger logger);
    }
}
