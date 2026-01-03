using System.Collections.Generic;
using System.Threading.Tasks;
using GodotTools.IdeMessaging.Requests;
using Newtonsoft.Json;

namespace GodotTools.IdeMessaging
{
    // ReSharper disable once UnusedType.Global
    public abstract class ClientMessageHandler : IMessageHandler
    {
        private readonly Dictionary<string, Peer.RequestHandler> requestHandlers;

        protected ClientMessageHandler()
        {
            requestHandlers = InitializeRequestHandlers();
        }

        public async Task<MessageContent> HandleRequest(Peer peer, string id, MessageContent content, ILogger logger)
        {
            if (!requestHandlers.TryGetValue(id, out var handler))
            {
                logger.LogError($"Received unknown request: {id}");
                return new MessageContent(MessageStatus.RequestNotSupported, "null");
            }

            try
            {
                var response = await handler(peer, content);
                return new MessageContent(response.Status, JsonConvert.SerializeObject(response));
            }
            catch (JsonException)
            {
                logger.LogError($"Received request with invalid body: {id}");
                return new MessageContent(MessageStatus.InvalidRequestBody, "null");
            }
        }

        private Dictionary<string, Peer.RequestHandler> InitializeRequestHandlers()
        {
            return new Dictionary<string, Peer.RequestHandler>
            {
                [OpenFileRequest.Id] = async (peer, content) =>
                {
                    var request = JsonConvert.DeserializeObject<OpenFileRequest>(content.Body);
                    return await HandleOpenFile(request!);
                }
            };
        }

        protected abstract Task<Response> HandleOpenFile(OpenFileRequest request);
    }
}
