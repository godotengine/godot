using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GodotTools.IdeMessaging.Utils;

namespace GodotTools.IdeMessaging.CLI
{
    public class ForwarderMessageHandler : IMessageHandler
    {
        private readonly StreamWriter outputWriter;
        private readonly SemaphoreSlim outputWriteSem = new SemaphoreSlim(1);

        public ForwarderMessageHandler(StreamWriter outputWriter)
        {
            this.outputWriter = outputWriter;
        }

        public async Task<MessageContent> HandleRequest(Peer peer, string id, MessageContent content, ILogger logger)
        {
            await WriteRequestToOutput(id, content);
            return new MessageContent(MessageStatus.RequestNotSupported, "null");
        }

        private async Task WriteRequestToOutput(string id, MessageContent content)
        {
            using (await outputWriteSem.UseAsync())
            {
                await outputWriter.WriteLineAsync("======= Request =======");
                await outputWriter.WriteLineAsync(id);
                await outputWriter.WriteLineAsync(content.Body.Count(c => c == '\n').ToString(CultureInfo.InvariantCulture));
                await outputWriter.WriteLineAsync(content.Body);
                await outputWriter.WriteLineAsync("=======================");
                await outputWriter.FlushAsync();
            }
        }

        public async Task WriteResponseToOutput(string id, MessageContent content)
        {
            using (await outputWriteSem.UseAsync())
            {
                await outputWriter.WriteLineAsync("======= Response =======");
                await outputWriter.WriteLineAsync(id);
                await outputWriter.WriteLineAsync(content.Body.Count(c => c == '\n').ToString(CultureInfo.InvariantCulture));
                await outputWriter.WriteLineAsync(content.Body);
                await outputWriter.WriteLineAsync("========================");
                await outputWriter.FlushAsync();
            }
        }

        public async Task WriteLineToOutput(string eventName)
        {
            using (await outputWriteSem.UseAsync())
                await outputWriter.WriteLineAsync($"======= {eventName} =======");
        }
    }
}
