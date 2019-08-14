using System.Linq;
using System.Text;

namespace GodotTools.IdeConnection
{
    public class MessageComposer
    {
        private readonly StringBuilder stringBuilder = new StringBuilder();

        private static readonly char[] CharsToEscape = { '\\', '"' };

        public void AddArgument(string argument)
        {
            AddArgument(argument, quoted: argument.Contains(","));
        }

        public void AddArgument(string argument, bool quoted)
        {
            if (stringBuilder.Length > 0)
                stringBuilder.Append(',');

            if (quoted)
            {
                stringBuilder.Append('"');
                
                foreach (char @char in argument)
                {
                    if (CharsToEscape.Contains(@char))
                        stringBuilder.Append('\\');
                    stringBuilder.Append(@char);
                }
                
                stringBuilder.Append('"');
            }
            else
            {
                stringBuilder.Append(argument);
            }
        }

        public override string ToString()
        {
            return stringBuilder.ToString();
        }
    }
}
