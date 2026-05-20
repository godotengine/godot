using System;
using System.IO;
namespace SevenZip
{
	using CommandLineParser;
	
	public class CDoubleStream: Stream
	{
		public System.IO.Stream s1;
		public System.IO.Stream s2;
		public int fileIndex;
		public long skipSize;
		
		public override bool CanRead { get { return true; }}
		public override bool CanWrite { get { return false; }}
		public override bool CanSeek { get { return false; }}
		public override long Length { get { return s1.Length + s2.Length - skipSize; } }
		public override long Position
		{
			get { return 0;	}
			set { }
		}
		public override void Flush() { }
		public override int Read(byte[] buffer, int offset, int count) 
		{
			int numTotal = 0;
			while (count > 0)
			{
				if (fileIndex == 0)
				{
					int num = s1.Read(buffer, offset, count);
					offset += num;
					count -= num;
					numTotal += num;
					if (num == 0)
						fileIndex++;
				}
				if (fileIndex == 1)
				{
					numTotal += s2.Read(buffer, offset, count);
					return numTotal;
				}
			}
			return numTotal;
		}
		public override void Write(byte[] buffer, int offset, int count)
		{
			throw (new Exception("can't Write"));
		}
		public override long Seek(long offset, System.IO.SeekOrigin origin)
		{
			throw (new Exception("can't Seek"));
		}
		public override void SetLength(long value)
		{
			throw (new Exception("can't SetLength"));
		}
	}
	
	class LzmaAlone
	{
		enum Key
		{
			Help1 = 0,
			Help2,
			Mode,
			Dictionary,
			FastBytes,
			LitContext,
			LitPos,
			PosBits,
			MatchFinder,
			EOS,
			StdIn,
			StdOut,
			Train
		};

		static void PrintHelp()
		{
			System.Console.WriteLine("\nUsage:  LZMA <e|d> [<switches>...] inputFile outputFile\n" +
				"  e: encode file\n" +
				"  d: decode file\n" +
				"  b: Benchmark\n" +
				"<Switches>\n" +
				// "  -a{N}:  set compression mode - [0, 1], default: 1 (max)\n" +
				"  -d{N}:  set dictionary - [0, 29], default: 23 (8MB)\n" +
				"  -fb{N}: set number of fast bytes - [5, 273], default: 128\n" +
				"  -lc{N}: set number of literal context bits - [0, 8], default: 3\n" +
				"  -lp{N}: set number of literal pos bits - [0, 4], default: 0\n" +
				"  -pb{N}: set number of pos bits - [0, 4], default: 2\n" +
				"  -mf{MF_ID}: set Match Finder: [bt2, bt4], default: bt4\n" +
				"  -eos:   write End Of Stream marker\n"
				// + "  -si:    read data from stdin\n"
				// + "  -so:    write data to stdout\n"
				);
		}

		static bool GetNumber(string s, out Int32 v)
		{
			v = 0;
			for (int i = 0; i < s.Length; i++)
			{
				char c = s[i];
				if (c < '0' || c > '9')
					return false;
				v *= 10;
				v += (Int32)(c - '0');
			}
			return true;
		}

		static int IncorrectCommand()
		{
			throw (new Exception("Command line error"));
			// System.Console.WriteLine("\nCommand line error\n");
			// return 1;
		}
		static int Main2(string[] args)
		{
			System.Console.WriteLine("\nLZMA# 4.61  2008-11-23\n");

			if (args.Length == 0)
			{
				PrintHelp();
				return 0;
			}

			SwitchForm[] kSwitchForms = new SwitchForm[13];
			int sw = 0;
			kSwitchForms[sw++] = new SwitchForm("?", SwitchType.Simple, false);
			kSwitchForms[sw++] = new SwitchForm("H", SwitchType.Simple, false);
			kSwitchForms[sw++] = new SwitchForm("A", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("D", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("FB", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("LC", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("LP", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("PB", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("MF", SwitchType.UnLimitedPostString, false, 1);
			kSwitchForms[sw++] = new SwitchForm("EOS", SwitchType.Simple, false);
			kSwitchForms[sw++] = new SwitchForm("SI", SwitchType.Simple, false);
			kSwitchForms[sw++] = new SwitchForm("SO", SwitchType.Simple, false);
			kSwitchForms[sw++] = new SwitchForm("T", SwitchType.UnLimitedPostString, false, 1);


			Parser parser = new Parser(sw);
			try
			{
				parser.ParseStrings(kSwitchForms, args);
			}
			catch
			{
				return IncorrectCommand();
			}

			if (parser[(int)Key.Help1].ThereIs || parser[(int)Key.Help2].ThereIs)
			{
				PrintHelp();
				return 0;
			}

			System.Collections.ArrayList nonSwitchStrings = parser.NonSwitchStrings;

			int paramIndex = 0;
			if (paramIndex >= nonSwitchStrings.Count)
				return IncorrectCommand();
			string command = (string)nonSwitchStrings[paramIndex++];
			command = command.ToLower();

			bool dictionaryIsDefined = false;
			Int32 dictionary = 1 << 21;
			if (parser[(int)Key.Dictionary].ThereIs)
			{
				Int32 dicLog;
				if (!GetNumber((string)parser[(int)Key.Dictionary].PostStrings[0], out dicLog))
					IncorrectCommand();
				dictionary = (Int32)1 << dicLog;
				dictionaryIsDefined = true;
			}
			string mf = "bt4";
			if (parser[(int)Key.MatchFinder].ThereIs)
				mf = (string)parser[(int)Key.MatchFinder].PostStrings[0];
			mf = mf.ToLower();

			if (command == "b")
			{
				const Int32 kNumDefaultItereations = 10;
				Int32 numIterations = kNumDefaultItereations;
				if (paramIndex < nonSwitchStrings.Count)
					if (!GetNumber((string)nonSwitchStrings[paramIndex++], out numIterations))
						numIterations = kNumDefaultItereations;
				return LzmaBench.LzmaBenchmark(numIterations, (UInt32)dictionary);
			}

			string train = "";
			if (parser[(int)Key.Train].ThereIs)
				train = (string)parser[(int)Key.Train].PostStrings[0];

			bool encodeMode = false;
			if (command == "e")
				encodeMode = true;
			else if (command == "d")
				encodeMode = false;
			else
				IncorrectCommand();

			bool stdInMode = parser[(int)Key.StdIn].ThereIs;
			bool stdOutMode = parser[(int)Key.StdOut].ThereIs;

			Stream inStream = null;
			if (stdInMode)
			{
				throw (new Exception("Not implemeted"));
			}
			else
			{
				if (paramIndex >= nonSwitchStrings.Count)
					IncorrectCommand();
				string inputName = (string)nonSwitchStrings[paramIndex++];
				inStream = new FileStream(inputName, FileMode.Open, FileAccess.Read);
			}

			FileStream outStream = null;
			if (stdOutMode)
			{
				throw (new Exception("Not implemeted"));
			}
			else
			{
				if (paramIndex >= nonSwitchStrings.Count)
					IncorrectCommand();
				string outputName = (string)nonSwitchStrings[paramIndex++];
				outStream = new FileStream(outputName, FileMode.Create, FileAccess.Write);
			}

			FileStream trainStream = null;
			if (train.Length != 0)
				trainStream = new FileStream(train, FileMode.Open, FileAccess.Read);

			if (encodeMode)
			{
				if (!dictionaryIsDefined)
					dictionary = 1 << 23;

				Int32 posStateBits = 2;
				Int32 litContextBits = 3; // for normal files
				// UInt32 litContextBits = 0; // for 32-bit data
				Int32 litPosBits = 0;
				// UInt32 litPosBits = 2; // for 32-bit data
				Int32 algorithm = 2;
				Int32 numFastBytes = 128;

				bool eos = parser[(int)Key.EOS].ThereIs || stdInMode;

				if (parser[(int)Key.Mode].ThereIs)
					if (!GetNumber((string)parser[(int)Key.Mode].PostStrings[0], out algorithm))
						IncorrectCommand();

				if (parser[(int)Key.FastBytes].ThereIs)
					if (!GetNumber((string)parser[(int)Key.FastBytes].PostStrings[0], out numFastBytes))
						IncorrectCommand();
				if (parser[(int)Key.LitContext].ThereIs)
					if (!GetNumber((string)parser[(int)Key.LitContext].PostStrings[0], out litContextBits))
						IncorrectCommand();
				if (parser[(int)Key.LitPos].ThereIs)
					if (!GetNumber((string)parser[(int)Key.LitPos].PostStrings[0], out litPosBits))
						IncorrectCommand();
				if (parser[(int)Key.PosBits].ThereIs)
					if (!GetNumber((string)parser[(int)Key.PosBits].PostStrings[0], out posStateBits))
						IncorrectCommand();

				CoderPropID[] propIDs = 
				{
					CoderPropID.DictionarySize,
					CoderPropID.PosStateBits,
					CoderPropID.LitContextBits,
					CoderPropID.LitPosBits,
					CoderPropID.Algorithm,
					CoderPropID.NumFastBytes,
					CoderPropID.MatchFinder,
					CoderPropID.EndMarker
				};
				object[] properties = 
				{
					(Int32)(dictionary),
					(Int32)(posStateBits),
					(Int32)(litContextBits),
					(Int32)(litPosBits),
					(Int32)(algorithm),
					(Int32)(numFastBytes),
					mf,
					eos
				};

				Compression.LZMA.Encoder encoder = new Compression.LZMA.Encoder();
				encoder.SetCoderProperties(propIDs, properties);
				encoder.WriteCoderProperties(outStream);
				Int64 fileSize;
				if (eos || stdInMode)
					fileSize = -1;
				else
					fileSize = inStream.Length;
				for (int i = 0; i < 8; i++)
					outStream.WriteByte((Byte)(fileSize >> (8 * i)));
				if (trainStream != null)
				{
					CDoubleStream doubleStream = new CDoubleStream();
					doubleStream.s1 = trainStream;
					doubleStream.s2 = inStream;
					doubleStream.fileIndex = 0;
					inStream = doubleStream;
					long trainFileSize = trainStream.Length;
					doubleStream.skipSize = 0;
					if (trainFileSize > dictionary)
						doubleStream.skipSize = trainFileSize - dictionary;
					trainStream.Seek(doubleStream.skipSize, SeekOrigin.Begin);
					encoder.SetTrainSize((uint)(trainFileSize - doubleStream.skipSize));
				}
				encoder.Code(inStream, outStream, -1, -1, null);
			}
			else if (command == "d")
			{
				byte[] properties = new byte[5];
				if (inStream.Read(properties, 0, 5) != 5)
					throw (new Exception("input .lzma is too short"));
				Compression.LZMA.Decoder decoder = new Compression.LZMA.Decoder();
				decoder.SetDecoderProperties(properties);
				if (trainStream != null)
				{
					if (!decoder.Train(trainStream))
						throw (new Exception("can't train"));
				}
				long outSize = 0;
				for (int i = 0; i < 8; i++)
				{
					int v = inStream.ReadByte();
					if (v < 0)
						throw (new Exception("Can't Read 1"));
					outSize |= ((long)(byte)v) << (8 * i);
				}
				long compressedSize = inStream.Length - inStream.Position;
				decoder.Code(inStream, outStream, compressedSize, outSize, null);
			}
			else
				throw (new Exception("Command Error"));
			return 0;
		}

		[STAThread]
		static int Main(string[] args)
		{
			try
			{
				return Main2(args);
			}
			catch (Exception e)
			{
				Console.WriteLine("{0} Caught exception #1.", e);
				// throw e;
				return 1;
			}
		}
	}
}
