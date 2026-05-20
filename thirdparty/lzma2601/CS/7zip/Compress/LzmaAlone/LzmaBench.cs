// LzmaBench.cs

using System;
using System.IO;

namespace SevenZip
{
	/// <summary>
	/// LZMA Benchmark
	/// </summary>
	internal abstract class LzmaBench
	{
		const UInt32 kAdditionalSize = (6 << 20);
		const UInt32 kCompressedAdditionalSize = (1 << 10);
		const UInt32 kMaxLzmaPropSize = 10;

		class CRandomGenerator
		{
			UInt32 A1;
			UInt32 A2;
			public CRandomGenerator() { Init(); }
			public void Init() { A1 = 362436069; A2 = 521288629; }
			public UInt32 GetRnd()
			{
				return
					((A1 = 36969 * (A1 & 0xffff) + (A1 >> 16)) << 16) ^
					((A2 = 18000 * (A2 & 0xffff) + (A2 >> 16)));
			}
		};

		class CBitRandomGenerator
		{
			CRandomGenerator RG = new CRandomGenerator();
			UInt32 Value;
			int NumBits;
			public void Init()
			{
				Value = 0;
				NumBits = 0;
			}
			public UInt32 GetRnd(int numBits)
			{
				UInt32 result;
				if (NumBits > numBits)
				{
					result = Value & (((UInt32)1 << numBits) - 1);
					Value >>= numBits;
					NumBits -= numBits;
					return result;
				}
				numBits -= NumBits;
				result = (Value << numBits);
				Value = RG.GetRnd();
				result |= Value & (((UInt32)1 << numBits) - 1);
				Value >>= numBits;
				NumBits = 32 - numBits;
				return result;
			}
		};

		class CBenchRandomGenerator
		{
			CBitRandomGenerator RG = new CBitRandomGenerator();
			UInt32 Pos;
			UInt32 Rep0;
			
			public UInt32 BufferSize;
			public Byte[] Buffer = null;

			public CBenchRandomGenerator() { }

			public void Set(UInt32 bufferSize)
			{
				Buffer = new Byte[bufferSize];
				Pos = 0;
				BufferSize = bufferSize;
			}
			UInt32 GetRndBit() { return RG.GetRnd(1); }
			UInt32 GetLogRandBits(int numBits)
			{
				UInt32 len = RG.GetRnd(numBits);
				return RG.GetRnd((int)len);
			}
			UInt32 GetOffset()
			{
				if (GetRndBit() == 0)
					return GetLogRandBits(4);
				return (GetLogRandBits(4) << 10) | RG.GetRnd(10);
			}
			UInt32 GetLen1() { return RG.GetRnd(1 + (int)RG.GetRnd(2)); }
			UInt32 GetLen2() { return RG.GetRnd(2 + (int)RG.GetRnd(2)); }
			public void Generate()
			{
				RG.Init();
				Rep0 = 1;
				while (Pos < BufferSize)
				{
					if (GetRndBit() == 0 || Pos < 1)
						Buffer[Pos++] = (Byte)RG.GetRnd(8);
					else
					{
						UInt32 len;
						if (RG.GetRnd(3) == 0)
							len = 1 + GetLen1();
						else
						{
							do
								Rep0 = GetOffset();
							while (Rep0 >= Pos);
							Rep0++;
							len = 2 + GetLen2();
						}
						for (UInt32 i = 0; i < len && Pos < BufferSize; i++, Pos++)
							Buffer[Pos] = Buffer[Pos - Rep0];
					}
				}
			}
		};

		class CrcOutStream : System.IO.Stream
		{
			public CRC CRC = new CRC();
			public void Init() { CRC.Init(); }
			public UInt32 GetDigest() { return CRC.GetDigest(); }

			public override bool CanRead { get { return false; } }
			public override bool CanSeek { get { return false; } }
			public override bool CanWrite { get { return true; } }
			public override Int64 Length { get { return 0; } }
			public override Int64 Position { get { return 0; } set { } }
			public override void Flush() { }
			public override long Seek(long offset, SeekOrigin origin) { return 0; }
			public override void SetLength(long value) { }
			public override int Read(byte[] buffer, int offset, int count) { return 0; }

			public override void WriteByte(byte b)
			{
				CRC.UpdateByte(b);
			}
			public override void Write(byte[] buffer, int offset, int count)
			{
				CRC.Update(buffer, (uint)offset, (uint)count);
			}
		};

		class CProgressInfo : ICodeProgress
		{
			public Int64 ApprovedStart;
			public Int64 InSize;
			public System.DateTime Time;
			public void Init() { InSize = 0; }
			public void SetProgress(Int64 inSize, Int64 outSize)
			{
				if (inSize >= ApprovedStart && InSize == 0)
				{
					Time = DateTime.UtcNow;
					InSize = inSize;
				}
			}
		}
		const int kSubBits = 8;

		static UInt32 GetLogSize(UInt32 size)
		{
			for (int i = kSubBits; i < 32; i++)
				for (UInt32 j = 0; j < (1 << kSubBits); j++)
					if (size <= (((UInt32)1) << i) + (j << (i - kSubBits)))
						return (UInt32)(i << kSubBits) + j;
			return (32 << kSubBits);
		}

		static UInt64 MyMultDiv64(UInt64 value, UInt64 elapsedTime)
		{
			UInt64 freq = TimeSpan.TicksPerSecond;
			UInt64 elTime = elapsedTime;
			while (freq > 1000000)
			{
				freq >>= 1;
				elTime >>= 1;
			}
			if (elTime == 0)
				elTime = 1;
			return value * freq / elTime;
		}

		static UInt64 GetCompressRating(UInt32 dictionarySize, UInt64 elapsedTime, UInt64 size)
		{
			UInt64 t = GetLogSize(dictionarySize) - (18 << kSubBits);
			UInt64 numCommandsForOne = 1060 + ((t * t * 10) >> (2 * kSubBits));
			UInt64 numCommands = (UInt64)(size) * numCommandsForOne;
			return MyMultDiv64(numCommands, elapsedTime);
		}

		static UInt64 GetDecompressRating(UInt64 elapsedTime, UInt64 outSize, UInt64 inSize)
		{
			UInt64 numCommands = inSize * 220 + outSize * 20;
			return MyMultDiv64(numCommands, elapsedTime);
		}

		static UInt64 GetTotalRating(
			UInt32 dictionarySize,
			UInt64 elapsedTimeEn, UInt64 sizeEn,
			UInt64 elapsedTimeDe,
			UInt64 inSizeDe, UInt64 outSizeDe)
		{
			return (GetCompressRating(dictionarySize, elapsedTimeEn, sizeEn) +
				GetDecompressRating(elapsedTimeDe, inSizeDe, outSizeDe)) / 2;
		}

		static void PrintValue(UInt64 v)
		{
			string s = v.ToString();
			for (int i = 0; i + s.Length < 6; i++)
				System.Console.Write(" ");
			System.Console.Write(s);
		}

		static void PrintRating(UInt64 rating)
		{
			PrintValue(rating / 1000000);
			System.Console.Write(" MIPS");
		}

		static void PrintResults(
			UInt32 dictionarySize,
			UInt64 elapsedTime,
			UInt64 size,
			bool decompressMode, UInt64 secondSize)
		{
			UInt64 speed = MyMultDiv64(size, elapsedTime);
			PrintValue(speed / 1024);
			System.Console.Write(" KB/s  ");
			UInt64 rating;
			if (decompressMode)
				rating = GetDecompressRating(elapsedTime, size, secondSize);
			else
				rating = GetCompressRating(dictionarySize, elapsedTime, size);
			PrintRating(rating);
		}

		static public int LzmaBenchmark(Int32 numIterations, UInt32 dictionarySize)
		{
			if (numIterations <= 0)
				return 0;
			if (dictionarySize < (1 << 18))
			{
				System.Console.WriteLine("\nError: dictionary size for benchmark must be >= 19 (512 KB)");
				return 1;
			}
			System.Console.Write("\n       Compressing                Decompressing\n\n");

			Compression.LZMA.Encoder encoder = new Compression.LZMA.Encoder();
			Compression.LZMA.Decoder decoder = new Compression.LZMA.Decoder();


			CoderPropID[] propIDs = 
			{ 
				CoderPropID.DictionarySize,
			};
			object[] properties = 
			{
				(Int32)(dictionarySize),
			};

			UInt32 kBufferSize = dictionarySize + kAdditionalSize;
			UInt32 kCompressedBufferSize = (kBufferSize / 2) + kCompressedAdditionalSize;

			encoder.SetCoderProperties(propIDs, properties);
			System.IO.MemoryStream propStream = new System.IO.MemoryStream();
			encoder.WriteCoderProperties(propStream);
			byte[] propArray = propStream.ToArray();

			CBenchRandomGenerator rg = new CBenchRandomGenerator();

			rg.Set(kBufferSize);
			rg.Generate();
			CRC crc = new CRC();
			crc.Init();
			crc.Update(rg.Buffer, 0, rg.BufferSize);

			CProgressInfo progressInfo = new CProgressInfo();
			progressInfo.ApprovedStart = dictionarySize;

			UInt64 totalBenchSize = 0;
			UInt64 totalEncodeTime = 0;
			UInt64 totalDecodeTime = 0;
			UInt64 totalCompressedSize = 0;

			MemoryStream inStream = new MemoryStream(rg.Buffer, 0, (int)rg.BufferSize);
			MemoryStream compressedStream = new MemoryStream((int)kCompressedBufferSize);
			CrcOutStream crcOutStream = new CrcOutStream();
			for (Int32 i = 0; i < numIterations; i++)
			{
				progressInfo.Init();
				inStream.Seek(0, SeekOrigin.Begin);
				compressedStream.Seek(0, SeekOrigin.Begin);
				encoder.Code(inStream, compressedStream, -1, -1, progressInfo);
				TimeSpan sp2 = DateTime.UtcNow - progressInfo.Time;
				UInt64 encodeTime = (UInt64)sp2.Ticks;

				long compressedSize = compressedStream.Position;
				if (progressInfo.InSize == 0)
					throw (new Exception("Internal ERROR 1282"));

				UInt64 decodeTime = 0;
				for (int j = 0; j < 2; j++)
				{
					compressedStream.Seek(0, SeekOrigin.Begin);
					crcOutStream.Init();

					decoder.SetDecoderProperties(propArray);
					UInt64 outSize = kBufferSize;
					System.DateTime startTime = DateTime.UtcNow;
					decoder.Code(compressedStream, crcOutStream, 0, (Int64)outSize, null);
					TimeSpan sp = (DateTime.UtcNow - startTime);
					decodeTime = (ulong)sp.Ticks;
					if (crcOutStream.GetDigest() != crc.GetDigest())
						throw (new Exception("CRC Error"));
				}
				UInt64 benchSize = kBufferSize - (UInt64)progressInfo.InSize;
				PrintResults(dictionarySize, encodeTime, benchSize, false, 0);
				System.Console.Write("     ");
				PrintResults(dictionarySize, decodeTime, kBufferSize, true, (ulong)compressedSize);
				System.Console.WriteLine();

				totalBenchSize += benchSize;
				totalEncodeTime += encodeTime;
				totalDecodeTime += decodeTime;
				totalCompressedSize += (ulong)compressedSize;
			}
			System.Console.WriteLine("---------------------------------------------------");
			PrintResults(dictionarySize, totalEncodeTime, totalBenchSize, false, 0);
			System.Console.Write("     ");
			PrintResults(dictionarySize, totalDecodeTime,
					kBufferSize * (UInt64)numIterations, true, totalCompressedSize);
			System.Console.WriteLine("    Average");
			return 0;
		}
	}
}
