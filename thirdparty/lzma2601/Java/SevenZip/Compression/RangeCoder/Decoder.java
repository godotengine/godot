package SevenZip.Compression.RangeCoder;
import java.io.IOException;

public class Decoder
{
	static final int kTopMask = ~((1 << 24) - 1);
	
	static final int kNumBitModelTotalBits = 11;
	static final int kBitModelTotal = (1 << kNumBitModelTotalBits);
	static final int kNumMoveBits = 5;
	
	int Range;
	int Code;

	java.io.InputStream Stream;
	
	public final void SetStream(java.io.InputStream stream)
	{ 
		Stream = stream; 
	}
	
	public final void ReleaseStream()
	{ 
		Stream = null; 
	}
	
	public final void Init() throws IOException
	{
		Code = 0;
		Range = -1;
		for (int i = 0; i < 5; i++)
			Code = (Code << 8) | Stream.read();
	}
	
	public final int DecodeDirectBits(int numTotalBits) throws IOException
	{
		int result = 0;
		for (int i = numTotalBits; i != 0; i--)
		{
			Range >>>= 1;
			int t = ((Code - Range) >>> 31);
			Code -= Range & (t - 1);
			result = (result << 1) | (1 - t);
			
			if ((Range & kTopMask) == 0)
			{
				Code = (Code << 8) | Stream.read();
				Range <<= 8;
			}
		}
		return result;
	}
	
	public int DecodeBit(short []probs, int index) throws IOException
	{
		int prob = probs[index];
		int newBound = (Range >>> kNumBitModelTotalBits) * prob;
		if ((Code ^ 0x80000000) < (newBound ^ 0x80000000))
		{
			Range = newBound;
			probs[index] = (short)(prob + ((kBitModelTotal - prob) >>> kNumMoveBits));
			if ((Range & kTopMask) == 0)
			{
				Code = (Code << 8) | Stream.read();
				Range <<= 8;
			}
			return 0;
		}
		else
		{
			Range -= newBound;
			Code -= newBound;
			probs[index] = (short)(prob - ((prob) >>> kNumMoveBits));
			if ((Range & kTopMask) == 0)
			{
				Code = (Code << 8) | Stream.read();
				Range <<= 8;
			}
			return 1;
		}
	}
	
	public static void InitBitModels(short []probs)
	{
		for (int i = 0; i < probs.length; i++)
			probs[i] = (kBitModelTotal >>> 1);
	}
}
