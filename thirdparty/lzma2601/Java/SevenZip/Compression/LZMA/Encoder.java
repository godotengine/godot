package SevenZip.Compression.LZMA;

import SevenZip.Compression.RangeCoder.BitTreeEncoder;
import SevenZip.Compression.LZMA.Base;
import SevenZip.Compression.LZ.BinTree;
import SevenZip.ICodeProgress;
import java.io.IOException;

public class Encoder
{
	public static final int EMatchFinderTypeBT2 = 0;
	public static final int EMatchFinderTypeBT4 = 1;




	static final int kIfinityPrice = 0xFFFFFFF;

	static byte[] g_FastPos = new byte[1 << 11];

	static
	{
		int kFastSlots = 22;
		int c = 2;
		g_FastPos[0] = 0;
		g_FastPos[1] = 1;
		for (int slotFast = 2; slotFast < kFastSlots; slotFast++)
		{
			int k = (1 << ((slotFast >> 1) - 1));
			for (int j = 0; j < k; j++, c++)
				g_FastPos[c] = (byte)slotFast;
		}
	}

	static int GetPosSlot(int pos)
	{
		if (pos < (1 << 11))
			return g_FastPos[pos];
		if (pos < (1 << 21))
			return (g_FastPos[pos >> 10] + 20);
		return (g_FastPos[pos >> 20] + 40);
	}

	static int GetPosSlot2(int pos)
	{
		if (pos < (1 << 17))
			return (g_FastPos[pos >> 6] + 12);
		if (pos < (1 << 27))
			return (g_FastPos[pos >> 16] + 32);
		return (g_FastPos[pos >> 26] + 52);
	}

	int _state = Base.StateInit();
	byte _previousByte;
	int[] _repDistances = new int[Base.kNumRepDistances];

	void BaseInit()
	{
		_state = Base.StateInit();
		_previousByte = 0;
		for (int i = 0; i < Base.kNumRepDistances; i++)
			_repDistances[i] = 0;
	}

	static final int kDefaultDictionaryLogSize = 22;
	static final int kNumFastBytesDefault = 0x20;

	class LiteralEncoder
	{
		class Encoder2
		{
			short[] m_Encoders = new short[0x300];

			public void Init() { SevenZip.Compression.RangeCoder.Encoder.InitBitModels(m_Encoders); }



			public void Encode(SevenZip.Compression.RangeCoder.Encoder rangeEncoder, byte symbol) throws IOException
			{
				int context = 1;
				for (int i = 7; i >= 0; i--)
				{
					int bit = ((symbol >> i) & 1);
					rangeEncoder.Encode(m_Encoders, context, bit);
					context = (context << 1) | bit;
				}
			}

			public void EncodeMatched(SevenZip.Compression.RangeCoder.Encoder rangeEncoder, byte matchByte, byte symbol) throws IOException
			{
				int context = 1;
				boolean same = true;
				for (int i = 7; i >= 0; i--)
				{
					int bit = ((symbol >> i) & 1);
					int state = context;
					if (same)
					{
						int matchBit = ((matchByte >> i) & 1);
						state += ((1 + matchBit) << 8);
						same = (matchBit == bit);
					}
					rangeEncoder.Encode(m_Encoders, state, bit);
					context = (context << 1) | bit;
				}
			}

			public int GetPrice(boolean matchMode, byte matchByte, byte symbol)
			{
				int price = 0;
				int context = 1;
				int i = 7;
				if (matchMode)
				{
					for (; i >= 0; i--)
					{
						int matchBit = (matchByte >> i) & 1;
						int bit = (symbol >> i) & 1;
						price += SevenZip.Compression.RangeCoder.Encoder.GetPrice(m_Encoders[((1 + matchBit) << 8) + context], bit);
						context = (context << 1) | bit;
						if (matchBit != bit)
						{
							i--;
							break;
						}
					}
				}
				for (; i >= 0; i--)
				{
					int bit = (symbol >> i) & 1;
					price += SevenZip.Compression.RangeCoder.Encoder.GetPrice(m_Encoders[context], bit);
					context = (context << 1) | bit;
				}
				return price;
			}
		}

		Encoder2[] m_Coders;
		int m_NumPrevBits;
		int m_NumPosBits;
		int m_PosMask;

		public void Create(int numPosBits, int numPrevBits)
		{
			if (m_Coders != null && m_NumPrevBits == numPrevBits && m_NumPosBits == numPosBits)
				return;
			m_NumPosBits = numPosBits;
			m_PosMask = (1 << numPosBits) - 1;
			m_NumPrevBits = numPrevBits;
			int numStates = 1 << (m_NumPrevBits + m_NumPosBits);
			m_Coders = new Encoder2[numStates];
			for (int i = 0; i < numStates; i++)
				m_Coders[i] = new Encoder2();
		}

		public void Init()
		{
			int numStates = 1 << (m_NumPrevBits + m_NumPosBits);
			for (int i = 0; i < numStates; i++)
				m_Coders[i].Init();
		}

		public Encoder2 GetSubCoder(int pos, byte prevByte)
		{ return m_Coders[((pos & m_PosMask) << m_NumPrevBits) + ((prevByte & 0xFF) >>> (8 - m_NumPrevBits))]; }
	}

	class LenEncoder
	{
		short[] _choice = new short[2];
		BitTreeEncoder[] _lowCoder = new BitTreeEncoder[Base.kNumPosStatesEncodingMax];
		BitTreeEncoder[] _midCoder = new BitTreeEncoder[Base.kNumPosStatesEncodingMax];
		BitTreeEncoder _highCoder = new BitTreeEncoder(Base.kNumHighLenBits);


		public LenEncoder()
		{
			for (int posState = 0; posState < Base.kNumPosStatesEncodingMax; posState++)
			{
				_lowCoder[posState] = new BitTreeEncoder(Base.kNumLowLenBits);
				_midCoder[posState] = new BitTreeEncoder(Base.kNumMidLenBits);
			}
		}

		public void Init(int numPosStates)
		{
			SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_choice);

			for (int posState = 0; posState < numPosStates; posState++)
			{
				_lowCoder[posState].Init();
				_midCoder[posState].Init();
			}
			_highCoder.Init();
		}

		public void Encode(SevenZip.Compression.RangeCoder.Encoder rangeEncoder, int symbol, int posState) throws IOException
		{
			if (symbol < Base.kNumLowLenSymbols)
			{
				rangeEncoder.Encode(_choice, 0, 0);
				_lowCoder[posState].Encode(rangeEncoder, symbol);
			}
			else
			{
				symbol -= Base.kNumLowLenSymbols;
				rangeEncoder.Encode(_choice, 0, 1);
				if (symbol < Base.kNumMidLenSymbols)
				{
					rangeEncoder.Encode(_choice, 1, 0);
					_midCoder[posState].Encode(rangeEncoder, symbol);
				}
				else
				{
					rangeEncoder.Encode(_choice, 1, 1);
					_highCoder.Encode(rangeEncoder, symbol - Base.kNumMidLenSymbols);
				}
			}
		}

		public void SetPrices(int posState, int numSymbols, int[] prices, int st)
		{
			int a0 = SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_choice[0]);
			int a1 = SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_choice[0]);
			int b0 = a1 + SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_choice[1]);
			int b1 = a1 + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_choice[1]);
			int i = 0;
			for (i = 0; i < Base.kNumLowLenSymbols; i++)
			{
				if (i >= numSymbols)
					return;
				prices[st + i] = a0 + _lowCoder[posState].GetPrice(i);
			}
			for (; i < Base.kNumLowLenSymbols + Base.kNumMidLenSymbols; i++)
			{
				if (i >= numSymbols)
					return;
				prices[st + i] = b0 + _midCoder[posState].GetPrice(i - Base.kNumLowLenSymbols);
			}
			for (; i < numSymbols; i++)
				prices[st + i] = b1 + _highCoder.GetPrice(i - Base.kNumLowLenSymbols - Base.kNumMidLenSymbols);
		}
	};

	public static final int kNumLenSpecSymbols = Base.kNumLowLenSymbols + Base.kNumMidLenSymbols;

	class LenPriceTableEncoder extends LenEncoder
	{
		int[] _prices = new int[Base.kNumLenSymbols<<Base.kNumPosStatesBitsEncodingMax];
		int _tableSize;
		int[] _counters = new int[Base.kNumPosStatesEncodingMax];

		public void SetTableSize(int tableSize) { _tableSize = tableSize; }

		public int GetPrice(int symbol, int posState)
		{
			return _prices[posState * Base.kNumLenSymbols + symbol];
		}

		void UpdateTable(int posState)
		{
			SetPrices(posState, _tableSize, _prices, posState * Base.kNumLenSymbols);
			_counters[posState] = _tableSize;
		}

		public void UpdateTables(int numPosStates)
		{
			for (int posState = 0; posState < numPosStates; posState++)
				UpdateTable(posState);
		}

		public void Encode(SevenZip.Compression.RangeCoder.Encoder rangeEncoder, int symbol, int posState) throws IOException
		{
			super.Encode(rangeEncoder, symbol, posState);
			if (--_counters[posState] == 0)
				UpdateTable(posState);
		}
	}

	static final int kNumOpts = 1 << 12;
	class Optimal
	{
		public int State;

		public boolean Prev1IsChar;
		public boolean Prev2;

		public int PosPrev2;
		public int BackPrev2;

		public int Price;
		public int PosPrev;
		public int BackPrev;

		public int Backs0;
		public int Backs1;
		public int Backs2;
		public int Backs3;

		public void MakeAsChar() { BackPrev = -1; Prev1IsChar = false; }
		public void MakeAsShortRep() { BackPrev = 0; ; Prev1IsChar = false; }
		public boolean IsShortRep() { return (BackPrev == 0); }
	};
	Optimal[] _optimum = new Optimal[kNumOpts];
	SevenZip.Compression.LZ.BinTree _matchFinder = null;
	SevenZip.Compression.RangeCoder.Encoder _rangeEncoder = new SevenZip.Compression.RangeCoder.Encoder();

	short[] _isMatch = new short[Base.kNumStates<<Base.kNumPosStatesBitsMax];
	short[] _isRep = new short[Base.kNumStates];
	short[] _isRepG0 = new short[Base.kNumStates];
	short[] _isRepG1 = new short[Base.kNumStates];
	short[] _isRepG2 = new short[Base.kNumStates];
	short[] _isRep0Long = new short[Base.kNumStates<<Base.kNumPosStatesBitsMax];

	BitTreeEncoder[] _posSlotEncoder = new BitTreeEncoder[Base.kNumLenToPosStates]; // kNumPosSlotBits

	short[] _posEncoders = new short[Base.kNumFullDistances-Base.kEndPosModelIndex];
	BitTreeEncoder _posAlignEncoder = new BitTreeEncoder(Base.kNumAlignBits);

	LenPriceTableEncoder _lenEncoder = new LenPriceTableEncoder();
	LenPriceTableEncoder _repMatchLenEncoder = new LenPriceTableEncoder();

	LiteralEncoder _literalEncoder = new LiteralEncoder();

	int[] _matchDistances = new int[Base.kMatchMaxLen*2+2];

	int _numFastBytes = kNumFastBytesDefault;
	int _longestMatchLength;
	int _numDistancePairs;

	int _additionalOffset;

	int _optimumEndIndex;
	int _optimumCurrentIndex;

	boolean _longestMatchWasFound;

	int[] _posSlotPrices = new int[1<<(Base.kNumPosSlotBits+Base.kNumLenToPosStatesBits)];
	int[] _distancesPrices = new int[Base.kNumFullDistances<<Base.kNumLenToPosStatesBits];
	int[] _alignPrices = new int[Base.kAlignTableSize];
	int _alignPriceCount;

	int _distTableSize = (kDefaultDictionaryLogSize * 2);

	int _posStateBits = 2;
	int _posStateMask = (4 - 1);
	int _numLiteralPosStateBits = 0;
	int _numLiteralContextBits = 3;

	int _dictionarySize = (1 << kDefaultDictionaryLogSize);
	int _dictionarySizePrev = -1;
	int _numFastBytesPrev = -1;

	long nowPos64;
	boolean _finished;
	java.io.InputStream _inStream;

	int _matchFinderType = EMatchFinderTypeBT4;
	boolean _writeEndMark = false;

	boolean _needReleaseMFStream = false;

	void Create()
	{
		if (_matchFinder == null)
		{
			SevenZip.Compression.LZ.BinTree bt = new SevenZip.Compression.LZ.BinTree();
			int numHashBytes = 4;
			if (_matchFinderType == EMatchFinderTypeBT2)
				numHashBytes = 2;
			bt.SetType(numHashBytes);
			_matchFinder = bt;
		}
		_literalEncoder.Create(_numLiteralPosStateBits, _numLiteralContextBits);

		if (_dictionarySize == _dictionarySizePrev && _numFastBytesPrev == _numFastBytes)
			return;
		_matchFinder.Create(_dictionarySize, kNumOpts, _numFastBytes, Base.kMatchMaxLen + 1);
		_dictionarySizePrev = _dictionarySize;
		_numFastBytesPrev = _numFastBytes;
	}

	public Encoder()
	{
		for (int i = 0; i < kNumOpts; i++)
			_optimum[i] = new Optimal();
		for (int i = 0; i < Base.kNumLenToPosStates; i++)
			_posSlotEncoder[i] = new BitTreeEncoder(Base.kNumPosSlotBits);
	}

	void SetWriteEndMarkerMode(boolean writeEndMarker)
	{
		_writeEndMark = writeEndMarker;
	}

	void Init()
	{
		BaseInit();
		_rangeEncoder.Init();

		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_isMatch);
		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_isRep0Long);
		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_isRep);
		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_isRepG0);
		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_isRepG1);
		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_isRepG2);
		SevenZip.Compression.RangeCoder.Encoder.InitBitModels(_posEncoders);







		_literalEncoder.Init();
		for (int i = 0; i < Base.kNumLenToPosStates; i++)
			_posSlotEncoder[i].Init();



		_lenEncoder.Init(1 << _posStateBits);
		_repMatchLenEncoder.Init(1 << _posStateBits);

		_posAlignEncoder.Init();

		_longestMatchWasFound = false;
		_optimumEndIndex = 0;
		_optimumCurrentIndex = 0;
		_additionalOffset = 0;
	}

	int ReadMatchDistances() throws java.io.IOException
	{
		int lenRes = 0;
		_numDistancePairs = _matchFinder.GetMatches(_matchDistances);
		if (_numDistancePairs > 0)
		{
			lenRes = _matchDistances[_numDistancePairs - 2];
			if (lenRes == _numFastBytes)
				lenRes += _matchFinder.GetMatchLen((int)lenRes - 1, _matchDistances[_numDistancePairs - 1],
					Base.kMatchMaxLen - lenRes);
		}
		_additionalOffset++;
		return lenRes;
	}

	void MovePos(int num) throws java.io.IOException
	{
		if (num > 0)
		{
			_matchFinder.Skip(num);
			_additionalOffset += num;
		}
	}

	int GetRepLen1Price(int state, int posState)
	{
		return SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isRepG0[state]) +
				SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isRep0Long[(state << Base.kNumPosStatesBitsMax) + posState]);
	}

	int GetPureRepPrice(int repIndex, int state, int posState)
	{
		int price;
		if (repIndex == 0)
		{
			price = SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isRepG0[state]);
			price += SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRep0Long[(state << Base.kNumPosStatesBitsMax) + posState]);
		}
		else
		{
			price = SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRepG0[state]);
			if (repIndex == 1)
				price += SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isRepG1[state]);
			else
			{
				price += SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRepG1[state]);
				price += SevenZip.Compression.RangeCoder.Encoder.GetPrice(_isRepG2[state], repIndex - 2);
			}
		}
		return price;
	}

	int GetRepPrice(int repIndex, int len, int state, int posState)
	{
		int price = _repMatchLenEncoder.GetPrice(len - Base.kMatchMinLen, posState);
		return price + GetPureRepPrice(repIndex, state, posState);
	}

	int GetPosLenPrice(int pos, int len, int posState)
	{
		int price;
		int lenToPosState = Base.GetLenToPosState(len);
		if (pos < Base.kNumFullDistances)
			price = _distancesPrices[(lenToPosState * Base.kNumFullDistances) + pos];
		else
			price = _posSlotPrices[(lenToPosState << Base.kNumPosSlotBits) + GetPosSlot2(pos)] +
				_alignPrices[pos & Base.kAlignMask];
		return price + _lenEncoder.GetPrice(len - Base.kMatchMinLen, posState);
	}

	int Backward(int cur)
	{
		_optimumEndIndex = cur;
		int posMem = _optimum[cur].PosPrev;
		int backMem = _optimum[cur].BackPrev;
		do
		{
			if (_optimum[cur].Prev1IsChar)
			{
				_optimum[posMem].MakeAsChar();
				_optimum[posMem].PosPrev = posMem - 1;
				if (_optimum[cur].Prev2)
				{
					_optimum[posMem - 1].Prev1IsChar = false;
					_optimum[posMem - 1].PosPrev = _optimum[cur].PosPrev2;
					_optimum[posMem - 1].BackPrev = _optimum[cur].BackPrev2;
				}
			}
			int posPrev = posMem;
			int backCur = backMem;

			backMem = _optimum[posPrev].BackPrev;
			posMem = _optimum[posPrev].PosPrev;

			_optimum[posPrev].BackPrev = backCur;
			_optimum[posPrev].PosPrev = cur;
			cur = posPrev;
		}
		while (cur > 0);
		backRes = _optimum[0].BackPrev;
		_optimumCurrentIndex = _optimum[0].PosPrev;
		return _optimumCurrentIndex;
	}

	int[] reps = new int[Base.kNumRepDistances];
	int[] repLens = new int[Base.kNumRepDistances];
	int backRes;

	int GetOptimum(int position) throws IOException
	{
		if (_optimumEndIndex != _optimumCurrentIndex)
		{
			int lenRes = _optimum[_optimumCurrentIndex].PosPrev - _optimumCurrentIndex;
			backRes = _optimum[_optimumCurrentIndex].BackPrev;
			_optimumCurrentIndex = _optimum[_optimumCurrentIndex].PosPrev;
			return lenRes;
		}
		_optimumCurrentIndex = _optimumEndIndex = 0;

		int lenMain, numDistancePairs;
		if (!_longestMatchWasFound)
		{
			lenMain = ReadMatchDistances();
		}
		else
		{
			lenMain = _longestMatchLength;
			_longestMatchWasFound = false;
		}
		numDistancePairs = _numDistancePairs;

		int numAvailableBytes = _matchFinder.GetNumAvailableBytes() + 1;
		if (numAvailableBytes < 2)
		{
			backRes = -1;
			return 1;
		}
		if (numAvailableBytes > Base.kMatchMaxLen)
			numAvailableBytes = Base.kMatchMaxLen;

		int repMaxIndex = 0;
		int i;
		for (i = 0; i < Base.kNumRepDistances; i++)
		{
			reps[i] = _repDistances[i];
			repLens[i] = _matchFinder.GetMatchLen(0 - 1, reps[i], Base.kMatchMaxLen);
			if (repLens[i] > repLens[repMaxIndex])
				repMaxIndex = i;
		}
		if (repLens[repMaxIndex] >= _numFastBytes)
		{
			backRes = repMaxIndex;
			int lenRes = repLens[repMaxIndex];
			MovePos(lenRes - 1);
			return lenRes;
		}

		if (lenMain >= _numFastBytes)
		{
			backRes = _matchDistances[numDistancePairs - 1] + Base.kNumRepDistances;
			MovePos(lenMain - 1);
			return lenMain;
		}

		byte currentByte = _matchFinder.GetIndexByte(0 - 1);
		byte matchByte = _matchFinder.GetIndexByte(0 - _repDistances[0] - 1 - 1);

		if (lenMain < 2 && currentByte != matchByte && repLens[repMaxIndex] < 2)
		{
			backRes = -1;
			return 1;
		}

		_optimum[0].State = _state;

		int posState = (position & _posStateMask);

		_optimum[1].Price = SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isMatch[(_state << Base.kNumPosStatesBitsMax) + posState]) +
				_literalEncoder.GetSubCoder(position, _previousByte).GetPrice(!Base.StateIsCharState(_state), matchByte, currentByte);
		_optimum[1].MakeAsChar();

		int matchPrice = SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isMatch[(_state << Base.kNumPosStatesBitsMax) + posState]);
		int repMatchPrice = matchPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRep[_state]);

		if (matchByte == currentByte)
		{
			int shortRepPrice = repMatchPrice + GetRepLen1Price(_state, posState);
			if (shortRepPrice < _optimum[1].Price)
			{
				_optimum[1].Price = shortRepPrice;
				_optimum[1].MakeAsShortRep();
			}
		}

		int lenEnd = ((lenMain >= repLens[repMaxIndex]) ? lenMain : repLens[repMaxIndex]);

		if (lenEnd < 2)
		{
			backRes = _optimum[1].BackPrev;
			return 1;
		}

		_optimum[1].PosPrev = 0;

		_optimum[0].Backs0 = reps[0];
		_optimum[0].Backs1 = reps[1];
		_optimum[0].Backs2 = reps[2];
		_optimum[0].Backs3 = reps[3];

		int len = lenEnd;
		do
			_optimum[len--].Price = kIfinityPrice;
		while (len >= 2);

		for (i = 0; i < Base.kNumRepDistances; i++)
		{
			int repLen = repLens[i];
			if (repLen < 2)
				continue;
			int price = repMatchPrice + GetPureRepPrice(i, _state, posState);
			do
			{
				int curAndLenPrice = price + _repMatchLenEncoder.GetPrice(repLen - 2, posState);
				Optimal optimum = _optimum[repLen];
				if (curAndLenPrice < optimum.Price)
				{
					optimum.Price = curAndLenPrice;
					optimum.PosPrev = 0;
					optimum.BackPrev = i;
					optimum.Prev1IsChar = false;
				}
			}
			while (--repLen >= 2);
		}

		int normalMatchPrice = matchPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isRep[_state]);

		len = ((repLens[0] >= 2) ? repLens[0] + 1 : 2);
		if (len <= lenMain)
		{
			int offs = 0;
			while (len > _matchDistances[offs])
				offs += 2;
			for (; ; len++)
			{
				int distance = _matchDistances[offs + 1];
				int curAndLenPrice = normalMatchPrice + GetPosLenPrice(distance, len, posState);
				Optimal optimum = _optimum[len];
				if (curAndLenPrice < optimum.Price)
				{
					optimum.Price = curAndLenPrice;
					optimum.PosPrev = 0;
					optimum.BackPrev = distance + Base.kNumRepDistances;
					optimum.Prev1IsChar = false;
				}
				if (len == _matchDistances[offs])
				{
					offs += 2;
					if (offs == numDistancePairs)
						break;
				}
			}
		}

		int cur = 0;

		while (true)
		{
			cur++;
			if (cur == lenEnd)
				return Backward(cur);
			int newLen = ReadMatchDistances();
			numDistancePairs = _numDistancePairs;
			if (newLen >= _numFastBytes)
			{

				_longestMatchLength = newLen;
				_longestMatchWasFound = true;
				return Backward(cur);
			}
			position++;
			int posPrev = _optimum[cur].PosPrev;
			int state;
			if (_optimum[cur].Prev1IsChar)
			{
				posPrev--;
				if (_optimum[cur].Prev2)
				{
					state = _optimum[_optimum[cur].PosPrev2].State;
					if (_optimum[cur].BackPrev2 < Base.kNumRepDistances)
						state = Base.StateUpdateRep(state);
					else
						state = Base.StateUpdateMatch(state);
				}
				else
					state = _optimum[posPrev].State;
				state = Base.StateUpdateChar(state);
			}
			else
				state = _optimum[posPrev].State;
			if (posPrev == cur - 1)
			{
				if (_optimum[cur].IsShortRep())
					state = Base.StateUpdateShortRep(state);
				else
					state = Base.StateUpdateChar(state);
			}
			else
			{
				int pos;
				if (_optimum[cur].Prev1IsChar && _optimum[cur].Prev2)
				{
					posPrev = _optimum[cur].PosPrev2;
					pos = _optimum[cur].BackPrev2;
					state = Base.StateUpdateRep(state);
				}
				else
				{
					pos = _optimum[cur].BackPrev;
					if (pos < Base.kNumRepDistances)
						state = Base.StateUpdateRep(state);
					else
						state = Base.StateUpdateMatch(state);
				}
				Optimal opt = _optimum[posPrev];
				if (pos < Base.kNumRepDistances)
				{
					if (pos == 0)
					{
						reps[0] = opt.Backs0;
						reps[1] = opt.Backs1;
						reps[2] = opt.Backs2;
						reps[3] = opt.Backs3;
					}
					else if (pos == 1)
					{
						reps[0] = opt.Backs1;
						reps[1] = opt.Backs0;
						reps[2] = opt.Backs2;
						reps[3] = opt.Backs3;
					}
					else if (pos == 2)
					{
						reps[0] = opt.Backs2;
						reps[1] = opt.Backs0;
						reps[2] = opt.Backs1;
						reps[3] = opt.Backs3;
					}
					else
					{
						reps[0] = opt.Backs3;
						reps[1] = opt.Backs0;
						reps[2] = opt.Backs1;
						reps[3] = opt.Backs2;
					}
				}
				else
				{
					reps[0] = (pos - Base.kNumRepDistances);
					reps[1] = opt.Backs0;
					reps[2] = opt.Backs1;
					reps[3] = opt.Backs2;
				}
			}
			_optimum[cur].State = state;
			_optimum[cur].Backs0 = reps[0];
			_optimum[cur].Backs1 = reps[1];
			_optimum[cur].Backs2 = reps[2];
			_optimum[cur].Backs3 = reps[3];
			int curPrice = _optimum[cur].Price;

			currentByte = _matchFinder.GetIndexByte(0 - 1);
			matchByte = _matchFinder.GetIndexByte(0 - reps[0] - 1 - 1);

			posState = (position & _posStateMask);

			int curAnd1Price = curPrice +
				SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isMatch[(state << Base.kNumPosStatesBitsMax) + posState]) +
				_literalEncoder.GetSubCoder(position, _matchFinder.GetIndexByte(0 - 2)).
				GetPrice(!Base.StateIsCharState(state), matchByte, currentByte);

			Optimal nextOptimum = _optimum[cur + 1];

			boolean nextIsChar = false;
			if (curAnd1Price < nextOptimum.Price)
			{
				nextOptimum.Price = curAnd1Price;
				nextOptimum.PosPrev = cur;
				nextOptimum.MakeAsChar();
				nextIsChar = true;
			}

			matchPrice = curPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isMatch[(state << Base.kNumPosStatesBitsMax) + posState]);
			repMatchPrice = matchPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRep[state]);

			if (matchByte == currentByte &&
				!(nextOptimum.PosPrev < cur && nextOptimum.BackPrev == 0))
			{
				int shortRepPrice = repMatchPrice + GetRepLen1Price(state, posState);
				if (shortRepPrice <= nextOptimum.Price)
				{
					nextOptimum.Price = shortRepPrice;
					nextOptimum.PosPrev = cur;
					nextOptimum.MakeAsShortRep();
					nextIsChar = true;
				}
			}

			int numAvailableBytesFull = _matchFinder.GetNumAvailableBytes() + 1;
			numAvailableBytesFull = Math.min(kNumOpts - 1 - cur, numAvailableBytesFull);
			numAvailableBytes = numAvailableBytesFull;

			if (numAvailableBytes < 2)
				continue;
			if (numAvailableBytes > _numFastBytes)
				numAvailableBytes = _numFastBytes;
			if (!nextIsChar && matchByte != currentByte)
			{
				// try Literal + rep0
				int t = Math.min(numAvailableBytesFull - 1, _numFastBytes);
				int lenTest2 = _matchFinder.GetMatchLen(0, reps[0], t);
				if (lenTest2 >= 2)
				{
					int state2 = Base.StateUpdateChar(state);

					int posStateNext = (position + 1) & _posStateMask;
					int nextRepMatchPrice = curAnd1Price +
						SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isMatch[(state2 << Base.kNumPosStatesBitsMax) + posStateNext]) +
						SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRep[state2]);
					{
						int offset = cur + 1 + lenTest2;
						while (lenEnd < offset)
							_optimum[++lenEnd].Price = kIfinityPrice;
						int curAndLenPrice = nextRepMatchPrice + GetRepPrice(
								0, lenTest2, state2, posStateNext);
						Optimal optimum = _optimum[offset];
						if (curAndLenPrice < optimum.Price)
						{
							optimum.Price = curAndLenPrice;
							optimum.PosPrev = cur + 1;
							optimum.BackPrev = 0;
							optimum.Prev1IsChar = true;
							optimum.Prev2 = false;
						}
					}
				}
			}

			int startLen = 2; // speed optimization 

			for (int repIndex = 0; repIndex < Base.kNumRepDistances; repIndex++)
			{
				int lenTest = _matchFinder.GetMatchLen(0 - 1, reps[repIndex], numAvailableBytes);
				if (lenTest < 2)
					continue;
				int lenTestTemp = lenTest;
				do
				{
					while (lenEnd < cur + lenTest)
						_optimum[++lenEnd].Price = kIfinityPrice;
					int curAndLenPrice = repMatchPrice + GetRepPrice(repIndex, lenTest, state, posState);
					Optimal optimum = _optimum[cur + lenTest];
					if (curAndLenPrice < optimum.Price)
					{
						optimum.Price = curAndLenPrice;
						optimum.PosPrev = cur;
						optimum.BackPrev = repIndex;
						optimum.Prev1IsChar = false;
					}
				}
				while (--lenTest >= 2);
				lenTest = lenTestTemp;

				if (repIndex == 0)
					startLen = lenTest + 1;

				// if (_maxMode)
				if (lenTest < numAvailableBytesFull)
				{
					int t = Math.min(numAvailableBytesFull - 1 - lenTest, _numFastBytes);
					int lenTest2 = _matchFinder.GetMatchLen(lenTest, reps[repIndex], t);
					if (lenTest2 >= 2)
					{
						int state2 = Base.StateUpdateRep(state);

						int posStateNext = (position + lenTest) & _posStateMask;
						int curAndLenCharPrice =
								repMatchPrice + GetRepPrice(repIndex, lenTest, state, posState) +
								SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isMatch[(state2 << Base.kNumPosStatesBitsMax) + posStateNext]) +
								_literalEncoder.GetSubCoder(position + lenTest,
								_matchFinder.GetIndexByte(lenTest - 1 - 1)).GetPrice(true,
								_matchFinder.GetIndexByte(lenTest - 1 - (reps[repIndex] + 1)),
								_matchFinder.GetIndexByte(lenTest - 1));
						state2 = Base.StateUpdateChar(state2);
						posStateNext = (position + lenTest + 1) & _posStateMask;
						int nextMatchPrice = curAndLenCharPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isMatch[(state2 << Base.kNumPosStatesBitsMax) + posStateNext]);
						int nextRepMatchPrice = nextMatchPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRep[state2]);

						// for(; lenTest2 >= 2; lenTest2--)
						{
							int offset = lenTest + 1 + lenTest2;
							while (lenEnd < cur + offset)
								_optimum[++lenEnd].Price = kIfinityPrice;
							int curAndLenPrice = nextRepMatchPrice + GetRepPrice(0, lenTest2, state2, posStateNext);
							Optimal optimum = _optimum[cur + offset];
							if (curAndLenPrice < optimum.Price)
							{
								optimum.Price = curAndLenPrice;
								optimum.PosPrev = cur + lenTest + 1;
								optimum.BackPrev = 0;
								optimum.Prev1IsChar = true;
								optimum.Prev2 = true;
								optimum.PosPrev2 = cur;
								optimum.BackPrev2 = repIndex;
							}
						}
					}
				}
			}

			if (newLen > numAvailableBytes)
			{
				newLen = numAvailableBytes;
				for (numDistancePairs = 0; newLen > _matchDistances[numDistancePairs]; numDistancePairs += 2) ;
				_matchDistances[numDistancePairs] = newLen;
				numDistancePairs += 2;
			}
			if (newLen >= startLen)
			{
				normalMatchPrice = matchPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isRep[state]);
				while (lenEnd < cur + newLen)
					_optimum[++lenEnd].Price = kIfinityPrice;

				int offs = 0;
				while (startLen > _matchDistances[offs])
					offs += 2;

				for (int lenTest = startLen; ; lenTest++)
				{
					int curBack = _matchDistances[offs + 1];
					int curAndLenPrice = normalMatchPrice + GetPosLenPrice(curBack, lenTest, posState);
					Optimal optimum = _optimum[cur + lenTest];
					if (curAndLenPrice < optimum.Price)
					{
						optimum.Price = curAndLenPrice;
						optimum.PosPrev = cur;
						optimum.BackPrev = curBack + Base.kNumRepDistances;
						optimum.Prev1IsChar = false;
					}

					if (lenTest == _matchDistances[offs])
					{
						if (lenTest < numAvailableBytesFull)
						{
							int t = Math.min(numAvailableBytesFull - 1 - lenTest, _numFastBytes);
							int lenTest2 = _matchFinder.GetMatchLen(lenTest, curBack, t);
							if (lenTest2 >= 2)
							{
								int state2 = Base.StateUpdateMatch(state);

								int posStateNext = (position + lenTest) & _posStateMask;
								int curAndLenCharPrice = curAndLenPrice +
									SevenZip.Compression.RangeCoder.Encoder.GetPrice0(_isMatch[(state2 << Base.kNumPosStatesBitsMax) + posStateNext]) +
									_literalEncoder.GetSubCoder(position + lenTest,
									_matchFinder.GetIndexByte(lenTest - 1 - 1)).
									GetPrice(true,
									_matchFinder.GetIndexByte(lenTest - (curBack + 1) - 1),
									_matchFinder.GetIndexByte(lenTest - 1));
								state2 = Base.StateUpdateChar(state2);
								posStateNext = (position + lenTest + 1) & _posStateMask;
								int nextMatchPrice = curAndLenCharPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isMatch[(state2 << Base.kNumPosStatesBitsMax) + posStateNext]);
								int nextRepMatchPrice = nextMatchPrice + SevenZip.Compression.RangeCoder.Encoder.GetPrice1(_isRep[state2]);

								int offset = lenTest + 1 + lenTest2;
								while (lenEnd < cur + offset)
									_optimum[++lenEnd].Price = kIfinityPrice;
								curAndLenPrice = nextRepMatchPrice + GetRepPrice(0, lenTest2, state2, posStateNext);
								optimum = _optimum[cur + offset];
								if (curAndLenPrice < optimum.Price)
								{
									optimum.Price = curAndLenPrice;
									optimum.PosPrev = cur + lenTest + 1;
									optimum.BackPrev = 0;
									optimum.Prev1IsChar = true;
									optimum.Prev2 = true;
									optimum.PosPrev2 = cur;
									optimum.BackPrev2 = curBack + Base.kNumRepDistances;
								}
							}
						}
						offs += 2;
						if (offs == numDistancePairs)
							break;
					}
				}
			}
		}
	}

	boolean ChangePair(int smallDist, int bigDist)
	{
		int kDif = 7;
		return (smallDist < (1 << (32 - kDif)) && bigDist >= (smallDist << kDif));
	}

	void WriteEndMarker(int posState) throws IOException
	{
		if (!_writeEndMark)
			return;

		_rangeEncoder.Encode(_isMatch, (_state << Base.kNumPosStatesBitsMax) + posState, 1);
		_rangeEncoder.Encode(_isRep, _state, 0);
		_state = Base.StateUpdateMatch(_state);
		int len = Base.kMatchMinLen;
		_lenEncoder.Encode(_rangeEncoder, len - Base.kMatchMinLen, posState);
		int posSlot = (1 << Base.kNumPosSlotBits) - 1;
		int lenToPosState = Base.GetLenToPosState(len);
		_posSlotEncoder[lenToPosState].Encode(_rangeEncoder, posSlot);
		int footerBits = 30;
		int posReduced = (1 << footerBits) - 1;
		_rangeEncoder.EncodeDirectBits(posReduced >> Base.kNumAlignBits, footerBits - Base.kNumAlignBits);
		_posAlignEncoder.ReverseEncode(_rangeEncoder, posReduced & Base.kAlignMask);
	}

	void Flush(int nowPos) throws IOException
	{
		ReleaseMFStream();
		WriteEndMarker(nowPos & _posStateMask);
		_rangeEncoder.FlushData();
		_rangeEncoder.FlushStream();
	}

	public void CodeOneBlock(long[] inSize, long[] outSize, boolean[] finished) throws IOException
	{
		inSize[0] = 0;
		outSize[0] = 0;
		finished[0] = true;

		if (_inStream != null)
		{
			_matchFinder.SetStream(_inStream);
			_matchFinder.Init();
			_needReleaseMFStream = true;
			_inStream = null;
		}

		if (_finished)
			return;
		_finished = true;


		long progressPosValuePrev = nowPos64;
		if (nowPos64 == 0)
		{
			if (_matchFinder.GetNumAvailableBytes() == 0)
			{
				Flush((int)nowPos64);
				return;
			}

			ReadMatchDistances();
			int posState = (int)(nowPos64) & _posStateMask;
			_rangeEncoder.Encode(_isMatch, (_state << Base.kNumPosStatesBitsMax) + posState, 0);
			_state = Base.StateUpdateChar(_state);
			byte curByte = _matchFinder.GetIndexByte(0 - _additionalOffset);
			_literalEncoder.GetSubCoder((int)(nowPos64), _previousByte).Encode(_rangeEncoder, curByte);
			_previousByte = curByte;
			_additionalOffset--;
			nowPos64++;
		}
		if (_matchFinder.GetNumAvailableBytes() == 0)
		{
			Flush((int)nowPos64);
			return;
		}
		while (true)
		{

			int len = GetOptimum((int)nowPos64);
			int pos = backRes;
			int posState = ((int)nowPos64) & _posStateMask;
			int complexState = (_state << Base.kNumPosStatesBitsMax) + posState;
			if (len == 1 && pos == -1)
			{
				_rangeEncoder.Encode(_isMatch, complexState, 0);
				byte curByte = _matchFinder.GetIndexByte((int)(0 - _additionalOffset));
				LiteralEncoder.Encoder2 subCoder = _literalEncoder.GetSubCoder((int)nowPos64, _previousByte);
				if (!Base.StateIsCharState(_state))
				{
					byte matchByte = _matchFinder.GetIndexByte((int)(0 - _repDistances[0] - 1 - _additionalOffset));
					subCoder.EncodeMatched(_rangeEncoder, matchByte, curByte);
				}
				else
					subCoder.Encode(_rangeEncoder, curByte);
				_previousByte = curByte;
				_state = Base.StateUpdateChar(_state);
			}
			else
			{
				_rangeEncoder.Encode(_isMatch, complexState, 1);
				if (pos < Base.kNumRepDistances)
				{
					_rangeEncoder.Encode(_isRep, _state, 1);
					if (pos == 0)
					{
						_rangeEncoder.Encode(_isRepG0, _state, 0);
						if (len == 1)
							_rangeEncoder.Encode(_isRep0Long, complexState, 0);
						else
							_rangeEncoder.Encode(_isRep0Long, complexState, 1);
					}
					else
					{
						_rangeEncoder.Encode(_isRepG0, _state, 1);
						if (pos == 1)
							_rangeEncoder.Encode(_isRepG1, _state, 0);
						else
						{
							_rangeEncoder.Encode(_isRepG1, _state, 1);
							_rangeEncoder.Encode(_isRepG2, _state, pos - 2);
						}
					}
					if (len == 1)
						_state = Base.StateUpdateShortRep(_state);
					else
					{
						_repMatchLenEncoder.Encode(_rangeEncoder, len - Base.kMatchMinLen, posState);
						_state = Base.StateUpdateRep(_state);
					}
					int distance = _repDistances[pos];
					if (pos != 0)
					{
						for (int i = pos; i >= 1; i--)
							_repDistances[i] = _repDistances[i - 1];
						_repDistances[0] = distance;
					}
				}
				else
				{
					_rangeEncoder.Encode(_isRep, _state, 0);
					_state = Base.StateUpdateMatch(_state);
					_lenEncoder.Encode(_rangeEncoder, len - Base.kMatchMinLen, posState);
					pos -= Base.kNumRepDistances;
					int posSlot = GetPosSlot(pos);
					int lenToPosState = Base.GetLenToPosState(len);
					_posSlotEncoder[lenToPosState].Encode(_rangeEncoder, posSlot);

					if (posSlot >= Base.kStartPosModelIndex)
					{
						int footerBits = (int)((posSlot >> 1) - 1);
						int baseVal = ((2 | (posSlot & 1)) << footerBits);
						int posReduced = pos - baseVal;

						if (posSlot < Base.kEndPosModelIndex)
							BitTreeEncoder.ReverseEncode(_posEncoders,
									baseVal - posSlot - 1, _rangeEncoder, footerBits, posReduced);
						else
						{
							_rangeEncoder.EncodeDirectBits(posReduced >> Base.kNumAlignBits, footerBits - Base.kNumAlignBits);
							_posAlignEncoder.ReverseEncode(_rangeEncoder, posReduced & Base.kAlignMask);
							_alignPriceCount++;
						}
					}
					int distance = pos;
					for (int i = Base.kNumRepDistances - 1; i >= 1; i--)
						_repDistances[i] = _repDistances[i - 1];
					_repDistances[0] = distance;
					_matchPriceCount++;
				}
				_previousByte = _matchFinder.GetIndexByte(len - 1 - _additionalOffset);
			}
			_additionalOffset -= len;
			nowPos64 += len;
			if (_additionalOffset == 0)
			{
				// if (!_fastMode)
				if (_matchPriceCount >= (1 << 7))
					FillDistancesPrices();
				if (_alignPriceCount >= Base.kAlignTableSize)
					FillAlignPrices();
				inSize[0] = nowPos64;
				outSize[0] = _rangeEncoder.GetProcessedSizeAdd();
				if (_matchFinder.GetNumAvailableBytes() == 0)
				{
					Flush((int)nowPos64);
					return;
				}

				if (nowPos64 - progressPosValuePrev >= (1 << 12))
				{
					_finished = false;
					finished[0] = false;
					return;
				}
			}
		}
	}

	void ReleaseMFStream()
	{
		if (_matchFinder != null && _needReleaseMFStream)
		{
			_matchFinder.ReleaseStream();
			_needReleaseMFStream = false;
		}
	}

	void SetOutStream(java.io.OutputStream outStream)
	{ _rangeEncoder.SetStream(outStream); }
	void ReleaseOutStream()
	{ _rangeEncoder.ReleaseStream(); }

	void ReleaseStreams()
	{
		ReleaseMFStream();
		ReleaseOutStream();
	}

	void SetStreams(java.io.InputStream inStream, java.io.OutputStream outStream,
			long inSize, long outSize)
	{
		_inStream = inStream;
		_finished = false;
		Create();
		SetOutStream(outStream);
		Init();

		// if (!_fastMode)
		{
			FillDistancesPrices();
			FillAlignPrices();
		}

		_lenEncoder.SetTableSize(_numFastBytes + 1 - Base.kMatchMinLen);
		_lenEncoder.UpdateTables(1 << _posStateBits);
		_repMatchLenEncoder.SetTableSize(_numFastBytes + 1 - Base.kMatchMinLen);
		_repMatchLenEncoder.UpdateTables(1 << _posStateBits);

		nowPos64 = 0;
	}

	long[] processedInSize = new long[1]; long[] processedOutSize = new long[1]; boolean[] finished = new boolean[1];
	public void Code(java.io.InputStream inStream, java.io.OutputStream outStream,
			long inSize, long outSize, ICodeProgress progress) throws IOException
	{
		_needReleaseMFStream = false;
		try
		{
			SetStreams(inStream, outStream, inSize, outSize);
			while (true)
			{



				CodeOneBlock(processedInSize, processedOutSize, finished);
				if (finished[0])
					return;
				if (progress != null)
				{
					progress.SetProgress(processedInSize[0], processedOutSize[0]);
				}
			}
		}
		finally
		{
			ReleaseStreams();
		}
	}

	public static final int kPropSize = 5;
	byte[] properties = new byte[kPropSize];

	public void WriteCoderProperties(java.io.OutputStream outStream) throws IOException
	{
		properties[0] = (byte)((_posStateBits * 5 + _numLiteralPosStateBits) * 9 + _numLiteralContextBits);
		for (int i = 0; i < 4; i++)
			properties[1 + i] = (byte)(_dictionarySize >> (8 * i));
		outStream.write(properties, 0, kPropSize);
	}

	int[] tempPrices = new int[Base.kNumFullDistances];
	int _matchPriceCount;

	void FillDistancesPrices()
	{
		for (int i = Base.kStartPosModelIndex; i < Base.kNumFullDistances; i++)
		{
			int posSlot = GetPosSlot(i);
			int footerBits = (int)((posSlot >> 1) - 1);
			int baseVal = ((2 | (posSlot & 1)) << footerBits);
			tempPrices[i] = BitTreeEncoder.ReverseGetPrice(_posEncoders,
				baseVal - posSlot - 1, footerBits, i - baseVal);
		}

		for (int lenToPosState = 0; lenToPosState < Base.kNumLenToPosStates; lenToPosState++)
		{
			int posSlot;
			BitTreeEncoder encoder = _posSlotEncoder[lenToPosState];

			int st = (lenToPosState << Base.kNumPosSlotBits);
			for (posSlot = 0; posSlot < _distTableSize; posSlot++)
				_posSlotPrices[st + posSlot] = encoder.GetPrice(posSlot);
			for (posSlot = Base.kEndPosModelIndex; posSlot < _distTableSize; posSlot++)
				_posSlotPrices[st + posSlot] += ((((posSlot >> 1) - 1) - Base.kNumAlignBits) << SevenZip.Compression.RangeCoder.Encoder.kNumBitPriceShiftBits);

			int st2 = lenToPosState * Base.kNumFullDistances;
			int i;
			for (i = 0; i < Base.kStartPosModelIndex; i++)
				_distancesPrices[st2 + i] = _posSlotPrices[st + i];
			for (; i < Base.kNumFullDistances; i++)
				_distancesPrices[st2 + i] = _posSlotPrices[st + GetPosSlot(i)] + tempPrices[i];
		}
		_matchPriceCount = 0;
	}

	void FillAlignPrices()
	{
		for (int i = 0; i < Base.kAlignTableSize; i++)
			_alignPrices[i] = _posAlignEncoder.ReverseGetPrice(i);
		_alignPriceCount = 0;
	}


	public boolean SetAlgorithm(int algorithm)
	{
		/*
		_fastMode = (algorithm == 0);
		_maxMode = (algorithm >= 2);
		*/
		return true;
	}

	public boolean SetDictionarySize(int dictionarySize)
	{
		int kDicLogSizeMaxCompress = 29;
		if (dictionarySize < (1 << Base.kDicLogSizeMin) || dictionarySize > (1 << kDicLogSizeMaxCompress))
			return false;
		_dictionarySize = dictionarySize;
		int dicLogSize;
		for (dicLogSize = 0; dictionarySize > (1 << dicLogSize); dicLogSize++) ;
		_distTableSize = dicLogSize * 2;
		return true;
	}

	public boolean SetNumFastBytes(int numFastBytes)
	{
		if (numFastBytes < 5 || numFastBytes > Base.kMatchMaxLen)
			return false;
		_numFastBytes = numFastBytes;
		return true;
	}

	public boolean SetMatchFinder(int matchFinderIndex)
	{
		if (matchFinderIndex < 0 || matchFinderIndex > 2)
			return false;
		int matchFinderIndexPrev = _matchFinderType;
		_matchFinderType = matchFinderIndex;
		if (_matchFinder != null && matchFinderIndexPrev != _matchFinderType)
		{
			_dictionarySizePrev = -1;
			_matchFinder = null;
		}
		return true;
	}

	public boolean SetLcLpPb(int lc, int lp, int pb)
	{
		if (
				lp < 0 || lp > Base.kNumLitPosStatesBitsEncodingMax ||
				lc < 0 || lc > Base.kNumLitContextBitsMax ||
				pb < 0 || pb > Base.kNumPosStatesBitsEncodingMax)
			return false;
		_numLiteralPosStateBits = lp;
		_numLiteralContextBits = lc;
		_posStateBits = pb;
		_posStateMask = ((1) << _posStateBits) - 1;
		return true;
	}

	public void SetEndMarkerMode(boolean endMarkerMode)
	{
		_writeEndMark = endMarkerMode;
	}
}

