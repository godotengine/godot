/*
 *    MIT License
 *
 *    Copyright (c) 2016 Błażej Szczygieł
 * 	  Copyright (c) 2026-present The Simplicity Group & contributors
 * 	  See AUTHORS.md for more information
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all
 *    copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include "include/demuxer/WebMDemuxer.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <utility>

#include <webm/dom_types.h>
#include <webm/status.h>

namespace
{
	constexpr std::int32_t kDemuxerError = 1;
	constexpr std::uint64_t kAlphaBlockAddId = 1;
}

WebMFrame::WebMFrame() :
	bufferSize(0),
	bufferCapacity(0),
	buffer(NULL),
	alphaBufferSize(0),
	alphaBufferCapacity(0),
	alphaBuffer(NULL),
	time(0),
	key(false)
{
}

WebMFrame::~WebMFrame()
{
	free(buffer);
	free(alphaBuffer);
}

class WebMDemuxer::ParserCallback : public webm::Callback
{
public:
	std::vector<ParsedFrame> *frames;

	// Zero-based indexes into the video/audio track lists.
	int requestedVideoTrack;
	int requestedAudioTrack;

	// Actual Matroska/WebM track numbers.
	std::uint64_t videoTrack;
	std::uint64_t audioTrack;

	int videoTrackCount;
	int audioTrackCount;

	VIDEO_CODEC vCodec;
	AUDIO_CODEC aCodec;

	int width;
	int height;

	double sampleRate;
	int channels;
	int audioDepth;

	std::vector<unsigned char> audioExtradata;

	double timecodeScale;
	double duration;

	std::int64_t clusterTimecode;
	std::int16_t blockTimecode;

	std::uint64_t currentTrackNumber;
	bool currentKey;

	// Index of the most recently decoded frame belonging to the current
	// BlockGroup. Used to attach BlockAdditional alpha data.
	size_t currentFrameIndex;
	bool currentFrameIsVideo;
	bool inBlockGroup;

	ParserCallback() :
		frames(NULL),
		requestedVideoTrack(0),
		requestedAudioTrack(0),
		videoTrack(0),
		audioTrack(0),
		videoTrackCount(0),
		audioTrackCount(0),
		vCodec(NO_VIDEO),
		aCodec(NO_AUDIO),
		width(0),
		height(0),
		sampleRate(0),
		channels(0),
		audioDepth(0),
		timecodeScale(1000000),
		duration(0),
		clusterTimecode(0),
		blockTimecode(0),
		currentTrackNumber(0),
		currentKey(false),
		currentFrameIndex(SIZE_MAX),
		currentFrameIsVideo(false),
		inBlockGroup(false)
	{
	}

	void setFrames(std::vector<ParsedFrame> *p_frames)
	{
		frames = p_frames;
	}

	virtual webm::Status OnInfo(const webm::ElementMetadata &metadata, const webm::Info &info) override
	{
		(void)metadata;

		if (info.timecode_scale.is_present())
		{
			timecodeScale = static_cast<double>(info.timecode_scale.value());
		}

		if (info.duration.is_present())
		{
			duration = info.duration.value() * timecodeScale / 1e9;
		}

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnClusterBegin(const webm::ElementMetadata &metadata, const webm::Cluster &cluster, webm::Action *action) override
	{
		(void)metadata;

		*action = webm::Action::kRead;

		if (cluster.timecode.is_present())
		{
			clusterTimecode = static_cast<std::int64_t>(cluster.timecode.value());
		}
		else
		{
			clusterTimecode = 0;
		}

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnSimpleBlockBegin(const webm::ElementMetadata &metadata, const webm::SimpleBlock &simpleBlock, webm::Action *action) override
	{
		(void)metadata;

		*action = webm::Action::kRead;

		currentTrackNumber = simpleBlock.track_number;
		blockTimecode = simpleBlock.timecode;
		currentKey = simpleBlock.is_key_frame;

		currentFrameIndex = SIZE_MAX;
		currentFrameIsVideo = false;

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnBlockGroupBegin(const webm::ElementMetadata &metadata, webm::Action *action) override
	{
		(void)metadata;

		*action = webm::Action::kRead;

		currentFrameIndex = SIZE_MAX;
		currentFrameIsVideo = false;
		inBlockGroup = true;

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnBlockBegin(const webm::ElementMetadata &metadata, const webm::Block &block, webm::Action *action) override
	{
		(void)metadata;

		*action = webm::Action::kRead;

		currentTrackNumber = block.track_number;
		blockTimecode = block.timecode;

		// BlockGroup does not expose the SimpleBlock keyframe flag here.
		// Keyframe detection can be refined later using ReferenceBlock.
		currentKey = false;

		currentFrameIndex = SIZE_MAX;
		currentFrameIsVideo = false;

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnFrame(const webm::FrameMetadata &metadata, webm::Reader *reader, std::uint64_t *bytesRemaining) override
	{
		(void)metadata;

		if (!frames)
		{
			return webm::Callback::OnFrame(metadata, reader, bytesRemaining);
		}

		const bool isVideo = currentTrackNumber == videoTrack;
		const bool isAudio = currentTrackNumber == audioTrack;

		if (!isVideo && !isAudio)
		{
			return webm::Callback::OnFrame(metadata, reader, bytesRemaining);
		}

		ParsedFrame frame;

		frame.video = isVideo;
		frame.key = isVideo && currentKey;

		frame.time = (static_cast<double>(clusterTimecode) + static_cast<double>(blockTimecode)) * timecodeScale / 1e9;

		if (*bytesRemaining > static_cast<std::uint64_t>(SIZE_MAX))
		{
			return webm::Status(kDemuxerError);
		}

		frame.buffer.resize(static_cast<size_t>(*bytesRemaining));

		std::uint64_t totalBytesRead = 0;

		while (*bytesRemaining > 0)
		{
			std::uint64_t bytesRead = 0;

			webm::Status status = reader->Read(static_cast<std::size_t>(*bytesRemaining), frame.buffer.data() + totalBytesRead, &bytesRead);

			if (bytesRead > 0)
			{
				totalBytesRead += bytesRead;
				*bytesRemaining -= bytesRead;
			}

			if (!status.ok())
			{
				return status;
			}

			if (bytesRead == 0)
			{
				return webm::Status(webm::Status::kOkPartial);
			}
		}

		frame.buffer.resize(static_cast<size_t>(totalBytesRead));

		frames->push_back(std::move(frame));

		currentFrameIndex = frames->size() - 1;
		currentFrameIsVideo = isVideo;

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnBlockGroupEnd(const webm::ElementMetadata &metadata, const webm::BlockGroup &blockGroup) override
	{
		(void)metadata;

		inBlockGroup = false;

		if (!frames || currentFrameIndex == SIZE_MAX || currentFrameIndex >= frames->size() || !currentFrameIsVideo || !blockGroup.additions.is_present())
		{
			return webm::Status(webm::Status::kOkCompleted);
		}

		const webm::BlockAdditions &additions = blockGroup.additions.value();

		for (const webm::Element<webm::BlockMore> &blockMoreElement : additions.block_mores)
		{
			if (!blockMoreElement.is_present())
			{
				continue;
			}

			const webm::BlockMore &blockMore = blockMoreElement.value();

			if (!blockMore.id.is_present() || !blockMore.data.is_present())
			{
				continue;
			}

			if (blockMore.id.value() != kAlphaBlockAddId)
			{
				continue;
			}

			const std::vector<std::uint8_t> &data = blockMore.data.value();

			frames->at(currentFrameIndex).alphaBuffer = data;

			break;
		}

		return webm::Status(webm::Status::kOkCompleted);
	}

	virtual webm::Status OnTrackEntry(const webm::ElementMetadata &metadata, const webm::TrackEntry &trackEntry) override
	{
		(void)metadata;

		if (!trackEntry.track_number.is_present() || !trackEntry.track_type.is_present() || !trackEntry.codec_id.is_present())
		{
			return webm::Status(webm::Status::kOkCompleted);
		}

		const std::uint64_t trackNumber = trackEntry.track_number.value();

		const std::string &codecId = trackEntry.codec_id.value();

		if (trackEntry.track_type.value() == webm::TrackType::kVideo)
		{
			const int trackIndex = videoTrackCount++;

			if (trackIndex == requestedVideoTrack)
			{
				videoTrack = trackNumber;

				if (codecId == "V_VP8")
				{
					vCodec = VIDEO_VP8;
				}
				else if (codecId == "V_VP9")
				{
					vCodec = VIDEO_VP9;
				}

				if (trackEntry.video.is_present())
				{
					const webm::Video &video = trackEntry.video.value();

					if (video.pixel_width.is_present())
					{
						width = static_cast<int>(video.pixel_width.value());
					}

					if (video.pixel_height.is_present())
					{
						height = static_cast<int>(video.pixel_height.value());
					}
				}
			}
		}
		else if (trackEntry.track_type.value() == webm::TrackType::kAudio)
		{
			const int trackIndex = audioTrackCount++;

			if (trackIndex == requestedAudioTrack)
			{
				audioTrack = trackNumber;

				if (codecId == "A_VORBIS")
				{
					aCodec = AUDIO_VORBIS;
				}
				else if (codecId == "A_OPUS")
				{
					aCodec = AUDIO_OPUS;
				}

				if (trackEntry.codec_private.is_present())
				{
					audioExtradata = trackEntry.codec_private.value();
				}

				if (trackEntry.audio.is_present())
				{
					const webm::Audio &audio = trackEntry.audio.value();

					if (audio.sampling_frequency.is_present())
					{
						sampleRate = audio.sampling_frequency.value();
					}

					if (audio.channels.is_present())
					{
						channels = static_cast<int>(audio.channels.value());
					}

					if (audio.bit_depth.is_present())
					{
						audioDepth = static_cast<int>(audio.bit_depth.value());
					}
				}
			}
		}

		return webm::Status(webm::Status::kOkCompleted);
	}
};

WebMDemuxer::WebMDemuxer(webm::Reader *reader, int videoTrack, int audioTrack) :
	m_reader(reader),
	m_parser(),
	m_callback(new ParserCallback()),
	m_frameIndex(0),
	m_length(0),
	m_videoTrack(videoTrack),
	m_audioTrack(audioTrack),
	m_vCodec(NO_VIDEO),
	m_aCodec(NO_AUDIO),
	m_width(0),
	m_height(0),
	m_sampleRate(0),
	m_channels(0),
	m_audioDepth(0),
	m_isOpen(false),
	m_eos(false),
	m_parseError(false)
{
	m_callback->setFrames(&m_frames);

	m_callback->requestedVideoTrack = videoTrack;
	m_callback->requestedAudioTrack = audioTrack;

	webm::Status status = m_parser.Feed(m_callback, m_reader);

	if (m_callback->vCodec != NO_VIDEO)
	{
		m_vCodec = m_callback->vCodec;
		m_width = m_callback->width;
		m_height = m_callback->height;
	}

	if (m_callback->aCodec != NO_AUDIO)
	{
		m_aCodec = m_callback->aCodec;
		m_sampleRate = m_callback->sampleRate;
		m_channels = m_callback->channels;
		m_audioDepth = m_callback->audioDepth;
		m_audioExtradata = m_callback->audioExtradata;
	}

	m_length = m_callback->duration;

	if (!status.ok())
	{
		m_parseError = true;
		return;
	}

	m_isOpen = m_vCodec != NO_VIDEO || m_aCodec != NO_AUDIO;

	if (status.completed_ok())
	{
		m_eos = true;
	}
}

WebMDemuxer::~WebMDemuxer()
{
	delete m_callback;
}

double WebMDemuxer::getLength() const
{
	return m_length;
}

WebMDemuxer::VIDEO_CODEC WebMDemuxer::getVideoCodec() const
{
	return m_vCodec;
}

int WebMDemuxer::getWidth() const
{
	return m_width;
}

int WebMDemuxer::getHeight() const
{
	return m_height;
}

WebMDemuxer::AUDIO_CODEC WebMDemuxer::getAudioCodec() const
{
	return m_aCodec;
}

const unsigned char *WebMDemuxer::getAudioExtradata(size_t &size) const
{
	if (m_audioExtradata.empty())
	{
		size = 0;
		return NULL;
	}

	size = m_audioExtradata.size();

	return m_audioExtradata.data();
}

double WebMDemuxer::getSampleRate() const
{
	return m_sampleRate;
}

int WebMDemuxer::getChannels() const
{
	return m_channels;
}

int WebMDemuxer::getAudioDepth() const
{
	return m_audioDepth;
}

bool WebMDemuxer::readFrame(WebMFrame *videoFrame, WebMFrame *audioFrame)
{
	if (videoFrame)
	{
		videoFrame->bufferSize = 0;
		videoFrame->alphaBufferSize = 0;
	}

	if (audioFrame)
	{
		audioFrame->bufferSize = 0;
		audioFrame->alphaBufferSize = 0;
	}

	while (m_frameIndex < m_frames.size())
	{
		ParsedFrame &parsedFrame = m_frames[m_frameIndex++];

		WebMFrame *frame = parsedFrame.video ? videoFrame : audioFrame;

		if (!frame)
		{
			continue;
		}

		const size_t frameSize = parsedFrame.buffer.size();

		if (frameSize > static_cast<size_t>(LONG_MAX))
		{
			return false;
		}

		if (frameSize > static_cast<size_t>(frame->bufferCapacity))
		{
			unsigned char *newBuffer = static_cast<unsigned char *>(realloc(frame->buffer, frameSize));

			if (!newBuffer)
			{
				return false;
			}

			frame->buffer = newBuffer;
			frame->bufferCapacity = static_cast<long>(frameSize);
		}

		if (frameSize > 0)
		{
			memcpy(frame->buffer, parsedFrame.buffer.data(), frameSize);
		}

		frame->bufferSize = static_cast<long>(frameSize);

		frame->time = parsedFrame.time;
		frame->key = parsedFrame.key;

		const size_t alphaSize = parsedFrame.alphaBuffer.size();

		if (alphaSize > static_cast<size_t>(LONG_MAX))
		{
			return false;
		}

		if (alphaSize > static_cast<size_t>(frame->alphaBufferCapacity))
		{
			unsigned char *newAlphaBuffer = static_cast<unsigned char *>(realloc(frame->alphaBuffer, alphaSize));

			if (!newAlphaBuffer)
			{
				return false;
			}

			frame->alphaBuffer = newAlphaBuffer;
			frame->alphaBufferCapacity = static_cast<long>(alphaSize);
		}

		if (alphaSize > 0)
		{
			memcpy(frame->alphaBuffer, parsedFrame.alphaBuffer.data(), alphaSize);
		}

		frame->alphaBufferSize = static_cast<long>(alphaSize);

		return true;
	}

	return false;
}