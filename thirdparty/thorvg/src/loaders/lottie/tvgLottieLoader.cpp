/*
 * Copyright (c) 2023 - 2024 the ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tvgLottieLoader.h"
#include "tvgLottieModel.h"
#include "tvgLottieParser.h"
#include "tvgLottieBuilder.h"
#include "tvgStr.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

void LottieLoader::run(unsigned tid)
{
    //update frame
    if (comp) {
        builder->update(comp, frameNo);
    //initial loading
    } else {
        LottieParser parser(content, dirName);
        if (!parser.parse()) return;
        comp = parser.comp;
        builder->build(comp);
    }
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

LottieLoader::LottieLoader() : FrameModule(FileType::Lottie), builder(new LottieBuilder)
{

}


LottieLoader::~LottieLoader()
{
    this->done();

    if (copy) free((char*)content);
    free(dirName);

    //TODO: correct position?
    delete(comp);
    delete(builder);
}


bool LottieLoader::header()
{
    //A single thread doesn't need to perform intensive tasks.
    if (TaskScheduler::threads() == 0) {
        run(0);
        if (comp) {
            w = static_cast<float>(comp->w);
            h = static_cast<float>(comp->h);
            frameDuration = comp->duration();
            frameCnt = comp->frameCnt();
            return true;
        } else {
            return false;
        }
    }

    //Quickly validate the given Lottie file without parsing in order to get the animation info.
    auto startFrame = 0.0f;
    auto endFrame = 0.0f;
    auto frameRate = 0.0f;
    uint32_t depth = 0;

    auto p = content;

    while (*p != '\0') {
        if (*p == '{') {
            ++depth;
            ++p;
            continue;
        }
        if (*p == '}') {
            --depth;
            ++p;
            continue;
        }
        if (depth != 1) {
            ++p;
            continue;
        }
        //version.
        if (!strncmp(p, "\"v\":", 4)) {
            p += 4;
            continue;
        }

        //framerate
        if (!strncmp(p, "\"fr\":", 5)) {
            p += 5;
            auto e = strstr(p, ",");
            if (!e) e = strstr(p, "}");
            frameRate = strToFloat(p, nullptr);
            p = e;
            continue;
        }

        //start frame
        if (!strncmp(p, "\"ip\":", 5)) {
            p += 5;
            auto e = strstr(p, ",");
            if (!e) e = strstr(p, "}");
            startFrame = strToFloat(p, nullptr);
            p = e;
            continue;
        }

        //end frame
        if (!strncmp(p, "\"op\":", 5)) {
            p += 5;
            auto e = strstr(p, ",");
            if (!e) e = strstr(p, "}");
            endFrame = strToFloat(p, nullptr);
            p = e;
            continue;
        }

        //width
        if (!strncmp(p, "\"w\":", 4)) {
            p += 4;
            auto e = strstr(p, ",");
            if (!e) e = strstr(p, "}");
            w = strToFloat(p, nullptr);
            p = e;
            continue;
        }
        //height
        if (!strncmp(p, "\"h\":", 4)) {
            p += 4;
            auto e = strstr(p, ",");
            if (!e) e = strstr(p, "}");
            h = strToFloat(p, nullptr);
            p = e;
            continue;
        }
        ++p;
    }

    if (frameRate < FLT_EPSILON) {
        TVGLOG("LOTTIE", "Not a Lottie file? Frame rate is 0!");
        return false;
    }

    frameCnt = (endFrame - startFrame);
    frameDuration = frameCnt / frameRate;

    TVGLOG("LOTTIE", "info: frame rate = %f, duration = %f size = %f x %f", frameRate, frameDuration, w, h);

    return true;
}


bool LottieLoader::open(const char* data, uint32_t size, bool copy)
{
    if (copy) {
        content = (char*)malloc(size);
        if (!content) return false;
        memcpy((char*)content, data, size);
    } else content = data;

    this->size = size;
    this->copy = copy;

    return header();
}


bool LottieLoader::open(const string& path)
{
    auto f = fopen(path.c_str(), "r");
    if (!f) return false;

    fseek(f, 0, SEEK_END);

    size = ftell(f);
    if (size == 0) {
        fclose(f);
        return false;
    }

    auto content = (char*)(malloc(sizeof(char) * size + 1));
    fseek(f, 0, SEEK_SET);
    auto ret = fread(content, sizeof(char), size, f);
    if (ret < size) {
        fclose(f);
        return false;
    }
    content[size] = '\0';

    fclose(f);

    this->dirName = strDirname(path.c_str());
    this->content = content;
    this->copy = true;

    return header();
}


bool LottieLoader::resize(Paint* paint, float w, float h)
{
    if (!paint) return false;

    auto sx = w / this->w;
    auto sy = h / this->h;
    Matrix m = {sx, 0, 0, 0, sy, 0, 0, 0, 1};
    paint->transform(m);

    //apply the scale to the base clipper
    const Paint* clipper;
    paint->composite(&clipper);
    if (clipper) const_cast<Paint*>(clipper)->transform(m);

    return true;
}


bool LottieLoader::read()
{
    if (!content || size == 0) return false;

    //the loading has been already completed
    if (comp || !LoadModule::read()) return true;

    TaskScheduler::request(this);

    return true;
}


Paint* LottieLoader::paint()
{
    done();

    if (!comp) return nullptr;
    comp->initiated = true;
    return comp->root->scene;
}


bool LottieLoader::override(const char* slot)
{
    if (!comp) done();

    if (!comp || comp->slots.count == 0) return false;

    auto success = true;

    //override slots
    if (slot) {
        //Copy the input data because the JSON parser will encode the data immediately.
        auto temp = strdup(slot);

        //parsing slot json
        LottieParser parser(temp, dirName);

        auto idx = 0;
        while (auto sid = parser.sid(idx == 0)) {
            for (auto s = comp->slots.begin(); s < comp->slots.end(); ++s) {
                if (strcmp((*s)->sid, sid)) continue;
                if (!parser.apply(*s)) success = false;
                break;
            }
            ++idx;
        }

        if (idx < 1) success = false;
        free(temp);
        overriden = success;

    //reset slots
    } else if (overriden) {
        for (auto s = comp->slots.begin(); s < comp->slots.end(); ++s) {
            (*s)->reset();
        }
        overriden = false;
    }
    return success;
}


bool LottieLoader::frame(float no)
{
    //Skip update if frame diff is too small.
    if (fabsf(this->frameNo - no) < 0.0009f) return false;

    this->done();

    //This ensures that the perfect last frame is reached.
    no *= 1000.0f;
    no = roundf(no);
    no *= 0.001f;

    this->frameNo = no + startFrame();

    TaskScheduler::request(this);

    return true;
}


float LottieLoader::startFrame()
{
    return frameCnt * segmentBegin;
}


float LottieLoader::totalFrame()
{
    return (segmentEnd - segmentBegin) * frameCnt;
}


float LottieLoader::curFrame()
{
    return frameNo - startFrame();
}


float LottieLoader::duration()
{
    if (segmentBegin == 0.0f && segmentEnd == 1.0f) return frameDuration;

    if (!comp) done();
    if (!comp) return 0.0f;

    return frameCnt * (segmentEnd - segmentBegin) / comp->frameRate;
}


void LottieLoader::sync()
{
    this->done();
}


uint32_t LottieLoader::markersCnt()
{
    if (!comp) done();
    if (!comp) return 0;
    return comp->markers.count;
}


const char* LottieLoader::markers(uint32_t index)
{
    if (!comp) done();
    if (!comp || index >= comp->markers.count) return nullptr;
    auto marker = comp->markers.begin() + index;
    return (*marker)->name;
}


bool LottieLoader::segment(const char* marker, float& begin, float& end)
{
    if (!comp) done();
    if (!comp) return false;
    
    for (auto m = comp->markers.begin(); m < comp->markers.end(); ++m) {
        if (!strcmp(marker, (*m)->name)) {
            begin = (*m)->time / frameCnt;
            end = ((*m)->time + (*m)->duration) / frameCnt;
            return true;
        }
    }
    return false;
}
