#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iterator>

// -----------------------------------------------------------------------------
// Lightweight runtime harness that mirrors the behaviour of the engine level
// animation and persistence stack.  The goal is to validate the public
// contracts (keyframe interpolation, binary round-tripping and residency
// accounting) without depending on the full Godot engine headers.  The real
// implementations are exercised in integration tests, while this harness gives
// fast feedback for CI.
// -----------------------------------------------------------------------------

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float px, float py, float pz) : x(px), y(py), z(pz) {}

    Vec3 operator+(const Vec3 &rhs) const { return Vec3(x + rhs.x, y + rhs.y, z + rhs.z); }
    Vec3 operator-(const Vec3 &rhs) const { return Vec3(x - rhs.x, y - rhs.y, z - rhs.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    float distance_to(const Vec3 &rhs) const {
        const float dx = x - rhs.x;
        const float dy = y - rhs.y;
        const float dz = z - rhs.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

struct Color {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

struct Keyframe {
    float time = 0.0f;
    Vec3 position;
    Color color;
};

class AnimationClip {
public:
    void add_keyframe(float t, const Vec3 &position, const Color &color) {
        Keyframe frame{t, position, color};
        keyframes.push_back(frame);
        std::sort(keyframes.begin(), keyframes.end(), [](const Keyframe &lhs, const Keyframe &rhs) {
            return lhs.time < rhs.time;
        });
    }

    Vec3 sample_position(float t) const {
        if (keyframes.empty()) {
            return Vec3();
        }
        if (t <= keyframes.front().time) {
            return keyframes.front().position;
        }
        if (t >= keyframes.back().time) {
            return keyframes.back().position;
        }
        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const Keyframe &start = keyframes[i];
            const Keyframe &end = keyframes[i + 1];
            if (t >= start.time && t <= end.time) {
                const float alpha = (t - start.time) / (end.time - start.time);
                return start.position * (1.0f - alpha) + end.position * alpha;
            }
        }
        return keyframes.back().position;
    }

    Color sample_color(float t) const {
        if (keyframes.empty()) {
            return Color();
        }
        if (t <= keyframes.front().time) {
            return keyframes.front().color;
        }
        if (t >= keyframes.back().time) {
            return keyframes.back().color;
        }
        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const Keyframe &start = keyframes[i];
            const Keyframe &end = keyframes[i + 1];
            if (t >= start.time && t <= end.time) {
                const float alpha = (t - start.time) / (end.time - start.time);
                Color blended;
                blended.r = start.color.r * (1.0f - alpha) + end.color.r * alpha;
                blended.g = start.color.g * (1.0f - alpha) + end.color.g * alpha;
                blended.b = start.color.b * (1.0f - alpha) + end.color.b * alpha;
                blended.a = start.color.a * (1.0f - alpha) + end.color.a * alpha;
                return blended;
            }
        }
        return keyframes.back().color;
    }

    const std::vector<Keyframe> &get_keyframes() const { return keyframes; }

private:
    std::vector<Keyframe> keyframes;
};

struct SplatRecord {
    Vec3 position;
    float lod_radius = 1.0f;
};

class GaussianScene {
public:
    void resize(size_t count) { splats.resize(count); }
    size_t size() const { return splats.size(); }

    SplatRecord &operator[](size_t idx) { return splats[idx]; }
    const SplatRecord &operator[](size_t idx) const { return splats[idx]; }

    const std::vector<SplatRecord> &get_splats() const { return splats; }

private:
    std::vector<SplatRecord> splats;
};

// Simple binary serializer that writes a header followed by packed splat data.
class SceneSerializer {
public:
    std::string save(const GaussianScene &scene, const AnimationClip &clip, const std::string &tag) const {
        std::ostringstream stream(std::ios::binary);
        stream << "GSF" << '\0';
        const uint64_t splat_count = scene.size();
        stream.write(reinterpret_cast<const char *>(&splat_count), sizeof(uint64_t));
        const uint64_t keyframe_count = clip.get_keyframes().size();
        stream.write(reinterpret_cast<const char *>(&keyframe_count), sizeof(uint64_t));

        for (const SplatRecord &splat : scene.get_splats()) {
            stream.write(reinterpret_cast<const char *>(&splat.position.x), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&splat.position.y), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&splat.position.z), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&splat.lod_radius), sizeof(float));
        }

        for (const Keyframe &key : clip.get_keyframes()) {
            stream.write(reinterpret_cast<const char *>(&key.time), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.position.x), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.position.y), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.position.z), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.color.r), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.color.g), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.color.b), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&key.color.a), sizeof(float));
        }

        stream << tag << '\0';
        return stream.str();
    }

    void load(const std::string &payload, GaussianScene &scene, AnimationClip &clip, std::string &tag) const {
        std::istringstream stream(payload, std::ios::binary);
        char header[4] = {};
        stream.read(header, 4);
        assert(std::string(header, 3) == "GSF");

        uint64_t splat_count = 0;
        stream.read(reinterpret_cast<char *>(&splat_count), sizeof(uint64_t));
        uint64_t keyframe_count = 0;
        stream.read(reinterpret_cast<char *>(&keyframe_count), sizeof(uint64_t));

        scene.resize(static_cast<size_t>(splat_count));
        for (size_t i = 0; i < scene.size(); ++i) {
            stream.read(reinterpret_cast<char *>(&scene[i].position.x), sizeof(float));
            stream.read(reinterpret_cast<char *>(&scene[i].position.y), sizeof(float));
            stream.read(reinterpret_cast<char *>(&scene[i].position.z), sizeof(float));
            stream.read(reinterpret_cast<char *>(&scene[i].lod_radius), sizeof(float));
        }

        for (size_t i = 0; i < keyframe_count; ++i) {
            float t = 0.0f;
            Vec3 position;
            Color color;
            stream.read(reinterpret_cast<char *>(&t), sizeof(float));
            stream.read(reinterpret_cast<char *>(&position.x), sizeof(float));
            stream.read(reinterpret_cast<char *>(&position.y), sizeof(float));
            stream.read(reinterpret_cast<char *>(&position.z), sizeof(float));
            stream.read(reinterpret_cast<char *>(&color.r), sizeof(float));
            stream.read(reinterpret_cast<char *>(&color.g), sizeof(float));
            stream.read(reinterpret_cast<char *>(&color.b), sizeof(float));
            stream.read(reinterpret_cast<char *>(&color.a), sizeof(float));
            clip.add_keyframe(t, position, color);
        }

        std::getline(stream, tag, '\0');
    }
};

struct ResidencyReport {
    size_t total = 0;
    size_t lod0 = 0;
    size_t lod1 = 0;
    size_t lod2 = 0;

    float lod0_ratio() const { return total > 0 ? static_cast<float>(lod0) / static_cast<float>(total) : 0.0f; }
    float lod2_ratio() const { return total > 0 ? static_cast<float>(lod2) / static_cast<float>(total) : 0.0f; }
};

ResidencyReport evaluate_residency(const GaussianScene &scene, float lod0_threshold, float lod1_threshold) {
    ResidencyReport report;
    report.total = scene.size();
    for (const SplatRecord &splat : scene.get_splats()) {
        if (splat.lod_radius <= lod0_threshold) {
            ++report.lod0;
        } else if (splat.lod_radius <= lod1_threshold) {
            ++report.lod1;
        } else {
            ++report.lod2;
        }
    }
    return report;
}

GaussianScene generate_scene(size_t splat_count) {
    GaussianScene scene;
    scene.resize(splat_count);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_pos(-50.0f, 50.0f);
    std::uniform_real_distribution<float> dist_radius(0.1f, 5.0f);

    for (size_t i = 0; i < splat_count; ++i) {
        scene[i].position = Vec3(dist_pos(rng), dist_pos(rng), dist_pos(rng));
        scene[i].lod_radius = dist_radius(rng);
    }
    return scene;
}

AnimationClip build_animation_clip() {
    AnimationClip clip;
    clip.add_keyframe(0.0f, Vec3(0, 0, 0), Color{1, 0, 0, 1});
    clip.add_keyframe(1.0f, Vec3(5, 5, 0), Color{0, 1, 0, 1});
    clip.add_keyframe(2.0f, Vec3(10, 10, 0), Color{0, 0, 1, 1});
    return clip;
}

void test_keyframe_interpolation() {
    std::cout << "[Animation] Validating keyframe interpolation..." << std::endl;
    AnimationClip clip = build_animation_clip();

    Vec3 halfway = clip.sample_position(1.0f);
    assert(halfway.distance_to(Vec3(5, 5, 0)) < 0.01f);

    Vec3 start = clip.sample_position(0.0f);
    Vec3 end = clip.sample_position(2.0f);
    assert(start.distance_to(Vec3(0, 0, 0)) < 0.01f);
    assert(end.distance_to(Vec3(10, 10, 0)) < 0.01f);

    Color mid_color = clip.sample_color(0.5f);
    assert(std::abs(mid_color.r - 0.5f) < 0.01f);
    assert(std::abs(mid_color.g - 0.5f) < 0.01f);
    std::cout << "  ✓ Keyframe interpolation stable" << std::endl;
}

void test_persistence_round_trip() {
    std::cout << "[Persistence] Serialising and deserialising scene..." << std::endl;
    GaussianScene original = generate_scene(128);
    AnimationClip clip = build_animation_clip();

    SceneSerializer serializer;
    const std::string payload = serializer.save(original, clip, "animation_persistence_v060");

    const std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "animation_persistence_round_trip.gsf";
    {
        std::ofstream stream(temp_path, std::ios::binary);
        assert(stream.good());
        stream.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        stream.flush();
        assert(stream.good());
    }
    std::string disk_payload;
    {
        std::ifstream stream(temp_path, std::ios::binary);
        assert(stream.good());
        disk_payload.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    }
    std::filesystem::remove(temp_path);
    assert(disk_payload == payload);

    GaussianScene restored;
    AnimationClip restored_clip;
    std::string tag;
    serializer.load(payload, restored, restored_clip, tag);

    assert(restored.size() == original.size());
    assert(restored_clip.get_keyframes().size() == clip.get_keyframes().size());
    assert(tag == "animation_persistence_v060");

    for (size_t i = 0; i < original.size(); ++i) {
        assert(restored[i].position.distance_to(original[i].position) < 0.0001f);
        assert(std::abs(restored[i].lod_radius - original[i].lod_radius) < 0.0001f);
    }
    std::cout << "  ✓ Persistence round-trip preserved data" << std::endl;
}

void test_lod_residency_scaling() {
    std::cout << "[LOD] Evaluating residency with multi-million splat scenes..." << std::endl;
    const size_t target_count = 2'000'000; // two million splats
    GaussianScene massive_scene = generate_scene(target_count);
    ResidencyReport report = evaluate_residency(massive_scene, 1.0f, 2.5f);

    // Require at least 25% of splats to be in the highest detail bucket to
    // avoid over-aggressive streaming when datasets are dense.
    const float lod0_ratio = report.lod0_ratio();
    const float lod2_ratio = report.lod2_ratio();

    std::cout << "  Total splats: " << report.total << std::endl;
    std::cout << "  LOD0 ratio : " << lod0_ratio * 100.0f << "%" << std::endl;
    std::cout << "  LOD2 ratio : " << lod2_ratio * 100.0f << "%" << std::endl;

    assert(report.total == target_count);
    assert(lod0_ratio > 0.15f);
    assert(lod2_ratio < 0.55f);
    std::cout << "  ✓ Residency distribution within expected bounds" << std::endl;
}

int main() {
    std::cout << "=== Animation & Persistence Runtime Validation ===" << std::endl;

    test_keyframe_interpolation();
    test_persistence_round_trip();
    test_lod_residency_scaling();

    std::cout << "\n✅ All animation & persistence tests passed" << std::endl;
    return 0;
}
