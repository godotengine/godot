using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;

namespace GodotTools.Export
{
    public class AotCache
    {
        private readonly string _cacheFilePath;
        private readonly Cache _cache = new Cache();
        private bool _hasUnsavedChanges = false;

        public AotCache(string cacheFilePath)
        {
            _cacheFilePath = cacheFilePath;

            if (File.Exists(_cacheFilePath))
                LoadCache(_cacheFilePath, out _cache);
        }

        private static byte[] ComputeSha256Checksum(string filePath)
        {
            using (var sha256 = SHA256.Create())
            {
                using (var streamReader = File.OpenRead(filePath))
                    return sha256.ComputeHash(streamReader);
            }
        }

        private static bool CompareHashes(byte[] a, byte[] b)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }

        private class Cache
        {
            [JsonProperty("assemblies")]
            public Dictionary<string, CachedChecksums> Assemblies { get; set; } =
                new Dictionary<string, CachedChecksums>();
        }

        private struct CachedChecksums
        {
            [JsonProperty("input_checksum")] public string InputChecksumBase64 { get; set; }
            [JsonProperty("output_checksum")] public string OutputChecksumBase64 { get; set; }
        }

        private static void LoadCache(string cacheFilePath, out Cache cache)
        {
            using (var streamReader = new StreamReader(cacheFilePath, Encoding.UTF8))
            using (var jsonReader = new JsonTextReader(streamReader))
            {
                cache = new JsonSerializer().Deserialize<Cache>(jsonReader);
            }
        }

        private static void SaveCache(string cacheFilePath, Cache cache)
        {
            using (var streamWriter = new StreamWriter(cacheFilePath, append: false, Encoding.UTF8))
            using (var jsonWriter = new JsonTextWriter(streamWriter))
            {
                new JsonSerializer().Serialize(jsonWriter, cache);
            }
        }

        private bool TryGetCachedChecksums(string name, out CachedChecksums cachedChecksums)
            => _cache.Assemblies.TryGetValue(name, out cachedChecksums);

        private void ChangeCache(string name, byte[] inputChecksum, byte[] outputChecksum)
        {
            _cache.Assemblies[name] = new CachedChecksums()
            {
                InputChecksumBase64 = Convert.ToBase64String(inputChecksum),
                OutputChecksumBase64 = Convert.ToBase64String(outputChecksum)
            };
            _hasUnsavedChanges = true;
        }

        public void SaveCache()
        {
            if (!_hasUnsavedChanges)
                return;
            SaveCache(_cacheFilePath, _cache);
            _hasUnsavedChanges = false;
        }

        private bool IsCached(string name, byte[] inputChecksum, string output)
        {
            if (!File.Exists(output))
            {
                return false;
            }

            if (!TryGetCachedChecksums(name, out var cachedChecksums))
                return false;

            if (string.IsNullOrEmpty(cachedChecksums.InputChecksumBase64) ||
                string.IsNullOrEmpty(cachedChecksums.OutputChecksumBase64))
                return false;

            var cachedInputChecksum = Convert.FromBase64String(cachedChecksums.InputChecksumBase64);

            if (!CompareHashes(inputChecksum, cachedInputChecksum))
                return false;

            var outputChecksum = ComputeSha256Checksum(output);
            var cachedOutputChecksum = Convert.FromBase64String(cachedChecksums.OutputChecksumBase64);

            if (!CompareHashes(outputChecksum, cachedOutputChecksum))
                return false;

            return true;
        }

        public void RunCached(string name, string input, string output, Action action)
        {
            var inputChecksum = ComputeSha256Checksum(input);

            if (IsCached(name, inputChecksum, output))
            {
                Console.WriteLine($"AOT compiler cache: '{name}' already compiled.");
                return;
            }

            action();

            var outputChecksum = ComputeSha256Checksum(output);

            ChangeCache(name, inputChecksum, outputChecksum);
        }
    }
}
