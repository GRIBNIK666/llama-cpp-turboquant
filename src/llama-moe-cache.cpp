#include "llama-moe-cache.h"
#include "llama-model.h"
#include "llama-impl.h"

#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <unistd.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

// ═══════════════════════════════════════════════════
// Stats
// ═══════════════════════════════════════════════════

double llama_moe_cache_stats::hit_rate() const {
    uint64_t total = hits.load() + misses.load();
    return total > 0 ? (double)hits.load() / (double)total : 0.0;
}

void llama_moe_cache_stats::print() const {
    LLAMA_LOG_INFO("\n=== MoE Cache Stats ===\n");
    LLAMA_LOG_INFO("  hits:       %8" PRIu64 "\n", hits.load());
    LLAMA_LOG_INFO("  misses:     %8" PRIu64 "\n", misses.load());
    LLAMA_LOG_INFO("  evictions:  %8" PRIu64 "\n", evictions.load());
    LLAMA_LOG_INFO("  hit_rate:   %8.2f%%\n", hit_rate() * 100.0);
    LLAMA_LOG_INFO("  willneed:   %8" PRIu64 "\n", willneed_calls.load());
    LLAMA_LOG_INFO("  dontneed:   %8" PRIu64 "\n", dontneed_calls.load());
    LLAMA_LOG_INFO("=======================\n");
}

void llama_moe_cache_stats::reset() {
    hits = 0;
    misses = 0;
    evictions = 0;
    willneed_calls = 0;
    dontneed_calls = 0;
}

// ═══════════════════════════════════════════════════
// Platform-specific madvise
// ═══════════════════════════════════════════════════

static long page_size() {
    static long ps = 0;
    if (ps == 0) {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        ps = si.dwPageSize;
#else
        ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) ps = 4096;
#endif
    }
    return ps;
}

static void moe_advise_willneed(void * addr, size_t len) {
    if (!addr || len == 0) return;

    long ps = page_size();
    uintptr_t start = (uintptr_t)addr & ~(uintptr_t)(ps - 1);
    uintptr_t end   = ((uintptr_t)addr + len + ps - 1) & ~(uintptr_t)(ps - 1);
    size_t aligned_len = end - start;

#if defined(__linux__) || defined(__APPLE__)
    int ret = posix_madvise((void *)start, aligned_len, POSIX_MADV_WILLNEED);
    if (ret != 0) {
        LLAMA_LOG_DEBUG("moe_cache: posix_madvise(WILLNEED) failed: %s\n", strerror(ret));
    }
#elif defined(_WIN32)
    WIN32_MEMORY_RANGE_ENTRY entry;
    entry.VirtualAddress = (void *)start;
    entry.NumberOfBytes  = aligned_len;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#endif
}

static void moe_advise_dontneed(void * addr, size_t len) {
    if (!addr || len == 0) return;

    long ps = page_size();
    // For DONTNEED, round inward to avoid affecting adjacent experts' pages
    uintptr_t start = ((uintptr_t)addr + ps - 1) & ~(uintptr_t)(ps - 1);
    uintptr_t end   = ((uintptr_t)addr + len) & ~(uintptr_t)(ps - 1);
    if (end <= start) return;
    size_t aligned_len = end - start;

#if defined(__APPLE__)
    // POSIX_MADV_DONTNEED is advisory-only on macOS; use MADV_FREE for actual release
    madvise((void *)start, aligned_len, MADV_FREE);
#elif defined(__linux__)
    posix_madvise((void *)start, aligned_len, POSIX_MADV_DONTNEED);
#elif defined(_WIN32)
    DiscardVirtualMemory((void *)start, aligned_len);
#endif
}

// ═══════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════

bool llama_moe_cache::init(const llama_moe_cache_config & config,
                           const llama_model & model) {
    const auto & hparams = model.hparams;

    if (hparams.n_expert == 0) {
        LLAMA_LOG_WARN("%s: model has no experts, MoE cache disabled\n", __func__);
        return false;
    }

    n_cache_experts_     = config.n_cache_experts;
    n_experts_per_layer_ = hparams.n_expert;
    n_layers_            = hparams.n_layer();
    policy_              = config.eviction_policy;

    if (policy_ == LLAMA_MOE_CACHE_SLRU) {
        protected_capacity_    = n_cache_experts_ * 4 / 5;  // 80%
        probationary_capacity_ = n_cache_experts_ - protected_capacity_;
    }

    layers_.resize(n_layers_);

    // Per-layer tensor info will be populated by set_layer_tensors() after model load
    stats_.reset();
    return true;
}

void llama_moe_cache::set_layer_tensors(int layer,
                                        const std::vector<llama_expert_tensor_info> & infos) {
    if (layer < 0 || layer >= n_layers_) return;
    layers_[layer].tensors = infos;
}

bool llama_moe_cache::finalize() {
    int layers_with_experts = 0;
    for (int il = 0; il < n_layers_; il++) {
        if (!layers_[il].tensors.empty()) {
            layers_with_experts++;
        }
    }

    if (layers_with_experts == 0) {
        LLAMA_LOG_WARN("%s: no MoE tensor metadata found, cache disabled\n", __func__);
        return false;
    }

    size_t expert_bytes = 0;
    for (int il = 0; il < n_layers_; il++) {
        if (layers_[il].tensors.empty()) continue;
        for (const auto & t : layers_[il].tensors) {
            expert_bytes += t.expert_stride;
        }
        break; // all layers have same size
    }

    LLAMA_LOG_INFO("%s: MoE cache: %d experts budget, %d experts/layer, %d MoE layers, policy=%s\n",
                   __func__, n_cache_experts_, n_experts_per_layer_, layers_with_experts,
                   policy_ == LLAMA_MOE_CACHE_SLRU ? "slru" : "lru");
    LLAMA_LOG_INFO("%s: MoE cache: ~%.2f MB per expert\n",
                   __func__, (double)expert_bytes / (1024.0 * 1024.0));

    return true;
}

void llama_moe_cache::shutdown() {
    if (!profile_path.empty()) {
        save_profile(profile_path);
    }
    stats_.print();
}

// ═══════════════════════════════════════════════════
// LRU touch
// ═══════════════════════════════════════════════════

void llama_moe_cache::touch(uint64_t key) {
    // Must be called under lock
    auto it = lru_map_.find(key);
    if (it != lru_map_.end()) {
        lru_order_.erase(it->second);
    }
    lru_order_.push_front(key);
    lru_map_[key] = lru_order_.begin();
}

// ═══════════════════════════════════════════════════
// SLRU internals (must be called under lock)
// ═══════════════════════════════════════════════════

void llama_moe_cache::slru_touch(uint64_t key) {
    auto it = slru_map_.find(key);
    if (it == slru_map_.end()) return;

    if (slru_in_protected_.count(key)) {
        // Already in protected — move to front
        slru_protected_.erase(it->second);
        slru_protected_.push_front(key);
        slru_map_[key] = slru_protected_.begin();
    } else {
        // In probationary — promote to protected
        slru_probationary_.erase(it->second);
        slru_protected_.push_front(key);
        slru_in_protected_.insert(key);
        slru_map_[key] = slru_protected_.begin();

        // If protected overflows, demote tail to probationary
        if ((int)slru_protected_.size() > protected_capacity_ && !slru_protected_.empty()) {
            uint64_t demoted = slru_protected_.back();
            slru_protected_.pop_back();
            slru_in_protected_.erase(demoted);
            slru_probationary_.push_front(demoted);
            slru_map_[demoted] = slru_probationary_.begin();
        }
    }
}

void llama_moe_cache::slru_add_new(uint64_t key) {
    slru_probationary_.push_front(key);
    slru_map_[key] = slru_probationary_.begin();
}

void llama_moe_cache::slru_evict_cold() {
    int total = (int)(slru_protected_.size() + slru_probationary_.size());
    while (total > n_cache_experts_) {
        if (!slru_probationary_.empty()) {
            uint64_t key = slru_probationary_.back();
            slru_probationary_.pop_back();
            slru_map_.erase(key);

            int layer  = (int)(key / n_experts_per_layer_);
            int expert = (int)(key % n_experts_per_layer_);
            advise_dontneed(layer, expert);
            stats_.evictions.fetch_add(1);
            total--;
        } else if (!slru_protected_.empty()) {
            // Demote protected tail to probationary (not a true eviction)
            uint64_t demoted = slru_protected_.back();
            slru_protected_.pop_back();
            slru_in_protected_.erase(demoted);
            slru_probationary_.push_front(demoted);
            slru_map_[demoted] = slru_probationary_.begin();
            // total unchanged — next iteration will evict from probationary
        } else {
            break;
        }
    }
}

// ═══════════════════════════════════════════════════
// Policy dispatch (must be called under lock)
// ═══════════════════════════════════════════════════

void llama_moe_cache::record_access(uint64_t key) {
    if (policy_ == LLAMA_MOE_CACHE_SLRU) {
        if (slru_map_.count(key)) {
            slru_touch(key);
        } else {
            slru_add_new(key);
        }
    } else {
        touch(key);
    }
}

bool llama_moe_cache::is_cached(uint64_t key) const {
    if (policy_ == LLAMA_MOE_CACHE_SLRU) {
        return slru_map_.count(key) > 0;
    }
    return lru_map_.count(key) > 0;
}

// ═══════════════════════════════════════════════════
// madvise wrappers
// ═══════════════════════════════════════════════════

void llama_moe_cache::advise_willneed(int layer, int expert) {
    if (layer < 0 || layer >= n_layers_) return;
    const auto & li = layers_[layer];
    for (const auto & t : li.tensors) {
        if (!t.base_addr || expert >= t.n_expert) continue;
        void * addr = (char *)t.base_addr + (size_t)expert * t.expert_stride;
        moe_advise_willneed(addr, t.expert_stride);
    }
    stats_.willneed_calls.fetch_add(1);
}

void llama_moe_cache::advise_dontneed(int layer, int expert) {
    if (layer < 0 || layer >= n_layers_) return;
    const auto & li = layers_[layer];
    for (const auto & t : li.tensors) {
        if (!t.base_addr || expert >= t.n_expert) continue;
        void * addr = (char *)t.base_addr + (size_t)expert * t.expert_stride;
        moe_advise_dontneed(addr, t.expert_stride);
    }
    stats_.dontneed_calls.fetch_add(1);
}

// ═══════════════════════════════════════════════════
// on_experts_used — called after graph compute
// ═══════════════════════════════════════════════════

void llama_moe_cache::on_experts_used(int layer, const int32_t * expert_ids, int n_ids) {
    if (n_cache_experts_ <= 0 || !expert_ids || n_ids <= 0) return;

    std::set<int32_t> unique_ids(expert_ids, expert_ids + n_ids);

    std::lock_guard<std::mutex> lock(mutex_);

    for (int32_t eid : unique_ids) {
        if (eid < 0 || eid >= n_experts_per_layer_) continue;

        uint64_t key = (uint64_t)layer * n_experts_per_layer_ + eid;
        access_counts_[key]++;

        bool was_cached = is_cached(key);
        record_access(key);

        if (was_cached) {
            stats_.hits.fetch_add(1);
        } else {
            stats_.misses.fetch_add(1);
            advise_willneed(layer, eid);
        }
    }
}

// ═══════════════════════════════════════════════════
// evict_cold — DONTNEED beyond budget
// ═══════════════════════════════════════════════════

void llama_moe_cache::evict_cold() {
    if (n_cache_experts_ <= 0) return;

    std::lock_guard<std::mutex> lock(mutex_);

    if (policy_ == LLAMA_MOE_CACHE_SLRU) {
        slru_evict_cold();
    } else {
        while ((int)lru_order_.size() > n_cache_experts_) {
            uint64_t key = lru_order_.back();
            lru_order_.pop_back();
            lru_map_.erase(key);

            int layer  = (int)(key / n_experts_per_layer_);
            int expert = (int)(key % n_experts_per_layer_);
            advise_dontneed(layer, expert);
            stats_.evictions.fetch_add(1);
        }
    }
}

// ═══════════════════════════════════════════════════
// Release all expert pages (post-load cleanup)
// ═══════════════════════════════════════════════════

void llama_moe_cache::release_all_expert_pages() {
    LLAMA_LOG_INFO("%s: releasing all expert pages via DONTNEED\n", __func__);
    for (int il = 0; il < n_layers_; il++) {
        if (layers_[il].tensors.empty()) continue;
        for (int ie = 0; ie < n_experts_per_layer_; ie++) {
            advise_dontneed(il, ie);
        }
    }
    LLAMA_LOG_INFO("%s: done\n", __func__);
}

// ═══════════════════════════════════════════════════
// Batch acquire (PP mode)
// ═══════════════════════════════════════════════════

void llama_moe_cache::acquire_batch(int layer, const int32_t * expert_ids, int n_ids) {
    if (n_cache_experts_ <= 0 || !expert_ids || n_ids <= 0) return;

    std::set<int32_t> unique_ids(expert_ids, expert_ids + n_ids);

    // If most experts are needed, just WILLNEED all of them
    if ((int)unique_ids.size() >= n_experts_per_layer_ * 4 / 5) {
        for (int e = 0; e < n_experts_per_layer_; e++) {
            unique_ids.insert(e);
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);

    for (int32_t eid : unique_ids) {
        if (eid < 0 || eid >= n_experts_per_layer_) continue;

        uint64_t key = (uint64_t)layer * n_experts_per_layer_ + eid;
        access_counts_[key]++;

        bool was_cached = is_cached(key);
        record_access(key);

        if (was_cached) {
            stats_.hits.fetch_add(1);
        } else {
            stats_.misses.fetch_add(1);
            advise_willneed(layer, eid);
        }
    }
}

// ═══════════════════════════════════════════════════
// Prefetch (no LRU update, thread-safe)
// ═══════════════════════════════════════════════════

void llama_moe_cache::prefetch_experts(int layer, const int32_t * expert_ids, int n_ids) {
    if (!expert_ids || n_ids <= 0) return;

    std::set<int32_t> unique_ids(expert_ids, expert_ids + n_ids);
    for (int32_t eid : unique_ids) {
        if (eid >= 0 && eid < n_experts_per_layer_) {
            advise_willneed(layer, eid);
        }
    }
}

std::vector<int32_t> llama_moe_cache::get_predicted_experts(int layer, int n_top) {
    // Extract access counts for this layer, sort by count descending
    std::vector<std::pair<int32_t, uint64_t>> layer_counts;

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & [key, count] : access_counts_) {
        int l = (int)(key / n_experts_per_layer_);
        int e = (int)(key % n_experts_per_layer_);
        if (l == layer) {
            layer_counts.push_back({e, count});
        }
    }

    std::sort(layer_counts.begin(), layer_counts.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    std::vector<int32_t> result;
    int n = std::min(n_top, (int)layer_counts.size());
    for (int i = 0; i < n; i++) {
        result.push_back(layer_counts[i].first);
    }
    return result;
}

void llama_moe_cache::prefetch_predicted_all_layers(int n_per_layer) {
    for (int il = 0; il < n_layers_; il++) {
        if (layers_[il].tensors.empty()) continue;
        auto predicted = get_predicted_experts(il, n_per_layer);
        if (!predicted.empty()) {
            prefetch_experts(il, predicted.data(), (int)predicted.size());
        }
    }
}

// ═══════════════════════════════════════════════════
// Warm-up
// ═══════════════════════════════════════════════════

void llama_moe_cache::warmup_topn(int n_per_layer) {
    LLAMA_LOG_INFO("%s: warming up top-%d experts per layer\n", __func__, n_per_layer);

    std::lock_guard<std::mutex> lock(mutex_);

    for (int il = 0; il < n_layers_; il++) {
        if (layers_[il].tensors.empty()) continue;
        int n = std::min(n_per_layer, n_experts_per_layer_);
        for (int ie = 0; ie < n; ie++) {
            uint64_t key = (uint64_t)il * n_experts_per_layer_ + ie;
            record_access(key);
            advise_willneed(il, ie);
        }
    }

    LLAMA_LOG_INFO("%s: warm-up complete\n", __func__);
}

// ═══════════════════════════════════════════════════
// Profile save/load
// ═══════════════════════════════════════════════════

void llama_moe_cache::save_profile(const std::string & path) const {
    if (path.empty()) return;

    // Sort by access count descending
    std::vector<std::pair<uint64_t, uint64_t>> entries(access_counts_.begin(), access_counts_.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    std::ofstream f(path);
    if (!f) {
        LLAMA_LOG_WARN("%s: failed to save profile to %s\n", __func__, path.c_str());
        return;
    }

    f << "{\n";
    f << "  \"n_experts_per_layer\": " << n_experts_per_layer_ << ",\n";
    f << "  \"n_layers\": " << n_layers_ << ",\n";
    f << "  \"experts\": [\n";

    bool first = true;
    for (const auto & [key, count] : entries) {
        if (count == 0) continue;
        int layer  = (int)(key / n_experts_per_layer_);
        int expert = (int)(key % n_experts_per_layer_);
        if (!first) f << ",\n";
        f << "    {\"l\":" << layer << ",\"e\":" << expert << ",\"c\":" << count << "}";
        first = false;
    }

    f << "\n  ]\n}\n";
    f.close();

    LLAMA_LOG_INFO("%s: saved profile (%zu experts) to %s\n",
                   __func__, entries.size(), path.c_str());
}

void llama_moe_cache::load_profile(const std::string & path) {
    if (path.empty()) return;

    std::ifstream f(path);
    if (!f) {
        LLAMA_LOG_INFO("%s: no profile at %s, starting cold\n", __func__, path.c_str());
        return;
    }

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Simple JSON parsing — find all {"l":N,"e":N,"c":N} entries
    // Sort by count descending and WILLNEED up to n_cache_experts_
    struct entry { int layer; int expert; uint64_t count; };
    std::vector<entry> entries;

    size_t pos = 0;
    while ((pos = content.find("\"l\":", pos)) != std::string::npos) {
        entry e = {};
        if (sscanf(content.c_str() + pos, "\"l\":%d,\"e\":%d,\"c\":%" SCNu64,
                   &e.layer, &e.expert, &e.count) == 3) {
            entries.push_back(e);
        }
        pos += 4;
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto & a, const auto & b) { return a.count > b.count; });

    std::lock_guard<std::mutex> lock(mutex_);

    int loaded = 0;
    for (const auto & e : entries) {
        if (loaded >= n_cache_experts_) break;
        if (e.layer < 0 || e.layer >= n_layers_) continue;
        if (e.expert < 0 || e.expert >= n_experts_per_layer_) continue;

        uint64_t key = (uint64_t)e.layer * n_experts_per_layer_ + e.expert;
        record_access(key);
        advise_willneed(e.layer, e.expert);
        loaded++;
    }

    LLAMA_LOG_INFO("%s: loaded profile from %s, pre-fetched %d/%zu experts\n",
                   __func__, path.c_str(), loaded, entries.size());
}
