#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;

// ─── Eviction policies ───

enum llama_moe_policy_type {
    LLAMA_MOE_CACHE_LRU  = 0,
    LLAMA_MOE_CACHE_SLRU = 1,
};

// ─── Config ───

struct llama_moe_cache_config {
    int  n_cache_experts   = 0;     // 0 = disabled; max experts kept resident
    int  eviction_policy   = LLAMA_MOE_CACHE_LRU;
    bool print_stats       = false;
    std::string warmup_profile_path;
};

// ─── Per-expert tensor info (populated at model load) ───
// Points into the mmap'd region — no file I/O needed.

struct llama_expert_tensor_info {
    void * base_addr     = nullptr;  // tensor->data (start of expert 0)
    size_t expert_stride = 0;        // tensor->nb[2] (bytes per expert)
    int64_t n_expert     = 0;        // tensor->ne[2]
};

// ─── Stats (thread-safe) ───

struct llama_moe_cache_stats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> willneed_calls{0};
    std::atomic<uint64_t> dontneed_calls{0};

    double hit_rate() const;
    void   print() const;
    void   reset();
};

// ─── Main cache class ───
//
// Uses madvise(WILLNEED/DONTNEED) to control which expert tensor pages
// are resident in physical memory. Tracks expert usage via an LRU list
// and evicts cold experts when the number of resident experts exceeds
// the configured budget (n_cache_experts).
//
// Shared experts (ffn_*_shexp) are separate tensors and are NOT tracked
// by this cache — they are always fully resident.

class llama_moe_cache {
public:
    bool init(const llama_moe_cache_config & config,
              const llama_model & model);

    // Called after model tensors are loaded to set per-layer expert tensor info
    void set_layer_tensors(int layer,
                           const std::vector<llama_expert_tensor_info> & infos);

    // Called after all set_layer_tensors() to validate and log
    bool finalize();

    void shutdown();

    // Called after graph compute with the selected expert IDs for a layer.
    // Marks experts as recently used (WILLNEED) and updates the LRU.
    // expert_ids may contain duplicates; n_ids = n_expert_used * n_tokens.
    void on_experts_used(int layer, const int32_t * expert_ids, int n_ids);

    // Evict cold experts beyond the cache budget (DONTNEED).
    void evict_cold();

    // DONTNEED all expert pages — call after model load to release pages
    // that were touched during load_all_data(). The cache/profile will
    // then page in only what's actually needed.
    void release_all_expert_pages();

    // Batch mode: pre-fetch all unique experts for a layer at once.
    void acquire_batch(int layer, const int32_t * expert_ids, int n_ids);

    // Warm-up: WILLNEED for top-N experts per layer.
    void warmup_topn(int n_per_layer);

    // Profile persistence: save/load hot expert set to avoid cold starts.
    // File is a small JSON (~few KB).
    void save_profile(const std::string & path) const;
    void load_profile(const std::string & path);

    // Prefetch: issue WILLNEED for experts without updating LRU/stats.
    // Thread-safe (madvise is thread-safe, only reads immutable layer data).
    void prefetch_experts(int layer, const int32_t * expert_ids, int n_ids);

    // Get top-N predicted experts for a layer based on access history.
    std::vector<int32_t> get_predicted_experts(int layer, int n_top);

    // Pre-graph: issue WILLNEED for top predicted experts across all layers.
    void prefetch_predicted_all_layers(int n_per_layer);

    bool has_profile_data() const { return !access_counts_.empty(); }
    int  get_n_layers() const { return n_layers_; }
    int  get_n_experts_per_layer() const { return n_experts_per_layer_; }

    bool enabled() const { return n_cache_experts_ > 0; }

    const llama_moe_cache_stats & stats() const { return stats_; }

    std::string profile_path;  // if non-empty, save profile on shutdown

private:
    int n_cache_experts_      = 0;
    int n_experts_per_layer_  = 0;
    int n_layers_             = 0;

    // Per-layer per-expert tensor metadata
    struct layer_info {
        std::vector<llama_expert_tensor_info> tensors; // gate[_up], [up], down
    };
    std::vector<layer_info> layers_;

    int policy_ = LLAMA_MOE_CACHE_LRU;

    // LRU tracking: key = layer * n_experts_per_layer + expert
    std::list<uint64_t> lru_order_;                                      // front = MRU
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_map_;

    // SLRU tracking (two-segment LRU)
    std::list<uint64_t> slru_protected_;       // front = MRU, 80% capacity
    std::list<uint64_t> slru_probationary_;    // front = MRU, 20% capacity
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> slru_map_;
    std::unordered_set<uint64_t> slru_in_protected_;
    int protected_capacity_    = 0;
    int probationary_capacity_ = 0;

    // Per-expert access counts (for profile save)
    std::unordered_map<uint64_t, uint64_t> access_counts_;

    mutable std::mutex mutex_;
    llama_moe_cache_stats stats_;

    // Dispatch: branches on policy_
    void record_access(uint64_t key);    // touch or slru_touch
    bool is_cached(uint64_t key) const;  // check if key is in cache

    // LRU internals
    void touch(uint64_t key);

    // SLRU internals
    void slru_touch(uint64_t key);       // promote/move within segments
    void slru_add_new(uint64_t key);     // insert into probationary
    void slru_evict_cold();              // evict from probationary first

    void advise_willneed(int layer, int expert);
    void advise_dontneed(int layer, int expert);
};
