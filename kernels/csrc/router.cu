// Page4 B16R16 GQA4 conditional router for the system benchmarks.
//
// Four entry points cover one decode step: append_decode writes a token's Value latent,
// Base coordinates, and residual code (and the exact Key into mapped host memory when the
// Keys live there); page_lse reconstructs every page's Keys from Base plus residual and
// returns the per-head page log-sum-exp; select_pages turns those into one page list per
// KV head and packs the support tokens; select_pack_refresh does the same and additionally
// maps the support onto a per-row Key slot cache, fetching only the misses from host memory.
//
// Numerics follow the PyTorch reference (inference/routing.py): every BF16 matmul, bias,
// RoPE, and score boundary is rounded to BF16 (round_bfloat16) so the routed pages match.
#include <torch/extension.h>
#include "warp.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "the router kernels use BF16 tensor-core MMA and need sm_80 or newer"
#endif

namespace {
using basiskv::warp_max;
using basiskv::warp_sum;

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kQueriesPerKv = BASIS_GQA;
constexpr int kQueryKeyDim = 128;
constexpr int kHalfHeadDim = kQueryKeyDim / 2;
constexpr int kValueRank = BASIS_VALUE_DIM;
constexpr int kBaseRank = BASIS_BASE_RANK;
constexpr int kResidualRank = BASIS_RESIDUAL_RANK;
constexpr int kPageSize = BASIS_PAGE_SIZE;
constexpr int kScanWarps = 8;  // 16 scored tokens per warp, 32 Page4 pages per block
constexpr int kTokensPerBlock = kScanWarps * 16;
constexpr int kPagesPerBlock = kTokensPerBlock / kPageSize;
constexpr int kMaxSupport = BASIS_ROUTED_TOKENS + BASIS_RECENT_TOKENS;
static_assert(kPageSize == 4 && kQueriesPerKv == 4,
              "the page scan is written for Page4 with four query heads per KV head");
static_assert(kBaseRank == 8 || kBaseRank == 16, "the Base reconstruction MMA covers ranks 8 and 16");
static_assert(kResidualRank > 0 && kResidualRank <= kQueryKeyDim, "residual rank");

#include "page_select.cuh"

__device__ __forceinline__ float page_sum(float value) {
#pragma unroll
  for (int offset = kPageSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, kPageSize);
  }
  return value;
}

__device__ __forceinline__ float page_max(float value) {
#pragma unroll
  for (int offset = kPageSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, kPageSize));
  }
  return value;
}

__device__ __forceinline__ float round_bfloat16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

// query_code[b, h, :] = q[b, h] @ residual_query[h]; one warp per (batch, query head).
__global__ void residual_query_code_kernel(const c10::BFloat16* __restrict__ query,
                                           const c10::BFloat16* __restrict__ residual_query,
                                           c10::BFloat16* __restrict__ query_code,
                                           int64_t query_heads, int64_t query_stride_batch,
                                           int64_t query_stride_head) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / query_heads;
  const int64_t query_head = row % query_heads;
  const int64_t query_base = batch * query_stride_batch + query_head * query_stride_head;
#pragma unroll
  for (int residual = 0; residual < kResidualRank; ++residual) {
    float partial = 0.0f;
#pragma unroll
    for (int feature = lane; feature < kQueryKeyDim; feature += kWarpSize) {
      partial = fmaf(
          static_cast<float>(query[query_base + feature]),
          static_cast<float>(
              residual_query[(query_head * kQueryKeyDim + feature) * kResidualRank + residual]),
          partial);
    }
    const float result = warp_sum(partial);
    if (lane == 0) {
      query_code[row * kResidualRank + residual] = static_cast<c10::BFloat16>(result);
    }
  }
}

// Each warp owns 16 tokens. The Base reconstruction runs as one MMA per 8-feature tile and
// its accumulator is consumed from registers (bias, RoPE, dot with the four queries), so the
// reconstructed K128 page never touches shared memory. Residual scores are added per token,
// then each (page, head) reduces to a log-sum-exp.
__global__ void page_lse_kernel(
    const c10::BFloat16* __restrict__ query, const c10::BFloat16* __restrict__ base_code,
    const c10::BFloat16* __restrict__ residual_code, const c10::BFloat16* __restrict__ base_right,
    const c10::BFloat16* __restrict__ base_bias, const c10::BFloat16* __restrict__ rope_cos,
    const c10::BFloat16* __restrict__ rope_sin, const c10::BFloat16* __restrict__ query_code,
    float* __restrict__ output, int64_t kv_heads, int64_t tokens, int64_t pages,
    int64_t query_stride_batch, int64_t query_stride_head, int64_t base_stride_batch,
    int64_t base_stride_head, int64_t base_stride_token, int64_t residual_stride_batch,
    int64_t residual_stride_head, int64_t residual_stride_token, int64_t rope_stride_token,
    int64_t output_stride_batch, int64_t output_stride_head, int64_t output_stride_query,
    float scale) {
  __shared__ __align__(16) __nv_bfloat16 shared_base[kTokensPerBlock * kBaseRank];
  __shared__ __align__(16) __nv_bfloat16 shared_right[kQueryKeyDim * kBaseRank];
  __shared__ __nv_bfloat16 shared_query[kQueriesPerKv * kQueryKeyDim];
  __shared__ __nv_bfloat16 shared_bias[kQueryKeyDim];
  __shared__ __nv_bfloat16 shared_residual_query[kQueriesPerKv * kResidualRank];
  __shared__ float shared_scores[kQueriesPerKv * kTokensPerBlock];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  const int lane_group = lane / 4;
  const int lane_in_group = lane % 4;
  const int64_t page_groups = (pages + kPagesPerBlock - 1) / kPagesPerBlock;
  const int64_t page_group = static_cast<int64_t>(blockIdx.x) % page_groups;
  const int64_t kv_row = static_cast<int64_t>(blockIdx.x) / page_groups;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t token_start = page_group * kTokensPerBlock;

  for (int index = thread; index < kTokensPerBlock * kBaseRank; index += kThreads) {
    const int token_in_block = index / kBaseRank;
    const int rank = index % kBaseRank;
    const int64_t token = token_start + token_in_block;
    shared_base[index] =
        token < tokens ? static_cast<__nv_bfloat16>(
                             base_code[batch * base_stride_batch + kv_head * base_stride_head +
                                       token * base_stride_token + rank])
                       : __float2bfloat16_rn(0.0f);
  }
  // Keep global reads coalesced, then transpose and RoPE-pair in shared memory.
  for (int index = thread; index < kBaseRank * kQueryKeyDim; index += kThreads) {
    const int rank = index / kQueryKeyDim;
    const int feature = index % kQueryKeyDim;
    const int paired_feature = (feature % kHalfHeadDim) * 2 + feature / kHalfHeadDim;
    shared_right[paired_feature * kBaseRank + rank] =
        static_cast<__nv_bfloat16>(base_right[kv_head * kBaseRank * kQueryKeyDim + index]);
  }
  for (int index = thread; index < kQueriesPerKv * kQueryKeyDim; index += kThreads) {
    const int query_head = index / kQueryKeyDim;
    const int feature = index % kQueryKeyDim;
    shared_query[index] = static_cast<__nv_bfloat16>(
        query[batch * query_stride_batch + (kv_head * kQueriesPerKv + query_head) * query_stride_head +
              feature]);
  }
  for (int index = thread; index < kQueryKeyDim; index += kThreads) {
    shared_bias[index] = static_cast<__nv_bfloat16>(base_bias[kv_head * kQueryKeyDim + index]);
  }
  for (int index = thread; index < kQueriesPerKv * kResidualRank; index += kThreads) {
    shared_residual_query[index] = static_cast<__nv_bfloat16>(
        query_code[(batch * kv_heads + kv_head) * kQueriesPerKv * kResidualRank + index]);
  }
  __syncthreads();

  const int token_zero = warp * 16 + lane_group;
  const int token_one = token_zero + 8;
  const unsigned left_zero =
      *reinterpret_cast<unsigned*>(shared_base + token_zero * kBaseRank + lane_in_group * 2);
  const unsigned left_one =
      *reinterpret_cast<unsigned*>(shared_base + token_one * kBaseRank + lane_in_group * 2);
#if BASIS_BASE_RANK == 16
  const unsigned left_zero_high =
      *reinterpret_cast<unsigned*>(shared_base + token_zero * kBaseRank + lane_in_group * 2 + 8);
  const unsigned left_one_high =
      *reinterpret_cast<unsigned*>(shared_base + token_one * kBaseRank + lane_in_group * 2 + 8);
#endif
  float dot_sums[2][kQueriesPerKv] = {};
#pragma unroll 1
  for (int feature_group = 0; feature_group < 16; ++feature_group) {
    const unsigned right_low = *reinterpret_cast<unsigned*>(
        shared_right + (feature_group * 8 + lane_group) * kBaseRank + lane_in_group * 2);
    float accumulator_zero_low;
    float accumulator_zero_high;
    float accumulator_one_low;
    float accumulator_one_high;
#if BASIS_BASE_RANK == 8
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=f"(accumulator_zero_low), "=f"(accumulator_zero_high), "=f"(accumulator_one_low),
          "=f"(accumulator_one_high)
        : "r"(left_zero), "r"(left_one), "r"(right_low), "f"(0.0f));
#else
    const unsigned right_high = *reinterpret_cast<unsigned*>(
        shared_right + (feature_group * 8 + lane_group) * kBaseRank + lane_in_group * 2 + 8);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(accumulator_zero_low), "=f"(accumulator_zero_high), "=f"(accumulator_one_low),
          "=f"(accumulator_one_high)
        : "r"(left_zero), "r"(left_one), "r"(left_zero_high), "r"(left_one_high), "r"(right_low),
          "r"(right_high), "f"(0.0f));
#endif
    const int feature = feature_group * 4 + lane_in_group;
    const float reconstructed_low[2] = {accumulator_zero_low, accumulator_one_low};
    const float reconstructed_high[2] = {accumulator_zero_high, accumulator_one_high};
#pragma unroll
    for (int token_pair = 0; token_pair < 2; ++token_pair) {
      const int64_t token = token_start + token_zero + token_pair * 8;
      const float biased_low = round_bfloat16(round_bfloat16(reconstructed_low[token_pair]) +
                                              __bfloat162float(shared_bias[feature]));
      const float biased_high =
          round_bfloat16(round_bfloat16(reconstructed_high[token_pair]) +
                         __bfloat162float(shared_bias[feature + kHalfHeadDim]));
      const float cosine =
          token < tokens ? static_cast<float>(rope_cos[token * rope_stride_token + feature]) : 0.0f;
      const float sine =
          token < tokens ? static_cast<float>(rope_sin[token * rope_stride_token + feature]) : 0.0f;
      const float rotated_low =
          round_bfloat16(round_bfloat16(biased_low * cosine) - round_bfloat16(biased_high * sine));
      const float rotated_high =
          round_bfloat16(round_bfloat16(biased_high * cosine) + round_bfloat16(biased_low * sine));
#pragma unroll
      for (int query_head = 0; query_head < kQueriesPerKv; ++query_head) {
        dot_sums[token_pair][query_head] =
            fmaf(rotated_low, __bfloat162float(shared_query[query_head * kQueryKeyDim + feature]),
                 dot_sums[token_pair][query_head]);
        dot_sums[token_pair][query_head] =
            fmaf(rotated_high,
                 __bfloat162float(shared_query[query_head * kQueryKeyDim + feature + kHalfHeadDim]),
                 dot_sums[token_pair][query_head]);
      }
    }
  }
#pragma unroll
  for (int token_pair = 0; token_pair < 2; ++token_pair) {
#pragma unroll
    for (int query_head = 0; query_head < kQueriesPerKv; ++query_head) {
      float score = dot_sums[token_pair][query_head];
      score += __shfl_xor_sync(0xffffffffu, score, 1, 4);
      score += __shfl_xor_sync(0xffffffffu, score, 2, 4);
      if (lane_in_group == 0) {
        shared_scores[query_head * kTokensPerBlock + token_zero + token_pair * 8] = score;
      }
    }
  }
  __syncthreads();

  for (int item = warp; item < kTokensPerBlock / kWarpSize * kQueriesPerKv; item += kScanWarps) {
    const int segment = item / kQueriesPerKv;
    const int query_head = item % kQueriesPerKv;
    const int64_t page = page_group * kPagesPerBlock + (segment * kWarpSize + lane) / kPageSize;
    const int64_t token = token_start + segment * kWarpSize + lane;
    float score = -CUDART_INF_F;
    if (token < tokens) {
      float residual_score = 0.0f;
#pragma unroll
      for (int residual = 0; residual < kResidualRank; ++residual) {
        residual_score =
            fmaf(__bfloat162float(shared_residual_query[query_head * kResidualRank + residual]),
                 static_cast<float>(
                     residual_code[batch * residual_stride_batch + kv_head * residual_stride_head +
                                   token * residual_stride_token + residual]),
                 residual_score);
      }
      score = round_bfloat16(
          round_bfloat16(
              round_bfloat16(
                  shared_scores[query_head * kTokensPerBlock + segment * kWarpSize + lane]) +
              round_bfloat16(residual_score)) *
          scale);
    }
    const float maximum = page_max(score);
    const float sum = page_sum(__expf(score - maximum));
    if (lane % kPageSize == 0 && page < pages) {
      output[batch * output_stride_batch + kv_head * output_stride_head +
             query_head * output_stride_query + page] =
          maximum == -CUDART_INF_F ? -CUDART_INF_F : maximum + __logf(sum);
    }
  }
}

// One block per (batch, KV head): store the Value latent, project it to Base coordinates,
// reconstruct and rotate the predicted Key, encode the residual, and optionally copy the
// exact Key into mapped host memory.
__global__ void append_decode_kernel(
    const c10::BFloat16* __restrict__ key, const c10::BFloat16* __restrict__ value,
    const c10::BFloat16* __restrict__ base_left, const c10::BFloat16* __restrict__ base_right,
    const c10::BFloat16* __restrict__ base_bias, const c10::BFloat16* __restrict__ residual_encoder,
    const c10::BFloat16* __restrict__ rope_cos, const c10::BFloat16* __restrict__ rope_sin,
    c10::BFloat16* __restrict__ value_cache, c10::BFloat16* __restrict__ base_cache,
    c10::BFloat16* __restrict__ residual_cache, c10::BFloat16* __restrict__ host_key,
    int64_t kv_heads, int64_t capacity, int64_t start, int64_t key_stride_batch,
    int64_t key_stride_head, int64_t value_stride_batch, int64_t value_stride_head,
    int64_t value_cache_stride_batch, int64_t value_cache_stride_head,
    int64_t value_cache_stride_token, int64_t base_cache_stride_batch,
    int64_t base_cache_stride_head, int64_t base_cache_stride_token,
    int64_t residual_cache_stride_batch, int64_t residual_cache_stride_head,
    int64_t residual_cache_stride_token) {
  __shared__ __align__(16) __nv_bfloat16 shared_value[kValueRank];
  __shared__ __align__(16) __nv_bfloat16 shared_base[kBaseRank];
  __shared__ __align__(16) __nv_bfloat16 shared_residual[kQueryKeyDim];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t kv_head = row % kv_heads;
  const int64_t batch = row / kv_heads;
  const int64_t value_input_base = batch * value_stride_batch + kv_head * value_stride_head;
  const int64_t key_input_base = batch * key_stride_batch + kv_head * key_stride_head;

  if (host_key != nullptr && thread < kQueryKeyDim / 8) {
    const auto* source = reinterpret_cast<const uint4*>(key + key_input_base);
    auto* destination =
        reinterpret_cast<uint4*>(host_key + (row * capacity + start) * kQueryKeyDim);
    destination[thread] = source[thread];
  }

  for (int feature = thread; feature < kValueRank; feature += kThreads) {
    const c10::BFloat16 loaded = value[value_input_base + feature];
    shared_value[feature] = *reinterpret_cast<const __nv_bfloat16*>(&loaded);
    value_cache[batch * value_cache_stride_batch + kv_head * value_cache_stride_head +
                start * value_cache_stride_token + feature] = loaded;
  }
  __syncthreads();

  if (thread < kBaseRank) {
    float accumulator = 0.0f;
#pragma unroll
    for (int feature = 0; feature < kValueRank; ++feature) {
      accumulator =
          fmaf(__bfloat162float(shared_value[feature]),
               static_cast<float>(base_left[(kv_head * kValueRank + feature) * kBaseRank + thread]),
               accumulator);
    }
    shared_base[thread] = __float2bfloat16_rn(accumulator);
    base_cache[batch * base_cache_stride_batch + kv_head * base_cache_stride_head +
               start * base_cache_stride_token + thread] = static_cast<c10::BFloat16>(accumulator);
  }
  __syncthreads();

  if (thread < kHalfHeadDim) {
    float first_accumulator = 0.0f;
    float second_accumulator = 0.0f;
#pragma unroll
    for (int base = 0; base < kBaseRank; ++base) {
      const float coordinate = __bfloat162float(shared_base[base]);
      first_accumulator =
          fmaf(coordinate,
               static_cast<float>(base_right[(kv_head * kBaseRank + base) * kQueryKeyDim + thread]),
               first_accumulator);
      second_accumulator =
          fmaf(coordinate,
               static_cast<float>(
                   base_right[(kv_head * kBaseRank + base) * kQueryKeyDim + thread + kHalfHeadDim]),
               second_accumulator);
    }
    const float first_pre =
        round_bfloat16(round_bfloat16(first_accumulator) +
                       static_cast<float>(base_bias[kv_head * kQueryKeyDim + thread]));
    const float second_pre = round_bfloat16(
        round_bfloat16(second_accumulator) +
        static_cast<float>(base_bias[kv_head * kQueryKeyDim + thread + kHalfHeadDim]));
    const float cosine = static_cast<float>(rope_cos[thread]);
    const float sine = static_cast<float>(rope_sin[thread]);
    const float first_post =
        round_bfloat16(round_bfloat16(first_pre * cosine) - round_bfloat16(second_pre * sine));
    const float second_post =
        round_bfloat16(round_bfloat16(second_pre * cosine) + round_bfloat16(first_pre * sine));
    shared_residual[thread] = __float2bfloat16_rn(
        round_bfloat16(static_cast<float>(key[key_input_base + thread]) - first_post));
    shared_residual[thread + kHalfHeadDim] = __float2bfloat16_rn(
        round_bfloat16(static_cast<float>(key[key_input_base + thread + kHalfHeadDim]) - second_post));
  }
  __syncthreads();

  for (int residual = warp; residual < kResidualRank; residual += kThreads / kWarpSize) {
    float partial = 0.0f;
#pragma unroll
    for (int feature = lane; feature < kQueryKeyDim; feature += kWarpSize) {
      partial =
          fmaf(__bfloat162float(shared_residual[feature]),
               static_cast<float>(
                   residual_encoder[(kv_head * kQueryKeyDim + feature) * kResidualRank + residual]),
               partial);
    }
    const float result = warp_sum(partial);
    if (lane == 0) {
      residual_cache[batch * residual_cache_stride_batch + kv_head * residual_cache_stride_head +
                     start * residual_cache_stride_token + residual] =
          static_cast<c10::BFloat16>(result);
    }
  }
}

// Map selected tokens to persistent Key slots after Page4 selection.
__global__ void plan_slots_kernel(
    const int64_t* __restrict__ support_ids, int64_t* __restrict__ resident,
    int* __restrict__ lookup, int64_t* __restrict__ slots,
    int* __restrict__ missing, int* __restrict__ counts, int capacity, int support_count) {
  __shared__ uint8_t used[kMaxSupport];
  __shared__ int free_slots[kMaxSupport];
  __shared__ int requested[kMaxSupport];
  __shared__ int free_count, miss_count, hit_count, valid_count;
  const int thread = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  {
    if (thread == 0) {
      free_count = miss_count = hit_count = valid_count = 0;
    }
    for (int index = thread; index < support_count; index += kThreads) {
      used[index] = 0;
    }
    __syncthreads();
    // 1. Support tokens already resident keep their slot.
    for (int index = thread; index < support_count; index += kThreads) {
      const int64_t token = support_ids[row * support_count + index];
      const bool valid = token >= 0 && token < capacity;
      const int slot = valid ? lookup[row * capacity + token] : -1;
      const bool hit = slot >= 0 && slot < support_count && resident[row * support_count + slot] == token;
      requested[index] = hit ? slot : -1;
      if (hit) {
        used[slot] = 1;
        atomicAdd(&hit_count, 1);
      }
      if (valid) {
        atomicAdd(&valid_count, 1);
      }
    }
    __syncthreads();
    // 2. The other slots are free for the misses.
    for (int slot = thread; slot < support_count; slot += kThreads) {
      if (!used[slot]) {
        free_slots[atomicAdd(&free_count, 1)] = slot;
      }
    }
    __syncthreads();
    // 3. Misses take free slots and are listed for the fetch.
    for (int index = thread; index < support_count; index += kThreads) {
      const int64_t token = support_ids[row * support_count + index];
      int slot = requested[index];
      int missing_token = -1;
      if (token >= 0 && token < capacity) {
        if (slot < 0) {
          slot = free_slots[atomicAdd(&miss_count, 1)];
          missing_token = static_cast<int>(token);
        }
        resident[row * support_count + slot] = token;
        lookup[row * capacity + token] = slot;
      } else {
        slot = -1;
      }
      slots[row * support_count + index] = slot;
      missing[row * support_count + index] = missing_token;
    }
    if (thread == 0) {
      counts[row * 2] = hit_count;
      counts[row * 2 + 1] = valid_count;
    }
  }
}

// Copy the missing Keys from mapped host memory into their slots, 16 bytes per thread.
__global__ void fetch_missing_kernel(const __nv_bfloat16* __restrict__ host_key,
                                     __nv_bfloat16* __restrict__ cache,
                                     const int* __restrict__ missing,
                                     const int64_t* __restrict__ slots, int capacity,
                                     int support_count, int rows) {
  constexpr int kVectorsPerKey = kQueryKeyDim / 8;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * support_count * kVectorsPerKey) {
    return;
  }
  const int selected = index / kVectorsPerKey;
  const int vector = index % kVectorsPerKey;
  const int row = selected / support_count;
  const int token = missing[selected];
  if (token >= 0) {
    const int64_t slot = slots[selected];
    const uint4* source = reinterpret_cast<const uint4*>(
        host_key + (static_cast<int64_t>(row) * capacity + token) * kQueryKeyDim);
    uint4* destination = reinterpret_cast<uint4*>(
        cache + (static_cast<int64_t>(row) * support_count + slot) * kQueryKeyDim);
    destination[vector] = source[vector];
  }
}

// ---------------------------------------------------------------------------------------
// Host entry points. Shapes: B batch, G KV heads, H = 4G query heads, T tokens, C capacity.

// CUDA-mapped pinned host storage for exact Keys, [B, G, C, 128] BF16 owned by a CPU tensor.
at::Tensor host_key_buffer(int64_t batch, int64_t kv_heads, int64_t capacity) {
  const size_t elements = static_cast<size_t>(batch) * kv_heads * capacity * kQueryKeyDim;
  void* pointer = nullptr;
  const cudaError_t status =
      cudaHostAlloc(&pointer, elements * sizeof(c10::BFloat16), cudaHostAllocMapped);
  assert(status == cudaSuccess);
  return at::from_blob(
      pointer, {batch, kv_heads, capacity, kQueryKeyDim},
      [](void* allocation) { cudaFreeHost(allocation); },
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
}

// Device alias of a host_key_buffer allocation; kernels read and write it directly.
int64_t device_pointer(const at::Tensor& host_key) {
  assert(host_key.device().is_cpu() && host_key.scalar_type() == at::kBFloat16);
  void* pointer = nullptr;
  const cudaError_t status = cudaHostGetDevicePointer(&pointer, host_key.data_ptr(), 0);
  assert(status == cudaSuccess);
  return reinterpret_cast<int64_t>(pointer);
}

// Copy key [B, G, n, 128] (CUDA) into host_key[:, :, start:start + n] on the current stream.
at::Tensor append_host_key(const at::Tensor& host_key, const at::Tensor& key, int64_t start) {
  assert(host_key.device().is_cpu() && key.is_cuda());
  assert(host_key.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16);
  assert(host_key.dim() == 4 && key.dim() == 4 && key.stride(3) == 1);
  assert(host_key.size(0) == key.size(0) && host_key.size(1) == key.size(1));
  assert(host_key.size(3) == key.size(3) && start >= 0 && start + key.size(2) <= host_key.size(2));

  c10::cuda::CUDAGuard guard(key.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  auto* destination = host_key.mutable_data_ptr<c10::BFloat16>();
  const auto* source = key.const_data_ptr<c10::BFloat16>();
  const size_t row_bytes = static_cast<size_t>(key.size(3)) * sizeof(c10::BFloat16);
  const size_t destination_pitch = static_cast<size_t>(host_key.stride(2)) * sizeof(c10::BFloat16);
  const size_t source_pitch = static_cast<size_t>(key.stride(2)) * sizeof(c10::BFloat16);
  for (int64_t batch = 0; batch < key.size(0); ++batch) {
    for (int64_t head = 0; head < key.size(1); ++head) {
      const int64_t destination_offset =
          batch * host_key.stride(0) + head * host_key.stride(1) + start * host_key.stride(2);
      const int64_t source_offset = batch * key.stride(0) + head * key.stride(1);
      const cudaError_t status = cudaMemcpy2DAsync(
          destination + destination_offset, destination_pitch, source + source_offset, source_pitch,
          row_bytes, static_cast<size_t>(key.size(2)), cudaMemcpyDeviceToHost, stream);
      assert(status == cudaSuccess);
    }
  }
  return host_key;
}

// query [B, H, 1, 128]; base_code [B, G, T, BASE_RANK]; residual_code [B, G, T, RESIDUAL_RANK];
// base_right [G, BASE_RANK, 128]; base_bias [G, 128]; residual_query [H, 128, RESIDUAL_RANK];
// rope_cos / rope_sin [T, 64]; query_code [B, G, 4, RESIDUAL_RANK] scratch;
// output [B, G, 4, ceil(T / 4)] FP32 page log-sum-exp.
at::Tensor page_lse(const at::Tensor& query, const at::Tensor& base_code,
                    const at::Tensor& residual_code, const at::Tensor& base_right,
                    const at::Tensor& base_bias, const at::Tensor& residual_query,
                    const at::Tensor& rope_cos, const at::Tensor& rope_sin,
                    const at::Tensor& query_code, const at::Tensor& output, double scale) {
  const int64_t batch = base_code.size(0), kv_heads = base_code.size(1), tokens = base_code.size(2);
  const int64_t pages = (tokens + kPageSize - 1) / kPageSize;
  for (const at::Tensor* tensor : {&query, &base_code, &residual_code, &base_right, &base_bias,
                                   &residual_query, &rope_cos, &rope_sin, &query_code}) {
    assert(tensor->is_cuda() && tensor->scalar_type() == at::kBFloat16);
  }
  assert(output.is_cuda() && output.scalar_type() == at::kFloat);
  assert(query.sizes() == at::IntArrayRef({batch, kv_heads * kQueriesPerKv, 1, kQueryKeyDim}));
  assert(base_code.dim() == 4 && base_code.size(3) == kBaseRank);
  assert(residual_code.sizes() == at::IntArrayRef({batch, kv_heads, tokens, kResidualRank}));
  assert(base_right.sizes() == at::IntArrayRef({kv_heads, kBaseRank, kQueryKeyDim}));
  assert(base_bias.sizes() == at::IntArrayRef({kv_heads, kQueryKeyDim}));
  assert(residual_query.sizes() == at::IntArrayRef({query.size(1), kQueryKeyDim, kResidualRank}));
  assert(rope_cos.sizes() == at::IntArrayRef({tokens, kHalfHeadDim}) && rope_sin.sizes() == rope_cos.sizes());
  assert(query_code.sizes() == at::IntArrayRef({batch, kv_heads, kQueriesPerKv, kResidualRank}));
  assert(output.sizes() == at::IntArrayRef({batch, kv_heads, kQueriesPerKv, pages}));
  assert(query.stride(3) == 1 && base_code.stride(3) == 1 && residual_code.stride(3) == 1);
  assert(rope_cos.stride(1) == 1 && rope_sin.stride(1) == 1 && output.stride(3) == 1);
  assert(base_right.is_contiguous() && base_bias.is_contiguous() && residual_query.is_contiguous());
  assert(query_code.is_contiguous() && std::isfinite(scale) && scale > 0.0);

  c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  residual_query_code_kernel<<<static_cast<unsigned int>(batch * query.size(1)), kWarpSize, 0,
                               stream>>>(query.const_data_ptr<c10::BFloat16>(),
                                         residual_query.const_data_ptr<c10::BFloat16>(),
                                         query_code.mutable_data_ptr<c10::BFloat16>(),
                                         query.size(1), query.stride(0), query.stride(1));
  page_lse_kernel<<<static_cast<unsigned int>(batch * kv_heads *
                                              ((pages + kPagesPerBlock - 1) / kPagesPerBlock)),
                    kScanWarps * kWarpSize, 0, stream>>>(
      query.const_data_ptr<c10::BFloat16>(), base_code.const_data_ptr<c10::BFloat16>(),
      residual_code.const_data_ptr<c10::BFloat16>(), base_right.const_data_ptr<c10::BFloat16>(),
      base_bias.const_data_ptr<c10::BFloat16>(), rope_cos.const_data_ptr<c10::BFloat16>(),
      rope_sin.const_data_ptr<c10::BFloat16>(), query_code.const_data_ptr<c10::BFloat16>(),
      output.mutable_data_ptr<float>(), kv_heads, tokens, pages, query.stride(0), query.stride(1),
      base_code.stride(0), base_code.stride(1), base_code.stride(2), residual_code.stride(0),
      residual_code.stride(1), residual_code.stride(2), rope_cos.stride(0), output.stride(0),
      output.stride(1), output.stride(2), static_cast<float>(scale));
  assert(cudaGetLastError() == cudaSuccess);
  return output;
}

// key [B, G, 1, 128]; value [B, G, 1, VALUE_DIM]; base_left [G, VALUE_DIM, BASE_RANK];
// base_right [G, BASE_RANK, 128]; base_bias [G, 128]; residual_encoder [G, 128, RESIDUAL_RANK];
// rope_cos / rope_sin [1, 64] for this position; caches [B, G, C, width] written at `start`;
// host_key_pointer is a device_pointer() of a [B, G, C, 128] buffer, or 0.
void append_decode(const at::Tensor& key, const at::Tensor& value, const at::Tensor& base_left,
                   const at::Tensor& base_right, const at::Tensor& base_bias,
                   const at::Tensor& residual_encoder, const at::Tensor& rope_cos,
                   const at::Tensor& rope_sin, const at::Tensor& value_cache,
                   const at::Tensor& base_cache, const at::Tensor& residual_cache, int64_t start,
                   int64_t host_key_pointer) {
  const int64_t batch = key.size(0), kv_heads = key.size(1), capacity = value_cache.size(2);
  for (const at::Tensor* tensor : {&key, &value, &base_left, &base_right, &base_bias,
                                   &residual_encoder, &rope_cos, &rope_sin, &value_cache,
                                   &base_cache, &residual_cache}) {
    assert(tensor->is_cuda() && tensor->scalar_type() == at::kBFloat16);
  }
  assert(key.sizes() == at::IntArrayRef({batch, kv_heads, 1, kQueryKeyDim}));
  assert(value.sizes() == at::IntArrayRef({batch, kv_heads, 1, kValueRank}));
  assert(base_left.sizes() == at::IntArrayRef({kv_heads, kValueRank, kBaseRank}));
  assert(base_right.sizes() == at::IntArrayRef({kv_heads, kBaseRank, kQueryKeyDim}));
  assert(base_bias.sizes() == at::IntArrayRef({kv_heads, kQueryKeyDim}));
  assert(residual_encoder.sizes() == at::IntArrayRef({kv_heads, kQueryKeyDim, kResidualRank}));
  assert(rope_cos.sizes() == at::IntArrayRef({1, kHalfHeadDim}) && rope_sin.sizes() == rope_cos.sizes());
  assert(value_cache.sizes() == at::IntArrayRef({batch, kv_heads, capacity, kValueRank}));
  assert(base_cache.sizes() == at::IntArrayRef({batch, kv_heads, capacity, kBaseRank}));
  assert(residual_cache.sizes() == at::IntArrayRef({batch, kv_heads, capacity, kResidualRank}));
  assert(start >= 0 && start < capacity);
  assert(key.stride(3) == 1 && value.stride(3) == 1 && rope_cos.stride(1) == 1 && rope_sin.stride(1) == 1);
  assert(value_cache.stride(3) == 1 && base_cache.stride(3) == 1 && residual_cache.stride(3) == 1);
  assert(base_left.is_contiguous() && base_right.is_contiguous());
  assert(base_bias.is_contiguous() && residual_encoder.is_contiguous());

  c10::cuda::CUDAGuard guard(key.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  append_decode_kernel<<<static_cast<unsigned int>(batch * kv_heads), kThreads, 0, stream>>>(
      key.const_data_ptr<c10::BFloat16>(), value.const_data_ptr<c10::BFloat16>(),
      base_left.const_data_ptr<c10::BFloat16>(), base_right.const_data_ptr<c10::BFloat16>(),
      base_bias.const_data_ptr<c10::BFloat16>(), residual_encoder.const_data_ptr<c10::BFloat16>(),
      rope_cos.const_data_ptr<c10::BFloat16>(), rope_sin.const_data_ptr<c10::BFloat16>(),
      value_cache.mutable_data_ptr<c10::BFloat16>(), base_cache.mutable_data_ptr<c10::BFloat16>(),
      residual_cache.mutable_data_ptr<c10::BFloat16>(),
      reinterpret_cast<c10::BFloat16*>(host_key_pointer), kv_heads, capacity, start,
      key.stride(0), key.stride(1), value.stride(0), value.stride(1), value_cache.stride(0),
      value_cache.stride(1), value_cache.stride(2), base_cache.stride(0), base_cache.stride(1),
      base_cache.stride(2), residual_cache.stride(0), residual_cache.stride(1),
      residual_cache.stride(2));
  assert(cudaGetLastError() == cudaSuccess);
}

// Scores [B,G,4,pages] are consumed: the selector reuses head 0 for candidate keys.
// support_ids [B,G,2048+64] pads tokens beyond an incomplete last page with -1.
void select_pages(const at::Tensor& page_log_mass, const at::Tensor& pages_output,
                  const at::Tensor& support_ids, const at::Tensor& scratch,
                  int64_t historical, int64_t end) {
  const int64_t batch = page_log_mass.size(0), kv_heads = page_log_mass.size(1);
  const int pages = static_cast<int>(page_log_mass.size(3));
  const int64_t rows = batch * kv_heads;
  assert(page_log_mass.is_cuda() && page_log_mass.scalar_type() == at::kFloat);
  assert(page_log_mass.dim() == 4 && page_log_mass.size(2) == kQueriesPerKv);
  assert(page_log_mass.stride(3) == 1);
  for (const at::Tensor* tensor : {&pages_output, &support_ids}) {
    assert(tensor->is_cuda() && tensor->is_contiguous() && tensor->scalar_type() == at::kLong);
  }
  assert(scratch.is_cuda() && scratch.is_contiguous() && scratch.scalar_type() == at::kFloat);
  assert(scratch.numel() == rows * kScratchWords);
  assert(historical > BASIS_ROUTED_TOKENS && end - historical == kRecentTokens);
  assert(pages == (historical + kPageSize - 1) / kPageSize && pages <= kMaxGroups * kGroupPages);
  assert(pages_output.sizes() == at::IntArrayRef({batch, kv_heads, kRoutedPages}));
  assert(support_ids.sizes() == at::IntArrayRef({batch, kv_heads, kMaxSupport}));
  const int blocks_per_row = (pages + kGroupThreads * kGroupPages - 1) / (kGroupThreads * kGroupPages);
  c10::cuda::CUDAGuard guard(page_log_mass.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(page_log_mass.get_device()).stream();
  page_group_statistics_kernel<<<static_cast<unsigned int>(rows * blocks_per_row), kGroupThreads, 0, stream>>>(
      page_log_mass.const_data_ptr<float>(), scratch.mutable_data_ptr<float>(), kv_heads, pages,
      blocks_per_row, page_log_mass.stride(0), page_log_mass.stride(1), page_log_mass.stride(2));
  select_pages_grouped_kernel<<<static_cast<unsigned int>(rows), kSelectThreads, 0, stream>>>(
      page_log_mass.mutable_data_ptr<float>(), scratch.mutable_data_ptr<float>(),
      pages_output.mutable_data_ptr<int64_t>(), support_ids.mutable_data_ptr<int64_t>(),
      historical, end, kv_heads, pages, blocks_per_row,
      page_log_mass.stride(0), page_log_mass.stride(1), page_log_mass.stride(2));
  assert(cudaGetLastError() == cudaSuccess);
}

// select_pages plus the Key slot cache refresh. host_key_pointer addresses [B, G, C, 128];
// cache [B, G, support, 128] holds the resident Keys by slot; resident [B, G, support] int64
// and lookup [B, G, C] int32 persist across steps (start at -1); slots and missing
// [B, G, support] and counts [B, G, 2] (hits, valid) are this step's outputs.
void select_pack_refresh(const at::Tensor& page_log_mass, const at::Tensor& pages_output,
                         const at::Tensor& support_ids, int64_t host_key_pointer,
                         const at::Tensor& cache, const at::Tensor& resident,
                         const at::Tensor& lookup, const at::Tensor& slots,
                         const at::Tensor& missing, const at::Tensor& counts, const at::Tensor& scratch,
                         int64_t historical, int64_t end) {
  const int64_t rows = page_log_mass.size(0) * page_log_mass.size(1);
  const int64_t support_count = support_ids.size(2), capacity = lookup.size(2);
  for (const at::Tensor* tensor : {&cache, &resident, &lookup, &slots, &missing, &counts}) {
    assert(tensor->is_cuda() && tensor->is_contiguous());
  }
  assert(host_key_pointer != 0 && capacity >= end);
  assert(cache.scalar_type() == at::kBFloat16 && resident.scalar_type() == at::kLong);
  assert(lookup.scalar_type() == at::kInt && slots.scalar_type() == at::kLong);
  assert(missing.scalar_type() == at::kInt && counts.scalar_type() == at::kInt);
  assert(cache.numel() == rows * support_count * kQueryKeyDim && resident.numel() == rows * support_count);
  assert(slots.numel() == rows * support_count && missing.numel() == rows * support_count);
  assert(counts.numel() == rows * 2);

  select_pages(page_log_mass, pages_output, support_ids, scratch, historical, end);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(page_log_mass.get_device()).stream();
  plan_slots_kernel<<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
      support_ids.const_data_ptr<int64_t>(), resident.mutable_data_ptr<int64_t>(),
      lookup.mutable_data_ptr<int>(), slots.mutable_data_ptr<int64_t>(),
      missing.mutable_data_ptr<int>(), counts.mutable_data_ptr<int>(),
      static_cast<int>(capacity), static_cast<int>(support_count));
  assert(cudaGetLastError() == cudaSuccess);
  const int vectors = static_cast<int>(rows * support_count) * (kQueryKeyDim / 8);
  fetch_missing_kernel<<<(vectors + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(host_key_pointer),
      reinterpret_cast<__nv_bfloat16*>(cache.mutable_data_ptr<at::BFloat16>()),
      missing.const_data_ptr<int>(), slots.const_data_ptr<int64_t>(), static_cast<int>(capacity),
      static_cast<int>(support_count), static_cast<int>(rows));
  assert(cudaGetLastError() == cudaSuccess);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  namespace py = pybind11;
  module.doc() = "Page4 B16R16 GQA4 conditional router: append, page scan, selection, slot refresh";
  module.def("selection_workspace_size", []() { return kScratchWords; });
  module.def("host_key_buffer", &host_key_buffer, py::arg("batch"), py::arg("kv_heads"),
             py::arg("capacity"));
  module.def("device_pointer", &device_pointer, py::arg("host_key"));
  module.def("append_host_key", &append_host_key, py::arg("host_key"), py::arg("key"),
             py::arg("start"));
  module.def("page_lse", &page_lse, py::arg("query"), py::arg("base_code"),
             py::arg("residual_code"), py::arg("base_right"), py::arg("base_bias"),
             py::arg("residual_query"), py::arg("rope_cos"), py::arg("rope_sin"),
             py::arg("query_code"), py::arg("output"), py::arg("scale"));
  module.def("append_decode", &append_decode, py::arg("key"), py::arg("value"),
             py::arg("base_left"), py::arg("base_right"), py::arg("base_bias"),
             py::arg("residual_encoder"), py::arg("rope_cos"), py::arg("rope_sin"),
             py::arg("value_cache"), py::arg("base_cache"), py::arg("residual_cache"),
             py::arg("start"), py::arg("host_key_pointer") = 0);
  module.def("select_pages", &select_pages, py::arg("page_log_mass"), py::arg("pages"),
             py::arg("support_ids"), py::arg("scratch"), py::arg("historical"), py::arg("end"));
  module.def("select_pack_refresh", &select_pack_refresh, py::arg("page_log_mass"),
             py::arg("pages"), py::arg("support_ids"), py::arg("host_key_pointer"),
             py::arg("cache"), py::arg("resident"), py::arg("lookup"), py::arg("slots"),
             py::arg("missing"), py::arg("counts"), py::arg("scratch"),
             py::arg("historical"), py::arg("end"));
}
