// Decode attention over selected pages of a GPU-resident K128 cache with compact Values.
//
// One block per (batch, KV head, split); one warp per query head of the group. Pages are
// streamed through shared memory in tiles of four tokens with an online softmax. With more
// than one split the blocks write FP32 partials that reduce_splits_kernel merges.
#include <torch/extension.h>
#include "warp.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace {
using basiskv::warp_max;
using basiskv::warp_sum;

constexpr int kWarpSize = 32;
constexpr int kQueriesPerKv = BASIS_GQA;
constexpr int kThreads = kQueriesPerKv * kWarpSize;
constexpr int kQueryKeyDim = 128;
constexpr int kValueDim = BASIS_VALUE_DIM;
constexpr int kPageSize = BASIS_PAGE_SIZE;
constexpr int kTokenTile = kPageSize < 4 ? kPageSize : 4;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kElementsPerVector = kVectorBytes / sizeof(uint16_t);
constexpr int kKeyVectors = kQueryKeyDim / kElementsPerVector;
constexpr int kValueRegisters = (kValueDim + kWarpSize - 1) / kWarpSize;

__device__ __forceinline__ float2 bf16_pair(uint32_t bits) {
  return make_float2(__bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(bits))),
                     __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(bits >> 16))));
}

__device__ __forceinline__ uint32_t vector_word(uint4 value, int index) {
  switch (index) {
    case 0:
      return value.x;
    case 1:
      return value.y;
    case 2:
      return value.z;
    default:
      return value.w;
  }
}

__device__ __forceinline__ float vector_dot(const c10::BFloat16* query, const c10::BFloat16* key) {
  const uint4 query_vector = *reinterpret_cast<const uint4*>(query);
  const uint4 key_vector = *reinterpret_cast<const uint4*>(key);
  float result = 0.0f;
#pragma unroll
  for (int pair = 0; pair < kElementsPerVector / 2; ++pair) {
    const float2 query_pair = bf16_pair(vector_word(query_vector, pair));
    const float2 key_pair = bf16_pair(vector_word(key_vector, pair));
    result = fmaf(query_pair.x, key_pair.x, result);
    result = fmaf(query_pair.y, key_pair.y, result);
  }
  return result;
}

template <bool kDirectOutput>
__global__ void paged_attention_kernel(
    const c10::BFloat16* __restrict__ key, const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ value, const int64_t* __restrict__ page_ids,
    float* __restrict__ workspace, c10::BFloat16* __restrict__ output, int64_t kv_heads,
    int64_t capacity, int64_t sequence_length, int64_t page_slots, int64_t splits,
    int64_t query_stride_batch, int64_t query_stride_head, int64_t value_stride_batch,
    int64_t value_stride_head, int64_t value_stride_token, int64_t page_stride_batch,
    int64_t page_stride_head, int64_t page_stride_slot, int64_t workspace_stride_row,
    int64_t workspace_stride_split, int64_t output_stride_batch, int64_t output_stride_head,
    int64_t output_stride_feature, float scale) {
  __shared__ __align__(kVectorBytes) c10::BFloat16 shared_query[kQueriesPerKv][kQueryKeyDim];
  __shared__ __align__(kVectorBytes) c10::BFloat16 shared_key[kTokenTile][kQueryKeyDim];
  __shared__ __align__(kVectorBytes) c10::BFloat16 shared_value[kTokenTile][kValueDim];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int split = static_cast<int>(linear % splits);
  const int64_t kv_row = linear / splits;
  const int64_t kv_head = kv_row % kv_heads;
  const int64_t batch = kv_row / kv_heads;
  const int64_t query_head = kv_head * kQueriesPerKv + warp;
  const int64_t query_heads = kv_heads * kQueriesPerKv;

  const int query_vector = static_cast<int>(threadIdx.x);
  if (query_vector < kQueriesPerKv * kKeyVectors) {
    const int head = query_vector / kKeyVectors;
    const int vector = query_vector % kKeyVectors;
    const int feature = vector * kElementsPerVector;
    const int64_t source =
        batch * query_stride_batch + (kv_head * kQueriesPerKv + head) * query_stride_head + feature;
    reinterpret_cast<uint4*>(shared_query)[query_vector] =
        *reinterpret_cast<const uint4*>(query + source);
  }
  __syncthreads();

  float accumulator[kValueRegisters] = {};
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  for (int64_t page_slot = split; page_slot < page_slots; page_slot += splits) {
    const int64_t page_id = page_ids[batch * page_stride_batch + kv_head * page_stride_head +
                                     page_slot * page_stride_slot];
    if (page_id < 0) {
      continue;
    }
    const int64_t page_start = page_id * kPageSize;
    for (int tile_start = 0; tile_start < kPageSize; tile_start += kTokenTile) {
      for (int key_vector = threadIdx.x; key_vector < kTokenTile * kKeyVectors;
           key_vector += blockDim.x) {
        const int token = key_vector / kKeyVectors;
        const int vector = key_vector % kKeyVectors;
        const int64_t source_token = page_start + tile_start + token;
        uint4 loaded = make_uint4(0, 0, 0, 0);
        if (source_token < sequence_length) {
          const int64_t source =
              ((batch * kv_heads + kv_head) * capacity + source_token) * kQueryKeyDim +
              vector * kElementsPerVector;
          loaded = *reinterpret_cast<const uint4*>(key + source);
        }
        reinterpret_cast<uint4*>(shared_key)[key_vector] = loaded;
      }
      for (int index = threadIdx.x; index < kTokenTile * kValueDim; index += blockDim.x) {
        const int token = index / kValueDim;
        const int feature = index % kValueDim;
        const int64_t source_token = page_start + tile_start + token;
        shared_value[token][feature] =
            source_token < sequence_length
                ? value[batch * value_stride_batch + kv_head * value_stride_head +
                        source_token * value_stride_token + feature]
                : c10::BFloat16(0.0f);
      }
      __syncthreads();

#pragma unroll
      for (int token = 0; token < kTokenTile; ++token) {
        if (page_start + tile_start + token < sequence_length) {
          float dot = 0.0f;
          if (lane < kKeyVectors) {
            dot = vector_dot(shared_query[warp] + lane * kElementsPerVector,
                             shared_key[token] + lane * kElementsPerVector);
          }
          dot = warp_sum(dot);
          const float score = dot * scale;
          const float next_max = fmaxf(running_max, score);
          const float previous_scale = __expf(running_max - next_max);
          const float probability = __expf(score - next_max);
#pragma unroll
          for (int slot = 0; slot < kValueRegisters; ++slot) {
            const int feature = lane + slot * kWarpSize;
            if (feature < kValueDim) {
              accumulator[slot] =
                  fmaf(probability, static_cast<float>(shared_value[token][feature]),
                       accumulator[slot] * previous_scale);
            }
          }
          running_sum = running_sum * previous_scale + probability;
          running_max = next_max;
        }
      }
      __syncthreads();
    }
  }

  if constexpr (kDirectOutput) {
    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    const int64_t output_base = batch * output_stride_batch + query_head * output_stride_head;
#pragma unroll
    for (int slot = 0; slot < kValueRegisters; ++slot) {
      const int feature = lane + slot * kWarpSize;
      if (feature < kValueDim)
        output[output_base + feature * output_stride_feature] =
            static_cast<c10::BFloat16>(accumulator[slot] * inverse_sum);
    }
  } else {
    const int64_t row = batch * query_heads + query_head;
    const int64_t base = row * workspace_stride_row + split * workspace_stride_split;
#pragma unroll
    for (int slot = 0; slot < kValueRegisters; ++slot) {
      const int feature = lane + slot * kWarpSize;
      if (feature < kValueDim)
        workspace[base + feature] = accumulator[slot];
    }
    if (lane == 0) {
      workspace[base + kValueDim] = running_max;
      workspace[base + kValueDim + 1] = running_sum;
    }
  }
}

__global__ void reduce_splits_kernel(const float* __restrict__ workspace,
                                     c10::BFloat16* __restrict__ output, int64_t splits,
                                     int64_t query_heads, int64_t workspace_stride_row,
                                     int64_t workspace_stride_split, int64_t output_stride_batch,
                                     int64_t output_stride_head, int64_t output_stride_feature) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / query_heads;
  const int64_t query_head = row % query_heads;
  const int64_t row_base = row * workspace_stride_row;

  float local_max = -CUDART_INF_F;
  for (int split = lane; split < splits; split += kWarpSize) {
    local_max = fmaxf(local_max, workspace[row_base + split * workspace_stride_split + kValueDim]);
  }
  const float maximum = warp_max(local_max);
  float local_sum = 0.0f;
  for (int split = lane; split < splits; split += kWarpSize) {
    const int64_t base = row_base + split * workspace_stride_split;
    local_sum =
        fmaf((workspace[base + kValueDim + 1] > 0.0f ? __expf(workspace[base + kValueDim] - maximum)
                                                     : 0.0f),
             workspace[base + kValueDim + 1], local_sum);
  }
  const float total = warp_sum(local_sum);

  for (int feature = lane; feature < kValueDim; feature += kWarpSize) {
    float combined = 0.0f;
    for (int split = 0; split < splits; ++split) {
      const int64_t base = row_base + split * workspace_stride_split;
      combined = fmaf(
          (workspace[base + kValueDim + 1] > 0.0f ? __expf(workspace[base + kValueDim] - maximum)
                                                  : 0.0f),
          workspace[base + feature], combined);
    }
    const int64_t output_base = batch * output_stride_batch + query_head * output_stride_head;
    output[output_base + feature * output_stride_feature] =
        static_cast<c10::BFloat16>(total > 0.0f ? combined / total : 0.0f);
  }
}

// key [batch, kv_heads, capacity, 128] contiguous; query [batch, kv_heads * GQA, 1, 128];
// value [batch, kv_heads, >= sequence_length, VALUE_DIM]; page_ids [batch, kv_heads, slots]
// with negative IDs as padding; workspace [batch * query_heads, >= splits, VALUE_DIM + 2]
// FP32; output [batch, query_heads, 1, VALUE_DIM]. Empty support returns zeros.
at::Tensor attention(const at::Tensor& key, const at::Tensor& query, const at::Tensor& value,
                     const at::Tensor& page_ids, const at::Tensor& workspace,
                     const at::Tensor& output, int64_t sequence_length, double scale,
                     int64_t splits) {
  assert(key.is_cuda() && query.is_cuda() && value.is_cuda() && page_ids.is_cuda());
  assert(key.scalar_type() == at::kBFloat16 && query.scalar_type() == at::kBFloat16);
  assert(value.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16);
  assert(page_ids.scalar_type() == at::kLong && workspace.scalar_type() == at::kFloat);
  assert(key.is_contiguous() && key.dim() == 4 && key.size(3) == kQueryKeyDim);
  assert(query.dim() == 4 && query.size(2) == 1 && query.size(3) == kQueryKeyDim);
  assert(query.size(0) == key.size(0) && query.size(1) == kQueriesPerKv * key.size(1));
  assert(value.size(0) == key.size(0) && value.size(1) == key.size(1) && value.size(3) == kValueDim);
  assert(sequence_length > 0 && sequence_length <= key.size(2) && sequence_length <= value.size(2));
  assert(page_ids.size(0) == key.size(0) && page_ids.size(1) == key.size(1));
  assert(splits > 0 && splits <= page_ids.size(2));
  assert(workspace.size(0) == query.size(0) * query.size(1) && workspace.size(1) >= splits);
  assert(workspace.size(2) == kValueDim + 2);
  assert(output.sizes() == at::IntArrayRef({query.size(0), query.size(1), 1, kValueDim}));
  assert(query.stride(3) == 1 && value.stride(3) == 1 && output.stride(3) == 1);

  c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  const int64_t kv_rows = key.size(0) * key.size(1);
  const int64_t query_heads = query.size(1);
  const auto launch = [&](auto direct) {
    paged_attention_kernel<decltype(direct)::value>
        <<<static_cast<unsigned int>(kv_rows * splits), kThreads, 0, stream>>>(
            key.const_data_ptr<c10::BFloat16>(), query.const_data_ptr<c10::BFloat16>(),
            value.const_data_ptr<c10::BFloat16>(), page_ids.const_data_ptr<int64_t>(),
            workspace.mutable_data_ptr<float>(), output.mutable_data_ptr<c10::BFloat16>(),
            key.size(1), key.size(2), sequence_length, page_ids.size(2), splits, query.stride(0),
            query.stride(1), value.stride(0), value.stride(1), value.stride(2), page_ids.stride(0),
            page_ids.stride(1), page_ids.stride(2), workspace.stride(0), workspace.stride(1),
            output.stride(0), output.stride(1), output.stride(3), static_cast<float>(scale));
  };
  if (splits == 1) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
    reduce_splits_kernel<<<static_cast<unsigned int>(query.size(0) * query_heads), kWarpSize, 0,
                           stream>>>(workspace.const_data_ptr<float>(),
                                     output.mutable_data_ptr<c10::BFloat16>(), splits, query_heads,
                                     workspace.stride(0), workspace.stride(1), output.stride(0),
                                     output.stride(1), output.stride(3));
  }
  assert(cudaGetLastError() == cudaSuccess);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Decode attention over selected pages of a K128 cache with compact Values";
  module.def("attention", &attention, pybind11::arg("key"), pybind11::arg("query"),
             pybind11::arg("value"), pybind11::arg("page_ids"), pybind11::arg("workspace"),
             pybind11::arg("output"), pybind11::arg("sequence_length"), pybind11::arg("scale"),
             pybind11::arg("splits"));
}
