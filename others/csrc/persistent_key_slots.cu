#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kHeadDim = 128;
constexpr int kMaximumSupport = 2048;
constexpr int kElementsPerVector = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kVectorsPerKey = kHeadDim / kElementsPerVector;

// One block owns one (batch, KV head) row. Lookup entries can be stale;
// resident[slot] confirms ownership before a cached Key is reused.
__global__ void plan_slots(const int64_t* ids, int* lookup, int64_t* resident,
                           int64_t* selected_slots, int* missing, int* counts, int capacity,
                           int budget, bool reuse) {
  __shared__ uint8_t used[kMaximumSupport];
  __shared__ int free_slots[kMaximumSupport];
  __shared__ int requested[kMaximumSupport];
  __shared__ int free_count, miss_count, hit_count, valid_count;
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  if (thread == 0) {
    free_count = miss_count = hit_count = valid_count = 0;
  }
  for (int item = thread; item < budget; item += blockDim.x) {
    used[item] = 0;
  }
  __syncthreads();

  // 1. Protect slots already holding one of this step's selected Keys.
  for (int item = thread; item < budget; item += blockDim.x) {
    const int64_t token = ids[row * budget + item];
    const bool valid = token >= 0 && token < capacity;
    const int slot = valid && reuse ? lookup[row * capacity + token] : -1;
    const bool hit = slot >= 0 && slot < budget && resident[row * budget + slot] == token;
    requested[item] = hit ? slot : -1;
    if (hit) {
      used[slot] = 1;
      atomicAdd(&hit_count, 1);
    }
    if (valid)
      atomicAdd(&valid_count, 1);
  }
  __syncthreads();

  // 2. Make the remaining slots available to selected Keys that missed.
  for (int slot = thread; slot < budget; slot += blockDim.x) {
    if (!used[slot])
      free_slots[atomicAdd(&free_count, 1)] = slot;
  }
  __syncthreads();
  for (int item = thread; item < budget; item += blockDim.x) {
    const int64_t token = ids[row * budget + item];
    int slot = requested[item];
    int missing_token = -1;
    if (token >= 0 && token < capacity) {
      if (slot < 0) {
        slot = free_slots[atomicAdd(&miss_count, 1)];
        missing_token = token;
      }
      resident[row * budget + slot] = token;
      lookup[row * capacity + token] = slot;
    } else {
      slot = -1;
    }
    selected_slots[row * budget + item] = slot;
    missing[row * budget + item] = missing_token;
  }
  if (thread == 0) {
    counts[row * 2] = hit_count;
    counts[row * 2 + 1] = valid_count;
  }
}

// 3. Copy only misses from CUDA-mapped host memory, in 16-byte vectors.
__global__ void fetch_missing(const __nv_bfloat16* host, __nv_bfloat16* cache, const int* missing,
                              const int64_t* slots, int capacity, int budget, int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * budget * kVectorsPerKey)
    return;
  const int selected = index / kVectorsPerKey;
  const int vector = index % kVectorsPerKey;
  const int row = selected / budget;
  const int token = missing[selected];
  if (token >= 0) {
    const int64_t slot = slots[selected];
    const uint4* source = reinterpret_cast<const uint4*>(
        host + (static_cast<int64_t>(row) * capacity + token) * kHeadDim);
    uint4* destination =
        reinterpret_cast<uint4*>(cache + (static_cast<int64_t>(row) * budget + slot) * kHeadDim);
    destination[vector] = source[vector];
  }
}

}  // namespace

void refresh(int64_t pointer, const at::Tensor& cache, const at::Tensor& ids,
             const at::Tensor& resident, const at::Tensor& lookup, const at::Tensor& slots,
             const at::Tensor& missing, const at::Tensor& counts, bool reuse) {
  c10::cuda::CUDAGuard guard(cache.device());
  const int rows = ids.size(0) * ids.size(1);
  const int budget = ids.size(2);
  const int capacity = lookup.size(2);
  assert(budget > 0 && budget <= kMaximumSupport && cache.size(3) == kHeadDim);
  assert(cache.is_contiguous() && ids.is_contiguous());
  assert(cache.numel() == rows * budget * kHeadDim && resident.numel() == rows * budget);
  assert(slots.numel() == rows * budget && missing.numel() == rows * budget);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  plan_slots<<<rows, kThreads, 0, stream>>>(ids.data_ptr<int64_t>(), lookup.data_ptr<int>(),
                                            resident.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
                                            missing.data_ptr<int>(), counts.data_ptr<int>(),
                                            capacity, budget, reuse);
  const int vectors = rows * budget * kVectorsPerKey;
  fetch_missing<<<(vectors + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(pointer),
      reinterpret_cast<__nv_bfloat16*>(cache.data_ptr<at::BFloat16>()), missing.data_ptr<int>(),
      slots.data_ptr<int64_t>(), capacity, budget, rows);
  assert(cudaGetLastError() == cudaSuccess);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("refresh", &refresh);
}
