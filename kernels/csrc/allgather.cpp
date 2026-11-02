#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {

void check(ncclResult_t status) {
  assert(status == ncclSuccess);
}

ncclDataType_t nccl_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return ncclFloat16;
    case at::kBFloat16:
      return ncclBfloat16;
    case at::kFloat:
      return ncclFloat32;
    case at::kByte:
      return ncclUint8;
    default:
      assert(false);
      return ncclUint8;
  }
}

class Plan;

class Communicator : public std::enable_shared_from_this<Communicator> {
 public:
  Communicator(const py::bytes& identity, int rank, int world_size, int device)
      : rank(rank), world_size(world_size), device(device) {
    const std::string bytes = identity;
    assert(bytes.size() == sizeof(ncclUniqueId) && rank >= 0 && rank < world_size);
    ncclUniqueId id;
    std::memcpy(&id, bytes.data(), sizeof(id));
    c10::cuda::CUDAGuard guard(device);
    check(ncclCommInitRank(&comm, world_size, id, rank));
  }

  ~Communicator() {
    if (comm != nullptr) {
      c10::cuda::CUDAGuard guard(device);
      ncclCommAbort(comm);
    }
  }

  void close() {
    if (comm != nullptr) {
      c10::cuda::CUDAGuard guard(device);
      cudaDeviceSynchronize();
      check(ncclCommDestroy(comm));
      comm = nullptr;
    }
  }

  std::shared_ptr<Plan> prepare(at::Tensor arena, int64_t local_width);

  ncclComm_t comm = nullptr;
  const int rank, world_size, device;
};

class Plan {
 public:
  Plan(std::shared_ptr<Communicator> owner, at::Tensor arena, int64_t local_width)
      : owner_(std::move(owner)), arena_(std::move(arena)) {
    assert(arena_.is_cuda() && arena_.is_contiguous() && arena_.dim() == 2);
    assert(arena_.get_device() == owner_->device);
    assert(local_width > 0 && arena_.size(0) == local_width * owner_->world_size);
    local = arena_.narrow(0, owner_->rank * local_width, local_width);
    count_ = local.numel();
    dtype_ = nccl_dtype(arena_.scalar_type());
    stream_ = c10::cuda::getCurrentCUDAStream(owner_->device).stream();
  }

  // The serving caller keeps the preparation device and stream current. NCCL
  // supports this in-place source slice, including during CUDA Graph capture.
  at::Tensor gather() {
    assert(owner_->comm != nullptr);
    assert(c10::cuda::getCurrentCUDAStream(owner_->device).stream() == stream_);
    if (owner_->world_size > 1) {
      check(ncclAllGather(local.data_ptr(), arena_.data_ptr(), count_, dtype_, owner_->comm,
                          stream_));
    }
    return arena_;
  }

  at::Tensor local;

 private:
  std::shared_ptr<Communicator> owner_;
  at::Tensor arena_;
  size_t count_;
  ncclDataType_t dtype_;
  cudaStream_t stream_;
};

std::shared_ptr<Plan> Communicator::prepare(at::Tensor arena, int64_t local_width) {
  return std::make_shared<Plan>(shared_from_this(), std::move(arena), local_width);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("unique_id_size", [] { return sizeof(ncclUniqueId); });
  module.def("unique_id", [] {
    ncclUniqueId identity;
    check(ncclGetUniqueId(&identity));
    return py::bytes(reinterpret_cast<const char*>(&identity), sizeof(identity));
  });
  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan")
      .def_readonly("local", &Plan::local)
      .def("gather", &Plan::gather);
  py::class_<Communicator, std::shared_ptr<Communicator>>(module, "Communicator")
      .def(py::init<const py::bytes&, int, int, int>())
      .def("prepare", &Communicator::prepare)
      .def("close", &Communicator::close);
}
