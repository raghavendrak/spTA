#ifndef MODEL_UTILS_HPP_
#define MODEL_UTILS_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "gputucker/reconstruction.cuh"

namespace supertensor {
namespace gputucker {

struct FinalModelMetrics {
  double observed_fit{0.0};
  double observed_error_sq{0.0};
  double cross_term{0.0};
  double model_norm_sq{0.0};
  double residual_sq{0.0};
  double full_fit{0.0};
};

inline void DenseModeMultiplySquare(const std::vector<double> &input,
                                    std::vector<double> *output,
                                    const std::vector<uint64_t> &dims,
                                    int mode,
                                    const std::vector<double> &matrix) {
  const int order = static_cast<int>(dims.size());
  const uint64_t mode_dim = dims[mode];

  uint64_t outer = 1;
  for (int axis = 0; axis < mode; ++axis) {
    outer *= dims[axis];
  }

  uint64_t inner = 1;
  for (int axis = mode + 1; axis < order; ++axis) {
    inner *= dims[axis];
  }

  for (uint64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
    const uint64_t outer_base = outer_idx * mode_dim * inner;
    for (uint64_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
      for (uint64_t row = 0; row < mode_dim; ++row) {
        double sum = 0.0;
        const double *mat_row = &matrix[row * mode_dim];
        for (uint64_t col = 0; col < mode_dim; ++col) {
          sum += mat_row[col] * input[outer_base + col * inner + inner_idx];
        }
        (*output)[outer_base + row * inner + inner_idx] = sum;
      }
    }
  }
}

template <typename TensorType, typename MatrixType>
double ComputeModelNormSq(TensorType *tensor, TensorType *core_tensor,
                          MatrixType ***factor_matrices, int rank) {
  using index_t = typename TensorType::index_t;
  using value_t = typename TensorType::value_t;

  const int order = tensor->order;
  std::vector<std::vector<double> > grams(
      static_cast<size_t>(order),
      std::vector<double>(static_cast<size_t>(rank) * rank, 0.0));

  for (int axis = 0; axis < order; ++axis) {
    const index_t partition_count = tensor->partition_dims[axis];
    const index_t base_block_rows = tensor->block_dims[axis];
    const index_t total_rows = tensor->dims[axis];

    for (index_t part = 0; part < partition_count; ++part) {
      index_t sub_rows = base_block_rows;
      if (part + 1 == partition_count) {
        sub_rows = total_rows - part * base_block_rows;
      }

      MatrixType *sub_factor = factor_matrices[axis][part];
      for (index_t row = 0; row < sub_rows; ++row) {
        const uint64_t row_base = static_cast<uint64_t>(row) * rank;
        for (int i = 0; i < rank; ++i) {
          const double lhs = static_cast<double>(sub_factor[row_base + i]);
          double *gram_row = &grams[axis][static_cast<size_t>(i) * rank];
          for (int j = 0; j < rank; ++j) {
            gram_row[j] += lhs * static_cast<double>(sub_factor[row_base + j]);
          }
        }
      }
    }
  }

  const uint64_t core_size = core_tensor->nnz_count;
  value_t *core_values = core_tensor->blocks[0]->values;

  std::vector<uint64_t> dims(static_cast<size_t>(order), 0);
  for (int axis = 0; axis < order; ++axis) {
    dims[axis] = static_cast<uint64_t>(core_tensor->dims[axis]);
  }

  std::vector<double> core_data(core_size, 0.0);
  std::vector<double> current(core_size, 0.0);
  std::vector<double> scratch(core_size, 0.0);
  for (uint64_t idx = 0; idx < core_size; ++idx) {
    core_data[idx] = static_cast<double>(core_values[idx]);
    current[idx] = core_data[idx];
  }

  for (int axis = 0; axis < order; ++axis) {
    DenseModeMultiplySquare(current, &scratch, dims, axis, grams[axis]);
    current.swap(scratch);
  }

  double model_norm_sq = 0.0;
  for (uint64_t idx = 0; idx < core_size; ++idx) {
    model_norm_sq += core_data[idx] * current[idx];
  }

  return std::max(0.0, model_norm_sq);
}

template <typename TensorType, typename ErrorType>
void ComputeObservedStats(TensorType *tensor, ErrorType **recon_values,
                          double *observed_error_sq, double *cross_term) {
  using block_t = typename TensorType::block_t;

  double error_sq = 0.0;
  double cross = 0.0;

#pragma omp parallel for reduction(+ : error_sq, cross)
  for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id) {
    block_t *curr_block = tensor->blocks[block_id];
    for (uint64_t nnz = 0; nnz < curr_block->nnz_count; ++nnz) {
      const double x = static_cast<double>(curr_block->values[nnz]);
      const double xhat = static_cast<double>(recon_values[block_id][nnz]);
      const double diff = x - xhat;
      error_sq += diff * diff;
      cross += x * xhat;
    }
  }

  *observed_error_sq = error_sq;
  *cross_term = cross;
}

template <typename IndexType, typename ValueType>
__global__ void ComputingRefreshedCoreKernel(
    std::uintptr_t *X_indices, ValueType *X_values,
    std::uintptr_t *core_indices, ValueType *refreshed_core,
    std::uintptr_t *factors, const int order, const int rank,
    uint64_t nnz_count, uint64_t core_nnz_count) {
  using index_t = IndexType;
  using value_t = ValueType;

  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t stride = blockDim.x * gridDim.x;

  __shared__ int sh_rank;
  __shared__ std::uintptr_t *sh_X_idx_addr[gputucker::constants::kMaxOrder];
  __shared__ std::uintptr_t *sh_core_idx_addr[gputucker::constants::kMaxOrder];
  __shared__ std::uintptr_t *sh_factors[gputucker::constants::kMaxOrder];

  if (threadIdx.x == 0) {
    sh_rank = rank;
    for (int axis = 0; axis < order; ++axis) {
      sh_X_idx_addr[axis] =
          reinterpret_cast<std::uintptr_t *>(X_indices[axis]);
      sh_core_idx_addr[axis] =
          reinterpret_cast<std::uintptr_t *>(core_indices[axis]);
      sh_factors[axis] = reinterpret_cast<std::uintptr_t *>(factors[axis]);
    }
  }
  __syncthreads();

  while (tid < nnz_count) {
    const value_t x_val = X_values[tid];
    for (uint64_t i = 0; i < core_nnz_count; ++i) {
      value_t contrib = x_val;
      for (int axis = 0; axis < order; ++axis) {
        contrib *= ((value_t *)(sh_factors[axis]))[
            ((index_t *)sh_X_idx_addr[axis])[tid] * sh_rank +
            ((index_t *)sh_core_idx_addr[axis])[i]];
      }
      atomicAdd(&refreshed_core[i], contrib);
    }
    tid += stride;
  }
}

template <typename TensorType, typename MatrixType, typename CudaAgentType,
          typename SchedulerType>
std::vector<typename TensorType::value_t> RecomputeFinalCoreValues(
    TensorType *tensor, TensorType *core_tensor, MatrixType ***factor_matrices,
    int rank, int device_count, CudaAgentType **cuda_agents,
    SchedulerType *scheduler) {
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  const int order = tensor->order;
  const uint64_t core_nnz_count = core_tensor->nnz_count;

  std::vector<value_t> refreshed_core(core_nnz_count, (value_t)0);

  for (int dev_id = 0; dev_id < device_count; ++dev_id) {
    cuda_agents[dev_id]->SetDeviceBuffers(tensor, rank,
                                          scheduler->nnz_count_per_task);
  }

#pragma omp parallel num_threads(device_count)
  {
    const int dev_slot = omp_get_thread_num();
    CudaAgentType *cuda_agent = cuda_agents[dev_slot];
    common::cuda::_CUDA_API_CALL(cudaSetDevice(cuda_agent->get_device_id()));

    auto dev_bufs = cuda_agent->dev_buf;
    auto dev_prof = cuda_agent->get_device_properties();
    cudaStream_t *streams =
        static_cast<cudaStream_t *>(cuda_agent->get_cuda_streams());
    const int stream_count = cuda_agent->get_stream_count();
    const int max_grid_size = dev_prof->maxGridSize[0] / stream_count;

    common::cuda::_CUDA_API_CALL(cudaMemset(
        dev_bufs.core_values.get_ptr(0), 0,
        sizeof(value_t) * core_nnz_count));

    std::uintptr_t ***h_X_idx_addr =
        static_cast<std::uintptr_t ***>(common::cuda::pinned_malloc(
            sizeof(std::uintptr_t **) * stream_count));
    std::uintptr_t ***h_fact_addr =
        static_cast<std::uintptr_t ***>(common::cuda::pinned_malloc(
            sizeof(std::uintptr_t **) * stream_count));
    std::uintptr_t **h_core_idx_addr =
        static_cast<std::uintptr_t **>(common::cuda::pinned_malloc(
            sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder));

    for (int i = 0; i < stream_count; ++i) {
      h_X_idx_addr[i] =
          static_cast<std::uintptr_t **>(common::cuda::pinned_malloc(
              sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder));
      h_fact_addr[i] =
          static_cast<std::uintptr_t **>(common::cuda::pinned_malloc(
              sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder));
    }

    for (int axis = 0; axis < order; ++axis) {
      h_core_idx_addr[axis] =
          reinterpret_cast<std::uintptr_t *>(dev_bufs.core_indices[axis].get_ptr(0));
    }
    common::cuda::h2dcpy(dev_bufs.core_idx_addr.get_ptr(0), h_core_idx_addr,
                         sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder);
    for (int axis = 0; axis < order; ++axis) {
      common::cuda::h2dcpy(dev_bufs.core_indices[axis].get_ptr(0),
                           core_tensor->blocks[0]->indices[axis],
                           sizeof(index_t) * core_tensor->nnz_count);
    }

    auto tasks = scheduler->tasks[dev_slot];
    index_t *block_dims = tensor->block_dims;
    for (uint64_t iter = 0; iter < tasks.size(); ++iter) {
      uint64_t block_id = tasks[iter].block_id;
      uint64_t avail_nnz_count = tasks[iter].nnz_count;
      uint64_t nnz_offset = tasks[iter].offset;
      int stream_offset = tasks[iter].stream_offset;

      block_t *curr_block = tensor->blocks[block_id];
      index_t *curr_block_coord = curr_block->get_block_coord();

      for (int axis = 0; axis < order; ++axis) {
        h_X_idx_addr[stream_offset][axis] =
            reinterpret_cast<std::uintptr_t *>(
                dev_bufs.X_indices[axis].get_ptr(stream_offset));
        common::cuda::h2dcpy_async(
            dev_bufs.X_indices[axis].get_ptr(stream_offset),
            &curr_block->indices[axis][nnz_offset],
            sizeof(index_t) * avail_nnz_count, streams[stream_offset]);

        h_fact_addr[stream_offset][axis] =
            reinterpret_cast<std::uintptr_t *>(
                dev_bufs.factors[axis].get_ptr(stream_offset));
        common::cuda::h2dcpy_async(
            dev_bufs.factors[axis].get_ptr(stream_offset),
            factor_matrices[axis][curr_block_coord[axis]],
            sizeof(value_t) * block_dims[axis] * rank,
            streams[stream_offset]);
      }

      common::cuda::h2dcpy_async(
          dev_bufs.X_idx_addr.get_ptr(stream_offset),
          h_X_idx_addr[stream_offset],
          sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder,
          streams[stream_offset]);
      common::cuda::h2dcpy_async(
          dev_bufs.factor_addr.get_ptr(stream_offset),
          h_fact_addr[stream_offset],
          sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder,
          streams[stream_offset]);
      common::cuda::h2dcpy_async(
          dev_bufs.X_values.get_ptr(stream_offset),
          &curr_block->values[nnz_offset],
          sizeof(value_t) * avail_nnz_count, streams[stream_offset]);

      const index_t block_size = 1024;
      const index_t grid_size =
          std::min(max_grid_size,
                   std::max(1, (int)((avail_nnz_count + block_size - 1) /
                                     block_size)));

      dim3 blocks_per_grid(grid_size, 1, 1);
      dim3 threads_per_block(block_size, 1, 1);

      gputucker::ComputingRefreshedCoreKernel<index_t, value_t>
          <<<blocks_per_grid, threads_per_block, 0, streams[stream_offset]>>>(
              (std::uintptr_t *)dev_bufs.X_idx_addr.get_ptr(stream_offset),
              (value_t *)dev_bufs.X_values.get_ptr(stream_offset),
              (std::uintptr_t *)dev_bufs.core_idx_addr.get_ptr(0),
              (value_t *)dev_bufs.core_values.get_ptr(0),
              (std::uintptr_t *)dev_bufs.factor_addr.get_ptr(stream_offset),
              order, rank, avail_nnz_count, core_nnz_count);
    }

    for (int stream_offset = 0; stream_offset < stream_count; ++stream_offset) {
      common::cuda::_CUDA_API_CALL(
          cudaStreamSynchronize(streams[stream_offset]));
    }

    std::vector<value_t> local_core(core_nnz_count, (value_t)0);
    common::cuda::d2hcpy(local_core.data(), dev_bufs.core_values.get_ptr(0),
                         sizeof(value_t) * core_nnz_count);

#pragma omp critical
    {
      for (uint64_t i = 0; i < core_nnz_count; ++i) {
        refreshed_core[i] += local_core[i];
      }
    }
  }

  return refreshed_core;
}

template <typename TensorType, typename MatrixType, typename ErrorType,
          typename CudaAgentType, typename SchedulerType>
FinalModelMetrics EvaluateReturnedModel(
    TensorType *tensor, TensorType *core_tensor, MatrixType ***factor_matrices,
    ErrorType **error_T, int rank, int device_count, CudaAgentType **cuda_agents,
    SchedulerType *scheduler) {
  FinalModelMetrics metrics;

  Reconstruction<TensorType, MatrixType, ErrorType, CudaAgentType, SchedulerType>(
      tensor, core_tensor, factor_matrices, &metrics.observed_fit, error_T, rank,
      device_count, cuda_agents, scheduler);

  ComputeObservedStats(tensor, error_T, &metrics.observed_error_sq,
                       &metrics.cross_term);
  metrics.model_norm_sq =
      ComputeModelNormSq(tensor, core_tensor, factor_matrices, rank);

  const double input_norm = static_cast<double>(tensor->norm);
  const double input_norm_sq = input_norm * input_norm;
  metrics.residual_sq = std::max(
      0.0, input_norm_sq + metrics.model_norm_sq - 2.0 * metrics.cross_term);
  metrics.full_fit =
      input_norm == 0.0 ? 1.0 : (1.0 - std::sqrt(metrics.residual_sq) / input_norm);

  return metrics;
}

template <typename TensorType, typename MatrixType>
void WriteModelArtifacts(const std::string &output_dir, TensorType *tensor,
                         TensorType *core_tensor, MatrixType ***factor_matrices,
                         int rank, const FinalModelMetrics &metrics,
                         const FinalModelMetrics *refreshed_metrics = nullptr,
                         double original_eval_time = -1.0,
                         double refreshed_core_time = -1.0,
                         double refreshed_eval_time = -1.0) {
  if (output_dir.empty()) {
    return;
  }

  boost::filesystem::path out_path(output_dir);
  boost::filesystem::create_directories(out_path);

  {
    std::ofstream summary((out_path / "summary.txt").string().c_str());
    summary << "observed_fit=" << metrics.observed_fit << "\n";
    summary << "observed_error_sq=" << metrics.observed_error_sq << "\n";
    summary << "cross_term=" << metrics.cross_term << "\n";
    summary << "model_norm_sq=" << metrics.model_norm_sq << "\n";
    summary << "residual_sq=" << metrics.residual_sq << "\n";
    summary << "full_fit=" << metrics.full_fit << "\n";
    if (refreshed_metrics != nullptr) {
      summary << "refreshed_observed_fit=" << refreshed_metrics->observed_fit << "\n";
      summary << "refreshed_observed_error_sq=" << refreshed_metrics->observed_error_sq << "\n";
      summary << "refreshed_cross_term=" << refreshed_metrics->cross_term << "\n";
      summary << "refreshed_model_norm_sq=" << refreshed_metrics->model_norm_sq << "\n";
      summary << "refreshed_residual_sq=" << refreshed_metrics->residual_sq << "\n";
      summary << "refreshed_full_fit=" << refreshed_metrics->full_fit << "\n";
    }
    if (original_eval_time >= 0.0) {
      summary << "original_eval_time_s=" << original_eval_time << "\n";
    }
    if (refreshed_core_time >= 0.0) {
      summary << "refreshed_core_time_s=" << refreshed_core_time << "\n";
    }
    if (refreshed_eval_time >= 0.0) {
      summary << "refreshed_eval_time_s=" << refreshed_eval_time << "\n";
      if (original_eval_time >= 0.0 && refreshed_core_time >= 0.0) {
        summary << "postprocess_total_time_s="
                << (original_eval_time + refreshed_core_time + refreshed_eval_time)
                << "\n";
      }
    }
    summary << "note=full_fit is computed on the returned GPUTucker model using "
               "||X||^2 + ||Xhat||^2 - 2<X,Xhat>\n";
    summary << "factor_format=mode<axis>.bin stores uint64 rows, uint64 cols, "
               "then double row-major data\n";
    summary << "core_format=core.bin stores uint64 order, uint64 dims[order], "
               "then double row-major data\n";
  }

  for (int axis = 0; axis < tensor->order; ++axis) {
    std::ofstream factor_out(
        (out_path / ("mode" + std::to_string(axis) + ".bin")).string().c_str(),
        std::ios::binary);
    const uint64_t rows = static_cast<uint64_t>(tensor->dims[axis]);
    const uint64_t cols = static_cast<uint64_t>(rank);
    factor_out.write(reinterpret_cast<const char *>(&rows), sizeof(uint64_t));
    factor_out.write(reinterpret_cast<const char *>(&cols), sizeof(uint64_t));

    const typename TensorType::index_t partition_count = tensor->partition_dims[axis];
    const typename TensorType::index_t base_block_rows = tensor->block_dims[axis];
    const typename TensorType::index_t total_rows = tensor->dims[axis];

    for (typename TensorType::index_t part = 0; part < partition_count; ++part) {
      typename TensorType::index_t sub_rows = base_block_rows;
      if (part + 1 == partition_count) {
        sub_rows = total_rows - part * base_block_rows;
      }

      MatrixType *sub_factor = factor_matrices[axis][part];
      std::vector<double> buffer(static_cast<size_t>(sub_rows) * rank, 0.0);
      for (typename TensorType::index_t row = 0; row < sub_rows; ++row) {
        const uint64_t row_base = static_cast<uint64_t>(row) * rank;
        for (int col = 0; col < rank; ++col) {
          buffer[row_base + col] = static_cast<double>(sub_factor[row_base + col]);
        }
      }

      factor_out.write(reinterpret_cast<const char *>(buffer.data()),
                       sizeof(double) * buffer.size());
    }
  }

  {
    std::ofstream core_out((out_path / "core.bin").string().c_str(),
                           std::ios::binary);
    const uint64_t order = static_cast<uint64_t>(core_tensor->order);
    core_out.write(reinterpret_cast<const char *>(&order), sizeof(uint64_t));
    for (int axis = 0; axis < core_tensor->order; ++axis) {
      const uint64_t dim = static_cast<uint64_t>(core_tensor->dims[axis]);
      core_out.write(reinterpret_cast<const char *>(&dim), sizeof(uint64_t));
    }

    const uint64_t core_size = core_tensor->nnz_count;
    std::vector<double> core_buffer(core_size, 0.0);
    for (uint64_t idx = 0; idx < core_size; ++idx) {
      core_buffer[idx] = static_cast<double>(core_tensor->blocks[0]->values[idx]);
    }
    core_out.write(reinterpret_cast<const char *>(core_buffer.data()),
                   sizeof(double) * core_buffer.size());
  }
}

} // namespace gputucker
} // namespace supertensor

#endif /* MODEL_UTILS_HPP_ */
