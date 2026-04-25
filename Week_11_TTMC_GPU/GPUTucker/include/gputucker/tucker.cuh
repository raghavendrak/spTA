#ifndef TUCKER_CUH_
#define TUCKER_CUH_

#include <omp.h>

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/cuda_helper.hpp"

#include "gputucker/constants.hpp"
#include "gputucker/cuda_agent.hpp"
#include "gputucker/model_utils.hpp"
#include "gputucker/optimizer.hpp"
#include "gputucker/reconstruction.cuh"
#include "gputucker/scheduler.hpp"
#include "gputucker/tensor_manager.hpp"
#include "gputucker/update.cuh"

namespace supertensor {
namespace gputucker {

namespace {

struct FactorFilePayload {
  uint64_t rows{0};
  uint64_t cols{0};
  std::vector<float> values;
};

inline FactorFilePayload ReadFactorFileFloat32(const std::string &path,
                                               uint64_t expected_rows,
                                               uint64_t max_cols) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("cannot open factors file: " + path);
  }

  fin.seekg(0, std::ios::end);
  const std::streamsize file_size = fin.tellg();
  fin.seekg(0, std::ios::beg);
  if (file_size < 0) {
    throw std::runtime_error("cannot stat factors file: " + path);
  }

  const std::streamsize bytes_per_row =
      static_cast<std::streamsize>(expected_rows * sizeof(float));

  auto read_values = [&](uint64_t rows, uint64_t cols, std::streamsize offset) {
    FactorFilePayload payload;
    payload.rows = rows;
    payload.cols = cols;
    payload.values.resize(static_cast<size_t>(rows * cols));
    fin.seekg(offset, std::ios::beg);
    fin.read(reinterpret_cast<char *>(payload.values.data()),
             static_cast<std::streamsize>(payload.values.size() * sizeof(float)));
    if (!fin) {
      throw std::runtime_error("short read on factors file: " + path);
    }
    return payload;
  };

  if (file_size >= static_cast<std::streamsize>(2 * sizeof(uint64_t))) {
    uint64_t fr = 0, fc = 0;
    fin.read(reinterpret_cast<char *>(&fr), sizeof(uint64_t));
    fin.read(reinterpret_cast<char *>(&fc), sizeof(uint64_t));
    if (!fin) {
      throw std::runtime_error("failed to read factors header: " + path);
    }

    const std::streamsize header_bytes =
        static_cast<std::streamsize>(2 * sizeof(uint64_t) + fr * fc * sizeof(float));
    if (fr == expected_rows && fc > 0 && fc <= max_cols && header_bytes == file_size) {
      return read_values(fr, fc, static_cast<std::streamsize>(2 * sizeof(uint64_t)));
    }
    fin.clear();
  }

  if (bytes_per_row <= 0 || file_size % bytes_per_row != 0) {
    throw std::runtime_error("factor file size mismatch for " + path);
  }

  const uint64_t inferred_cols = static_cast<uint64_t>(file_size / bytes_per_row);
  if (inferred_cols == 0 || inferred_cols > max_cols) {
    throw std::runtime_error("factor column mismatch for " + path);
  }
  return read_values(expected_rows, inferred_cols, 0);
}

template <typename ValueType>
void OrthonormalizeTrailingColumns(uint64_t rows, uint64_t cols,
                                   uint64_t start_col, ValueType *A) {
  for (uint64_t c = start_col; c < cols; ++c) {
    for (uint64_t k = 0; k < c; ++k) {
      ValueType dot = 0;
      for (uint64_t r = 0; r < rows; ++r) {
        dot += A[r * cols + k] * A[r * cols + c];
      }
      for (uint64_t r = 0; r < rows; ++r) {
        A[r * cols + c] -= dot * A[r * cols + k];
      }
    }

    ValueType norm = 0;
    for (uint64_t r = 0; r < rows; ++r) {
      norm += A[r * cols + c] * A[r * cols + c];
    }
    norm = std::sqrt(norm);
    if (norm < (ValueType)1e-10) {
      norm = (ValueType)1;
    }
    for (uint64_t r = 0; r < rows; ++r) {
      A[r * cols + c] /= norm;
    }
  }
}

template <typename ValueType>
void init_factor_orthonormal(uint64_t rows, uint64_t cols,
                             unsigned int mode_idx, ValueType *A) {
  const char *factors_dir = std::getenv("TUCKER_FACTORS_DIR");
  if (factors_dir && *factors_dir) {
    std::string path = std::string(factors_dir) + "/mode" +
                       std::to_string(mode_idx) + ".bin";
    FactorFilePayload payload = ReadFactorFileFloat32(path, rows, cols);
    if (payload.rows != rows || payload.cols == 0 || payload.cols > cols) {
      throw std::runtime_error("factor shape mismatch for " + path);
    }

    const uint64_t load_cols = payload.cols;
    for (uint64_t r = 0; r < rows; ++r) {
      for (uint64_t c = 0; c < load_cols; ++c) {
        A[r * cols + c] =
            static_cast<ValueType>(payload.values[r * load_cols + c]);
      }
    }
    if (load_cols < cols) {
      std::mt19937 gen(42 + mode_idx);
      std::normal_distribution<ValueType> dist((ValueType)0, (ValueType)1);
      for (uint64_t c = load_cols; c < cols; ++c) {
        for (uint64_t r = 0; r < rows; ++r) {
          A[r * cols + c] = dist(gen);
        }
      }
      OrthonormalizeTrailingColumns(rows, cols, load_cols, A);
      std::cout << "[factors] loaded " << path << " (" << rows << " x "
                << load_cols << ") and extended to " << cols << " cols\n";
    } else {
      std::cout << "[factors] loaded " << path << " (" << rows << " x "
                << cols << ")\n";
    }
    return;
  }

  std::mt19937 gen(42 + mode_idx);
  std::normal_distribution<ValueType> dist((ValueType)0, (ValueType)1);
  for (uint64_t i = 0; i < rows * cols; ++i) {
    A[i] = dist(gen);
  }

  for (uint64_t c = 0; c < cols; ++c) {
    ValueType norm = 0;
    for (uint64_t r = 0; r < rows; ++r) {
      norm += A[r * cols + c] * A[r * cols + c];
    }
    norm = std::sqrt(norm);
    if (norm < (ValueType)1e-10) {
      norm = (ValueType)1;
    }
    for (uint64_t r = 0; r < rows; ++r) {
      A[r * cols + c] /= norm;
    }
    for (uint64_t c2 = c + 1; c2 < cols; ++c2) {
      ValueType dot = 0;
      for (uint64_t r = 0; r < rows; ++r) {
        dot += A[r * cols + c] * A[r * cols + c2];
      }
      for (uint64_t r = 0; r < rows; ++r) {
        A[r * cols + c2] -= dot * A[r * cols + c];
      }
    }
  }
}

} // namespace

/**
 * @brief Perform Tucker decomposition on the input tensor.
 *
 * This function performs Tucker decomposition on the input tensor using the specified Tucker rank.
 *
 * @tparam TensorType The type of the tensor being decomposed.
 * @param input_tensor Pointer to the input tensor.
 * @param tucker_rank The Tucker rank.
 * @param gpu_count The number of GPUs to use.
 *
 */
template <typename TensorType>
void TuckerDecomposition(
    TensorType *input_tensor, int tucker_rank, int gpu_count,
    int max_iteration = gputucker::constants::kMaxIteration,
    double tol = gputucker::constants::kLambda,
    const std::string &output_dir = std::string()) {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using block_t = typename tensor_t::block_t;

  using optimizer_t = Optimizer<tensor_t>;
  using cuda_agent_t = CudaAgent<tensor_t>;
  using scheduler_t = Scheduler<tensor_t, optimizer_t>;

  /* Initialize Cuda Agents */
  MYPRINT("\t... Initialize CUDA Agents\n");

  cuda_agent_t **cuda_agents = allocate<cuda_agent_t *>(gpu_count);
  for (unsigned dev_id = 0; dev_id < gpu_count; ++dev_id) {
    cuda_agents[dev_id] = new cuda_agent_t(dev_id);
    cuda_agents[dev_id]->AllocateMaximumBuffer();
  }

  size_t avail_gpu_mem = cuda_agents[0]->get_allocated_size();
  std::cout << "Available GPU memory: " << common::HumanReadable{(uintmax_t)avail_gpu_mem} << std::endl;

  // Find optimal partition parameters from optimizer
  optimizer_t *optimizer = new optimizer_t;
  optimizer->Initialize(gpu_count, tucker_rank, avail_gpu_mem, input_tensor);
  index_t *partition_dims = optimizer->FindPartitionParms();

  // Create tensor blocks ( = sub-tensors )
  tensor_t *tensor = new tensor_t(input_tensor);
  CreateTensorBlocks<tensor_t, optimizer_t>(&input_tensor, &tensor, optimizer);
  tensor->ToString();

  MYPRINT("\t... Initialize Scheduler\n");
  scheduler_t *scheduler = new scheduler_t(gpu_count);
  scheduler->Schedule(tensor, optimizer);

  unsigned short order = tensor->order;
  index_t *dims = tensor->dims;
  index_t *block_dims = tensor->block_dims;

  MYPRINT("... Ready to fill in the factor matrices and the core tensor\n");
  value_t **factor_matrices[gputucker::constants::kMaxOrder];

  printf("\t... Make the factor matrices\n");
  // Allocate sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    factor_matrices[axis] = static_cast<value_t **>(common::cuda::pinned_malloc(sizeof(value_t *) * partition_dims[axis]));
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      factor_matrices[axis][part] = static_cast<value_t *>(common::cuda::pinned_malloc(sizeof(value_t) * sub_factor_row * tucker_rank));
    }
  }

  // Initialize factor matrices in full row-major form first so they can match
  // SHAKTI's exact dump format when TUCKER_FACTORS_DIR is provided.
  for (int axis = 0; axis < order; ++axis) {
    printf("\t\t... Fill the factor matrix [%d]\n", axis);
    const uint64_t full_rows = dims[axis];
    std::vector<value_t> full_factor(static_cast<size_t>(full_rows) * tucker_rank);
    init_factor_orthonormal(full_rows, static_cast<uint64_t>(tucker_rank),
                            static_cast<unsigned int>(axis),
                            full_factor.data());

    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      index_t sub_factor_row = block_dims[axis];
      if (part + 1 == partition_dims[axis]) {
        sub_factor_row = dims[axis] - part * block_dims[axis];
      }
      const uint64_t global_row_offset = static_cast<uint64_t>(part) * block_dims[axis];
      for (index_t row = 0; row < sub_factor_row; ++row) {
        const uint64_t src_row = global_row_offset + row;
        for (int col = 0; col < tucker_rank; ++col) {
          factor_matrices[axis][part][row * tucker_rank + col] =
              full_factor[src_row * tucker_rank + static_cast<uint64_t>(col)];
        }
      }
    }
  }

  // Core tensor
  printf("\t... Make the core tensor\n");
  tensor_t *core_tensor = new tensor_t(order);
  index_t *core_dims = gputucker::allocate<index_t>(order);
  index_t *core_part_dims = gputucker::allocate<index_t>(order);
  uint64_t core_nnz_count = 1;
  for (int axis = 0; axis < order; ++axis) {
    core_dims[axis] = tucker_rank;
    core_part_dims[axis] = 1;
    core_nnz_count *= tucker_rank;
  }

  core_tensor->set_dims(core_dims);
  core_tensor->set_nnz_count(core_nnz_count);
  core_tensor->MakeBlocks(1, &core_nnz_count);

  block_t *curr_block = core_tensor->blocks[0];

  // #pragma omp parallel for
  for (uint64_t i = 0; i < core_nnz_count; ++i) {
    curr_block->values[i] = gputucker::frand<double>(0, 1);
    index_t mult = 1;
    for (short axis = order; --axis >= 0;) {
      index_t idx = 0;
      if (axis == order - 1) {
        idx = i % core_dims[axis];
      } else if (axis == 0) {
        idx = i / mult;
      } else {
        idx = (i / mult) % core_dims[axis];
      }
      curr_block->indices[axis][i] = idx;
      mult *= core_dims[axis];
    }
    assert(mult == core_nnz_count);
  }
  printf("\t... Initialize the intermediate data (delta, B and C, errorT)\n");
  const uint64_t block_count = tensor->block_count;
  const index_t max_block_dim = tensor->get_max_block_dim();
  const index_t max_partition_dim = tensor->get_max_partition_dim();

  using matrix_t = double;
  value_t **delta = gputucker::allocate<value_t *>(block_count);
  matrix_t **B = gputucker::allocate<matrix_t *>(max_partition_dim);
  matrix_t **C = gputucker::allocate<matrix_t *>(max_partition_dim);
  value_t **error_T = gputucker::allocate<value_t *>(block_count);

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t *curr_block = tensor->blocks[block_id];
    delta[block_id] = gputucker::allocate<value_t>(curr_block->nnz_count * tucker_rank);
    error_T[block_id] = gputucker::allocate<value_t>(curr_block->nnz_count);
  }
  for (index_t part = 0; part < max_partition_dim; ++part) {
    B[part] = gputucker::allocate<matrix_t>(max_block_dim * tucker_rank * tucker_rank);
    C[part] = gputucker::allocate<matrix_t>(max_block_dim * tucker_rank);
  }

  for (unsigned dev_id = 0; dev_id < gpu_count; ++dev_id) {
    cuda_agents[dev_id]->set_stream_count(optimizer->cuda_stream_count);
  }

  int iter = 0;
  double p_fit = -1;
  double fit = -1;

  double avg_time = omp_get_wtime();

  while (1) {
    double itertime = omp_get_wtime(), steptime;
    steptime = itertime;
    gputucker::UpdateFactorMatrices<tensor_t, matrix_t, value_t, cuda_agent_t, scheduler_t>(tensor, core_tensor, factor_matrices, delta, B, C,
                                                                                            tucker_rank, gpu_count, cuda_agents, scheduler);
    printf("Factor Time : %lf\n", omp_get_wtime() - steptime);

    steptime = omp_get_wtime();
    gputucker::Reconstruction<tensor_t, value_t, value_t, cuda_agent_t, scheduler_t>(tensor, core_tensor, factor_matrices, &fit, error_T, tucker_rank,
                                                                                     gpu_count, cuda_agents, scheduler);
    printf("Recon Time : %lf\n\n", omp_get_wtime() - steptime);
    steptime = omp_get_wtime();

    ++iter;

    std::cout << "iter " << iter << "\t ObservedFit: " << fit << std::endl;
    printf("iter%d :      ObservedFit : %lf\tElapsed Time : %lf\n\n", iter, fit, omp_get_wtime() - itertime);
    if (iter >= max_iteration || (p_fit != -1 && gputucker::abs<double>(p_fit - fit) <= tol)) {
      break;
    }
    p_fit = fit;
  }

  printf("[ Final Model Evaluation ]\n");
  const double original_eval_start = omp_get_wtime();
  FinalModelMetrics final_metrics =
      gputucker::EvaluateReturnedModel<tensor_t, value_t, value_t,
                                       cuda_agent_t, scheduler_t>(
          tensor, core_tensor, factor_matrices, error_T, tucker_rank, gpu_count,
          cuda_agents, scheduler);
  const double original_eval_time = omp_get_wtime() - original_eval_start;

  const uint64_t final_core_nnz_count = core_tensor->nnz_count;
  std::vector<value_t> original_core_values(final_core_nnz_count, (value_t)0);
  for (uint64_t i = 0; i < final_core_nnz_count; ++i) {
    original_core_values[i] = core_tensor->blocks[0]->values[i];
  }

  const double refreshed_core_start = omp_get_wtime();
  std::vector<value_t> refreshed_core_values =
      gputucker::RecomputeFinalCoreValues<tensor_t, value_t, cuda_agent_t,
                                          scheduler_t>(
          tensor, core_tensor, factor_matrices, tucker_rank, gpu_count,
          cuda_agents, scheduler);
  const double refreshed_core_time = omp_get_wtime() - refreshed_core_start;

  for (uint64_t i = 0; i < final_core_nnz_count; ++i) {
    core_tensor->blocks[0]->values[i] = refreshed_core_values[i];
  }

  const double refreshed_eval_start = omp_get_wtime();
  FinalModelMetrics refreshed_metrics =
      gputucker::EvaluateReturnedModel<tensor_t, value_t, value_t,
                                       cuda_agent_t, scheduler_t>(
          tensor, core_tensor, factor_matrices, error_T, tucker_rank, gpu_count,
          cuda_agents, scheduler);
  const double refreshed_eval_time = omp_get_wtime() - refreshed_eval_start;

  for (uint64_t i = 0; i < final_core_nnz_count; ++i) {
    core_tensor->blocks[0]->values[i] = original_core_values[i];
  }

  const double postprocess_total_time =
      original_eval_time + refreshed_core_time + refreshed_eval_time;

  std::printf("[GPUTucker Final Model]: observed_fit = %.8e, observed_error_sq = %.8e\n",
              final_metrics.observed_fit, final_metrics.observed_error_sq);
  std::printf("[GPUTucker Final Model]: cross_term = %.8e, model_norm_sq = %.8e, residual_sq = %.8e, full_fit = %.8e\n",
              final_metrics.cross_term, final_metrics.model_norm_sq,
              final_metrics.residual_sq, final_metrics.full_fit);
  std::printf("[GPUTucker Refreshed Core]: observed_fit = %.8e, observed_error_sq = %.8e\n",
              refreshed_metrics.observed_fit, refreshed_metrics.observed_error_sq);
  std::printf("[GPUTucker Refreshed Core]: cross_term = %.8e, model_norm_sq = %.8e, residual_sq = %.8e, full_fit = %.8e\n",
              refreshed_metrics.cross_term, refreshed_metrics.model_norm_sq,
              refreshed_metrics.residual_sq, refreshed_metrics.full_fit);
  std::printf("[GPUTucker Postprocess Time]: original_eval_s = %.6f, refreshed_core_s = %.6f, refreshed_eval_s = %.6f, total_s = %.6f\n",
              original_eval_time, refreshed_core_time, refreshed_eval_time,
              postprocess_total_time);
  std::printf("[GPUTucker Summary]: iterations = %d, final_observed_fit = %.8e, final_full_fit = %.8e, refreshed_observed_fit = %.8e, refreshed_full_fit = %.8e\n",
              iter, final_metrics.observed_fit, final_metrics.full_fit,
              refreshed_metrics.observed_fit, refreshed_metrics.full_fit);

  if (!output_dir.empty()) {
    gputucker::WriteModelArtifacts(output_dir, tensor, core_tensor,
                                  factor_matrices, tucker_rank, final_metrics,
                                  &refreshed_metrics, original_eval_time,
                                  refreshed_core_time, refreshed_eval_time);
    std::printf("[GPUTucker Output]: wrote model artifacts to %s\n",
                output_dir.c_str());
  }

  MYPRINT("DONE\n");
}

} // namespace gputucker
} // namespace supertensor

#endif /* TUCKER_CUH_ */
