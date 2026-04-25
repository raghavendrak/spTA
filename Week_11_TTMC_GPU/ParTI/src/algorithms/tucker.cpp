/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI/algorithm.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <ParTI/device.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memblock.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/utils.hpp>

#ifdef PARTI_USE_CUDA
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#endif

namespace pti {

namespace {

struct FactorFilePayload {
    uint64_t rows{0};
    uint64_t cols{0};
    std::vector<float> values;
};

FactorFilePayload read_factor_file_float32(
    const std::string& path,
    size_t expected_rows,
    size_t max_cols
) {
    std::ifstream fin(path, std::ios::binary);
    ptiCheckError(!fin, ERR_UNKNOWN, ("cannot open factors file: " + path).c_str());

    fin.seekg(0, std::ios::end);
    const std::streamsize file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    ptiCheckError(file_size < 0, ERR_UNKNOWN, ("cannot stat factors file: " + path).c_str());

    const std::streamsize bytes_per_row =
        static_cast<std::streamsize>(expected_rows * sizeof(float));

    auto read_values = [&](uint64_t rows, uint64_t cols, std::streamsize offset) {
        FactorFilePayload payload;
        payload.rows = rows;
        payload.cols = cols;
        payload.values.resize(static_cast<size_t>(rows * cols));
        fin.seekg(offset, std::ios::beg);
        fin.read(reinterpret_cast<char*>(payload.values.data()),
                 static_cast<std::streamsize>(payload.values.size() * sizeof(float)));
        ptiCheckError(!fin, ERR_UNKNOWN, ("short read on factors file: " + path).c_str());
        return payload;
    };

    if(file_size >= static_cast<std::streamsize>(2 * sizeof(uint64_t))) {
        uint64_t fr = 0, fc = 0;
        fin.read(reinterpret_cast<char*>(&fr), sizeof(uint64_t));
        fin.read(reinterpret_cast<char*>(&fc), sizeof(uint64_t));
        ptiCheckError(!fin, ERR_UNKNOWN, ("failed to read factors header: " + path).c_str());

        const std::streamsize header_bytes =
            static_cast<std::streamsize>(2 * sizeof(uint64_t) + fr * fc * sizeof(float));
        if(fr == expected_rows &&
           fc > 0 &&
           fc <= max_cols &&
           header_bytes == file_size) {
            return read_values(fr, fc, static_cast<std::streamsize>(2 * sizeof(uint64_t)));
        }
        fin.clear();
    }

    ptiCheckError(bytes_per_row <= 0 || file_size % bytes_per_row != 0,
                  ERR_SHAPE_MISMATCH,
                  ("factor file size mismatch for " + path).c_str());

    const uint64_t inferred_cols = static_cast<uint64_t>(file_size / bytes_per_row);
    ptiCheckError(inferred_cols == 0 || inferred_cols > max_cols,
                  ERR_SHAPE_MISMATCH,
                  ("factor column mismatch for " + path).c_str());

    return read_values(static_cast<uint64_t>(expected_rows), inferred_cols, 0);
}

template <typename Scalar>
void extend_factor_columns(
    Scalar *temp,
    size_t rows,
    size_t stride,
    size_t start_col,
    size_t end_col,
    unsigned mode_idx
) {
    std::mt19937 gen(42 + static_cast<unsigned>(mode_idx));
    std::normal_distribution<Scalar> dist((Scalar)0, (Scalar)1);
    for(size_t c = start_col; c < end_col; ++c) {
        for(size_t r = 0; r < rows; ++r) {
            temp[r * stride + c] = dist(gen);
        }
        for(size_t k = 0; k < c; ++k) {
            Scalar dot = 0;
            for(size_t r = 0; r < rows; ++r) {
                dot += temp[r * stride + k] * temp[r * stride + c];
            }
            for(size_t r = 0; r < rows; ++r) {
                temp[r * stride + c] -= dot * temp[r * stride + k];
            }
        }
        Scalar norm = 0;
        for(size_t r = 0; r < rows; ++r) {
            norm += temp[r * stride + c] * temp[r * stride + c];
        }
        norm = std::sqrt(norm);
        if(norm < (Scalar)1e-10) norm = (Scalar)1;
        for(size_t r = 0; r < rows; ++r) {
            temp[r * stride + c] /= norm;
        }
    }
}

} // namespace

void init_factor_orthonormal(
    Tensor&   mtx,
    size_t    mode_idx
) {
    ptiCheckError(mtx.nmodes != 2, ERR_SHAPE_MISMATCH, "mtx.nmodes != 2");
    ptiCheckError(mtx.storage_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "mtx.storage_order[0] != 0");
    ptiCheckError(mtx.storage_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "mtx.storage_order[1] != 1");

    // ParTI stores factor matrix as R_n x I_n (row-major).
    // Reference stores it as I_n x R_n. Generate in reference layout, then transpose.
    size_t R_n = mtx.shape(cpu)[0];
    size_t I_n = mtx.shape(cpu)[1];
    size_t stride = mtx.strides(cpu)[1];

    std::unique_ptr<Scalar[]> temp(new Scalar[I_n * R_n]);

    const char* factors_dir = std::getenv("TUCKER_FACTORS_DIR");
    if(factors_dir && *factors_dir) {
        std::string path = std::string(factors_dir) + "/mode" + std::to_string(mode_idx) + ".bin";
        FactorFilePayload payload = read_factor_file_float32(path, I_n, R_n);
        ptiCheckError(payload.rows != I_n || payload.cols == 0 || payload.cols > R_n,
                      ERR_SHAPE_MISMATCH,
                      ("factor shape mismatch for " + path).c_str());

        const size_t load_cols = static_cast<size_t>(payload.cols);
        for(size_t r = 0; r < I_n; ++r) {
            for(size_t c = 0; c < load_cols; ++c) {
                temp[r * R_n + c] =
                    static_cast<Scalar>(payload.values[r * load_cols + c]);
            }
        }
        if(load_cols < R_n) {
            extend_factor_columns(temp.get(), I_n, R_n, load_cols, R_n,
                                  static_cast<unsigned>(mode_idx));
            std::printf("[factors] loaded %s (%zu x %zu) and extended to %zu cols\n",
                        path.c_str(), I_n, load_cols, R_n);
        } else {
            std::printf("[factors] loaded %s (%zu x %zu)\n", path.c_str(), I_n, R_n);
        }
    } else {
        std::mt19937 gen(42 + static_cast<unsigned>(mode_idx));
        std::normal_distribution<Scalar> dist((Scalar)0, (Scalar)1);
        for(size_t i = 0; i < I_n * R_n; ++i) temp[i] = dist(gen);

        // Gram-Schmidt orthonormalization on columns of I_n x R_n matrix
        for(size_t c = 0; c < R_n; ++c) {
            Scalar norm = 0;
            for(size_t r = 0; r < I_n; ++r) norm += temp[r * R_n + c] * temp[r * R_n + c];
            norm = std::sqrt(norm);
            if(norm < (Scalar)1e-10) norm = (Scalar)1;
            for(size_t r = 0; r < I_n; ++r) temp[r * R_n + c] /= norm;
            for(size_t c2 = c + 1; c2 < R_n; ++c2) {
                Scalar dot = 0;
                for(size_t r = 0; r < I_n; ++r) dot += temp[r * R_n + c] * temp[r * R_n + c2];
                for(size_t r = 0; r < I_n; ++r) temp[r * R_n + c2] -= dot * temp[r * R_n + c];
            }
        }
    }

    // Transpose into ParTI's R_n x I_n layout
    Scalar* values = mtx.values(cpu);
    for(size_t i = 0; i < I_n; ++i) {
        for(size_t j = 0; j < R_n; ++j) {
            values[j * stride + i] = temp[i * R_n + j];
        }
    }
    for(size_t j = 0; j < R_n; ++j) {
        for(size_t i = I_n; i < stride; ++i) {
            values[j * stride + i] = 0;
        }
    }
    for(size_t i = R_n * stride; i < mtx.chunk_size; ++i) {
        values[i] = 0;
    }
}

Tensor nvecs(
    SparseTensor& t,
    size_t        n,
    size_t        r,
    Device*       device
) {

    Tensor tm = unfold(t, n);
    Tensor u, s;

    // device = session.devices[cpu]; // Experiments show that cuSOLVER is slow when M >> N
    svd(&u, false, true, s, nullptr, false, false, tm, device);
    size_t const* u_shape = u.shape(cpu);
    size_t u_nrows = u_shape[0];
    size_t u_ncols = u_shape[1];
    assert(u_nrows == t.shape(cpu)[n]);

    size_t const result_shape[2] = { u_nrows, r };
    Tensor result(2, result_shape);
    size_t result_m = result_shape[0];
    size_t result_n = result_shape[1];
    size_t result_stride = result.strides(cpu)[1];
    size_t u_stride = u.strides(cpu)[1];

    for(size_t i = 0; i < result_m; ++i) {
        for(size_t j = 0; j < std::min(result_n, u_ncols); ++j) {
            result.values(cpu)[i * result_stride + j] = u.values(cpu)[i * u_stride + j];
        }
    }

    return result;

}

SparseTensor tucker_decomposition(
    SparseTensor&   X,
    size_t const    R[],
    size_t const    dimorder[],
    Device*         device,
    Device*         solve_device,
    enum tucker_decomposition_init_type init,
    double          tol,
    unsigned        maxiters
) {
    ptiCheckError(X.dense_order.size() != 0, ERR_SHAPE_MISMATCH, "X should be fully sparse");

    if(solve_device == nullptr) {
        solve_device = device;
    }

    size_t N = X.nmodes;
    double normX = X.norm(device);

    // Rank truncation per mode: R_n <= I_n (orthonormal columns in R^{I_n}).
    std::vector<size_t> R_cap(N);
    for(size_t n = 0; n < N; ++n) {
        R_cap[n] = R[n];
        size_t I_n = X.shape(cpu)[n];
        if(R_cap[n] > I_n) {
            std::printf("  [truncate] R[%zu] %zu -> %zu (dim[%zu])\n",
                        n, R[n], I_n, n);
            R_cap[n] = I_n;
        }
    }

    std::unique_ptr<Tensor[]> U(new Tensor[N]);
    size_t U_shape[2];
    for(size_t ni = 1; ni < N; ++ni) {
        size_t n = dimorder[ni];
        U_shape[0] = R_cap[n];
        U_shape[1] = X.shape(cpu)[n];
        U[n].reset(2, U_shape);
        if(false && init == TUCKER_INIT_NVECS) {
            U[n] = nvecs(X, n, R_cap[n], device);
        } else {
            init_factor_orthonormal(U[n], n);
        }
    }
    SparseTensor core;

    std::unique_ptr<size_t []> sort_order(new size_t [N]);
    std::unique_ptr<SparseTensor []> X_sort_cache(new SparseTensor [N]);
    Timer timer_sort(cpu);
    timer_sort.start();
    for(size_t n = 0; n < N; ++n) {
        Timer timer_sort_i(cpu);
        timer_sort_i.start();
        for(size_t m = 0; m < N; ++m) {
            if(m < n) {
                sort_order[N - m - 1] = m;
            } else if(m != n) {
                sort_order[N - m] = m;
            }
        }
        sort_order[0] = n;
        X_sort_cache[n] = X.clone();
        X_sort_cache[n].sort_index(sort_order.get());
        timer_sort_i.stop();
        timer_sort_i.print_elapsed_time("Tucker Sort");
    }
    timer_sort.stop();
    timer_sort.print_elapsed_time("Tucker Sort Total");


    double fit = 0;
    SparseTensor Utilde_next;
    unsigned iterations_run = 0;
    for(unsigned iter = 0; iter < maxiters; ++iter) {
        auto iter_wall_start = std::chrono::high_resolution_clock::now();
        Timer timer_iter(cpu);
        timer_iter.start();

        double fitold = fit;

        Timer timer_loop(cpu);
        timer_loop.start();
        SparseTensor* Utilde = &X;
        for(size_t ni = 0; ni < N; ++ni) {
            std::printf("\n");
            size_t n = dimorder[ni];

            Timer timer_ttm_chain(cpu);
            timer_ttm_chain.start();
            Utilde = &X_sort_cache[n];
            for(size_t m = 0; m < N; ++m) {
                if(m != n) {
                    std::printf("[Tucker Decomp]: Iter %u, n = %zu, m = %zu\n", iter, n, m);
                    std::fflush(stdout);
                    Utilde_next = tensor_times_matrix(*Utilde, U[m], m, device, true);
                    Utilde = &Utilde_next;
                }
            }
            timer_ttm_chain.stop();
            timer_ttm_chain.print_elapsed_time("TTM Chain");

            if(device->mem_node != cpu) {
                std::printf("[Tucker TTM Chain]: Releasing GPU memory of X_sort_cache[%zu]\n", n);
                X_sort_cache[n].values.mark_dirty(cpu);
                X_sort_cache[n].values.free(device->mem_node);
                for(size_t m = 0; m < N; ++m) {
                    X_sort_cache[n].indices[m].mark_dirty(cpu);
                    X_sort_cache[n].indices[m].free(device->mem_node);
                }
            }

            Timer timer_svd(solve_device->device_id);
            timer_svd.start();
            // Mode n is sparse, while other modes are dense
            U[n] = nvecs(*Utilde, n, R_cap[n], solve_device);
            timer_svd.stop();
            timer_svd.print_elapsed_time("SVD");

            transpose_matrix_inplace(U[n], true, false, solve_device);
        }   // End loop of nmodes
        timer_loop.stop();
        timer_loop.print_elapsed_time("Tucker Decomp Loop");

        std::printf("\n");
        std::fflush(stdout);

        Timer timer_core(cpu);
        timer_core.start();
        core = tensor_times_matrix(*Utilde, U[dimorder[N-1]], dimorder[N-1], device, true);
        timer_core.stop();
        timer_core.print_elapsed_time("Tucker Decomp Core");

        Timer timer_fit(cpu);
        timer_fit.start();
        double normCore = core.norm(device);
        double normResidual = std::sqrt(normX * normX - normCore * normCore);
        fit = 1 - normResidual / normX;
        double fitchange = std::fabs(fitold - fit);
        timer_fit.stop();
        timer_fit.print_elapsed_time("Tucker Decomp Norm");

        std::printf("[Tucker Dcomp]: normX = %lg, normCore = %lg\n", normX, normCore);
        std::printf("[Tucker Dcomp]: fit = %lg, fitchange = %lg\n", fit, fitchange);
        std::fflush(stdout);

        iterations_run = iter + 1;

        auto iter_wall_end = std::chrono::high_resolution_clock::now();
        auto iter_wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
            iter_wall_end - iter_wall_start).count();
        std::printf("[Tucker Dcomp IterTime]: iter = %u, runtime_us = %lld\n",
                    iter, static_cast<long long>(iter_wall_us));
        std::fflush(stdout);

        timer_iter.stop();
        timer_iter.print_elapsed_time("Tucker Decomp Iter");
        if(iter != 0 && fitchange < tol) {
            break;
        }
    }   // End of iterations

    std::printf("[Tucker Dcomp Summary]: iterations = %u, final_fit = %lg\n",
                iterations_run, fit);
    std::fflush(stdout);

    return core;
}

}
