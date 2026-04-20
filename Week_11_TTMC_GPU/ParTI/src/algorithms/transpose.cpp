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
#include <type_traits>
#include <utility>
#include <ParTI/device.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memblock.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/utils.hpp>

#ifdef PARTI_USE_CBLAS
#include <cblas.h>
#endif

#ifdef PARTI_USE_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace pti {

namespace {

static_assert(
    std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value,
    "Transpose supports only float or double Scalar types"
);

#ifdef PARTI_USE_CUDA
template <typename T>
struct CublasGeam;

template <>
struct CublasGeam<float> {
    static cublasStatus_t run(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        const float* alpha,
        const float* A,
        int lda,
        const float* beta,
        const float* B,
        int ldb,
        float* C,
        int ldc
    ) {
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
};

template <>
struct CublasGeam<double> {
    static cublasStatus_t run(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        const double* alpha,
        const double* A,
        int lda,
        const double* beta,
        const double* B,
        int ldb,
        double* C,
        int ldc
    ) {
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
};
#endif

#ifdef PARTI_USE_OPENBLAS
template <typename T>
struct CblasOmatcopy;

template <>
struct CblasOmatcopy<float> {
    static void run(
        CBLAS_ORDER order,
        CBLAS_TRANSPOSE trans,
        int rows,
        int cols,
        float alpha,
        const float* a,
        int lda,
        float* b,
        int ldb
    ) {
        cblas_somatcopy(order, trans, rows, cols, alpha, a, lda, b, ldb);
    }
};

template <>
struct CblasOmatcopy<double> {
    static void run(
        CBLAS_ORDER order,
        CBLAS_TRANSPOSE trans,
        int rows,
        int cols,
        double alpha,
        const double* a,
        int lda,
        double* b,
        int ldb
    ) {
        cblas_domatcopy(order, trans, rows, cols, alpha, a, lda, b, ldb);
    }
};
#endif

}

void transpose_matrix_inplace(
    Tensor& X,
    bool do_transpose,
    bool want_fortran_style,
    Device *device
) {

    ptiCheckError(X.nmodes != 2, ERR_SHAPE_MISMATCH, "X.nmodes != 2");

    size_t* storage_order = X.storage_order(cpu);
    bool currently_fortran_style;
    if(storage_order[0] == 0 && storage_order[1] == 1) {
        currently_fortran_style = false;
    } else if(storage_order[0] == 1 && storage_order[1] == 0) {
        currently_fortran_style = true;
    } else {
        ptiCheckError(true, ERR_SHAPE_MISMATCH, "X is not a matrix");
    }

    size_t* shape = X.shape(cpu);
    size_t* strides = X.strides(cpu);

    if(do_transpose != (currently_fortran_style != want_fortran_style)) {
        size_t m = shape[storage_order[0]]; // Result rows
        size_t n = shape[storage_order[1]]; // Result cols
        size_t ldm = ceil_div<size_t>(m, 8) * 8;
        size_t ldn = strides[storage_order[1]];
        MemBlock<Scalar[]> result_matrix;
        result_matrix.allocate(device->mem_node, n * ldm);

        if(CudaDevice *cuda_device = dynamic_cast<CudaDevice *>(device)) {

#ifdef PARTI_USE_CUDA

            cudaSetDevice(cuda_device->cuda_device);
            cudaMemset(result_matrix(device->mem_node), 0, n * ldm * sizeof (Scalar));

            cublasHandle_t handle = (cublasHandle_t) cuda_device->GetCublasHandle();

            cublasStatus_t status = cublasSetPointerMode(
                handle,
                CUBLAS_POINTER_MODE_HOST
            );
            ptiCheckError(status, ERR_CUDA_LIBRARY, "cuBLAS error");

            Scalar const alpha = 1;
            Scalar const beta = 0;
            status = CublasGeam<Scalar>::run(
                handle,                             // handle
                CUBLAS_OP_T,                        // transa
                CUBLAS_OP_N,                        // transb
                m,                                  // m
                n,                                  // n
                &alpha,                             // alpha
                X.values(device->mem_node),         // A
                ldn,                                // lda
                &beta,                              // beta
                nullptr,                            // B
                ldm,                                // ldb
                result_matrix(device->mem_node),    // C
                ldm                                 // ldc
            );
            ptiCheckError(status, ERR_CUDA_LIBRARY, "cuBLAS error");

            cudaSetDevice(cuda_device->cuda_device);
            cudaDeviceSynchronize();

#else

            (void) cuda_device;
            ptiCheckError(true, ERR_BUILD_CONFIG, "CUDA not enabled");
#endif

        } else if(dynamic_cast<CpuDevice *>(device) != nullptr) {

            memset(result_matrix(device->mem_node), 0, n * ldm * sizeof (Scalar));

#ifdef PARTI_USE_OPENBLAS

            CblasOmatcopy<Scalar>::run(
                CblasColMajor,                          // CORDER
                CblasTrans,                             // CTRANS
                n,                                      // crows
                m,                                      // ccols
                1,                                      // calpha
                X.values(device->mem_node),             // a
                ldn,                                    // clda
                result_matrix(device->mem_node),        // b
                ldm                                     // cldb
            );

#else

            Scalar *result_matrix_values = result_matrix(device->mem_node);
            const Scalar *X_values = X.values(device->mem_node);
            for(size_t i = 0; i < n; ++i) {
                for(size_t j = 0; j < m; ++j) {
                    result_matrix_values[i * ldm + j] = X_values[j * ldn + i];
                }
            }

#endif

        } else {
            ptiCheckError(true, ERR_VALUE_ERROR, "Invalid device type");
        }

        strides[storage_order[0]] = ldm;
        X.values = std::move(result_matrix);
    }

    if(do_transpose) {
        std::swap(shape[0], shape[1]);
        std::swap(strides[0], strides[1]);
    }

    if(want_fortran_style) {
        storage_order[0] = 1;
        storage_order[1] = 0;
    } else {
        storage_order[0] = 0;
        storage_order[1] = 1;
    }

    assert(strides[0] >= shape[0]);
    assert(strides[1] >= shape[1]);

}

}
