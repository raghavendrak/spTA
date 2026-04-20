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
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <ParTI/device.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memblock.hpp>
#include <ParTI/tensor.hpp>
#include <ParTI/utils.hpp>
#include <ParTI/timer.hpp>

#ifdef PARTI_USE_LAPACKE
#include <lapacke.h>
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
    "SVD supports only float or double Scalar types"
);

void init_matrix(
    Tensor& X,
    size_t nrows,
    size_t ncols,
    bool fortran_style = true,
    bool initialize = true
) {
    size_t shape[2] = { nrows, ncols };
    X.reset(2, shape, initialize);
    size_t* storage_order = X.storage_order(cpu);
    if(fortran_style) {
        storage_order[0] = 1;
        storage_order[1] = 0;
    }
}

#ifdef PARTI_USE_CUDA
template <typename T>
struct CusolverGesvd;

template <>
struct CusolverGesvd<float> {
    static cusolverStatus_t buffer_size(cusolverDnHandle_t handle, int m, int n, int* lwork) {
        return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
    }

    static cusolverStatus_t run(
        cusolverDnHandle_t handle,
        signed char jobu,
        signed char jobvt,
        int m,
        int n,
        float* A,
        int lda,
        float* S,
        float* U,
        int ldu,
        float* VT,
        int ldvt,
        float* work,
        int lwork,
        float* rwork,
        int* devInfo
    ) {
        return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
    }
};

template <>
struct CusolverGesvd<double> {
    static cusolverStatus_t buffer_size(cusolverDnHandle_t handle, int m, int n, int* lwork) {
        return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
    }

    static cusolverStatus_t run(
        cusolverDnHandle_t handle,
        signed char jobu,
        signed char jobvt,
        int m,
        int n,
        double* A,
        int lda,
        double* S,
        double* U,
        int ldu,
        double* VT,
        int ldvt,
        double* work,
        int lwork,
        double* rwork,
        int* devInfo
    ) {
        return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
    }
};
#endif

#ifdef PARTI_USE_LAPACKE
template <typename T>
struct LapackeGesvd;

template <>
struct LapackeGesvd<float> {
    static lapack_int run(
        int matrix_layout,
        char jobu,
        char jobvt,
        lapack_int m,
        lapack_int n,
        float* a,
        lapack_int lda,
        float* s,
        float* u,
        lapack_int ldu,
        float* vt,
        lapack_int ldvt,
        float* superb
    ) {
        return LAPACKE_sgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    }
};

template <>
struct LapackeGesvd<double> {
    static lapack_int run(
        int matrix_layout,
        char jobu,
        char jobvt,
        lapack_int m,
        lapack_int n,
        double* a,
        lapack_int lda,
        double* s,
        double* u,
        lapack_int ldu,
        double* vt,
        lapack_int ldvt,
        double* superb
    ) {
        return LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    }
};
#endif

}

void svd(
    Tensor* U,
    bool U_want_transpose,
    bool U_want_minimal,
    Tensor& S,
    Tensor* V,
    bool V_want_transpose,
    bool V_want_minimal,
    Tensor& X,
    Device* device
) {

    size_t const* X_shape = X.shape(cpu);

    bool X_transposed = X_shape[0] < X_shape[1];
    transpose_matrix_inplace(X, X_transposed, true, device);
    if(X_transposed) {
        std::swap(U, V);
        std::swap(U_want_transpose, V_want_transpose);
        std::swap(U_want_minimal, V_want_minimal);
    }

    size_t svd_m = X_shape[0];
    size_t svd_n = X_shape[1];
    size_t svd_lda = X.strides(cpu)[0];

    assert(svd_m >= svd_n);
    assert(svd_m >= 1);
    assert(svd_lda >= svd_m);
    assert(svd_n >= 1);

    if(U != nullptr) {
        if(U_want_minimal) {
            init_matrix(*U, svd_m, svd_n, true, true);
        } else {
            init_matrix(*U, svd_m, svd_m, true, true);
        }
    }
    init_matrix(S, 1, svd_n, false, true);
    if(V != nullptr) {
        init_matrix(*V, svd_n, svd_n, true, true);
    }
    size_t svd_ldu = U ? U->strides(cpu)[0] : svd_m;
    size_t svd_ldvt = V ? V->strides(cpu)[0] : svd_n;

    assert(svd_ldu >= svd_m);
    assert(svd_ldu >= 1);
    assert(svd_ldvt >= svd_n);
    assert(svd_ldvt >= 1);

    if(CudaDevice *cuda_device = dynamic_cast<CudaDevice *>(device)) {

#ifdef PARTI_USE_CUDA

        size_t const int_max = static_cast<size_t>(std::numeric_limits<int>::max());
        ptiCheckError(
            svd_m > int_max || svd_n > int_max || svd_lda > int_max || svd_ldu > int_max || svd_ldvt > int_max,
            ERR_SHAPE_MISMATCH,
            "SVD matrix dimensions exceed int range for cuSOLVER"
        );

        int const cuda_m = static_cast<int>(svd_m);
        int const cuda_n = static_cast<int>(svd_n);
        int const cuda_lda = static_cast<int>(svd_lda);
        int const cuda_ldu = static_cast<int>(svd_ldu);
        int const cuda_ldvt = static_cast<int>(svd_ldvt);

        cusolverDnHandle_t handle = (cusolverDnHandle_t) cuda_device->GetCusolverDnHandle();
        cusolverStatus_t status;

        int svd_work_size;
        status = CusolverGesvd<Scalar>::buffer_size(
            handle,                                // handle
            cuda_m,                                // m
            cuda_n,                                // n
            &svd_work_size                         // lwork
        );
        ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");

        MemBlock<Scalar[]> svd_work;
        svd_work.allocate(device->mem_node, svd_work_size);
        MemBlock<Scalar[]> svd_rwork;
        svd_rwork.allocate(device->mem_node, std::min(svd_m, svd_n) - 1);
        MemBlock<int> svd_devInfo;
        svd_devInfo.allocate(device->mem_node);

        Timer timer_svd(device->device_id);
        timer_svd.start();
        status = CusolverGesvd<Scalar>::run(
            handle,                                     // handle
            U ? U_want_minimal ? 'S' : 'A' : 'N',       // jobu
            V ? 'A' : 'N',                              // jobvt
            cuda_m,                                     // m
            cuda_n,                                     // n
            X.values(device->mem_node),                 // A
            cuda_lda,                                   // lda (lda >= max(1, m))
            S.values(device->mem_node),                 // S
            U ? U->values(device->mem_node) : nullptr,  // U
            cuda_ldu,                                   // ldu
            V ? V->values(device->mem_node) : nullptr,  // VT
            cuda_ldvt,                                  // ldvt
            svd_work(device->mem_node),                 // work
            svd_work_size,                              // lwork
            svd_rwork(device->mem_node),                // rwork
            svd_devInfo(device->mem_node)               // devInfo
        );
        ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");
        
        cudaSetDevice(cuda_device->cuda_device);
        cudaDeviceSynchronize();
        timer_svd.stop();
        timer_svd.print_elapsed_time("cusolver gesvd");

        int svd_devInfo_value = *svd_devInfo(cpu);
        ptiCheckError(svd_devInfo_value != 0, ERR_CUDA_LIBRARY, ("devInfo = " + std::to_string(svd_devInfo_value)).c_str());

#else

        (void) cuda_device;
        ptiCheckError(true, ERR_BUILD_CONFIG, "CUDA not enabled");

#endif

    } else if(dynamic_cast<CpuDevice *>(device) != nullptr) {

#ifdef PARTI_USE_LAPACKE

        size_t lapack_int_max = (size_t)std::numeric_limits<lapack_int>::max();
        ptiCheckError(svd_m > lapack_int_max || svd_n > lapack_int_max || svd_lda > lapack_int_max || svd_ldu > lapack_int_max || svd_ldvt > lapack_int_max,
            ERR_SHAPE_MISMATCH, "SVD matrix dimensions exceed lapack_int range");

        lapack_int const lapack_m = static_cast<lapack_int>(svd_m);
        lapack_int const lapack_n = static_cast<lapack_int>(svd_n);
        lapack_int const lapack_lda = static_cast<lapack_int>(svd_lda);
        lapack_int const lapack_ldu = static_cast<lapack_int>(svd_ldu);
        lapack_int const lapack_ldvt = static_cast<lapack_int>(svd_ldvt);

        MemBlock<Scalar[]> svd_superb;
        svd_superb.allocate(device->mem_node, std::min(svd_m, svd_n) - 1);

        Timer timer_svd(cpu);
        timer_svd.start();
        lapack_int status = LapackeGesvd<Scalar>::run(
            LAPACK_COL_MAJOR,                           // matrix_layout
            U ? U_want_minimal ? 'S' : 'A' : 'N',       // jobu
            V ? 'A' : 'N',                              // jobvt
            lapack_m,                                   // m
            lapack_n,                                   // n
            X.values(device->mem_node),                 // a
            lapack_lda,                                 // lda
            S.values(device->mem_node),                 // s
            U ? U->values(device->mem_node) : nullptr,  // U
            lapack_ldu,                                 // ldu
            V ? V->values(device->mem_node) : nullptr,  // VT
            lapack_ldvt,                                // ldvt
            svd_superb(device->mem_node)                // superb
        );
        ptiCheckError(status, ERR_LAPACK_LIBRARY, "LAPACKE error");
        timer_svd.stop();
        timer_svd.print_elapsed_time("lapack gesvd");

#else

        ptiCheckError(true, ERR_BUILD_CONFIG, "LAPACKE not enabled");

#endif

    } else {
        ptiCheckError(true, ERR_VALUE_ERROR, "Invalid device type");
    }

    if(U != nullptr) {
        // After the optional wide-matrix swap, U already names the buffer whose
        // logical contents should satisfy U_want_transpose; only the storage
        // layout still needs normalization to row-major.
        transpose_matrix_inplace(*U, U_want_transpose, false, device);
    }
    if(V != nullptr) {
        // cuSOLVER/LAPACK always return VT. After the optional wide-matrix
        // swap, V still needs the same "VT -> V unless transpose requested"
        // handling independent of X_transposed.
        transpose_matrix_inplace(*V, !V_want_transpose, false, device);
    }
}

}
