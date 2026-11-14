#include "../include/gemm/microkernel.hpp"
#include "../include/matrix_storage.hpp"
#include "test.hpp"

#include <cstddef>
#include <vector>

using mlinalg::Index;
using mlinalg::detail::gemm_kernel_shape;
using mlinalg::detail::gemm_microkernel;

namespace {

template <typename T> using aligned_vector = std::vector<T, mlinalg::aligned_allocator<T>>;

/// Small integers in [-4, 4]: every product and partial sum below is exact even in float, so the
/// kernel can be compared for equality whatever order it accumulates in.
template <typename T> auto make_value(Index i, int salt) -> T {
    return static_cast<T>((((i * 7) + (salt * 13)) % 9) - 4);
}

/// Marks C coefficients outside the tile, which the kernel must not touch.
template <typename T> constexpr T sentinel = static_cast<T>(-99);

/// Runs gemm_microkernel<T, Packets, NR> on packed slivers of depth kc into a C tile with leading
/// dimension ldc that already holds values, and checks it against a scalar reference.
template <typename T, int Packets, int NR> void check_kernel(Index kc, Index ldc) {
    constexpr Index mr = Index{Packets} * mlinalg::Packet<T>::size;
    constexpr Index nr = NR;
    ASSERT_GE(ldc, mr);

    aligned_vector<T> ap(static_cast<std::size_t>(mr * kc));
    aligned_vector<T> bp(static_cast<std::size_t>(kc * nr));
    for (Index i = 0; i < mr * kc; ++i) {
        ap[i] = make_value<T>(i, 1);
    }
    for (Index i = 0; i < kc * nr; ++i) {
        bp[i] = make_value<T>(i, 2);
    }

    // Offset by one so C is not aligned: the kernel must only use unaligned accesses on C.
    std::vector<T> storage(static_cast<std::size_t>((ldc * nr) + 1), sentinel<T>);
    T*             c = storage.data() + 1;
    for (Index j = 0; j < nr; ++j) {
        for (Index i = 0; i < mr; ++i) {
            c[(j * ldc) + i] = make_value<T>((j * mr) + i, 3);
        }
    }

    std::vector<T> expected(storage);
    for (Index j = 0; j < nr; ++j) {
        for (Index i = 0; i < mr; ++i) {
            T sum = expected[static_cast<std::size_t>((j * ldc) + i + 1)];
            for (Index k = 0; k < kc; ++k) {
                sum += ap[(k * mr) + i] * bp[(k * nr) + j];
            }
            expected[static_cast<std::size_t>((j * ldc) + i + 1)] = sum;
        }
    }

    gemm_microkernel<T, Packets, NR>(kc, ap.data(), bp.data(), c, ldc);

    for (std::size_t n = 0; n < storage.size(); ++n) {
        ASSERT_EQ(storage[n], expected[n]) << "kc " << kc << ", ldc " << ldc << ", offset " << n;
    }
}

template <typename T> class GemmMicrokernelTest : public ::testing::Test {};

using GemmScalars = ::testing::Types<float, double, int>;
TYPED_TEST_SUITE(GemmMicrokernelTest, GemmScalars);

constexpr Index depths[] = {0, 1, 2, 3, 7, 64, 257};

} // namespace

TYPED_TEST(GemmMicrokernelTest, DefaultShapeAccumulatesIntoTile) {
    using T     = TypeParam;
    using shape = gemm_kernel_shape<T>;
    for (const Index kc : depths) {
        check_kernel<T, shape::packets, shape::nr>(kc, shape::mr);
    }
}

TYPED_TEST(GemmMicrokernelTest, DefaultShapeLeavesPaddingRowsUntouched) {
    using T     = TypeParam;
    using shape = gemm_kernel_shape<T>;
    for (const Index kc : depths) {
        check_kernel<T, shape::packets, shape::nr>(kc, shape::mr + 3);
    }
}

TYPED_TEST(GemmMicrokernelTest, OtherShapes) {
    using T = TypeParam;
    for (const Index kc : depths) {
        check_kernel<T, 1, 1>(kc, mlinalg::Packet<T>::size);
        check_kernel<T, 3, 5>(kc, (3 * mlinalg::Packet<T>::size) + 1);
    }
}
