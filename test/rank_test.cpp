#include "../include/matrix.hpp"
#include "test.hpp"

#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>

using mlinalg::conj_transpose;
using mlinalg::Dynamic;
using mlinalg::Index;
using mlinalg::Matrix;
using mlinalg::RowVector;
using mlinalg::transpose;
using mlinalg::Vector;

namespace {

using M23 = Matrix<double, 2, 3>;
using M33 = Matrix<double, 3, 3>;
using MXX = Matrix<double, Dynamic, Dynamic>;
using MFX = Matrix<float, Dynamic, Dynamic>;
using C   = std::complex<double>;
using MCX = Matrix<C, Dynamic, Dynamic>;
using MIX = Matrix<int, Dynamic, Dynamic>;
using MI3 = Matrix<int, 3, 3>;
using VX  = Vector<double>;
using RX  = RowVector<double>;

template <typename M>
concept HasRank = requires(const M& m) {
    m.rank();
    m.dim();
};

/// Deterministic integer test matrix with entries in [-bound, bound].
auto pseudo_random_int(Index rows, Index cols, unsigned seed, int bound) -> MIX {
    MIX      m(rows, cols);
    unsigned state = seed;
    for (int& c : m) {
        state = (state * 1103515245U) + 12345U;
        c     = static_cast<int>((state >> 8U) % static_cast<unsigned>((2 * bound) + 1)) - bound;
    }
    return m;
}

template <typename T> auto convert(const MIX& m) -> Matrix<T, Dynamic, Dynamic> {
    Matrix<T, Dynamic, Dynamic> result(m.rows(), m.cols());
    for (Index i = 0; i < m.rows(); ++i) {
        for (Index j = 0; j < m.cols(); ++j) {
            result(i, j) = static_cast<T>(m(i, j));
        }
    }
    return result;
}

/// rows x cols integer matrix of rank at most k: a product of rows x k and k x cols factors.
auto low_rank_int(Index rows, Index cols, Index k, unsigned seed) -> MIX {
    const MIX left  = pseudo_random_int(rows, k, seed, 3);
    const MIX right = pseudo_random_int(k, cols, seed + 77, 3);
    return left * right;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(HasRank<M23> && HasRank<M33> && HasRank<MXX> && HasRank<MFX> && HasRank<MCX>);
static_assert(HasRank<MIX> && HasRank<MI3> && HasRank<Matrix<std::int64_t, 2, 5>>);
static_assert(HasRank<VX> && HasRank<RX>);
static_assert(!HasRank<Matrix<unsigned, 3, 3>> && !HasRank<Matrix<bool, 2, 2>>);

static_assert(std::is_same_v<decltype(std::declval<const MCX&>().rank()), Index>);
static_assert(std::is_same_v<decltype(std::declval<const MIX&>().dim()), Index>);

// Integer ranks are exact and usable in constant expressions.
static_assert(MI3{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}.rank() == 2);
static_assert(MI3{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}.dim() == 1);
static_assert(MI3{{2, 0, 0}, {0, 3, 0}, {0, 0, 5}}.rank() == 3);
static_assert(Matrix<int, 2, 3>{{1, 2, 3}, {2, 4, 6}}.rank() == 1);
static_assert(Matrix<int, 2, 3>{{1, 2, 3}, {2, 4, 6}}.dim() == 2);
static_assert(MI3{}.rank() == 0 && MI3{}.dim() == 3);

// ------------------------------------------------------------------------------------------------
// Exact rank of integer matrices
// ------------------------------------------------------------------------------------------------

TEST(RankInteger, KnownMatrices) {
    EXPECT_EQ((MIX{{1, 2, 3}, {2, 4, 6}}.rank()), 1);
    EXPECT_EQ((MIX{{1, 2, 3, 4}, {2, 4, 6, 8}, {0, 1, 0, 1}, {5, 1, 2, 2}}.rank()), 3);
    EXPECT_EQ((MIX{{2, -1, 0, 3}, {1, 4, 2, -2}, {0, 3, -1, 1}, {5, 0, 2, 2}}.rank()), 4);
    EXPECT_EQ((MIX{{1, 2}, {3, 4}, {5, 6}, {7, 8}}.rank()), 2);
}

TEST(RankInteger, SkipsColumnsWithoutPivot) {
    // The first two columns contribute one pivot between them, found only in the last row.
    const MIX a{{0, 0, 1, 2}, {0, 0, 2, 4}, {0, 3, 1, 1}};
    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(a.dim(), 2);
}

TEST(RankInteger, ZeroAndEmpty) {
    EXPECT_EQ(MIX(3, 4).rank(), 0);
    EXPECT_EQ(MIX(3, 4).dim(), 4);
    EXPECT_EQ(MIX().rank(), 0);
    EXPECT_EQ(MIX().dim(), 0);
    EXPECT_EQ(MIX(0, 3).rank(), 0);
    EXPECT_EQ(MIX(0, 3).dim(), 3);
}

TEST(RankInteger, LowRankProducts) {
    for (Index k = 1; k <= 3; ++k) {
        SCOPED_TRACE(k);
        // Random factors of these sizes have full rank k (checked below against the float rank).
        EXPECT_EQ(low_rank_int(5, 4, k, 10 + static_cast<unsigned>(k)).rank(), k);
        EXPECT_EQ(low_rank_int(4, 6, k, 20 + static_cast<unsigned>(k)).rank(), k);
    }
}

// ------------------------------------------------------------------------------------------------
// Numerical rank of floating-point and complex matrices
// ------------------------------------------------------------------------------------------------

TEST(Rank, KnownMatrices) {
    // Rank 2 on paper; in floating point the third singular value is ~1e-16, not 0.
    const M33 a{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(a.dim(), 1);

    const M23 b{{1, 2, 3}, {2, 4, 6}};
    EXPECT_EQ(b.rank(), 1);
    EXPECT_EQ(b.dim(), 2);

    const MXX identity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    EXPECT_EQ(identity.rank(), 3);
    EXPECT_EQ(identity.dim(), 0);
}

TEST(Rank, ZeroAndEmpty) {
    EXPECT_EQ(MXX(3, 4).rank(), 0);
    EXPECT_EQ(MXX(3, 4).dim(), 4);
    EXPECT_EQ(MXX().rank(), 0);
    EXPECT_EQ(MXX().dim(), 0);
    EXPECT_EQ(MXX(0, 3).rank(), 0);
    EXPECT_EQ(MXX(0, 3).dim(), 3);
    EXPECT_EQ(MXX(3, 0).rank(), 0);
    EXPECT_EQ(MXX(3, 0).dim(), 0);
}

TEST(Rank, Vectors) {
    const VX col{0, 3, 0};
    EXPECT_EQ(col.rank(), 1);
    EXPECT_EQ(col.dim(), 0); // one column, independent

    const RX row{0, 3, 0};
    EXPECT_EQ(row.rank(), 1);
    EXPECT_EQ(row.dim(), 2); // three unknowns, one equation

    EXPECT_EQ(VX(4).rank(), 0);
    EXPECT_EQ(VX(4).dim(), 1);
}

TEST(Rank, AgreesWithExactIntegerRank) {
    for (Index k = 1; k <= 4; ++k) {
        for (unsigned seed = 1; seed <= 4; ++seed) {
            SCOPED_TRACE(testing::Message() << "k=" << k << " seed=" << seed);
            const MIX tall = low_rank_int(6, 4, k, (100 * seed) + static_cast<unsigned>(k));
            const MIX wide = low_rank_int(4, 7, k, (200 * seed) + static_cast<unsigned>(k));
            EXPECT_EQ(convert<double>(tall).rank(), tall.rank());
            EXPECT_EQ(convert<double>(wide).rank(), wide.rank());
            EXPECT_EQ(convert<float>(tall).rank(), tall.rank());
            EXPECT_EQ(convert<C>(wide).rank(), wide.rank());
        }
    }
}

TEST(Rank, RankNullityAndTranspose) {
    for (Index k = 0; k <= 3; ++k) {
        SCOPED_TRACE(k);
        const MXX a = k == 0
                          ? MXX(5, 3)
                          : convert<double>(low_rank_int(5, 3, k, 40 + static_cast<unsigned>(k)));
        EXPECT_EQ(a.rank() + a.dim(), a.cols());

        const MXX at = transpose(a);
        EXPECT_EQ(at.rank(), a.rank());
        EXPECT_EQ(at.rank() + at.dim(), at.cols());
    }
}

TEST(Rank, ThresholdIsRelativeToTheLargestSingularValue) {
    // Uniformly tiny or huge matrices keep their rank.
    const MXX tiny{{1e-30, 0}, {0, 2e-30}};
    EXPECT_EQ(tiny.rank(), 2);
    const MXX huge{{1e30, 0}, {0, 2e30}};
    EXPECT_EQ(huge.rank(), 2);

    // A singular value far above the threshold counts...
    const MXX clearly_independent{{1, 1}, {1, 1 + 1e-8}};
    EXPECT_EQ(clearly_independent.rank(), 2);
    // ...one below max(rows, cols) * epsilon * sigma_max does not.
    const MXX negligible{{1, 0}, {0, 1e-17}};
    EXPECT_EQ(negligible.rank(), 1);
    EXPECT_EQ(negligible.dim(), 1);
}

TEST(Rank, MatchesInvertibility) {
    const MXX regular{{4, 3, 2, 1}, {0, 1, 2, 3}, {1, 0, 1, 0}, {2, 2, 0, 1}}; // det 8
    EXPECT_EQ(regular.rank(), 4);
    EXPECT_TRUE(regular.lu().is_invertible());

    // Last row = row 0 - row 1 + row 2.
    const MXX singular{{1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1}};
    EXPECT_EQ(singular.rank(), 3);
    EXPECT_EQ(singular.dim(), 1);
    EXPECT_FALSE(singular.lu().is_invertible());

    // Flipping one sign makes it regular (det -2).
    const MXX flipped{{1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, -1}};
    EXPECT_EQ(flipped.rank(), 4);
}

TEST(Rank, ComplexScalars) {
    // Hermitian with eigenvalues 0 and 2: the second column is i times the first.
    const MCX hermitian{{C(1, 0), C(0, 1)}, {C(0, -1), C(1, 0)}};
    EXPECT_EQ(hermitian.rank(), 1);
    EXPECT_EQ(hermitian.dim(), 1);

    const MCX regular{{C(1, 1), C(2, 0)}, {C(0, -1), C(3, 2)}};
    EXPECT_EQ(regular.rank(), 2);

    // Complex rank-one outer product u * v^H.
    const MCX u{{C(1, 2)}, {C(0, -1)}, {C(3, 0)}};
    const MCX v{{C(2, -1)}, {C(1, 1)}, {C(0, 4)}, {C(-1, 0)}};
    const MCX outer = u * conj_transpose(v);
    ASSERT_EQ(outer.rows(), 3);
    ASSERT_EQ(outer.cols(), 4);
    EXPECT_EQ(outer.rank(), 1);
    EXPECT_EQ(outer.dim(), 3);
}

TEST(Rank, FloatScalars) {
    const MFX a{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    EXPECT_EQ(a.rank(), 2);
    const MFX b{{2, 0}, {0, 3}, {1, 1}};
    EXPECT_EQ(b.rank(), 2);
    EXPECT_EQ(b.dim(), 0);
}

TEST(Rank, DoesNotModifyTheMatrix) {
    const MXX                    a    = convert<double>(low_rank_int(4, 6, 2, 9));
    const MXX                    copy = a;
    [[maybe_unused]] const Index r    = a.rank();
    for (Index i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a.data()[i], copy.data()[i]);
    }
}
