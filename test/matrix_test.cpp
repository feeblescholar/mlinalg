#include "../include/ast/ast.hpp"
#include "../include/matrix.hpp"
#include "test.hpp"

#include <complex>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

using mlinalg::AddOp;
using mlinalg::Dynamic;
using mlinalg::Expression;
using mlinalg::Index;
using mlinalg::Matrix;
using mlinalg::MatrixLeaf;
using mlinalg::MatrixWrapper;
using mlinalg::MulOp;
using mlinalg::RowVector;
using mlinalg::ScalarMulOp;
using mlinalg::SubOp;
using mlinalg::Vector;

namespace {

using M22  = Matrix<double, 2, 2>;
using M23  = Matrix<double, 2, 3>;
using M32  = Matrix<double, 3, 2>;
using MX3  = Matrix<double, Dynamic, 3>;
using MXX  = Matrix<double, Dynamic, Dynamic>;
using MF23 = Matrix<float, 2, 3>;
using V3   = Matrix<double, 3, 1>;
using VX   = Vector<double>;
using RX   = RowVector<double>;

auto is_aligned(const void* ptr, std::size_t alignment = 32) -> bool {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

/// Compare every coefficient against an expected row-major list.
template <typename M> void expect_coeffs(const M& m, std::initializer_list<double> expected) {
    ASSERT_EQ(static_cast<std::size_t>(m.rows() * m.cols()), expected.size());
    const auto* it = expected.begin();
    for (Index i = 0; i < m.rows(); ++i) {
        for (Index j = 0; j < m.cols(); ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), *it++) << "at (" << i << ", " << j << ")";
        }
    }
}

template <typename M> auto linear(const M& m) -> std::vector<typename M::value_type> {
    return {m.begin(), m.end()};
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Compile-time properties
// ------------------------------------------------------------------------------------------------

static_assert(M23::RowsAtCompileTime == 2 && M23::ColsAtCompileTime == 3);
static_assert(MX3::RowsAtCompileTime == Dynamic && MX3::ColsAtCompileTime == 3);
static_assert(std::is_same_v<MF23::value_type, float>);

// Vectors are matrices with one dimension fixed to 1.
static_assert(std::is_same_v<VX, Matrix<double, Dynamic, 1>>);
static_assert(std::is_same_v<RX, Matrix<double, 1, Dynamic>>);
static_assert(VX::IsVector && RX::IsVector && V3::IsVector && !M23::IsVector && !MXX::IsVector);

// A Matrix can be wrapped as an AST leaf.
static_assert(MatrixLeaf<M23> && MatrixLeaf<MXX> && MatrixLeaf<VX>);
static_assert(Expression<MatrixWrapper<M23>> && Expression<MatrixWrapper<MXX>>);

static_assert(alignof(M22) >= 32 && alignof(MXX) >= 32);

// Sized constructors exist only where a dimension is Dynamic.
static_assert(std::is_constructible_v<MXX, Index, Index>);
static_assert(std::is_constructible_v<MX3, Index, Index>);
static_assert(!std::is_constructible_v<M23, Index, Index>);
static_assert(std::is_constructible_v<VX, Index> && std::is_constructible_v<RX, Index>);
static_assert(!std::is_constructible_v<V3, Index> && !std::is_constructible_v<MXX, Index>);
static_assert(!std::is_convertible_v<Index, VX>); // the length constructor is explicit

// Nested lists for matrices, flat lists for vectors.
static_assert(std::is_constructible_v<MXX, std::initializer_list<std::initializer_list<double>>>);
static_assert(!std::is_constructible_v<MXX, std::initializer_list<double>>);
static_assert(std::is_constructible_v<VX, std::initializer_list<double>>);
static_assert(!std::is_constructible_v<VX, std::initializer_list<std::initializer_list<double>>>);

// Evaluation constructor and assignment: same scalar type and statically compatible shape.
static_assert(std::is_convertible_v<AddOp<MatrixWrapper<M23>, MatrixWrapper<M23>>, M23>);
static_assert(std::is_convertible_v<AddOp<MatrixWrapper<M23>, MatrixWrapper<M23>>, MXX>);
static_assert(std::is_convertible_v<MulOp<MatrixWrapper<MX3>, MatrixWrapper<M32>>, M22>);
static_assert(!std::is_constructible_v<M32, AddOp<MatrixWrapper<M23>, MatrixWrapper<M23>>>);
static_assert(!std::is_constructible_v<MF23, AddOp<MatrixWrapper<M23>, MatrixWrapper<M23>>>);
static_assert(std::is_assignable_v<MXX&, MulOp<MatrixWrapper<M23>, MatrixWrapper<M32>>>);
static_assert(!std::is_assignable_v<M23&, MulOp<MatrixWrapper<M23>, MatrixWrapper<M32>>>);

// Wrapping a temporary is still rejected.
static_assert(!std::is_constructible_v<MatrixWrapper<M23>, M23&&>);

// ------------------------------------------------------------------------------------------------
// Constexpr support (fixed-size matrices)
// ------------------------------------------------------------------------------------------------

static_assert([] {
    const M23 m{{1, 2, 3}, {4, 5, 6}};
    // Column-major storage.
    const double* d = m.data();
    return m.rows() == 2 && m.cols() == 3 && m(1, 0) == 4.0 && m(0, 2) == 3.0 && d[0] == 1.0 &&
           d[1] == 4.0 && d[2] == 2.0 && d[5] == 6.0;
}());

static_assert([] {
    const M22 a{{1, 2}, {3, 4}};
    const M22 b{{5, 6}, {7, 8}};
    const M22 c =
        SubOp(MulOp(MatrixWrapper(a), MatrixWrapper(b)), ScalarMulOp(2.0, MatrixWrapper(a)));
    // a * b = [[19, 22], [43, 50]]
    return c(0, 0) == 17.0 && c(0, 1) == 18.0 && c(1, 0) == 37.0 && c(1, 1) == 42.0;
}());

static_assert([] {
    M22 m;
    m(1, 0) = 5.0;
    return m(0, 0) == 0.0 && m(1, 0) == 5.0 && m.data()[1] == 5.0;
}());

// ------------------------------------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------------------------------------

TEST(Matrix, DefaultFixedIsZero) {
    const M23 m;
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m.size(), 6);
    expect_coeffs(m, {0, 0, 0, 0, 0, 0});
}

TEST(Matrix, DefaultDynamicIsEmpty) {
    const MXX m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.size(), 0);
    EXPECT_EQ(m.begin(), m.end());
}

TEST(Matrix, DefaultKeepsFixedDimension) {
    const MX3 m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 3);

    const VX v;
    EXPECT_EQ(v.rows(), 0);
    EXPECT_EQ(v.cols(), 1);
}

TEST(Matrix, SizedIsZero) {
    const MXX m(3, 4);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    for (const double c : m) {
        EXPECT_EQ(c, 0.0);
    }

    const MX3 p(5, 3);
    EXPECT_EQ(p.rows(), 5);
    EXPECT_EQ(p.cols(), 3);
}

TEST(Matrix, BracedTwoIntegersOnDynamicMatrixIsSized) {
    // Only vectors accept a flat list, so this is the (rows, cols) constructor.
    const MXX m{2, 3};
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
}

TEST(Matrix, LargeSizedUsesHeapAndStaysAligned) {
    MXX m(10, 10);
    EXPECT_EQ(m.size(), 100);
    EXPECT_TRUE(is_aligned(m.data()));
    m(9, 9) = 1.0;
    EXPECT_EQ(m.data()[99], 1.0);
}

TEST(Matrix, NestedInitializerListIsRowWise) {
    const MXX m{{1, 2, 3}, {4, 5, 6}};
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    expect_coeffs(m, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(linear(m), (std::vector<double>{1, 4, 2, 5, 3, 6})); // column-major
}

TEST(Matrix, NestedInitializerListFixedAndPartiallyFixed) {
    const M32 f{{1, 2}, {3, 4}, {5, 6}};
    expect_coeffs(f, {1, 2, 3, 4, 5, 6});

    const MX3 p{{1, 2, 3}};
    EXPECT_EQ(p.rows(), 1);
    expect_coeffs(p, {1, 2, 3});

    const MX3 empty = std::initializer_list<std::initializer_list<double>>{};
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 3);
}

TEST(Matrix, CopyAssignFromInitializerList) {
    MXX m(4, 4);
    m = {{1, 2}, {3, 4}};
    EXPECT_EQ(m.rows(), 2);
    expect_coeffs(m, {1, 2, 3, 4});
}

// ------------------------------------------------------------------------------------------------
// Vectors
// ------------------------------------------------------------------------------------------------

TEST(Vector, FlatInitializerList) {
    const VX v{1, 2, 3};
    EXPECT_EQ(v.rows(), 3);
    EXPECT_EQ(v.cols(), 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v(2), 3.0);
    EXPECT_EQ(v(1, 0), 2.0);

    const RX r{4, 5};
    EXPECT_EQ(r.rows(), 1);
    EXPECT_EQ(r.cols(), 2);
    EXPECT_EQ(r[1], 5.0);
    EXPECT_EQ(r(0, 1), 5.0);

    const V3 f{7, 8, 9};
    EXPECT_EQ(f[2], 9.0);
}

TEST(Vector, BracesVersusParentheses) {
    const VX one{4};
    EXPECT_EQ(one.size(), 1);
    EXPECT_EQ(one[0], 4.0);

    const VX sized(4);
    EXPECT_EQ(sized.size(), 4);
    EXPECT_EQ(sized[3], 0.0);

    const RX row(3);
    EXPECT_EQ(row.rows(), 1);
    EXPECT_EQ(row.cols(), 3);
}

TEST(Vector, ElementWrite) {
    VX v(3);
    v[0]    = 1.0;
    v(1)    = 2.0;
    v(2, 0) = 3.0;
    EXPECT_EQ(linear(v), (std::vector<double>{1, 2, 3}));
}

TEST(Vector, Resize) {
    VX v(2);
    v.resize(40);
    EXPECT_EQ(v.rows(), 40);
    EXPECT_EQ(v.cols(), 1);

    RX r;
    r.resize(5);
    EXPECT_EQ(r.rows(), 1);
    EXPECT_EQ(r.cols(), 5);
}

// ------------------------------------------------------------------------------------------------
// Element access, iteration, resize, copy, move, swap
// ------------------------------------------------------------------------------------------------

TEST(Matrix, ElementAccessIsColumnMajor) {
    MXX m(2, 3);
    m(1, 2) = 7.0;
    EXPECT_EQ(m.data()[(2 * 2) + 1], 7.0);
    EXPECT_EQ(std::distance(m.begin(), m.end()), 6);
}

TEST(Matrix, DataIsAligned) {
    const M22 f;
    const MXX d(2, 2);
    EXPECT_TRUE(is_aligned(f.data()));
    EXPECT_TRUE(is_aligned(d.data()));
}

TEST(Matrix, ResizeChangesDynamicDimensions) {
    MXX m(2, 2);
    m.resize(6, 7);
    EXPECT_EQ(m.rows(), 6);
    EXPECT_EQ(m.cols(), 7);

    MX3 p;
    p.resize(4, 3);
    EXPECT_EQ(p.rows(), 4);

    M22 f{{1, 2}, {3, 4}};
    f.resize(2, 2); // no-op for fixed sizes
    expect_coeffs(f, {1, 2, 3, 4});
}

TEST(Matrix, CopyIsDeep) {
    const MXX a{{1, 2}, {3, 4}};
    MXX       b = a;
    b(0, 0)     = 9.0;
    expect_coeffs(a, {1, 2, 3, 4});
    expect_coeffs(b, {9, 2, 3, 4});

    MXX c;
    c = a;
    expect_coeffs(c, {1, 2, 3, 4});
}

TEST(Matrix, MoveLeavesSourceEmpty) {
    MXX a(8, 8);
    a(7, 7)              = 1.0;
    const double* buffer = a.data();
    const MXX     b      = std::move(a);
    EXPECT_EQ(b.data(), buffer); // heap buffer is stolen
    EXPECT_EQ(b(7, 7), 1.0);
    EXPECT_EQ(a.size(), 0); // NOLINT(bugprone-use-after-move): moved-from state is specified
}

TEST(Matrix, Swap) {
    MXX a{{1, 2}};
    MXX b{{3}, {4}, {5}};
    swap(a, b);
    EXPECT_EQ(a.rows(), 3);
    EXPECT_EQ(b.rows(), 1);
    expect_coeffs(a, {3, 4, 5});
    expect_coeffs(b, {1, 2});

    M22 f{{1, 2}, {3, 4}};
    M22 g;
    f.swap(g);
    expect_coeffs(f, {0, 0, 0, 0});
    expect_coeffs(g, {1, 2, 3, 4});
}

// ------------------------------------------------------------------------------------------------
// Evaluation of expressions
// ------------------------------------------------------------------------------------------------

TEST(Matrix, EvaluationConstructor) {
    const MXX a{{1, 2, 3}, {4, 5, 6}};
    const MXX b{{6, 5, 4}, {3, 2, 1}};
    const MXX sum = AddOp(MatrixWrapper(a), MatrixWrapper(b));
    EXPECT_EQ(sum.rows(), 2);
    EXPECT_EQ(sum.cols(), 3);
    expect_coeffs(sum, {7, 7, 7, 7, 7, 7});
}

TEST(Matrix, EvaluationIntoFixedFromDynamicOperands) {
    const MXX a{{1, 2, 3}, {4, 5, 6}};
    const MXX b{{7, 8}, {9, 10}, {11, 12}};
    const M22 product = MulOp(MatrixWrapper(a), MatrixWrapper(b));
    expect_coeffs(product, {58, 64, 139, 154});
}

TEST(Matrix, EvaluationOfWrapperOfDifferentType) {
    const M23 fixed{{1, 2, 3}, {4, 5, 6}};
    const MXX copy = MatrixWrapper(fixed);
    EXPECT_EQ(copy.rows(), 2);
    expect_coeffs(copy, {1, 2, 3, 4, 5, 6});
}

TEST(Matrix, AssignmentEvaluatesAndResizes) {
    const MXX a{{1, 2, 3}, {4, 5, 6}};
    const MXX b{{1, 0}, {0, 1}, {1, 1}};
    MXX       c(5, 5);
    c = MulOp(MatrixWrapper(a), MatrixWrapper(b));
    EXPECT_EQ(c.rows(), 2);
    EXPECT_EQ(c.cols(), 2);
    expect_coeffs(c, {4, 5, 10, 11});
}

TEST(Matrix, AssignmentHandlesAliasing) {
    MXX       a{{1, 2}, {3, 4}};
    const MXX b{{5, 6}, {7, 8}};
    a = MulOp(MatrixWrapper(a), MatrixWrapper(b));
    expect_coeffs(a, {19, 22, 43, 50});

    // Shape change of an operand that is also the destination.
    MXX       r{{1, 2, 3}};
    const MXX col{{1}, {1}, {1}};
    r = MulOp(MatrixWrapper(col), MatrixWrapper(r));
    EXPECT_EQ(r.rows(), 3);
    EXPECT_EQ(r.cols(), 3);
    expect_coeffs(r, {1, 2, 3, 1, 2, 3, 1, 2, 3});
}

TEST(Matrix, VectorFromExpression) {
    const MXX m{{1, 2}, {3, 4}};
    const VX  x{1, 1};
    const VX  y = MulOp(MatrixWrapper(m), MatrixWrapper(x));
    EXPECT_EQ(linear(y), (std::vector<double>{3, 7}));
}

TEST(Matrix, ComplexScalars) {
    using C = std::complex<double>;
    const Matrix<C, Dynamic, Dynamic> a{{C(1, 1), C(0, 2)}};
    const Matrix<C, Dynamic, Dynamic> twice = ScalarMulOp(C(2, 0), MatrixWrapper(a));
    EXPECT_EQ(twice(0, 0), C(2, 2));
    EXPECT_EQ(twice(0, 1), C(0, 4));
}

TEST(Matrix, IntegerScalars) {
    const Matrix<int, 2, 2> a{{1, 2}, {3, 4}};
    const Matrix<int, 2, 2> b = SubOp(MatrixWrapper(a), MatrixWrapper(a));
    for (const int c : b) {
        EXPECT_EQ(c, 0);
    }
}
