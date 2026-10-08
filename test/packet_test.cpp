#include "../include/packet.hpp"
#include "test.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using mlinalg::Packet;

namespace {

template <typename T> struct real_of {
    using type = T;
};

template <typename T> struct real_of<std::complex<T>> {
    using type = T;
};

template <typename T> using real_of_t = typename real_of<T>::type;

/// Deterministic, non-zero test values; `salt` gives different operands distinct values.
template <typename T> auto make_value(int i, int salt) -> T {
    if constexpr (std::is_same_v<T, int>) {
        // Mixed signs and non-multiples, so division exercises truncation toward zero.
        const int x = (i * 7) + (salt * 13) + 1;
        return (i % 2 == 0) ? x : -x;
    }
    using R      = real_of_t<T>;
    const auto x = static_cast<R>(i + 1) * static_cast<R>(0.75) + static_cast<R>(salt);
    if constexpr (mlinalg::detail::is_complex_v<T>) {
        return T(x, static_cast<R>(salt - i) * static_cast<R>(0.5) - static_cast<R>(1.25));
    } else {
        return (i % 2 == 0) ? x : -x;
    }
}

template <typename T> auto tolerance() -> real_of_t<T> {
    if constexpr (std::is_integral_v<T>) {
        return 0; // integer results must be exact
    } else if constexpr (std::is_same_v<real_of_t<T>, float>) {
        return 1e-5F;
    } else {
        return static_cast<real_of_t<T>>(1e-12);
    }
}

template <typename T> void expect_close(const T& actual, const T& expected) {
    using std::abs;
    const auto scale = std::max(abs(expected), real_of_t<T>(1));
    EXPECT_LE(abs(actual - expected), tolerance<T>() * scale)
        << "actual = " << actual << ", expected = " << expected;
}

constexpr std::size_t kBufferAlignment = 32;

template <typename T> auto is_aligned(const T* ptr) -> bool {
    return reinterpret_cast<std::uintptr_t>(ptr) % kBufferAlignment == 0;
}

/// Aligned buffer holding one packet's worth of values (plus one spare for unaligned tests).
template <typename T> struct Buffer {
    static constexpr int N = Packet<T>::size;
    alignas(kBufferAlignment) std::array<T, N + 1> data{};

    static auto filled(int salt) -> Buffer {
        Buffer buffer;
        for (int i = 0; i < N + 1; ++i) {
            buffer.data[static_cast<std::size_t>(i)] = make_value<T>(i, salt);
        }
        return buffer;
    }

    auto operator[](int i) -> T& { return data[static_cast<std::size_t>(i)]; }
};

/// Applies a binary packet op and checks every lane against the scalar reference op.
template <typename T, typename PacketOp, typename ScalarOp>
void check_binary(PacketOp packet_op, ScalarOp scalar_op) {
    auto      lhs = Buffer<T>::filled(1);
    auto      rhs = Buffer<T>::filled(3);
    Buffer<T> out;

    packet_op(Packet<T>::load(lhs.data.data()), Packet<T>::load(rhs.data.data()))
        .store(out.data.data());

    for (int i = 0; i < Packet<T>::size; ++i) {
        SCOPED_TRACE(testing::Message() << "lane " << i);
        expect_close(out[i], scalar_op(lhs[i], rhs[i]));
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Backend selection
// ------------------------------------------------------------------------------------------------

#if defined(MLINALG_SIMD_AVX2)
static_assert(Packet<float>::size == 8);
static_assert(Packet<double>::size == 4);
static_assert(Packet<int>::size == 8);
static_assert(Packet<std::complex<float>>::size == 4);
static_assert(Packet<std::complex<double>>::size == 2);
#elif defined(MLINALG_SIMD_NEON)
static_assert(Packet<float>::size == 4);
static_assert(Packet<double>::size == 2);
static_assert(Packet<int>::size == 4);
static_assert(Packet<std::complex<float>>::size == 2);
static_assert(Packet<std::complex<double>>::size == 1);
#else
static_assert(Packet<float>::size == 1);
static_assert(Packet<double>::size == 1);
static_assert(Packet<int>::size == 1);
#endif

// Types without a SIMD specialization always use the portable one-lane packet.
static_assert(Packet<long double>::size == 1);
static_assert(Packet<std::complex<long double>>::size == 1);

static_assert(mlinalg::PacketScalar<float> && mlinalg::PacketScalar<std::complex<double>>);
static_assert(mlinalg::PacketScalar<int>);
static_assert(!mlinalg::PacketScalar<long> && !mlinalg::PacketScalar<unsigned> &&
              !mlinalg::PacketScalar<std::complex<int>>);

// One packet must occupy exactly one register's worth of scalars.
static_assert(sizeof(Packet<float>) == sizeof(float) * Packet<float>::size);
static_assert(sizeof(Packet<int>) == sizeof(int) * Packet<int>::size);
static_assert(sizeof(Packet<std::complex<double>>) ==
              sizeof(std::complex<double>) * Packet<std::complex<double>>::size);

// ------------------------------------------------------------------------------------------------
// Typed tests over every scalar type
// ------------------------------------------------------------------------------------------------

template <typename T> class PacketTest : public testing::Test {};

using PacketTypes = testing::Types<int, float, double, std::complex<float>, std::complex<double>,
                                   long double, std::complex<long double>>;
TYPED_TEST_SUITE(PacketTest, PacketTypes);

TYPED_TEST(PacketTest, AlignedLoadStoreRoundTrip) {
    using T       = TypeParam;
    auto      src = Buffer<T>::filled(2);
    Buffer<T> dst;
    ASSERT_TRUE(is_aligned(src.data.data()));

    Packet<T>::load(src.data.data()).store(dst.data.data());
    for (int i = 0; i < Packet<T>::size; ++i) {
        EXPECT_EQ(dst[i], src[i]) << "lane " << i;
    }
}

TYPED_TEST(PacketTest, UnalignedLoadStoreRoundTrip) {
    using T       = TypeParam;
    auto      src = Buffer<T>::filled(2);
    Buffer<T> dst;

    // Offsetting by one element breaks register alignment whenever size > 1.
    Packet<T>::loadu(src.data.data() + 1).storeu(dst.data.data() + 1);
    for (int i = 1; i <= Packet<T>::size; ++i) {
        EXPECT_EQ(dst[i], src[i]) << "lane " << i;
    }
    EXPECT_EQ(dst[0], T{}) << "storeu wrote outside the packet";
}

TYPED_TEST(PacketTest, Broadcast) {
    using T         = TypeParam;
    const T   value = make_value<T>(4, 7);
    Buffer<T> dst;

    Packet<T>::broadcast(value).store(dst.data.data());
    for (int i = 0; i < Packet<T>::size; ++i) {
        EXPECT_EQ(dst[i], value) << "lane " << i;
    }
}

TYPED_TEST(PacketTest, Add) {
    using T = TypeParam;
    check_binary<T>([](auto a, auto b) { return add(a, b); }, [](T a, T b) { return a + b; });
}

TYPED_TEST(PacketTest, Sub) {
    using T = TypeParam;
    check_binary<T>([](auto a, auto b) { return sub(a, b); }, [](T a, T b) { return a - b; });
}

TYPED_TEST(PacketTest, Mul) {
    using T = TypeParam;
    check_binary<T>([](auto a, auto b) { return mul(a, b); }, [](T a, T b) { return a * b; });
}

TYPED_TEST(PacketTest, Div) {
    using T = TypeParam;
    check_binary<T>([](auto a, auto b) { return div(a, b); }, [](T a, T b) { return a / b; });
}

TYPED_TEST(PacketTest, Fma) {
    using T     = TypeParam;
    auto      a = Buffer<T>::filled(1);
    auto      b = Buffer<T>::filled(3);
    auto      c = Buffer<T>::filled(5);
    Buffer<T> out;

    fma(Packet<T>::load(a.data.data()), Packet<T>::load(b.data.data()),
        Packet<T>::load(c.data.data()))
        .store(out.data.data());

    for (int i = 0; i < Packet<T>::size; ++i) {
        SCOPED_TRACE(testing::Message() << "lane " << i);
        expect_close(out[i], a[i] * b[i] + c[i]);
    }
}

TYPED_TEST(PacketTest, ReduceAdd) {
    using T  = TypeParam;
    auto src = Buffer<T>::filled(2);

    T expected{};
    for (int i = 0; i < Packet<T>::size; ++i) {
        expected += src[i];
    }
    expect_close(Packet<T>::load(src.data.data()).reduce_add(), expected);
}

TYPED_TEST(PacketTest, ChainedOperations) {
    // (a + b) * c - a / b, evaluated purely through the Packet API.
    using T     = TypeParam;
    auto      a = Buffer<T>::filled(1);
    auto      b = Buffer<T>::filled(3);
    auto      c = Buffer<T>::filled(5);
    Buffer<T> out;

    const auto pa = Packet<T>::load(a.data.data());
    const auto pb = Packet<T>::load(b.data.data());
    const auto pc = Packet<T>::load(c.data.data());
    sub(mul(add(pa, pb), pc), div(pa, pb)).store(out.data.data());

    for (int i = 0; i < Packet<T>::size; ++i) {
        SCOPED_TRACE(testing::Message() << "lane " << i);
        expect_close(out[i], (a[i] + b[i]) * c[i] - a[i] / b[i]);
    }
}

// ------------------------------------------------------------------------------------------------
// Edge cases specific to the hand-written complex kernels
// ------------------------------------------------------------------------------------------------

TEST(PacketComplex, MulAndDivByPureImaginaryUnit) {
    using C = std::complex<double>;
    Buffer<C> src;
    Buffer<C> out;
    for (int i = 0; i < Packet<C>::size; ++i) {
        src[i] = C(1.0 + i, -2.0 * i);
    }
    const auto unit = Packet<C>::broadcast(C(0.0, 1.0));
    const auto p    = Packet<C>::load(src.data.data());

    mul(p, unit).store(out.data.data());
    for (int i = 0; i < Packet<C>::size; ++i) {
        EXPECT_EQ(out[i], src[i] * C(0.0, 1.0));
    }

    div(p, unit).store(out.data.data());
    for (int i = 0; i < Packet<C>::size; ++i) {
        EXPECT_EQ(out[i], src[i] / C(0.0, 1.0));
    }
}

TEST(PacketComplex, ReduceAddKeepsRealAndImaginaryApart) {
    using C = std::complex<float>;
    Buffer<C> src;
    for (int i = 0; i < Packet<C>::size; ++i) {
        src[i] = C(1.0F, 100.0F);
    }
    const auto sum = Packet<C>::load(src.data.data()).reduce_add();
    EXPECT_EQ(sum.real(), static_cast<float>(Packet<C>::size));
    EXPECT_EQ(sum.imag(), 100.0F * static_cast<float>(Packet<C>::size));
}

// ------------------------------------------------------------------------------------------------
// Integer division (emulated through double, since AVX2 and NEON lack SIMD integer division)
// ------------------------------------------------------------------------------------------------

TEST(PacketInt, DivTruncatesTowardZeroAcrossFullRange) {
    constexpr int kMax = std::numeric_limits<int>::max();
    constexpr int kMin = std::numeric_limits<int>::min();

    // {numerator, denominator}: sign combinations, exact multiples, |divisor| == 1, and extreme
    // magnitudes where an inexact double quotient would round to the wrong integer.
    const std::vector<std::pair<int, int>> cases = {
        {7, 2},           {-7, 2},          {7, -2},          {-7, -2},      {6, 3},
        {-6, 3},          {0, 5},           {1, 7},           {kMax, 1},     {kMax, -1},
        {kMin, 1},        {kMax, 2},        {kMin, 2},        {kMin, kMax},  {kMax, kMin},
        {kMax, kMax - 1}, {kMax - 1, kMax}, {kMin + 1, -1},   {kMax, 46341}, {kMin, -46341},
        {kMax, 3},        {-kMax, 7},       {123456789, -10}, {-1, kMax},    {kMin, kMin},
    };

    constexpr int N = Packet<int>::size;
    for (std::size_t start = 0; start < cases.size(); start += N) {
        alignas(32) std::array<int, N> num{};
        alignas(32) std::array<int, N> den{};
        alignas(32) std::array<int, N> out{};
        for (int lane = 0; lane < N; ++lane) {
            const auto& c = cases[(start + static_cast<std::size_t>(lane)) % cases.size()];
            num[static_cast<std::size_t>(lane)] = c.first;
            den[static_cast<std::size_t>(lane)] = c.second;
        }

        div(Packet<int>::load(num.data()), Packet<int>::load(den.data())).store(out.data());

        for (std::size_t lane = 0; lane < N; ++lane) {
            EXPECT_EQ(out[lane], num[lane] / den[lane]) << num[lane] << " / " << den[lane];
        }
    }
}

TEST(PacketInt, FmaAndReduceAreExact) {
    constexpr int                  N = Packet<int>::size;
    alignas(32) std::array<int, N> a{};
    alignas(32) std::array<int, N> b{};
    alignas(32) std::array<int, N> out{};
    for (int i = 0; i < N; ++i) {
        a[static_cast<std::size_t>(i)] = 1000 * (i + 1);
        b[static_cast<std::size_t>(i)] = -(i + 3);
    }
    const auto pa = Packet<int>::load(a.data());
    const auto pb = Packet<int>::load(b.data());

    fma(pa, pb, Packet<int>::broadcast(17)).store(out.data());
    int expected_sum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(out[i], a[i] * b[i] + 17);
        expected_sum += a[i];
    }
    EXPECT_EQ(pa.reduce_add(), expected_sum);
}
