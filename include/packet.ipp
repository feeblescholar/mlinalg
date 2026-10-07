#ifndef PACKET_IPP
#define PACKET_IPP

// Definitions for the portable Packet<T> primary template. Do not include this file directly.

namespace mlinalg {

template <PacketScalar T> Packet<T>::Packet(native_type value) noexcept : m_value(value) {}

template <PacketScalar T> auto Packet<T>::native() const noexcept -> native_type {
    return m_value;
}

template <PacketScalar T> auto Packet<T>::load(const T* ptr) noexcept -> Packet {
    return Packet(*ptr);
}

template <PacketScalar T> auto Packet<T>::loadu(const T* ptr) noexcept -> Packet {
    return Packet(*ptr);
}

template <PacketScalar T> auto Packet<T>::broadcast(T value) noexcept -> Packet {
    return Packet(value);
}

template <PacketScalar T> void Packet<T>::store(T* ptr) const noexcept {
    *ptr = m_value;
}

template <PacketScalar T> void Packet<T>::storeu(T* ptr) const noexcept {
    *ptr = m_value;
}

template <PacketScalar T> auto Packet<T>::reduce_add() const noexcept -> T {
    return m_value;
}

template <PacketScalar T> auto add(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T> {
    return Packet<T>(lhs.native() + rhs.native());
}

template <PacketScalar T> auto sub(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T> {
    return Packet<T>(lhs.native() - rhs.native());
}

template <PacketScalar T> auto mul(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T> {
    return Packet<T>(lhs.native() * rhs.native());
}

template <PacketScalar T> auto div(Packet<T> lhs, Packet<T> rhs) noexcept -> Packet<T> {
    return Packet<T>(lhs.native() / rhs.native());
}

template <PacketScalar T> auto fma(Packet<T> a, Packet<T> b, Packet<T> c) noexcept -> Packet<T> {
    // Deliberately not std::fma: without hardware FMA it is emulated in software and very slow.
    return Packet<T>(a.native() * b.native() + c.native());
}

} // namespace mlinalg

#endif // PACKET_IPP
