# Toolchain

Every tool is pinned to an exact version. Bump them deliberately, in a dedicated change, and update
every location listed below at once.

| Tool | Version | Pinned in |
| --- | --- | --- |
| GCC (Linux) | 16.2.0 | `linux.yml`, `lint.yml` (image digest) |
| Clang (Linux) | 23.1.4 | `linux.yml` (URL and SHA-256) |
| GCC (Windows, WinLibs UCRT) | 16.2.0 | `windows.yml` (URL and SHA-256) |
| Apple Clang (macOS) | 21.0.0, from Xcode 27.1 | `macos.yml`, `lint.yml` (`DEVELOPER_DIR`) |
| clang-format, clang-tidy | 23.1.4 | `.github/lint-requirements.txt`, `.pre-commit-config.yaml` |
| CMake | 4.4.5 | `lukka/get-cmake` in every workflow except the format job |
| Ninja | 1.13.2 | `lukka/get-cmake` in every workflow except the format job |
| GoogleTest | 1.18.0 | `lib/googletest` submodule |

Runner images are pinned by label: `ubuntu-26.04`, `xcode-27` and `windows-2025-vs2026`.

## Building

```sh
cmake --workflow --preset debug       # or release, asan-ubsan, tsan
```

## Local lint

```sh
pipx install clang-format==23.1.4
pipx install clang-tidy==23.1.4
cmake --preset debug
git ls-files -z 'include/*' 'test/*.cpp' 'test/*.hpp' | xargs -0 clang-format --dry-run -Werror
.github/scripts/clang-tidy.sh build/debug
```

## AVX2 on Apple Silicon

An x86-64 VM runs the same GCC image as CI. QEMU emulates AVX2 and FMA, so this tests correctness
only; it is far too slow for benchmarks.

```sh
brew install colima qemu lima-additional-guestagents docker
colima start --profile avx --arch x86_64 --cpu-type max --cpus 6 --memory 8 --disk 40
# seccomp=unconfined lets TSan disable ASLR for itself, as in CI. Only the tsan preset needs it.
docker --context colima-avx run --rm --security-opt seccomp=unconfined -v "$PWD":/src:ro gcc:16.2.0 bash -c '
    apt-get update -qq && apt-get install -y -qq cmake >/dev/null
    cp -r /src /work && cd /work && rm -rf build
    CXX=g++ cmake --workflow --preset release'

colima stop --profile avx
```
