#!/usr/bin/env python3
# Converts PrismML's PQ2_0 GGUF files (Ternary Bonsai 2) to upstream llama.cpp's
# Q2_0, with no loss: the app's llama.cpp loads Q2_0 and applies the model's
# Hadamard rotation (prism.hadamard.* metadata) itself, but refuses PQ2_0, a type
# only PrismML's fork knows.
#
#   scripts/bonsai-pq2-to-q2.py Ternary-Bonsai-2-27B-PQ2_0.gguf
#   scripts/bonsai-pq2-to-q2.py in.gguf out.gguf
#
# Python 3.8 or newer and nothing else; about a minute for the 27B, and 6% more
# disk than the input. The output goes to <name with Q2_0>.gguf beside the input.
#
# Both types hold the same ternary weights with an fp16 scale per block, and
# use the same 2-bit codes (0, 1, 2 for -1, 0, +1, the lowest bits first). A
# PQ2_0 block is 128 weights (the scale, then 32 bytes of codes); a Q2_0 block
# is 64 (the scale, then 16 bytes). Each PQ2_0 block becomes two Q2_0 blocks
# with its scale and one half of its codes each, so every weight dequantizes to
# the same value. Code 3, which ternary weights never use, stops the conversion.
# Every other tensor and every key is copied as it is, except general.file_type.
# Checked against PrismML's own Q2_0 files: the 2B converts to the same bytes.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import hashlib
import os
import struct
import sys

PQ2_0, Q2_0 = 142, 42                 # ggml type ids: PrismML's fork, upstream
FTYPE_PQ2_0, FTYPE_Q2_0 = 141, 41     # general.file_type
GGML_TYPE_COUNT = 43                  # upstream knows the types below this
PQ_BLOCK, PQ_BYTES = 128, 34
Q_BLOCK, Q_BYTES = 64, 18
CHUNK = 1 << 20                       # blocks converted at a time

# Scalar sizes of GGUF metadata types (8 is a string, 9 an array).
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# For each byte of codes, 1 when one of its four codes is 3.
HAS_CODE_3 = bytes(1 if any((b >> s) & 3 == 3 for s in (0, 2, 4, 6)) else 0 for b in range(256))


def fail(text):
    sys.exit("bonsai-pq2-to-q2: " + text)


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            fail("the file ends inside its header")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64())


def skip_value(r, t, raw):
    """Reads one metadata value of type t, appending its bytes to raw."""
    if t == 8:
        n = r.take(8)
        raw += n + r.take(struct.unpack("<Q", n)[0])
    elif t == 9:
        head = r.take(12)
        raw += head
        et, n = struct.unpack("<IQ", head)
        if et == 8:
            for _ in range(n):
                skip_value(r, 8, raw)
        elif et in SCALAR:
            raw += r.take(n * SCALAR[et])
        else:
            fail("unknown array element type %d" % et)
    elif t in SCALAR:
        raw += r.take(SCALAR[t])
    else:
        fail("unknown metadata type %d" % t)


def to_q2_0(block_bytes):
    """PQ2_0 blocks to Q2_0 blocks: the scale twice, the codes split in halves."""
    n = len(block_bytes) // PQ_BYTES
    out = bytearray(n * 2 * Q_BYTES)
    for j in range(2):  # the fp16 scale
        out[j::2 * Q_BYTES] = block_bytes[j::PQ_BYTES]
        out[Q_BYTES + j::2 * Q_BYTES] = block_bytes[j::PQ_BYTES]
    for j in range(16):  # codes of weights 0..63, then 64..127
        out[2 + j::2 * Q_BYTES] = block_bytes[2 + j::PQ_BYTES]
        out[Q_BYTES + 2 + j::2 * Q_BYTES] = block_bytes[18 + j::PQ_BYTES]
    codes = bytearray(out)
    codes[0::Q_BYTES] = bytes(2 * n)
    codes[1::Q_BYTES] = bytes(2 * n)
    if 1 in codes.translate(HAS_CODE_3):
        fail("a block holds code 3, which is not a ternary weight; this is not a file this script knows")
    return out


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] in ("-h", "--help"):
        sys.exit("usage: bonsai-pq2-to-q2.py IN.gguf [OUT.gguf]")
    src = sys.argv[1]
    if len(sys.argv) == 3:
        dst = sys.argv[2]
    else:
        base = os.path.basename(src)
        name = base.replace("PQ2_0", "Q2_0") if "PQ2_0" in base else base[:-5] + "-Q2_0.gguf"
        dst = os.path.join(os.path.dirname(src), name)
    if os.path.abspath(dst) == os.path.abspath(src):
        fail("the output would replace the input")

    f = open(src, "rb")
    r = Reader(f)
    if r.take(4) != b"GGUF":
        fail("%s is not a GGUF file" % src)
    version = r.u32()
    if version != 3:
        fail("GGUF version %d; this script reads version 3" % version)
    n_tensors, n_kv = r.u64(), r.u64()

    kvs = []          # (raw key bytes, type, raw value bytes)
    alignment = 32
    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        raw = bytearray()
        skip_value(r, t, raw)
        if key == b"general.alignment" and t == 4:
            alignment = struct.unpack("<I", raw)[0]
        if key == b"general.file_type" and t == 4 and struct.unpack("<I", raw)[0] == FTYPE_PQ2_0:
            raw = bytearray(struct.pack("<I", FTYPE_Q2_0))
        kvs.append((key, t, bytes(raw)))

    tensors = []      # [name, dims, type, offset]
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        t, off = r.u32(), r.u64()
        tensors.append([name, dims, t, off])
    data_start = (f.tell() + alignment - 1) // alignment * alignment
    file_size = os.fstat(f.fileno()).st_size

    # Each tensor's bytes run to the next tensor's start (padding included),
    # so types this script does not size itself are copied whole.
    by_offset = sorted(tensors, key=lambda x: x[3])
    span = {}
    for i, x in enumerate(by_offset):
        end = by_offset[i + 1][3] if i + 1 < len(by_offset) else file_size - data_start
        span[x[0]] = end - x[3]

    converted = 0
    for name, dims, t, off in tensors:
        n = 1
        for d in dims:
            n *= d
        if t == PQ2_0:
            if dims[0] % PQ_BLOCK:
                fail("%s: rows of %d weights, not whole PQ2_0 blocks" % (name.decode(), dims[0]))
            if n // PQ_BLOCK * PQ_BYTES > span[name]:
                fail("%s: the file holds less data than its shape needs" % name.decode())
            converted += 1
        elif t >= GGML_TYPE_COUNT:
            fail("%s has type %d, which upstream llama.cpp does not know either (PrismML's PTQ1_0 is 143); "
                 "only PQ2_0 (142) converts" % (name.decode(), t))
    if not converted:
        fail("%s has no PQ2_0 tensors: nothing to convert" % src)

    # The new layout: tensors in the order listed, each at an aligned offset.
    sizes, out_off, at = {}, {}, 0
    for name, dims, t, off in tensors:
        n = 1
        for d in dims:
            n *= d
        sizes[name] = n // Q_BLOCK * Q_BYTES if t == PQ2_0 else span[name]
        out_off[name] = at
        at = (at + sizes[name] + alignment - 1) // alignment * alignment

    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, n_tensors, n_kv))
    for key, t, raw in kvs:
        header += struct.pack("<Q", len(key)) + key + struct.pack("<I", t) + raw
    for name, dims, t, off in tensors:
        header += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dims))
        header += struct.pack("<%dQ" % len(dims), *dims)
        header += struct.pack("<IQ", Q2_0 if t == PQ2_0 else t, out_off[name])
    header += bytes((-len(header)) % alignment)

    part = dst + ".part"
    sha = hashlib.sha256()
    with open(part, "wb") as out:
        def emit(b):
            out.write(b)
            sha.update(b)
        emit(header)
        for i, (name, dims, t, off) in enumerate(tensors):
            f.seek(data_start + off)
            if t == PQ2_0:
                left = sizes[name] // (2 * Q_BYTES) * PQ_BYTES
                while left:
                    take = min(left, CHUNK * PQ_BYTES)
                    emit(to_q2_0(f.read(take)))
                    left -= take
            else:
                left = sizes[name]
                while left:
                    b = f.read(min(left, 64 << 20))
                    if not b:
                        fail("%s: the file ends inside its data" % name.decode())
                    emit(b)
                    left -= len(b)
            pad = (-sizes[name]) % alignment
            if pad:
                emit(bytes(pad))
            if sys.stderr.isatty():
                sys.stderr.write("\rbonsai-pq2-to-q2: %d of %d tensors" % (i + 1, len(tensors)))
        out.flush()
        os.fsync(out.fileno())
    os.replace(part, dst)
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    print("bonsai-pq2-to-q2: %s: %d PQ2_0 tensors as Q2_0, %d others copied, %.2f GiB" %
          (dst, converted, len(tensors) - converted, os.path.getsize(dst) / 1073741824))
    print("sha256 %s" % sha.hexdigest())


if __name__ == "__main__":
    main()
