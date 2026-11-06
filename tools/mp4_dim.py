#!/usr/bin/env python3
"""纯标准库读取 mp4 的 width/height（从 moov/trak/tkhd 的 16.16 定点数），无需 ffmpeg/imageio。"""
import struct, sys

def boxes(buf, start=0, end=None):
    end = len(buf) if end is None else end
    i = start
    while i + 8 <= end:
        size = struct.unpack(">I", buf[i:i+4])[0]
        typ = buf[i+4:i+8]
        if size == 1:
            size = struct.unpack(">Q", buf[i+8:i+16])[0]
            hdr = 16
        else:
            hdr = 8
        if size < hdr:
            break
        yield typ, i + hdr, i + size
        i += size

def find_dim(path):
    buf = open(path, "rb").read()
    for typ, s, e in boxes(buf):
        if typ == b"moov":
            for t2, s2, e2 in boxes(buf, s, e):
                if t2 == b"trak":
                    for t3, s3, e3 in boxes(buf, s2, e2):
                        if t3 == b"tkhd":
                            # tkhd: version(1)+flags(3) ... 结尾 8 字节是 16.16 的 width,height
                            w, h = struct.unpack(">II", buf[e3-8:e3])
                            return round(w / 65536), round(h / 65536)
    return None

for p in sys.argv[1:]:
    print(p, find_dim(p))
