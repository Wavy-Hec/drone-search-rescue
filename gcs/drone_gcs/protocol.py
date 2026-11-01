"""Wire format of the simulator -> GCS link. Standard library + numpy only.

One message is one length-prefixed frame on a stream socket (UNIX domain or TCP)::

    offset  size        field
    0       4           magic b"DGC1"
    4       4           header_len   uint32, little endian
    8       8           payload_len  uint64, little endian
    16      header_len  header: UTF-8 JSON object (see below)
    16+H    payload_len payload: the raw buffers listed in header["buffers"], concatenated, each one
                        starting at its own "offset" (8-byte aligned, zero padding in between)

Header keys: ``proto`` ("drone_gcs"), ``v`` (protocol version, 1), ``type`` (message type), ``seq`` (per-sender
counter), ``drone_id``, ``sender`` (name), ``wall_time`` (sender ``time.time()``), ``fields`` (JSON scalars,
lists and dicts) and ``buffers``: a list of ``{"name", "dtype", "shape", "offset", "nbytes", "encoding"}``.

``dtype`` is numpy's explicit array-protocol type string (``"<f4"``, ``"|u1"``, ``"|b1"``, ``"<i4"``...), so
the two ends may run different numpy major versions: nothing is pickled. ``encoding`` is ``"raw"`` (the
little-endian bytes of a C-contiguous array) or ``"jpeg"`` (``shape`` is then the decoded shape and ``nbytes``
the compressed size). The message types and their fields are documented in PROTOCOL.md.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_NAME = "drone_gcs"
PROTOCOL_VERSION = 1
MAGIC = b"DGC1"
PREFIX = struct.Struct("<4sIQ")
PREFIX_SIZE = PREFIX.size  # 16 bytes
ALIGN = 8
MAX_HEADER_BYTES = 8 << 20
MAX_PAYLOAD_BYTES = 1 << 30

# message types
HELLO = "HELLO"
MISSION = "MISSION"
TELEMETRY = "TELEMETRY"
SCAN = "SCAN"
CAMERA = "CAMERA"
GT_OBJECTS = "GT_OBJECTS"
END = "END"
MESSAGE_TYPES = (HELLO, MISSION, TELEMETRY, SCAN, CAMERA, GT_OBJECTS, END)

LATCHED_TYPES = frozenset({MISSION, GT_OBJECTS})
"""Re-sent automatically by the Sender on every (re)connect, so the GCS can start or restart at any time."""

CONTROL_TYPES = frozenset({HELLO, MISSION, GT_OBJECTS, END})
"""Never dropped by the bounded queues (data types are dropped oldest-first when a queue is full)."""

_ALLOWED_KINDS = frozenset("biuf")  # bool, signed int, unsigned int, float


class ProtocolError(ValueError):
    """A frame or message that does not follow the protocol."""


def jsonable(obj: Any) -> Any:
    """Convert ``obj`` to plain JSON types (numpy scalars/small arrays -> Python numbers/lists).

    Non-finite floats become ``None`` (strict JSON has no NaN/inf).
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # torch tensors and friends
        return jsonable(obj.tolist())
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def is_proxy_array(x: Any) -> bool:
    """``isaaclab.utils.warp.ProxyArray`` (or a look-alike with ``.torch`` / ``.warp`` properties).

    Checked on the *class*: a ProxyArray forwards unknown attributes to its torch view (a deprecation bridge
    that warns once per process), so ``hasattr(x, "detach")`` on the instance would be True and would use up
    Isaac Lab's one-time warning meant for the user's own code."""
    t = type(x)
    return hasattr(t, "torch") and hasattr(t, "warp")


def to_numpy(x: Any) -> np.ndarray:
    """Array view of numpy arrays, lists, torch tensors (any device), warp arrays or Isaac Lab ProxyArrays.

    Duck-typed: nothing here imports torch or warp.
    """
    if isinstance(x, np.ndarray):
        return x
    if is_proxy_array(x) or (hasattr(x, "torch") and not hasattr(x, "detach")):  # (or another torch-view wrapper)
        x = x.torch
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy") and callable(x.numpy):  # warp.array
        return np.asarray(x.numpy())
    return np.asarray(x)


def little_endian(a: np.ndarray) -> np.ndarray:
    """C-contiguous, little-endian version of ``a`` (bool / int / uint / float only)."""
    a = np.asarray(a)
    if a.dtype.kind not in _ALLOWED_KINDS:
        raise ProtocolError(f"unsupported dtype {a.dtype} (only bool, int, uint and float arrays are sent)")
    bo = a.dtype.byteorder
    if bo == ">" or (bo == "=" and sys.byteorder == "big"):
        a = a.astype(a.dtype.newbyteorder("<"))
    if not a.flags.c_contiguous:
        a = np.array(a, order="C", copy=True)  # (np.ascontiguousarray would turn 0-d arrays into 1-d)
    return a


# ---------------------------------------------------------------------------------------------------- JPEG
def jpeg_encode(rgb: np.ndarray, quality: int = 90) -> bytes | None:
    """JPEG bytes of an RGB (H, W, 3) or gray (H, W) uint8 image; ``None`` if neither cv2 nor PIL is importable."""
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim not in (2, 3):
        raise ProtocolError("JPEG encoding needs a uint8 (H, W) or (H, W, 3) image")
    try:
        import cv2  # noqa: PLC0415  (optional, lazy)

        img = rgb[..., ::-1] if rgb.ndim == 3 else rgb  # OpenCV encodes BGR
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass
    try:
        import io

        from PIL import Image  # noqa: PLC0415

        out = io.BytesIO()
        Image.fromarray(rgb).save(out, format="JPEG", quality=int(quality))
        return out.getvalue()
    except ImportError:
        return None


def jpeg_decode(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB (H, W, 3) or gray (H, W) uint8 array."""
    try:
        import cv2  # noqa: PLC0415

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ProtocolError("corrupt JPEG buffer")
        return np.ascontiguousarray(img[..., ::-1]) if img.ndim == 3 else img
    except ImportError:
        import io

        from PIL import Image  # noqa: PLC0415

        return np.asarray(Image.open(io.BytesIO(data)))


# ---------------------------------------------------------------------------------------------------- message
@dataclass
class Message:
    """A decoded message. ``fields`` holds the JSON part, ``arrays`` the decoded buffers."""

    type: str
    fields: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    seq: int = 0
    drone_id: int = 0
    sender: str = ""
    wall_time: float = 0.0
    recv_time: float = 0.0  # receiver time.monotonic()
    conn_id: int = 0
    nbytes: int = 0

    @property
    def t(self) -> float | None:
        """Simulation time [s] (``fields["t"]``) or ``None``."""
        t = self.fields.get("t")
        return None if t is None else float(t)

    def get(self, key: str, default: Any = None) -> Any:
        """A buffer if present, else a JSON field, else ``default``."""
        if key in self.arrays:
            return self.arrays[key]
        return self.fields.get(key, default)

    def array(self, key: str, dtype: Any = None, default: Any = None) -> np.ndarray | None:
        """``get(key)`` as a numpy array (``default`` when missing or null)."""
        v = self.get(key)
        if v is None:
            return default
        return np.asarray(v, dtype=dtype)


def encode_frame(
    msg_type: str,
    fields: dict | None = None,
    arrays: dict | None = None,
    *,
    seq: int = 0,
    drone_id: int = 0,
    sender: str = "",
    wall_time: float = 0.0,
    encodings: dict | None = None,
    jpeg_quality: int = 90,
) -> list:
    """Encode one message. Returns the frame as a list of bytes-like chunks (send them in order)."""
    encodings = encodings or {}
    metas: list[dict] = []
    chunks: list = []
    offset = 0
    for name, arr in (arrays or {}).items():
        if arr is None:
            continue
        a = np.asarray(arr)
        enc = encodings.get(name, "raw")
        data = None
        if enc == "jpeg":
            data = jpeg_encode(a, jpeg_quality)
            if data is None:  # no encoder available: fall back to raw
                enc = "raw"
        if enc == "raw":
            a = little_endian(a)
            data = memoryview(a.reshape(-1).view(np.uint8)) if a.size else b""
        elif enc != "jpeg":
            raise ProtocolError(f"unknown encoding {enc!r}")
        pad = (-offset) % ALIGN
        if pad:
            chunks.append(b"\0" * pad)
            offset += pad
        metas.append(
            {
                "name": str(name),
                "dtype": a.dtype.str,
                "shape": [int(s) for s in a.shape],
                "offset": offset,
                "nbytes": len(data),
                "encoding": enc,
            }
        )
        chunks.append(data)
        offset += len(data)
    header = {
        "proto": PROTOCOL_NAME,
        "v": PROTOCOL_VERSION,
        "type": str(msg_type),
        "seq": int(seq),
        "drone_id": int(drone_id),
        "sender": str(sender),
        "wall_time": float(wall_time),
        "fields": jsonable(fields or {}),
        "buffers": metas,
    }
    hb = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(hb) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header too large ({len(hb)} bytes): send big arrays as buffers, not fields")
    if offset > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload too large ({offset} bytes)")
    return [PREFIX.pack(MAGIC, len(hb), offset), hb, *chunks]


def encode_bytes(msg_type: str, fields: dict | None = None, arrays: dict | None = None, **kw) -> bytes:
    """``encode_frame`` joined into one bytes object (handy for tests and files)."""
    return b"".join(bytes(c) for c in encode_frame(msg_type, fields, arrays, **kw))


def parse_prefix(prefix: bytes) -> tuple[int, int]:
    """Validate the 16-byte prefix; returns ``(header_len, payload_len)``."""
    if len(prefix) != PREFIX_SIZE:
        raise ProtocolError("short prefix")
    magic, hlen, plen = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r} (not a drone_gcs stream?)")
    if hlen > MAX_HEADER_BYTES or plen > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"frame too large (header {hlen} B, payload {plen} B)")
    return hlen, plen


def decode_message(header_bytes: bytes, payload, *, decode_jpeg: bool = True) -> Message:
    """Decode a header (JSON bytes) and its payload (bytes / bytearray / memoryview) into a :class:`Message`.

    Raw buffers become zero-copy numpy views of ``payload`` (writable if ``payload`` is a bytearray).
    """
    try:
        header = json.loads(bytes(header_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad header: {e}") from e
    if not isinstance(header, dict) or header.get("proto") != PROTOCOL_NAME:
        raise ProtocolError("not a drone_gcs header")
    if int(header.get("v", -1)) != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {header.get('v')} != {PROTOCOL_VERSION}")
    mv = memoryview(payload)
    if mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    arrays: dict[str, np.ndarray] = {}
    for meta in header.get("buffers", []):
        try:
            name = str(meta["name"])
            off, n = int(meta["offset"]), int(meta["nbytes"])
            shape = tuple(int(s) for s in meta["shape"])
            dt = np.dtype(meta["dtype"])
            enc = meta.get("encoding", "raw")
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError(f"bad buffer description {meta!r}: {e}") from e
        if off < 0 or n < 0 or off + n > len(mv):
            raise ProtocolError(f"buffer {name!r} outside the payload")
        if dt.kind not in _ALLOWED_KINDS:
            raise ProtocolError(f"buffer {name!r}: unsupported dtype {dt}")
        if enc == "raw":
            count = math.prod(shape)
            if count * dt.itemsize != n:
                raise ProtocolError(f"buffer {name!r}: {n} bytes do not match shape {shape} x {dt}")
            arr = np.frombuffer(mv[off : off + n], dtype=dt).reshape(shape) if n else np.zeros(shape, dt)
            if not dt.isnative:
                arr = arr.astype(dt.newbyteorder("="))
        elif enc == "jpeg":
            if not decode_jpeg:
                arr = np.frombuffer(mv[off : off + n], dtype=np.uint8)
            else:
                arr = jpeg_decode(bytes(mv[off : off + n]))
        else:
            raise ProtocolError(f"buffer {name!r}: unknown encoding {enc!r}")
        arrays[name] = arr
    fields = header.get("fields") or {}
    if not isinstance(fields, dict):
        raise ProtocolError("fields must be a JSON object")
    return Message(
        type=str(header.get("type", "")),
        fields=fields,
        arrays=arrays,
        seq=int(header.get("seq", 0)),
        drone_id=int(header.get("drone_id", 0)),
        sender=str(header.get("sender", "")),
        wall_time=float(header.get("wall_time") or 0.0),
    )


def decode_bytes(frame: bytes, **kw) -> Message:
    """Decode one complete frame held in memory (inverse of :func:`encode_bytes`)."""
    hlen, plen = parse_prefix(frame[:PREFIX_SIZE])
    if len(frame) != PREFIX_SIZE + hlen + plen:
        raise ProtocolError("frame length does not match its prefix")
    msg = decode_message(frame[PREFIX_SIZE : PREFIX_SIZE + hlen], bytearray(frame[PREFIX_SIZE + hlen :]), **kw)
    msg.nbytes = len(frame)
    return msg
