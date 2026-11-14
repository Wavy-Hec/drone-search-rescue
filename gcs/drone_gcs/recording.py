"""Recording of the link: the frames a ground station received, saved as they came over the wire, and a reader
that plays them back. Standard library + numpy only.

File format::

    offset  size  field
    0       8     magic b"DGCREC1\\n"
    8       ...   records, one per received frame, in arrival order

    one record:
    0       8     receive time: the station's ``time.time()`` [s], float64, little endian
    8       4     connection id (1, 2, ... in the order the senders connected), uint32, little endian
    12      ...   the frame exactly as it was received: 16-byte prefix, header, payload (see protocol.py)

A path ending in ``.gz`` is gzip-compressed (level 1: fast, and raw camera frames still shrink a lot).
"""

from __future__ import annotations

import contextlib
import gzip
import logging
import os
import struct
import threading
from collections.abc import Iterator
from typing import IO

from .protocol import PREFIX_SIZE, Message, ProtocolError, decode_message, parse_prefix

__all__ = ["LinkRecorder", "read_recording", "MAGIC"]

log = logging.getLogger("drone_gcs.recording")

MAGIC = b"DGCREC1\n"
RECORD = struct.Struct("<dI")  # receive time, connection id
_HEAD_SIZE = RECORD.size + PREFIX_SIZE  # the fixed-size start of a record: it tells how long the rest is


def _open(path: str | os.PathLike, mode: str) -> IO[bytes]:
    """Open a recording in binary ``mode``; a name ending in ``.gz`` goes through gzip."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, compresslevel=1)
    return open(path, mode)


class LinkRecorder:
    """Writes received frames to a recording file. ``write`` may be called from several threads.

    A write error (disk full, ...) is logged and ends the recording; it is not raised, because the caller is
    the live link, which must go on."""

    def __init__(self, path: str | os.PathLike):
        self.path = path
        self._file: IO[bytes] | None = _open(path, "wb")
        self._file.write(MAGIC)
        self._lock = threading.Lock()
        self.frames = 0  # frames written
        self.nbytes = len(MAGIC)  # bytes written (before compression)

    def write(
        self,
        recv_wall: float,
        conn_id: int,
        prefix: bytes | bytearray,
        header: bytes | bytearray,
        payload: bytes | bytearray,
    ) -> None:
        """Append one frame (its three parts as received) that arrived at ``time.time()`` = ``recv_wall`` on
        connection ``conn_id``. Ignored after :meth:`close` or a write error."""
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(RECORD.pack(recv_wall, conn_id))
                self._file.write(prefix)
                self._file.write(header)
                self._file.write(payload)
            except OSError as e:
                log.error("recording to %s stopped after %d frames: %s", self.path, self.frames, e)
                with contextlib.suppress(OSError):  # the buffered rest cannot be written either
                    self._file.close()
                self._file = None
                return
            self.frames += 1
            self.nbytes += RECORD.size + len(prefix) + len(header) + len(payload)

    def close(self) -> None:
        """Finish the file (calling it again does nothing)."""
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.close()  # writes what is still buffered
            except OSError as e:
                log.error("recording %s is incomplete: %s", self.path, e)
            self._file = None


def read_recording(path: str | os.PathLike, decode_jpeg: bool = True) -> Iterator[Message]:
    """The messages of a recording, decoded, in file order.

    Each message has ``recv_time`` = seconds since the first record, ``conn_id`` and ``nbytes`` set, and its
    arrays are writable (the payload is decoded from a bytearray, as the Receiver does). A last record that is
    cut short (the station was killed while writing) ends the iteration with a warning. Raises
    :class:`~drone_gcs.protocol.ProtocolError` for a file that is not a recording.
    """
    with _open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ProtocolError(f"{path} is not a drone_gcs link recording (it does not start with {MAGIC!r})")
        first_wall = None
        n = 0
        try:
            while True:
                head = f.read(_HEAD_SIZE)
                if not head:
                    return  # the file ends after a complete record
                if len(head) < _HEAD_SIZE:
                    break
                recv_wall, conn_id = RECORD.unpack_from(head)
                hlen, plen = parse_prefix(head[RECORD.size :])
                header = f.read(hlen)
                payload = bytearray(f.read(plen))
                if len(header) < hlen or len(payload) < plen:
                    break
                msg = decode_message(header, payload, decode_jpeg=decode_jpeg)
                if first_wall is None:
                    first_wall = recv_wall
                msg.recv_time = recv_wall - first_wall
                msg.conn_id = conn_id
                msg.nbytes = PREFIX_SIZE + hlen + plen
                n += 1
                yield msg
        except EOFError:  # gzip: the compressed stream itself is cut short
            pass
        log.warning("%s: the last record is cut short (was the station killed?); %d complete records read", path, n)
