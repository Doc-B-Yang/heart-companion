"""Bounded JSON output through the existing REPL stream, without UART reinit.

UART0 is owned by MicroPython on CH340 boards. At most 32 UTF-8 bytes are
written per service call; one pending report bounds memory and sampling delay.
"""
import sys


class USBTransport:
    def __init__(self, buffer_size=4096, stream=None):
        self.buffer_size = buffer_size
        self.stream = sys.stdout if stream is None else stream
        self.chunked = True
        self.pending = ''
        self.offset = 0
        self.dropped = 0

    def send(self, payload):
        data = payload + '\n'
        if self.pending or len(data.encode('utf-8')) >= self.buffer_size:
            self.dropped += 1
            return False
        self.pending, self.offset = data, 0
        return True

    def service(self):
        if not self.pending:
            return
        block = self.pending[self.offset:self.offset + 32]
        while len(block.encode('utf-8')) > 32:
            block = block[:-1]
        try:
            count = self.stream.write(block)
            if not isinstance(count, int) or not 0 <= count <= len(block):
                raise OSError('Invalid stream write count')
            self.offset += count
            if self.offset == len(self.pending):
                self.pending = ''
        except OSError:
            self.dropped += 1
            # Terminate any truncated JSON line before the next report.
            self.pending, self.offset = '\n', 0
