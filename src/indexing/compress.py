import struct

class VariableByteEncoder:
    @staticmethod
    def encode_number(n: int) -> bytes:
        """Encodes a single integer using Variable Byte Encoding."""
        if n == 0:
            return struct.pack('B', 128)
            
        bytes_list = []
        while True:
            bytes_list.insert(0, n % 128)
            if n < 128:
                break
            n = n // 128
        # Set the continuation bit of the last byte to 1 (add 128)
        bytes_list[-1] += 128
        return struct.pack(f'{len(bytes_list)}B', *bytes_list)

    @staticmethod
    def decode(stream: bytes) -> list[int]:
        """Decodes a stream of bytes back into a list of integers."""
        numbers = []
        n = 0
        for byte in stream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers
        
    @staticmethod
    def gap_encode(postings: list[int]) -> list[int]:
        """Converts a sorted list of doc IDs into a list of gaps."""
        if not postings:
            return []
        encoded = [postings[0]]
        for i in range(1, len(postings)):
            encoded.append(postings[i] - postings[i-1])
        return encoded
        
    @staticmethod
    def gap_decode(gaps: list[int]) -> list[int]:
        """Reconstructs the original doc IDs from a list of gaps."""
        if not gaps:
            return []
        decoded = [gaps[0]]
        for i in range(1, len(gaps)):
            decoded.append(decoded[-1] + gaps[i])
        return decoded
    @staticmethod
    def encode_list(numbers: list[int]) -> bytes:
        """VB-encodes a list of integers into one byte string."""
        out = bytearray()
        for n in numbers:
            out.extend(VariableByteEncoder.encode_number(n))
        return bytes(out)


class GammaEncoder:
    """Elias gamma codes (IIR 5.3.2). Only defined for n >= 1, so callers encode gaps (which are >= 1)
    or add 1 to values that may be 0 (e.g. the first position)."""
    @staticmethod
    def encode_number(n: int) -> str:
        """Returns the gamma code of n as a bit string: unary(length) + offset."""
        offset = bin(n)[3:]  # binary of n without the leading 1
        return '1' * len(offset) + '0' + offset

    @staticmethod
    def encode_list(numbers: list[int]) -> bytes:
        bits = ''.join(GammaEncoder.encode_number(n) for n in numbers)
        bits += '0' * (-len(bits) % 8)  # pad to a whole byte
        return int(bits, 2).to_bytes(len(bits) // 8, 'big') if bits else b''

    @staticmethod
    def decode(stream: bytes, count: int) -> list[int]:
        bits = ''.join(f'{b:08b}' for b in stream)
        numbers, i = [], 0
        while len(numbers) < count:
            length = 0
            while bits[i] == '1':
                length += 1
                i += 1
            i += 1  # skip the 0 that ends the unary part
            numbers.append(int('1' + bits[i:i + length], 2))
            i += length
        return numbers
