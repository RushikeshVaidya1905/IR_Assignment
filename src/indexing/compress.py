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