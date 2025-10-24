from abc import ABC, abstractmethod


def _total_number(per_layer_counter) -> int:
    """Helper function to sum values from a dictionary-producing function"""
    total = 0
    per_layer_number = per_layer_counter()
    for layer_number in per_layer_number.values():
        total += layer_number
    return total


class Counter(ABC):
    def __init__(self):
        super().__init__()

    def total_bytes(self) -> int:
        return _total_number(self.per_layer_bytes)

    @abstractmethod
    def per_layer_bytes(self) -> dict:
        pass

    def total_flops(self, input_size) -> int:
        return _total_number(lambda: self.per_layer_flops(input_size))

    @abstractmethod
    def per_layer_flops(self, input_size) -> dict:
        pass

    def total_parameters(self) -> int:
        return _total_number(self.per_layer_parameters)

    @abstractmethod
    def per_layer_parameters(self) -> dict:
        pass
