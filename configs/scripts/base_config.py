from abc import ABC, abstractmethod


def type_str_to_torch_dtype(type_str: str) -> str:
    """Convert a type string to a PyTorch dtype string."""
    type_map = {
        "float32": "torch.float32",
        "float64": "torch.float64",
        "complex64": "torch.complex64",
        "complex128": "torch.complex128",
    }
    return type_map.get(type_str, type_str)


def type_str_to_cuda_type(type_str: str) -> str:
    """Convert a type string to a CUDA type string."""
    type_map = {
        "float32": "float",
        "float64": "double",
        "complex64": "float2",
        "complex128": "double2",
    }
    return type_map.get(type_str, type_str)


class BaseFFT1dConfig(ABC):
    """Base abstract class for generating 1D FFT configurations.

    Attributes
    ----------
    function_name : str
        The name of the CUDA function which this configuration should generate template
        instantiations and C++ code calls for.
    signal_length : int
        The length of the input signal, which must match the FFT size for in-place FFTs.
        Out-of-place FFTs may have a different signal length.
    fft_size : int
        The size of the FFT, which must match the signal length for in-place FFTs. For
        FFT configs which are implicitly zero-padded this total size of the input signal
        plus the zero padding.
    input_data_type : str
        The data type of the input signal, which should be one of
        ["float16", "float32", "float64", "complex32", "complex64", "complex128"].
        NOTE: Not all data types are implemented in the underlying CUDA code yet.
    output_data_type : str
        The data type of the output signal, which should be one of
        ["float16", "float32", "float64", "complex32", "complex64", "complex128"].
        NOTE: Not all data types are implemented in the underlying CUDA code yet.
    is_forward_fft : bool
        A boolean indicating whether this configuration is for a forward FFT (True) or
        an inverse FFT (False).
    elements_per_thread : int
        The number of elements processed per thread. Tunable parameter for performance.
    ffts_per_block : int
        The number of FFTs processed per block. Tunable parameter for performance.

    Methods
    -------
    """

    function_name: str
    signal_length: int
    fft_size: int
    input_data_type: str
    output_data_type: str
    is_forward_fft: bool
    elements_per_thread: int
    ffts_per_block: int

    def __init__(
        self,
        function_name: str,
        signal_length: int,
        fft_size: int,
        input_data_type: str,
        output_data_type: str,
        is_forward_fft: bool,
        elements_per_thread: int,
        ffts_per_block: int,
    ):
        """Initialize the configuration for a 1D FFT."""
        self.function_name = function_name
        self.signal_length = signal_length
        self.fft_size = fft_size
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.is_forward_fft = is_forward_fft
        self.elements_per_thread = elements_per_thread
        self.ffts_per_block = ffts_per_block

    ##############################################################
    ### Abstract methods for generating static CUDA assertions ###
    ##############################################################
    @abstractmethod
    def get_fft_size_assert(self) -> str:
        """Generate a single element for an FFT size parameter assertion."""
        pass

    @abstractmethod
    def get_signal_length_assert(self) -> str:
        """Generate a single element for a signal length parameter assertion."""

    @abstractmethod
    def get_input_data_type_assert(self) -> str:
        """Generate a single element for an input data type parameter assertion."""
        pass

    @abstractmethod
    def get_output_data_type_assert(self) -> str:
        """Generate a single element for an output data type parameter assertion."""
        pass

    ###########################################################
    ### Abstract method for generating a templated instance ###
    ###########################################################
    @abstractmethod
    def get_template_instantiation(self) -> str:
        """Generate a string for the template instantiation of the FFT function."""
        pass
