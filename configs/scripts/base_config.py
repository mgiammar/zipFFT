from abc import ABC, abstractmethod
import json
import yaml
from typing import Any


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
    get_fft_size_assert() -> str
        Generate a single element for an FFT size parameter assertion.
    get_signal_length_assert() -> str
        Generate a single element for a signal length parameter assertion.
    get_input_data_type_assert() -> str
        Generate a single element for an input data type parameter assertion.
    get_output_data_type_assert() -> str
        Generate a single element for an output data type parameter assertion.
    get_function_signature() -> str
        Generate a string for the function signature of the FFT function.
    get_template_instantiation() -> str
        Generate a string for the template instantiation of the FFT function.
    get_binding_case_call() -> str
        Generate a string for the binding case call of the FFT function.

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

    @abstractmethod
    def get_fft_size_assert(self) -> str:
        """Generate a single element for an FFT size parameter assertion."""
        pass

    @abstractmethod
    def get_signal_length_assert(self) -> str:
        """Generate a single element for a signal length parameter assertion."""
        pass

    @abstractmethod
    def get_input_data_type_assert(self) -> str:
        """Generate a single element for an input data type parameter assertion."""
        pass

    @abstractmethod
    def get_output_data_type_assert(self) -> str:
        """Generate a single element for an output data type parameter assertion."""
        pass

    @abstractmethod
    def get_function_signature(self) -> str:
        """Generate a string for the function signature of the FFT function.

        This will likely be unique per FFT configuration, and the parameters/pointers
        of the signature will depend on the surrounding code.
        """
        pass

    def get_template_instantiation(self, *args, **kwargs) -> str:
        """Generate a string for the template instantiation of the FFT function."""
        return f"template int {self.get_function_signature(*args, **kwargs)};"

    def get_binding_case_call(self, *args, **kwargs) -> str:
        """Generate a string for the binding case call of the FFT function.

        Note
        ----
        This is a switch case against the FFT size parameter since all FFT configs have
        a FFT size, but not all have a signal length or input/output data type
        parameter. Nested switch cases are handled by the associated Generator class.
        """
        return (
            f"case {self.fft_size}:\n"
            f"{self.get_function_signature(*args, **kwargs)};\n"
            "break;"
        )


class BaseFFT1dGenerator(ABC):
    """
    Base abstract class for generating 1D FFT configurations.

    This class provides a framework for generating and managing 1D FFT (Fast Fourier Transform)
    configuration files and related code artifacts. It is intended to be subclassed to implement
    specific FFT configuration generators.

    Attributes
    ----------
    config_list (list[BaseFFT1dConfig]): List of FFT configuration objects to be processed.
    binding_cases_file (str): Path to the file where binding cases will be generated.
    implementation_file (str): Path to the file where template instantiations will be generated.
    static_assertions_file (str): Path to the file where static assertions will be generated.

    Methods
    -------
    __init__(config_list, binding_cases_file=None, implementation_file=None, static_assertions_file=None):
        Initialize the generator with a list of configurations and optional file paths.
        If file paths are not provided, default paths will be used, but the defaults
        need to be defined in the subclass.
    from_yaml(yaml_path, config_class=BaseFFT1dConfig, **kwargs):
        Class method to load a generator from a YAML configuration file.
    from_json(json_path, config_class=BaseFFT1dConfig, **kwargs):
        Class method to load a generator from a JSON configuration file.
    _get_forward_configs():
        Return a list of configurations for forward FFTs.
    _get_inverse_configs():
        Return a list of configurations for inverse FFTs.
    _group_configs_by_fft_size(configs):
        Group configurations by their FFT size.
    _get_warning_header():
        Generate a warning header for the generated files.
    _generate_fft_size_switch_block(configs):
        Generate a switch block for a set of configurations with the same FFT size.
    generate_template_instantiations():
        Generate the template instantiations for the FFT configurations.
    default_file_paths():
        Abstract method. Return the default paths for the generated files.
    generate_binding_cases():
        Abstract method. Generate the binding cases for the FFT configurations.
    generate_static_assertions():
        Abstract method. Generate static assertions for the FFT configurations.
    write_template_instantiations(file_path):
        Write the template instantiations to a file.
    write_binding_cases(file_path=None):
        Write the binding cases to a file.
    write_static_assertions(file_path=None):
        Write the static assertions to a file.
    """

    config_list: list[BaseFFT1dConfig] = None
    binding_cases_file: str = None
    implementation_file: str = None
    static_assertions_file: str = None

    def __init__(
        self,
        config_list: list[BaseFFT1dConfig] = None,
        binding_cases_file: str = None,
        implementation_file: str = None,
        static_assertions_file: str = None,
    ):
        """Initialize the generator with a list of configurations."""
        self.config_list = config_list
        self.binding_cases_file = binding_cases_file
        self.implementation_file = implementation_file
        self.static_assertions_file = static_assertions_file

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        config_class: type[BaseFFT1dConfig] = BaseFFT1dConfig,
        **kwargs: dict[Any, Any],
    ) -> "BaseFFT1dGenerator":
        """Load a generator from a YAML configuration file."""
        with open(yaml_path, "r") as file:
            config_data = yaml.safe_load(file)

        config_list = [config_class(**data) for data in config_data]
        return cls(config_list, **kwargs)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        config_class: type[BaseFFT1dConfig] = BaseFFT1dConfig,
        **kwargs: dict[Any, Any],
    ) -> "BaseFFT1dGenerator":
        """Load a generator from a JSON configuration file."""
        with open(json_path, "r") as file:
            config_data = json.load(file)

        config_list = [config_class(**data) for data in config_data]
        return cls(config_list, **kwargs)

    def _get_forward_configs(self) -> list[BaseFFT1dConfig]:
        """Return a list of configurations for forward FFTs."""
        return [config for config in self.config_list if config.is_forward_fft]

    def _get_inverse_configs(self) -> list[BaseFFT1dConfig]:
        """Return a list of configurations for inverse FFTs."""
        return [config for config in self.config_list if not config.is_forward_fft]

    def _group_configs_by_fft_size(
        self, configs: list[BaseFFT1dConfig]
    ) -> dict[int, list[BaseFFT1dConfig]]:
        """Group configurations by their FFT size."""
        grouped_configs = {}
        for config in configs:
            if config.fft_size not in grouped_configs:
                grouped_configs[config.fft_size] = []
            grouped_configs[config.fft_size].append(config)
        return grouped_configs

    def _get_warning_header(self) -> str:
        """Generate a warning header for the generated files."""
        return (
            "// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"// !!! This file was auto-generated by {self.__class__.__name__}. !!!\n"
            "// !!! Do not edit unless you know what you are doing.   !!!\n"
            "// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
        )

    def _generate_fft_size_switch_block(self, configs: list[BaseFFT1dConfig]) -> str:
        """Generate a switch block for a set of configurations with same FFTSize."""
        # Extra check to make sure all configurations have the same FFT size
        assert set([config.fft_size for config in configs]) == {configs[0].fft_size}, (
            "All configurations must have the same FFT size for this method. "
            f"Found FFT sizes of: {[config.fft_size for config in configs]}."
        )

        # Only generate the cases, not the actual switch statement
        cases = [config.get_binding_case_call() for config in configs]
        return "\n".join(cases)

    def generate_template_instantiations(self) -> str:
        """Generate the template instantiations for the FFT configurations."""
        tmp = "\n".join(
            config.get_template_instantiation() for config in self.config_list
        )
        return self._get_warning_header() + tmp + "\n"

    @abstractmethod
    def default_file_paths(self) -> tuple[str, str, str]:
        """Return the default paths for the generated files."""
        pass

    @abstractmethod
    def generate_binding_cases(self) -> str:
        """Generate the binding cases for the FFT configurations."""
        pass

    @abstractmethod
    def generate_static_assertions(self) -> str:
        """Generate static assertions for the FFT configurations."""
        pass

    def write_template_instantiations(self, file_path: str) -> None:
        """Write the template instantiations to a file."""
        if file_path is None:
            file_path = self.implementation_file
        with open(file_path, "w") as file:
            file.write(self.generate_template_instantiations())

    def write_binding_cases(self, file_path: str = None) -> None:
        """Write the binding cases to a file."""
        if file_path is None:
            file_path = self.binding_cases_file
        with open(file_path, "w") as file:
            file.write(self.generate_binding_cases())

    def write_static_assertions(self, file_path: str = None) -> None:
        """Write the static assertions to a file."""
        if file_path is None:
            file_path = self.static_assertions_file
        with open(file_path, "w") as file:
            file.write(self.generate_static_assertions())

    def write_all_files(self) -> None:
        """Write all files: template instantiations, binding cases, and static assertions."""
        self.write_template_instantiations(self.implementation_file)
        self.write_binding_cases(self.binding_cases_file)
        self.write_static_assertions(self.static_assertions_file)
