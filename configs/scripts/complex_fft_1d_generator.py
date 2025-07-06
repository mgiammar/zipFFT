import yaml
import json
from pathlib import Path

from base_config import (
    BaseFFT1dConfig,
    type_str_to_torch_dtype,
    type_str_to_cuda_type,
)

# Constants for static assert messages
STATIC_ASSERT_FFT_SIZE_MESSAGE = "Unsupported FFT size"
STATIC_ASSERT_TYPE_MESSAGE = "Unsupported input data type"


class ComplexToComplexFFT1DConfig(BaseFFT1dConfig):
    """Container class for generating C2C 1D FFT configurations.

    This class handles complex-to-complex FFT configurations where the
    following are not used:
    - signal_length: Unused because no zero-padding is applied. Input and output are
                     the same lengths.
    - output_data_type: Unused because the input and output data types are the same.

    Attributes
    ----------
    function_name: Name of the FFT function to generate. Here, it is
        "block_complex_fft_1d".
    fft_size: Size of the FFT
    input_data_type: Input data type as string
    is_forward_fft: True for forward FFT, False for inverse
    elements_per_thread: Number of elements processed per thread
    ffts_per_block: Number of FFTs per block
    output_data_type: Unused, kept for compatibility
    signal_length: Unused, kept for compatibility
    """

    def __init__(
        self,
        fft_size: int,
        input_data_type: str,
        is_forward_fft: bool,
        elements_per_thread: int,
        ffts_per_block: int,
        output_data_type: str = None,
        signal_length: int = None,
    ):
        """Initialize the configuration for a complex-to-complex 1D FFT.

        Parameters
        ----------
        fft_size : int
            Size of the FFT.
        input_data_type : str
            Input data type as string.
        is_forward_fft : bool
            True for forward FFT, False for inverse.
        elements_per_thread : int
            Number of elements processed per thread.
        ffts_per_block : int
            Number of FFTs per block.
        output_data_type : str, optional
            Unused, kept for compatibility.
        signal_length : int, optional
            Unused, kept for compatibility.
        """
        super().__init__(
            function_name="block_complex_fft_1d",
            signal_length=signal_length,
            fft_size=fft_size,
            input_data_type=input_data_type,
            output_data_type=output_data_type,
            is_forward_fft=is_forward_fft,
            elements_per_thread=elements_per_thread,
            ffts_per_block=ffts_per_block,
        )

    def get_fft_size_assert(self) -> str:
        """Generate FFT size assertion condition.

        Returns
        -------
        str
            Assertion string for FFT size.
        """
        return f"FFTSize == {self.fft_size}"

    def get_input_data_type_assert(self) -> str:
        """Generate input data type assertion condition.

        Returns
        -------
        str
            Assertion string for input data type.
        """
        cuda_type = type_str_to_cuda_type(self.input_data_type)
        return f"std::is_same_v<T, {cuda_type}>"

    def get_template_instantiation(self) -> str:
        """Generate template instantiation string for the FFT function.

        Returns
        -------
        str
            Template instantiation string.
        """
        cuda_type = type_str_to_cuda_type(self.input_data_type)
        fwd_inv_param = "true" if self.is_forward_fft else "false"

        return (
            f"template int {self.function_name}<{cuda_type}, {self.fft_size}u, "
            f"{fwd_inv_param}, {self.elements_per_thread}, {self.ffts_per_block}>"
            f"({cuda_type}* data);"
        )

    def get_signal_length_assert(self) -> str:
        """Generate signal length assertion (unused for C2C FFT).

        Returns
        -------
        str
            Empty string.
        """
        return ""

    def get_output_data_type_assert(self) -> str:
        """Generate output data type assertion (unused for C2C FFT).

        Returns
        -------
        str
            Empty string.
        """
        return ""


class ComplexToComplexFFT1DGenerator:
    """Generator for complex-to-complex 1D FFT configurations.

    This class manages a collection of ComplexToComplexFFT1DConfig instances
    and generates various code artifacts including CUDA static asserts,
    template instantiations, and Python binding switch statements.

    Attributes
    ----------
    config_list : list of ComplexToComplexFFT1DConfig
        List of ComplexToComplexFFT1DConfig instances.
    fwd_implementations_file : str
        File name (path) to output the template instantiations for the forward FFT implementations.
    inv_implementations_file : str
        File name (path) to output the template instantiations for the inverse FFT implementations.
    fwd_static_assertions_file : str
        File name (path) to output the static asserts for the forward FFT implementations.
    inv_static_assertions_file : str
        File name (path) to output the static asserts for the inverse FFT implementations.
    fwd_binding_cases_file : str
        File name (path) to output the Python binding switch cases for the forward FFT implementations.
    inv_binding_cases_file : str
        File name (path) to output the Python binding switch cases for the inverse FFT implementations.
    """

    config_list: list[ComplexToComplexFFT1DConfig]

    # Use pathlib to get paths relative to this file
    _script_dir = Path(__file__).parent
    _gen_dir = _script_dir / ".." / ".." / "src" / "generated"

    fwd_implementations_file: str = _gen_dir / "fwd_fft_c2c_1d_implementations.inc"
    inv_implementations_file: str = _gen_dir / "inv_fft_c2c_1d_implementations.inc"
    fwd_static_assertions_file: str = _gen_dir / "fwd_fft_c2c_1d_assertions.inc"
    inv_static_assertions_file: str = _gen_dir / "inv_fft_c2c_1d_assertions.inc"
    fwd_binding_cases_file: str = _gen_dir / "fwd_fft_c2c_1d_binding_cases.inc"
    inv_binding_cases_file: str = _gen_dir / "inv_fft_c2c_1d_binding_cases.inc"

    def __init__(
        self,
        config_list: list[ComplexToComplexFFT1DConfig],
        fwd_implementations_file: str = None,
        inv_implementations_file: str = None,
        fwd_static_assertions_file: str = None,
        inv_static_assertions_file: str = None,
        fwd_binding_cases_file: str = None,
        inv_binding_cases_file: str = None,
    ):
        """Initialize the generator with a list of configurations.

        Parameters
        ----------
        config_list : list of ComplexToComplexFFT1DConfig
            List of ComplexToComplexFFT1DConfig instances.
        fwd_implementations_file : str, optional
            Path to output file for forward FFT implementations. If None, uses default path.
        inv_implementations_file : str, optional
            Path to output file for inverse FFT implementations. If None, uses default path.
        fwd_static_assertions_file : str, optional
            Path to output file for forward FFT static assertions. If None, uses default path.
        inv_static_assertions_file : str, optional
            Path to output file for inverse FFT static assertions. If None, uses default path.
        fwd_binding_cases_file : str, optional
            Path to output file for forward FFT Python binding cases. If None, uses default path.
        inv_binding_cases_file : str, optional
            Path to output file for inverse FFT Python binding cases. If None, uses default path.
        """
        self.config_list = config_list

        if fwd_implementations_file:
            self.fwd_implementations_file = fwd_implementations_file
        if inv_implementations_file:
            self.inv_implementations_file = inv_implementations_file
        if fwd_static_assertions_file:
            self.fwd_static_assertions_file = fwd_static_assertions_file
        if inv_static_assertions_file:
            self.inv_static_assertions_file = inv_static_assertions_file
        if fwd_binding_cases_file:
            self.fwd_binding_cases_file = fwd_binding_cases_file
        if inv_binding_cases_file:
            self.inv_binding_cases_file = inv_binding_cases_file

    def _filter_configs_by_direction(
        self, is_forward: bool
    ) -> list[ComplexToComplexFFT1DConfig]:
        """Filter configurations by FFT direction.

        Parameters
        ----------
        is_forward : bool
            True for forward FFT configs, False for inverse FFT.

        Returns
        -------
        list of ComplexToComplexFFT1DConfig
            Filtered list of configurations.
        """
        return [
            config for config in self.config_list if config.is_forward_fft == is_forward
        ]

    @classmethod
    def from_yaml(cls, config_yaml_path: str) -> "ComplexToComplexFFT1DGenerator":
        """Parse YAML configuration into ComplexToComplexFFT1DGenerator.

        Parameters
        ----------
        config_yaml_path : str
            Path to YAML file containing configuration.

        Returns
        -------
        ComplexToComplexFFT1DGenerator
            Instance of ComplexToComplexFFT1DGenerator.
        """
        with open(config_yaml_path, "r") as f:
            config_dicts = yaml.safe_load(f)

        return cls.parse_dicts(config_dicts)

    @classmethod
    def from_json(cls, config_json_path: str) -> "ComplexToComplexFFT1DGenerator":
        """Parse JSON configuration into ComplexToComplexFFT1DGenerator.

        Parameters
        ----------
        config_json_path : str
            Path to JSON file containing configuration.

        Returns
        -------
        ComplexToComplexFFT1DGenerator
            Instance of ComplexToComplexFFT1DGenerator.
        """
        with open(config_json_path, "r") as f:
            config_dicts = json.load(f)

        return cls.parse_dicts(config_dicts)

    @classmethod
    def parse_dicts(cls, config_dicts: list[dict]) -> "ComplexToComplexFFT1DGenerator":
        """Parse list of dictionaries into ComplexToComplexFFT1DGenerator.

        Parameters
        ----------
        config_dicts : list of dict
            List of configuration dictionaries.

        Returns
        -------
        ComplexToComplexFFT1DGenerator
            Instance of ComplexToComplexFFT1DGenerator.

        Note
        ----
        This function does not have robust error handling or type checking.
        """
        configs = [
            ComplexToComplexFFT1DConfig(**config_dict) for config_dict in config_dicts
        ]
        return cls(configs)

    def _get_warning_header(self) -> str:
        """Generate warning header for auto-generated files.

        Returns
        -------
        str
            Warning header string.
        """
        return (
            "// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "// !!! This file was auto-generated by the class   !!!\n"
            "// !!! ComplexToComplexFFT1DGenerator. Do not edit !!!\n"
            "// !!! unless you know what you are doing.         !!!\n"
            "// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
        )

    def _collect_assert_conditions(
        self, configs: list[ComplexToComplexFFT1DConfig]
    ) -> tuple[list[str], list[str]]:
        """Collect assertion conditions from given configurations.

        Parameters
        ----------
        configs : list of ComplexToComplexFFT1DConfig
            List of configurations to process.

        Returns
        -------
        tuple of list of str
            Tuple of (fft_size_conditions, input_type_conditions).
        """
        fft_size_conditions = []
        input_type_conditions = []

        for config in configs:
            fft_size_conditions.append(config.get_fft_size_assert())
            input_type_conditions.append(config.get_input_data_type_assert())

        # Filter out duplicate conditions
        fft_size_conditions = list(set(fft_size_conditions))
        input_type_conditions = list(set(input_type_conditions))

        return fft_size_conditions, input_type_conditions

    def generate_cuda_static_asserts(self, is_forward: bool = True) -> str:
        """Generate CUDA static assert statements for configurations.

        Parameters
        ----------
        is_forward : bool, optional
            True for forward FFT, False for inverse FFT.

        Returns
        -------
        str
            String containing static assert statements.
        """
        configs = self._filter_configs_by_direction(is_forward)
        fft_size_conditions, input_type_conditions = self._collect_assert_conditions(
            configs
        )

        # Create combined assertion conditions
        fft_size_assert = " || ".join(fft_size_conditions)
        input_type_assert = " || ".join(input_type_conditions)

        # Format complete assert statements
        fft_size_statement = (
            f'static_assert({fft_size_assert}, "{STATIC_ASSERT_FFT_SIZE_MESSAGE}");'
        )
        input_type_statement = (
            f'static_assert({input_type_assert}, "{STATIC_ASSERT_TYPE_MESSAGE}");'
        )

        return (
            f"{self._get_warning_header()}"
            f"{fft_size_statement}\n"
            f"{input_type_statement}\n"
        )

    def write_cuda_static_asserts(self, is_forward: bool = None) -> None:
        """Write CUDA static assert statements to files.

        Parameters
        ----------
        is_forward : bool, optional
            True for forward only, False for inverse only, None for both.
        """
        if is_forward is None or is_forward:
            with open(self.fwd_static_assertions_file, "w") as f:
                f.write(self.generate_cuda_static_asserts(is_forward=True))

        if is_forward is None or not is_forward:
            with open(self.inv_static_assertions_file, "w") as f:
                f.write(self.generate_cuda_static_asserts(is_forward=False))

    def generate_template_instantiations(self, is_forward: bool = True) -> str:
        """Generate template instantiation statements for configurations.

        Parameters
        ----------
        is_forward : bool, optional
            True for forward FFT, False for inverse FFT.

        Returns
        -------
        str
            String containing template instantiations.
        """
        configs = self._filter_configs_by_direction(is_forward)
        instantiations = [config.get_template_instantiation() for config in configs]

        return f"{self._get_warning_header()}{chr(10).join(instantiations)}\n"

    def write_template_instantiations(self, is_forward: bool = None) -> None:
        """Write template instantiation statements to files.

        Parameters
        ----------
        is_forward : bool, optional
            True for forward only, False for inverse only, None for both.
        """
        if is_forward is None or is_forward:
            with open(self.fwd_implementations_file, "w") as f:
                f.write(self.generate_template_instantiations(is_forward=True))

        if is_forward is None or not is_forward:
            with open(self.inv_implementations_file, "w") as f:
                f.write(self.generate_template_instantiations(is_forward=False))

    def _generate_switch_case(self, config: ComplexToComplexFFT1DConfig) -> str:
        """Generate a single switch case statement for a configuration.

        Parameters
        ----------
        config : ComplexToComplexFFT1DConfig
            Configuration to generate switch case for.

        Returns
        -------
        str
            Switch case statement string.
        """
        cuda_type = type_str_to_cuda_type(config.input_data_type)
        is_forward = str(config.is_forward_fft).lower()

        return (
            f"case {config.fft_size}: "
            f"{config.function_name}<{cuda_type}, "
            f"{config.fft_size}u, {is_forward}, "
            f"{config.elements_per_thread}, {config.ffts_per_block}>(data_ptr); "
            "break;"
        )

    def _generate_default_case(self, configs: list[ComplexToComplexFFT1DConfig]) -> str:
        """Generate default case for switch statement with supported sizes.

        Parameters
        ----------
        configs : list of ComplexToComplexFFT1DConfig
            List of configurations to get supported sizes from.

        Returns
        -------
        str
            Default case statement string.
        """
        supported_sizes = [config.fft_size for config in configs]
        supported_sizes_str = ", ".join(map(str, supported_sizes))

        return (
            "default:\n"
            f'    std::string supported_sizes = "[{supported_sizes_str}]";\n'
            f'    TORCH_CHECK(false, "Unsupported FFT size " + std::to_string(fft_size) + '
            f'", supported sizes are: " + supported_sizes);'
        )

    def generate_python_binding_switch_statements(self, is_forward: bool = True) -> str:
        """Generate Python binding switch statements for configurations.

        Parameters
        ----------
        is_forward : bool, optional
            True for forward FFT, False for inverse FFT.

        Returns
        -------
        str
            String containing switch statement cases.
        """
        configs = self._filter_configs_by_direction(is_forward)
        switch_cases = [self._generate_switch_case(config) for config in configs]

        # Add default case
        switch_cases.append(self._generate_default_case(configs))

        return f"{self._get_warning_header()}{chr(10).join(switch_cases)}\n"

    def write_python_binding_switch_statements(self, is_forward: bool = None) -> None:
        """Write Python binding switch statements to files.

        Parameters
        ----------
        is_forward : bool, optional
            True for forward only, False for inverse only, None for both.
        """
        if is_forward is None or is_forward:
            with open(self.fwd_binding_cases_file, "w") as f:
                f.write(self.generate_python_binding_switch_statements(is_forward=True))

        if is_forward is None or not is_forward:
            with open(self.inv_binding_cases_file, "w") as f:
                f.write(
                    self.generate_python_binding_switch_statements(is_forward=False)
                )

    def write_all_files(self) -> None:
        """Write all output files for both forward and inverse FFTs."""
        self.write_cuda_static_asserts()
        self.write_template_instantiations()
        self.write_python_binding_switch_statements()


if __name__ == "__main__":
    config_gen = ComplexToComplexFFT1DGenerator.from_yaml(
        "/home/mgiammar/git_repositories/zipFFT/configs/fft_c2c_1d.yaml"
    )

    # Generate and print examples for both forward and inverse
    fwd_cuda_static_asserts = config_gen.generate_cuda_static_asserts(is_forward=True)
    inv_cuda_static_asserts = config_gen.generate_cuda_static_asserts(is_forward=False)
    fwd_template_instantiations = config_gen.generate_template_instantiations(
        is_forward=True
    )
    inv_template_instantiations = config_gen.generate_template_instantiations(
        is_forward=False
    )
    fwd_python_binding_cases = config_gen.generate_python_binding_switch_statements(
        is_forward=True
    )
    inv_python_binding_cases = config_gen.generate_python_binding_switch_statements(
        is_forward=False
    )

    print("Generated Forward CUDA Static Asserts:\n", fwd_cuda_static_asserts)
    print("Generated Inverse CUDA Static Asserts:\n", inv_cuda_static_asserts)
    print("Generated Forward Template Instantiations:\n", fwd_template_instantiations)
    print("Generated Inverse Template Instantiations:\n", inv_template_instantiations)
    print(
        "Generated Forward Python Binding Switch Statements:\n",
        fwd_python_binding_cases,
    )
    print(
        "Generated Inverse Python Binding Switch Statements:\n",
        inv_python_binding_cases,
    )

    # Write all files
    config_gen.write_all_files()
