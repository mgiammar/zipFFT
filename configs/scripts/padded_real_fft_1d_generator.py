import yaml
import json
from pathlib import Path

from base_config import (
    BaseFFT1dConfig,
    type_str_to_torch_dtype,
    type_str_to_cuda_type,
)

STATIC_ASSERT_FFT_SIZE_MESSAGE = "Unsupported FFT size"
STATIC_ASSERT_TYPE_MESSAGE = "Unsupported input/output data type"


class PaddedRealToComplexFFT1dConfig(BaseFFT1dConfig):
    """Container for real-to-complex implicitly zero-padded 1D FFT configurations."""

    def __init__(
        self,
        signal_length: int,
        fft_size: int,
        input_data_type: str,
        output_data_type: str,
        is_forward_fft: bool,
        elements_per_thread: int,
        ffts_per_block: int,
    ):
        super().__init__(
            function_name="padded_block_real_fft_1d",
            signal_length=signal_length,
            fft_size=fft_size,
            input_data_type=input_data_type,
            output_data_type=output_data_type,
            is_forward_fft=is_forward_fft,
            elements_per_thread=elements_per_thread,
            ffts_per_block=ffts_per_block,
        )

    def get_fft_size_assert(self) -> str:
        return f"FFTSize == {self.fft_size}"

    def get_signal_length_assert(self) -> str:
        return f"SignalLength == {self.signal_length}"

    def get_input_data_type_assert(self) -> str:
        cuda_type = type_str_to_cuda_type(self.input_data_type)
        return f"std::is_same_v<Input_T, {cuda_type}>"

    def get_output_data_type_assert(self) -> str:
        cuda_type = type_str_to_cuda_type(self.output_data_type)
        return f"std::is_same_v<Output_T, {cuda_type}>"

    def get_template_instantiation(self) -> str:
        in_type = type_str_to_cuda_type(self.input_data_type)
        out_type = type_str_to_cuda_type(self.output_data_type)
        is_forward_fft = "true" if self.is_forward_fft else "false"
        return (
            f"template int {self.function_name}<{in_type}, {out_type}, "
            f"{self.signal_length}, {self.fft_size}, "
            f"{is_forward_fft}, {self.elements_per_thread}, "
            f"{self.ffts_per_block}>({in_type}* input_ptr, {out_type}* output_ptr);"
        )


class PaddedRealToComplexFFT1dGenerator:
    """Generates configurations for real-to-complex implicitly zero-padded 1D FFTs."""

    config_list: list[PaddedRealToComplexFFT1dConfig]

    _script_dir = Path(__file__).parent
    _gen_dir = _script_dir / ".." / ".." / "src" / "generated"

    # fmt: off
    fwd_implementations_file:   Path = _gen_dir / "fwd_padded_fft_r2c_1d_implementations.inc"
    fwd_static_assertions_file: Path = _gen_dir / "fwd_padded_fft_r2c_1d_assertions.inc"
    fwd_binding_cases_file:     Path = _gen_dir / "fwd_padded_fft_r2c_1d_binding_cases.inc"
    inv_implementations_file:   Path = _gen_dir / "inv_padded_fft_c2r_1d_implementations.inc"
    inv_static_assertions_file: Path = _gen_dir / "inv_padded_fft_c2r_1d_assertions.inc"
    inv_binding_cases_file:     Path = _gen_dir / "inv_padded_fft_c2r_1d_binding_cases.inc"
    # fmt: on

    def __init__(
        self,
        config_list: list[PaddedRealToComplexFFT1dConfig],
        fwd_implementations_file: str = None,
        inv_implementations_file: str = None,
        fwd_static_assertions_file: str = None,
        inv_static_assertions_file: str = None,
        fwd_binding_cases_file: str = None,
        inv_binding_cases_file: str = None,
    ):
        self.config_list = config_list
        if fwd_implementations_file:
            self.fwd_implementations_file = Path(fwd_implementations_file)
        if inv_implementations_file:
            self.inv_implementations_file = Path(inv_implementations_file)
        if fwd_static_assertions_file:
            self.fwd_static_assertions_file = Path(fwd_static_assertions_file)
        if inv_static_assertions_file:
            self.inv_static_assertions_file = Path(inv_static_assertions_file)
        if fwd_binding_cases_file:
            self.fwd_binding_cases_file = Path(fwd_binding_cases_file)
        if inv_binding_cases_file:
            self.inv_binding_cases_file = Path(inv_binding_cases_file)

    def _filter_configs_by_direction(
        self, is_forward: bool
    ) -> list[PaddedRealToComplexFFT1dConfig]:
        """Filter configurations by the direction of the FFT (forward or inverse)."""
        return [
            config for config in self.config_list if config.is_forward_fft == is_forward
        ]

    def _group_configs_by_signal_length(
        self, is_forward: bool
    ) -> dict[int, list[PaddedRealToComplexFFT1dConfig]]:
        """Group configurations by signal length."""
        grouped_configs = {}
        for config in self.config_list:
            if config.is_forward_fft == is_forward:
                if config.signal_length not in grouped_configs:
                    grouped_configs[config.signal_length] = []
                grouped_configs[config.signal_length].append(config)
        return grouped_configs

    @classmethod
    def from_yaml(cls, config_yaml_path: str) -> "PaddedRealToComplexFFT1dGenerator":
        with open(config_yaml_path, "r") as f:
            config_dicts = yaml.safe_load(f)
        return cls.parse_dicts(config_dicts)

    @classmethod
    def parse_dicts(
        cls, config_dicts: list[dict]
    ) -> "PaddedRealToComplexFFT1dGenerator":
        """Parse a list of configuration dictionaries into a generator."""
        configs = [
            PaddedRealToComplexFFT1dConfig(**config_dict)
            for config_dict in config_dicts
        ]
        return cls(configs)

    def _get_warning_header(self) -> str:
        """Generate a warning header for the generated files."""
        return (
            "// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "// !!! This file was auto-generated by the class   !!!\n"
            "// !!! PaddedRealToComplexFFT1dGenerator. Do not   !!!\n"
            "// !!! edit unless you know what you are doing.    !!!\n"
            "// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
        )

    def _collect_assert_conditions(
        self, configs: list[PaddedRealToComplexFFT1dConfig]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Collect static assertion conditions for a list of configurations."""
        # NOTE: These asserts constitute a superset of the implemented functions
        # since not all combinations of signal lengths and FFT sizes are generated.
        signal_length_conditions = []
        fft_size_conditions = []
        input_type_conditions = []
        output_type_conditions = []
        for config in configs:
            signal_length_conditions.append(config.get_signal_length_assert())
            fft_size_conditions.append(config.get_fft_size_assert())
            input_type_conditions.append(config.get_input_data_type_assert())
            output_type_conditions.append(config.get_output_data_type_assert())
        return (
            list(set(signal_length_conditions)),
            list(set(fft_size_conditions)),
            list(set(input_type_conditions)),
            list(set(output_type_conditions)),
        )

    def generate_cuda_static_asserts(self, is_forward: bool = True) -> str:
        """Generate static CUDA assertions for the given configurations."""
        configs = self._filter_configs_by_direction(is_forward)
        signal_length_conds, fft_size_conds, input_type_conds, output_type_conds = (
            self._collect_assert_conditions(configs)
        )
        signal_length_assert = " || ".join(signal_length_conds)
        fft_size_assert = " || ".join(fft_size_conds)
        input_type_assert = " || ".join(input_type_conds)
        output_type_assert = " || ".join(output_type_conds)
        signal_length_statement = f'static_assert({signal_length_assert}, "{STATIC_ASSERT_FFT_SIZE_MESSAGE} (signal length)");'
        fft_size_statement = (
            f'static_assert({fft_size_assert}, "{STATIC_ASSERT_FFT_SIZE_MESSAGE}");'
        )
        input_type_statement = f'static_assert({input_type_assert}, "{STATIC_ASSERT_TYPE_MESSAGE} (input wrong type)");'
        output_type_statement = f'static_assert({output_type_assert}, "{STATIC_ASSERT_TYPE_MESSAGE} (output wrong type)");'
        return (
            f"{self._get_warning_header()}"
            f"{signal_length_statement}\n"
            f"{fft_size_statement}\n"
            f"{input_type_statement}\n"
            f"{output_type_statement}\n"
        )

    def write_cuda_static_asserts(self, is_forward: bool = None) -> None:
        if is_forward is None or is_forward:
            self.fwd_static_assertions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.fwd_static_assertions_file, "w") as f:
                f.write(self.generate_cuda_static_asserts(is_forward=True))
        if is_forward is None or not is_forward:
            self.inv_static_assertions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.inv_static_assertions_file, "w") as f:
                f.write(self.generate_cuda_static_asserts(is_forward=False))

    def generate_template_instantiations(self, is_forward: bool = True) -> str:
        configs = self._filter_configs_by_direction(is_forward)
        instantiations = [config.get_template_instantiation() for config in configs]
        return f"{self._get_warning_header()}{chr(10).join(instantiations)}\n"

    def write_template_instantiations(self, is_forward: bool = None) -> None:
        if is_forward is None or is_forward:
            self.fwd_implementations_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.fwd_implementations_file, "w") as f:
                f.write(self.generate_template_instantiations(is_forward=True))
        if is_forward is None or not is_forward:
            self.inv_implementations_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.inv_implementations_file, "w") as f:
                f.write(self.generate_template_instantiations(is_forward=False))

    def _generate_switch_cases_single_signal_length(
        self, signal_length: int, configs: list[PaddedRealToComplexFFT1dConfig]
    ) -> str:
        """Generate a switch case for a specific signal length.

        NOTE: These switch cases are nested two deep since both the signal length and
        FFT size are used to determine the correct template instantiation.
        """
        cases = []
        for config in configs:
            if config.signal_length == signal_length:
                in_type = type_str_to_cuda_type(config.input_data_type)
                out_type = type_str_to_cuda_type(config.output_data_type)
                is_forward_fft = "true" if config.is_forward_fft else "false"
                cases.append(
                    f"case {config.fft_size}: {config.function_name}<{in_type}, {out_type}, "
                    f"{config.signal_length}, {config.fft_size}, "
                    f"{is_forward_fft}, {config.elements_per_thread}u, "
                    f"{config.ffts_per_block}u>(input_ptr, output_ptr); break;"
                )

        # Default case if no matching FFT size is found
        cases.append(
            'default: TORCH_CHECK(false, "Unsupported FFT size ", fft_size, " for signal length ", signal_length);'
        )

        full_case = (
            f"switch (fft_size) {{\n"
            f"{chr(10).join(['    ' + case for case in cases])}\n"
            f"}}"
        )
        return full_case
    
    def _generate_default_case(self, configs: list[PaddedRealToComplexFFT1dConfig]) -> str:
        """Generate a default case for unsupported configurations."""
        signal_lengths = set(config.signal_length for config in configs)
        return (
            f"TORCH_CHECK(false, \"Unsupported signal length \", signal_length, "
            f"\". Supported lengths: {', '.join(map(str, sorted(signal_lengths)))}, "
            f"but got\", signal_length);"
        )
        
    def generate_python_binding_switch_statements(self, is_forward: bool = True) -> str:
        """Generate Python binding switch statements for the given configurations."""
        configs = self._filter_configs_by_direction(is_forward)
        grouped_configs = self._group_configs_by_signal_length(is_forward)

        switch_statements = []
        for signal_length, signal_length_configs in grouped_configs.items():
            switch_case = self._generate_switch_cases_single_signal_length(
                signal_length, signal_length_configs
            )
            switch_statements.append(switch_case)

        switch_code = ""
        for signal_length, signal_length_configs in grouped_configs.items():
            switch_code += f"case {signal_length}:\n"
            switch_code += "    " + self._generate_switch_cases_single_signal_length(signal_length, signal_length_configs).replace("\n", "\n    ") + "\n    break;\n"

        switch_code += "default:\n"
        switch_code += f"    {self._generate_default_case(configs)}\n"

        return f"{self._get_warning_header()}switch (signal_length) {{\n{switch_code}}}\n"
    
    def write_python_binding_switch_statements(self, is_forward: bool = None) -> None:
        if is_forward is None or is_forward:
            self.fwd_binding_cases_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.fwd_binding_cases_file, "w") as f:
                f.write(self.generate_python_binding_switch_statements(is_forward=True))
        if is_forward is None or not is_forward:
            self.inv_binding_cases_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.inv_binding_cases_file, "w") as f:
                f.write(self.generate_python_binding_switch_statements(is_forward=False))
                
    def write_all_files(self) -> None:
        """Write all generated files to disk."""
        self.write_cuda_static_asserts()
        self.write_template_instantiations()
        self.write_python_binding_switch_statements()
        
        
if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "padded_fft_r2c_1d.yaml"
    gen = PaddedRealToComplexFFT1dGenerator.from_yaml(str(config_path))
    gen.write_all_files()
