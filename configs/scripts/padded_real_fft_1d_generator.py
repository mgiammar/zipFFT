import yaml
import json
from pathlib import Path

from base_config import (
    BaseFFT1dConfig,
    BaseFFT1dGenerator,
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

    def get_function_signature(
        self,
        include_pointer_type: bool = True,
        pointer_name: str = "input_ptr",
        output_pointer_name: str = "output_ptr",
    ) -> str:
        in_type = type_str_to_cuda_type(self.input_data_type)
        out_type = type_str_to_cuda_type(self.output_data_type)
        is_forward_fft = "true" if self.is_forward_fft else "false"
        if include_pointer_type:
            call = f"({in_type}* {pointer_name}, {out_type}* {output_pointer_name})"
        else:
            call = f"({pointer_name}, {output_pointer_name})"
        return (
            f"{self.function_name}<{in_type}, {out_type}, {self.signal_length}, "
            f"{self.fft_size}, {is_forward_fft}, {self.elements_per_thread}u, "
            f"{self.ffts_per_block}u>{call}"
        )


class PaddedRealToComplexFFT1dGenerator(BaseFFT1dGenerator):
    _script_dir = Path(__file__).parent
    _gen_dir = (_script_dir / ".." / ".." / "src" / "generated").resolve()

    def __init__(
        self,
        config_list=None,
        is_forward=True,
        binding_cases_file=None,
        implementation_file=None,
        static_assertions_file=None,
        **kwargs,
    ):
        self.is_forward = is_forward
        filtered = [
            cfg for cfg in (config_list or []) if cfg.is_forward_fft == is_forward
        ]
        defaults = self.default_file_paths()
        binding_cases_file = binding_cases_file or defaults[0]
        implementation_file = implementation_file or defaults[1]
        static_assertions_file = static_assertions_file or defaults[2]
        super().__init__(
            config_list=filtered,
            binding_cases_file=binding_cases_file,
            implementation_file=implementation_file,
            static_assertions_file=static_assertions_file,
            **kwargs,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        config_class=PaddedRealToComplexFFT1dConfig,
        is_forward=True,
        **kwargs,
    ):
        with open(yaml_path, "r") as file:
            config_data = yaml.safe_load(file)
        config_list = [config_class(**data) for data in config_data]
        return cls(config_list=config_list, is_forward=is_forward, **kwargs)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        config_class=PaddedRealToComplexFFT1dConfig,
        is_forward=True,
        **kwargs,
    ):
        with open(json_path, "r") as file:
            config_data = json.load(file)
        config_list = [config_class(**data) for data in config_data]
        return cls(config_list=config_list, is_forward=is_forward, **kwargs)

    def default_file_paths(self) -> tuple[str, str, str]:
        prefix = "fwd_" if self.is_forward else "inv_"
        return (
            str(self._gen_dir / f"{prefix}padded_fft_r2c_1d_binding_cases.inc"),
            str(self._gen_dir / f"{prefix}padded_fft_r2c_1d_implementations.inc"),
            str(self._gen_dir / f"{prefix}padded_fft_r2c_1d_assertions.inc"),
        )

    def generate_static_assertions(self) -> str:
        configs = self.config_list

        signal_length_conditions = [cfg.get_signal_length_assert() for cfg in configs]
        fft_size_conditions = [cfg.get_fft_size_assert() for cfg in configs]
        input_type_conditions = [cfg.get_input_data_type_assert() for cfg in configs]
        output_type_conditions = [cfg.get_output_data_type_assert() for cfg in configs]

        signal_length_conditions = list(set(filter(bool, signal_length_conditions)))
        fft_size_conditions = list(set(filter(bool, fft_size_conditions)))
        input_type_conditions = list(set(filter(bool, input_type_conditions)))
        output_type_conditions = list(set(filter(bool, output_type_conditions)))

        signal_length_conditions.sort()
        fft_size_conditions.sort()
        input_type_conditions.sort()
        output_type_conditions.sort()
        signal_length_assert = " || ".join(signal_length_conditions)

        fft_size_assert = " || ".join(fft_size_conditions)
        input_type_assert = " || ".join(input_type_conditions)
        output_type_assert = " || ".join(output_type_conditions)

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

    def generate_binding_cases(self) -> str:
        configs = self.config_list
        # Group by signal_length for nested switch
        grouped = {}
        for cfg in configs:
            grouped.setdefault(cfg.signal_length, []).append(cfg)
        switch_code = ""
        for signal_length, group in grouped.items():
            switch_code += f"case {signal_length}:\n"
            switch_code += "    switch (fft_size) {\n"
            for cfg in group:
                switch_code += (
                    "        "
                    + cfg.get_binding_case_call(
                        include_pointer_type=False,
                        pointer_name="input_ptr",
                        output_pointer_name="output_ptr",
                    ).replace("\n", "\n        ")
                    + "\n"
                )
            switch_code += (
                "        default:\n"
                '            TORCH_CHECK(false, "Unsupported FFT size ", fft_size, " for signal length ", signal_length);\n'
                "    }\n"
                "    break;\n"
            )
        switch_code += (
            "default:\n"
            '    TORCH_CHECK(false, "Unsupported signal length ", signal_length);\n'
        )
        return (
            f"{self._get_warning_header()}switch (signal_length) {{\n{switch_code}}}\n"
        )


if __name__ == "__main__":
    fwd_gen = PaddedRealToComplexFFT1dGenerator.from_yaml(
        yaml_path="configs/padded_fft_r2c_1d.yaml",
        is_forward=True,
    )
    # inv_gen = PaddedRealToComplexFFT1dGenerator.from_yaml(
    #     yaml_path="configs/padded_fft_r2c_1d.yaml",
    #     is_forward=False,
    # )
    fwd_gen.write_all_files()
    # inv_gen.write_all_files()
