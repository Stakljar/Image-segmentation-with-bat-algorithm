import tkinter as tk
from typing import Callable, Any
from gui.segmented_image_window import SegmentedImageWindow

class ParametersWindow(tk.Toplevel):
    def __init__(self, parent, ba_algorithm, ba_algorithms, file_path, width, height):
        super().__init__(parent)
        self.parent = parent
        self.file_path = file_path
        self.grab_set()
        self.ba_algorithm = ba_algorithm
        self.ba_algorithms = ba_algorithms
        self.title("Parameters")
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        self.create_widgets()

    def create_widgets(self) -> None:
        parameters_frame = tk.Frame(self)
        parameters_frame.pack(pady=15)

        integer_parameters: dict[str, int] = {"Number of bats": 20, "Number of iterations": 20, "Dimension": 3}
        decimal_parameters: dict[str, float] = {"Minimum frequency": 0.0, "Maximum frequency": 2.0, "Sigma": 0.1, "Alpha": 0.9, "Gamma": 0.1}
        self.entries: list[tk.StringVar] = []

        i = self.add_params(entries=self.entries, frame=parameters_frame, parameters=integer_parameters, offset=0,
                            validatecommand=lambda value: True if(str.isdigit(value) or value=="") else False)
        i = self.add_params(entries=self.entries, frame=parameters_frame, parameters=decimal_parameters, offset=i + 1, validatecommand=lambda value: 
                            True if (str.isdigit(value) or (value == "" or (value.count('.') == 1 and value.replace('.', '').isdigit()))) else False)
        segment_button = tk.Button(self, text="Segment", bg="blue", fg="white", command=self.segment_image)
        segment_button.pack(pady=10)

    def add_params(self, entries: list[tk.StringVar], frame: tk.Frame, parameters: dict[str, Any], offset: int, validatecommand: Callable[[str], bool]) -> int:
        for i, (param_label, default_value) in enumerate(parameters.items()):
            label = tk.Label(frame, text=param_label)
            label.grid(row=i+offset, column=0, padx=15, pady=5, sticky="w")
            entry_var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=entry_var)
            entry.insert(0, default_value)
            entry.grid(row=i+offset, column=1, padx=15, pady=5, sticky="e")
            entries.append(entry_var)
            entry.config(validate="key", validatecommand=(self.register(validatecommand), "%P"))
        return i + offset
    
    def segment_image(self):
        SegmentedImageWindow(self, self.file_path, [param.get() for param in self.entries], self.ba_algorithm, self.ba_algorithms)
