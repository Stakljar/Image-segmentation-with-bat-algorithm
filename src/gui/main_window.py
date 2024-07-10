import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from gui.parameters_window import ParametersWindow
from gui.semantic_segmented_image_window import SemanticSegmentedImageWindow

class MainWindow(tk.Tk):
    def __init__(self, width: float, height: float):
        super().__init__()
        self.title("Image segmentation with bat algorithm")
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        self.add_elements()
        self.mainloop()

    def add_elements(self): 
        menu = tk.Menu(self, tearoff=False)
        self.configure(menu=menu)
        file = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=file)
        file.add_command(label="Open", command=self.open_image)
    
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        actions_panel = tk.Frame(self, bd=2, relief="groove")
        image_panel = tk.Frame(self)
        actions_panel.grid(row=0, column=0, sticky="nsew")
        actions_panel.pack_propagate(False)
        image_panel.grid(row=0, column=1, sticky="nsew")
        image_panel.pack_propagate(False)

        grayscale_segmentation_label = tk.Label(actions_panel, text="Select bat algorithm for\ngrayscale image segmentation", font=("Arial", 10, "bold"))
        grayscale_segmentation_label.pack(pady=(25, 10))
        self.ba_algorithms_for_grayscale_segmentation = ["Standard bat algorithm with Otsu", "Chaotic bat algorithm with Otsu",
                                                         "BA with inertia weight with Otsu", "KMeans BA"]
        self.selected_grayscale_segmentation_option = tk.StringVar(actions_panel, value=self.ba_algorithms_for_grayscale_segmentation[0])
        grayscale_segmentation_option_menu = tk.OptionMenu(actions_panel, self.selected_grayscale_segmentation_option, *self.ba_algorithms_for_grayscale_segmentation)
        grayscale_segmentation_option_menu.configure(bg="white", activebackground="white")
        grayscale_segmentation_option_menu.pack(pady=10)
        proceed_with_grayscale_segmentation_button = tk.Button(actions_panel, text="Proceed", bg="green", fg="white", command=lambda: self.display_parameters_screen(0))
        proceed_with_grayscale_segmentation_button.pack(pady=(10, 20))

        self.ba_algorithms_for_colored_segmentation = ["KMeans BA (colored)"]
        segmentation_label = tk.Label(actions_panel, text="Perform image segmentation\nwith KMeans BA", font=("Arial", 10, "bold"))
        segmentation_label.pack(padx=15, pady=(20, 10))
        proceed_with_colored_segmentation_button = tk.Button(actions_panel, text="Proceed", bg="green", fg="white", command=lambda: self.display_parameters_screen(1))
        proceed_with_colored_segmentation_button.pack(pady=10)

        semantic_segmentation_label = tk.Label(actions_panel, text="Perform semantic segmentation\nwith CNN BA", font=("Arial", 10, "bold"))
        semantic_segmentation_label.pack(padx=15, pady=(20, 10))
        generate_semantic_segmentation_button = tk.Button(actions_panel, text="Segment", bg="blue", fg="white", command=self.perform_semantic_segmentation)
        generate_semantic_segmentation_button.pack(pady=10)
        self.image_label = tk.Label(image_panel, text="Load image here", font=("Arial", 13, "bold"))
        self.image_label.pack(expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            self.file_path = file_path
            self.display_image(file_path)

    def display_image(self, file_path: str):
        image = ImageTk.PhotoImage(Image.open(file_path).resize((540, 500)))
        self.image_label.config(image=image)
        self.image_label.image = image

    def is_image_present(self):
        if(not self.image_label["image"]):
            tk.messagebox.showerror("Error", "Load image first.")
            return False
        return True
    
    def display_parameters_screen(self, type: int):
        if(not self.is_image_present()):
            return
        if(type == 0):
            ba_algorithm = self.selected_grayscale_segmentation_option.get()
            
        else:
            ba_algorithm = self.ba_algorithms_for_colored_segmentation[0]
        ParametersWindow(self, ba_algorithm, self.ba_algorithms_for_grayscale_segmentation + self.ba_algorithms_for_colored_segmentation,
                         self.file_path, 360, 340)
        
    def perform_semantic_segmentation(self):
        if(not self.is_image_present()):
            return
        SemanticSegmentedImageWindow(self, self.file_path)
