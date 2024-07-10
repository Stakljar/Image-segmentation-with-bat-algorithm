import tkinter as tk
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from threading import Thread
import torchvision
import torch
from PIL import Image
from utils.utils import get_pascal_voc_classes_colors

class SemanticSegmentedImageWindow(tk.Toplevel):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.parent = parent
        self.image_path = image_path
        self.model_path = "./model/fcn_resnet50_trained_on_pascal_voc_2007.pth"
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = torchvision.models.segmentation.fcn_resnet50().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.grab_set()
        self.title("Segmented image plot")
        self.resizable(False, False)
        self.bind("<<Segmented>>", self.plot)
        Thread(target=self.segment, daemon=True).start()

    def segment(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(self.image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transform(image)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0).to(self.device))["out"]
            predicted = torch.argmax(outputs, dim=1)
            segmented_image = predicted[0]
        self.segmented_image = Image.fromarray(segmented_image.byte().cpu().numpy())
        self.segmented_image.putpalette(get_pascal_voc_classes_colors())
        self.event_generate("<<Segmented>>")

    def plot(self, _):
        fig = Figure(figsize=(5, 5), dpi=100)
        plt = fig.add_subplot(111)
        plt.imshow(self.segmented_image)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self)
        canvas.get_tk_widget().pack()
