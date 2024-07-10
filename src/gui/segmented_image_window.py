import tkinter as tk
import cv2
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from bat_algorithm.algorithms.chaotic_bat_algorithm import ChaoticBatAlgorithm
from bat_algorithm.algorithms.bat_algorithm_with_inertia_weight import BatAlgorithmWithInertiaWeight
from bat_algorithm.algorithms.bat_algorithm import BatAlgorithm
from methods.otsu_method import OtsuMethod
from functions.chaotic_maps import tent
from functions.inertia_weights import linear_decreasing
from bat_algorithm.algorithms.bat_algorithm_with_kmeans import BatAlgorithmWithKMeans
from threading import Thread
from utils.utils import segment_image_by_thresholds, flatten_rows_and_cols_of_image, segment_image_by_clusters
from methods.kmeans import KMeans

class SegmentedImageWindow(tk.Toplevel):
    def __init__(self, parent, image_path, parameters, ba_algorithm, ba_algorithms):
        super().__init__(parent)
        self.parent = parent
        self.parameters = parameters
        self.ba_algorithm = ba_algorithm
        self.ba_algorithms = ba_algorithms
        self.image_path = image_path
        self.grab_set()
        self.title("Segmented image plot")
        self.resizable(False, False)
        self.bind("<<Segmented>>", self.plot)
        Thread(target=self.segment, daemon=True).start()

    def segment(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if(self.ba_algorithm != self.ba_algorithms[3] and self.ba_algorithm != self.ba_algorithms[4]):
            otsu_method = OtsuMethod(image = image)
            if(self.ba_algorithm == self.ba_algorithms[0]):
                best_positions, _ = BatAlgorithm(total_bats=int(self.parameters[0]), num_iterations=int(self.parameters[1]), dimension=int(self.parameters[2]),
                        min_position=0, max_position=255, min_frequency=float(self.parameters[3]),
                        max_frequency=float(self.parameters[4]), sigma=float(self.parameters[5]), 
                        alpha=float(self.parameters[6]), gamma=float(self.parameters[7])).run(objective_function=otsu_method.calculate_within_class_variance)
            elif(self.ba_algorithm == self.ba_algorithms[1]):
                best_positions, _ = ChaoticBatAlgorithm(chaos_map=lambda x: tent(x, 1.9), total_bats=int(self.parameters[0]),
                        num_iterations=int(self.parameters[1]), dimension=int(self.parameters[2]),
                        min_position=0, max_position=255, min_frequency=float(self.parameters[3]),
                        max_frequency=float(self.parameters[4]), sigma=float(self.parameters[5]), 
                        alpha=float(self.parameters[6]), gamma=float(self.parameters[7])).run(objective_function=otsu_method.calculate_within_class_variance)
            elif(self.ba_algorithm == self.ba_algorithms[2]):
                best_positions, _ = BatAlgorithmWithInertiaWeight(weight_function=lambda t, t_max:
                        linear_decreasing(t=t, t_max=t_max, w_min=0.4, w_max=0.9), total_bats=int(self.parameters[0]),
                        num_iterations=int(self.parameters[1]), dimension=int(self.parameters[2]),
                        min_position=0, max_position=255, min_frequency=float(self.parameters[3]),
                        max_frequency=float(self.parameters[4]), sigma=float(self.parameters[5]), 
                        alpha=float(self.parameters[6]), gamma=float(self.parameters[7])).run(objective_function=otsu_method.calculate_within_class_variance)
            segmented_image = segment_image_by_thresholds(otsu_method.image, best_positions)
        else:
            if(self.ba_algorithm == self.ba_algorithms[4]):
                image = cv2.imread(self.image_path)
            pixels = flatten_rows_and_cols_of_image(image)
            best_positions, _, best_labels = BatAlgorithmWithKMeans(kmeans=KMeans(data=pixels, max_iterations=2), total_bats=int(self.parameters[0]),
                    num_iterations=int(self.parameters[1]), dimension=int(self.parameters[2]), min_position=0, max_position=255,
                    min_frequency=float(self.parameters[3]), max_frequency=float(self.parameters[4]), sigma=float(self.parameters[5]), 
                    alpha=float(self.parameters[6]), gamma=float(self.parameters[7])).run()
            segmented_image = segment_image_by_clusters(image=image, centroids=best_positions, labels=best_labels)
        self.segmented_image = segmented_image
        self.event_generate("<<Segmented>>")

    def plot(self, _):
        fig = Figure(figsize=(5, 5), dpi=100)
        plt = fig.add_subplot(111)
        plt.imshow(cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2RGB))
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self)
        canvas.get_tk_widget().pack()
