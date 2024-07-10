from bat_algorithm.bat import Bat
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy.typing as npt
from bat_algorithm.bat_algorithm_params import BatAlgorithmParams
import copy
from typing import Dict, Any

class CNNBatAlgorithm(BatAlgorithmParams):
    def __init__(self, train_dataset, val_dataset, total_bats: int,
                 num_iterations: int, min_position: float | list[float] | npt.NDArray[np.float64]=[4, 0.001],
                 max_position: float | list[float] | npt.NDArray[np.float64]=[32, 0.1], max_epochs: int=30,
                 min_frequency: float=0., max_frequency: float=1., device="cuda", sigma: float=0.1, gamma: float=0.1, alpha: float=0.97) -> None:

        super().__init__(total_bats=total_bats, num_iterations=num_iterations, dimension=2, min_position=min_position,
                max_position=max_position, min_frequency=min_frequency, max_frequency=max_frequency, sigma=sigma, gamma=gamma, alpha=alpha)
        self.device = device
        self.num_workers: int = 0
        self.max_epochs = max_epochs

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.criterion = nn.CrossEntropyLoss()

    def _train_and_validate_model(self, hyper_parameters) -> tuple[float, Dict[str, Any] | None, int]:
        batch_size: int = int(np.round(hyper_parameters[0]))
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        model = torchvision.models.segmentation.fcn_resnet50().to(self.device)
        self.optimizer = optim.SGD(model.parameters(), lr=hyper_parameters[1], momentum=0.9)
        lowest_loss: float = np.inf
        best_model_state = None
        best_num_epochs: int = 999
        for i in range(self.max_epochs):
            model.train()
            for data in train_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = model(images)["out"]
                loss = self.criterion(outputs, labels.squeeze(1).long())
                loss.backward()
                self.optimizer.step()
            model.eval()
            running_loss: float = 0.0
            with torch.no_grad():
                for data in val_loader:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = model(images)["out"]
                    loss = self.criterion(outputs, labels.squeeze(1).long()) 
                    running_loss += loss.item()
            average_loss: float = running_loss / len(val_loader)
            if(average_loss < lowest_loss):
                best_model_state = copy.deepcopy(model.state_dict())
                lowest_loss = average_loss
                best_num_epochs = i + 1
        return lowest_loss, best_model_state, best_num_epochs
    
    def _initialize_starting_values(self) -> tuple[list[Bat], list[Dict[str, Any] | None], list[int]]:
        bats: list[Bat] = []
        model_states: list[Dict[str, Any] | None] = []
        nums_epochs: list[int] = []
        for _ in range(self.total_bats):
            position: npt.NDArray[np.float64] = np.random.uniform(self.min_position, self.max_position, self.dimension)
            velocity: npt.NDArray[np.float64] = np.zeros(self.dimension)
            frequency: float = 0.
            pulse_rate: float = 0.
            marginal_pulse_rate: float = 1.
            loudness: float = 1.
            fitness, model_state, num_epoch = self._train_and_validate_model(position)
            bats.append(Bat(
                position=position,
                velocity=velocity,
                frequency=frequency,
                loudness=loudness,
                pulse_rate=pulse_rate,
                marginal_pulse_rate=marginal_pulse_rate,
                fitness=fitness
            ))
            model_states.append(model_state)
            nums_epochs.append(num_epoch)
        return bats, model_states, nums_epochs

    def run(self) -> tuple[npt.NDArray[np.float64], float, Dict[str, Any] | None, int]:
        bats, model_states, nums_epochs = self._initialize_starting_values()
        t: int = 0
        best_bat = min(bats, key=lambda bat: bat.fitness)
        best_index, best_bat = min(enumerate(bats), key=lambda x: x[1].fitness)
        best_model_state: Dict[str, Any] | None = model_states[best_index]
        best_num_epochs: int = nums_epochs[best_index]
        new_fitness: float
        while(t < self.num_iterations):
            t = t + 1
            for bat in bats:
                bat.frequency = np.random.uniform(self.min_frequency, self.max_frequency)
                bat.velocity = bat.velocity + (bat.position-best_bat.position) * bat.frequency
                temp_position: npt.NDArray[np.float64] = bat.position + bat.velocity
                if(np.random.rand() > bat.pulse_rate):
                    temp_position = best_bat.position + self.sigma * np.random.normal(0, 1, self.dimension) * np.mean([bat.loudness for bat in bats])
                temp_position = np.clip(temp_position, self.min_position, self.max_position)
                new_fitness, model_state, num_epochs = self._train_and_validate_model(temp_position)
                if(np.random.rand() < bat.loudness and new_fitness < bat.fitness):
                    bat.position = temp_position
                    bat.fitness = new_fitness
                    bat.pulse_rate = bat.marginal_pulse_rate * (1 - np.exp(-self.gamma*t))
                    bat.loudness = self.alpha * bat.loudness
                    if(new_fitness < best_bat.fitness):
                        best_bat = bat
                        best_model_state = model_state
                        best_num_epochs = num_epochs
        return best_bat.position, best_bat.fitness, best_model_state, best_num_epochs
