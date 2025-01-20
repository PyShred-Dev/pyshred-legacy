import numpy as np

class ReconstructionResult:
    def __init__(self, recon_dict, sensor_measurements, time):
        self.recon_dict = recon_dict
        self.sensor_measurements = sensor_measurements
        self.time = time