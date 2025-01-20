import numpy as np
import os
def load_plasma():
    package_dir = os.path.dirname(__file__)
    plasma_path = os.path.join(package_dir, 'plasma_data.npz')
    with np.load(plasma_path) as data:
        Jex = data['Jex']
        Jey = data['Jey']
        Jez = data['Jez']
    return {'Jex':Jex, 'Jey':Jey, 'Jez':Jez}