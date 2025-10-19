import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension

def Dividir_datos(ruta_csv: str, test_size: float = 0.2, validation: bool = True, random_state: int = 42
                 ):
    """
    Lee CSV y divide en train / (val, test).
    Si validation=True: primero separa train / temp (test_size),
    luego divide temp en validation y test a partes iguales.
    Devuelve (train_df, val_df, test_df).
    """
    df = pd.read_csv(ruta_csv)
    if df.shape[0] == 0:
        raise ValueError(f"CSV vacío: {ruta_csv}")

    datos_entrenamiento, datos_prueba = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    if validation:
        # dividir el 'prueba' en validación y prueba (mitades)
        datos_validacion, datos_prueba = train_test_split(datos_prueba, test_size=0.5, random_state=random_state, shuffle=True)
        print(len(datos_entrenamiento), len(datos_validacion), len(datos_prueba))
        return datos_entrenamiento.reset_index(drop=True), datos_validacion.reset_index(drop=True), datos_prueba.reset_index(drop=True)
    
    print(len(datos_entrenamiento), len(datos_prueba))
    return datos_entrenamiento.reset_index(drop=True), pd.DataFrame(columns=df.columns), datos_prueba.reset_index(drop=True)


class CSVDataset(Dataset):
    def __init__(self, root, dataframe,filename_col=None, label_col=None, transform=None,
                 loader=default_loader, extensions=IMG_EXTENSIONS):
        """
        Dataset a partir de un DataFrame de pandas.
        - root: ruta base de las imágenes
        - dataframe: objeto pd.DataFrame con columnas de ruta y etiqueta
        - filename_col / label_col: nombres de columnas (si no se especifican, usa primera y última)
        """
        self.root = root
        self.loader = loader
        self.transform = transform

        df = dataframe
        if df.shape[1] < 2:
            raise ValueError("El DataFrame debe tener al menos 2 columnas (filename y clase).")

        # Columnas a usar
        filename_col = filename_col or df.columns[0]
        label_col = label_col or df.columns[-1]

        # Clases y mapeo
        self.classes = df[label_col].value_counts().index.tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Construir lista de muestras válidas
        self.samples = []
        for fname, label in zip(df[filename_col], df[label_col]):
            if pd.isna(fname) or pd.isna(label):
                continue
            path = os.path.join(root, f"{fname}.jpeg")
            if os.path.exists(path) and has_file_allowed_extension(path, extensions):
                self.samples.append((path, self.class_to_idx[label]))

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target

class UnlabeledDataSet(torch.utils.data.Dataset):
    """Carga imágenes desde una carpeta (recursiva). Devuelve (image, path)."""
    def __init__(self, root, transform=None, loader=default_loader, extensions=IMG_EXTENSIONS):
        self.transform = transform
        self.loader = loader
        self.samples = []
        for rd, _, files in os.walk(root):
            for f in files:
                path = os.path.join(rd, f)
                if has_file_allowed_extension(path, extensions):
                    self.samples.append(path)
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path