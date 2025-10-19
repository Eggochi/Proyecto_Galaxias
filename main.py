import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
#import torch.nn.functional as F
import PrepararDatos as prep

def main():

    # ======================
    # CONFIGURACIÓN
    # ======================
    #Datos
    data_dir = 'smallGZ1/Imagenes'  # Cambia esta ruta a donde tengas tus imagenes
    cvs_file = 'smallGZ1/smallCSV_class.csv'  # Ruta al archivo CSV con rutas y etiquetas
    partition_size = 0.2  # Proporción para validación y test
    
    #Entrenamiento
    batch_size = 16     # Ajusta si te da "out of memory" (usa 8 o 12)
    num_epochs = 10
    learning_rate = 0.001

    #Early Stopping
    patience = 3        # Número de épocas sin mejora antes de parar
    min_delta = 0.001   # Mejora mínima para considerar que hay progreso

    #dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # ======================
    # TRANSFORMACIONES
    # ======================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
                         ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
                         ])
    
    # ======================
    # CARGA DE DATOS
    # ======================

    training_data, validation_data, test_data = prep.Dividir_datos(cvs_file, test_size=partition_size, validation=True)

    train_dataset   = prep.CSVDataset(root=data_dir,dataframe=training_data ,filename_col='OBJID', label_col='CLASS', transform=train_transform)
    val_dataset     = prep.CSVDataset(root=data_dir,dataframe=validation_data,filename_col='OBJID', label_col='CLASS', transform=val_transform)
    test_dataset    = prep.CSVDataset(root=data_dir,dataframe=test_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"Clases: {train_dataset.classes}")

    # ======================
    # MODELO: EfficientNet-B0
    # ======================
    modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Congelar capas base
    for param in modelo.parameters():
        param.requires_grad = False

    # Reemplazar la capa final (classifier)
    num_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(num_features, num_classes)
    modelo = modelo.to(device)

    # ======================
    # FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
    # ======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.classifier[1].parameters(), lr=learning_rate)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda' in device.type)

    # ======================
    # ENTRENAMIENTO
    # ======================

    best_val = 0.0
    counter = 0         # Contador de épocas sin mejora
    best_model_wts = None

    for epoch in range(num_epochs):
        modelo.train()
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward con mixed precision
            with torch.cuda.amp.autocast():
                outputs = modelo(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # ======================
        # VALIDACIÓN
        # ======================
        modelo.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = modelo(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Época {epoch+1}/{num_epochs} - Pérdida: {running_loss/len(train_loader):.4f} - Precisión: {acc:.2f}%")

        # Early Stopping
        if acc - best_val > min_delta:
            best_val = acc
            counter = 0
            best_model_wts = modelo.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping activado.")
                break

    # =====================
    # PRUEBA
    # =====================
    #modelo.eval()
    #with torch.no_grad():
    #    correct, total = 0, 0
    #    for inputs, labels in val_loader:
    #        inputs, labels = inputs.to(device), labels.to(device)
    #        outputs = modelo(inputs)

    #        probs=F.softmax(outputs,dim=1)
    #        conf, preds = torch.max(probs, 1)
    #        correct += (preds == labels).sum().item()
    #        total += labels.size(0) 

    # ======================
    # GUARDAR MODELO
    # ======================
    if best_model_wts:
        modelo.load_state_dict(best_model_wts)
        torch.save(modelo.state_dict(), "efficientnet_b0_best.pth")
        print("Modelo con mejor validación guardado como 'efficientnet_b0_best.pth'")


if __name__ == "__main__":
    main()
