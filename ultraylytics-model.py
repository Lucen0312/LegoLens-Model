import torch
from ultralytics import YOLO
def main():
    model = YOLO('yolov8n.pt')

    # Set the model configuration
    model.cfg = {
        'data': 'data.yaml',
        'epochs': 10,
        'batch_size': 16,
        'img_size': 416,
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'optimizer': 'sgd',
        'pretrained_weights': None,
        'augment': True,  # Enable data augmentation; crucial for this dataset
    }

    model.train()

    results = model.val()
    print(results)

    model.save('ultralytics.pt')

if __name__ == '__main__':
    main()
    