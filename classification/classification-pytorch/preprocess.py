from torchvision import transforms

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))#torchvision.transforms.Normalize(mean, std, inplace=False)
        ])

